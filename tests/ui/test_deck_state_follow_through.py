"""The Deck bar and wheel follow the same state the panels do.

Pins the lane-A Deck fixes: the wheel's Play and Bake slots change with the
state the way the bar's buttons do, Apply/Step/Materials are greyed with a
reason where the panel would not offer them, Restart and Cancel Bake ask first
(INVOKE_DEFAULT), tooltips are plain language, a Step moves the readout, the
Liquid Preview toggle warns like the panel, the LIVE chip names the setting by
its label, and the bar hides with the viewport's Overlays toggle.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from deck_doubles import install_deck_bpy, reload_deck

PKG = "fluidSim_warp_blender51.blender"


class _FakeOp:
    def __init__(self, identifier, ran):
        self.identifier, self.ran = identifier, ran

    def get_rna_type(self):
        return SimpleNamespace(identifier=self.identifier)

    def poll(self):
        return True

    def __call__(self, call_context, **properties):
        self.ran.append((self.identifier, call_context, properties))
        return {"FINISHED"}


def _fake_module(monkeypatch, name, **attrs):
    module = types.ModuleType(f"{PKG}.{name}")
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, f"{PKG}.{name}", module)
    # ``from .. import name`` reads the package attribute once the real module was imported.
    monkeypatch.setattr(sys.modules[PKG], name, module, raising=False)
    return module


@pytest.fixture
def deck(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    ran = []
    names = ("step", "apply_changes", "restart_simulation", "bake_cancel", "bake_start",
             "bake_resume", "install_render_nodes", "preview_pause")
    bpy.ops = SimpleNamespace(
        fluidsim=SimpleNamespace(**{n: _FakeOp(f"FLUIDSIM_OT_{n}", ran) for n in names}),
        screen=SimpleNamespace(animation_play=_FakeOp("SCREEN_OT_animation_play", ran)))
    scene = SimpleNamespace(fluidsim=SimpleNamespace(deck_shown=True, deck_stage="SIMULATE",
                                                     cache_playback_enabled=False,
                                                     stat_step_ms=0.0),
                            frame_current=1, name="Scene")
    bpy.context = SimpleNamespace(scene=scene, area=None, region=None,
                                  preferences=bpy.context.preferences)
    deck.bpy, deck.ran, deck.scene = bpy, ran, scene
    deck.state = {"activity": "IDLE", "playing": False, "owns": True}
    monkeypatch.setattr(deck.controls, "deck_state", lambda _ctx: (
        deck.state["activity"], deck.state["playing"], deck.state["owns"]))
    deck.pending = {"action": "NONE"}
    _fake_module(monkeypatch, "pending_settings",
                 pending_presentation=lambda _scene: dict(deck.pending),
                 declaring_group=lambda _scene, _name: None)
    deck.session = SimpleNamespace(active=False, owns_scene=lambda _s: deck.state["owns"],
                                   solver=SimpleNamespace(frame=1))
    _fake_module(monkeypatch, "scene_bridge", session=lambda: deck.session)
    deck.domain = {"object": object()}
    _fake_module(monkeypatch, "scene_access",
                 active_domain_object=lambda _scene: deck.domain["object"],
                 scene_identity=lambda scene: id(scene))
    return deck


def _status(deck, control):
    return deck.controls.control_status(deck.bpy.context, control)


# Defect 1: the wheel gates as strictly as the bar -----------------------------

def test_wheel_play_slot_follows_the_bar_transport(deck):
    controls, wheel = deck.controls, deck.wheel
    slot = wheel.STAGE_SLOTS["SIMULATE"].index(controls.PLAY)
    assert wheel.stage_controls(deck.bpy.context, "SIMULATE")[slot] is controls.PLAY
    deck.state["playing"] = True
    assert wheel.stage_controls(deck.bpy.context, "SIMULATE")[slot] is controls.PAUSE
    deck.state["activity"] = "PRODUCTION_BAKE"
    assert wheel.stage_controls(deck.bpy.context, "SIMULATE")[slot] is controls.CANCEL_BAKE


def test_apply_is_greyed_when_nothing_is_pending(deck):
    ready, reason = _status(deck, deck.controls.APPLY)
    assert not ready and "No changes" in reason
    deck.pending["action"] = "RESET"
    assert _status(deck, deck.controls.APPLY) == (True, "")


def test_step_is_greyed_in_cache_playback_and_when_another_scene_owns_the_solver(deck):
    STEP = deck.controls.STEP
    assert _status(deck, STEP) == (True, "")
    deck.state["activity"] = "CACHE_PLAYBACK"
    ready, reason = _status(deck, STEP)
    assert not ready and "Cache playback" in reason
    deck.state.update(activity="LIVE_PREVIEW", owns=False)
    deck.session.active = True
    ready, reason = _status(deck, STEP)
    assert not ready and "Another scene" in reason


def test_materials_are_greyed_without_a_domain_and_during_a_bake(deck):
    materials = next(c for c in deck.wheel.STAGE_SLOTS["LOOK"]
                     if c.target == "fluidsim.install_render_nodes")
    assert _status(deck, materials) == (True, "")
    deck.state["activity"] = "PRODUCTION_BAKE"
    ready, reason = _status(deck, materials)
    assert not ready and "Production Bake" in reason
    deck.domain["object"] = None
    ready, reason = _status(deck, materials)
    assert not ready and "domain" in reason.lower()


# Defects 2 and 3: Bake follows the cache; Restart and Cancel ask first ---------

def test_bake_slot_shows_progress_resume_or_restart(deck, monkeypatch):
    controls = deck.controls
    baking, action = {"on": False}, {"value": "START"}
    monkeypatch.setattr(controls, "scene_is_baking", lambda _s: baking["on"])
    monkeypatch.setattr(controls, "_bake_action", lambda _s: action["value"])
    context = deck.bpy.context
    assert controls.resolve(context, controls.BAKE) is controls.BAKE
    action["value"] = "RESUME"
    assert controls.resolve(context, controls.BAKE).target == "fluidsim.bake_resume"
    action["value"] = "RESTART"
    restart = controls.resolve(context, controls.BAKE)
    assert restart.target == "fluidsim.bake_start" and "Restart" in restart.label
    assert dict(restart.properties) == {"restart_existing": True}
    baking["on"] = True
    progress = controls.resolve(context, controls.BAKE)
    assert progress.target == "fluidsim.bake_cancel"
    # The bar's Bake button is the resolved one, so it is never a dead greyed button.
    layout_tools = tuple(controls.resolve(context, c) for c in controls.TOOL_CONTROLS)
    assert progress in layout_tools and controls.BAKE not in layout_tools


@pytest.mark.parametrize("name", ["RESTART", "CANCEL_BAKE", "BAKE_RESTART", "BAKE_PROGRESS"])
def test_restart_and_cancel_bake_run_with_invoke_default(deck, name):
    control = getattr(deck.controls, name)
    deck.controls.run_control(deck.bpy.context, control)
    assert deck.ran[-1][1] == "INVOKE_DEFAULT"
    if name == "BAKE_RESTART":
        assert deck.ran[-1][2] == {"restart_existing": True}


# Defect 4: plain-language tooltips ---------------------------------------------

def test_tooltips_never_show_idnames_and_set_up_scene_says_what_it_does(deck):
    controls = deck.controls
    for control in (*controls.all_controls(), controls.BAKE_PROGRESS, controls.BAKE_RESUME,
                    controls.BAKE_RESTART):
        assert f"({control.target})" not in control.tooltip
        assert "fluidsim." not in control.tooltip
    setup = next(c for c in controls.STAGE_SLOTS["SETUP"] if c.target == "fluidsim.setup_scene")
    assert "selected mesh" in setup.tooltip and "role" in setup.tooltip


# Defect 5: a Step moves the readout and the sparkline --------------------------

def test_step_advances_the_readout_frame_and_adds_a_sparkline_sample(deck):
    bar, scene = deck.bar, deck.scene
    scene.fluidsim.stat_step_ms = 5.0
    first = bar.sample_readout(scene)
    assert first.frame == 1 and len(first.samples) == 1
    deck.session.solver.frame = 2  # a Step: the solver moves, the timeline does not
    scene.fluidsim.stat_step_ms = 6.0
    second = bar.sample_readout(scene)
    assert second.frame == 2 and second.samples == (5.0, 6.0)


# Defect 6: Liquid Preview warns like the panel ---------------------------------

def test_liquid_preview_on_with_no_particles_warns(deck, monkeypatch):
    controls = deck.controls
    group = SimpleNamespace(liquid_preview_enabled=False)
    monkeypatch.setattr(controls, "property_owner", lambda _ctx, _c: group)
    _fake_module(monkeypatch, "output_publication", latest_published_points=lambda: None)
    reports = []
    controls.run_control(deck.bpy.context, controls.LIQUID_PREVIEW,
                         report=lambda kind, message: reports.append((kind, message)))
    assert group.liquid_preview_enabled is True
    assert reports == [({"WARNING"}, controls.NO_PUBLISHED_PARTICLES)]
    # Switching it off, or with particles published, says nothing.
    _fake_module(monkeypatch, "output_publication",
                 latest_published_points=lambda: SimpleNamespace(particle_count=10))
    controls.run_control(deck.bpy.context, controls.LIQUID_PREVIEW,
                         report=lambda kind, message: reports.append((kind, message)))
    controls.run_control(deck.bpy.context, controls.LIQUID_PREVIEW,
                         report=lambda kind, message: reports.append((kind, message)))
    assert len(reports) == 1


# Defect 7: the LIVE chip names the setting by its UI label ---------------------

def test_live_chip_shows_the_property_label_not_its_name(deck):
    bar = deck.bar
    rna = SimpleNamespace(properties={"initial_velocity": SimpleNamespace(name="Initial Velocity")})
    group = SimpleNamespace(bl_rna=rna)
    bar.note_deck_write(deck.scene, "initial_velocity", group)
    assert bar.chip_state(deck.scene).text.endswith("Initial Velocity")
    # Without the group the registered settings classes give the label.
    deck.bpy.types.FLUIDSIM_PG_domain = SimpleNamespace(bl_rna=SimpleNamespace(
        properties={"viscosity_enabled": SimpleNamespace(name="Viscosity")}))
    assert bar.field_label("viscosity_enabled") == "Viscosity"
    assert bar.field_label("unknown_field_name") == "Unknown field name"


# Defect 8: the bar follows the viewport's Overlays toggle ----------------------

def test_bar_hides_when_viewport_overlays_are_off(deck, monkeypatch):
    bar = deck.bar
    monkeypatch.setattr(deck.controls, "scene_is_baking", lambda _s: False)
    monkeypatch.setattr(deck.controls, "_bake_action", lambda _s: "START")
    monkeypatch.setattr(bar, "bake_fraction", lambda _s: None)
    overlay = SimpleNamespace(show_overlays=True)
    context = SimpleNamespace(scene=deck.scene, region=SimpleNamespace(width=1600),
                              preferences=deck.bpy.context.preferences,
                              space_data=SimpleNamespace(type="VIEW_3D", overlay=overlay))
    assert bar.current_layout(context) is not None
    overlay.show_overlays = False
    assert bar.current_layout(context) is None
