"""Liquid Deck gizmos write through the same property path as the panels.

Spec trap 5 (docs/specs/4.0/SPEC-liquid-deck.md): a gizmo drag must not change
simulation results silently. The gizmo's ``invoke``/``modal``/``exit`` are called
directly, as a ``--background`` session would, and the property write goes through
an RNA stand-in that fires the same ``update=`` callback ``properties.py`` declares,
so the Pending Settings classification can be compared with a panel edit.
"""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from deck_doubles import install_deck_bpy, reload_deck
from settings_doubles import full_settings, split_settings

ROOT = Path(__file__).resolve().parents[2]
PROPERTIES = ROOT / "fluidSim_warp_blender51" / "properties.py"
_UPDATE_ACTIONS = {
    "_request_sync": "SYNC",
    "_request_reset": "RESET",
    "_request_rebuild": "REBUILD",
    "_request_live": "LIVE",
}


def _object_update_actions():
    """``FLUIDSIM_PG_object`` field -> the SceneUpdate its ``update=`` callback requests."""
    tree = ast.parse(PROPERTIES.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FLUIDSIM_PG_object":
            actions = {}
            for statement in node.body:
                if not (isinstance(statement, ast.AnnAssign)
                        and isinstance(statement.annotation, ast.Call)):
                    continue
                for keyword in statement.annotation.keywords:
                    if keyword.arg == "update" and isinstance(keyword.value, ast.Name):
                        action = _UPDATE_ACTIONS.get(keyword.value.id)
                        if action:
                            actions[statement.target.id] = action
            return actions
    raise LookupError("FLUIDSIM_PG_object")


class _RnaObjectSettings:
    """Attribute writes fire the declared ``update=`` callback, as Blender's RNA does."""

    def __init__(self, scene_ref, pending, actions, **values):
        object.__setattr__(self, "_scene_ref", scene_ref)
        object.__setattr__(self, "_pending", pending)
        object.__setattr__(self, "_actions", actions)
        object.__setattr__(self, "writes", [])
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        if isinstance(value, list):
            value = tuple(value)
        object.__setattr__(self, name, value)
        self.writes.append(name)
        action = self._actions.get(name)
        if action is not None:
            self._pending.request_scene_update(
                self._scene_ref(), self._pending.SceneUpdate[action])


def _environment(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    output_publication = types.ModuleType("fluidSim_warp_blender51.blender.output_publication")
    output_publication.refresh_existing_output_settings = lambda *_a, **_k: None
    output_publication.update_outputs = lambda *_a, **_k: None
    output_publication.prepare_outputs = lambda *_a, **_k: None
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.output_publication", output_publication)
    preferences = types.ModuleType("fluidSim_warp_blender51.preferences")
    preferences.configured_device = lambda: "cuda:0"
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.preferences", preferences)
    for name in ("pending_settings", "scene_bridge"):
        sys.modules.pop(f"fluidSim_warp_blender51.blender.{name}", None)
    pending = importlib.import_module("fluidSim_warp_blender51.blender.pending_settings")
    deck = reload_deck()
    return bpy, pending, deck


def _emitter_scene(pending, **object_values):
    fs = full_settings(live=True, status="", voxel_size=0.1)
    scene = SimpleNamespace(fluidsim=split_settings(fs), objects=[], frame_start=1,
                            frame_current=1, name="Scene")
    values = dict(
        role="SOURCE", point_source_representation="AUTO", mesh_evaluation_mode="EVALUATED",
        emission_mode="CONTINUOUS_INFLOW", auto_inject_seed=False, emit_rate=8.0,
        source_sampling_mode="MULTIPLIER", source_sampling_multiplier=1.0,
        collider_dynamic=False, collider_static=False, collider_initial_velocity=(0.0, 0.0, 0.0),
        initial_velocity=(0.0, 0.0, 0.0), use_object_velocity=True,
    )
    values.update(object_values)
    settings = _RnaObjectSettings(lambda: scene, pending, _object_update_actions(), **values)
    emitter = SimpleNamespace(name="Emitter", fluidsim=settings, location=(0.0, 0.0, 1.0))
    scene.objects.append(emitter)
    pending.capture_applied_state(scene)
    return scene, emitter


def _context(scene, obj):
    return SimpleNamespace(scene=scene, active_object=obj, object=obj,
                           region=SimpleNamespace(), region_data=SimpleNamespace())


def _event(x, y=0.0, value="PRESS"):
    return SimpleNamespace(mouse_region_x=x, mouse_region_y=y, type="MOUSEMOVE", value=value)


@pytest.fixture
def env(monkeypatch):
    bpy, pending, deck = _environment(monkeypatch)
    # One metre of world motion per 100 px across the screen: x -> world x, y -> world z.
    monkeypatch.setattr(deck.gizmos, "mouse_to_world",
                        lambda _context, mouse, _depth: (mouse[0] / 100.0, 0.0, mouse[1] / 100.0))
    return SimpleNamespace(bpy=bpy, pending=pending, deck=deck)


def _chip(env, scene):
    return env.deck.bar.chip_state(scene, now=0.5)


def test_object_update_actions_are_parsed_from_properties():
    actions = _object_update_actions()
    assert actions["initial_velocity"] == "SYNC"
    assert actions["emit_rate"] == "RESET"
    assert actions["source_sampling_multiplier"] == "RESET"
    assert actions["collider_friction"] == "SYNC"


def test_emitter_arrow_drag_sets_the_velocity_property_like_the_panel(env, monkeypatch):
    monkeypatch.setattr(env.deck.bar, "_now", lambda: 0.0)
    scene, emitter = _emitter_scene(env.pending)
    context = _context(scene, emitter)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_velocity()
    assert gizmo.invoke(context, _event(0.0)) == {"RUNNING_MODAL"}
    assert gizmo.modal(context, _event(50.0, 25.0), set()) == {"RUNNING_MODAL"}
    gizmo.exit(context, False)

    scale = env.deck.gizmos.ARROW_SCALE
    expected = (0.5 / scale, 0.0, 0.25 / scale)
    assert emitter.fluidsim.initial_velocity == pytest.approx(expected)
    assert "initial_velocity" in emitter.fluidsim.writes

    # The panel path: type the same value into the same property.
    panel_scene, panel_emitter = _emitter_scene(env.pending)
    panel_emitter.fluidsim.initial_velocity = expected
    gizmo_pending = env.pending.pending_presentation(scene)
    panel_pending = env.pending.pending_presentation(panel_scene)
    assert gizmo_pending["action"] == panel_pending["action"] == "NONE"
    assert gizmo_pending["reasons"] == panel_pending["reasons"]
    chip = _chip(env, scene)
    assert chip.kind == "LIVE" and "initial velocity" in chip.text.lower()


def test_emitter_arrow_cancel_restores_the_value(env):
    scene, emitter = _emitter_scene(env.pending, initial_velocity=(1.0, 0.0, 0.0))
    context = _context(scene, emitter)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_velocity()
    gizmo.invoke(context, _event(0.0))
    gizmo.modal(context, _event(80.0), set())
    gizmo.exit(context, True)
    assert emitter.fluidsim.initial_velocity == pytest.approx((1.0, 0.0, 0.0))


def test_rate_dial_drag_matches_the_panel_reset_classification(env):
    scene, emitter = _emitter_scene(env.pending)
    context = _context(scene, emitter)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_dial()
    gizmo.binding = "rate"
    gizmo.invoke(context, _event(0.0))
    gizmo.modal(context, _event(100.0), set())
    gizmo.exit(context, False)
    value = emitter.fluidsim.source_sampling_multiplier
    assert value != pytest.approx(1.0)

    panel_scene, panel_emitter = _emitter_scene(env.pending)
    panel_emitter.fluidsim.source_sampling_multiplier = value
    gizmo_pending = env.pending.pending_presentation(scene)
    panel_pending = env.pending.pending_presentation(panel_scene)
    assert gizmo_pending["action"] == panel_pending["action"] == "RESET_PENDING"
    assert gizmo_pending["reasons"] == panel_pending["reasons"]
    chip = _chip(env, scene)
    assert chip.kind == "RESET_PENDING"
    assert [op for op, _label in chip.actions] == ["fluidsim.apply_changes",
                                                  "fluidsim.revert_pending"]


def test_rate_dial_binds_the_property_the_panel_draws(env):
    gizmos = env.deck.gizmos
    multiplier = SimpleNamespace(source_sampling_mode="MULTIPLIER")
    legacy = SimpleNamespace(source_sampling_mode="LEGACY_ABSOLUTE")
    other = SimpleNamespace(source_sampling_mode="AUTO")
    assert gizmos.rate_property(multiplier) == "source_sampling_multiplier"
    assert gizmos.rate_property(legacy) == "emit_rate"
    assert gizmos.rate_property(other) is None


def test_dial_clamps_to_the_property_range(env):
    scene, emitter = _emitter_scene(env.pending)
    emitter.fluidsim.role = "COLLIDER"
    object.__setattr__(emitter.fluidsim, "collider_friction", 0.9)
    context = _context(scene, emitter)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_dial()
    gizmo.binding = "friction"
    gizmo.invoke(context, _event(0.0))
    gizmo.modal(context, _event(10_000.0), set())
    gizmo.exit(context, False)
    assert emitter.fluidsim.collider_friction == pytest.approx(1.0)


def test_dynamic_toggle_binds_only_when_the_property_exists(env):
    gizmos = env.deck.gizmos
    assert gizmos.dynamic_property(SimpleNamespace(collider_friction=0.0)) is None
    assert gizmos.dynamic_property(SimpleNamespace(collider_dynamic=False)) == "collider_dynamic"


def _domain(divisions=80):
    corners = [(x, y, z) for x in (-2.5, 2.5) for y in (-2.0, 2.0) for z in (-1.5, 1.5)]
    fs = full_settings(resolution_divisions=divisions)
    scene = SimpleNamespace(fluidsim=split_settings(fs), objects=[], name="Scene")
    domain = scene.fluidsim.domain_object
    domain.name = "Domain"
    domain.bound_box = corners
    domain.location = [0.0, 0.0, 0.0]
    domain.scale = [1.0, 1.0, 1.0]
    domain.matrix_world = None
    scene.objects.append(domain)
    return scene, domain


def test_domain_face_drag_previews_the_same_grid_as_typing_the_size(env):
    from fluidSim_warp_blender51.sim.quality_contract import (
        resolve_quality,
        voxel_size_for_divisions,
    )

    scene, domain = _domain()
    context = _context(scene, domain)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_face()
    gizmo.axis, gizmo.side = 0, 1
    gizmo.invoke(context, _event(0.0))
    gizmo.modal(context, _event(140.0), set())  # +1.4 m on the +X face
    preview = env.deck.gizmos.active_domain_preview()
    assert preview is not None

    # Typing the new X dimension (6.4 m) in the panel resolves this grid.
    typed_bounds = ((-3.2, -2.0, -1.5), (3.2, 2.0, 1.5))
    typed = resolve_quality(
        SimpleNamespace(voxel_size=voxel_size_for_divisions(80, 6.4)), typed_bounds)
    assert preview.quality.voxel_size == pytest.approx(typed.voxel_size)
    assert preview.quality.padded_voxel_dimensions == typed.padded_voxel_dimensions
    assert preview.quality.unpadded_voxel_dimensions == typed.unpadded_voxel_dimensions
    assert "80 × 50 × 38" in preview.label
    # Written live while dragging: the opposite face stays put.
    assert domain.scale == pytest.approx([1.28, 1.0, 1.0])
    assert domain.location == pytest.approx([0.7, 0.0, 0.0])

    gizmo.exit(context, False)
    assert env.deck.gizmos.active_domain_preview() is None
    # Release keeps the last drag step; divisions are untouched.
    assert domain.scale == pytest.approx([1.28, 1.0, 1.0])
    assert domain.location == pytest.approx([0.7, 0.0, 0.0])
    assert domain.fluidsim_domain.resolution_divisions == 80


def test_domain_face_drag_follows_every_step(env):
    scene, domain = _domain()
    context = _context(scene, domain)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_face()
    gizmo.axis, gizmo.side = 0, 1
    gizmo.invoke(context, _event(0.0))
    seen = []
    for x in (40.0, 80.0, 140.0):
        gizmo.modal(context, _event(x), set())
        seen.append(domain.scale[0])
    assert seen == sorted(seen) and len(set(seen)) == 3
    assert seen[-1] == pytest.approx(1.28)


def test_domain_face_drag_cancel_restores_the_start_box(env):
    scene, domain = _domain()
    context = _context(scene, domain)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_face()
    gizmo.axis, gizmo.side = 2, -1
    gizmo.invoke(context, _event(0.0, 0.0))
    gizmo.modal(context, _event(0.0, -100.0), set())
    assert domain.scale != [1.0, 1.0, 1.0]
    gizmo.exit(context, True)
    assert domain.scale == [1.0, 1.0, 1.0]
    assert domain.location == [0.0, 0.0, 0.0]
    assert env.deck.gizmos.active_domain_preview() is None


def _drag_domain(env):
    scene, domain = _domain()
    context = _context(scene, domain)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_face()
    gizmo.axis, gizmo.side = 0, 1
    gizmo.invoke(context, _event(0.0))
    gizmo.modal(context, _event(140.0), set())
    return scene, domain, context, gizmo


@pytest.mark.parametrize("active", ["none", "other"])
def test_domain_face_release_writes_the_domain_it_started_on(env, active):
    scene, domain, context, gizmo = _drag_domain(env)
    other = SimpleNamespace(name="Cube", location=[5.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0])
    scene.objects.append(other)
    context.active_object = None if active == "none" else other
    gizmo.exit(context, False)
    assert domain.scale == pytest.approx([1.28, 1.0, 1.0])
    assert domain.location == pytest.approx([0.7, 0.0, 0.0])
    assert other.scale == [1.0, 1.0, 1.0] and other.location == [5.0, 0.0, 0.0]


@pytest.mark.parametrize("cancel", [False, True])
def test_domain_face_is_a_no_op_once_the_domain_is_gone(env, cancel):
    scene, domain, context, gizmo = _drag_domain(env)
    written = (list(domain.scale), list(domain.location))
    scene.objects.remove(domain)
    context.active_object = None
    gizmo.modal(context, _event(200.0), set())
    gizmo.exit(context, cancel)
    assert (domain.scale, domain.location) == written
    assert env.deck.gizmos.active_domain_preview() is None


@pytest.mark.parametrize("kind", ["velocity", "dial"])
def test_cancel_restores_the_emitter_it_started_on_after_the_active_object_is_gone(env, kind):
    scene, emitter = _emitter_scene(env.pending, initial_velocity=(1.0, 0.0, 0.0))
    context = _context(scene, emitter)
    if kind == "velocity":
        gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_velocity()
        prop, start = "initial_velocity", (1.0, 0.0, 0.0)
    else:
        gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_dial()
        gizmo.binding = "rate"
        prop, start = "source_sampling_multiplier", 1.0
    gizmo.invoke(context, _event(0.0))
    gizmo.modal(context, _event(80.0), set())
    assert getattr(emitter.fluidsim, prop) != pytest.approx(start)
    context.active_object = None
    gizmo.exit(context, True)
    assert getattr(emitter.fluidsim, prop) == pytest.approx(start)
    # Once the emitter is gone too, exit does nothing.
    scene.objects.remove(emitter)
    gizmo.exit(context, True)


def test_toggle_flips_the_collider_it_started_on_and_no_ops_when_gone(env):
    scene, collider = _emitter_scene(env.pending, role="COLLIDER")
    context = _context(scene, collider)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_toggle()
    gizmo.invoke(context, _event(0.0))
    context.active_object = None
    gizmo.exit(context, False)
    assert collider.fluidsim.collider_dynamic is True
    scene.objects.remove(collider)
    gizmo.exit(context, False)
    assert collider.fluidsim.collider_dynamic is True


def test_face_resize_math_keeps_the_opposite_face_fixed(env):
    resize = env.deck.gizmos.face_resize
    scale, shift = resize(-2.5, 2.5, 1.0, side=1, delta=1.4)
    assert scale == pytest.approx(1.28) and shift == pytest.approx(0.7)
    scale, shift = resize(-2.5, 2.5, 1.0, side=-1, delta=1.4)
    assert scale == pytest.approx(1.28) and shift == pytest.approx(-0.7)
    # A drag past the opposite face stops at the minimum size instead of inverting.
    scale, _shift = resize(-2.5, 2.5, 1.0, side=1, delta=-50.0)
    assert scale > 0.0


# Undo, precision, snapping, ranges, toggle clicks, matrix_world and the bake gate ----

def _record_undo(env):
    pushes = []
    env.bpy.ops.ed = SimpleNamespace(undo_push=lambda message: pushes.append(message))
    return pushes


def _velocity_drag(env, scene, emitter):
    context = _context(scene, emitter)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_velocity()
    gizmo.invoke(context, _event(0.0))
    return context, gizmo


@pytest.mark.parametrize("cancel", [False, True])
def test_a_committed_drag_pushes_one_undo_step_and_a_cancelled_one_none(env, cancel):
    pushes = _record_undo(env)
    scene, emitter = _emitter_scene(env.pending)
    context, gizmo = _velocity_drag(env, scene, emitter)
    for x in (10.0, 20.0, 30.0):
        gizmo.modal(context, _event(x), set())
    gizmo.exit(context, cancel)

    dial = env.deck.gizmos.FLUIDSIM_GT_deck_dial()
    dial.binding = "rate"
    dial.invoke(context, _event(0.0))
    dial.modal(context, _event(40.0), set())
    dial.exit(context, cancel)

    scene_d, domain = _domain()
    context_d = _context(scene_d, domain)
    face = env.deck.gizmos.FLUIDSIM_GT_deck_face()
    face.axis, face.side = 0, 1
    face.invoke(context_d, _event(0.0))
    face.modal(context_d, _event(50.0), set())
    face.exit(context_d, cancel)
    assert len(pushes) == (0 if cancel else 3)


def test_a_toggle_click_pushes_one_undo_step(env):
    pushes = _record_undo(env)
    scene, collider = _emitter_scene(env.pending, role="COLLIDER")
    context = _context(scene, collider)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_toggle()
    gizmo.invoke(context, _event(0.0))
    gizmo.exit(context, False)
    assert len(pushes) == 1


def test_undo_push_tolerates_a_refusing_operator(env):
    def refuse(message):
        raise RuntimeError("context is incorrect")
    env.bpy.ops.ed = SimpleNamespace(undo_push=refuse)
    scene, emitter = _emitter_scene(env.pending)
    context, gizmo = _velocity_drag(env, scene, emitter)
    gizmo.modal(context, _event(40.0), set())
    gizmo.exit(context, False)
    assert emitter.fluidsim.initial_velocity[0] == pytest.approx(0.4 / env.deck.gizmos.ARROW_SCALE)


def test_shift_mid_drag_slows_from_where_it_was_pressed_without_a_jump(env):
    gizmos = env.deck.gizmos
    scene, emitter = _emitter_scene(env.pending)
    context, gizmo = _velocity_drag(env, scene, emitter)
    gizmo.modal(context, _event(50.0), set())
    before = emitter.fluidsim.initial_velocity[0]
    gizmo.modal(context, _event(60.0), {"PRECISE"})
    precise = emitter.fluidsim.initial_velocity[0]
    step = 0.1 / gizmos.ARROW_SCALE  # 10 px = 0.1 m of pointer travel
    assert precise == pytest.approx(before + step * gizmos.PRECISE_FACTOR)
    # Releasing Shift continues at full speed from there, again without a jump.
    gizmo.modal(context, _event(70.0), set())
    assert emitter.fluidsim.initial_velocity[0] == pytest.approx(precise + step)


def test_shift_mid_drag_on_a_dial_does_not_jump(env):
    gizmos = env.deck.gizmos
    scene, emitter = _emitter_scene(env.pending)
    context = _context(scene, emitter)
    dial = gizmos.FLUIDSIM_GT_deck_dial()
    dial.binding = "rate"
    dial.invoke(context, _event(0.0))
    dial.modal(context, _event(60.0), set())
    before = emitter.fluidsim.source_sampling_multiplier
    dial.modal(context, _event(90.0), {"PRECISE"})
    per_pixel = 8.0 / gizmos.DIAL_PIXELS
    assert emitter.fluidsim.source_sampling_multiplier == pytest.approx(
        before + 30.0 * per_pixel * gizmos.PRECISE_FACTOR)


def test_ctrl_snaps_velocity_fill_and_friction(env):
    gizmos = env.deck.gizmos
    scene, emitter = _emitter_scene(env.pending)
    context, gizmo = _velocity_drag(env, scene, emitter)
    gizmo.modal(context, _event(9.0, 4.0), {"SNAP"})  # 0.36, 0, 0.16 m/s
    assert emitter.fluidsim.initial_velocity == pytest.approx((0.5, 0.0, 0.0))

    dial = gizmos.FLUIDSIM_GT_deck_dial()
    dial.binding = "rate"
    dial.invoke(context, _event(0.0))
    dial.modal(context, _event(13.0), {"SNAP"})  # 1 + 0.347
    assert emitter.fluidsim.source_sampling_multiplier == pytest.approx(1.25)

    object.__setattr__(emitter.fluidsim, "collider_friction", 0.0)
    friction = gizmos.FLUIDSIM_GT_deck_dial()
    friction.binding = "friction"
    friction.invoke(context, _event(0.0))
    friction.modal(context, _event(20.0), {"SNAP"})  # 0.0667
    assert emitter.fluidsim.collider_friction == pytest.approx(0.05)


def test_ctrl_snaps_the_dragged_domain_size_to_a_tenth_of_a_metre(env):
    scene, domain = _domain()
    context = _context(scene, domain)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_face()
    gizmo.axis, gizmo.side = 0, 1
    gizmo.invoke(context, _event(0.0))
    gizmo.modal(context, _event(137.0), {"SNAP"})  # 5 m + 1.37 m -> 6.4 m
    assert domain.scale[0] * 5.0 == pytest.approx(6.4)
    assert domain.location[0] - 2.5 * domain.scale[0] == pytest.approx(-2.5)  # -X face fixed


class _Rna:
    def __init__(self, **ranges):
        self.properties = {name: SimpleNamespace(hard_min=r[0], hard_max=r[1], soft_min=r[2],
                                                 soft_max=r[3]) for name, r in ranges.items()}


@pytest.mark.parametrize("with_rna", [False, True])
def test_legacy_rate_dial_clamps_to_the_property_range_not_32(env, with_rna):
    from fluidSim_warp_blender51.sim import constants

    scene, emitter = _emitter_scene(env.pending, source_sampling_mode="LEGACY_ABSOLUTE",
                                    emit_rate=30.0)
    hard = float(constants.MAX_PARTICLES_PER_VOXEL)
    if with_rna:
        object.__setattr__(emitter.fluidsim, "bl_rna",
                           _Rna(emit_rate=(0.0, hard, 0.0, 32.0)))
    context = _context(scene, emitter)
    dial = env.deck.gizmos.FLUIDSIM_GT_deck_dial()
    dial.binding = "rate"
    dial.invoke(context, _event(0.0))
    dial.modal(context, _event(30.0), set())  # 30 px of a 0-32 sweep: +3.2
    assert emitter.fluidsim.emit_rate == pytest.approx(33.2)
    dial.modal(context, _event(10_000.0), set())
    assert emitter.fluidsim.emit_rate == pytest.approx(hard)


def test_dial_reads_a_narrower_rna_range(env):
    scene, emitter = _emitter_scene(env.pending)
    object.__setattr__(emitter.fluidsim, "bl_rna",
                       _Rna(source_sampling_multiplier=(0.0, 4.0, 0.0, 4.0)))
    context = _context(scene, emitter)
    dial = env.deck.gizmos.FLUIDSIM_GT_deck_dial()
    dial.binding = "rate"
    dial.invoke(context, _event(0.0))
    dial.modal(context, _event(10_000.0), set())
    assert emitter.fluidsim.source_sampling_multiplier == pytest.approx(4.0)


def test_a_drag_released_over_the_toggle_does_not_flip_it(env):
    scene, collider = _emitter_scene(env.pending, role="COLLIDER")
    context = _context(scene, collider)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_toggle()
    gizmo.invoke(context, _event(0.0))
    gizmo.modal(context, _event(40.0), set())
    gizmo.modal(context, _event(0.0), set())  # back over the toggle
    gizmo.exit(context, False)
    assert collider.fluidsim.collider_dynamic is False
    # A click with a pixel of jitter still flips it.
    gizmo.invoke(context, _event(0.0))
    gizmo.modal(context, _event(1.0, 1.0), set())
    gizmo.exit(context, False)
    assert collider.fluidsim.collider_dynamic is True


def _delta_domain():
    """Domain whose matrix_world differs from location/scale: delta location +2 m X,
    delta scale x2 on X, so the world box is X -3..7 m."""
    scene, domain = _domain()
    domain.matrix_world = [[2.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    domain.parent, domain.constraints = None, []
    return scene, domain


def test_domain_face_drag_follows_matrix_world(env):
    gizmos = env.deck.gizmos
    scene, domain = _delta_domain()
    context = _context(scene, domain)
    gizmo = gizmos.FLUIDSIM_GT_deck_face()
    gizmo.axis, gizmo.side = 0, 1
    gizmo.invoke(context, _event(0.0))
    assert gizmos.active_domain_preview().bounds[0][0] == pytest.approx(-3.0)
    assert gizmos.active_domain_preview().bounds[1][0] == pytest.approx(7.0)
    gizmo.modal(context, _event(140.0), set())  # +1.4 m on the +X face: 10 m -> 11.4 m
    lo, hi = gizmos.active_domain_preview().bounds
    assert (lo[0], hi[0]) == pytest.approx((-3.0, 8.4))
    # Written so matrix_world (delta scale x2, delta location +2) gives that box.
    assert domain.scale[0] == pytest.approx(1.14)
    world_scale = 2.0 * domain.scale[0]
    world_x = 2.0 + domain.location[0]
    assert (world_x - 2.5 * world_scale, world_x + 2.5 * world_scale) == pytest.approx(
        (-3.0, 8.4))


def test_domain_handles_are_not_offered_for_a_constrained_domain(env):
    gizmos = env.deck.gizmos
    _scene, domain = _delta_domain()
    assert gizmos._axis_aligned(domain)
    domain.constraints = [SimpleNamespace(enabled=True)]
    assert not gizmos._axis_aligned(domain)
    domain.constraints = [SimpleNamespace(enabled=False)]
    assert gizmos._axis_aligned(domain)


@pytest.fixture
def baking(env, monkeypatch):
    from fluidSim_warp_blender51.blender import production_bake

    state = SimpleNamespace(here=True, anywhere=True)
    monkeypatch.setattr(production_bake, "bake_is_active", lambda _scene: state.here)
    monkeypatch.setattr(production_bake, "any_bake_active", lambda: state.anywhere)
    return state


@pytest.mark.parametrize("where", ["here", "other_scene"])
def test_no_gizmo_starts_a_drag_during_a_production_bake(env, baking, where):
    baking.here = where == "here"
    pushes = _record_undo(env)
    gizmos = env.deck.gizmos
    scene, emitter = _emitter_scene(env.pending, role="COLLIDER", collider_friction=0.0,
                                    initial_velocity=(1.0, 0.0, 0.0))
    context = _context(scene, emitter)
    velocity = gizmos.FLUIDSIM_GT_deck_velocity()
    dial = gizmos.FLUIDSIM_GT_deck_dial()
    dial.binding = "friction"
    toggle = gizmos.FLUIDSIM_GT_deck_toggle()
    scene_d, domain = _domain()
    face = gizmos.FLUIDSIM_GT_deck_face()
    face.axis, face.side = 0, 1
    for gizmo, ctx in ((velocity, context), (dial, context), (toggle, context),
                       (face, _context(scene_d, domain))):
        assert gizmo.invoke(ctx, _event(0.0)) == {"CANCELLED"}
        gizmo.modal(ctx, _event(100.0), set())
        gizmo.exit(ctx, False)
    assert emitter.fluidsim.initial_velocity == (1.0, 0.0, 0.0)
    assert emitter.fluidsim.collider_friction == 0.0
    assert emitter.fluidsim.collider_dynamic is False
    assert domain.scale == [1.0, 1.0, 1.0] and domain.location == [0.0, 0.0, 0.0]
    assert gizmos.active_domain_preview() is None
    assert pushes == []


def test_handles_draw_dimmed_during_a_production_bake(env, baking):
    gizmos = env.deck.gizmos
    group = SimpleNamespace()
    handles = [SimpleNamespace(alpha=0.8, alpha_highlight=1.0) for _ in range(2)]
    assert gizmos.show_held(group, SimpleNamespace(scene=None), handles) is True
    assert [h.alpha for h in handles] == pytest.approx([0.8 * gizmos.DIM_ALPHA] * 2)
    assert [h.alpha_highlight for h in handles] == [h.alpha for h in handles]
    baking.here = baking.anywhere = False
    assert gizmos.show_held(group, SimpleNamespace(scene=None), handles) is False
    assert [(h.alpha, h.alpha_highlight) for h in handles] == [(0.8, 1.0)] * 2
