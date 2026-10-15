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
        collider_dynamic=False, collider_initial_velocity=(0.0, 0.0, 0.0),
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
    assert chip.kind == "LIVE" and "initial_velocity" in chip.text


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
    # Nothing is written while dragging.
    assert domain.scale == [1.0, 1.0, 1.0]

    gizmo.exit(context, False)
    assert env.deck.gizmos.active_domain_preview() is None
    # Commit on release: the opposite face stays put, divisions are untouched.
    assert domain.scale == pytest.approx([1.28, 1.0, 1.0])
    assert domain.location == pytest.approx([0.7, 0.0, 0.0])
    assert domain.fluidsim_domain.resolution_divisions == 80


def test_domain_face_drag_cancel_writes_nothing(env):
    scene, domain = _domain()
    context = _context(scene, domain)
    gizmo = env.deck.gizmos.FLUIDSIM_GT_deck_face()
    gizmo.axis, gizmo.side = 2, -1
    gizmo.invoke(context, _event(0.0, 0.0))
    gizmo.modal(context, _event(0.0, -100.0), set())
    gizmo.exit(context, True)
    assert domain.scale == [1.0, 1.0, 1.0]
    assert domain.location == [0.0, 0.0, 0.0]
    assert env.deck.gizmos.active_domain_preview() is None


def test_face_resize_math_keeps_the_opposite_face_fixed(env):
    resize = env.deck.gizmos.face_resize
    scale, shift = resize(-2.5, 2.5, 1.0, side=1, delta=1.4)
    assert scale == pytest.approx(1.28) and shift == pytest.approx(0.7)
    scale, shift = resize(-2.5, 2.5, 1.0, side=-1, delta=1.4)
    assert scale == pytest.approx(1.28) and shift == pytest.approx(-0.7)
    # A drag past the opposite face stops at the minimum size instead of inverting.
    scale, _shift = resize(-2.5, 2.5, 1.0, side=1, delta=-50.0)
    assert scale > 0.0
