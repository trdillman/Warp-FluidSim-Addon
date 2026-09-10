"""Exhaustive/disjoint Pending Settings transition registry contracts."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
import sys
import textwrap
import types
from types import SimpleNamespace


def _modules(monkeypatch):
    bpy = types.ModuleType("bpy")
    bpy.context = SimpleNamespace()
    bpy.app = SimpleNamespace(handlers=SimpleNamespace(frame_change_post=[]))

    class _Operator:
        @classmethod
        def poll_message_set(cls, _message):
            pass

        def report(self, *_args, **_kwargs):
            pass

    bpy.types = SimpleNamespace(
        PropertyGroup=object,
        Text=object,
        Object=object,
        Material=object,
        Image=object,
        Operator=_Operator,
        Panel=object,
        AddonPreferences=object,
    )
    bpy.utils = SimpleNamespace(
        register_class=lambda _cls: None, unregister_class=lambda _cls: None
    )
    bpy.props = types.ModuleType("bpy.props")
    for name in (
        "BoolProperty",
        "CollectionProperty",
        "EnumProperty",
        "FloatProperty",
        "FloatVectorProperty",
        "IntProperty",
        "PointerProperty",
        "StringProperty",
    ):
        setattr(bpy.props, name, lambda **kwargs: kwargs.get("default"))
    monkeypatch.setitem(sys.modules, "bpy.props", bpy.props)
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    for name in (
        "fluidSim_warp_blender51.properties",
        "fluidSim_warp_blender51.blender.pending_settings",
        "fluidSim_warp_blender51.blender.scene_bridge",
    ):
        sys.modules.pop(name, None)
    properties = importlib.import_module("fluidSim_warp_blender51.properties")
    pending = importlib.import_module(
        "fluidSim_warp_blender51.blender.pending_settings"
    )
    bridge = importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")
    return properties, pending, bridge


def _declared_scene_fields(properties):
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(properties.FLUIDSIM_PG_settings))
    )
    return {
        node.target.id
        for node in tree.body[0].body
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Call)
        )
    }


def test_transition_registry_is_exhaustive_and_disjoint(monkeypatch):
    properties, pending, _bridge = _modules(monkeypatch)
    declared = _declared_scene_fields(properties)

    pending.validate_transition_registry(declared)

    assert set(pending.SETTING_TRANSITION_REGISTRY) == declared
    assert all(
        isinstance(tier, pending.SettingTier)
        for tier in pending.SETTING_TRANSITION_REGISTRY.values()
    )
    assert len(pending.SETTING_TRANSITION_REGISTRY) == len(declared)


def test_special_particle_capacity_policy_is_explicit(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)

    assert pending.SETTING_TRANSITION_REGISTRY["max_particles"] is (
        pending.SettingTier.REINITIALIZE
    )
    assert pending.SETTING_TRANSITION_REGISTRY["auto_grow_particles"] is (
        pending.SettingTier.LIVE
    )
    assert (
        pending.SETTING_TRANSITION_REGISTRY["particle_growth_failure_policy"]
        is pending.SettingTier.LIVE
    )
    # Preserve the characterized Phase-3 behavior: this authored value is not
    # pushed into an active solver by the surface refresh callback.
    assert (
        pending.SETTING_TRANSITION_REGISTRY["spacetime_resync_surface"]
        is pending.SettingTier.AUTHORED_NEXT_RUN
    )


def test_quality_tier_is_reinitialize_and_detected_by_reinit_signature(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)

    # The scene-level Quality dial rescales solver knobs, so a change must
    # re-derive grid allocation (rebuild), not read as a pure presentation edit.
    assert (
        pending.SETTING_TRANSITION_REGISTRY["quality"]
        is pending.SettingTier.REINITIALIZE
    )
    assert "quality" not in pending._PRESENTATION_FIELDS
    # The reinit signature (which drives auto-detection) reads scene.fluidsim,
    # exactly where quality is authored, so quality must be in its field set.
    assert "quality" in pending._REINIT_SETTING_FIELDS


def test_quality_property_wires_the_rebuild_update_callback(monkeypatch):
    """A REINITIALIZE tier classification is inert without an RNA callback.

    The stub in ``_modules`` discards every ``bpy.props.*`` keyword (it only
    ever returns ``default``), so a change to the ``update=`` argument cannot
    be observed by calling the property -- it has to be read out of source.
    Without this, a UI edit to Quality would only refresh cached presentation
    on the depsgraph handler and never reach ``request_scene_update()``,
    leaving an active solver running the previous voxel size/substeps/sampling
    until the user manually applies the pending action or rewinds.
    """

    properties, _pending, _bridge = _modules(monkeypatch)
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(properties.FLUIDSIM_PG_settings))
    )
    (quality_node,) = (
        node
        for node in tree.body[0].body
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "quality"
        )
    )
    update_kwargs = {
        kw.arg: kw.value
        for kw in quality_node.annotation.keywords
        if kw.arg == "update"
    }
    assert "update" in update_kwargs, (
        "quality's EnumProperty has no update= callback; UI edits to Quality "
        "will not trigger a REINITIALIZE rebuild"
    )
    assert (
        isinstance(update_kwargs["update"], ast.Name)
        and update_kwargs["update"].id == "_request_rebuild"
    )


def test_species_toggle_rebuilds_storage_but_rate_updates_live(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)

    assert (
        pending.SETTING_TRANSITION_REGISTRY["species_enabled"]
        is pending.SettingTier.RESET
    )
    assert (
        pending.SETTING_TRANSITION_REGISTRY["species_mixing_rate"]
        is pending.SettingTier.LIVE
    )


def test_scene_bridge_no_longer_owns_pending_policy(monkeypatch):
    properties, pending, bridge = _modules(monkeypatch)
    source = inspect.getsource(bridge)

    assert "class SceneUpdate" not in source
    assert "_INTERACTION_STATES" not in source
    assert "def pending_presentation" not in source
    assert "def request_scene_update" not in source
    assert "def apply_pending_" not in source
    assert "def build_snippets" not in source
    assert "def validate_snippet_pg" not in source
    assert "def gather_sources" not in source
    assert "def gather_colliders" not in source
    assert "def register_handlers" not in source
    assert pending.SceneUpdate.REBUILD.value == "REBUILD"
    assert "pending_settings.request_scene_update" in inspect.getsource(properties)


def test_adapters_delegate_pending_policy_to_deep_owner(monkeypatch):
    _properties, _pending, _bridge = _modules(monkeypatch)
    root = Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51"
    sources = [
        (root / "operator_domains" / "common.py").read_text(encoding="utf-8"),
        (root / "panels.py").read_text(encoding="utf-8"),
        (root / "blender" / "ui_common.py").read_text(encoding="utf-8"),
        (root / "blender" / "ui_scene.py").read_text(encoding="utf-8"),
        (root / "blender" / "viewport_overlay.py").read_text(encoding="utf-8"),
        (root / "blender" / "playback.py").read_text(encoding="utf-8"),
    ]

    assert "def _pending" in sources[0]
    assert "pending_ui_state" in sources[1]
    assert "pending_settings import pending_presentation" in sources[2]
    assert "pending_ui_state" in sources[3]
    assert "pending_settings import pending_presentation" in sources[4]
    assert "pending_settings.apply_pending_on_rewind" in sources[5]
