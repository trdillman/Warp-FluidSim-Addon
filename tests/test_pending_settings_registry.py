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
        PropertyGroup=object, Text=object, Object=object, Material=object,
        Image=object, Operator=_Operator, Panel=object, AddonPreferences=object,
    )
    bpy.utils = SimpleNamespace(register_class=lambda _cls: None,
                                unregister_class=lambda _cls: None)
    bpy.props = types.ModuleType("bpy.props")
    for name in (
            "BoolProperty", "CollectionProperty", "EnumProperty",
            "FloatProperty", "FloatVectorProperty", "IntProperty",
            "PointerProperty", "StringProperty"):
        setattr(bpy.props, name, lambda **kwargs: kwargs.get("default"))
    monkeypatch.setitem(sys.modules, "bpy.props", bpy.props)
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    for name in (
            "fluidSim_warp_blender51.properties",
            "fluidSim_warp_blender51.blender.pending_settings",
            "fluidSim_warp_blender51.blender.scene_bridge"):
        sys.modules.pop(name, None)
    properties = importlib.import_module("fluidSim_warp_blender51.properties")
    pending = importlib.import_module(
        "fluidSim_warp_blender51.blender.pending_settings")
    bridge = importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")
    return properties, pending, bridge


def _declared_scene_fields(properties):
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(properties.FLUIDSIM_PG_settings)))
    return {
        node.target.id
        for node in tree.body[0].body
        if (isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Call))
    }


def test_transition_registry_is_exhaustive_and_disjoint(monkeypatch):
    properties, pending, _bridge = _modules(monkeypatch)
    declared = _declared_scene_fields(properties)

    pending.validate_transition_registry(declared)

    assert set(pending.SETTING_TRANSITION_REGISTRY) == declared
    assert all(isinstance(tier, pending.SettingTier)
               for tier in pending.SETTING_TRANSITION_REGISTRY.values())
    assert len(pending.SETTING_TRANSITION_REGISTRY) == len(declared)


def test_special_particle_capacity_policy_is_explicit(monkeypatch):
    _properties, pending, _bridge = _modules(monkeypatch)

    assert pending.SETTING_TRANSITION_REGISTRY["max_particles"] is (
        pending.SettingTier.AUTHORED_NEXT_RUN)
    assert pending.SETTING_TRANSITION_REGISTRY["auto_grow_particles"] is (
        pending.SettingTier.LIVE)
    assert pending.SETTING_TRANSITION_REGISTRY[
        "particle_growth_failure_policy"] is pending.SettingTier.LIVE
    # Preserve the characterized Phase-3 behavior: this authored value is not
    # pushed into an active solver by the surface refresh callback.
    assert pending.SETTING_TRANSITION_REGISTRY[
        "spacetime_resync_surface"] is pending.SettingTier.AUTHORED_NEXT_RUN


def test_scene_bridge_no_longer_owns_pending_policy(monkeypatch):
    properties, pending, bridge = _modules(monkeypatch)
    source = inspect.getsource(bridge)

    assert "class SceneUpdate" not in source
    assert "_INTERACTION_STATES" not in source
    assert "def pending_presentation" not in source
    assert "def request_scene_update" not in source
    assert "def apply_pending_" not in source
    assert pending.SceneUpdate.REBUILD.value == "REBUILD"
    assert "pending_settings.request_scene_update" in inspect.getsource(properties)


def test_adapters_delegate_pending_policy_to_deep_owner(monkeypatch):
    _properties, _pending, _bridge = _modules(monkeypatch)
    root = Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51"
    sources = [
        (root / "operators.py").read_text(encoding="utf-8"),
        (root / "panels.py").read_text(encoding="utf-8"),
        (root / "blender" / "viewport_overlay.py").read_text(encoding="utf-8"),
        (root / "blender" / "playback.py").read_text(encoding="utf-8"),
    ]

    assert "def _pending" in sources[0]
    assert "pending_settings import pending_presentation" in sources[1]
    assert "pending_settings import pending_presentation" in sources[2]
    assert "pending_settings.apply_pending_on_rewind" in sources[3]
