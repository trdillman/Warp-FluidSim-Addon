"""Scene-ownership regressions for Blender property mode callbacks."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _load_properties(monkeypatch):
    fake_bpy = types.ModuleType("bpy")

    class _Scene:
        pass

    class _Object:
        pass

    def prop(**kwargs):
        return kwargs.get("default")

    fake_bpy.types = SimpleNamespace(
        PropertyGroup=object,
        Scene=_Scene,
        Object=_Object,
        Material=object,
        Image=object,
        Text=object,
    )
    fake_bpy.props = SimpleNamespace(
        BoolProperty=lambda **kwargs: prop(**kwargs),
        CollectionProperty=lambda **kwargs: prop(**kwargs),
        EnumProperty=lambda **kwargs: prop(**kwargs),
        FloatProperty=lambda **kwargs: prop(**kwargs),
        FloatVectorProperty=lambda **kwargs: prop(**kwargs),
        IntProperty=lambda **kwargs: prop(**kwargs),
        PointerProperty=lambda **kwargs: prop(**kwargs),
        StringProperty=lambda **kwargs: prop(**kwargs),
    )
    fake_bpy.utils = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_bpy.props)
    sys.modules.pop("fluidSim_warp_blender51.properties", None)
    return fake_bpy, importlib.import_module("fluidSim_warp_blender51.properties")


@pytest.mark.parametrize(
    ("callback_name", "value_name", "playback_method"),
    [
        ("_on_live_changed", "live", "set_live_enabled"),
        (
            "_on_cache_playback_changed",
            "cache_playback_enabled",
            "set_cache_playback_enabled",
        ),
    ],
)
def test_mode_callback_uses_property_group_scene_owner_after_context_switch(
    monkeypatch, callback_name, value_name, playback_method
):
    fake_bpy, properties = _load_properties(monkeypatch)
    scene_a = fake_bpy.types.Scene()
    scene_b = fake_bpy.types.Scene()
    calls = []
    fake_playback = SimpleNamespace(
        set_live_enabled=lambda scene, enabled: calls.append(
            ("set_live_enabled", scene, enabled)
        ),
        set_cache_playback_enabled=lambda scene, enabled: calls.append(
            ("set_cache_playback_enabled", scene, enabled)
        ),
    )
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setattr(blender_pkg, "playback", fake_playback, raising=False)
    settings = SimpleNamespace(id_data=scene_a, status="")
    setattr(settings, value_name, True)

    getattr(properties, callback_name)(settings, SimpleNamespace(scene=scene_b))

    assert calls == [(playback_method, scene_a, True)]


@pytest.mark.parametrize(
    ("callback_name", "value_name", "playback_method"),
    [
        ("_on_live_changed", "live", "set_live_enabled"),
        (
            "_on_cache_playback_changed",
            "cache_playback_enabled",
            "set_cache_playback_enabled",
        ),
    ],
)
def test_mode_callback_falls_back_to_context_for_ownerless_fixture(
    monkeypatch, callback_name, value_name, playback_method
):
    fake_bpy, properties = _load_properties(monkeypatch)
    context_scene = fake_bpy.types.Scene()
    calls = []
    fake_playback = SimpleNamespace(
        set_live_enabled=lambda scene, enabled: calls.append(
            ("set_live_enabled", scene, enabled)
        ),
        set_cache_playback_enabled=lambda scene, enabled: calls.append(
            ("set_cache_playback_enabled", scene, enabled)
        ),
    )
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setattr(blender_pkg, "playback", fake_playback, raising=False)
    settings = SimpleNamespace(status="")
    setattr(settings, value_name, False)

    getattr(properties, callback_name)(
        settings, SimpleNamespace(scene=context_scene)
    )

    assert calls == [(playback_method, context_scene, False)]
