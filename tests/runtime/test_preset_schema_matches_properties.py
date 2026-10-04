"""A preset may not promise what the Blender property would silently change.

``presets.schema.SETTING_SPECS`` validates preset values without ``bpy``; the
authored value lands in a Blender property whose ``min``/``max`` clamp
silently. If the schema admits a value the property clamps, a preset "applies"
as something other than what it declares. These tests pin the two declarations
of each setting's type, hard range and choices to one another.
"""

from __future__ import annotations

import ast
import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.presets.schema import SETTING_SPECS

_KIND = {
    "IntProperty": int,
    "FloatProperty": float,
    "BoolProperty": bool,
    "EnumProperty": str,
    "StringProperty": str,
}


@pytest.fixture(scope="module")
def declarations():
    """{setting: [(group_name, property_kind, {keyword: value})]} from the real
    ``properties.py`` annotations, evaluated in its own namespace."""
    mp = pytest.MonkeyPatch()
    try:
        fake_bpy = types.ModuleType("bpy")
        fake_bpy.types = SimpleNamespace(
            PropertyGroup=object, Scene=type("_Scene", (), {}),
            Object=type("_Object", (), {}), Material=object, Image=object, Text=object)
        fake_bpy.props = SimpleNamespace(**{
            name: (lambda **kwargs: kwargs.get("default"))
            for name in (
                "BoolProperty", "CollectionProperty", "EnumProperty", "FloatProperty",
                "FloatVectorProperty", "IntProperty", "PointerProperty", "StringProperty")})
        fake_bpy.utils = SimpleNamespace()
        mp.setitem(sys.modules, "bpy", fake_bpy)
        mp.setitem(sys.modules, "bpy.props", fake_bpy.props)
        mp.delitem(sys.modules, "fluidSim_warp_blender51.properties", raising=False)
        properties = importlib.import_module("fluidSim_warp_blender51.properties")
        namespace = vars(properties)

        found: dict[str, list] = {}
        for group in (properties.FLUIDSIM_PG_scene, properties.FLUIDSIM_PG_domain):
            for name, source in group.__annotations__.items():
                call = ast.parse(source, mode="eval").body
                if not isinstance(call, ast.Call):
                    continue
                kwargs = {
                    keyword.arg: eval(  # noqa: S307 - our own annotation source
                        compile(ast.Expression(keyword.value), "<annotation>", "eval"),
                        namespace)
                    for keyword in call.keywords
                }
                found.setdefault(name, []).append(
                    (group.__name__, ast.unparse(call.func), kwargs))
        return found
    finally:
        mp.undo()
        sys.modules.pop("fluidSim_warp_blender51.properties", None)


def test_every_preset_setting_is_a_declared_property(declarations):
    assert sorted(set(SETTING_SPECS) - set(declarations)) == []


@pytest.mark.parametrize("key", sorted(SETTING_SPECS))
def test_preset_spec_agrees_with_the_property_it_assigns(key, declarations):
    spec = SETTING_SPECS[key]
    assert declarations[key], f"{key} is not declared on a settings group"
    for group, kind, kwargs in declarations[key]:
        where = f"{group}.{key}"
        assert _KIND[kind] is spec.value_type, (
            f"{where} is a {kind} but the preset schema types it "
            f"{spec.value_type.__name__}")

        if spec.value_type in (int, float):
            hard_min, hard_max = kwargs.get("min"), kwargs.get("max")
            if hard_min is not None:
                assert spec.minimum is not None and spec.minimum >= hard_min, (
                    f"{where} clamps below {hard_min} but the preset schema admits "
                    f"{spec.minimum}")
            if hard_max is not None:
                assert spec.maximum is not None and spec.maximum <= hard_max, (
                    f"{where} clamps above {hard_max} but the preset schema admits "
                    f"{spec.maximum}")

        if kind == "EnumProperty":
            identifiers = {item[0] for item in kwargs["items"]}
            assert set(spec.choices) == identifiers, (
                f"{where} offers {sorted(identifiers)} but the preset schema "
                f"allows {sorted(spec.choices)}")
