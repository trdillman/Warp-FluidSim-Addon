"""Unit tests for the Custom Effects Authoring Surface."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from fluidsim_testpkg.sim.effects.enums import EffectCadence, EffectDomain, EffectStage
from fluidsim_testpkg.sim.effects.templates import (
    get_template_definition,
    starter_template_items,
)
from fluidsim_testpkg.sim.effects.validation import validate_effect

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "blender" / "effect_workspace.py"
EFFECTS = ROOT / "blender" / "effects.py"


def test_starter_templates_are_valid_and_pass_ast_validation():
    items = starter_template_items()
    assert len(items) >= 5
    keys = [item[0] for item in items]
    assert "PARTICLE_FORCE" in keys
    assert "PARTICLE_ATTRIBUTE" in keys
    assert "VOXEL_ACCELERATION" in keys
    assert "VELOCITY_DAMPING" in keys
    assert "BLANK" in keys

    from fluidsim_testpkg.sim.effects.schema import EffectInstanceV2

    for key, _label, _desc in items:
        definition = get_template_definition(key)
        assert len(definition.name) > 0
        assert definition.built_in is False
        assert len(definition.kernels) >= 1
        for kernel in definition.kernels:
            assert len(kernel.source.strip()) > 0
            assert kernel.domain in {EffectDomain.PARTICLES, EffectDomain.ACTIVE_VOXELS}
            assert kernel.stage in set(EffectStage)
            assert kernel.cadence in set(EffectCadence)

        instance = EffectInstanceV2(f"test-{key.lower()}", definition)
        res = validate_effect(instance, safe_mode=True)
        assert res.valid is True, f"Template {key} failed AST validation: {res.issues}"


def test_authoring_operators_registered_in_workspace():
    source = WORKSPACE.read_text(encoding="utf-8")
    expected_operators = (
        "fluidsim.effect_create_custom",
        "fluidsim.effect_convert_to_custom",
        "fluidsim.effect_parameter_add",
        "fluidsim.effect_parameter_remove",
        "fluidsim.effect_parameter_move",
        "fluidsim.effect_kernel_add",
        "fluidsim.effect_kernel_remove",
        "fluidsim.effect_field_add",
        "fluidsim.effect_field_remove",
        "fluidsim.effect_generate_boilerplate",
        "fluidsim.effect_validate_kernel",
        "fluidsim.effect_test_sandbox",
        "fluidsim.effect_install_to_library",
        "fluidsim.open_effect_authoring_workspace",
    )
    for op_id in expected_operators:
        assert f'bl_idname = "{op_id}"' in source, f"Missing operator {op_id}"

    assert "FLUIDSIM_OT_effect_create_custom" in source
    assert "FLUIDSIM_OT_effect_parameter_add" in source
    assert "FLUIDSIM_OT_effect_kernel_add" in source
    assert "FLUIDSIM_OT_effect_field_add" in source
    assert "FLUIDSIM_OT_effect_generate_boilerplate" in source


def test_boilerplate_generator_contains_variables_and_bounds(monkeypatch):
    fake_bpy = ModuleType("bpy")
    fake_bpy.types = SimpleNamespace(Operator=object, Panel=object)
    fake_props = ModuleType("bpy.props")
    for prop in ("IntProperty", "BoolProperty", "EnumProperty", "FloatProperty", "StringProperty"):
        setattr(fake_props, prop, lambda **_kwargs: None)
    fake_bpy.props = fake_props
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_props)

    effects_module = ModuleType("fluidsim_testpkg.blender.effects")
    effects_module.bound_parameter_keys = lambda _item, kernel: tuple(
        k.strip() for k in kernel.parameter_keys_csv.split(",") if k.strip())
    fields_module = ModuleType("fluidsim_testpkg.sim.effects.fields")
    fields_module.FIELD_SPECS = {}
    source_module = ModuleType("fluidsim_testpkg.sim.effects.source")
    source_module.generate_kernel_source = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, effects_module.__name__, effects_module)
    monkeypatch.setitem(sys.modules, fields_module.__name__, fields_module)
    monkeypatch.setitem(sys.modules, source_module.__name__, source_module)

    spec = importlib.util.spec_from_file_location(
        "fluidsim_testpkg.blender.effect_workspace_test", WORKSPACE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    kernel = SimpleNamespace(
        kernel_id="test_kernel",
        label="Test Kernel",
        domain="PARTICLES",
        stage="FORCES",
        cadence="EVERY_SUBSTEP",
        parameter_keys_csv="strength, radius",
        fields=[
            SimpleNamespace(field="particle_velocity", access="READ_WRITE_CURRENT"),
        ],
    )
    item = SimpleNamespace(
        parameters=[
            SimpleNamespace(key="strength", label="Strength", kind="FLOAT", unit="m/s²", description="Force magnitude"),
            SimpleNamespace(key="radius", label="Radius", kind="FLOAT", unit="m", description="Effect radius"),
        ]
    )

    doc = mod._generate_kernel_boilerplate(kernel, item)
    assert "FluidSim Warp Kernel: test_kernel" in doc
    assert "particle_index" in doc
    assert "position" in doc
    assert "particle_velocity" in doc
    assert "strength (FLOAT): Strength [m/s²] - Force magnitude" in doc
    assert "radius (FLOAT): Radius [m] - Effect radius" in doc


def test_effects_sync_helpers_defined():
    source = EFFECTS.read_text(encoding="utf-8")
    assert "def parameter_definition_from_pg(" in source
    assert "def kernel_definition_from_pg(" in source
    assert "def rebuild_definition_json(" in source
    assert "def add_custom_effect_from_template(" in source
    assert "def convert_instance_to_custom(" in source
    assert "parameter_keys_csv" in source
