"""Regression tests for custom-effect authoring review findings."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fluidsim_testpkg.sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    ParameterKind,
)
from fluidsim_testpkg.sim.effects.schema import KernelDefinitionV2, ParameterDefinitionV2

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "blender" / "effect_workspace.py"


@pytest.fixture()
def effects_mod(monkeypatch):
    fake_bpy = MagicMock()
    fake_bpy.types = SimpleNamespace(
        PropertyGroup=object, Operator=object, Panel=object, UIList=object
    )
    fake_props = MagicMock()
    fake_bpy.props = fake_props
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_props)
    name = "fluidsim_testpkg.blender.effects_regression"
    spec = importlib.util.spec_from_file_location(name, ROOT / "blender" / "effects.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_unique_id_survives_removals(effects_mod):
    # Removing the middle item leaves {param_1, param_3}; len()+1 would reuse param_3.
    assert effects_mod.unique_id("param", ["param_1", "param_3"]) == "param_4"
    assert effects_mod.unique_id("kernel", []) == "kernel_1"
    assert effects_mod.unique_id("kernel", ["kernel_1"]) == "kernel_2"


def test_pg_limit_clears_when_unticked_and_keeps_exact_value(effects_mod):
    assert effects_mod._pg_limit(False, 5.0, 5.0) is None
    exact = 0.1  # not representable in float32
    stored = effects_mod._as_float32(exact)
    assert stored != exact
    assert effects_mod._pg_limit(True, stored, exact) == exact
    assert effects_mod._pg_limit(True, 2.5, exact) == 2.5
    assert effects_mod._pg_limit(True, 3.0, None) == 3.0


def _parameter_pg(**overrides):
    values = dict(
        key="strength",
        label="Strength",
        kind="FLOAT",
        unit="",
        description="",
        enum_items_json="",
        has_minimum=False,
        has_maximum=False,
        minimum=0.0,
        maximum=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parameter_limits_can_be_cleared(effects_mod, monkeypatch):
    monkeypatch.setattr(effects_mod, "_parameter_value", lambda _item: 1.0)
    base = ParameterDefinitionV2(
        key="strength", label="Strength", kind=ParameterKind.FLOAT,
        default=1.0, minimum=0.0, maximum=10.0,
    )
    result = effects_mod.parameter_definition_from_pg(_parameter_pg(), base)
    assert result.minimum is None and result.maximum is None


def _kernel_pg(csv, authored):
    return SimpleNamespace(
        kernel_id="k",
        label="K",
        domain=EffectDomain.PARTICLES.value,
        stage=EffectStage.BEFORE_PRESSURE.value,
        cadence=EffectCadence.EVERY_SUBSTEP.value,
        source_text=None,
        fields=[],
        parameter_keys_csv=csv,
        parameter_keys_authored=authored,
        safe_mode=True,
        frame_start=1,
        frame_end=250,
        frame_step=1,
    )


def _kernel_base():
    return KernelDefinitionV2(
        kernel_id="k",
        label="K",
        domain=EffectDomain.PARTICLES,
        stage=EffectStage.BEFORE_PRESSURE,
        cadence=EffectCadence.EVERY_SUBSTEP,
        source="pass",
        parameter_keys=("a", "b"),
    )


def test_kernel_parameter_keys_can_be_emptied_once_authored(effects_mod):
    result = effects_mod.kernel_definition_from_pg(_kernel_pg("", True), _kernel_base())
    assert result.parameter_keys == ()


def test_kernel_parameter_keys_fall_back_for_legacy_blends(effects_mod):
    result = effects_mod.kernel_definition_from_pg(_kernel_pg("", False), _kernel_base())
    assert result.parameter_keys == ("a", "b")


def test_kernel_parameter_keys_follow_csv_edits(effects_mod):
    result = effects_mod.kernel_definition_from_pg(_kernel_pg("b", False), _kernel_base())
    assert result.parameter_keys == ("b",)


def test_workspace_operators_use_real_report_attributes():
    source = WORKSPACE.read_text(encoding="utf-8")
    assert "report.passed" in source
    assert "report.success" not in source
    assert "invoke_confirm" in source  # library overwrite is confirmed


def test_edited_default_is_exported_and_unedited_default_stays_exact(effects_mod):
    # An edited value becomes the exported default.
    assert effects_mod._pg_default(2.5, 1.0) == 2.5
    assert effects_mod._pg_default((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) == (0.0, 1.0, 0.0)
    assert effects_mod._pg_default("HIGH", "LOW") == "HIGH"
    assert effects_mod._pg_default(True, False) is True
    # An untouched float32 PG value keeps the exact definition value (stable digest).
    stored = effects_mod._as_float32(0.1)
    assert effects_mod._pg_default(stored, 0.1) == 0.1
    assert effects_mod._pg_default((stored, 0.0, 1.0), (0.1, 0.0, 1.0)) == (0.1, 0.0, 1.0)
    assert effects_mod._pg_default(3.0, None) == 3.0


def test_save_and_export_bake_current_values_as_defaults():
    source = WORKSPACE.read_text(encoding="utf-8")
    ui = (ROOT / "blender" / "effect_ui.py").read_text(encoding="utf-8")
    assert "effects.exportable_definition(" in source
    assert "effects.exportable_definition(" in ui


def test_save_to_user_library_targets_the_scanned_library_folder():
    source = WORKSPACE.read_text(encoding="utf-8")
    body = source.split("def _package_path(item):", 1)[1].split("def invoke", 1)[0]
    assert "user_library_path()" in body
    assert "effect_user_root" not in body


def test_parameter_editor_covers_every_kind():
    source = WORKSPACE.read_text(encoding="utf-8")
    for prop in ("bool_value", "vector_value", "color_value", "enum_value", "enum_items_json"):
        assert f'p_box.prop(param, "{prop}"' in source or f'r_vec.prop(param, "{prop}"' in source, prop


def _remove_target(csv, authored, definition_json=""):
    kernel = SimpleNamespace(
        kernel_id="voxel_buoyancy", parameter_keys_csv=csv, parameter_keys_authored=authored
    )
    return SimpleNamespace(definition_json=definition_json, kernels=[kernel]), kernel


def test_unbind_parameter_key_strips_authored_csv(effects_mod):
    pg, kernel = _remove_target("strength, radius", True)
    effects_mod.unbind_parameter_key(pg, "strength")
    assert kernel.parameter_keys_csv == "radius"
    effects_mod.unbind_parameter_key(pg, "missing")
    assert kernel.parameter_keys_csv == "radius"


def test_unbind_parameter_key_handles_legacy_fallback_keys(effects_mod):
    import json

    from fluidsim_testpkg.sim.effects.templates import voxel_acceleration_template

    definition = voxel_acceleration_template()
    assert definition.kernels[0].parameter_keys == ("buoyancy",)
    pg, kernel = _remove_target("", False, json.dumps(definition.to_dict()))
    effects_mod.unbind_parameter_key(pg, "buoyancy")
    assert kernel.parameter_keys_csv == ""
    assert kernel.parameter_keys_authored is True  # emptied csv now really unbinds
    result = effects_mod.kernel_definition_from_pg(
        SimpleNamespace(**{**vars(_kernel_pg("", True)), "kernel_id": "voxel_buoyancy"}),
        definition.kernels[0],
    )
    assert result.parameter_keys == ()
