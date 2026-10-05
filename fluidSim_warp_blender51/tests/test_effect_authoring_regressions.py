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
