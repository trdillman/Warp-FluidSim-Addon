from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fluidsim_testpkg.resources.examples.custom_effects.example_effects import (
    EXAMPLE_EFFECT_DEFINITIONS,
)
from fluidsim_testpkg.sim.effects.compiler import compile_effect
from fluidsim_testpkg.sim.effects.enums import ParameterKind
from fluidsim_testpkg.sim.effects.package import read_effect_package
from fluidsim_testpkg.sim.effects.sandbox import sandbox_test
from fluidsim_testpkg.sim.effects.schema import EffectDefinitionV2, EffectInstanceV2
from fluidsim_testpkg.sim.effects.validation import validate_effect

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "resources" / "examples" / "custom_effects"

REQUIRES_WARP = pytest.mark.skipif(
    importlib.util.find_spec("warp") is None,
    reason="real Warp JIT validation runs when Warp is installed",
)

EXPECTED_EFFECT_IDS = {
    "example.thermal_updraft",
    "example.point_attractor",
    "example.helical_vortex",
    "example.directional_wind",
    "example.centrifugal_spinner",
    "example.surface_tension_bead",
    "example.viscous_heating",
    "example.periodic_ripples",
    "example.bounding_box_confinement",
    "example.voxel_viscosity_decay",
}


def test_example_effects_inventory():
    assert len(EXAMPLE_EFFECT_DEFINITIONS) == 10
    actual_ids = {defn.effect_id for defn in EXAMPLE_EFFECT_DEFINITIONS}
    assert actual_ids == EXPECTED_EFFECT_IDS
    for definition in EXAMPLE_EFFECT_DEFINITIONS:
        assert EffectDefinitionV2.from_dict(definition.to_dict()) == definition
        assert len(definition.summary) >= 15
        assert len(definition.description) >= 30
        assert definition.category in {"Thermal", "Forces", "Surface", "Waves", "Boundary", "Grid"}


def test_example_effects_packages_exist_and_roundtrip():
    for definition in EXAMPLE_EFFECT_DEFINITIONS:
        filename = f"{definition.effect_id.split('.')[-1]}-{definition.version}.fseffect"
        pkg_path = EXAMPLES_DIR / filename
        assert pkg_path.is_file(), f"Package file {filename} does not exist"
        assert pkg_path.stat().st_size > 500

        read_defn = read_effect_package(pkg_path)
        assert read_defn.effect_id == definition.effect_id
        assert read_defn.name == definition.name
        assert read_defn.version == definition.version
        assert len(read_defn.kernels) == len(definition.kernels)
        assert read_defn == definition


def test_example_effects_parameters_have_metadata():
    for definition in EXAMPLE_EFFECT_DEFINITIONS:
        for param in definition.parameters:
            assert param.label
            assert param.description
            assert param.unit
            if param.kind in {ParameterKind.FLOAT, ParameterKind.INT}:
                assert param.minimum is not None or param.maximum is not None


@REQUIRES_WARP
def test_all_10_example_effects_validate_compile_and_sandbox():
    for definition in EXAMPLE_EFFECT_DEFINITIONS:
        instance = EffectInstanceV2(f"test-inst-{definition.effect_id}", definition)
        val = validate_effect(instance)
        assert val.valid, (definition.effect_id, val.issues)

        compiled = compile_effect(instance, device="cpu")
        assert compiled.success, (definition.effect_id, compiled.issues)

        sb = sandbox_test(compiled, device="cpu")
        assert sb.passed, (definition.effect_id, sb.issues)
