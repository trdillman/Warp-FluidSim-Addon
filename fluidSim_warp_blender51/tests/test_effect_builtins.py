from __future__ import annotations

import importlib.util

import pytest
from fluidsim_testpkg.sim.effects.builtins import builtin_effect_definitions
from fluidsim_testpkg.sim.effects.compiler import compile_effect
from fluidsim_testpkg.sim.effects.enums import ParameterKind, TrustState
from fluidsim_testpkg.sim.effects.library import EffectLibrary
from fluidsim_testpkg.sim.effects.sandbox import sandbox_test
from fluidsim_testpkg.sim.effects.schema import EffectDefinitionV2, EffectInstanceV2
from fluidsim_testpkg.sim.effects.trust import TrustStore, content_trust_digest
from fluidsim_testpkg.sim.effects.validation import validate_effect

EXPECTED = {
    "fluidsim.vortex_force",
    "fluidsim.curl_turbulence",
    "fluidsim.radial_pulse",
    "fluidsim.planar_split",
    "fluidsim.velocity_damping",
    "fluidsim.temperature_lift",
    "fluidsim.particle_speed_limit",
    "fluidsim.voxel_cooling",
}

REQUIRES_WARP = pytest.mark.skipif(
    importlib.util.find_spec("warp") is None,
    reason="real Warp JIT validation runs in the supplied-Warp gate",
)


def test_builtin_inventory_is_complete_unique_and_strict():
    definitions = builtin_effect_definitions()
    assert {item.effect_id for item in definitions} == EXPECTED
    assert len(definitions) == len(EXPECTED)
    for definition in definitions:
        assert EffectDefinitionV2.from_dict(definition.to_dict()) == definition
        assert definition.built_in is True
        assert len(definition.summary) >= 20
        assert len(definition.description) >= 40
        assert definition.cost_hint in {"LOW", "MEDIUM", "HIGH"}


def test_spatial_builtins_use_object_controllers():
    definitions = {item.effect_id: item for item in builtin_effect_definitions()}
    for effect_id in (
        "fluidsim.vortex_force",
        "fluidsim.radial_pulse",
        "fluidsim.planar_split",
    ):
        assert any(item.kind is ParameterKind.OBJECT for item in definitions[effect_id].parameters)


def test_parameters_have_artist_metadata():
    for definition in builtin_effect_definitions():
        for parameter in definition.parameters:
            assert parameter.label
            assert parameter.description
            assert parameter.unit
            if parameter.kind in {ParameterKind.FLOAT, ParameterKind.INT}:
                assert parameter.minimum is not None or parameter.maximum is not None


def test_temperature_lift_proves_multi_stage_effect_support():
    definition = next(
        item for item in builtin_effect_definitions()
        if item.effect_id == "fluidsim.temperature_lift"
    )
    assert len(definition.kernels) == 2
    assert len({kernel.stage for kernel in definition.kernels}) == 2


@REQUIRES_WARP
def test_every_builtin_validates_compiles_and_sandboxes_on_supplied_warp():
    for definition in builtin_effect_definitions():
        instance = EffectInstanceV2(f"test-{definition.effect_id}", definition)
        validation = validate_effect(instance)
        assert validation.valid, (definition.effect_id, validation.issues)
        compiled = compile_effect(instance, device="cpu")
        assert compiled.success, (definition.effect_id, compiled.issues)
        report = sandbox_test(compiled, device="cpu")
        assert report.passed, (definition.effect_id, report.issues)


def test_builtins_are_trusted_without_local_records(tmp_path):
    store = TrustStore(tmp_path / "trust.json")
    for definition in builtin_effect_definitions():
        assert store.status(
            content_trust_digest(definition), built_in=definition.built_in
        ) is TrustState.BUILT_IN_TRUSTED


def test_library_search_and_categories():
    library = EffectLibrary(builtin_effect_definitions())
    assert library.get("fluidsim.vortex_force").name == "Vortex Force"
    assert {item.effect_id for item in library.search("temperature")} == {
        "fluidsim.temperature_lift",
        "fluidsim.voxel_cooling",
    }
    assert "Forces" in library.categories()
    assert "Utilities" in library.categories()
