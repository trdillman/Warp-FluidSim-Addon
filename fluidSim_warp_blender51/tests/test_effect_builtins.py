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


def _shipped_definitions():
    from fluidsim_testpkg.resources.examples.custom_effects.example_effects import (
        EXAMPLE_EFFECT_DEFINITIONS,
    )
    from fluidsim_testpkg.sim.effects.templates import TEMPLATES

    yield from builtin_effect_definitions()
    yield from (builder() for builder in TEMPLATES.values())
    yield from EXAMPLE_EFFECT_DEFINITIONS


def test_shipped_effects_write_only_where_the_solver_consumes_the_write():
    # B68: built-ins, starter templates and example packages wrote particle
    # velocity after particle-to-grid transfer, where PIC/APIC discard it.
    for definition in _shipped_definitions():
        result = validate_effect(EffectInstanceV2("shipped", definition))
        assert result.issues == (), (definition.effect_id, result.issues)


def _retired_builtins():
    import json
    from pathlib import Path

    path = Path(__file__).parent / "data" / "retired_builtin_effects_1_0_0.json"
    return [EffectDefinitionV2.from_dict(item)
            for item in json.loads(path.read_text(encoding="utf-8"))]


def test_previously_shipped_builtins_stay_trusted_in_saved_scenes(tmp_path):
    # A .blend embeds the built-in definition it was authored with; the B68
    # stage fix changed their content digests, which must not turn every
    # saved built-in effect into untrusted code.
    store = TrustStore(tmp_path / "trust.json")
    retired = _retired_builtins()
    assert len(retired) == len(EXPECTED)
    for definition in retired:
        digest = content_trust_digest(definition)
        assert store.status(digest, built_in=True) is TrustState.BUILT_IN_TRUSTED
        # The built_in flag alone (spoofable in a .blend) never grants trust.
        assert store.status(digest, built_in=False) is TrustState.UNTRUSTED_IMPORTED


def test_library_search_and_categories():
    library = EffectLibrary(builtin_effect_definitions())
    assert library.get("fluidsim.vortex_force").name == "Vortex Force"
    assert {item.effect_id for item in library.search("temperature")} == {
        "fluidsim.temperature_lift",
        "fluidsim.voxel_cooling",
    }
    assert "Forces" in library.categories()
    assert "Utilities" in library.categories()
