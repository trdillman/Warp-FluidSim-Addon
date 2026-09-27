from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace

import pytest
from fluidsim_testpkg.sim.effects.compiler import (
    _import_generated,
    build_candidate_effects,
    compile_effect,
)
from fluidsim_testpkg.sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    FieldAccess,
    ParameterKind,
    TrustState,
)
from fluidsim_testpkg.sim.effects.sandbox import sandbox_test
from fluidsim_testpkg.sim.effects.schema import (
    EffectDefinitionV2,
    EffectInstanceV2,
    FieldBindingV2,
    KernelDefinitionV2,
    ParameterDefinitionV2,
)
from fluidsim_testpkg.sim.effects.source import GeneratedKernelSource, generate_kernel_source
from fluidsim_testpkg.sim.effects.trust import TrustStore, content_trust_digest

REQUIRES_WARP = pytest.mark.skipif(
    importlib.util.find_spec("warp") is None,
    reason="real Warp JIT validation runs in the supplied-Warp gate",
)


def _particle_instance(source: str, *, built_in=True):
    definition = EffectDefinitionV2(
        effect_id="test.compiler.particle",
        version="1.0.0",
        name="Compiler Particle",
        summary="Compiler Particle",
        description="Compiler Particle",
        category="Tests",
        author="Tests",
        license="GPL-3.0-or-later",
        built_in=built_in,
        parameters=(
            ParameterDefinitionV2("strength", "Strength", ParameterKind.FLOAT, 0.5),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="particle_kernel",
                label="Particle Kernel",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source=source,
                fields=(
                    FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT),
                ),
                parameter_keys=("strength",),
            ),
        ),
    )
    return EffectInstanceV2("particle-1", definition)


def _voxel_instance(source: str):
    definition = EffectDefinitionV2(
        effect_id="test.compiler.voxel",
        version="1.0.0",
        name="Compiler Voxel",
        summary="Compiler Voxel",
        description="Compiler Voxel",
        category="Tests",
        author="Tests",
        license="GPL-3.0-or-later",
        built_in=True,
        kernels=(
            KernelDefinitionV2(
                kernel_id="voxel_kernel",
                label="Voxel Kernel",
                domain=EffectDomain.ACTIVE_VOXELS,
                stage=EffectStage.BEFORE_PRESSURE,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source=source,
                fields=(FieldBindingV2("temperature", FieldAccess.READ_WRITE_CURRENT),),
            ),
        ),
    )
    return EffectInstanceV2("voxel-1", definition)


@REQUIRES_WARP
def test_real_warp_compile_and_sandbox_pass_for_particle_effect():
    instance = _particle_instance(
        "particle_velocity[particle_index] = velocity * strength"
    )
    compiled = compile_effect(instance, device="cpu")
    assert compiled.success is True
    assert compiled.kernels[0].compile_report.compiled is True
    report = sandbox_test(compiled, device="cpu")
    assert report.passed is True
    assert report.kernels[0].canaries_intact is True
    assert report.kernels[0].finite is True


@REQUIRES_WARP
def test_scalar_vec3_assignment_is_rejected_by_real_warp_compile():
    compiled = compile_effect(
        _voxel_instance("temperature[flat_voxel_index] = wp.vec3(1.0, 2.0, 3.0)"),
        device="cpu",
    )
    assert compiled.success is False
    assert any("float" in issue.message.lower() or "vec3" in issue.message.lower()
               for issue in compiled.issues)
    assert compiled.issues[0].line == 1


@REQUIRES_WARP
def test_missing_warp_builtin_is_rejected_by_real_warp_compile():
    compiled = compile_effect(
        _particle_instance("particle_velocity[particle_index] = wp.not_a_builtin(velocity)"),
        device="cpu",
    )
    assert compiled.success is False
    assert compiled.issues


@REQUIRES_WARP
def test_candidate_build_is_fail_closed_and_enforces_trust(tmp_path):
    store = TrustStore(tmp_path / "trust.json")
    untrusted = _particle_instance(
        "particle_velocity[particle_index] = velocity * strength",
        built_in=False,
    )
    result = build_candidate_effects((untrusted,), device="cpu", trust_store=store)
    assert result.success is False
    assert result.compiled_effects == ()
    assert result.trust_states[untrusted.instance_id] is TrustState.UNTRUSTED_IMPORTED

    store.trust(content_trust_digest(untrusted.definition), "Compiler Test")
    trusted = build_candidate_effects((untrusted,), device="cpu", trust_store=store)
    assert trusted.success is True
    assert len(trusted.compiled_effects) == 1


@REQUIRES_WARP
def test_candidate_failure_does_not_return_partial_compiled_stack(tmp_path):
    store = TrustStore(tmp_path / "trust.json")
    valid = _particle_instance("particle_velocity[particle_index] = velocity")
    invalid = replace(
        _particle_instance("particle_velocity[particle_index] = wp.not_a_builtin(velocity)"),
        instance_id="invalid-2",
    )
    result = build_candidate_effects((valid, invalid), device="cpu", trust_store=store)
    assert result.success is False
    assert result.compiled_effects == ()


def test_import_refuses_module_changed_on_disk(monkeypatch, tmp_path):
    generated = generate_kernel_source(
        _particle_instance("particle_velocity[particle_index] = velocity"), 0)
    tampered = tmp_path / f"{generated.module_name}.py"

    def swap_after_write(self, directory=None):
        tampered.write_text("raise SystemExit('tampered')\n", encoding="utf-8")
        return tampered

    monkeypatch.setattr(GeneratedKernelSource, "write_cache_file", swap_after_write)
    with pytest.raises(RuntimeError, match="changed on disk"):
        _import_generated(generated, tmp_path)
    assert generated.module_name not in sys.modules


@REQUIRES_WARP
def test_default_compile_writes_into_private_module_directory(
        _private_effect_module_directory):
    compiled = compile_effect(
        _particle_instance("particle_velocity[particle_index] = velocity * 0.25"),
        device="cpu",
    )
    assert compiled.success is True
    module_file = compiled.kernels[0].compile_report.module_file
    assert module_file.startswith(str(_private_effect_module_directory))
