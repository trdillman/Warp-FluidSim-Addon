from __future__ import annotations

from fluidsim_testpkg.sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    FieldAccess,
)
from fluidsim_testpkg.sim.effects.fields import FIELD_SPECS, validate_field_binding
from fluidsim_testpkg.sim.effects.schema import FieldBindingV2, KernelDefinitionV2


def _kernel(domain: EffectDomain, stage: EffectStage, binding: FieldBindingV2):
    return KernelDefinitionV2(
        kernel_id="k",
        label="K",
        domain=domain,
        stage=stage,
        cadence=EffectCadence.EVERY_SUBSTEP,
        source="pass",
        fields=(binding,),
    )


def test_registry_covers_existing_particle_and_voxel_fields():
    expected = {
        "velocity_u", "velocity_v", "velocity_w", "weight_u", "weight_v", "weight_w",
        "temperature", "viscosity", "density", "pressure", "divergence",
        "surface_sdf", "collision_sdf", "churn", "voxel_type", "voxel_counts",
        "particle_position", "particle_velocity", "particle_age", "particle_life",
        "particle_temperature", "particle_viscosity", "particle_density",
        "particle_time_offset",
    }
    assert expected.issubset(FIELD_SPECS)


def test_particle_kernel_rejects_voxel_field_in_safe_mode():
    binding = FieldBindingV2("temperature", FieldAccess.READ)
    issues = validate_field_binding(
        binding,
        _kernel(EffectDomain.PARTICLES, EffectStage.AFTER_PARTICLE_TRANSFER, binding),
        safe_mode=True,
    )
    assert any(issue.code == "FIELD_DOMAIN_MISMATCH" for issue in issues)


def test_voxel_kernel_rejects_particle_field_in_safe_mode():
    binding = FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT)
    issues = validate_field_binding(
        binding,
        _kernel(EffectDomain.ACTIVE_VOXELS, EffectStage.BEFORE_PRESSURE, binding),
        safe_mode=True,
    )
    assert any(issue.code == "FIELD_DOMAIN_MISMATCH" for issue in issues)


def test_safe_mode_rejects_unsafe_arbitrary_access():
    binding = FieldBindingV2("particle_velocity", FieldAccess.UNSAFE_ARBITRARY)
    issues = validate_field_binding(
        binding,
        _kernel(EffectDomain.PARTICLES, EffectStage.AFTER_PARTICLE_TRANSFER, binding),
        safe_mode=True,
    )
    assert any(issue.code == "UNSAFE_ACCESS_REQUIRES_EXPERT" for issue in issues)


def test_legal_particle_current_write_passes():
    binding = FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT)
    assert validate_field_binding(
        binding,
        _kernel(EffectDomain.PARTICLES, EffectStage.AFTER_PARTICLE_TRANSFER, binding),
        safe_mode=True,
    ) == ()
