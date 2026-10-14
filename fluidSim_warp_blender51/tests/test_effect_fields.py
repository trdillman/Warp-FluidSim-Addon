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
    for stage in (EffectStage.BEFORE_EMISSION, EffectStage.AFTER_PARTICLE_UPDATE):
        assert validate_field_binding(
            binding,
            _kernel(EffectDomain.PARTICLES, stage, binding),
            safe_mode=True,
        ) == ()


def _issues(domain, stage, field, access):
    binding = FieldBindingV2(field, access)
    return validate_field_binding(binding, _kernel(domain, stage, binding), safe_mode=True)


def test_voxel_writes_before_emission_are_rejected_as_discarded():
    # B68: the solver's grid clear (_clear_grid_fields), which runs right after
    # BEFORE_EMISSION every substep, zeroes these fields before anything reads them.
    for field in ("velocity_u", "velocity_v", "velocity_w", "weight_u", "weight_v",
                  "weight_w", "temperature", "viscosity", "density"):
        for access in (FieldAccess.WRITE_CURRENT, FieldAccess.READ_WRITE_CURRENT):
            issues = _issues(EffectDomain.ACTIVE_VOXELS, EffectStage.BEFORE_EMISSION,
                             field, access)
            discarded = [issue for issue in issues if issue.code == "FIELD_WRITE_DISCARDED"]
            assert discarded and discarded[0].severity == "ERROR", (field, access)


def test_voxel_reads_and_carried_fields_before_emission_stay_legal():
    assert _issues(EffectDomain.ACTIVE_VOXELS, EffectStage.BEFORE_EMISSION,
                   "temperature", FieldAccess.READ) == ()
    # churn is remapped across the topology rebuild, so the write survives.
    assert _issues(EffectDomain.ACTIVE_VOXELS, EffectStage.BEFORE_EMISSION,
                   "churn", FieldAccess.READ_WRITE_CURRENT) == ()
    assert _issues(EffectDomain.ACTIVE_VOXELS, EffectStage.AFTER_PARTICLE_TRANSFER,
                   "temperature", FieldAccess.READ_WRITE_CURRENT) == ()


def test_discarded_voxel_fields_match_the_solver_grid_clear():
    import inspect

    from fluidsim_testpkg.sim.effects.fields import GRID_CLEARED_FIELDS
    from fluidsim_testpkg.sim.solver import FluidSolver

    clear_source = inspect.getsource(FluidSolver._clear_grid_fields)
    zero_call = clear_source.split("k_zero_fluid_grid", 1)[1].split("device=", 1)[0]
    for field in GRID_CLEARED_FIELDS:
        accessor = FIELD_SPECS[field].runtime_accessor
        if accessor.startswith("vel."):
            assert "self.vel.arrays()" in zero_call, field
        else:
            assert f"self.{accessor}" in zero_call, field


def test_particle_velocity_write_after_transfer_warns_it_is_replaced():
    # B68: after P2G, PIC/APIC G2P replace particle velocity with the grid
    # value and FLIP keeps only its ratio of the edit, unprojected.
    for stage in (EffectStage.AFTER_PARTICLE_TRANSFER, EffectStage.BEFORE_PRESSURE):
        for access in (FieldAccess.WRITE_CURRENT, FieldAccess.READ_WRITE_CURRENT):
            issues = _issues(EffectDomain.PARTICLES, stage, "particle_velocity", access)
            assert [(issue.code, issue.severity) for issue in issues] == [
                ("PARTICLE_VELOCITY_WRITE_OVERWRITTEN", "WARNING")], (stage, access)
            assert "BEFORE_EMISSION" in issues[0].hint
    assert _issues(EffectDomain.PARTICLES, EffectStage.AFTER_PARTICLE_TRANSFER,
                   "particle_velocity", FieldAccess.READ) == ()
    assert _issues(EffectDomain.PARTICLES, EffectStage.AFTER_PARTICLE_TRANSFER,
                   "particle_temperature", FieldAccess.READ_WRITE_CURRENT) == ()
