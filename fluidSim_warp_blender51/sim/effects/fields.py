"""Authoritative solver-field registry for custom effects."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import EffectDomain, EffectStage, FieldAccess
from .schema import EffectIssue, FieldBindingV2, KernelDefinitionV2


@dataclass(frozen=True)
class FieldSpec:
    key: str
    dtype: str
    domain: EffectDomain
    units: str
    legal_stages: frozenset[EffectStage]
    allowed_access: frozenset[FieldAccess]
    runtime_accessor: str
    current_index: str
    prerequisites: frozenset[str] = frozenset()
    persistent: bool = False
    structural: bool = False
    description: str = ""


_ALL_STAGES = frozenset(EffectStage)
_PARTICLE_ACCESS = frozenset({FieldAccess.READ, FieldAccess.WRITE_CURRENT, FieldAccess.READ_WRITE_CURRENT})
_GRID_READ_WRITE = frozenset({FieldAccess.READ, FieldAccess.WRITE_CURRENT, FieldAccess.READ_WRITE_CURRENT})
_GRID_READ = frozenset({FieldAccess.READ})


def _particle(key: str, dtype: str, accessor: str, units: str = "", *, persistent: bool = True, description: str = "") -> FieldSpec:
    return FieldSpec(
        key=key,
        dtype=dtype,
        domain=EffectDomain.PARTICLES,
        units=units,
        legal_stages=_ALL_STAGES,
        allowed_access=_PARTICLE_ACCESS,
        runtime_accessor=accessor,
        current_index="particle_index",
        prerequisites=frozenset({"PARTICLES"}),
        persistent=persistent,
        description=description,
    )


def _voxel(key: str, dtype: str, accessor: str, units: str = "", *, read_only: bool = False, prerequisites=(), structural: bool = False, description: str = "") -> FieldSpec:
    return FieldSpec(
        key=key,
        dtype=dtype,
        domain=EffectDomain.ACTIVE_VOXELS,
        units=units,
        legal_stages=_ALL_STAGES,
        allowed_access=_GRID_READ if read_only else _GRID_READ_WRITE,
        runtime_accessor=accessor,
        current_index="flat_voxel_index",
        prerequisites=frozenset({"TOPOLOGY", *prerequisites}),
        persistent=False,
        structural=structural,
        description=description,
    )


FIELD_SPECS: dict[str, FieldSpec] = {
    "velocity_u": _voxel("velocity_u", "float", "vel.u", "m/s"),
    "velocity_v": _voxel("velocity_v", "float", "vel.v", "m/s"),
    "velocity_w": _voxel("velocity_w", "float", "vel.w", "m/s"),
    "weight_u": _voxel("weight_u", "float", "vel.wu", "weight"),
    "weight_v": _voxel("weight_v", "float", "vel.wv", "weight"),
    "weight_w": _voxel("weight_w", "float", "vel.ww", "weight"),
    "temperature": _voxel("temperature", "float", "temperature.data", "scene temperature"),
    "viscosity": _voxel("viscosity", "float", "viscosity_field.data", "m²/s"),
    "density": _voxel("density", "float", "density.data", "kg/m³"),
    "pressure": _voxel("pressure", "float", "pressure_solver.pressure", "solver pressure"),
    "divergence": _voxel("divergence", "float", "pressure_solver.divergence", "1/s"),
    "surface_sdf": _voxel("surface_sdf", "float", "surface_sdf.data", "m", read_only=True, prerequisites=("SURFACE_SDF",)),
    "collision_sdf": _voxel("collision_sdf", "float", "collision_sdf.data", "m", read_only=True, prerequisites=("COLLISION",)),
    "churn": _voxel("churn", "float", "churn.data", "1/s"),
    "voxel_type": _voxel("voxel_type", "int", "voxel_type.data", "bitmask", read_only=True, structural=True),
    "voxel_counts": _voxel("voxel_counts", "int", "voxel_counts.data", "particles", read_only=True, structural=True),
    "particle_position": _particle("particle_position", "vec3", "particles.position", "m"),
    "particle_velocity": _particle("particle_velocity", "vec3", "particles.velocity", "m/s"),
    "particle_age": _particle("particle_age", "float", "particles.attr:age", "s"),
    "particle_life": _particle("particle_life", "float", "particles.attr:life", "s"),
    "particle_temperature": _particle("particle_temperature", "float", "particles.attr:temperature", "scene temperature"),
    "particle_viscosity": _particle("particle_viscosity", "float", "particles.attr:viscosity", "m²/s"),
    "particle_density": _particle("particle_density", "float", "particles.attr:density", "kg/m³"),
    "particle_time_offset": _particle("particle_time_offset", "float", "particles.time_offset", "s"),
}


# Voxel fields the solver zeroes in _clear_grid_fields immediately after the
# BEFORE_EMISSION stage of every substep; a write there never reaches a reader.
GRID_CLEARED_FIELDS = frozenset({
    "velocity_u", "velocity_v", "velocity_w", "weight_u", "weight_v", "weight_w",
    "temperature", "viscosity", "density",
})
# Stages between particle-to-grid transfer and grid-to-particle transfer: G2P
# replaces particle velocity with the grid value under PIC/APIC, and FLIP keeps
# only flip_ratio of an edit made here, without pressure projection.
_AFTER_P2G_BEFORE_G2P = frozenset({
    EffectStage.AFTER_PARTICLE_TRANSFER, EffectStage.BEFORE_PRESSURE,
})


def _stage_consumption_issues(
    binding: FieldBindingV2,
    kernel: KernelDefinitionV2,
) -> list[EffectIssue]:
    """Writes the solver overwrites before any phase reads them (B68)."""
    if binding.access is FieldAccess.READ:
        return []
    if (kernel.domain is EffectDomain.ACTIVE_VOXELS
            and kernel.stage is EffectStage.BEFORE_EMISSION
            and binding.field in GRID_CLEARED_FIELDS):
        return [EffectIssue(
            "FIELD_WRITE_DISCARDED",
            f"Field {binding.field!r} is cleared right after BEFORE_EMISSION, "
            "so a write at this stage never reaches the simulation.",
            kernel_id=kernel.kernel_id,
            hint="Write grid fields at AFTER_PARTICLE_TRANSFER or BEFORE_PRESSURE.",
        )]
    if (kernel.domain is EffectDomain.PARTICLES
            and kernel.stage in _AFTER_P2G_BEFORE_G2P
            and binding.field == "particle_velocity"):
        return [EffectIssue(
            "PARTICLE_VELOCITY_WRITE_OVERWRITTEN",
            f"particle_velocity written at {kernel.stage.value} is replaced by the "
            "grid velocity under PIC/APIC transfer, and FLIP keeps only part of "
            "it without pressure projection.",
            severity="WARNING",
            kernel_id=kernel.kernel_id,
            hint="Apply particle velocity changes at BEFORE_EMISSION (before "
                 "particle-to-grid transfer) or AFTER_PARTICLE_UPDATE.",
        )]
    return []


def validate_field_binding(
    binding: FieldBindingV2,
    kernel: KernelDefinitionV2,
    *,
    safe_mode: bool,
) -> tuple[EffectIssue, ...]:
    spec = FIELD_SPECS.get(binding.field)
    if spec is None:
        return (EffectIssue(
            "UNKNOWN_FIELD",
            f"Unknown solver field {binding.field!r}.",
            kernel_id=kernel.kernel_id,
            hint="Choose a field from the API browser.",
        ),)
    issues: list[EffectIssue] = []
    if spec.domain is not kernel.domain:
        issues.append(EffectIssue(
            "FIELD_DOMAIN_MISMATCH",
            f"Field {binding.field!r} belongs to {spec.domain.value}, not {kernel.domain.value}.",
            kernel_id=kernel.kernel_id,
            hint="Change the kernel domain or choose a field in the same domain.",
        ))
    if kernel.stage not in spec.legal_stages:
        issues.append(EffectIssue(
            "FIELD_STAGE_MISMATCH",
            f"Field {binding.field!r} is not available at stage {kernel.stage.value}.",
            kernel_id=kernel.kernel_id,
        ))
    if binding.access in {FieldAccess.ATOMIC_SCATTER, FieldAccess.UNSAFE_ARBITRARY} and safe_mode:
        issues.append(EffectIssue(
            "UNSAFE_ACCESS_REQUIRES_EXPERT",
            f"Access mode {binding.access.value} requires Expert mode.",
            kernel_id=kernel.kernel_id,
            hint="Use a current-element access mode or explicitly enable Expert mode.",
        ))
    elif binding.access not in spec.allowed_access and binding.access not in {
        FieldAccess.ATOMIC_SCATTER,
        FieldAccess.UNSAFE_ARBITRARY,
    }:
        issues.append(EffectIssue(
            "FIELD_ACCESS_NOT_ALLOWED",
            f"Field {binding.field!r} does not allow {binding.access.value} access.",
            kernel_id=kernel.kernel_id,
        ))
    if spec.structural and binding.access is not FieldAccess.READ:
        issues.append(EffectIssue(
            "STRUCTURAL_FIELD_WRITE_FORBIDDEN",
            f"Structural field {binding.field!r} is read-only.",
            kernel_id=kernel.kernel_id,
        ))
    if spec.domain is kernel.domain:
        issues.extend(_stage_consumption_issues(binding, kernel))
    return tuple(issues)


def requirements_for_bindings(bindings: tuple[FieldBindingV2, ...]) -> frozenset[str]:
    required: set[str] = set()
    for binding in bindings:
        spec = FIELD_SPECS.get(binding.field)
        if spec is not None:
            required.update(spec.prerequisites)
    return frozenset(required)
