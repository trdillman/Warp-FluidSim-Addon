"""Starter templates for custom effect authoring."""

from __future__ import annotations

from .enums import EffectCadence, EffectDomain, EffectStage, FieldAccess, ParameterKind
from .schema import (
    EffectDefinitionV2,
    FieldBindingV2,
    KernelDefinitionV2,
    ParameterDefinitionV2,
)


def _float(key, label, default, minimum=None, maximum=None, unit="NONE", description=""):
    return ParameterDefinitionV2(
        key=key,
        label=label,
        kind=ParameterKind.FLOAT,
        default=default,
        minimum=minimum,
        maximum=maximum,
        unit=unit,
        description=description,
    )


def _object(key, label, description=""):
    return ParameterDefinitionV2(
        key=key,
        label=label,
        kind=ParameterKind.OBJECT,
        default="",
        unit="OBJECT",
        description=description,
        animatable=False,
    )


def _kernel(
    kernel_id,
    label,
    source,
    *,
    domain=EffectDomain.PARTICLES,
    stage=EffectStage.AFTER_PARTICLE_TRANSFER,
    cadence=EffectCadence.EVERY_SUBSTEP,
    fields=(),
    params=(),
    description="",
):
    return KernelDefinitionV2(
        kernel_id=kernel_id,
        label=label,
        domain=domain,
        stage=stage,
        cadence=cadence,
        source=source.strip(),
        fields=tuple(fields),
        parameter_keys=tuple(params),
        description=description,
    )


def particle_force_template() -> EffectDefinitionV2:
    source = (
        "# Radial acceleration toward or away from the controller object\n"
        "delta = position - controller\n"
        "dist = wp.length(delta)\n"
        "if dist < radius and dist > 0.001:\n"
        "    factor = 1.0 - (dist / radius)\n"
        "    force_dir = wp.normalize(delta)\n"
        "    accel = force_dir * (strength * factor)\n"
        "    particle_velocity[particle_index] = particle_velocity[particle_index] + accel * solver_time_step"
    )
    return EffectDefinitionV2(
        effect_id="custom.particle_force",
        version="1.0.0",
        name="Particle Force",
        summary="Apply bounded radial or directional acceleration to liquid particles.",
        description="Scaffold for custom particle force fields with a spatial controller empty.",
        category="Force",
        author="Custom",
        license="GPL-3.0-or-later",
        built_in=False,
        parameters=(
            _float("strength", "Strength", 10.0, -1000.0, 1000.0, "m/s²", "Acceleration magnitude"),
            _float("radius", "Radius", 1.5, 0.01, 100.0, "m", "Effective radial falloff"),
            _object("controller", "Controller Object", "Spatial reference for effect origin"),
        ),
        kernels=(
            _kernel(
                "particle_force",
                "Apply Particle Force",
                source,
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.BEFORE_EMISSION,
                cadence=EffectCadence.EVERY_SUBSTEP,
                fields=(FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT),),
                params=("strength", "radius", "controller"),
                description="Radial acceleration applied before particle-to-grid transfer.",
            ),
        ),
        tags=("custom", "particle", "force"),
        cost_hint="LOW",
    )


def particle_attribute_template() -> EffectDefinitionV2:
    source = (
        "# Smooth exponential decay of particle temperature toward ambient\n"
        "cur_temp = particle_temperature[particle_index]\n"
        "diff = ambient_temp - cur_temp\n"
        "decay = 1.0 - wp.exp(-cooling_rate * solver_time_step)\n"
        "particle_temperature[particle_index] = cur_temp + diff * decay"
    )
    return EffectDefinitionV2(
        effect_id="custom.particle_attribute",
        version="1.0.0",
        name="Particle Attribute Modifier",
        summary="Modify particle attributes such as temperature, life, or density.",
        description="Scaffold for updating per-particle scalar channels each substep.",
        category="Attribute",
        author="Custom",
        license="GPL-3.0-or-later",
        built_in=False,
        parameters=(
            _float("cooling_rate", "Cooling Rate", 0.5, 0.0, 100.0, "1/s", "Decay rate constant"),
            _float("ambient_temp", "Ambient Temperature", 20.0, -100.0, 1000.0, "°C", "Target ambient temperature"),
        ),
        kernels=(
            _kernel(
                "particle_attribute_cooling",
                "Cooling Step",
                source,
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                cadence=EffectCadence.EVERY_SUBSTEP,
                fields=(FieldBindingV2("particle_temperature", FieldAccess.READ_WRITE_CURRENT),),
                params=("cooling_rate", "ambient_temp"),
                description="Decays particle temperature toward ambient.",
            ),
        ),
        tags=("custom", "particle", "temperature"),
        cost_hint="LOW",
    )


def voxel_acceleration_template() -> EffectDefinitionV2:
    source = (
        "# Add vertical buoyancy acceleration to the W velocity component\n"
        "velocity_w[flat_voxel_index] = velocity_w[flat_voxel_index] + buoyancy * solver_time_step"
    )
    return EffectDefinitionV2(
        effect_id="custom.voxel_acceleration",
        version="1.0.0",
        name="Active Voxel Force",
        summary="Apply acceleration directly to staggered grid velocity components.",
        description="Scaffold for sparse tiled leaf grid velocity manipulation before pressure solve.",
        category="Grid",
        author="Custom",
        license="GPL-3.0-or-later",
        built_in=False,
        parameters=(
            _float("buoyancy", "Buoyancy Strength", 9.8, -100.0, 100.0, "m/s²", "Vertical grid acceleration"),
        ),
        kernels=(
            _kernel(
                "voxel_buoyancy",
                "Apply Grid Buoyancy",
                source,
                domain=EffectDomain.ACTIVE_VOXELS,
                stage=EffectStage.BEFORE_PRESSURE,
                cadence=EffectCadence.EVERY_SUBSTEP,
                fields=(FieldBindingV2("velocity_w", FieldAccess.READ_WRITE_CURRENT),),
                params=("buoyancy",),
                description="Modifies leaf tile velocity before pressure projection.",
            ),
        ),
        tags=("custom", "voxel", "grid", "buoyancy"),
        cost_hint="LOW",
    )


def velocity_damping_template() -> EffectDefinitionV2:
    source = (
        "# Exponential velocity magnitude damping\n"
        "scale = wp.exp(-damping * solver_time_step)\n"
        "particle_velocity[particle_index] = particle_velocity[particle_index] * scale"
    )
    return EffectDefinitionV2(
        effect_id="custom.velocity_damping",
        version="1.0.0",
        name="Custom Velocity Damping",
        summary="Apply smooth timestep-aware exponential velocity decay.",
        description="Scaffold for particle damping post-advection.",
        category="Damping",
        author="Custom",
        license="GPL-3.0-or-later",
        built_in=False,
        parameters=(
            _float("damping", "Damping Factor", 0.2, 0.0, 10.0, "1/s", "Linear drag rate"),
        ),
        kernels=(
            _kernel(
                "damp_velocity",
                "Damp Velocity",
                source,
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_UPDATE,
                cadence=EffectCadence.EVERY_SUBSTEP,
                fields=(FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT),),
                params=("damping",),
                description="Reduces velocity after particle update.",
            ),
        ),
        tags=("custom", "particle", "damping"),
        cost_hint="LOW",
    )


def blank_custom_template() -> EffectDefinitionV2:
    source = (
        "# Custom Warp kernel body\n"
        "# Built-in: particle_index, position, velocity, time, solver_time_step\n"
        "particle_velocity[particle_index] = particle_velocity[particle_index] * factor"
    )
    return EffectDefinitionV2(
        effect_id="custom.blank",
        version="1.0.0",
        name="Blank Custom Effect",
        summary="Minimal starter effect for custom Warp kernel logic.",
        description="Scaffold for authoring custom logic from scratch.",
        category="Custom",
        author="Custom",
        license="GPL-3.0-or-later",
        built_in=False,
        parameters=(
            _float("factor", "Scaling Factor", 1.0, 0.0, 100.0, "NONE", "Uniform scaling factor"),
        ),
        kernels=(
            _kernel(
                "custom_kernel",
                "Custom Kernel",
                source,
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.BEFORE_EMISSION,
                cadence=EffectCadence.EVERY_SUBSTEP,
                fields=(FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT),),
                params=("factor",),
                description="Custom user kernel logic.",
            ),
        ),
        tags=("custom", "scratch"),
        cost_hint="LOW",
    )


TEMPLATES: dict[str, callable] = {
    "PARTICLE_FORCE": particle_force_template,
    "PARTICLE_ATTRIBUTE": particle_attribute_template,
    "VOXEL_ACCELERATION": voxel_acceleration_template,
    "VELOCITY_DAMPING": velocity_damping_template,
    "BLANK": blank_custom_template,
}


def starter_template_items() -> tuple[tuple[str, str, str], ...]:
    return (
        ("PARTICLE_FORCE", "Particle Force (Spatial Push/Attract)", "Radial acceleration force using controller empty"),
        ("PARTICLE_ATTRIBUTE", "Particle Attribute Modifier", "Scalar attribute manipulation such as temperature decay"),
        ("VOXEL_ACCELERATION", "Active Voxel Force (Grid Buoyancy)", "Direct grid acceleration on sparse leaf tiles"),
        ("VELOCITY_DAMPING", "Velocity Damping", "Timestep-aware exponential velocity decay"),
        ("BLANK", "Blank Custom Effect", "Minimal clean scaffold for new kernels"),
    )


def get_template_definition(template_key: str) -> EffectDefinitionV2:
    builder = TEMPLATES.get(template_key)
    if builder is None:
        raise KeyError(f"Unknown starter template {template_key!r}")
    return builder()
