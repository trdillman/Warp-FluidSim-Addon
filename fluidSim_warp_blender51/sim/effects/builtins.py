"""Curated artist-facing Custom Effects bundled with FluidSim."""

from __future__ import annotations

from functools import lru_cache

from .enums import EffectCadence, EffectDomain, EffectStage, FieldAccess, ParameterKind
from .schema import (
    EffectDefinitionV2,
    FieldBindingV2,
    KernelDefinitionV2,
    ParameterDefinitionV2,
)


def _float(key, label, default, minimum, maximum, unit, description):
    return ParameterDefinitionV2(
        key, label, ParameterKind.FLOAT, default,
        minimum=minimum, maximum=maximum, unit=unit,
        description=description,
    )


def _vector(key, label, default, unit, description):
    return ParameterDefinitionV2(
        key, label, ParameterKind.VECTOR, default,
        unit=unit, description=description,
    )


def _object(key, label, description):
    return ParameterDefinitionV2(
        key, label, ParameterKind.OBJECT, "",
        unit="OBJECT", description=description, animatable=False,
    )


def _kernel(kernel_id, label, source, *, domain=EffectDomain.PARTICLES,
            stage=EffectStage.BEFORE_EMISSION,
            cadence=EffectCadence.EVERY_SUBSTEP, fields=(), params=(), description=""):
    return KernelDefinitionV2(
        kernel_id=kernel_id,
        label=label,
        domain=domain,
        stage=stage,
        cadence=cadence,
        source=source,
        fields=tuple(fields),
        parameter_keys=tuple(params),
        description=description,
    )


# 1.1.0: particle-velocity kernels run at BEFORE_EMISSION, before
# particle-to-grid transfer, instead of AFTER_PARTICLE_TRANSFER where PIC/APIC
# discarded the edit (review B68). The 1.0.0 content digests stay trusted
# through trust.RETIRED_BUILTIN_DIGESTS so saved scenes keep running.
_VELOCITY_STAGE_VERSION = "1.1.0"


def _definition(effect_id, name, summary, description, category, *,
                parameters, kernels, tags=(), cost="LOW",
                version=_VELOCITY_STAGE_VERSION):
    return EffectDefinitionV2(
        effect_id=effect_id,
        version=version,
        name=name,
        summary=summary,
        description=description,
        category=category,
        author="FluidSim",
        license="GPL-3.0-or-later",
        built_in=True,
        package_origin="builtin://fluidsim/1.0.251",
        tags=tuple(tags),
        cost_hint=cost,
        parameters=tuple(parameters),
        kernels=tuple(kernels),
    )


@lru_cache(maxsize=1)
def builtin_effect_definitions() -> tuple[EffectDefinitionV2, ...]:
    velocity_rw = FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT)
    temperature_r = FieldBindingV2("particle_temperature", FieldAccess.READ)
    temperature_rw = FieldBindingV2("particle_temperature", FieldAccess.READ_WRITE_CURRENT)
    voxel_temperature_rw = FieldBindingV2("temperature", FieldAccess.READ_WRITE_CURRENT)

    vortex = _definition(
        "fluidsim.vortex_force", "Vortex Force",
        "Swirl liquid around an artist-positioned controller with radial falloff.",
        "Adds controlled tangential motion around a chosen axis, with optional suction and lift. "
        "Use the owned controller Empty to place the vortex directly in the Blender scene.",
        "Forces",
        parameters=(
            _object("controller", "Controller", "Scene object whose origin places the vortex center."),
            _vector("axis", "Axis", (0.0, 0.0, 1.0), "DIRECTION", "Local swirl axis in world space."),
            _float("strength", "Strength", 4.0, -100.0, 100.0, "m/s²", "Tangential acceleration at the controller center."),
            _float("radius", "Radius", 2.0, 0.001, 1000.0, "m", "Maximum radial influence distance."),
            _float("suction", "Suction", 0.0, -100.0, 100.0, "1/s²", "Radial pull toward the vortex axis."),
            _float("lift", "Lift", 0.0, -100.0, 100.0, "m/s²", "Acceleration along the vortex axis."),
        ),
        kernels=(_kernel(
            "vortex_force", "Apply Vortex",
            """offset = position - controller
axis_length = wp.length(axis)
distance = wp.length(offset)
if axis_length > 1.0e-6 and distance < radius:
    axis_n = axis / axis_length
    radial = offset - axis_n * wp.dot(offset, axis_n)
    radial_length = wp.length(radial)
    if radial_length > 1.0e-6:
        influence = wp.clamp(1.0 - radial_length / radius, 0.0, 1.0)
        tangent = wp.normalize(wp.cross(axis_n, radial))
        updated = particle_velocity[particle_index]
        updated = updated + tangent * (strength * influence * solver_time_step)
        updated = updated - radial * (suction * influence * solver_time_step)
        updated = updated + axis_n * (lift * influence * solver_time_step)
        particle_velocity[particle_index] = updated""",
            fields=(velocity_rw,), params=("controller", "axis", "strength", "radius", "suction", "lift"),
            description="Adds bounded tangential, radial, and axial acceleration to live liquid particles.",
        ),),
        tags=("swirl", "controller", "force"), cost="LOW",
    )

    turbulence = _definition(
        "fluidsim.curl_turbulence", "Curl Turbulence",
        "Add deterministic, animated divergence-like turbulent motion to particles.",
        "Applies a smooth trigonometric curl-style velocity perturbation. The effect is deterministic "
        "for a given seed, position, time, and stack order and requires no topology rebuild.",
        "Forces",
        parameters=(
            _float("strength", "Strength", 1.0, 0.0, 100.0, "m/s²", "Magnitude of the turbulent acceleration."),
            _float("frequency", "Frequency", 1.5, 0.001, 1000.0, "1/m", "Spatial frequency of turbulent features."),
            _float("drift", "Drift", 0.5, -100.0, 100.0, "1/s", "Animation rate of the turbulence pattern."),
            _float("seed", "Seed", 17.0, -100000.0, 100000.0, "NONE", "Deterministic phase offset."),
        ),
        kernels=(_kernel(
            "curl_turbulence", "Apply Turbulence",
            """sample = position * frequency + wp.vec3(seed, seed * 1.37, seed * 2.11)
phase = time * drift
noise = wp.vec3(
    wp.sin(sample[1] + phase) - wp.sin(sample[2] - phase),
    wp.sin(sample[2] + phase * 1.17) - wp.sin(sample[0] - phase),
    wp.sin(sample[0] + phase * 0.73) - wp.sin(sample[1] - phase * 1.31),
)
particle_velocity[particle_index] = particle_velocity[particle_index] + noise * (strength * solver_time_step)""",
            fields=(velocity_rw,), params=("strength", "frequency", "drift", "seed"),
            description="Adds deterministic spatially varying motion without changing particle positions directly.",
        ),),
        tags=("noise", "turbulence", "force"), cost="LOW",
    )

    pulse = _definition(
        "fluidsim.radial_pulse", "Radial Pulse",
        "Push or pull liquid radially from a controller with a finite radius.",
        "Creates a controllable spherical impulse field around an owned Empty. Positive strength pushes "
        "outward; negative strength pulls inward, with a smooth linear falloff to the radius.",
        "Forces",
        parameters=(
            _object("controller", "Controller", "Scene object whose origin places the pulse center."),
            _float("strength", "Strength", 6.0, -100.0, 100.0, "m/s²", "Outward or inward radial acceleration."),
            _float("radius", "Radius", 2.5, 0.001, 1000.0, "m", "Maximum pulse influence distance."),
        ),
        kernels=(_kernel(
            "radial_pulse", "Apply Pulse",
            """offset = position - controller
distance = wp.length(offset)
if distance > 1.0e-6 and distance < radius:
    influence = wp.clamp(1.0 - distance / radius, 0.0, 1.0)
    direction = offset / distance
    particle_velocity[particle_index] = particle_velocity[particle_index] + direction * (strength * influence * solver_time_step)""",
            fields=(velocity_rw,), params=("controller", "strength", "radius"),
            description="Applies a bounded radial acceleration around the controller origin.",
        ),),
        tags=("burst", "controller", "force"), cost="LOW",
    )

    split = _definition(
        "fluidsim.planar_split", "Planar Split",
        "Separate liquid to opposite sides of an artist-positioned plane.",
        "Uses a controller origin and world-space normal to push particles away from a plane. "
        "The finite width makes the effect easy to art-direct without global scene impact.",
        "Forces",
        parameters=(
            _object("controller", "Controller", "Scene object whose origin places the split plane."),
            _vector("normal", "Plane Normal", (1.0, 0.0, 0.0), "DIRECTION", "World-space normal of the split plane."),
            _float("strength", "Strength", 3.0, -100.0, 100.0, "m/s²", "Acceleration away from the plane."),
            _float("width", "Width", 1.0, 0.001, 1000.0, "m", "Half-width of the plane influence band."),
        ),
        kernels=(_kernel(
            "planar_split", "Apply Split",
            """normal_length = wp.length(normal)
if normal_length > 1.0e-6:
    normal_n = normal / normal_length
    signed_distance = wp.dot(position - controller, normal_n)
    absolute_distance = wp.abs(signed_distance)
    if absolute_distance < width:
        side = 1.0
        if signed_distance < 0.0:
            side = -1.0
        influence = wp.clamp(1.0 - absolute_distance / width, 0.0, 1.0)
        particle_velocity[particle_index] = particle_velocity[particle_index] + normal_n * (side * strength * influence * solver_time_step)""",
            fields=(velocity_rw,), params=("controller", "normal", "strength", "width"),
            description="Pushes current particles to opposing sides of the configured plane.",
        ),),
        tags=("plane", "controller", "separation"), cost="LOW",
    )

    damping = _definition(
        "fluidsim.velocity_damping", "Velocity Damping",
        "Reduce particle velocity smoothly without changing solver topology.",
        "Applies stable per-substep velocity damping to the current liquid particle. Use it to settle "
        "motion, tame injected energy, or create a localized viscous-looking art direction layer.",
        "Utilities",
        parameters=(
            _float("damping", "Damping", 1.0, 0.0, 1000.0, "1/s", "Fractional velocity reduction rate."),
        ),
        kernels=(_kernel(
            "velocity_damping", "Damp Velocity",
            """factor = wp.clamp(1.0 - damping * solver_time_step, 0.0, 1.0)
particle_velocity[particle_index] = particle_velocity[particle_index] * factor""",
            fields=(velocity_rw,), params=("damping",),
            description="Scales each current particle velocity by a bounded timestep-aware factor.",
        ),),
        tags=("damping", "stability", "utility"), cost="LOW",
    )

    temperature_lift = _definition(
        "fluidsim.temperature_lift", "Temperature Lift",
        "Convert particle temperature into buoyant lift and cool it after advection.",
        "A two-kernel effect: the first applies lift before particle-to-grid transfer; the second cools "
        "temperature after particle update. It demonstrates one artist effect spanning two solver stages.",
        "Thermal",
        parameters=(
            _vector("direction", "Lift Direction", (0.0, 0.0, 1.0), "DIRECTION", "World-space lift direction."),
            _float("ambient", "Ambient", 0.0, -10000.0, 10000.0, "TEMPERATURE", "Temperature with zero lift."),
            _float("buoyancy", "Buoyancy", 0.5, -100.0, 100.0, "m/s²/K", "Acceleration per temperature unit above ambient."),
            _float("cooling", "Cooling", 0.25, 0.0, 1000.0, "1/s", "Rate at which temperature returns to ambient."),
        ),
        kernels=(
            _kernel(
                "temperature_lift", "Apply Thermal Lift",
                """direction_length = wp.length(direction)
if direction_length > 1.0e-6:
    direction_n = direction / direction_length
    excess = particle_temperature[particle_index] - ambient
    particle_velocity[particle_index] = particle_velocity[particle_index] + direction_n * (excess * buoyancy * solver_time_step)""",
                fields=(temperature_r, velocity_rw),
                params=("direction", "ambient", "buoyancy"),
                stage=EffectStage.BEFORE_EMISSION,
                description="Converts excess particle temperature into directional acceleration.",
            ),
            _kernel(
                "temperature_cooling", "Cool Particle Temperature",
                """factor = wp.clamp(1.0 - cooling * solver_time_step, 0.0, 1.0)
particle_temperature[particle_index] = ambient + (particle_temperature[particle_index] - ambient) * factor""",
                fields=(temperature_rw,),
                params=("ambient", "cooling"),
                stage=EffectStage.AFTER_PARTICLE_UPDATE,
                description="Returns particle temperature toward ambient after advection.",
            ),
        ),
        tags=("temperature", "buoyancy", "multi-stage"), cost="MEDIUM",
    )

    speed_limit = _definition(
        "fluidsim.particle_speed_limit", "Particle Speed Limit",
        "Clamp particle speeds to an explicit artist-controlled maximum.",
        "Limits the magnitude of each liquid particle velocity while preserving direction. This is a "
        "predictable utility for art direction and for guarding aggressive custom-force stacks.",
        "Utilities",
        parameters=(
            _float("maximum_speed", "Maximum Speed", 10.0, 0.001, 10000.0, "m/s", "Maximum allowed particle speed."),
        ),
        kernels=(_kernel(
            "particle_speed_limit", "Limit Speed",
            """current = particle_velocity[particle_index]
speed = wp.length(current)
if speed > maximum_speed:
    particle_velocity[particle_index] = current * (maximum_speed / speed)""",
            fields=(velocity_rw,), params=("maximum_speed",),
            description="Clamps current-particle speed without changing velocity direction.",
        ),),
        tags=("clamp", "stability", "utility"), cost="LOW",
    )

    voxel_cooling = _definition(
        "fluidsim.voxel_cooling", "Voxel Cooling",
        "Relax active-grid temperature toward ambient once per frame.",
        "Runs on active sparse voxels and therefore requests topology. It provides a clear example of a "
        "grid-domain effect whose cadence is once per frame rather than every solver substep. No solver "
        "phase reads grid temperature (it is cleared every substep and is separate from particle "
        "temperature), so this effect does not change the liquid's motion or output; use Temperature "
        "Lift to act on particle temperature.",
        "Thermal",
        parameters=(
            _float("ambient", "Ambient", 0.0, -10000.0, 10000.0, "TEMPERATURE", "Target grid temperature."),
            _float("cooling", "Cooling", 0.5, 0.0, 1000.0, "1/s", "Grid temperature relaxation rate."),
        ),
        kernels=(_kernel(
            "voxel_cooling", "Cool Active Voxels",
            """factor = wp.clamp(1.0 - cooling * real_time_step, 0.0, 1.0)
temperature[flat_voxel_index] = ambient + (temperature[flat_voxel_index] - ambient) * factor""",
            domain=EffectDomain.ACTIVE_VOXELS,
            stage=EffectStage.AFTER_PARTICLE_TRANSFER,
            cadence=EffectCadence.ONCE_PER_FRAME,
            fields=(voxel_temperature_rw,), params=("ambient", "cooling"),
            description="Relaxes temperature on active non-collision voxels once per display frame.",
        ),),
        # Behaviour unchanged (documentation only), so the version stays.
        tags=("temperature", "grid", "cooling"), cost="MEDIUM", version="1.0.0",
    )

    return (
        vortex, turbulence, pulse, split, damping,
        temperature_lift, speed_limit, voxel_cooling,
    )
