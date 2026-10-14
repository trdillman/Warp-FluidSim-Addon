"""Curated example Custom Effects bundled as package templates for FluidSim.

These 10 custom effect examples demonstrate various physical behaviors,
domains (PARTICLES and ACTIVE_VOXELS), stages (BEFORE_EMISSION, AFTER_PARTICLE_TRANSFER,
BEFORE_PRESSURE, AFTER_PARTICLE_UPDATE), and controller-driven spatial falloffs.
Particle-velocity forces run at BEFORE_EMISSION, before particle-to-grid
transfer, so every transfer mode (PIC, FLIP, APIC) keeps them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

# Enable running both as package submodule and standalone script
try:
    from ....sim.effects.enums import (
        EffectCadence,
        EffectDomain,
        EffectStage,
        FieldAccess,
        ParameterKind,
    )
    from ....sim.effects.package import write_effect_package
    from ....sim.effects.schema import (
        EffectDefinitionV2,
        FieldBindingV2,
        KernelDefinitionV2,
        ParameterDefinitionV2,
    )
except (ImportError, ValueError):
    import importlib
    _root = Path(__file__).resolve().parents[3].name
    _enums = importlib.import_module(f"{_root}.sim.effects.enums")
    EffectCadence = _enums.EffectCadence
    EffectDomain = _enums.EffectDomain
    EffectStage = _enums.EffectStage
    FieldAccess = _enums.FieldAccess
    ParameterKind = _enums.ParameterKind
    _pkg_mod = importlib.import_module(f"{_root}.sim.effects.package")
    write_effect_package = _pkg_mod.write_effect_package
    _schema = importlib.import_module(f"{_root}.sim.effects.schema")
    EffectDefinitionV2 = _schema.EffectDefinitionV2
    FieldBindingV2 = _schema.FieldBindingV2
    KernelDefinitionV2 = _schema.KernelDefinitionV2
    ParameterDefinitionV2 = _schema.ParameterDefinitionV2


def _param_float(key: str, label: str, default: float, minimum: float, maximum: float, unit: str, description: str) -> ParameterDefinitionV2:
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


def _param_vector(key: str, label: str, default: tuple[float, float, float], unit: str, description: str) -> ParameterDefinitionV2:
    return ParameterDefinitionV2(
        key=key,
        label=label,
        kind=ParameterKind.VECTOR,
        default=default,
        unit=unit,
        description=description,
    )


def _param_object(key: str, label: str, description: str) -> ParameterDefinitionV2:
    return ParameterDefinitionV2(
        key=key,
        label=label,
        kind=ParameterKind.OBJECT,
        default="",
        unit="OBJECT",
        animatable=False,
        description=description,
    )


def _field(field: str, access: FieldAccess) -> FieldBindingV2:
    return FieldBindingV2(field, access)


# Authoritative shared field bindings
_VEL_RW = _field("particle_velocity", FieldAccess.READ_WRITE_CURRENT)
_VEL_R = _field("particle_velocity", FieldAccess.READ)
_TEMP_R = _field("particle_temperature", FieldAccess.READ)
_TEMP_RW = _field("particle_temperature", FieldAccess.READ_WRITE_CURRENT)
_VISC_RW = _field("viscosity", FieldAccess.READ_WRITE_CURRENT)


EXAMPLE_EFFECT_DEFINITIONS: tuple[EffectDefinitionV2, ...] = (
    # 1. Thermal Updraft
    EffectDefinitionV2(
        effect_id="example.thermal_updraft",
        version="1.0.0",
        name="Thermal Updraft",
        summary="Convective thermal buoyancy based on particle temperature.",
        description="Accelerates warm fluid particles upward against gravity proportional to the temperature delta above ambient.",
        category="Thermal",
        author="FluidSim Examples",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin="example://custom-effects/1.0.251",
        tags=("thermal", "buoyancy", "force"),
        cost_hint="LOW",
        parameters=(
            _param_float("ambient_temp", "Ambient Temperature", 20.0, -100.0, 1000.0, "°C", "Reference ambient temperature."),
            _param_float("buoyancy", "Buoyancy Coefficient", 0.5, -50.0, 50.0, "m/s²/°C", "Upward acceleration per degree above ambient."),
            _param_vector("up_direction", "Up Direction", (0.0, 0.0, 1.0), "DIRECTION", "World space upward direction vector."),
            _param_float("max_accel", "Max Acceleration", 20.0, 0.0, 100.0, "m/s²", "Maximum allowed upward acceleration."),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="thermal_updraft",
                label="Apply Thermal Updraft",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.BEFORE_EMISSION,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="""dir_len = wp.length(up_direction)
if dir_len > 1.0e-6:
    up_n = up_direction / dir_len
    temp_delta = particle_temperature[particle_index] - ambient_temp
    if temp_delta > 0.0:
        raw_accel = temp_delta * buoyancy
        accel = wp.clamp(raw_accel, 0.0, max_accel)
        particle_velocity[particle_index] = particle_velocity[particle_index] + up_n * (accel * solver_time_step)""",
                fields=(_VEL_RW, _TEMP_R),
                parameter_keys=("ambient_temp", "buoyancy", "up_direction", "max_accel"),
                description="Applies buoyant upward force based on particle temperature.",
            ),
        ),
    ),

    # 2. Point Attractor
    EffectDefinitionV2(
        effect_id="example.point_attractor",
        version="1.0.0",
        name="Point Attractor",
        summary="Attract or repel liquid particles around a controller object.",
        description="Pulls particles toward or pushes particles away from an artist controller object with distance falloff and softening.",
        category="Forces",
        author="FluidSim Examples",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin="example://custom-effects/1.0.251",
        tags=("attractor", "gravity", "controller"),
        cost_hint="LOW",
        parameters=(
            _param_object("controller", "Controller", "Scene object placing the attractor origin."),
            _param_float("strength", "Strength", 5.0, -100.0, 100.0, "m/s²", "Pull (positive) or push (negative) acceleration."),
            _param_float("radius", "Radius", 3.0, 0.01, 100.0, "m", "Maximum radius of attraction."),
            _param_float("softening", "Softening", 0.2, 0.001, 100.0, "m", "Softening distance to prevent singularity at center."),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="point_attractor",
                label="Apply Point Attractor",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.BEFORE_EMISSION,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="""offset = controller - position
dist = wp.length(offset)
if dist < radius and dist > 1.0e-6:
    dir_n = offset / dist
    effective_dist = dist + softening
    falloff = 1.0 - (dist / radius)
    accel = strength * falloff / (effective_dist * effective_dist)
    particle_velocity[particle_index] = particle_velocity[particle_index] + dir_n * (accel * solver_time_step)""",
                fields=(_VEL_RW,),
                parameter_keys=("controller", "strength", "radius", "softening"),
                description="Applies central radial attraction toward controller.",
            ),
        ),
    ),

    # 3. Helical Vortex
    EffectDefinitionV2(
        effect_id="example.helical_vortex",
        version="1.0.0",
        name="Helical Vortex",
        summary="Spiraling corkscrew flow around a directional cylinder axis.",
        description="Creates combined azimuthal swirl and axial propulsion within a cylindrical zone around a controller object.",
        category="Forces",
        author="FluidSim Examples",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin="example://custom-effects/1.0.251",
        tags=("vortex", "helix", "cylinder"),
        cost_hint="LOW",
        parameters=(
            _param_object("controller", "Controller", "Scene object placing cylinder origin."),
            _param_vector("axis", "Axis", (0.0, 0.0, 1.0), "DIRECTION", "Cylinder center axis vector."),
            _param_float("radius", "Radius", 2.0, 0.01, 100.0, "m", "Cylinder influence radius."),
            _param_float("swirl", "Swirl Strength", 5.0, -100.0, 100.0, "m/s²", "Tangential swirl acceleration around axis."),
            _param_float("axial_drive", "Axial Drive", 2.0, -100.0, 100.0, "m/s²", "Linear acceleration along cylinder axis."),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="helical_vortex",
                label="Apply Helical Vortex",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.BEFORE_EMISSION,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="""axis_len = wp.length(axis)
if axis_len > 1.0e-6:
    axis_n = axis / axis_len
    offset = position - controller
    along_axis = wp.dot(offset, axis_n)
    radial = offset - axis_n * along_axis
    radial_len = wp.length(radial)
    if radial_len > 1.0e-6 and radial_len < radius:
        tangent = wp.cross(axis_n, radial / radial_len)
        falloff = 1.0 - (radial_len / radius)
        accel = tangent * (swirl * falloff) + axis_n * (axial_drive * falloff)
        particle_velocity[particle_index] = particle_velocity[particle_index] + accel * solver_time_step""",
                fields=(_VEL_RW,),
                parameter_keys=("controller", "axis", "radius", "swirl", "axial_drive"),
                description="Swirls and propels particles helically along axis.",
            ),
        ),
    ),

    # 4. Directional Wind
    EffectDefinitionV2(
        effect_id="example.directional_wind",
        version="1.0.0",
        name="Directional Wind",
        summary="Aerodynamic drag force with time-varying periodic gusts.",
        description="Accelerates particles toward a target wind velocity with smooth falloff and oscillatory gusts inside a localized volume.",
        category="Forces",
        author="FluidSim Examples",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin="example://custom-effects/1.0.251",
        tags=("wind", "gust", "drag"),
        cost_hint="LOW",
        parameters=(
            _param_object("controller", "Controller", "Scene object placing wind zone center."),
            _param_vector("direction", "Direction", (1.0, 0.0, 0.0), "DIRECTION", "Wind flow direction."),
            _param_float("speed", "Base Speed", 6.0, 0.0, 100.0, "m/s", "Steady wind stream speed."),
            _param_float("drag", "Drag Coefficient", 2.0, 0.0, 20.0, "1/s", "Coupling drag rate between fluid and wind."),
            _param_float("influence_radius", "Influence Radius", 3.0, 0.01, 100.0, "m", "Spherical reach of the wind effect."),
            _param_float("gust_freq", "Gust Frequency", 1.5, 0.0, 20.0, "1/s", "Frequency of periodic gust fluctuations."),
            _param_float("gust_amp", "Gust Amplitude", 2.0, 0.0, 20.0, "m/s", "Velocity boost added during gust peaks."),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="directional_wind",
                label="Apply Directional Wind",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.BEFORE_EMISSION,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="""dir_len = wp.length(direction)
offset = position - controller
dist = wp.length(offset)
if dir_len > 1.0e-6 and dist < influence_radius:
    dir_n = direction / dir_len
    falloff = 1.0 - (dist / influence_radius)
    gust = wp.sin(time * gust_freq + dist) * gust_amp
    target_vel = dir_n * (speed + gust)
    rel_vel = target_vel - particle_velocity[particle_index]
    drag_accel = rel_vel * (drag * falloff)
    particle_velocity[particle_index] = particle_velocity[particle_index] + drag_accel * solver_time_step""",
                fields=(_VEL_RW,),
                parameter_keys=("controller", "direction", "speed", "drag", "influence_radius", "gust_freq", "gust_amp"),
                description="Applies gusting aerodynamic drag force toward wind direction.",
            ),
        ),
    ),

    # 5. Centrifugal Spinner
    EffectDefinitionV2(
        effect_id="example.centrifugal_spinner",
        version="1.0.0",
        name="Centrifugal Spinner",
        summary="Fling liquid radially outward from a rotating shaft.",
        description="Applies centrifugal acceleration omega^2 * r around a designated controller axis, simulating high-speed rotating turbines or centrifuges.",
        category="Forces",
        author="FluidSim Examples",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin="example://custom-effects/1.0.251",
        tags=("centrifugal", "rotation", "force"),
        cost_hint="LOW",
        parameters=(
            _param_object("controller", "Controller", "Scene object placing rotor center."),
            _param_vector("axis", "Rotation Axis", (0.0, 0.0, 1.0), "DIRECTION", "Axis of rotation in world space."),
            _param_float("angular_speed", "Angular Speed", 5.0, -100.0, 100.0, "rad/s", "Rotational angular velocity."),
            _param_float("max_radius", "Max Radius", 2.5, 0.01, 50.0, "m", "Maximum reach of the centrifugal field."),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="centrifugal_spinner",
                label="Apply Centrifugal Force",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.BEFORE_EMISSION,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="""axis_len = wp.length(axis)
offset = position - controller
if axis_len > 1.0e-6:
    axis_n = axis / axis_len
    along_axis = wp.dot(offset, axis_n)
    radial = offset - axis_n * along_axis
    r = wp.length(radial)
    if r > 1.0e-6 and r < max_radius:
        radial_n = radial / r
        accel_mag = (angular_speed * angular_speed) * r
        particle_velocity[particle_index] = particle_velocity[particle_index] + radial_n * (accel_mag * solver_time_step)""",
                fields=(_VEL_RW,),
                parameter_keys=("controller", "axis", "angular_speed", "max_radius"),
                description="Accelerates particles outward perpendicular to rotation axis.",
            ),
        ),
    ),

    # 6. Surface Tension Bead
    EffectDefinitionV2(
        effect_id="example.surface_tension_bead",
        version="1.0.0",
        name="Surface Tension Bead",
        summary="Dampen boundary droplet dispersion to form cohesive beads.",
        description="Finds sparsely filled surface voxels from the live particle count and damps their grid velocity, so liquid sheets and droplets stay tighter.",
        category="Surface",
        author="FluidSim Examples",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin="example://custom-effects/1.0.251",
        tags=("surface", "tension", "droplet"),
        cost_hint="LOW",
        parameters=(
            _param_float("surface_count", "Surface Particle Count", 6.0, 1.0, 64.0, "particles", "Voxels holding fewer particles than this count as liquid surface."),
            _param_float("damping_factor", "Damping Factor", 3.0, 0.0, 50.0, "1/s", "Velocity damping strength in the emptiest surface voxels."),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="surface_tension_bead",
                label="Apply Surface Beading",
                domain=EffectDomain.ACTIVE_VOXELS,
                stage=EffectStage.BEFORE_PRESSURE,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="""count = float(voxel_counts[flat_voxel_index])
if count > 0.0 and count < surface_count:
    deficit = (surface_count - count) / surface_count
    factor = wp.clamp(1.0 - damping_factor * deficit * solver_time_step, 0.0, 1.0)
    velocity_u[flat_voxel_index] = velocity_u[flat_voxel_index] * factor
    velocity_v[flat_voxel_index] = velocity_v[flat_voxel_index] * factor
    velocity_w[flat_voxel_index] = velocity_w[flat_voxel_index] * factor""",
                fields=(
                    _field("voxel_counts", FieldAccess.READ),
                    _field("velocity_u", FieldAccess.READ_WRITE_CURRENT),
                    _field("velocity_v", FieldAccess.READ_WRITE_CURRENT),
                    _field("velocity_w", FieldAccess.READ_WRITE_CURRENT),
                ),
                parameter_keys=("surface_count", "damping_factor"),
                description="Damps grid velocity in sparsely filled surface voxels.",
            ),
        ),
    ),

    # 7. Viscous Heating
    EffectDefinitionV2(
        effect_id="example.viscous_heating",
        version="1.0.0",
        name="Viscous Heating",
        summary="Dissipate kinetic speed into thermal heating increments.",
        description="Increases particle temperature when velocity exceeds an energy threshold, simulating friction heating in fast flows or lava.",
        category="Thermal",
        author="FluidSim Examples",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin="example://custom-effects/1.0.251",
        tags=("thermal", "kinetic", "heating"),
        cost_hint="LOW",
        parameters=(
            _param_float("heating_rate", "Heating Rate", 0.5, 0.0, 50.0, "°C/(m/s)", "Temperature increase per unit speed per second."),
            _param_float("speed_threshold", "Speed Threshold", 1.0, 0.0, 50.0, "m/s", "Minimum velocity magnitude before friction heating begins."),
            _param_float("max_temp", "Max Temperature", 500.0, -100.0, 5000.0, "°C", "Upper ceiling on simulated fluid temperature."),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="viscous_heating",
                label="Apply Friction Heating",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_UPDATE,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="""vel = particle_velocity[particle_index]
speed = wp.length(vel)
if speed > speed_threshold:
    excess_speed = speed - speed_threshold
    heat_inc = excess_speed * heating_rate * solver_time_step
    cur_temp = particle_temperature[particle_index]
    new_temp = wp.min(cur_temp + heat_inc, max_temp)
    particle_temperature[particle_index] = new_temp""",
                fields=(_VEL_R, _TEMP_RW),
                parameter_keys=("heating_rate", "speed_threshold", "max_temp"),
                description="Converts excess particle speed into local temperature rise.",
            ),
        ),
    ),

    # 8. Periodic Ripples
    EffectDefinitionV2(
        effect_id="example.periodic_ripples",
        version="1.0.0",
        name="Periodic Ripples",
        summary="Concentric circular traveling wave ripples around a controller.",
        description="Generates vertical oscillatory harmonic wave pulses propagating outward radially from a source controller.",
        category="Waves",
        author="FluidSim Examples",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin="example://custom-effects/1.0.251",
        tags=("waves", "ripples", "harmonic"),
        cost_hint="LOW",
        parameters=(
            _param_object("controller", "Controller", "Scene object placing wave origin."),
            _param_float("amplitude", "Wave Amplitude", 2.0, 0.0, 50.0, "m/s²", "Peak vertical acceleration."),
            _param_float("wavenumber", "Wavenumber", 4.0, 0.01, 100.0, "1/m", "Spatial frequency of concentric ripple rings."),
            _param_float("wave_speed", "Wave Speed", 3.0, 0.0, 50.0, "m/s", "Radial propagation speed of ripples."),
            _param_float("decay", "Spatial Decay", 0.4, 0.0, 10.0, "1/m", "Exponential distance damping rate."),
            _param_float("radius", "Max Radius", 4.0, 0.1, 50.0, "m", "Maximum radius of wave propagation."),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="periodic_ripples",
                label="Apply Periodic Ripples",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.BEFORE_EMISSION,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="""offset = position - controller
dist = wp.length(offset)
if dist < radius and dist > 1.0e-6:
    phase = wavenumber * dist - time * (wavenumber * wave_speed)
    decay_factor = wp.exp(-decay * dist)
    accel_z = amplitude * wp.sin(phase) * decay_factor
    particle_velocity[particle_index] = particle_velocity[particle_index] + wp.vec3(0.0, 0.0, accel_z * solver_time_step)""",
                fields=(_VEL_RW,),
                parameter_keys=("controller", "amplitude", "wavenumber", "wave_speed", "decay", "radius"),
                description="Generates harmonic radial wave ripples in particle velocity.",
            ),
        ),
    ),

    # 9. Bounding Box Confinement
    EffectDefinitionV2(
        effect_id="example.bounding_box_confinement",
        version="1.0.0",
        name="Bounding Box Confinement",
        summary="Elastic boundary barrier keeping particles inside a 3D box.",
        description="Pushes runaway particles inward with smooth restorative penalty forces when approaching or crossing controller half-extents.",
        category="Boundary",
        author="FluidSim Examples",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin="example://custom-effects/1.0.251",
        tags=("boundary", "confinement", "box"),
        cost_hint="LOW",
        parameters=(
            _param_object("controller", "Controller", "Scene object placing box center."),
            _param_vector("half_extents", "Half Extents", (2.0, 2.0, 2.0), "m", "Half width, length, and height of boundary box."),
            _param_float("margin", "Soft Margin", 0.3, 0.01, 2.0, "m", "Distance before boundary where spring begins resisting."),
            _param_float("stiffness", "Spring Stiffness", 30.0, 0.0, 200.0, "1/s²", "Restorative acceleration stiffness constant."),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="bounding_box_confinement",
                label="Apply Box Confinement",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.BEFORE_EMISSION,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="""rel_pos = position - controller
ax = wp.abs(rel_pos[0])
ay = wp.abs(rel_pos[1])
az = wp.abs(rel_pos[2])
hx = half_extents[0]
hy = half_extents[1]
hz = half_extents[2]
push_x = 0.0
push_y = 0.0
push_z = 0.0
if ax > (hx - margin):
    pen = ax - (hx - margin)
    sign = 1.0
    if rel_pos[0] > 0.0:
        sign = -1.0
    push_x = sign * pen * stiffness
if ay > (hy - margin):
    pen = ay - (hy - margin)
    sign = 1.0
    if rel_pos[1] > 0.0:
        sign = -1.0
    push_y = sign * pen * stiffness
if az > (hz - margin):
    pen = az - (hz - margin)
    sign = 1.0
    if rel_pos[2] > 0.0:
        sign = -1.0
    push_z = sign * pen * stiffness
accel = wp.vec3(push_x, push_y, push_z)
particle_velocity[particle_index] = particle_velocity[particle_index] + accel * solver_time_step""",
                fields=(_VEL_RW,),
                parameter_keys=("controller", "half_extents", "margin", "stiffness"),
                description="Applies inward penalty acceleration at boundary box limits.",
            ),
        ),
    ),

    # 10. Voxel Viscosity Decay
    EffectDefinitionV2(
        effect_id="example.voxel_viscosity_decay",
        version="1.0.0",
        name="Voxel Viscosity Decay",
        summary="Thin out a thick fluid over time toward a base viscosity.",
        description="Each substep, blends the grid viscosity splatted from particles toward a base viscosity by exp(-decay rate x time), so a thick fluid thins out over the shot. Needs viscosity enabled with Viscosity From Particles.",
        category="Grid",
        author="FluidSim Examples",
        license="GPL-3.0-or-later",
        built_in=False,
        package_origin="example://custom-effects/1.0.251",
        tags=("viscosity", "grid", "relaxation"),
        cost_hint="MEDIUM",
        parameters=(
            _param_float("base_viscosity", "Base Viscosity", 0.001, 0.0, 10.0, "m²/s", "Target baseline kinematic viscosity."),
            _param_float("decay_rate", "Decay Rate", 0.4, 0.0, 10.0, "1/s", "Rate at which grid viscosity relaxes to base over simulation time."),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="voxel_viscosity_decay",
                label="Relax Grid Viscosity",
                domain=EffectDomain.ACTIVE_VOXELS,
                stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                cadence=EffectCadence.EVERY_SUBSTEP,
                # The grid is re-splatted from particles every substep, so the
                # decay is a function of time, not an accumulated per-step factor.
                source="""factor = wp.exp(-decay_rate * time)
viscosity[flat_voxel_index] = base_viscosity + (viscosity[flat_voxel_index] - base_viscosity) * factor""",
                fields=(_VISC_RW,),
                parameter_keys=("base_viscosity", "decay_rate"),
                description="Relaxes active grid voxel viscosity toward baseline value.",
            ),
        ),
    ),
)


def generate_all_example_packages(
    target_directory: str | Path | None = None,
    definitions: Sequence[EffectDefinitionV2] = EXAMPLE_EFFECT_DEFINITIONS,
) -> tuple[Path, ...]:
    """Compile and save all example Custom Effects as versioned .fseffect packages."""
    out_dir = Path(target_directory) if target_directory is not None else Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for definition in definitions:
        filename = f"{definition.effect_id.split('.')[-1]}-{definition.version}.fseffect"
        destination = out_dir / filename
        report = write_effect_package(destination, definition)
        generated.append(report.path)
    return tuple(generated)


if __name__ == "__main__":
    packages = generate_all_example_packages()
    print(f"Generated {len(packages)} example effect packages:")
    for pkg in packages:
        print(f"  - {pkg.name}")
