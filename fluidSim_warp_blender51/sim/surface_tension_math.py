"""Pure physical helpers for FluidSim surface tension.

This module deliberately has no Warp or Blender dependency.  Release/static
validation can therefore verify the capillary stability and droplet trend
oracles before any runtime package is available.
"""

from __future__ import annotations

import math

_CAPILLARY_DENOMINATOR = 4.0 * math.pi


def capillary_dt(
    density: float,
    voxel_size: float,
    coefficient: float,
    safety: float,
) -> float:
    """Return the explicit capillary time-step limit in seconds.

    ``coefficient == 0`` disables the constraint and returns infinity.  For a
    positive coefficient, all physical inputs must be finite and positive.
    The formula is ``safety * sqrt(rho * h**3 / (4*pi*sigma))``.
    """

    values = {
        "density": float(density),
        "voxel_size": float(voxel_size),
        "coefficient": float(coefficient),
        "safety": float(safety),
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    coefficient_value = values["coefficient"]
    if coefficient_value < 0.0:
        raise ValueError("coefficient must be non-negative")
    if coefficient_value == 0.0:
        return float("inf")
    if values["density"] <= 0.0:
        raise ValueError("density must be positive")
    if values["voxel_size"] <= 0.0:
        raise ValueError("voxel_size must be positive")
    if not 0.0 < values["safety"] <= 1.0:
        raise ValueError("safety must be in (0, 1]")
    return values["safety"] * math.sqrt(
        values["density"] * values["voxel_size"] ** 3
        / (_CAPILLARY_DENOMINATOR * coefficient_value)
    )


def required_capillary_substeps(frame_dt: float, stable_dt: float) -> int:
    """Return the minimum integer substeps needed to satisfy ``stable_dt``."""

    frame_dt = float(frame_dt)
    stable_dt = float(stable_dt)
    if not math.isfinite(frame_dt) or frame_dt < 0.0:
        raise ValueError("frame_dt must be finite and non-negative")
    if frame_dt == 0.0:
        return 1
    if math.isinf(stable_dt) and stable_dt > 0.0:
        return 1
    if not math.isfinite(stable_dt) or stable_dt <= 0.0:
        raise ValueError("stable_dt must be finite and positive, or +inf")
    return max(1, int(math.ceil(frame_dt / stable_dt)))


def normal_support_voxels(
    force_band_voxels: float, smoothing_iterations: int
) -> float:
    """Return SDF-normal support required around the force band.

    Curvature samples one neighbouring normal beyond the force band, and each
    normal-smoothing pass expands the required stencil by one voxel.
    """

    force_band_voxels = float(force_band_voxels)
    if not math.isfinite(force_band_voxels) or force_band_voxels <= 0.0:
        raise ValueError("force_band_voxels must be finite and positive")
    if isinstance(smoothing_iterations, bool):
        raise ValueError("smoothing_iterations must be a non-negative integer")
    smoothing_value = int(smoothing_iterations)
    if smoothing_value != smoothing_iterations or smoothing_value < 0:
        raise ValueError("smoothing_iterations must be a non-negative integer")
    return max(force_band_voxels, 0.5) + smoothing_value + 1.0


def required_sdf_splat_radius_voxels(
    particle_radius_voxels: float,
    force_band_voxels: float,
    smoothing_iterations: int,
    sdf_filter_iterations: int = 0,
) -> int:
    """Return a conservative particle-SDF splat radius for capillarity.

    The particle union-of-spheres SDF must cover the force band plus every
    normal/curvature smoothing stencil.  One additional voxel accounts for a
    particle's sub-voxel location relative to the cell centre.
    """

    particle_radius_voxels = float(particle_radius_voxels)
    if not math.isfinite(particle_radius_voxels) or particle_radius_voxels < 0.0:
        raise ValueError("particle_radius_voxels must be finite and non-negative")
    if isinstance(sdf_filter_iterations, bool):
        raise ValueError("sdf_filter_iterations must be a non-negative integer")
    filter_value = int(sdf_filter_iterations)
    if filter_value != sdf_filter_iterations or filter_value < 0:
        raise ValueError("sdf_filter_iterations must be a non-negative integer")
    support = (
        normal_support_voxels(force_band_voxels, smoothing_iterations)
        + filter_value
    )
    return max(1, int(math.ceil(particle_radius_voxels + support + 1.0)))


def required_topology_expand_leaves(
    sdf_splat_radius_voxels: int, leaf_dim: int
) -> int:
    """Return sparse-leaf halo iterations needed for an SDF splat radius."""

    if isinstance(sdf_splat_radius_voxels, bool) or isinstance(leaf_dim, bool):
        raise ValueError("splat radius and leaf_dim must be positive integers")
    radius = int(sdf_splat_radius_voxels)
    dimension = int(leaf_dim)
    if radius != sdf_splat_radius_voxels or radius < 0:
        raise ValueError("sdf_splat_radius_voxels must be a non-negative integer")
    if dimension != leaf_dim or dimension <= 0:
        raise ValueError("leaf_dim must be a positive integer")
    return int(math.ceil(radius / dimension)) if radius else 0


def compact_cosine_delta(phi: float, half_width: float) -> float:
    """Host reference for the compact interface delta used by the Warp kernel."""

    phi = float(phi)
    half_width = float(half_width)
    if not math.isfinite(phi) or not math.isfinite(half_width):
        raise ValueError("phi and half_width must be finite")
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    if abs(phi) >= half_width:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * phi / half_width)) / half_width


def rayleigh_lamb_frequency(
    density: float,
    radius: float,
    coefficient: float,
    mode: int = 2,
) -> float:
    """Return the inviscid droplet angular frequency for a shape mode.

    This host-only utility supplies a physical trend oracle for regression
    tests.  It is not used to advance the numerical solver.
    """

    density = float(density)
    radius = float(radius)
    coefficient = float(coefficient)
    if isinstance(mode, bool):
        raise ValueError("droplet mode must be an integer of at least 2")
    mode_value = int(mode)
    if mode_value != mode:
        raise ValueError("droplet mode must be an integer of at least 2")
    mode = mode_value
    if not all(math.isfinite(value) for value in (density, radius, coefficient)):
        raise ValueError("droplet inputs must be finite")
    if density <= 0.0 or radius <= 0.0 or coefficient < 0.0:
        raise ValueError("density/radius must be positive and coefficient non-negative")
    if mode < 2:
        raise ValueError("droplet mode must be at least 2")
    return math.sqrt(
        mode * (mode - 1) * (mode + 2) * coefficient
        / (density * radius**3)
    )
