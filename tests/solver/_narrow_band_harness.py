"""Shared setup for the narrow-band acceptance tests (lane J).

A closed tank, 16 x 16 x 32 cells at h = 1/32, filled to 75 percent with a
2 x 2 x 2 particle lattice per voxel.  Small enough to run in well under a
minute on the test GPU, deep enough (24 cells) that the interior is many band
widths from the surface.

The solver-side API the tests rely on is the stop-2 surface described in
docs/specs/3.0/SPEC-narrow-band-particles.md ("Implementation notes"):

* ``solver.narrow_band``          -- state object, present only when enabled
* ``solver.narrow_band.phi_numpy()`` -- (leaf-flat) persistent liquid phi
* ``solver.narrow_band.reseed_overflow`` -- bool, set when a reseed was clipped
* ``solver.stats.liquid_voxels``  -- voxels inside the liquid, not particles
"""

from __future__ import annotations

import numpy as np

from fluidSim_warp_blender51.sim import constants as C
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

H = 1.0 / 32.0
DOMAIN_MAX = (0.5, 0.5, 1.0)
FILL_HEIGHT = 0.75
BAND_VOXELS = 3.0
# Frames chosen for the column test: the 16x16x32 tank settles to a visually
# quiet column in about 30 frames; the budget is roughly 60 s of wall clock.
SETTLE_FRAMES = 30
MAX_PARTICLES = 120_000


def settings(device, *, narrow_band, max_particles=MAX_PARTICLES, **extra):
    return SolverSettings(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=DOMAIN_MAX,
        voxel_size=H,
        leaf_capacity=256,
        max_particles=max_particles,
        surface_enabled=False,
        whitewater_enabled=False,
        narrow_band_enabled=narrow_band,
        narrow_band_voxels=BAND_VOXELS,
        device=device,
        **extra,
    )


def lattice(fill_height=FILL_HEIGHT, per_axis=2):
    spacing = H / per_axis
    counts = (
        int(DOMAIN_MAX[0] / spacing), int(DOMAIN_MAX[1] / spacing),
        int(fill_height / spacing))
    axes = [(np.arange(n) + 0.5) * spacing for n in counts]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)


def run_tank(device, *, narrow_band, frames=SETTLE_FRAMES, fill_height=FILL_HEIGHT,
             **extra):
    """Build, fill and step a tank; the caller owns (and must release) it."""
    solver = FluidSolver(settings(device, narrow_band=narrow_band, **extra))
    solver.inject_particles(lattice(fill_height))
    for _ in range(frames):
        solver.step_frame()
    return solver


def surface_height(solver):
    """Free-surface height from the particles: 99th percentile of z."""
    return float(np.percentile(solver.particle_positions()[:, 2], 99.0))


def pressure_profile(solver):
    """Mean pressure per z-layer over liquid-bearing cells -> (z_centres, p)."""
    grid = solver.grid
    count = int(grid.refresh_topology_counts())
    coords = grid.leaf_coords.numpy()[:count]
    pressure = solver.pressure_solver.pressure.numpy()[:count * C.LEAF_VOLUME]
    local = np.arange(C.LEAF_VOLUME)
    k = (local >> (2 * C.LEAF_SHIFT)) & C.LEAF_MASK
    z_index = (coords[:, 2:3] << C.LEAF_SHIFT) + k[None, :]
    z_index = z_index.ravel()
    p = pressure[:z_index.size]
    layers = np.unique(z_index)
    means = np.array([p[(z_index == layer) & (p != 0.0)].mean()
                      if np.any((z_index == layer) & (p != 0.0)) else 0.0
                      for layer in layers])
    return (layers + 0.5) * H, means
