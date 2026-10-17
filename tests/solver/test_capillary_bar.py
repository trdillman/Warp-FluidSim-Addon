"""A zero-g 2 mm water bar keeps its volume under surface tension.

Two faults made fine-grid surface tension collapse liquid:

* The pressure residual is in right-hand-side units (-divergence * h^2), so a
  fixed tolerance allowed 25x the divergence at 2 mm that it allows at 1 cm.
  The initial residual already passed, the solve ran 0-1 iterations and the
  bar compressed to a few voxels. ``effective_pressure_tolerance`` scales the
  tolerance by (h / 1 cm)^2 below 1 cm.
* The particle SDF is a union of spheres, so inside the liquid it hovers near
  zero and every gap between particles reads as a bubble. Surface tension now
  reads ``_capillary_phi``, whose interior is a distance from the free surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.pressure import effective_pressure_tolerance
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings
from fluidSim_warp_blender51.sim.surface_tension_math import normal_support_voxels

pytest.importorskip("warp")

H = 0.002
DOMAIN_MAX = (0.1, 0.1, 0.1)
BAR_MIN = np.array((0.02, 0.04, 0.04))
BAR_SIZE = np.array((0.06, 0.02, 0.02))
FRAMES = 8


def test_tolerance_is_unchanged_at_and_above_one_centimetre():
    for h in (0.01, 0.02, 0.045, 1.0):
        assert effective_pressure_tolerance(1e-5, h) == 1e-5


def test_tolerance_scales_with_voxel_area_below_one_centimetre():
    assert effective_pressure_tolerance(1e-5, 0.002) == pytest.approx(4e-7)
    assert effective_pressure_tolerance(1e-5, 0.005) == pytest.approx(2.5e-6)
    assert effective_pressure_tolerance(1e-5, 0.0) == 1e-5


def _bar_points():
    spacing = H / 2.0
    counts = np.round(BAR_SIZE / spacing).astype(int)
    axes = [BAR_MIN[i] + (np.arange(counts[i]) + 0.5) * spacing for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)


def _solver(device):
    points = _bar_points()
    solver = FluidSolver(SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=DOMAIN_MAX, voxel_size=H,
        leaf_capacity=512, max_particles=len(points) + 4096,
        gravity=(0.0, 0.0, 0.0), max_substeps=48,
        surface_tension_enabled=True, surface_tension_coefficient=0.072,
        surface_tension_density=1000.0,
        surface_enabled=False, whitewater_enabled=False, device=device))
    solver.inject_particles(points)
    return solver


def _occupied_voxels(positions):
    return len(np.unique(np.floor(positions / H).astype(np.int64), axis=0))


def test_zero_g_bar_keeps_its_volume(cuda_device):
    solver = _solver(cuda_device)
    try:
        start = _occupied_voxels(solver.particle_positions())
        for _ in range(FRAMES):
            solver.step_frame()
        end = _occupied_voxels(solver.particle_positions())
    finally:
        solver.release()
    assert end >= 0.85 * start, (start, end)


def test_capillary_distance_is_deep_inside_the_liquid(cuda_device):
    solver = _solver(cuda_device)
    try:
        solver.step_frame()
        assert solver.surface.ensure_solver_sdf_current()
        st = solver.surface_tension
        st.ensure_resources()
        settings = solver.settings
        band = normal_support_voxels(
            float(settings.surface_tension_band_voxels),
            int(settings.surface_tension_smoothing_iterations)) * H
        phi = st._capillary_phi(band).numpy()
        n = int(solver.grid.num_leaves.numpy()[0]) * 512
        fluid = (solver.voxel_type.data.numpy()[:n] & 4) != 0
        raw = solver.surface_sdf.data.numpy()[:n][fluid]
        rebuilt = phi[:n][fluid]
    finally:
        solver.release()
    # The bar is ten voxels thick: most liquid cells lie a voxel or more from
    # the surface. The raw union-of-spheres SDF reads about -0.3 h there.
    assert np.median(raw) > -0.5 * H
    assert np.median(rebuilt) <= -1.0 * H, np.quantile(rebuilt / H, [0.1, 0.5, 0.9])
