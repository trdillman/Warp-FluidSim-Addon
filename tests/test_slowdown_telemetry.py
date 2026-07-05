"""Regression tests for slowdown diagnostics and reset cache behavior."""

import inspect

from fluidSim_warp_blender51.diagnostics.profiler import PROFILE_FRAME_FIELDS
from fluidSim_warp_blender51.sim import constants as C
from fluidSim_warp_blender51.sim.kernels_fluid import k_particles_relax
from fluidSim_warp_blender51.sim.solver import FluidSolver, FrameStats
from fluidSim_warp_blender51.sim.surface import SurfaceExtractor


def test_surface_extractor_reset_buffers_releases_dense_cache():
    """Timeline reset must not keep the largest-ever marching-cubes buffers."""
    extractor = object.__new__(SurfaceExtractor)
    extractor._mc = object()
    extractor._dense = object()
    extractor._dense_dim = (128, 96, 64)

    SurfaceExtractor.reset_buffers(extractor)

    assert extractor._mc is None
    assert extractor._dense is None
    assert extractor._dense_dim is None


def test_solver_reset_clears_surface_extractor_buffers():
    """Playback rewind resets sticky surface extraction allocation buckets."""

    class Clearable:
        def __init__(self):
            self.calls = 0

        def clear(self):
            self.calls += 1

        def zero(self):
            self.calls += 1

    class Grid:
        active_voxel_count = 42

        def __init__(self):
            self.calls = []

        def begin_topology(self):
            self.calls.append("begin")

        def finalize_topology(self):
            self.calls.append("finalize")

    class Extractor:
        def __init__(self):
            self.reset_calls = 0

        def reset_buffers(self):
            self.reset_calls += 1

    solver = object.__new__(FluidSolver)
    solver.particles = Clearable()
    solver.whitewater = Clearable()
    solver.churn = Clearable()
    solver.grid = Grid()
    solver.surface_extractor = Extractor()
    solver._surface_mesh_cache = object()
    solver._surface_mesh_key = (99, 1.0)
    solver.frame = 99
    solver.sim_time = 1.0

    FluidSolver.reset(solver)

    assert solver.surface_extractor.reset_calls == 1
    assert solver._surface_mesh_cache is None
    assert solver._surface_mesh_key is None
    assert solver._nvox == 42
    assert solver.grid.calls == ["begin", "finalize"]


def test_frame_stats_exports_slowdown_diagnostics():
    """Frame JSON includes counters needed to explain rising frame cost."""
    stats = FrameStats()
    stats.whitewater_particles = 17
    stats.substeps = 3
    stats.pressure_iterations = 11
    stats.pressure_residual = 0.125

    data = stats.as_dict()

    assert data["whitewaterParticles"] == 17
    assert data["substeps"] == 3
    assert data["pressureIterations"] == 11
    assert data["pressureResidual"] == 0.125
    for field in (
        "whitewaterParticles",
        "substeps",
        "pressureIterations",
        "pressureResidual",
    ):
        assert field in PROFILE_FRAME_FIELDS


def test_relax_kernel_caps_per_bin_particle_scan():
    """Particle relax must not scan every particle in overcrowded bins."""
    source = inspect.getsource(k_particles_relax)

    assert C.DEFAULT_RELAX_BIN_SAMPLE_LIMIT == 32
    assert "DEFAULT_RELAX_BIN_SAMPLE_LIMIT" in source
    assert "range(limit)" in source
