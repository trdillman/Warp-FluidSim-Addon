"""Regression tests for slowdown diagnostics and reset cache behavior."""

import inspect
from types import SimpleNamespace

from fluidSim_warp_blender51.diagnostics.profiler import PROFILE_FRAME_FIELDS
from fluidSim_warp_blender51.sim import constants as C
from fluidSim_warp_blender51.sim.kernels_fluid import k_particles_relax
from fluidSim_warp_blender51.sim.particles import ParticleSystem
from fluidSim_warp_blender51.sim.solver import FluidSolver, FrameStats
from fluidSim_warp_blender51.sim.surface import SurfaceExtractor
from fluidSim_warp_blender51.sim.whitewater import WhitewaterSystem


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
    extractor = Extractor()
    solver.settings = SimpleNamespace(spacetime_enabled=False, spacetime_seed=1729)
    calibration_calls = []
    solver._calibrate_st_reference_mass = lambda: calibration_calls.append("calibrated")
    solver.surface = SimpleNamespace(reset=lambda: extractor.reset_buffers())
    solver.frame = 99
    solver.sim_time = 1.0

    FluidSolver.reset(solver)

    assert extractor.reset_calls == 1
    assert calibration_calls == []
    assert solver._nvox == 42
    assert solver.grid.calls == ["begin", "finalize"]


def test_rebuild_surface_owns_cache_invalidation_and_current_sdf_rebuild():
    solver = object.__new__(FluidSolver)
    calls = []
    solver.surface = SimpleNamespace(rebuild=lambda: calls.append("surface"))

    FluidSolver.rebuild_surface(solver)

    assert calls == ["surface"]


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


def test_compaction_accepts_active_count_hint():
    """Solver compaction calls must keep accepting active-work bounds."""
    assert "count_hint" in inspect.signature(ParticleSystem.compact).parameters
    assert "count_hint" in inspect.signature(WhitewaterSystem.compact).parameters
