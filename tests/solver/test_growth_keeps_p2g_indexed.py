"""Particle-pool growth inside the source phase must not starve P2G.

Regression for review item B01: a successful growth swaps in fresh, empty
ParticleBins and defers the emission (``_source`` returns False). step_frame
re-indexed only when something was emitted, so P2G gathered through zeroed
bin indices (probe: grid weight 3000 -> 106). Growth also dropped the
compiled source batch, so compiled field/pressure stages skipped the substep.
"""
from __future__ import annotations

import numpy as np
import pytest


def _first_substep_grid_mass(cuda_device, grow):
    wp = pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(-0.4, -0.4, 0.0), domain_max=(0.4, 0.4, 0.8),
        voxel_size=0.05, leaf_capacity=64, max_particles=4096,
        max_whitewater=1024, max_substeps=1, device=cuda_device)
    solver = FluidSolver(settings)
    try:
        rng = np.random.default_rng(1)
        solver.inject_particles(rng.uniform(
            (-0.2, -0.2, 0.1), (0.2, 0.2, 0.4), (3000, 3)).astype(np.float32),
            velocity=(0.5, 0.0, 0.0))
        original_source, original_p2g = solver._source, solver._p2g
        masses = []

        def source(dt):
            if grow and not masses:
                result = solver.ensure_particle_capacity(solver.particles.capacity + 1)
                assert result.grew, result
                return False  # emission deferred after growth
            return original_source(dt)

        def p2g():
            original_p2g()
            wp.synchronize_device(cuda_device)
            masses.append(float(solver.vel.wu.numpy()[:solver._nvox].sum()))

        solver._source, solver._p2g = source, p2g
        solver.step_frame()
        return masses[0]
    finally:
        solver.release()


def test_growth_during_sourcing_keeps_the_grid_transfer_complete(cuda_device):
    reference = _first_substep_grid_mass(cuda_device, grow=False)
    grown = _first_substep_grid_mass(cuda_device, grow=True)
    assert reference > 0.0
    assert grown == pytest.approx(reference, rel=1e-5)


def test_growth_keeps_the_compiled_source_batch():
    from dataclasses import dataclass
    from types import SimpleNamespace

    from fluidSim_warp_blender51.sim.sourcing import SourcePipeline

    @dataclass(frozen=True)
    class _Runtime:
        particles: object

    # The compiled batch holds region data only (no particle storage), so a
    # pool swap must not drop it mid-substep.
    pipeline = SourcePipeline.__new__(SourcePipeline)
    batch = object()
    pipeline._compiled_batch = batch
    pipeline.runtime = _Runtime(particles=None)
    pipeline.update_available_capacity = lambda available: None
    grown = SimpleNamespace(capacity=8, host_count=1)
    pipeline.replace_particle_storage(grown)

    assert pipeline.runtime.particles is grown
    assert pipeline._compiled_batch is batch
