import numpy as np
import pytest

from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

warp = pytest.importorskip("warp")


def _settings(**overrides):
    if not warp.is_cuda_available():
        pytest.skip("CUDA not available")
    values = dict(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.25,
        leaf_capacity=32,
        max_particles=16,
        max_whitewater=1,
        surface_enabled=False,
        whitewater_enabled=False,
        device="cuda:0",
    )
    values.update(overrides)
    return SolverSettings(**values)


def test_mixing_blends_existing_particle_colors_in_one_neighborhood():
    solver = FluidSolver(_settings(
        species_enabled=True,
        species_mixing_rate=20.0,
    ))
    try:
        solver.inject_particles(
            [(0.50, 0.50, 0.50)],
            color=(1.0, 0.0, 0.0, 1.0),
        )
        solver.inject_particles(
            [(0.51, 0.50, 0.50)],
            color=(0.0, 0.0, 1.0, 1.0),
        )
        solver._rebuild_topology(expand=0)

        solver._mix_particle_colors(0.1)

        colors = solver.particles.attr("color").numpy()[:2]
        assert colors[0, 0] < 1.0
        assert colors[0, 2] > 0.0
        assert colors[1, 0] > 0.0
        assert colors[1, 2] < 1.0
        np.testing.assert_allclose(colors[:, 3], 1.0, atol=1.0e-5)
    finally:
        solver.release()


def test_reset_can_disable_and_reenable_mixing_without_stale_color_storage():
    solver = FluidSolver(_settings(species_enabled=True))
    try:
        assert solver.particles.has_optional_channel("color")
        assert solver.color_mixing is not None

        solver.settings.species_enabled = False
        solver.reset()
        assert not solver.particles.has_optional_channel("color")
        assert solver.color_mixing is None

        solver.settings.species_enabled = True
        solver.reset()
        assert solver.particles.has_optional_channel("color")
        assert solver.color_mixing is not None
    finally:
        solver.release()
