"""A real solver's checkpoint restores into a fresh solver, field for field."""

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

warp = pytest.importorskip("warp")


def _settings(device):
    return SolverSettings(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.125,
        leaf_capacity=64,
        max_particles=4096,
        max_whitewater=64,
        surface_enabled=False,
        whitewater_enabled=False,
        device=device,
    )


def _lattice(origin, spacing, counts):
    axes = [origin[i] + spacing * np.arange(counts[i]) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)


def _assert_same(expected, actual, path="state"):
    """Recursive exact comparison of a checkpoint state (arrays bit for bit)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(expected) == set(actual), (
            f"{path}: keys differ: {sorted(set(expected) ^ set(actual))}")
        for key in expected:
            _assert_same(expected[key], actual[key], f"{path}[{key!r}]")
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual, err_msg=path)
    else:
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def test_checkpoint_round_trips_into_a_fresh_solver(cuda_device):
    settings = _settings(cuda_device)
    source = FluidSolver(settings)
    restored = FluidSolver(_settings(cuda_device))
    try:
        points = _lattice((0.25, 0.25, 0.15), 0.0625, (6, 6, 6))
        source.inject_particles(points, velocity=(0.5, 0.0, 0.0))
        for _ in range(3):
            source.step_frame()

        saved = source.export_checkpoint_state()
        # Space-time FLIP is removed; its scalars must not come back.
        assert not any(key.startswith("st_") for key in saved["scalars"])
        assert saved["scalars"]["particle_count"] == len(points)
        assert saved["scalars"]["frame"] == 3
        assert saved["scalars"]["sim_time"] > 0.0

        restored.import_checkpoint_state(saved)
        again = restored.export_checkpoint_state()

        _assert_same(saved, again)
        assert restored.frame == source.frame
        assert restored.sim_time == source.sim_time
        np.testing.assert_array_equal(
            restored.particles.position.numpy()[:len(points)],
            source.particles.position.numpy()[:len(points)])
    finally:
        restored.release()
        source.release()
