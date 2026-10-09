"""APIC state contracts: checkpoint payload and memory estimate (3.0 lane F)."""

from __future__ import annotations

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.enums import ParticleSampleType
from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

pytest.importorskip("warp")

MAX_PARTICLES = 4096
# One float32 mat33 is 36 bytes; the particle pool is double buffered for
# compaction, so the honest per-particle cost is 72.
AFFINE_BYTES_PER_PARTICLE = 72


def _settings(device, sample_type):
    return SolverSettings(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.125,
        leaf_capacity=64,
        max_particles=MAX_PARTICLES,
        max_whitewater=64,
        surface_enabled=False,
        whitewater_enabled=False,
        sample_type=int(sample_type),
        device=device,
    )


def _lattice(origin, spacing, counts):
    axes = [origin[i] + spacing * np.arange(counts[i]) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)


def _spinning_state(device, sample_type):
    solver = FluidSolver(_settings(device, sample_type))
    points = _lattice((0.25, 0.25, 0.15), 0.0625, (6, 6, 6))
    rel = points - 0.5
    velocities = np.stack(
        [-rel[:, 1], rel[:, 0], np.zeros(len(points))], axis=1).astype(np.float32)
    solver.inject_particles(points, velocities=velocities)
    for _ in range(3):
        solver.step_frame()
    return solver, len(points)


def _assert_same(expected, actual, path="state"):
    if isinstance(expected, dict):
        assert set(expected) == set(actual), path
        for key in expected:
            _assert_same(expected[key], actual[key], f"{path}[{key!r}]")
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual, err_msg=path)
    else:
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def test_apic_checkpoint_carries_particle_affine_and_round_trips(cuda_device):
    source, count = _spinning_state(cuda_device, ParticleSampleType.APIC)
    restored = FluidSolver(_settings(cuda_device, ParticleSampleType.APIC))
    try:
        saved = source.export_checkpoint_state()
        affine = saved["arrays"]["particle_affine"]
        assert affine.dtype == np.float32
        assert affine.shape == (count, 9)
        assert np.any(affine != 0.0), "a spinning fluid must carry affine motion"

        restored.import_checkpoint_state(saved)
        _assert_same(saved, restored.export_checkpoint_state())
        np.testing.assert_array_equal(
            restored.particles.affine.numpy()[:count].reshape(count, 9), affine)
    finally:
        restored.release()
        source.release()


def test_flip_checkpoint_leaves_particle_affine_empty(cuda_device):
    source, _ = _spinning_state(cuda_device, ParticleSampleType.FLIP)
    try:
        saved = source.export_checkpoint_state()
        affine = saved["arrays"].get("particle_affine")
        assert affine is None or np.asarray(affine).shape == (0, 9)
    finally:
        source.release()


def test_flip_solver_imports_an_apic_checkpoint_and_drops_the_affine(cuda_device):
    """Cross-mode import never raises; a FLIP solver has no use for the affine."""
    source, count = _spinning_state(cuda_device, ParticleSampleType.APIC)
    target = FluidSolver(_settings(cuda_device, ParticleSampleType.FLIP))
    try:
        saved = source.export_checkpoint_state()
        target.import_checkpoint_state(saved)

        assert not target.particles.affine_allocated
        again = target.export_checkpoint_state()
        assert np.asarray(again["arrays"]["particle_affine"]).shape == (0, 9)
        np.testing.assert_array_equal(
            again["arrays"]["particle_position"],
            saved["arrays"]["particle_position"])
        np.testing.assert_array_equal(
            again["arrays"]["particle_velocity"],
            saved["arrays"]["particle_velocity"])
        target.step_frame()
        assert np.all(np.isfinite(target.particles.velocity.numpy()[:count]))
    finally:
        target.release()
        source.release()


def test_apic_solver_imports_a_flip_checkpoint_with_zero_affine(cuda_device):
    """A FLIP checkpoint has no affine: an APIC solver starts it at zero."""
    source, count = _spinning_state(cuda_device, ParticleSampleType.FLIP)
    target = FluidSolver(_settings(cuda_device, ParticleSampleType.APIC))
    try:
        target.import_checkpoint_state(source.export_checkpoint_state())

        assert target.particles.affine_allocated
        assert not np.any(target.particles.affine.numpy())
        target.step_frame()
        assert np.all(np.isfinite(target.particles.velocity.numpy()[:count]))
        assert np.all(np.isfinite(target.particles.affine.numpy()[:count]))
    finally:
        target.release()
        source.release()


def test_memory_estimate_charges_the_affine_channel():
    flip = estimate_solver_bytes(_settings("cuda:0", ParticleSampleType.FLIP))
    pic = estimate_solver_bytes(_settings("cuda:0", ParticleSampleType.PIC))
    apic = estimate_solver_bytes(_settings("cuda:0", ParticleSampleType.APIC))
    assert pic == flip
    assert apic - flip >= AFFINE_BYTES_PER_PARTICLE * MAX_PARTICLES


def test_sample_type_values_are_stable_kernel_arguments():
    assert [(m.name, int(m)) for m in ParticleSampleType] == [
        ("PIC", 0), ("FLIP", 1), ("APIC", 2)]


def test_growth_plan_charges_the_extra_particle_bytes():
    from fluidSim_warp_blender51.sim.particle_capacity import plan_particle_growth

    kwargs = dict(gpu_total_bytes=10 << 30, gpu_free_bytes=10 << 30)
    plain = plan_particle_growth(1000, 5000, **kwargs)
    apic = plan_particle_growth(
        1000, 5000, extra_particle_bytes=AFFINE_BYTES_PER_PARTICLE, **kwargs)
    assert apic.peak_allocation_bytes - plain.peak_allocation_bytes == (
        AFFINE_BYTES_PER_PARTICLE * apic.target)
