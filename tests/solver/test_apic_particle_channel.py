"""APIC affine channel bookkeeping: compaction, emission, live mode switches."""

from __future__ import annotations

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.enums import ParticleSampleType
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

pytest.importorskip("warp")


def _settings(device, sample_type, max_particles=4096):
    return SolverSettings(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.125,
        leaf_capacity=64,
        max_particles=max_particles,
        max_whitewater=64,
        surface_enabled=False,
        whitewater_enabled=False,
        export_attributes=True,
        particles_attribute_uid=True,
        sample_type=int(sample_type),
        device=device,
    )


def _lattice(counts, origin=(0.25, 0.25, 0.15), spacing=0.0625):
    axes = [origin[i] + spacing * np.arange(counts[i]) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)


def _spin(points):
    rel = points - 0.5
    return np.stack(
        [-rel[:, 1], rel[:, 0], np.zeros(len(points))], axis=1).astype(np.float32)


def _spinning_apic(device, **kwargs):
    solver = FluidSolver(_settings(device, ParticleSampleType.APIC, **kwargs))
    points = _lattice((6, 6, 6))
    solver.inject_particles(points, velocities=_spin(points))
    for _ in range(3):
        solver.step_frame()
    return solver


def _affine_by_uid(solver):
    p = solver.particles
    n = p.refresh_host_count()
    uid = p.int_attrs["uid"].numpy()[:n]
    affine = p.affine.numpy()[:n].reshape(n, 9)
    assert len(np.unique(uid)) == n
    return {int(u): affine[k].copy() for k, u in enumerate(uid)}


def test_every_compaction_path_carries_affine_with_its_particle(cuda_device):
    solver = _spinning_apic(cuda_device)
    try:
        p = solver.particles
        before = _affine_by_uid(solver)
        assert any(np.any(row != 0.0) for row in before.values())

        n = p.refresh_host_count()
        alive = p.alive.numpy().copy()
        alive[:n:3] = 0
        p.alive.assign(alive)
        killed = int((alive[:n] == 0).sum())
        assert killed > 0

        p.compact()  # no hint: full-capacity path
        after = _affine_by_uid(solver)
        assert len(after) == len(before) - killed
        for uid, row in after.items():
            np.testing.assert_array_equal(row, before[uid], err_msg=f"uid {uid}")

        # Second path: compact with a count hint after killing more.
        n = p.refresh_host_count()
        alive = p.alive.numpy().copy()
        alive[1:n:4] = 0
        p.alive.assign(alive)
        p.compact(count_hint=n)
        again = _affine_by_uid(solver)
        for uid, row in again.items():
            np.testing.assert_array_equal(row, before[uid], err_msg=f"uid {uid}")
        assert len(again) < len(after)
    finally:
        solver.release()


def test_emitted_particles_start_with_zero_affine_even_after_compaction(
        cuda_device):
    solver = _spinning_apic(cuda_device)
    try:
        p = solver.particles
        n = p.refresh_host_count()
        alive = p.alive.numpy().copy()
        alive[:n:2] = 0
        p.alive.assign(alive)
        p.compact()
        live = p.refresh_host_count()

        fresh = _lattice((3, 3, 3), origin=(0.6, 0.6, 0.6))
        solver.inject_particles(fresh)
        total = p.refresh_host_count()
        assert total == live + len(fresh)
        assert not np.any(p.affine.numpy()[live:total])
        # Everything beyond the count is zero too (the emission invariant).
        assert not np.any(p.affine.numpy()[total:])
    finally:
        solver.release()


def test_emitted_particles_start_with_zero_affine_after_growth(cuda_device):
    solver = _spinning_apic(cuda_device, max_particles=512)
    try:
        before = _affine_by_uid(solver)
        old_capacity = solver.particles.capacity
        result = solver.ensure_particle_capacity(old_capacity * 4)
        assert result.status == "GROWN", result
        p = solver.particles
        assert p.capacity > old_capacity and p.affine_allocated

        after = _affine_by_uid(solver)
        for uid, row in before.items():
            np.testing.assert_array_equal(after[uid], row)

        live = p.refresh_host_count()
        solver.inject_particles(_lattice((3, 3, 3), origin=(0.6, 0.6, 0.6)))
        total = p.refresh_host_count()
        assert total == live + 27
        assert not np.any(p.affine.numpy()[live:])
    finally:
        solver.release()


def test_transfer_mode_switches_live_flip_apic_flip(cuda_device):
    solver = FluidSolver(_settings(cuda_device, ParticleSampleType.FLIP))
    try:
        points = _lattice((6, 6, 6))
        solver.inject_particles(points, velocities=_spin(points))
        for mode, expect_channel in (
                (ParticleSampleType.FLIP, False),
                (ParticleSampleType.APIC, True),
                (ParticleSampleType.FLIP, False),
                (ParticleSampleType.APIC, True)):
            solver.settings.sample_type = int(mode)
            for _ in range(2):
                solver.step_frame()
            p = solver.particles
            n = p.refresh_host_count()
            assert n == len(points)
            assert p.affine_allocated is expect_channel
            assert np.all(np.isfinite(p.position.numpy()[:n]))
            assert np.all(np.isfinite(p.velocity.numpy()[:n]))
            if expect_channel:
                assert np.all(np.isfinite(p.affine.numpy()[:n]))
    finally:
        solver.release()
