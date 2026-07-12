"""Small contracts preventing the measured live-playback hot paths returning."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim.particles import ParticleSystem
from fluidSim_warp_blender51.sim.pressure import PressureSolver
from fluidSim_warp_blender51.sim.solver import FluidSolver, SourceShapeDesc
from fluidSim_warp_blender51.sim.sourcing import (
    SourceShape,
    k_source_shapes_particles,
)


def test_particle_emission_saturates_device_and_host_counts():
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    device = "cuda:0"
    particles = ParticleSystem(capacity=64, device=device)
    wp.copy(particles.count, wp.array([60], dtype=wp.int32, device=device))
    shape = SourceShapeDesc(
        enabled=True, shape=2, stype=0, mode=1,
        position=(0.4, 0.4, 0.4), half_extents=(1.0, 1.0, 1.0),
        emit_rate=100.0,
    ).to_struct()
    shapes = wp.array([shape], dtype=SourceShape, device=device)
    leaf_coords = wp.array([(0, 0, 0)], dtype=wp.vec3i, device=device)
    num_leaves = wp.array([1], dtype=wp.int32, device=device)
    voxel_counts = wp.zeros(512, dtype=wp.int32, device=device)

    wp.launch(
        k_source_shapes_particles, dim=512,
        inputs=[shapes, 1, leaf_coords, num_leaves, wp.vec3(0.0), 0.1,
                voxel_counts, 1, particles.capacity],
        outputs=[particles.position, particles.velocity,
                 particles.attr("age"), particles.attr("life"),
                 particles.attr("temperature"), particles.attr("viscosity"),
                 particles.attr("density"), particles.time_offset,
                 particles.alive, particles.count],
        device=device,
    )

    assert particles.refresh_host_count() == particles.capacity
    assert int(particles.count.numpy()[0]) == particles.capacity
    assert np.isfinite(particles.position.numpy()[:particles.capacity]).all()


def test_pcg_reuses_scalar_buffers_without_device_copy():
    source = inspect.getsource(PressureSolver.solve_pcg)
    batch_source = inspect.getsource(PressureSolver._launch_pcg_batch)
    assert "wp.copy(self.rz, self.rz_new)" not in source
    assert "wp.copy(self.rz, self.rz_new)" not in batch_source
    assert "self.rz, self.rz_new = self.rz_new, self.rz" in batch_source


def test_full_particle_allocation_skips_emission_and_reindex_signal(monkeypatch):
    launches = []
    monkeypatch.setattr(wp, "launch", lambda kernel, **kwargs: launches.append(kernel))
    particles = SimpleNamespace(host_count=64, capacity=64)
    solver = SimpleNamespace(
        grid=SimpleNamespace(leaf_coords=None, num_leaves=None, origin=wp.vec3(0.0)),
        settings=SimpleNamespace(voxel_size=0.1),
        sources=[SimpleNamespace(enabled=True, stype=0, mode=1)],
        height_source=SimpleNamespace(enabled=False, emit_particles=False),
        particles=particles,
        voxel_type=SimpleNamespace(data=None),
        collision_sdf=SimpleNamespace(data=None),
        collision_vel=None,
        vel=SimpleNamespace(u=None, v=None, w=None),
        temperature=SimpleNamespace(data=None),
        viscosity_field=SimpleNamespace(data=None),
        device=None,
        _nvox=1,
        _shapes=lambda: None,
        _emit_seed=1,
    )

    emitted = FluidSolver._source(solver, 1.0 / 24.0)

    assert emitted is False
    assert len(launches) == 2  # source field + source type resolve only
