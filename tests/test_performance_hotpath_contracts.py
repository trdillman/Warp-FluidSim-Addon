"""Small contracts preventing the measured live-playback hot paths returning."""

from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim.particles import ParticleSystem
from fluidSim_warp_blender51.sim.constants import VOXEL_COLLISION, VOXEL_FLUID
from fluidSim_warp_blender51.sim.fields import SparseGrid
from fluidSim_warp_blender51.sim.pressure import PressureSolver
from fluidSim_warp_blender51.sim.solver import (
    FluidSolver,
    SolverSettings,
    SourceShapeDesc,
    _motion_leaf_radius,
)
from fluidSim_warp_blender51.sim.sourcing import (
    SourcePipeline,
    SourceShape,
    k_source_shapes_particles,
)
from fluidSim_warp_blender51.sim.whitewater import (
    WhitewaterSystem,
    k_emit_whitewater,
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
        particles=particles,
        voxel_type=SimpleNamespace(data=None),
        collision_sdf=SimpleNamespace(data=None),
        collision_vel=None,
        vel=SimpleNamespace(u=None, v=None, w=None),
        temperature=SimpleNamespace(data=None),
        viscosity_field=SimpleNamespace(data=None),
        device=None,
        _nvox=1,
    )
    pipeline = SimpleNamespace(
        owner=solver,
        device=None,
        sources=[SimpleNamespace(enabled=True, stype=0, mode=1)],
        height_source=SimpleNamespace(enabled=False, emit_particles=False),
        _shapes=lambda: None,
        _emit_seed=1,
    )

    emitted = SourcePipeline.apply(pipeline, 1.0 / 24.0)

    assert emitted is False
    assert len(launches) == 2  # source field + source type resolve only


def test_live_surface_publish_cadence_defaults_to_four_frames():
    assert SolverSettings().live_surface_publish_cadence == 4


def test_viewport_replaces_full_mesh_repair_with_index_validation():
    from fluidSim_warp_blender51.blender import viewport

    source = inspect.getsource(viewport._swap_mesh)
    assert "mesh.validate()" not in source
    assert "surface triangle index is outside the vertex range" in source


def test_skipped_surface_publish_preserves_existing_mesh(monkeypatch):
    from fluidSim_warp_blender51 import blender
    from fluidSim_warp_blender51.blender import viewport

    fake_materials = types.ModuleType("fluidSim_warp_blender51.blender.materials")
    fake_materials.ensure_surface_material = lambda *_args: None
    fake_materials.ensure_particle_material = lambda *_args: None
    fake_materials.ensure_whitewater_material = lambda *_args: None
    monkeypatch.setitem(sys.modules, fake_materials.__name__, fake_materials)
    monkeypatch.setattr(blender, "materials", fake_materials, raising=False)

    existing = SimpleNamespace(type="MESH", data=SimpleNamespace(vertices=[1]))
    monkeypatch.setattr(
        viewport.bpy, "data",
        SimpleNamespace(objects=SimpleNamespace(get=lambda _name: existing)),
        raising=False)
    swaps = []
    monkeypatch.setattr(viewport, "_swap_mesh", lambda *args: swaps.append(args))
    monkeypatch.setattr(viewport, "_clear_points", lambda _name: None)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        show_surface=True, surface_enabled=True, surface_attribute_mode="NONE",
        show_particles=False, show_whitewater=False, whitewater_enabled=False))
    frame = SimpleNamespace(publish_surface=False)

    viewport._render_outputs(scene, frame)

    assert swaps == []


def test_whitewater_compaction_is_frame_batched_without_substep_readback():
    substep_source = inspect.getsource(FluidSolver._whitewater)
    finalize_source = inspect.getsource(FluidSolver._finalize_whitewater)
    frame_source = inspect.getsource(FluidSolver.step_frame)

    assert "synchronize_device" not in substep_source
    assert "ww.compact()" not in substep_source
    assert "ww.free_slots" in substep_source
    assert "ww.free_count" in substep_source
    assert "ww.compact()" in finalize_source
    assert "self._finalize_whitewater()" in frame_source


def test_whitewater_topology_budget_covers_ballistic_motion():
    method_source = inspect.getsource(FluidSolver._whitewater_topology_expand)
    frame_source = inspect.getsource(FluidSolver.step_frame)

    assert _motion_leaf_radius(0.0, 0.0, 9.81, 0.25) == 0
    assert _motion_leaf_radius(16.0, 0.25, 0.0, 0.25) == 3
    assert "ww.velocity" in method_source
    assert "self._whitewater_topology_expand(frame_dt)" in frame_source


def test_whitewater_emission_reuses_dead_slot_at_capacity():
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    device = "cuda:0"
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8),
        voxel_size=0.1, leaf_capacity=4, device=device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.79, 0.79, 0.79))
    assert grid.finalize_topology() == 1
    ww = WhitewaterSystem(capacity=2, device=device)
    wp.copy(ww.count, wp.array([2], dtype=wp.int32, device=device))
    wp.copy(ww.alive, wp.array([1, 0], dtype=wp.int32, device=device))
    wp.copy(ww.free_slots, wp.array([1, 0], dtype=wp.int32, device=device))
    wp.copy(ww.free_count, wp.array([1], dtype=wp.int32, device=device))
    liquid_position = wp.array([(0.35, 0.35, 0.35)], dtype=wp.vec3, device=device)
    liquid_velocity = wp.array([(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    liquid_count = wp.array([1], dtype=wp.int32, device=device)
    churn = wp.ones(grid.voxel_capacity, dtype=float, device=device)
    surface_sdf = wp.zeros(grid.voxel_capacity, dtype=float, device=device)

    wp.launch(
        k_emit_whitewater, dim=1,
        inputs=[
            liquid_position, liquid_velocity, liquid_count, churn, surface_sdf,
            grid.leaf_table, grid.voxels_dim, grid.leaves_dim, grid.origin,
            grid.voxel_size, 1.0, 0.0, 100.0, 10.0, 0.0, 1.0, 1.0,
            123, ww.capacity,
        ],
        outputs=[
            ww.position, ww.velocity, ww.age, ww.life, ww.alive, ww.count,
            ww.free_slots, ww.free_count,
        ],
        device=device,
    )

    assert int(ww.count.numpy()[0]) == ww.capacity
    assert int(ww.free_count.numpy()[0]) == 0
    assert int(ww.alive.numpy()[1]) == 1


def test_pressure_index_contains_only_fluid_non_collision_voxels():
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    device = "cuda:0"
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8),
        voxel_size=0.1, leaf_capacity=4, device=device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.79, 0.79, 0.79))
    assert grid.finalize_topology() == 1
    types = np.zeros(grid.voxel_capacity, dtype=np.int32)
    types[[0, 7, 511]] = VOXEL_FLUID
    types[7] |= VOXEL_COLLISION
    solver = PressureSolver(grid, device)

    count = solver._build_pressure_index(
        wp.array(types, dtype=wp.int32, device=device))

    assert count == 2
    assert solver.pressure_indices[:count].numpy().tolist() == [0, 511]

    grid.begin_topology()
    assert grid.finalize_topology() == 0
    assert solver._build_pressure_index(
        wp.array(types, dtype=wp.int32, device=device)) == 0
