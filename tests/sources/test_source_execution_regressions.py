"""Regression tests for compiled/legacy source execution (review 2026-10-11, lane S).

Each behaviour test was written red against the unmodified code before its fix.
"""

from __future__ import annotations

import numpy as np

from fluidSim_warp_blender51.sim.enums import (
    SourceEmissionMode,
    SourceEvaluationCadence,
    SourceLifecycle,
    SourceOperation,
    SourcePayloadKind,
    SourceShapeShape,
    SourceShapeSourceMode,
    SourceShapeType,
)
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings
from fluidSim_warp_blender51.sim.source_regions import (
    PayloadPlan,
    PointRegionPlan,
    SourceKey,
    SourcePlan,
)
from fluidSim_warp_blender51.sim.sourcing import SourceShapeDesc


def _settings(device, *, max_particles=4096, auto_grow=True):
    settings = SolverSettings()
    settings.domain_min = (0.0, 0.0, 0.0)
    settings.domain_max = (2.0, 2.0, 2.0)
    settings.voxel_size = 0.25
    settings.max_particles = max_particles
    settings.auto_grow_particles = auto_grow
    settings.surface_enabled = False
    settings.whitewater_enabled = False
    settings.device = device
    return settings


def _prepare_substep(solver):
    solver._rebuild_topology(expand=0)
    solver._clear_grid_fields()
    solver._index_particles()


def _direct_points(count, stable_id="points", position=0.1):
    return SourcePlan(
        SourceKey(stable_id), SourceOperation.LIQUID, SourceLifecycle.DIRECT_APPEND,
        SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.full((count, 3), position, np.float32)),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))


# B42: removing every legacy emitter must also remove its frozen transaction
# shapes, or the next compiled-only transaction counts phantom demand.

def test_removed_legacy_emitters_leave_no_phantom_demand(cuda_device):
    solver = FluidSolver(_settings(cuda_device, max_particles=1200))
    try:
        box = SourceShapeDesc(
            enabled=True, shape=int(SourceShapeShape.BOX),
            stype=int(SourceShapeType.SOURCE), mode=int(SourceShapeSourceMode.PARTICLES),
            position=(1.0, 1.0, 1.0), half_extents=(0.5, 0.5, 0.5), emit_rate=8.0,
            emission_mode=int(SourceEmissionMode.CONTINUOUS_INFLOW), source_id="box")
        solver.set_source_batch(legacy_descriptors=(box,))
        _prepare_substep(solver)
        solver.sourcing.apply(1.0 / 24.0)
        emitted = solver.sourcing.last_apply_result.emitted
        assert emitted > 0
        # Empty the box again so a stale shape would want to refill it.
        count = solver.particles.host_count
        positions = solver.particles.position.numpy()
        positions[:count] = (1.9, 1.9, 1.9)
        solver.particles.position.assign(positions)

        solver.set_source_batch(native_plans=(_direct_points(50),))
        _prepare_substep(solver)
        solver.sourcing.apply(1.0 / 24.0)
        result = solver.sourcing.last_apply_result
        assert result.required == 50
        assert result.admitted and result.emitted == 50
        assert solver.particles.capacity == 1200
        assert solver.particles.host_count == emitted + 50
    finally:
        solver.release()


# B41: compiled coverage emission must spread a voxel's particles inside the
# voxel (deterministically), not stack them all on its centre.

def _rasterized_points(target=8.0):
    return SourcePlan(
        SourceKey("raster"), SourceOperation.LIQUID, SourceLifecycle.TOP_UP,
        SourceEvaluationCadence.STATIC,
        PointRegionPlan([[1.0, 1.0, 1.0]], rasterize_radius=0.3),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
                    target_particles_per_voxel=target))


def _emit_rasterized(device):
    solver = FluidSolver(_settings(device))
    try:
        solver.set_source_batch(native_plans=(_rasterized_points(),))
        _prepare_substep(solver)
        solver.sourcing.apply(1.0 / 24.0)
        count = solver.particles.host_count
        return solver.particles.position.numpy()[:count].copy()
    finally:
        solver.release()


def test_compiled_emission_jitters_particles_within_their_voxel(cuda_device):
    positions = _emit_rasterized(cuda_device)
    voxel = 0.25
    voxels = np.floor(positions / voxel).astype(np.int64)
    covered = np.unique(voxels, axis=0)
    assert len(positions) == 8 * len(covered)
    # Every particle is distinct and still inside the voxel it was emitted for.
    assert len(np.unique(positions, axis=0)) == len(positions)
    centres = (voxels + 0.5) * voxel
    assert np.all(np.abs(positions - centres) <= 0.5 * voxel + 1e-6)
    # The layout is a function of the grid and the source, not of thread order.
    again = _emit_rasterized(cuda_device)
    order = np.lexsort(positions.T)
    again_order = np.lexsort(again.T)
    np.testing.assert_array_equal(positions[order], again[again_order])


# P15: liquid-only analytic shapes and SOURCE mesh emitters skip their no-op
# full-grid field passes; collision/sink/activation shapes and mesh sinks
# still run them.

def _cube_mesh(device, lo=0.4, hi=1.6):
    import warp as wp

    vertices = np.array([(lo, lo, lo), (hi, lo, lo), (hi, hi, lo), (lo, hi, lo),
                         (lo, lo, hi), (hi, lo, hi), (hi, hi, hi), (lo, hi, hi)], np.float32)
    triangles = np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5),
                          (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
                          (3, 0, 4), (3, 4, 7)], np.int32)
    return wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=device),
                   indices=wp.array(triangles.reshape(-1), dtype=int, device=device))


def _shape(stype, mode=SourceShapeSourceMode.PARTICLES, source_id="shape"):
    return SourceShapeDesc(
        enabled=True, shape=int(SourceShapeShape.BOX), stype=int(stype), mode=int(mode),
        position=(1.0, 1.0, 1.0), half_extents=(0.5, 0.5, 0.5), emit_rate=2.0,
        emission_mode=int(SourceEmissionMode.CONTINUOUS_INFLOW), source_id=source_id)


def _mesh(device, stype, source_id="mesh"):
    from fluidSim_warp_blender51.sim.sourcing import MeshSourceDesc

    return MeshSourceDesc(
        enabled=True, stype=int(stype), mode=int(SourceShapeSourceMode.PARTICLES),
        mesh=_cube_mesh(device), aabb_min=(0.4,) * 3, aabb_max=(1.6,) * 3,
        emit_rate=2.0, source_id=source_id,
        emission_mode=int(SourceEmissionMode.CONTINUOUS_INFLOW))


def _field_launches(monkeypatch, solver, descriptors):
    from fluidSim_warp_blender51.sim import source_execution
    from fluidSim_warp_blender51.sim.source_kernels import k_mesh_source_field, k_source_shapes

    launched = []
    original = source_execution.fast_launch

    def recording(kernel, *args, **kwargs):
        launched.append(kernel)
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(source_execution, "fast_launch", recording)
    solver.set_source_batch(legacy_descriptors=descriptors)
    _prepare_substep(solver)
    solver.sourcing.apply(1.0 / 24.0)
    monkeypatch.setattr(source_execution, "fast_launch", original)
    return launched.count(k_source_shapes), launched.count(k_mesh_source_field)


def test_field_passes_run_only_for_roles_that_write(monkeypatch, cuda_device):
    solver = FluidSolver(_settings(cuda_device))
    try:
        assert _field_launches(monkeypatch, solver, (
            _shape(SourceShapeType.SOURCE),
            _shape(SourceShapeType.SOURCE, SourceShapeSourceMode.FIELD, "field"),
            _mesh(cuda_device, SourceShapeType.SOURCE))) == (0, 0)
        assert _field_launches(monkeypatch, solver, (
            _shape(SourceShapeType.SOURCE),
            _shape(SourceShapeType.SINK, source_id="sink"),
            _mesh(cuda_device, SourceShapeType.SINK))) == (1, 1)
        assert _field_launches(monkeypatch, solver, (
            _shape(SourceShapeType.ACTIVATION, source_id="act"),)) == (1, 0)
    finally:
        solver.release()


def _field_snapshot(solver):
    return [array.numpy().copy() for array in (
        solver.voxel_type.data, solver.collision_sdf.data, solver.collision_vel,
        solver.vel.u, solver.vel.v, solver.vel.w, solver.temperature.data,
        solver.viscosity_field.data)]


def test_skipped_field_passes_were_no_ops(cuda_device):
    import warp as wp
    from fluidSim_warp_blender51.sim.source_contracts import SourceShape
    from fluidSim_warp_blender51.sim.source_kernels import k_mesh_source_field, k_source_shapes

    solver = FluidSolver(_settings(cuda_device))
    try:
        solver.set_source_batch(legacy_descriptors=(_shape(SourceShapeType.SOURCE),))
        _prepare_substep(solver)
        rng = np.random.default_rng(5)
        for array in (solver.vel.u, solver.vel.v, solver.vel.w,
                      solver.temperature.data, solver.viscosity_field.data,
                      solver.collision_sdf.data):
            array.assign(rng.normal(size=array.shape).astype(np.float32))
        before = _field_snapshot(solver)
        grid, settings = solver.grid, solver.settings
        shapes = wp.array([
            _shape(SourceShapeType.SOURCE).to_struct(),
            _shape(SourceShapeType.SOURCE, SourceShapeSourceMode.FIELD).to_struct(),
        ], dtype=SourceShape, device=cuda_device)
        wp.launch(k_source_shapes, dim=solver._nvox,
                  inputs=[shapes, 2, grid.leaf_coords, grid.num_leaves, grid.origin,
                          settings.voxel_size, 1.0 / 24.0],
                  outputs=[solver.voxel_type.data, solver.collision_sdf.data,
                           solver.collision_vel, solver.vel.u, solver.vel.v, solver.vel.w,
                           solver.temperature.data, solver.viscosity_field.data],
                  device=cuda_device)
        source_mesh = _mesh(cuda_device, SourceShapeType.SOURCE)
        wp.launch(k_mesh_source_field, dim=solver._nvox,
                  inputs=[source_mesh.mesh.id, source_mesh.to_struct(),
                          source_mesh.query_radius(), grid.leaf_coords, grid.num_leaves,
                          grid.origin, settings.voxel_size],
                  outputs=[solver.voxel_type.data, solver.vel.u, solver.vel.v,
                           solver.vel.w, solver.temperature.data,
                           solver.viscosity_field.data],
                  device=cuda_device)
        for expected, actual in zip(before, _field_snapshot(solver)):
            np.testing.assert_array_equal(actual, expected)
    finally:
        solver.release()
