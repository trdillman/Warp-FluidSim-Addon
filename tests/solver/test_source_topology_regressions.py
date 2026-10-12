"""Source-driven topology regressions (review 2026-10-11, lane S).

Each behaviour test was written red against the unmodified code before its fix.
"""

from __future__ import annotations

import numpy as np

from fluidSim_warp_blender51.sim.enums import (
    SourceEvaluationCadence,
    SourceLifecycle,
    SourceOperation,
    SourcePayloadKind,
    SourceShapeShape,
)
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings
from fluidSim_warp_blender51.sim.source_regions import (
    AnalyticRegionPlan,
    PayloadPlan,
    PointRegionPlan,
    SourceKey,
    SourcePlan,
)


def _settings(device):
    settings = SolverSettings()
    settings.domain_min = (0.0, 0.0, 0.0)
    settings.domain_max = (3.2, 3.2, 3.2)  # 32 voxels = 4 leaves per axis
    settings.voxel_size = 0.1
    settings.max_particles = 4096
    settings.surface_enabled = False
    settings.whitewater_enabled = False
    settings.device = device
    return settings


def _active_leaves(solver):
    count = int(solver.grid.num_leaves.numpy()[0])
    return {tuple(int(v) for v in coord)
            for coord in solver.grid.leaf_coords.numpy()[:count]}


def _pressure_box(position, half_extents):
    return SourcePlan(
        SourceKey("field"), SourceOperation.PRESSURE, SourceLifecycle.EVERY_FRAME,
        SourceEvaluationCadence.STATIC,
        AnalyticRegionPlan(shape=SourceShapeShape.BOX, position=position,
                           half_extents=half_extents),
        PayloadPlan(kind=SourcePayloadKind.PRESSURE, pressure=1.0))


# B46/P02: compiled source leaves are marked exactly, not as inclusive world
# boxes that also activate the next leaf on each positive side.

def test_compiled_source_marks_exactly_its_leaves(cuda_device):
    solver = FluidSolver(_settings(cuda_device))
    try:
        solver.set_source_batch(native_plans=(
            _pressure_box((1.6, 1.6, 1.6), (0.35, 0.35, 0.35)),))
        solver._rebuild_topology(expand=0)
        region = solver.sourcing._compiled_batch.sources[0].region
        expected = {tuple(int(v) for v in coord) for coord in region.leaf_coords}
        assert expected == {(x, y, z) for x in (1, 2) for y in (1, 2) for z in (1, 2)}
        assert _active_leaves(solver) == expected
    finally:
        solver.release()


# B48: DIRECT_APPEND points need grid support on the frame they are appended,
# and only until they have fired.

def test_direct_append_points_have_topology_on_their_emission_frame(cuda_device):
    solver = FluidSolver(_settings(cuda_device))
    try:
        from fluidSim_warp_blender51.sim.source_regions import CompileContext, SourceCompiler

        points = np.array([(2.55, 2.55, 2.55), (2.65, 2.45, 2.75)], np.float32)
        plan = SourcePlan(
            SourceKey("append"), SourceOperation.LIQUID, SourceLifecycle.DIRECT_APPEND,
            SourceEvaluationCadence.STATIC, PointRegionPlan(points),
            PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))
        solver.set_source_batch(native_plans=(plan,))
        solver._rebuild_topology(expand=0)
        assert _active_leaves(solver) == {(3, 3, 3)}
        # Admission planning counts the same leaf.
        planning = CompileContext(0, 0, 0, "cpu", (0.0, 0.0, 0.0), 0.1, (0.0, 0.0, 0.0),
                                  (3.2, 3.2, 3.2), upload=False)
        footprint = SourceCompiler().compile((plan,), planning).footprints[0]
        assert footprint.leaf_coords == ((3, 3, 3),)

        solver._clear_grid_fields()
        solver._index_particles()
        solver.sourcing.apply(1.0 / 24.0)
        assert solver.sourcing.last_apply_result.emitted == 2
        # Once fired, the source no longer pins its leaf: moved particles
        # mark their own leaves.
        count = solver.particles.host_count
        positions = solver.particles.position.numpy()
        positions[:count] = (0.35, 0.35, 0.35)
        solver.particles.position.assign(positions)
        solver._rebuild_topology(expand=0)
        assert _active_leaves(solver) == {(0, 0, 0)}
    finally:
        solver.release()


# B45/A7: planning, topology and particle emission share one profile-aware
# height model (the level k_source_height fills to).

def _height(back_mode, profile=None, base=0.3):
    from fluidSim_warp_blender51.sim.enums import SourceEmissionMode, SourceHeightBackMode
    from fluidSim_warp_blender51.sim.sourcing import HeightSourceDesc

    return HeightSourceDesc(
        enabled=True, base_height=base, back_mode=int(SourceHeightBackMode[back_mode]),
        profile=None if profile is None else np.full((2, 2), profile, np.float32),
        emit_particles=True, emit_rate=4.0,
        emission_mode=int(SourceEmissionMode.CONTINUOUS_INFLOW))


def _height_settings(device):
    settings = _settings(device)
    settings.domain_max = (1.6, 1.6, 1.6)  # 16 voxels = 2 leaves per axis
    settings.max_particles = 200_000
    return settings


def test_height_profile_planning_covers_profile_peaks():
    from fluidSim_warp_blender51.sim.source_compat import height_descriptor_to_plan
    from fluidSim_warp_blender51.sim.source_regions import CompileContext, SourceCompiler

    context = CompileContext(0, 0, 0, "cpu", (0.0, 0.0, 0.0), 0.1, (0.0, 0.0, 0.0),
                             (1.6, 1.6, 1.6), upload=False)
    peaked = SourceCompiler().compile(
        (height_descriptor_to_plan(_height("PROFILE", 0.8)),), context).sources[0].region
    assert {int(coord[2]) for coord in peaked.topology_leaf_coords} == {0, 1}
    # Every voxel centre up to the profiled level 1.1 is covered.
    assert peaked.covered_voxel_count == 16 * 16 * 11
    flat = SourceCompiler().compile(
        (height_descriptor_to_plan(_height("FLAT")),), context).sources[0].region
    assert {int(coord[2]) for coord in flat.topology_leaf_coords} == {0}
    assert flat.covered_voxel_count == 16 * 16 * 3


def _height_solver(device, height):
    solver = FluidSolver(_height_settings(device))
    solver.set_height_source(height)
    solver._rebuild_topology(expand=0)
    solver._clear_grid_fields()
    solver._index_particles()
    return solver


def test_height_profile_topology_and_emission_follow_the_profile(cuda_device):
    solver = _height_solver(cuda_device, _height("PROFILE", 0.8))
    try:
        assert {leaf[2] for leaf in _active_leaves(solver)} == {0, 1}
        solver.sourcing.apply(1.0 / 24.0)
        count = solver.particles.host_count
        z = solver.particles.position.numpy()[:count, 2]
        # 16 x 16 columns x 11 voxels below the 1.1 level, 4 particles each.
        assert count == 16 * 16 * 11 * 4
        assert z.max() > 1.0 and z.max() <= 1.1 + 1e-6
    finally:
        solver.release()


def test_height_profile_below_base_emits_below_the_surface(cuda_device):
    solver = _height_solver(cuda_device, _height("PROFILE", -0.2))
    try:
        solver.sourcing.apply(1.0 / 24.0)
        count = solver.particles.host_count
        z = solver.particles.position.numpy()[:count, 2]
        assert count == 16 * 16 * 1 * 4
        assert z.max() <= 0.1 + 1e-6
    finally:
        solver.release()


def test_flat_height_topology_and_emission_are_unchanged(cuda_device):
    solver = _height_solver(cuda_device, _height("FLAT"))
    try:
        assert {leaf[2] for leaf in _active_leaves(solver)} == {0}
        solver.sourcing.apply(1.0 / 24.0)
        count = solver.particles.host_count
        z = solver.particles.position.numpy()[:count, 2]
        assert count == 16 * 16 * 3 * 4
        assert z.max() <= 0.3 + 1e-6
    finally:
        solver.release()
