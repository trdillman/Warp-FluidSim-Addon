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
