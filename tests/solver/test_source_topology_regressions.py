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
