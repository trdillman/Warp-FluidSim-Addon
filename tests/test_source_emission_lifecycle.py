"""Milestone 1: Initial Fill / Continuous Inflow emission lifecycle.

These tests encode the source-emission truth contract:

* Initial Fill emits the configured region once, regardless of CFL substep
  count, and re-arms only on Initialize/Reset.
* Continuous Inflow replenishes while Enabled.
* Disabling and re-enabling an Initial Fill source does not re-arm it.
* ``SourceEmissionMode`` is a product lifecycle separate from the solver
  representation ``SourceShapeSourceMode`` (FIELD/PARTICLES).
"""

import numpy as np
import pytest

from fluidSim_warp_blender51.sim import sourcing
from fluidSim_warp_blender51.sim.enums import (
    SourceEmissionMode, SourceEvaluationCadence, SourceLifecycle,
    SourceOperation, SourcePayloadKind,
    SourceShapeShape,
    SourceShapeSourceMode,
    SourceShapeType,
)
from fluidSim_warp_blender51.sim.sourcing import SourcePipeline, SourceShapeDesc
from fluidSim_warp_blender51.sim.source_regions import (
    PayloadPlan, PointRegionPlan, SourceKey, SourcePlan,
)
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings


# ---------------------------------------------------------------------------
# Host-level armed-state logic (no GPU, no kernel launches)
# ---------------------------------------------------------------------------


def _bare_pipeline():
    """A SourcePipeline without running __init__ (which needs a real owner)."""
    pipeline = SourcePipeline.__new__(SourcePipeline)
    pipeline._initial_fired = set()
    pipeline._shape_array = None
    pipeline._emit_shape_array = None
    pipeline._emit_shape_key = None
    pipeline.analytic_sources = []
    pipeline.mesh_sources = []
    return pipeline


def test_bake_struct_continuous_source_is_always_armed():
    pipeline = _bare_pipeline()
    src = SourceShapeDesc(
        emission_mode=SourceEmissionMode.CONTINUOUS_INFLOW, source_id="tap")
    assert pipeline._bake_struct(src).armed == 1
    # Continuous sources never enter the fired set, so they stay armed.
    pipeline._initial_fired.add("tap")
    assert pipeline._bake_struct(src).armed == 1


def test_bake_struct_initial_fill_disarms_after_first_fire():
    pipeline = _bare_pipeline()
    src = SourceShapeDesc(
        emission_mode=SourceEmissionMode.INITIAL_FILL, source_id="bucket")
    # Before firing: armed.
    assert pipeline._bake_struct(src).armed == 1
    # After the host records the fire: disarmed for the rest of the run.
    pipeline._initial_fired.add("bucket")
    assert pipeline._bake_struct(src).armed == 0


def test_reset_clears_armed_state_and_invalidates_caches():
    pipeline = _bare_pipeline()
    pipeline._initial_fired = {"bucket"}
    pipeline._shape_array = object()  # stale cached device array
    pipeline._emit_shape_array = object()

    pipeline.reset()

    assert pipeline._initial_fired == set()
    assert pipeline._shape_array is None
    assert pipeline._emit_shape_array is None


def test_initial_fill_source_id_is_stable_identity_for_re_arm():
    """Re-arming is keyed by stable source_id, not descriptor identity.

    A fresh descriptor rebuilt next frame with the same source_id must still
    be recognized as already-fired.
    """
    pipeline = _bare_pipeline()
    pipeline._initial_fired.add("bucket")
    first = SourceShapeDesc(
        emission_mode=SourceEmissionMode.INITIAL_FILL, source_id="bucket")
    rebuilt = SourceShapeDesc(
        emission_mode=SourceEmissionMode.INITIAL_FILL, source_id="bucket")
    assert first is not rebuilt
    # Both are disarmed because the stable id is already fired.
    assert pipeline._bake_struct(first).armed == 0
    assert pipeline._bake_struct(rebuilt).armed == 0


# ---------------------------------------------------------------------------
# Real-solver integration: actual particle emission on the GPU
# ---------------------------------------------------------------------------


def _device():
    return sourcing.wp.get_preferred_device()


def _settings(*, max_substeps):
    s = SolverSettings()
    s.domain_min = (-0.6, -0.6, 0.0)
    s.domain_max = (0.6, 0.6, 1.2)
    s.voxel_size = 0.1
    s.max_particles = 100_000
    s.whitewater_enabled = False
    s.surface_enabled = False
    s.min_substeps = 1
    s.max_substeps = max_substeps
    s.device = _device()
    return s


def _box_source(*, emission_mode, source_id="bucket", emit_rate=6.0):
    return SourceShapeDesc(
        enabled=True,
        shape=SourceShapeShape.BOX,
        stype=SourceShapeType.SOURCE,
        mode=SourceShapeSourceMode.PARTICLES,
        position=(0.0, 0.0, 0.75),
        rotation=(0.0, 0.0, 0.0, 1.0),
        half_extents=(0.3, 0.3, 0.25),  # z in [0.5, 1.0]
        radius=0.3,
        height=0.25,
        emit_rate=emit_rate,
        emission_mode=emission_mode,
        source_id=source_id,
    )


def _count_in_box(solver, x=(-0.3, 0.3), y=(-0.3, 0.3), z=(0.5, 1.0)):
    pos = solver.particles.position.numpy()
    alive = solver.particles.alive.numpy().astype(bool)
    if not alive.any():
        return 0
    p = pos[alive]
    inside = (
        (p[:, 0] >= x[0]) & (p[:, 0] <= x[1])
        & (p[:, 1] >= y[0]) & (p[:, 1] <= y[1])
        & (p[:, 2] >= z[0]) & (p[:, 2] <= z[1])
    )
    return int(inside.sum())


def _step(solver, frames=1):
    stats = None
    for _ in range(frames):
        stats = solver.step_frame()
    return stats


def test_initial_fill_fires_once_and_does_not_replenish_after_drain():
    solver = FluidSolver(_settings(max_substeps=2))
    solver.set_sources([_box_source(emission_mode=SourceEmissionMode.INITIAL_FILL)])

    _step(solver, frames=1)
    # The source fired exactly once this run.
    assert solver.sourcing._initial_fired == {"bucket"}
    filled = _count_in_box(solver)
    assert filled > 0, "Initial Fill emitted no particles on the first step"

    # Step several more frames; gravity drains the box. An initial-fill
    # source must NOT replenish, so the box empties out.
    _step(solver, frames=8)
    assert solver.sourcing._initial_fired == {"bucket"}  # never re-records
    drained = _count_in_box(solver)
    assert drained < filled * 0.5, (
        f"Initial Fill replenished the region ({drained=} vs {filled=}); "
        "it should emit once and stop")


def test_continuous_inflow_replenishes_after_drain():
    solver = FluidSolver(_settings(max_substeps=2))
    solver.set_sources([_box_source(emission_mode=SourceEmissionMode.CONTINUOUS_INFLOW)])

    _step(solver, frames=1)
    # Continuous sources never enter the fired set.
    assert solver.sourcing._initial_fired == set()
    filled = _count_in_box(solver)
    assert filled > 0, "Continuous Inflow emitted no particles"

    # After many frames the continuous source keeps the box populated even
    # though gravity is draining it.
    _step(solver, frames=8)
    assert solver.sourcing._initial_fired == set()
    refilled = _count_in_box(solver)
    assert refilled >= filled * 0.5, (
        f"Continuous Inflow failed to replenish ({refilled=} vs {filled=})")


def test_initial_fill_rearms_only_on_reset():
    solver = FluidSolver(_settings(max_substeps=1))
    solver.set_sources([_box_source(emission_mode=SourceEmissionMode.INITIAL_FILL)])
    _step(solver, frames=1)
    assert solver.sourcing._initial_fired == {"bucket"}

    solver.reset()
    # Reset re-arms: the fired set is cleared.
    assert solver.sourcing._initial_fired == set()

    # Re-applied sources fire again after reset.
    solver.set_sources([_box_source(emission_mode=SourceEmissionMode.INITIAL_FILL)])
    _step(solver, frames=1)
    assert solver.sourcing._initial_fired == {"bucket"}
    assert _count_in_box(solver) > 0


def test_initial_fill_emit_total_is_independent_of_substep_count():
    """The configured region fills once regardless of how many CFL substeps
    the first frame takes."""
    counts = {}
    for substeps in (1, 4):
        solver = FluidSolver(_settings(max_substeps=substeps))
        solver.set_sources(
            [_box_source(emission_mode=SourceEmissionMode.INITIAL_FILL,
                         emit_rate=8.0)])
        _step(solver, frames=1)
        counts[substeps] = _count_in_box(solver)
        assert counts[substeps] > 0

    hi, lo = max(counts.values()), min(counts.values())
    # Same region, same target occupancy -> substep count must not change the
    # total materially.
    assert hi - lo <= max(1, lo * 0.05), (
        f"Initial Fill total depends on substep count: {counts}")


def test_direct_point_append_is_once_per_runtime_and_rearms_on_reset():
    solver = FluidSolver(_settings(max_substeps=1))
    plan = SourcePlan(
        SourceKey("direct-points"), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.array(((-0.1, 0.0, 0.7), (0.1, 0.0, 0.7)), np.float32)),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )
    solver.set_source_plans((plan,))
    _step(solver)
    assert solver.particles.refresh_host_count() == 2
    _step(solver)
    assert solver.particles.refresh_host_count() == 2
    solver.reset()
    solver.set_source_plans((plan,))
    _step(solver)
    assert solver.particles.refresh_host_count() == 2
    solver.release()


def test_re_enabled_initial_fill_does_not_rearm_without_reset():
    """Simulate the per-frame descriptor rebuild: source present (armed fires),
    then absent (disabled), then present again with the same source_id. Without
    a reset it must NOT re-fire."""
    pipeline = _bare_pipeline()
    pipeline.analytic_sources = []

    def armed_value(src):
        return pipeline._bake_struct(src).armed

    src = SourceShapeDesc(
        emission_mode=SourceEmissionMode.INITIAL_FILL, source_id="bucket")
    # Frame 1: present and armed.
    assert armed_value(src) == 1
    pipeline._initial_fired.add("bucket")  # host records the fire

    # Frame 2: disabled -> descriptor simply absent from the source list. The
    # fired set is unchanged.
    assert "bucket" in pipeline._initial_fired

    # Frame 3: re-enabled, fresh descriptor, same source_id. Still disarmed.
    rebuilt = SourceShapeDesc(
        emission_mode=SourceEmissionMode.INITIAL_FILL, source_id="bucket")
    assert armed_value(rebuilt) == 0
    # Only reset re-arms.
    pipeline.reset()
    assert armed_value(rebuilt) == 1
