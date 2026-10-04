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
from fluidSim_warp_blender51.sim.sourcing import (
    MeshSourceDesc, SourceApplyResult, SourcePipeline, SourceShapeDesc,
)
from fluidSim_warp_blender51.sim.source_regions import (
    AnalyticRegionPlan, PayloadPlan, PointRegionPlan, SourceKey, SourcePlan,
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
    pipeline.last_injection_result = object()

    pipeline.reset()

    assert pipeline._initial_fired == set()
    assert pipeline._shape_array is None
    assert pipeline._emit_shape_array is None
    assert pipeline.last_injection_result is None


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
    if not sourcing.wp.is_cuda_available():
        pytest.skip("CUDA not available")
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


def test_height_fill_initial_fill_stops_injecting_mass_and_momentum():
    """Domain Height Fill honours its Fill Lifecycle.

    The fill velocity carries the liquid out of the fill volume, which leaves
    those voxels under-occupied. While the height source ignored its emission
    mode it refilled them every substep at the fill velocity -- an endless mass
    and momentum source that ran the particle count up until the sim blew up.
    """
    from fluidSim_warp_blender51.sim.sourcing import HeightSourceDesc

    solver = FluidSolver(_settings(max_substeps=2))
    solver.set_height_source(HeightSourceDesc(
        enabled=True, base_height=0.4, back_mode=1, velocity=(2.0, 0.0, 0.0),
        emit_particles=True, emission_mode=SourceEmissionMode.INITIAL_FILL))

    _step(solver, frames=1)
    filled = solver.particles.refresh_host_count()
    assert filled > 0, "Height Fill emitted no particles"
    assert solver.sourcing._initial_fired == {"builtin:height"}

    _step(solver, frames=12)
    assert solver.particles.refresh_host_count() == filled, (
        "Initial Fill height source kept emitting after it fired")
    solver.release()


def test_zero_rate_height_fill_matches_non_particle_grid_admission():
    from fluidSim_warp_blender51.sim import constants as C
    from fluidSim_warp_blender51.sim.sourcing import HeightSourceDesc

    tracked_counts = []
    for emit_particles in (False, True):
        solver = FluidSolver(_settings(max_substeps=1))
        solver.set_height_source(HeightSourceDesc(
            enabled=True, base_height=0.4, back_mode=1,
            emit_particles=emit_particles, emit_rate=0.0,
            emission_mode=SourceEmissionMode.INITIAL_FILL))

        _step(solver)

        tracked = solver.voxel_type.data.numpy() & C.VOXEL_SOURCE_REGION_TRACKING
        tracked_counts.append(int(np.count_nonzero(tracked)))
        assert solver.particles.refresh_host_count() == 0
        assert "builtin:height" not in solver.sourcing._initial_fired
        solver.release()

    assert tracked_counts[0] > 0
    assert tracked_counts[1] == tracked_counts[0]


def test_height_fill_over_capacity_does_not_stamp_or_consume_lifecycle():
    from fluidSim_warp_blender51.sim.sourcing import HeightSourceDesc

    settings = _settings(max_substeps=1)
    settings.max_particles = 4
    solver = FluidSolver(settings)
    solver.set_height_source(HeightSourceDesc(
        enabled=True, base_height=0.4, back_mode=1,
        emit_particles=True, emission_mode=SourceEmissionMode.INITIAL_FILL))
    seed = solver.sourcing._emit_seed

    _step(solver)

    assert solver.particles.refresh_host_count() == 0
    assert solver.sourcing.last_apply_result["admitted"] is False
    assert solver.sourcing.last_apply_result["required"] > 4
    assert "builtin:height" not in solver.sourcing._initial_fired
    assert solver.sourcing._emit_seed == seed
    solver.release()


def test_height_fill_still_applies_alongside_compiled_plan_sources():
    """The domain Height Fill is not a scene object and compiles to no plan.

    ``apply`` routes to the compiled path whenever plans exist and no legacy
    descriptors do, so a single plan-based source used to take the height fill
    down with it -- the domain came up empty with no diagnostic.
    """
    from fluidSim_warp_blender51.sim.sourcing import HeightSourceDesc

    solver = FluidSolver(_settings(max_substeps=1))
    solver.set_height_source(HeightSourceDesc(
        enabled=True, base_height=0.4, back_mode=1,
        emit_particles=True, emission_mode=SourceEmissionMode.INITIAL_FILL))
    solver.set_source_plans((SourcePlan(
        SourceKey("direct-points"), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.array(((-0.1, 0.0, 0.7),), np.float32)),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    ),))

    _step(solver, frames=1)
    assert solver.particles.refresh_host_count() > 1, (
        "compiled plan path dropped the Height Fill")
    assert solver.sourcing._initial_fired >= {"builtin:height"}
    solver.release()


def test_mixed_source_batch_owns_each_representation_once():
    solver = FluidSolver(_settings(max_substeps=1))
    assert solver.sourcing.last_injection_result is None
    legacy = _box_source(
        emission_mode=SourceEmissionMode.INITIAL_FILL,
        source_id="legacy-mixed", emit_rate=1.0)
    direct = SourcePlan(
        SourceKey("native-direct"), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.array(((0.45, 0.45, 0.9),), np.float32)),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )
    volume = SourcePlan(
        SourceKey("native-volume"), SourceOperation.LIQUID,
        SourceLifecycle.ONCE, SourceEvaluationCadence.STATIC,
        AnalyticRegionPlan(
            shape=SourceShapeShape.BOX, position=(-0.45, -0.45, 0.9),
            half_extents=(0.15, 0.15, 0.15)),
        PayloadPlan(
            kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
            target_particles_per_voxel=1.0),
    )

    solver.set_source_batch((legacy,), (direct, volume))
    assert solver.sourcing.sources == [legacy]
    assert solver.sourcing.plans == (direct, volume)
    _step(solver)
    first_count = solver.particles.refresh_host_count()
    positions = solver.particles.position.numpy()[:first_count]
    native_volume_count = int(np.count_nonzero(
        np.linalg.norm(
            positions - np.array((-0.45, -0.45, 0.9), np.float32), axis=1
        ) < 0.2
    ))
    native_direct_count = int(np.count_nonzero(
        np.linalg.norm(
            positions - np.array((0.45, 0.45, 0.9), np.float32), axis=1
        ) < 0.2
    ))
    legacy_count = int(np.count_nonzero(
        np.linalg.norm(
            positions - np.array((0.0, 0.0, 0.75), np.float32), axis=1
        ) < 0.3
    ))
    assert native_volume_count > 0
    assert native_direct_count == 1
    assert legacy_count > 0
    assert solver.sourcing._initial_fired == {
        "legacy-mixed", "native-direct", "native-volume"}

    _step(solver)
    assert solver.particles.refresh_host_count() == first_count
    assert solver.sourcing._initial_fired == {
        "legacy-mixed", "native-direct", "native-volume"}
    solver.release()


def test_zero_liquid_plan_stays_armed_while_field_plan_remains_active():
    solver = FluidSolver(_settings(max_substeps=1))
    zero = SourcePlan(
        SourceKey("zero-liquid"), SourceOperation.LIQUID,
        SourceLifecycle.ONCE, SourceEvaluationCadence.STATIC,
        AnalyticRegionPlan(
            shape=SourceShapeShape.BOX, position=(0.0, 0.0, 0.75),
            half_extents=(0.2, 0.2, 0.2)),
        PayloadPlan(
            kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
            target_particles_per_voxel=0.0),
    )
    pressure = SourcePlan(
        SourceKey("pressure-field"), SourceOperation.PRESSURE,
        SourceLifecycle.EVERY_FRAME, SourceEvaluationCadence.STATIC,
        AnalyticRegionPlan(
            shape=SourceShapeShape.BOX, position=(0.0, 0.0, 0.75),
            half_extents=(0.2, 0.2, 0.2)),
        PayloadPlan(kind=SourcePayloadKind.PRESSURE, pressure=3.0),
    )
    solver.set_source_plans((zero, pressure))

    _step(solver)

    assert solver.particles.refresh_host_count() == 0
    assert "zero-liquid" not in solver.sourcing._initial_fired
    assert solver.sourcing.last_pressure_apply_count > 0
    solver.release()


def test_direct_packet_exact_fit_preserves_attributes_and_public_totals():
    settings = _settings(max_substeps=1)
    settings.max_particles = 4
    solver = FluidSolver(settings)
    packets = (
        sourcing.DirectInjectionPacket(
            np.array(((-0.2, 0.0, 0.7),), np.float32),
            temperature=2.0, viscosity=3.0),
        sourcing.DirectInjectionPacket(
            np.array(((0.0, 0.0, 0.7), (0.2, 0.0, 0.7)), np.float32),
            temperature=5.0, viscosity=7.0),
    )

    result = solver.inject_particle_packets_exact(packets)

    assert result == sourcing.InjectionResult(
        admitted=True, emitted=3, required=3, available=4,
        capacity_limited=False, total=3)
    np.testing.assert_allclose(
        np.sort(solver.particles.attr("temperature").numpy()[:3]),
        (2.0, 5.0, 5.0))
    np.testing.assert_allclose(
        np.sort(solver.particles.attr("viscosity").numpy()[:3]),
        (3.0, 7.0, 7.0))
    injection = solver.inject_particles(
        np.array(((0.4, 0.0, 0.7),), np.float32))
    assert injection.admitted and injection.emitted == 1 and injection.total == 4
    solver.release()


def test_direct_injection_refuses_whole_batch_from_stale_host_count():
    settings = _settings(max_substeps=1)
    settings.max_particles = 3
    settings.auto_grow_particles = False
    solver = FluidSolver(settings)
    injection = solver.inject_particles(np.array(((0.0, 0.0, 0.5),), np.float32))
    assert injection.admitted and injection.emitted == 1

    # Deliberately stale mirror: device count is one, so only two slots remain.
    solver.particles.host_count = 0
    points = np.array(((-0.2, 0.0, 0.7), (0.0, 0.0, 0.7),
                       (0.2, 0.0, 0.7)), np.float32)
    injection = solver.inject_particles(points)
    assert not injection.admitted and injection.emitted == 0
    result = solver.sourcing.last_injection_result
    assert result.admitted is False
    assert result.required == 3
    assert result.available == 2
    assert result.emitted == 0
    assert solver.particles.refresh_host_count() == 1
    solver.release()


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


def test_zero_count_direct_append_does_not_mask_an_earlier_emission():
    """A later empty source must not erase an earlier source's emission.

    ``apply_compiled`` only refreshes the host particle count when something
    emitted. If a zero-count source overwrites that flag, the emitted particles
    are live on the device but never re-indexed for the substep, and the
    reported emission count is 0.
    """
    solver = FluidSolver(_settings(max_substeps=1))
    filled = SourcePlan(
        SourceKey("direct-filled"), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.array(((-0.1, 0.0, 0.7), (0.1, 0.0, 0.7)), np.float32)),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )
    empty = SourcePlan(
        SourceKey("direct-empty"), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.zeros((0, 3), np.float32)),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )
    solver.set_source_plans((filled, empty))
    _step(solver)
    assert solver.particles.refresh_host_count() == 2
    assert solver.sourcing.last_apply_result["emitted"] == 2
    solver.release()


def test_capacity_gated_liquid_source_stays_armed():
    """A source gated out by a full particle pool must not be marked fired.

    The gate reads ``particles.host_count``, which ``apply_compiled`` refreshes
    only at the end of the loop. A direct-append source earlier in the same
    batch fills the pool, so the box source must observe the pool as full, skip
    its emit, and stay armed for a later substep instead of being consumed.
    """
    settings = _settings(max_substeps=1)
    settings.max_particles = 2
    solver = FluidSolver(settings)
    assert solver.particles.capacity == 2
    direct = SourcePlan(
        SourceKey("direct-fill"), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.array(((-0.1, 0.0, 0.7), (0.1, 0.0, 0.7)), np.float32)),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )
    box = SourcePlan(
        SourceKey("once-box"), SourceOperation.LIQUID,
        SourceLifecycle.ONCE, SourceEvaluationCadence.STATIC,
        AnalyticRegionPlan(
            shape=SourceShapeShape.BOX, position=(0.0, 0.0, 0.75),
            half_extents=(0.3, 0.3, 0.25)),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )
    solver.set_source_plans((direct, box))
    _step(solver)
    assert solver.particles.refresh_host_count() == 0
    assert str(direct.stable_id) not in solver.sourcing._initial_fired
    assert str(box.stable_id) not in solver.sourcing._initial_fired
    assert solver.sourcing.last_apply_result["admitted"] is False
    assert solver.sourcing.last_apply_result["required"] > 2
    assert solver.sourcing.last_apply_result["available"] == 2
    solver.release()


def test_aggregate_direct_append_exact_fit_commits_every_source():
    settings = _settings(max_substeps=1)
    settings.max_particles = 4
    solver = FluidSolver(settings)
    first = SourcePlan(
        SourceKey("direct-a"), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.array(((-0.2, 0.0, 0.7), (0.0, 0.0, 0.7)), np.float32)),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )
    second = SourcePlan(
        SourceKey("direct-b"), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.array(((0.1, 0.0, 0.7), (0.2, 0.0, 0.7)), np.float32)),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )
    solver.set_source_plans((first, second))

    _step(solver)

    assert solver.particles.refresh_host_count() == 4
    assert solver.sourcing.last_apply_result == SourceApplyResult(
        emitted=4, capacity_limited=False,
        required=4, available=4, admitted=True)
    assert solver.sourcing._initial_fired == {"direct-a", "direct-b"}
    solver.release()


def test_legacy_initial_fill_over_capacity_is_zero_commit_and_retryable():
    settings = _settings(max_substeps=1)
    settings.max_particles = 8
    solver = FluidSolver(settings)
    solver.set_sources([_box_source(
        emission_mode=SourceEmissionMode.INITIAL_FILL,
        source_id="legacy-overflow", emit_rate=6.0)])
    seed = solver.sourcing._emit_seed

    _step(solver)

    report = solver.sourcing.last_apply_result
    assert report["admitted"] is False
    assert report["required"] > report["available"] == 8
    assert solver.particles.refresh_host_count() == 0
    assert "legacy-overflow" not in solver.sourcing._initial_fired
    assert solver.sourcing._emit_seed == seed

    # Retry the same stable lifecycle after an explicit smaller configuration.
    solver.set_sources([SourceShapeDesc(
        enabled=True, shape=SourceShapeShape.BOX,
        stype=SourceShapeType.SOURCE,
        mode=SourceShapeSourceMode.PARTICLES,
        position=(0.05, 0.05, 0.75), half_extents=(0.04, 0.04, 0.04),
        emit_rate=2.0, emission_mode=SourceEmissionMode.INITIAL_FILL,
        source_id="legacy-overflow",
    )])
    _step(solver)
    # Assert the source transaction itself: later solver phases may cull a
    # boundary-jittered particle and are outside this lifecycle contract.
    retry = solver.sourcing.last_apply_result
    assert retry["admitted"] is True
    assert retry["emitted"] == 2
    assert "legacy-overflow" in solver.sourcing._initial_fired
    solver.release()


def test_legacy_mesh_over_capacity_is_rejected_before_commit():
    settings = _settings(max_substeps=1)
    settings.max_particles = 1
    device = settings.device
    vertices = np.array(((-0.4, -0.4, 0.6), (0.4, -0.4, 0.6),
                         (-0.4, 0.4, 0.6)), np.float32)
    triangles = np.array(((0, 1, 2),), np.int32)
    mesh = sourcing.wp.Mesh(
        points=sourcing.wp.array(vertices, dtype=sourcing.wp.vec3, device=device),
        indices=sourcing.wp.array(
            triangles.reshape(-1), dtype=sourcing.wp.int32, device=device),
        support_winding_number=True,
    )
    solver = FluidSolver(settings)
    solver.set_sources([MeshSourceDesc(
        mesh=mesh, vertices=vertices, triangles=triangles,
        aabb_min=tuple(vertices.min(axis=0)),
        aabb_max=tuple(vertices.max(axis=0)),
        surface_thickness=0.15, emit_rate=2.0,
        emission_mode=SourceEmissionMode.INITIAL_FILL,
        source_id="mesh-overflow",
    )])

    _step(solver)

    assert solver.particles.refresh_host_count() == 0
    assert solver.sourcing.last_apply_result["admitted"] is False
    assert solver.sourcing.last_apply_result["required"] > 1
    assert "mesh-overflow" not in solver.sourcing._initial_fired
    solver.release()


def test_direct_inject_rejects_before_partial_append_when_auto_growth_is_disabled():
    settings = _settings(max_substeps=1)
    settings.max_particles = 2
    settings.auto_grow_particles = False
    solver = FluidSolver(settings)
    points = np.array(((-0.2, 0.0, 0.7), (0.0, 0.0, 0.7),
                       (0.2, 0.0, 0.7)), np.float32)

    injection = solver.inject_particles(points)
    assert not injection.admitted and injection.emitted == 0
    assert solver.particles.refresh_host_count() == 0
    result = solver.sourcing.last_apply_result
    assert result["emitted"] == 0
    assert result["capacity_limited"] is True
    assert result["required"] == 3
    assert result["available"] == 2
    assert result["admitted"] is False
    assert result["deferred"] is True
    assert result["growth_status"] == "DISABLED"
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
