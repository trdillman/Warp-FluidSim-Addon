"""Truth contracts for whitewater SDF dependency and capacity admission."""

from contextlib import nullcontext
import inspect
from types import SimpleNamespace

import warp as wp

from fluidSim_warp_blender51.sim.enums import SurfaceUpdateCadence
from fluidSim_warp_blender51.sim.solver import FluidSolver, FrameStats
from fluidSim_warp_blender51.sim.surface import SurfacePipeline
from fluidSim_warp_blender51.sim.whitewater import (
    WHITEWATER_POLICY_FIXED_CAPACITY,
    deterministic_admission,
    k_finalize_whitewater_admission,
    k_prepare_whitewater_admission,
    normalized_life_range,
)


def test_fixed_capacity_admission_is_stable_and_reports_honest_drops():
    flags = [0, 1, 1, 0, 1, 1]

    first = deterministic_admission(
        flags, live_count=3, free_count=1, capacity=4)
    second = deterministic_admission(
        flags, live_count=3, free_count=1, capacity=4)

    assert first == second == {
        "policy": WHITEWATER_POLICY_FIXED_CAPACITY,
        "required": 4,
        "available": 2,
        "admitted": 2,
        "dropped": 2,
        "winner_tids": (1, 2),
    }


def test_frame_admission_diagnostics_accumulate_available_across_substeps():
    device = "cpu"
    candidates = wp.array([1, 1], dtype=wp.int32, device=device)
    offsets = wp.array([0, 1], dtype=wp.int32, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    free_count = wp.zeros(1, dtype=wp.int32, device=device)
    plan = wp.zeros(4, dtype=wp.int32, device=device)
    diagnostics = wp.zeros(4, dtype=wp.int32, device=device)

    for _ in range(2):
        wp.launch(
            k_prepare_whitewater_admission, dim=1,
            inputs=[candidates, offsets, 2, count, free_count, 3],
            outputs=[plan, diagnostics], device=device)
        wp.launch(
            k_finalize_whitewater_admission, dim=1,
            inputs=[plan], outputs=[count, free_count], device=device)

    required, admitted, dropped, available = map(int, diagnostics.numpy())
    assert (required, available, admitted, dropped) == (4, 4, 3, 1)
    assert required - admitted == dropped
    assert admitted <= available


def test_life_range_is_non_negative_and_ordered():
    assert normalized_life_range(3.0, 1.0) == (3.0, 3.0)
    assert normalized_life_range(-2.0, -4.0) == (0.0, 0.0)
    assert normalized_life_range(1.0, 4.0) == (1.0, 4.0)


def test_solver_sdf_failure_fails_whitewater_closed_without_launching_churn(monkeypatch):
    pipeline = object.__new__(SurfacePipeline)
    pipeline.solver_sdf_key = None
    pipeline.solver_sdf_error = ""
    pipeline.solver = SimpleNamespace(
        frame=2,
        sim_time=0.5,
        settings=SimpleNamespace(
            voxel_size=0.1,
            particle_radius_scale=1.2,
            surface_splat_radius=2,
            surface_filter_type=2,
            surface_filter_iterations=0,
            surface_filter_strength=0.7,
            surface_dilate_iterations=0,
            surface_dilate_strength=1.0,
            surface_erode_iterations=0,
            surface_erode_strength=1.0,
            surface_smooth_type=2,
            surface_smooth_iterations=1,
            surface_smooth_strength=0.7,
        ),
        grid=object(),
        _nvox=8,
        _surface_positions=lambda: object(),
    )
    pipeline.sdf = pipeline.tmp = object()
    pipeline.build_sdf = lambda *_args: (_ for _ in ()).throw(RuntimeError("oom"))

    assert pipeline.ensure_solver_sdf_current() is False
    assert pipeline.solver_sdf_key is None
    assert "oom" in pipeline.solver_sdf_error

    messages = []
    solver = SimpleNamespace(
        settings=SimpleNamespace(whitewater_enabled=True),
        surface=pipeline,
        whitewater=SimpleNamespace(mark_sdf_unavailable=messages.append),
    )
    FluidSolver._whitewater(solver, 0.1, sdf_ready=False)
    assert messages == [pipeline.solver_sdf_error]


def test_phase_order_is_solver_sdf_then_whitewater():
    source = inspect.getsource(FluidSolver.step_frame)
    assert source.index("sdf_ready = self.surface.after_substep()") < source.index(
        "self._whitewater(dt, sdf_ready=sdf_ready)")


def test_frame_stats_publish_explicit_whitewater_contract():
    stats = FrameStats()
    stats.whitewater_required = 9
    stats.whitewater_available = 4
    stats.whitewater_admitted = 4
    stats.whitewater_dropped = 5
    stats.whitewater_status = "CAPACITY_EXHAUSTED"

    data = stats.as_dict()

    assert data["whitewaterRequired"] == 9
    assert data["whitewaterAvailable"] == 4
    assert data["whitewaterAdmitted"] == 4
    assert data["whitewaterDropped"] == 5
    assert data["whitewaterPolicy"] == WHITEWATER_POLICY_FIXED_CAPACITY
    assert data["whitewaterStatus"] == "CAPACITY_EXHAUSTED"


def _failing_after_substep_pipeline(*, surface_enabled, cadence):
    pipeline = object.__new__(SurfacePipeline)
    pipeline.solver = SimpleNamespace(
        settings=SimpleNamespace(
            surface_enabled=surface_enabled,
            surface_update_cadence=cadence,
            whitewater_enabled=True,
        ),
        _phase=lambda _name: nullcontext(),
    )
    pipeline.ensure_solver_sdf_current = lambda: False
    pipeline.build = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("render build must be skipped after SDF dependency failure"))
    return pipeline


def test_render_disabled_after_substep_propagates_dependency_failure():
    pipeline = _failing_after_substep_pipeline(
        surface_enabled=False, cadence=SurfaceUpdateCadence.MANUAL)

    assert pipeline.after_substep() is False


def test_substep_render_cadence_propagates_dependency_failure_without_raising():
    pipeline = _failing_after_substep_pipeline(
        surface_enabled=True, cadence=SurfaceUpdateCadence.SUBSTEP)

    assert pipeline.after_substep() is False
