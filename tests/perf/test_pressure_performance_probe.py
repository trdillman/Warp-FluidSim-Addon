"""Pressure-segment attribution probe (plan 2026-08-26-pressure-follow-on-performance, Task 1).

Report test: drives the *unmodified* production PCG pressure pipeline over three
pressure-heavy geometries (closed domain, free surface, moving collider) and
produces a structured JSON stage breakdown at the graph boundaries named by the
plan interface: assembly, index marking, matrix apply, precondition/update,
dot/reduction, graph launch/capture, checkpoint readback, projection, plus
total pressure and whole frame. Each record also binds active leaves/voxels,
unknown count, iterations, status, residual, divergence, and whole frame.

Timing method
-------------
CUDA events (``wp.Event(enable_timing=True)``) are recorded around the solver's
existing graph-boundary host calls. No production code changes, no per-kernel
synchronization is added, and no event is ever recorded inside a stream-capture
window: every instrumented boundary lies wholly outside the capture regions in
``PressureSolver`` (the ``ScopedCapture`` blocks run inside wrapped calls, so
the surrounding events bracket them). Segments production keeps *inside* a
captured graph (matrix apply, precondition/update, per-iteration dot/reduction)
cannot be split without changing execution, so after the faithful solves the
probe re-launches the already-recorded command bundle segment-by-segment
(mirroring ``PressureSolver._record_pcg_iterations``) to attribute them. Cold
capture and warm execution are always reported separately.

Run headless with pytest::

    python -m pytest -q tests/perf/test_pressure_performance_probe.py

or as a plain script (JSON per case is printed and mirrored under
``artifacts/real_world_performance/pressure/``)::

    python tests/perf/test_pressure_performance_probe.py

The first process pays Warp JIT for the pressure kernels (the pytest session
cache is redirected by ``tests/conftest.py``); later runs reuse the cache.
"""

from __future__ import annotations

import functools
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "support"))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.constants import (  # noqa: E402
    LEAF_VOLUME,
    VOXEL_COLLISION,
    VOXEL_FLUID,
)
from fluidSim_warp_blender51.sim.fields import MacVelocityField, SparseGrid  # noqa: E402
from fluidSim_warp_blender51.sim.pressure import PressureSolver  # noqa: E402
from _test_device import script_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Stage names required by the plan Task 1 interface, plus the rollup extras the
# probe measures alongside them.
REQUIRED_STAGES = (
    "assembly",
    "index_marking",
    "matrix_apply",
    "precondition_update",
    "dot_reduction",
    "graph_launch_capture",
    "checkpoint_readback",
    "projection",
)
TOTAL_PRESSURE_STAGE = "total_pressure"
WHOLE_FRAME_STAGE = "whole_frame"

GEOMETRY_NAMES = ("closed_domain", "free_surface", "moving_collider")

FIXTURES = {
    # One checkpoint batch: exercises cold command recording + conditional
    # graph capture against warm graph launch.
    "fixed_iterations": {
        "max_iterations": 25,
        "tolerance": 1.0e-6,
        "force_host_batched": False,
    },
    # Normal production-style tolerance: converges (or hits the cap) with the
    # device-resident conditional loop.
    "normal_tolerance": {
        "max_iterations": 200,
        "tolerance": 1.0e-5,
        "force_host_batched": False,
    },
    # Production host-batched fallback: per-batch graph launches plus the
    # per-checkpoint packed readback syncs.
    "host_batched": {
        "max_iterations": 25,
        "tolerance": 1.0e-6,
        "force_host_batched": True,
    },
}

WARM_REPLICATES = 3
REPLAY_ITERATIONS = 25
WARMUP_ITERATIONS = 25

# 64^3 voxels = 512 leaves = 262144 slots; enough unknowns that PCG segments
# are measurable at event resolution without dominating a test run.
_DOMAIN_SIZE = (3.2, 3.2, 3.2)
_VOXEL_SIZE = 0.05
_LEAF_CAPACITY = 512

_KNOWN_STATUSES = (
    "converged",
    "max_iterations",
    "stalled",
    "singular",
    "invalid",
)


# ---------------------------------------------------------------------------
# CUDA-event stage timing (graph-boundary, capture-safe)
# ---------------------------------------------------------------------------


class _Variant:
    """Cold/warm label resolved after the wrapped call returns."""

    def __init__(self):
        self.value = "unknown"


class _Span:
    """One CUDA-event pair plus a host wall measurement."""

    def __init__(self, timings, stage, variant):
        self._timings = timings
        self._stage = stage
        self._variant = variant
        self._start = None
        self._host_ns = 0

    def __enter__(self):
        self._start = wp.Event(device=self._timings.device, enable_timing=True)
        wp.record_event(self._start)
        self._host_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        host_ns = time.perf_counter_ns() - self._host_ns
        end = wp.Event(device=self._timings.device, enable_timing=True)
        wp.record_event(end)
        self._timings._pending.append((self._stage, self._variant, self._start, end, host_ns))
        return False


class _StageTimings:
    """Collects graph-boundary event spans; each :meth:`drain` closes a bucket.

    Buckets keep cold capture, each warm replicate, and the attribution replay
    separately so per-frame stage totals stay attributable.
    """

    def __init__(self, device):
        self.device = device
        self._pending = []
        self.buckets = []

    def span(self, stage, variant="fixed"):
        return _Span(self, stage, variant)

    def drain(self):
        """One coarse device sync per measured segment group (never per kernel)."""
        wp.synchronize_device(self.device)
        bucket = defaultdict(list)
        for stage, variant, start, end, host_ns in self._pending:
            gpu_ms = float(wp.get_event_elapsed_time(start, end, synchronize=False))
            variant = variant.value if isinstance(variant, _Variant) else variant
            bucket[(stage, variant)].append((gpu_ms, host_ns * 1.0e-6))
        self._pending.clear()
        self.buckets.append(dict(bucket))
        return bucket


def _wrap_fixed(solver, timings, method_name, stage):
    original = getattr(solver, method_name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        with timings.span(stage):
            return original(*args, **kwargs)

    setattr(solver, method_name, wrapper)


def _wrap_graph_stage(solver, timings, method_name):
    """Wrap a host call that may capture a CUDA graph on a cold key.

    The variant (``cold`` when the call minted a graph, ``warm`` when it
    replayed one) is resolved from the captured-graph counters around the call,
    so cold capture and warm launch stay separately attributable.
    """
    original = getattr(solver, method_name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        before = len(solver._pcg_graphs) + len(solver._pcg_conditional_graphs)
        variant = _Variant()
        with timings.span("graph_launch_capture", variant):
            result = original(*args, **kwargs)
        after = len(solver._pcg_graphs) + len(solver._pcg_conditional_graphs)
        variant.value = "cold" if after > before else "warm"
        return result

    setattr(solver, method_name, wrapper)


def _instrument_solver(solver, timings):
    """Attach graph-boundary event timing to one solver instance.

    Only method boundaries are wrapped, so execution is unchanged and no event
    can land inside a ScopedCapture window.
    """
    _wrap_fixed(solver, timings, "build_rhs", "assembly")
    _wrap_fixed(solver, timings, "build_preconditioner", "preconditioner_build")
    _wrap_fixed(solver, timings, "solve_pcg", "solve_total")
    _wrap_fixed(solver, timings, "project", "projection")
    _wrap_fixed(solver, timings, "_build_pressure_index", "index_marking")
    _wrap_fixed(solver, timings, "_prepare_pcg_commands", "command_record")
    _wrap_graph_stage(solver, timings, "_launch_pcg_batch")
    _wrap_graph_stage(solver, timings, "_launch_pcg_conditional")
    _wrap_graph_stage(solver, timings, "_get_pcg_conditional_graphs")
    return solver


def _bucket_stage_totals(bucket):
    """{stage: {variant: {gpu_ms, host_ms, calls}}} for one drained bucket."""
    totals = {}
    for (stage, variant), values in bucket.items():
        gpu = sum(v[0] for v in values)
        host = sum(v[1] for v in values)
        totals.setdefault(stage, {})[variant] = {
            "gpu_ms": gpu,
            "host_ms": host,
            "calls": len(values),
        }
    return totals


def _bucket_stage(bucket_totals, stage, variant="fixed"):
    entry = bucket_totals.get(stage, {}).get(variant)
    if entry is None:
        return {"gpu_ms": 0.0, "host_ms": 0.0, "calls": 0}
    return entry


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------


class _Geometry:
    def __init__(self, name, grid, voxel_type, vel, collision_vel, expected_unknowns):
        self.name = name
        self.grid = grid
        self.voxel_type = voxel_type
        self.vel = vel
        self.collision_vel = collision_vel
        self.expected_unknowns = expected_unknowns


def _build_geometry(device, name):
    """Materialize one pressure-heavy geometry on a 64^3 sparse domain.

    - ``closed_domain``: fluid interior with a solid shell on every face
      (Neumann walls; non-singular because wall faces carry solid terms).
    - ``free_surface``: a fluid block with air on all sides (Dirichlet faces).
    - ``moving_collider``: closed domain plus an embedded solid box advecting
      at a nonzero collision velocity, which feeds k_compute_divergence.
    """
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0),
        size=_DOMAIN_SIZE,
        voxel_size=_VOXEL_SIZE,
        leaf_capacity=_LEAF_CAPACITY,
        device=device,
    )
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), tuple(s - 0.001 for s in _DOMAIN_SIZE))
    leaves = grid.finalize_topology()
    assert leaves == _LEAF_CAPACITY, (name, leaves)

    # Flat slot layout: slot = leaf_slot * LEAF_VOLUME + (x + 8y + 64z), with
    # the leaf cell coordinates in grid.leaf_coords (allocation order).
    leaf_coords = grid.leaf_coords.numpy()[:leaves].astype(np.int64)
    local = np.arange(LEAF_VOLUME, dtype=np.int64)
    lx, ly, lz = local & 7, (local >> 3) & 7, local >> 6
    ijk = (
        leaf_coords[:, None, :] * 8
        + np.stack([lx, ly, lz], axis=1)[None, :, :]
    )  # (leaves, 512, 3)
    i, j, k = ijk[..., 0], ijk[..., 1], ijk[..., 2]
    nx = int(grid.voxels_dim[0])
    ny = int(grid.voxels_dim[1])
    shell = (
        (i == 0) | (i == nx - 1)
        | (j == 0) | (j == ny - 1)
        | (k == 0) | (k == ny - 1)
    )

    collider = np.zeros(i.shape, dtype=bool)
    if name == "closed_domain":
        vt = np.where(shell, VOXEL_COLLISION, VOXEL_FLUID)
    elif name == "free_surface":
        vt = np.where(k < ny // 2, VOXEL_FLUID, 0)
    elif name == "moving_collider":
        box = (
            (i >= 24) & (i < 40)
            & (j >= 24) & (j < 40)
            & (k >= 24) & (k < 40)
        )
        vt = np.where(shell | box, VOXEL_COLLISION, VOXEL_FLUID)
        collider = box & ~shell
    else:
        raise ValueError(f"unknown geometry {name!r}")

    vt = vt.astype(np.int32).reshape(-1)
    fluid = (vt & VOXEL_FLUID) != 0
    solid = (vt & VOXEL_COLLISION) != 0
    expected_unknowns = int(np.count_nonzero(fluid & ~solid))

    n = grid.voxel_capacity
    voxel_type = wp.array(vt, dtype=wp.int32, device=device)
    vel = MacVelocityField(grid)
    vel.u.fill_(1.0)
    vel.v.fill_(0.25)
    vel.w.fill_(0.1)

    collision_np = np.zeros((n, 3), dtype=np.float32)
    if collider.any():
        collision_np[collider.reshape(-1), 0] = 0.5
    collision_vel = wp.array(collision_np, dtype=wp.vec3, device=device)
    return _Geometry(name, grid, voxel_type, vel, collision_vel, expected_unknowns)


# ---------------------------------------------------------------------------
# Measured frame + attribution replay
# ---------------------------------------------------------------------------


def _frame_once(solver, timings, geom, fixture):
    """One production pressure frame: build -> precondition -> solve -> project.

    ``total_pressure`` brackets the pipeline proper; ``whole_frame`` adds the
    divergence binding readback a report frame performs.
    """
    # Restore the fixture velocity field: project() rewrites it in place, so
    # every replicate must start from identical data to stay comparable.
    geom.vel.u.fill_(1.0)
    geom.vel.v.fill_(0.25)
    geom.vel.w.fill_(0.1)
    with timings.span(WHOLE_FRAME_STAGE):
        with timings.span(TOTAL_PRESSURE_STAGE):
            solver.build_rhs(
                geom.vel, geom.voxel_type, geom.collision_vel, None)
            solver.build_preconditioner(geom.voxel_type)
            solver.solve_pcg(
                geom.voxel_type,
                fixture["max_iterations"],
                fixture["tolerance"],
            )
            if solver.should_project_pressure():
                solver.project(geom.vel, geom.voxel_type, geom.collision_vel)
        count = solver.grid.active_voxel_count
        wp.utils.array_inner(
            solver.divergence, solver.divergence,
            out=solver.residual, count=count)
        divergence_l2 = math.sqrt(
            max(0.0, float(solver.residual.numpy()[0])))
    # Bind solve facts before any later frame overwrites the solver mirrors.
    return {
        "execution_mode": solver.last_pcg_execution_mode,
        "iterations": int(solver.last_iterations),
        "status": solver.last_pressure_status,
        "pressure_valid": bool(solver.last_pressure_valid),
        "residual": float(solver.last_residual),
        "divergence_l2": divergence_l2,
    }


def _attribution_replay(solver, timings, iterations):
    """Re-launch the recorded command bundle segment-by-segment.

    Mirrors ``PressureSolver._record_pcg_iterations`` exactly (same commands,
    same order, same parity handling) but outside graph capture so matrix
    apply, precondition/update, and dot/reduction get their own event pairs.
    Runs after the faithful solves, so report facts are bound first.
    """
    if solver.pressure_count <= 0:
        return None
    full_n = solver.grid.active_voxel_count
    work_n = min(full_n, 1 << (solver.pressure_count - 1).bit_length())
    dot = solver._pcg_dot

    def _reduce(a, b, out):
        if dot is not None:
            dot.launch(a, b, out, work_n)
        else:
            wp.utils.array_inner(a, b, out=out, count=work_n)

    parity = 0 if solver.rz is solver._pcg_rz_a else 1
    for _ in range(iterations):
        with timings.span("matrix_apply"):
            solver._pcg_apply_cmd.launch()
        with timings.span("dot_reduction"):
            _reduce(
                solver._pcg_compact_p, solver._pcg_compact_ap,
                solver._pcg_p_ap_accum)
        with timings.span("precondition_update"):
            solver._pcg_alpha_cmds[parity].launch()
            solver._pcg_update_cmd.launch()
        with timings.span("dot_reduction"):
            rz_out = solver._pcg_rz_b if parity == 0 else solver._pcg_rz_a
            _reduce(solver._pcg_compact_r, solver._pcg_compact_z, rz_out)
        with timings.span("precondition_update"):
            solver._pcg_beta_cmds[parity].launch()
            solver._pcg_update_p_cmd.launch()
        parity ^= 1
    return work_n


# ---------------------------------------------------------------------------
# JIT warmup (once per process)
# ---------------------------------------------------------------------------

_WARMED = False


def _jit_warmup(device):
    """Absorb one-time Warp JIT/native-module cost on a 1-leaf grid.

    Without this, the first measured cold run would mix kernel compilation
    into the graph-capture segment it is supposed to attribute.
    """
    global _WARMED
    if _WARMED:
        return
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4), voxel_size=0.05,
        leaf_capacity=1, device=device,
    )
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.399, 0.399, 0.399))
    assert grid.finalize_topology() == 1
    solver = PressureSolver(grid, device)
    vel = MacVelocityField(grid)
    vel.u.fill_(1.0)
    voxel_type = wp.array(
        np.where(
            np.arange(grid.voxel_capacity) % 2 == 1,
            np.int32(VOXEL_FLUID), np.int32(0),
        ).astype(np.int32),
        dtype=wp.int32, device=device,
    )
    collision_vel = wp.zeros(grid.voxel_capacity, dtype=wp.vec3, device=device)
    solver.build_rhs(vel, voxel_type, collision_vel, None)
    solver.build_preconditioner(voxel_type)
    solver.solve_pcg(voxel_type, WARMUP_ITERATIONS, 1.0e-6)
    if solver.should_project_pressure():
        solver.project(vel, voxel_type, collision_vel)
    wp.synchronize_device(device)
    solver.release()
    _WARMED = True


# ---------------------------------------------------------------------------
# Case driver
# ---------------------------------------------------------------------------


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _stage_entry(totals_list, stage, variant="fixed"):
    """Mean per-frame totals for a stage across warm buckets."""
    gpu = [t[stage][variant]["gpu_ms"] for t in totals_list if stage in t
           and variant in t[stage]]
    host = [t[stage][variant]["host_ms"] for t in totals_list if stage in t
            and variant in t[stage]]
    return {
        "gpu_ms_mean": _mean(gpu),
        "host_ms_mean": _mean(host),
        "frames": len(gpu),
    }


def _checkpoint_readback_ms(bucket_totals):
    """Warm solve GPU time not attributed to wrapped graph-boundary segments.

    Covers the initialization launches, the initial and per-checkpoint dot /
    packed readbacks (the existing checkpoint sync points), and status
    classification. Derived, never measured per kernel.
    """
    solve = _bucket_stage(bucket_totals, "solve_total")["gpu_ms"]
    attributed = (
        _bucket_stage(bucket_totals, "index_marking")["gpu_ms"]
        + _bucket_stage(bucket_totals, "command_record")["gpu_ms"]
        + sum(
            entry["gpu_ms"]
            for entry in bucket_totals.get("graph_launch_capture", {}).values()
        )
    )
    return max(0.0, solve - attributed)


def _run_case(device, geometry_name, fixture_name):
    """Run one (geometry, fixture) probe case; returns the report dict."""
    fixture = FIXTURES[fixture_name]
    _jit_warmup(device)
    geom = _build_geometry(device, geometry_name)
    solver = PressureSolver(geom.grid, device)
    timings = _StageTimings(device)
    _instrument_solver(solver, timings)

    conditional_supported = bool(wp.is_conditional_graph_supported())
    _restore_conditional = None
    if fixture["force_host_batched"]:
        _restore_conditional = wp.is_conditional_graph_supported
        wp.is_conditional_graph_supported = lambda: False
    try:
        cold_record = _frame_once(solver, timings, geom, fixture)
        cold_totals = _bucket_stage_totals(timings.drain())
        unknown_count = int(solver.pressure_count)

        warm_records = []
        warm_totals = []
        for _ in range(WARM_REPLICATES):
            warm_records.append(_frame_once(solver, timings, geom, fixture))
            warm_totals.append(_bucket_stage_totals(timings.drain()))

        replay_work_n = _attribution_replay(solver, timings, REPLAY_ITERATIONS)
        replay_bucket = timings.drain()
    finally:
        if _restore_conditional is not None:
            wp.is_conditional_graph_supported = _restore_conditional
        native_dot_available = solver._pcg_dot is not None
        graph_counts = {
            "batch_graphs": len(solver._pcg_graphs),
            "conditional_graphs": len(solver._pcg_conditional_graphs),
        }
        solver.release()

    replay_totals = _bucket_stage_totals(replay_bucket)
    replay_stages = {
        stage: {
            "gpu_ms_total": _bucket_stage(replay_totals, stage)["gpu_ms"],
            "calls": _bucket_stage(replay_totals, stage)["calls"],
            "ms_per_iteration": (
                _bucket_stage(replay_totals, stage)["gpu_ms"] / REPLAY_ITERATIONS
            ),
        }
        for stage in ("matrix_apply", "precondition_update", "dot_reduction")
    }

    cold_solve = cold_record
    warm_aggregate = {stage: _stage_entry(warm_totals, stage)
                      for stage in ("assembly", "index_marking",
                                    "preconditioner_build", "command_record",
                                    "solve_total", "projection",
                                    TOTAL_PRESSURE_STAGE, WHOLE_FRAME_STAGE)}
    warm_graph = _stage_entry(warm_totals, "graph_launch_capture", "warm")
    cold_graph = cold_totals.get("graph_launch_capture", {})
    checkpoint_warm = _mean(
        [_checkpoint_readback_ms(t) for t in warm_totals])

    cold_capture = cold_graph.get("cold", {"gpu_ms": 0.0, "host_ms": 0.0,
                                           "calls": 0})
    plan_view = {
        "assembly": warm_aggregate["assembly"],
        "index_marking": warm_aggregate["index_marking"],
        "matrix_apply": {
            "ms_per_iteration": replay_stages["matrix_apply"]["ms_per_iteration"],
            "source": "attribution_replay",
        },
        "precondition_update": {
            "ms_per_iteration": replay_stages["precondition_update"][
                "ms_per_iteration"],
            "preconditioner_build_gpu_ms_mean":
                warm_aggregate["preconditioner_build"]["gpu_ms_mean"],
            "source": "attribution_replay",
        },
        "dot_reduction": {
            "ms_per_iteration": replay_stages["dot_reduction"][
                "ms_per_iteration"],
            "source": "attribution_replay",
        },
        # Cold capture vs warm launch stay separate per the plan.
        "graph_launch_capture": {
            "cold_gpu_ms": cold_capture["gpu_ms"],
            "cold_host_ms": cold_capture["host_ms"],
            "cold_calls": cold_capture["calls"],
            "warm_gpu_ms_mean": warm_graph["gpu_ms_mean"],
            "warm_host_ms_mean": warm_graph["host_ms_mean"],
            "source": "cuda_events_at_graph_boundaries",
        },
        "checkpoint_readback": {
            "warm_gpu_ms_mean": checkpoint_warm,
            "source": "derived_warm_residual_of_solve_total",
        },
        "projection": warm_aggregate["projection"],
        TOTAL_PRESSURE_STAGE: warm_aggregate[TOTAL_PRESSURE_STAGE],
        WHOLE_FRAME_STAGE: warm_aggregate[WHOLE_FRAME_STAGE],
    }

    report = {
        "schema": "fluidsim/pressure_performance_probe/v1",
        "generated_by": "tests/perf/test_pressure_performance_probe.py",
        "device": str(device),
        "warp_version": str(getattr(wp, "__version__", "unknown")),
        "conditional_graph_support": conditional_supported,
        "native_dot_available": native_dot_available,
        "graph_counts": graph_counts,
        "timing_method": {
            "gpu": "wp.Event(enable_timing=True) pairs at graph boundaries",
            "per_kernel_synchronization": False,
            "production_code_modified": False,
            "in_graph_segments": "attribution replay of the recorded commands",
        },
        "stage_definitions": {
            "assembly": "PressureSolver.build_rhs (divergence/RHS build)",
            "index_marking": "PressureSolver._build_pressure_index",
            "matrix_apply": "k_apply_matrix per PCG iteration (replay)",
            "precondition_update": ("alpha + preconditioned x/r update + beta"
                                    " + direction update per iteration"
                                    " (replay)"),
            "dot_reduction": "persistent dot launches per iteration (replay)",
            "graph_launch_capture": ("_launch_pcg_batch/_launch_pcg_conditional"
                                     "/_get_pcg_conditional_graphs; cold ="
                                     " capture, warm = replay"),
            "checkpoint_readback": ("warm solve GPU residual after index"
                                    " marking and graph segments:"
                                    " initialization launches, initial and"
                                    " per-checkpoint dot/readback syncs,"
                                    " classification"),
            "projection": "PressureSolver.project",
            "total_pressure": "build + precondition + solve + project",
            "whole_frame": "total_pressure + divergence binding readback",
        },
        "geometry": {
            "name": geometry_name,
            "voxels_dim": [int(geom.grid.voxels_dim[i]) for i in range(3)],
            "active_leaves": int(geom.grid.active_leaf_count),
            "active_voxels": int(geom.grid.active_voxel_count),
            "voxel_capacity": int(geom.grid.voxel_capacity),
            "expected_unknown_count": geom.expected_unknowns,
        },
        "fixture": dict(fixture, name=fixture_name),
        "cold": {
            "solve": cold_solve,
            "unknown_count": unknown_count,
            "stage_totals_ms": cold_totals,
        },
        "warm": {
            "replicates": [
                {"solve": record, "stage_totals_ms": totals}
                for record, totals in zip(warm_records, warm_totals)
            ],
            "stage_means_ms": warm_aggregate,
        },
        "graph_attribution_replay": {
            "iterations": REPLAY_ITERATIONS,
            "work_n": replay_work_n,
            "stages_ms": replay_stages,
            "note": ("replay launches run uncaptured, so each segment's"
                     " absolute time includes inter-kernel launch gaps and"
                     " is an upper bound; production keeps these segments"
                     " inside one graph (see graph_launch_capture warm)."
                     " Use the per-segment ratios to weight in-graph time."),
        },
        "plan_interface_timings_ms": plan_view,
    }
    _validate_report(report, fixture_name)
    _emit_report(report)
    return report


def _validate_report(report, fixture_name):
    """Structural + binding checks: the probe run must be interpretable."""
    interface = report["plan_interface_timings_ms"]
    for stage in REQUIRED_STAGES + (TOTAL_PRESSURE_STAGE, WHOLE_FRAME_STAGE):
        assert stage in interface, f"missing plan interface stage {stage!r}"

    geometry = report["geometry"]
    assert geometry["active_leaves"] > 0
    assert geometry["active_voxels"] > 0
    cold = report["cold"]
    assert cold["unknown_count"] == geometry["expected_unknown_count"], (
        cold["unknown_count"], geometry["expected_unknown_count"])
    assert cold["solve"]["status"] in _KNOWN_STATUSES, cold["solve"]
    assert cold["solve"]["residual"] >= 0.0 and math.isfinite(
        cold["solve"]["residual"])
    assert 0 <= cold["solve"]["iterations"] <= report["fixture"][
        "max_iterations"]
    assert cold["solve"]["execution_mode"] in ("cuda_conditional",
                                               "host_batched")
    if fixture_name == "host_batched":
        assert cold["solve"]["execution_mode"] == "host_batched"

    warm = report["warm"]
    assert len(warm["replicates"]) == WARM_REPLICATES
    first = warm["replicates"][0]["solve"]
    for replicate in warm["replicates"][1:]:
        solve = replicate["solve"]
        assert solve["status"] == first["status"]
        assert solve["iterations"] == first["iterations"]
        assert solve["residual"] == pytest.approx(first["residual"], rel=1.0e-3)

    assert cold["stage_totals_ms"].get(TOTAL_PRESSURE_STAGE) is not None
    total = cold["stage_totals_ms"][TOTAL_PRESSURE_STAGE]["fixed"]["gpu_ms"]
    assert total > 0.0

    replay = report["graph_attribution_replay"]
    assert replay["work_n"] and replay["work_n"] > 0
    for stage in ("matrix_apply", "precondition_update", "dot_reduction"):
        value = replay["stages_ms"][stage]
        assert math.isfinite(value["ms_per_iteration"])
        assert value["ms_per_iteration"] >= 0.0
    assert replay["stages_ms"]["matrix_apply"]["ms_per_iteration"] > 0.0


def _emit_report(report):
    text = json.dumps(report, indent=2)
    label = f"{report['geometry']['name']}/{report['fixture']['name']}"
    print(f"\n===== pressure performance probe: {label} =====")
    print(text)
    out_dir = ROOT / "artifacts" / "real_world_performance" / "pressure"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (
            f"probe_{report['geometry']['name']}_"
            f"{report['fixture']['name']}.json"
        )
        path.write_text(text, encoding="utf-8")
    except OSError as exc:  # pragma: no cover - read-only checkout
        print(f"[probe] artifact write skipped: {exc}")


# ---------------------------------------------------------------------------
# pytest + direct-python entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", sorted(FIXTURES))
@pytest.mark.parametrize("geometry", GEOMETRY_NAMES)
def test_pressure_probe_stage_report(cuda_device, geometry, fixture_name):
    """Plan Task 1 report test: structured per-stage pressure timings."""
    report = _run_case(cuda_device, geometry, fixture_name)
    interface = report["plan_interface_timings_ms"]
    for stage in REQUIRED_STAGES:
        assert stage in interface
    assert interface[TOTAL_PRESSURE_STAGE]["gpu_ms_mean"] > 0.0
    assert interface[WHOLE_FRAME_STAGE]["host_ms_mean"] > 0.0


def main() -> int:
    """Headless python entry point (no pytest, no Blender)."""
    wp.init()
    device = script_device()
    try:
        wp.get_device(device)
    except Exception as exc:
        print(f"[probe] device {device!r} unavailable: {exc}")
        return 2
    if not wp.get_device(device).is_cuda:
        print(f"[probe] device {device!r} is not CUDA; "
              "the probe requires a GPU (set FLUIDSIM_TEST_DEVICE).")
        return 2

    failures = 0
    for geometry in GEOMETRY_NAMES:
        for fixture_name in sorted(FIXTURES):
            try:
                report = _run_case(device, geometry, fixture_name)
            except AssertionError as exc:
                failures += 1
                print(f"[probe] FAIL {geometry}/{fixture_name}: {exc}")
                continue
            total = report["plan_interface_timings_ms"][TOTAL_PRESSURE_STAGE]
            print(
                f"[probe] ok {geometry}/{fixture_name}: "
                f"total_pressure {total['gpu_ms_mean']:.3f} ms/gpu, "
                f"{total['host_ms_mean']:.3f} ms/host (warm mean)"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
