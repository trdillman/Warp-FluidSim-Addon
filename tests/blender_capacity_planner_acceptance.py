"""Real CUDA acceptance for fail-closed source-capacity transactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluidSim_warp_blender51.sim.capacity_planner import (  # noqa: E402
    CapacityVerdict, DemandQuality, plan_initialization,
)
from fluidSim_warp_blender51.sim.enums import (  # noqa: E402
    SourceEvaluationCadence, SourceLifecycle, SourceOperation,
    SourcePayloadKind,
)
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings  # noqa: E402
from fluidSim_warp_blender51.sim.source_regions import (  # noqa: E402
    PayloadPlan, PointRegionPlan, SourceKey, SourcePlan,
)
from fluidSim_warp_blender51.sim.sourcing import HeightSourceDesc  # noqa: E402
from fluidSim_warp_blender51.sim.enums import SourceEmissionMode  # noqa: E402
from tools.build_demo_scenes import (  # noqa: E402
    fresh_scene, make_domain, make_source_box,
)


def settings(max_particles):
    value = SolverSettings()
    value.domain_min = (-0.6, -0.6, 0.0)
    value.domain_max = (0.6, 0.6, 1.2)
    value.voxel_size = 0.1
    value.leaf_capacity = 64
    value.max_particles = max_particles
    value.max_whitewater = 1
    value.whitewater_enabled = False
    value.surface_enabled = False
    value.min_substeps = 1
    value.max_substeps = 1
    value.device = "cuda:0"
    return value


def direct_plan(key, points):
    return SourcePlan(
        SourceKey(key), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(np.asarray(points, np.float32).reshape((-1, 3))),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("capacity acceptance requires cuda:0")

    cases = []

    # Exact-fit direct append commits all particles.
    exact = FluidSolver(settings(2))
    try:
        exact.set_source_plans((direct_plan(
            "exact", ((-0.1, 0.0, 0.7), (0.1, 0.0, 0.7))),))
        exact.step_frame()
        count = exact.particles.refresh_host_count()
        result = dict(exact.sourcing.last_apply_result)
        assert count == 2 and result["admitted"] is True
        cases.append({"name": "exact_fit", "count": count, "result": result})
    finally:
        exact.release()

    # Two individually valid direct sources exceed the aggregate pool by one.
    aggregate = FluidSolver(settings(3))
    try:
        aggregate.set_source_plans((
            direct_plan("first", ((-0.2, 0.0, 0.7), (-0.1, 0.0, 0.7))),
            direct_plan("second", ((0.1, 0.0, 0.7), (0.2, 0.0, 0.7))),
        ))
        aggregate.step_frame()
        count = aggregate.particles.refresh_host_count()
        result = dict(aggregate.sourcing.last_apply_result)
        assert count == 0
        assert result["admitted"] is False
        assert result["required"] == 4 and result["available"] == 3
        assert aggregate.sourcing._initial_fired == set()
        cases.append({
            "name": "aggregate_over_by_one", "count": count,
            "fired": sorted(aggregate.sourcing._initial_fired), "result": result,
        })
        grown_capacity = aggregate.particles.capacity
        assert grown_capacity > 3
        aggregate.step_frame()
        retry_count = aggregate.particles.refresh_host_count()
        retry_result = dict(aggregate.sourcing.last_apply_result)
        assert retry_count == 4, {
            "retry_count": retry_count,
            "capacity": aggregate.particles.capacity,
            "result": retry_result,
            "fired": sorted(aggregate.sourcing._initial_fired),
        }
        assert retry_result["admitted"] is True, retry_result
        cases.append({
            "name": "aggregate_growth_retry_commits_whole_batch",
            "count": retry_count,
            "capacity": grown_capacity,
            "result": retry_result,
        })
    finally:
        aggregate.release()

    # Legacy height emission must not stamp/emit/consume lifecycle on rejection.
    height = FluidSolver(settings(4))
    try:
        height.set_height_source(HeightSourceDesc(
            enabled=True, base_height=0.4, back_mode=1,
            emit_particles=True,
            emission_mode=SourceEmissionMode.INITIAL_FILL,
        ))
        seed_before = height.sourcing._emit_seed
        height.step_frame()
        count = height.particles.refresh_host_count()
        result = dict(height.sourcing.last_apply_result)
        assert count == 0 and result["admitted"] is False
        assert "builtin:height" not in height.sourcing._initial_fired
        assert height.sourcing._emit_seed == seed_before
        cases.append({
            "name": "height_fail_closed", "count": count,
            "seed_preserved": height.sourcing._emit_seed == seed_before,
            "result": result,
        })
    finally:
        height.release()

    # Pure preflight proves automatic seeds and source demand are aggregated.
    class Footprint:
        stable_id = "compiled-source"
        particle_demand = 3
        particle_quality = DemandQuality.UPPER_BOUND
        leaf_coords = ((0, 0, 0),)
        staging_bytes = 0
        reason = ""

    planner_settings = {
        "domain_min": (0.0, 0.0, 0.0),
        "domain_max": (0.8, 0.8, 0.8),
        "voxel_size": 0.1,
        "max_particles": 4,
        "leaf_capacity": 1,
        "topology_expand": 0,
        "max_whitewater": 0,
        "surface_enabled": False,
    }
    resources = {
        "gpu_total_bytes": int(device.total_memory),
        "gpu_free_bytes": int(device.free_memory),
    }
    plan = plan_initialization(
        planner_settings, (Footprint(),),
        (np.zeros((2, 3), np.float32),), (), resources)
    assert plan.verdict is CapacityVerdict.BLOCKED
    assert plan.particle_demand.value == 5
    assert any(blocker.code == "PARTICLE_CAPACITY" for blocker in plan.blockers)
    cases.append({
        "name": "planner_aggregate", "verdict": plan.verdict.value,
        "required": plan.particle_demand.value,
        "blockers": [blocker.code for blocker in plan.blockers],
    })

    # Add-on UX: failed initialize warns without mutation; explicit approval
    # applies the preserve-simulation recommendation and retries successfully.
    scene = fresh_scene()
    make_domain(scene, (-0.4, -0.4, 0.0), (0.4, 0.4, 0.8),
                "Capacity Approval Domain")
    from fluidSim_warp_blender51.blender import scene_bridge

    ds = scene_bridge.active_domain_settings(scene)
    ds.voxel_size = 0.1
    ds.leaf_capacity = 64
    ds.max_particles = 1
    ds.auto_grow_particles = False
    ds.max_whitewater = 1
    ds.particles_per_voxel = 8
    ds.surface_enabled = False
    ds.whitewater_enabled = False
    make_source_box(
        scene, (-0.3, -0.3, 0.1), (0.3, 0.3, 0.7),
        "Capacity Approval Source", emit_rate=8.0,
        emission_mode="INITIAL_FILL")
    max_before = int(ds.max_particles)
    init_result = set(bpy.ops.fluidsim.initialize())
    addon_plan = scene_bridge.session().capacity_plan
    assert init_result == {"CANCELLED"}
    assert int(ds.max_particles) == max_before
    assert not scene_bridge.session().active
    assert addon_plan.suggestions
    recommended = addon_plan.suggestions[0]
    assert recommended.key == "preserve_simulation"
    required_particles = int(addon_plan.particle_demand.value)
    assert recommended.changes["max_particles"] == required_particles
    approved_result = set(bpy.ops.fluidsim.apply_capacity_suggestion(
        "EXEC_DEFAULT",
        suggestion_key=recommended.key,
        expected_scene_signature=(
            scene_bridge.session().capacity_plan_signature_for_scene(scene)),
    ))
    assert approved_result == {"FINISHED"}
    assert int(ds.max_particles) == required_particles
    assert scene_bridge.session().active
    cases.append({
        "name": "approval_gated_preserve_simulation_retry",
        "initial_result": sorted(init_result),
        "approved_result": sorted(approved_result),
        "before": max_before,
        "after": int(ds.max_particles),
        "required": required_particles,
        "recommended_key": recommended.key,
    })
    scene_bridge.release_session(scene)

    # A Reinitialize warning must recheck its authored-reserve model even when
    # the owned runtime pool has grown far beyond it. Reset is the operation
    # that uses runtime capacity; suggestion approval here recreates the pool.
    ds.max_particles = 1024
    ds.auto_grow_particles = True
    authored_reserve = int(ds.max_particles)
    auto_result = set(bpy.ops.fluidsim.initialize())
    auto_session = scene_bridge.session()
    assert auto_result == {"FINISHED"}
    assert auto_session.active
    assert int(ds.max_particles) == authored_reserve
    growth = auto_session.solver.ensure_particle_capacity(65_536)
    assert growth.status in {"GROWN", "SUFFICIENT"}
    assert auto_session.solver.particles.capacity == 65_536
    assert int(ds.max_particles) == authored_reserve

    ds.auto_grow_particles = False
    blocked_result = set(bpy.ops.fluidsim.initialize())
    blocked_plan = auto_session.capacity_plan_for_scene(scene)
    assert blocked_result == {"CANCELLED"}
    assert auto_session.active
    assert auto_session.solver.particles.capacity == 65_536
    assert auto_session.capacity_plan_operation == "REINITIALIZE"
    assert any(
        blocker.code == "PARTICLE_CAPACITY"
        for blocker in blocked_plan.blockers)
    assert blocked_plan.suggestions
    blocked_signature = auto_session.capacity_plan_signature_for_scene(scene)
    refreshed_plan = scene_bridge.refresh_capacity_plan(scene)
    assert auto_session.capacity_plan_signature_for_scene(scene) == blocked_signature
    assert any(
        blocker.code == "PARTICLE_CAPACITY"
        for blocker in refreshed_plan.blockers)
    assert refreshed_plan.suggestions
    recommended = refreshed_plan.suggestions[0]
    assert recommended.key == "preserve_simulation"
    approved_result = set(bpy.ops.fluidsim.apply_capacity_suggestion(
        "EXEC_DEFAULT",
        suggestion_key=recommended.key,
        expected_scene_signature=blocked_signature,
    ))
    assert approved_result == {"FINISHED"}
    assert int(ds.max_particles) == int(refreshed_plan.particle_demand.value)
    cases.append({
        "name": "grown_pool_reinitialize_recheck_uses_authored_reserve",
        "initial_operator": sorted(auto_result),
        "blocked_operator": sorted(blocked_result),
        "approved_operator": sorted(approved_result),
        "authored_reserve": authored_reserve,
        "grown_runtime_capacity": 65_536,
        "required": int(refreshed_plan.particle_demand.value),
        "recommended_key": recommended.key,
        "signature_stable": True,
    })
    scene_bridge.release_session(scene)

    # A live direct append grows the active pool and commits the whole batch.
    runtime_settings = settings(2)
    runtime_settings.auto_grow_particles = True
    runtime = FluidSolver(runtime_settings)
    try:
        points = np.asarray((
            (-0.2, 0.0, 0.7), (0.0, 0.0, 0.7), (0.2, 0.0, 0.7),
        ), np.float32)
        injection = runtime.inject_particles(points)
        assert injection.admitted and injection.emitted == 3
        assert runtime.particles.refresh_host_count() == 3
        assert runtime.particles.capacity > 2
        cases.append({
            "name": "live_direct_append_auto_growth",
            "emitted": injection.emitted,
            "runtime_capacity": runtime.particles.capacity,
            "growth_status": runtime.last_particle_growth.status,
        })
    finally:
        runtime.release()

    report = {
        "schema": "fluidsim-capacity-acceptance-v1",
        "passed": True,
        "device": str(device),
        "cases": cases,
        "truth": {
            "partial_emission": False,
            "oom_guarantee_claimed": False,
            "settings_mutated_before_approval": False,
            "settings_mutated_after_explicit_approval": True,
        },
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print("CAPACITY-PLANNER-ACCEPTANCE-PASS")
    print(f"CAPACITY-PLANNER-REPORT {output}")


if __name__ == "__main__":
    main()
