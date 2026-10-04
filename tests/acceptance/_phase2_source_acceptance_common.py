"""Shared quantitative Blender Phase 2 source acceptance implementation."""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from _test_device import script_device  # noqa: E402
for _name in list(sys.modules):
    if _name == "fluidSim_warp_blender51" or _name.startswith("fluidSim_warp_blender51."):
        del sys.modules[_name]

import bpy
import numpy as np
import warp as wp
from mathutils import Matrix

import fluidSim_warp_blender51 as addon
from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs
from fluidSim_warp_blender51.sim.enums import *
from fluidSim_warp_blender51.sim.source_regions import *
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings
from tools.gate.validate_source_acceptance_report import validate

MARKERS = {
    "source_region": "SOURCE-REGION-ACCEPTANCE-PASS",
    "point_source": "POINT-SOURCE-ACCEPTANCE-PASS",
    "source_lifecycle": "SOURCE-LIFECYCLE-ACCEPTANCE-PASS",
    "animated_source": "ANIMATED-SOURCE-ACCEPTANCE-PASS",
    "volume_source": "VOLUME-SOURCE-ACCEPTANCE-PASS",
}


def cube(open_top=False):
    vertices = np.array([
        (-.5, -.5, -.5), (.5, -.5, -.5), (.5, .5, -.5), (-.5, .5, -.5),
        (-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5),
    ], np.float32) + 1.0
    triangles = np.array([
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ], np.int32)
    return vertices, triangles[:-2] if open_top else triangles


def context(frame=1):
    return CompileContext(frame, 0, 1, script_device(), (0, 0, 0), .25,
                          (0, 0, 0), (2, 2, 2), upload=True)


def source_plan(region, key="fixture", cadence=SourceEvaluationCadence.STATIC,
                lifecycle=SourceLifecycle.TOP_UP, payload=None):
    payload = payload or PayloadPlan(
        kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
        velocity=(0, 0, 0), temperature=4, viscosity=5, density=6,
        lifespan=7, use_lifespan=True,
    )
    return SourcePlan(SourceKey(key), SourceOperation.LIQUID, lifecycle,
                      cadence, region, payload)


def compile_timed(compiler, plan, ctx):
    started = time.perf_counter()
    batch = compiler.compile((plan,), ctx)
    wp.synchronize_device(ctx.device)
    return batch, (time.perf_counter() - started) * 1000.0


def settings(substeps=1, max_particles=20_000):
    value = SolverSettings()
    value.domain_min = (0.0, 0.0, 0.0)
    value.domain_max = (2.0, 2.0, 2.0)
    value.voxel_size = 0.25
    value.max_particles = max_particles
    value.surface_enabled = False
    value.whitewater_enabled = False
    value.min_substeps = substeps
    value.max_substeps = substeps
    value.pressure_iterations = 30
    value.gravity = (0.0, 0.0, 0.0)
    value.device = script_device()
    return value


def step_solver(plans, frames=1, *, substeps=1, max_particles=20_000):
    solver = FluidSolver(settings(substeps=substeps, max_particles=max_particles))
    solver.set_source_batch(native_plans=tuple(plans))
    samples = []
    stats = None
    for _ in range(frames):
        stats = solver.step_frame()
        samples.append(float(stats.step_time * 1000.0))
    return solver, stats, samples


def save_fixture(capability):
    """Persist the exact factory-startup acceptance scene reproducibly."""
    fixture_dir = ROOT / "artifacts" / "phase2_fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    path = fixture_dir / f"{capability}.blend"
    obj = bpy.data.objects.get("Phase2AcceptanceFixture")
    if obj is None:
        obj = bpy.data.objects.new("Phase2AcceptanceFixture", None)
        bpy.context.scene.collection.objects.link(obj)
    obj["fluidsim_phase2_capability"] = capability
    bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    return str(path.resolve())


def point_attribute_checks(solver, expected):
    count = solver.particles.host_count
    temperature = solver.particles.attr("temperature").numpy()[:count]
    order = np.argsort(temperature)
    velocity = solver.particles.velocity.numpy()[:count][order]
    checks = {
        "velocity": bool(np.allclose(
            velocity, expected.velocities[np.argsort(expected.temperature)], atol=0.15)),
        "temperature": bool(np.allclose(
            temperature[order], np.sort(expected.temperature), atol=1.0e-5)),
        "viscosity": bool(np.allclose(
            solver.particles.attr("viscosity").numpy()[:count][order],
            expected.viscosity[np.argsort(expected.temperature)], atol=1.0e-5)),
        "density": bool(np.allclose(
            solver.particles.attr("density").numpy()[:count][order],
            expected.density[np.argsort(expected.temperature)], atol=1.0e-5)),
        "lifespan": bool(np.allclose(
            solver.particles.attr("life").numpy()[:count][order],
            expected.lifespan[np.argsort(expected.temperature)], atol=1.0e-5)),
    }
    return checks


def main(capability):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)

    fixture_path = save_fixture(capability)
    wp.init()
    compiler = SourceCompiler()
    compile_samples = []
    attribute_checks = None
    details = {}

    if capability == "point_source":
        point_region = PointRegionPlan(
            [[.5, .5, .5], [1.5, 1.5, 1.5]],
            velocities=[[1, 0, 0], [0, 1, 0]], temperature=[2, 3],
            viscosity=[4, 5], density=[6, 7], lifespan=[8, 9],
        )
        plan = source_plan(
            point_region, lifecycle=SourceLifecycle.DIRECT_APPEND,
            payload=PayloadPlan(
                kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
                velocity=(-1, -1, -1), temperature=-1, viscosity=-1,
                density=-1, lifespan=-1, use_lifespan=True),
        )
    elif capability == "source_lifecycle":
        plan = source_plan(
            AnalyticRegionPlan(shape=SourceShapeShape.BOX, position=(1, 1, 1),
                               half_extents=(.5, .5, .5)),
            lifecycle=SourceLifecycle.ONCE)
    elif capability == "animated_source":
        plan = source_plan(
            AnalyticRegionPlan(shape=SourceShapeShape.SPHERE,
                               position=(.65, 1, 1), radius=.35),
            cadence=SourceEvaluationCadence.EVERY_SIM_FRAME)
    elif capability == "volume_source":
        import openvdb
        volume_path = str((ROOT / "artifacts" / "phase2_fixtures" /
                           "volume_source.vdb").resolve())
        grids = []
        for name in ("density", "surface", "temperature", "vel_x", "vel_y", "vel_z"):
            grid = openvdb.FloatGrid(); grid.name = name
            accessor = grid.getAccessor()
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if name == "density": value = 1.0
                        elif name == "surface": value = float(x) - 2.0
                        elif name == "temperature": value = float(x + y + z)
                        elif name == "vel_x": value = float(x)
                        elif name == "vel_y": value = float(y * 2)
                        else: value = float(z * 3)
                        accessor.setValueOn((x, y, z), value)
            grids.append(grid)
        vector = openvdb.Vec3SGrid(); vector.name = "velocity"
        vector_accessor = vector.getAccessor()
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    vector_accessor.setValueOn((x, y, z), (float(x), float(y*2), float(z*3)))
        grids.append(vector)
        openvdb.write(volume_path, grids=grids)
        transform = tuple(np.array(((.25,0,0,.25),(0,.25,0,.25),
                                    (0,0,.25,.25),(0,0,0,1)), np.float64).reshape(-1))
        plan = source_plan(VolumeRegionPlan(
            volume_path, mask_grid="density", sdf_grid="surface",
            scalar_grid="temperature", vector_grid="velocity",
            transform=transform))
    else:
        vertices, triangles = cube()
        plan = source_plan(MeshRegionPlan(vertices, triangles, solid=True))

    for frame in (1, 2, 3):
        batch, elapsed = compile_timed(compiler, plan, context(frame))
        compile_samples.append(elapsed)
    region = batch.sources[0].region

    if capability == "animated_source":
        scene = bpy.context.scene
        source_object = bpy.data.objects.new("AnimatedSourceAdapterProbe", None)
        scene.collection.objects.link(source_object)
        source_object.matrix_world = Matrix.Translation((.65, 1.0, 1.0))
        inputs = SceneInputs(); scene.frame_set(1)
        inputs.object_velocity(source_object, scene)
        stable_key = (f"session:{inputs._session_generation}:object:"
                      f"{source_object.as_pointer()}")
        adapter_plan = source_plan(
            AnalyticRegionPlan(shape=SourceShapeShape.SPHERE,
                               position=tuple(source_object.location), radius=.35),
            key=stable_key, cadence=SourceEvaluationCadence.EVERY_SIM_FRAME)
        solver, stats, step_samples = step_solver((adapter_plan,), frames=1)
        first_count = solver.particles.host_count
        source_object.matrix_world = Matrix.Translation((1.35, 1.0, 1.0))
        scene.frame_set(2)
        inherited_velocity = inputs.object_velocity(source_object, scene)
        moved = source_plan(
            AnalyticRegionPlan(shape=SourceShapeShape.SPHERE,
                               position=tuple(source_object.location), radius=.35),
            key=stable_key, cadence=SourceEvaluationCadence.EVERY_SIM_FRAME,
            payload=PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
                                velocity=inherited_velocity))
        solver.set_source_batch(native_plans=(moved,))
        stats = solver.step_frame()
        step_samples.append(float(stats.step_time * 1000.0))
        emitted_particles = solver.particles.host_count
        positions = solver.particles.position.numpy()[:emitted_particles]
        velocities = solver.particles.velocity.numpy()[:emitted_particles]
        moved_mask = positions[:, 0] > 1.0
        details.update(
            first_count=first_count, moved_count=emitted_particles,
            compiler_rebuilds=solver.sourcing.compiler.compile_count,
            adapter_stable_key=stable_key,
            inherited_velocity=[float(v) for v in inherited_velocity],
            moved_region_particles=int(np.count_nonzero(moved_mask)),
            moved_region_mean_x=float(np.mean(positions[moved_mask, 0]))
                if np.any(moved_mask) else 0.0,
            inherited_velocity_max_x=float(np.max(velocities[moved_mask, 0]))
                if np.any(moved_mask) else 0.0)
    elif capability == "source_lifecycle":
        solver, stats, step_samples = step_solver((plan,), frames=1)
        first_count = solver.particles.host_count
        first_emitted = solver.sourcing.last_apply_result["emitted"]
        stats = solver.step_frame()
        step_samples.append(float(stats.step_time * 1000.0))
        second_emitted = solver.sourcing.last_apply_result["emitted"]
        emitted_particles = first_count
        solver_four, _, _ = step_solver((plan,), frames=1, substeps=4)
        four_substep_count = solver_four.particles.host_count
        solver.reset(); solver.set_source_batch(native_plans=(plan,)); solver.step_frame()
        reset_count = solver.particles.host_count
        reopened, _, _ = step_solver((plan,), frames=1)
        reopened_count = reopened.particles.host_count
        scene_inputs = SceneInputs()
        identity_obj = bpy.data.objects.new("LifecycleIdentityProbe", None)
        bpy.context.scene.collection.objects.link(identity_obj)
        pointer = identity_obj.as_pointer()
        rename_key = f"session:{scene_inputs._session_generation}:object:{pointer}"
        identity_obj.name = "LifecycleIdentityProbeRenamed"
        renamed_key = f"session:{scene_inputs._session_generation}:object:{identity_obj.as_pointer()}"
        duplicate = identity_obj.copy(); bpy.context.scene.collection.objects.link(duplicate)
        duplicate_key = f"session:{scene_inputs._session_generation}:object:{duplicate.as_pointer()}"
        details.update(first_emitted=first_emitted,
                       second_emitted=second_emitted,
                       fired=sorted(solver.sourcing._initial_fired),
                       one_substep_count=first_count,
                       four_substep_count=four_substep_count,
                       reset_rearmed_count=reset_count,
                       reopen_fresh_count=reopened_count,
                       rename_key_stable=rename_key == renamed_key,
                       duplicate_key_distinct=rename_key != duplicate_key)
        solver_four.release(); reopened.release()
    else:
        solver, stats, step_samples = step_solver((plan,), frames=1)
        emitted_particles = solver.particles.host_count
        if capability == "point_source":
            attribute_checks = point_attribute_checks(solver, point_region)
            details["point_count"] = emitted_particles

    if capability == "source_region":
        pressure_plan = SourcePlan(
            SourceKey("pressure-runtime"), SourceOperation.PRESSURE,
            SourceLifecycle.EVERY_FRAME, SourceEvaluationCadence.STATIC,
            AnalyticRegionPlan(shape=SourceShapeShape.BOX, position=(1, 1, 1),
                               half_extents=(.5, .5, .5)),
            PayloadPlan(kind=SourcePayloadKind.PRESSURE, pressure=5.0))
        pressure_seed = source_plan(
            AnalyticRegionPlan(shape=SourceShapeShape.BOX, position=(1,1,1),
                               half_extents=(.5,.5,.5)),
            key="pressure-liquid-seed", lifecycle=SourceLifecycle.ONCE)
        pressure_solver, _, _ = step_solver((pressure_seed,), frames=1)
        control_solver, _, _ = step_solver((pressure_seed,), frames=1)
        pressure_particle_count = pressure_solver.particles.host_count
        pressure_solver.set_source_batch(native_plans=(pressure_plan,))
        control_solver.set_source_batch()
        pressure_solver.step_frame()
        control_solver.step_frame()
        pressure_velocity_max = max(
            float(np.max(np.abs(pressure_solver.vel.u.numpy()))),
            float(np.max(np.abs(pressure_solver.vel.v.numpy()))),
            float(np.max(np.abs(pressure_solver.vel.w.numpy()))))
        control_velocity_max = max(
            float(np.max(np.abs(control_solver.vel.u.numpy()))),
            float(np.max(np.abs(control_solver.vel.v.numpy()))),
            float(np.max(np.abs(control_solver.vel.w.numpy()))))
        details["pressure_influence"] = {
            "emitted_particles": pressure_solver.particles.host_count - pressure_particle_count,
            "applied_voxels": pressure_solver.sourcing.last_pressure_apply_count,
            "peak_pressure": float(np.max(np.abs(
                pressure_solver.pressure_solver.pressure.numpy()))),
            "projected_velocity_max": pressure_velocity_max,
            "control_velocity_max": control_velocity_max,
            "projected_velocity_delta": pressure_velocity_max - control_velocity_max,
        }
        pressure_solver.release(); control_solver.release()

        staggered_plan = SourcePlan(
            SourceKey("staggered-runtime"), SourceOperation.STAGGERED_VELOCITY,
            SourceLifecycle.EVERY_FRAME, SourceEvaluationCadence.STATIC,
            AnalyticRegionPlan(shape=SourceShapeShape.BOX, position=(1, 1, 1),
                               half_extents=(.5, .5, .5)),
            PayloadPlan(kind=SourcePayloadKind.STAGGERED_VECTOR,
                        vector=(1.0, 2.0, 3.0)))
        staggered_solver, _, _ = step_solver((staggered_plan,), frames=1)
        details["staggered_velocity"] = {
            "emitted_particles": staggered_solver.particles.host_count,
            "applied_voxels": staggered_solver.sourcing.last_post_p2g_apply_count,
            "face_component_max": [
                float(np.max(np.abs(staggered_solver.vel.u.numpy()))),
                float(np.max(np.abs(staggered_solver.vel.v.numpy()))),
                float(np.max(np.abs(staggered_solver.vel.w.numpy()))),
            ],
        }
        staggered_solver.release()

        shell_requested = .30
        shell_plan = source_plan(MeshRegionPlan(
            *cube(True), solid=False, shell_half_width=shell_requested),
            key="shell-acceptance")
        shell_batch = compiler.compile((shell_plan,), context())
        shell_region = shell_batch.sources[0].region
        shell_distances = shell_region.sdf[shell_region.coverage > 0]
        effective_width = max(shell_requested, context().voxel_size * .5)
        analytic_plan = source_plan(AnalyticRegionPlan(
            shape=SourceShapeShape.BOX, position=(1,1,1),
            half_extents=(.5,.5,.5)), key="analytic-reference")
        analytic_batch = compiler.compile((analytic_plan,), context())
        details["surface_shell"] = {
            "requested_half_width": shell_requested,
            "voxel_size": context().voxel_size,
            "policy": "effective half-width is max(requested, 0.5 * voxel_size)",
            "effective_half_width": effective_width,
            "measured_max_center_distance": float(np.max(shell_distances)),
            "covered_voxels": int(shell_region.covered_voxel_count),
            "open_mesh_accepted": not any(
                d.severity == "ERROR" for d in shell_batch.diagnostics),
        }
        mesh_covered = int(region.covered_voxel_count)
        analytic_covered = int(analytic_batch.sources[0].region.covered_voxel_count)
        details["analytic_mesh_reference"] = {
            "mesh_covered_voxels": mesh_covered,
            "analytic_covered_voxels": analytic_covered,
            "absolute_difference": abs(mesh_covered - analytic_covered),
        }

    if capability == "volume_source":
        component_plan = SourcePlan(
            SourceKey("volume-components"), SourceOperation.STAGGERED_VELOCITY,
            SourceLifecycle.EVERY_FRAME, SourceEvaluationCadence.STATIC,
            VolumeRegionPlan(
                volume_path, mask_grid="density", vector_x_grid="vel_x",
                vector_y_grid="vel_y", vector_z_grid="vel_z",
                transform=transform),
            PayloadPlan(kind=SourcePayloadKind.STAGGERED_VECTOR))
        component_batch = compiler.compile((component_plan,), context())
        component_region = component_batch.sources[0].region
        details["bundled_openvdb"] = {
            "module_path": str(Path(openvdb.__file__).resolve()),
            "fog_grid": bool(region.covered_voxel_count > 0),
            "sdf_grid": bool(region.sdf is not None and np.isfinite(region.sdf).any()),
            "scalar_grid": bool(region.scalar is not None and np.max(region.scalar) > 0),
            "vector_grid": bool(region.vector is not None and np.max(region.vector) > 0),
            "vector_components": bool(component_region.vector is not None
                and np.max(component_region.vector) > 0),
            "staggered_components": all(value is not None for value in (
                component_region.staggered_u, component_region.staggered_v,
                component_region.staggered_w)),
            "non_identity_transform": transform != VolumeRegionPlan(volume_path, mask_grid="density").transform,
            "transformed_bounds": [list(v) for v in plan.region.bounds()],
            "staging_path": "bundled openvdb copyToArray",
        }

    capacity_context = CompileContext(
        1,0,1,"cpu",(0,0,0),.05,(0,0,0),(2,2,2),
        leaf_capacity=1, max_source_bytes=1024, upload=False)
    capacity_batch = SourceCompiler().compile((source_plan(
        AnalyticRegionPlan(shape=SourceShapeShape.BOX, position=(1,1,1),
                           half_extents=(1,1,1)), "capacity"),), capacity_context)
    capacity_errors = [d for d in capacity_batch.diagnostics if d.severity == "ERROR"]
    capacity_solver, _, _ = step_solver((source_plan(
        AnalyticRegionPlan(shape=SourceShapeShape.BOX, position=(1,1,1),
                           half_extents=(1,1,1)), "capacity-runtime"),),
        frames=1, max_particles=32)
    details["capacity"] = {
        "diagnostic_codes": [d.code for d in capacity_errors],
        "diagnostic_messages": [d.message for d in capacity_errors],
        "emitted_particles": int(capacity_solver.particles.host_count),
        "capacity_limited": bool(capacity_solver.sourcing.last_apply_result["capacity_limited"]),
        "particle_capacity": int(capacity_solver.particles.capacity),
    }
    capacity_solver.release()

    vertices, open_triangles = cube(True)
    if capability == "source_region":
        invalid = compiler.compile((source_plan(
            MeshRegionPlan(vertices, open_triangles, solid=True), "invalid"),), context())
        invalid_case = "open_mesh_as_solid"
    elif capability == "volume_source":
        invalid = compiler.compile((source_plan(VolumeRegionPlan(
            volume_path, mask_grid="missing-grid"), "invalid-volume"),), context())
        invalid_case = "missing_named_volume_grid"
    elif capability == "point_source":
        try:
            PointRegionPlan([[0,0,0]], velocities=[[1,0,0],[2,0,0]])
        except ValueError as exc:
            invalid = None; rejected = True; invalid_message = str(exc)
        invalid_case = "point_attribute_length_mismatch"
    elif capability == "source_lifecycle":
        try:
            source_plan(AnalyticRegionPlan(), key="")
        except ValueError as exc:
            invalid = None; rejected = True; invalid_message = str(exc)
        invalid_case = "empty_runtime_source_identity"
    else:
        huge_triangles = np.tile(cube()[1], (4167, 1))[:50000]
        invalid = compiler.compile((source_plan(MeshRegionPlan(
            cube()[0], huge_triangles, solid=False, shell_half_width=.1),
            "invalid-animated", cadence=SourceEvaluationCadence.EVERY_SIM_FRAME),),
            CompileContext(1,0,1,"cpu",(0,0,0),.02,(0,0,0),(2,2,2),upload=False))
        invalid_case = "animated_mesh_work_exceeds_bound"
    if invalid is not None:
        rejected = any(d.severity == "ERROR" for d in invalid.diagnostics)
        invalid_message = next((d.message for d in invalid.diagnostics
                                if d.severity == "ERROR"), "")
    coverage_ratio = float(np.mean(region.coverage)) if region.coverage.size else 0.0

    invariants = [
        {"id": "flat_8x8x8_leaf_storage",
         "passed": region.coverage.shape[1:] == (512,),
         "expected": "512 voxels per flat leaf", "actual": str(region.coverage.shape)},
        {"id": "capability_invalid_input_rejected", "passed": rejected,
         "expected": f"{invalid_case} rejected with actionable message",
         "actual": invalid_message},
        {"id": "real_solver_advanced", "passed": solver.frame >= 1,
         "expected": "FluidSolver frame >= 1", "actual": f"frame={solver.frame}"},
        {"id": "real_liquid_emitted", "passed": emitted_particles > 0,
         "expected": "nonzero live particles", "actual": str(emitted_particles)},
        {"id": "capacity_limit_visible",
         "passed": bool(details["capacity"]["diagnostic_codes"])
             and details["capacity"]["capacity_limited"]
             and details["capacity"]["emitted_particles"]
                 <= details["capacity"]["particle_capacity"],
         "expected": "preflight diagnostic plus bounded partial runtime fill",
         "actual": str(details["capacity"])},
    ]
    if capability == "source_region":
        pressure_measure = details["pressure_influence"]
        staggered_measure = details["staggered_velocity"]
        invariants.extend([
            {"id": "pressure_field_nonparticle_influence",
             "passed": pressure_measure["emitted_particles"] == 0
             and pressure_measure["applied_voxels"] > 0
             and pressure_measure["peak_pressure"] >= 4.99
             and pressure_measure["projected_velocity_delta"] > 1.0e-4,
             "expected": "zero particles and projected velocity above a no-pressure control",
             "actual": str(pressure_measure)},
            {"id": "surface_shell_half_width_quantitative",
             "passed": details["surface_shell"]["open_mesh_accepted"]
             and details["surface_shell"]["covered_voxels"] > 0
             and details["surface_shell"]["measured_max_center_distance"]
                 <= details["surface_shell"]["effective_half_width"] + 1.0e-5
             and details["surface_shell"]["measured_max_center_distance"]
                 >= details["surface_shell"]["effective_half_width"]
                    - 0.5 * details["surface_shell"]["voxel_size"] - 1.0e-5,
             "expected": "open shell reaches the requested band within half-voxel sampling and never exceeds it",
             "actual": str(details["surface_shell"])},
            {"id": "mesh_solid_matches_analytic_reference",
             "passed": details["analytic_mesh_reference"]["absolute_difference"] == 0,
             "expected": "equal covered voxel count", "actual": str(details["analytic_mesh_reference"])},
            {"id": "staggered_face_components_post_p2g",
             "passed": staggered_measure["emitted_particles"] == 0
             and staggered_measure["applied_voxels"] > 0
             and np.allclose(staggered_measure["face_component_max"],
                             (1.0, 2.0, 3.0), atol=.02),
             "expected": "independent 1/2/3 MAC face components after P2G",
             "actual": str(staggered_measure)},
        ])
    if capability == "point_source":
        invariants.append({
            "id": "point_attributes_exact", "passed": all(attribute_checks.values()),
            "expected": "all owned point arrays preserved", "actual": str(attribute_checks)})
    if capability == "source_lifecycle":
        invariants.extend([{
            "id": "initial_fill_once", "passed": details["first_emitted"] > 0
            and details["second_emitted"] == 0,
            "expected": "first emission nonzero; second zero", "actual": str(details)},
            {"id": "compiled_substep_invariant",
             "passed": details["one_substep_count"] == details["four_substep_count"],
             "expected": "equal initial fill at 1 and 4 fixed substeps", "actual": str(details)},
            {"id": "identity_reset_reopen_semantics",
             "passed": details["rename_key_stable"] and details["duplicate_key_distinct"]
             and details["reset_rearmed_count"] > 0 and details["reopen_fresh_count"] > 0,
             "expected": "rename stable, duplicate distinct, reset/reopen rearm",
             "actual": str(details)}])
    if capability == "animated_source":
        invariants.extend([{
            "id": "animated_plan_recompiled", "passed": details["compiler_rebuilds"] >= 2,
            "expected": "at least two geometry compiles", "actual": str(details)},
            {"id": "animated_spatial_displacement",
             "passed": details["moved_region_particles"] > 0
             and details["moved_region_mean_x"] > 1.0,
             "expected": "particles occupy moved source region", "actual": str(details)},
            {"id": "adapter_object_velocity_inherited",
             "passed": details["inherited_velocity"][0] > 0.0
             and details["inherited_velocity_max_x"] > 0.0,
             "expected": "Blender SceneInputs motion produces positive emitted x velocity",
             "actual": str(details)}])
    if capability == "volume_source":
        volume_checks = details["bundled_openvdb"]
        invariants.append({
            "id": "phase2v_all_grid_roles_and_transform",
            "passed": all(volume_checks[key] for key in (
                "fog_grid", "sdf_grid", "scalar_grid", "vector_grid",
                "vector_components", "staggered_components",
                "non_identity_transform")),
            "expected": "bundled OpenVDB covers fog/SDF/scalar/vector/components/staggered/transform",
            "actual": str(volume_checks)})

    identity = {
        "blender_version": bpy.app.version_string,
        "blender_hash": (bpy.app.build_hash.decode() if isinstance(
            bpy.app.build_hash, bytes) else str(bpy.app.build_hash)),
        "addon_module": addon.__name__,
        "addon_version": [int(v) for v in importlib.import_module(
            addon.__name__ + ".version").VERSION],
        "addon_path": str(Path(addon.__file__).resolve()),
        "device": script_device(), "device_name": wp.get_device(script_device()).name,
    }
    runtime_errors = [d.message for d in batch.diagnostics if d.severity == "ERROR"]
    report = {
        "schema": "fluidsim-source-acceptance-v1",
        "ok": all(item["passed"] for item in invariants) and not runtime_errors,
        "capability": capability, "identity": identity,
        "fixture": {"name": capability, "blend_path": fixture_path},
        "measurements": {
            "active_voxels": int(stats.active_voxels),
            "emitted_particles": int(emitted_particles),
            "coverage_ratio": coverage_ratio,
            "attribute_checks": attribute_checks,
            "details": details,
        },
        "invalid_input": {"case": invalid_case, "rejected": rejected,
                          "message": invalid_message},
        "timings_ms": {
            "compile": float(compile_samples[0]),
            "step_median": float(statistics.median(step_samples)),
            "step_p95": float(max(step_samples)),
        },
        "invariants": invariants, "errors": runtime_errors,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf8")
    errors = validate(report)
    solver.release()
    compiler.release()
    if errors:
        raise RuntimeError("; ".join(errors))
    print(MARKERS[capability])
