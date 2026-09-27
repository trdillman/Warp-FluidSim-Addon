"""Authoritative Blender/CUDA acceptance for Simulation Admission."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_demo_scenes import fresh_scene, make_particle_seed


def check(label, condition, detail=""):
    if not condition:
        raise RuntimeError(f"{label}: {detail}")


def main():
    scene = fresh_scene()
    fs = scene.fluidsim
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.8))
    bpy.context.object.name = "Admission Setup Source"
    check("setup", bpy.ops.fluidsim.setup_scene() == {"FINISHED"})
    from fluidSim_warp_blender51.blender.scene_bridge import active_domain_settings

    domain = fs.domain_object
    check("authoritative domain present", domain is not None)
    check(
        "authoritative domain settings registered",
        hasattr(domain, "fluidsim_domain"),
    )
    settings = active_domain_settings(scene)
    check(
        "authoritative domain settings active",
        settings is not None,
    )
    # Guided Setup applies its product preset; admission-specific authored
    # values belong after that transaction and before snapshot capture.
    settings.voxel_size = 0.15
    settings.max_particles = 20_000
    settings.max_whitewater = 1
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    fs.show_particles = True
    # Reproduce a legacy scene authored with the unsafe independent wall
    # switch disabled. Admission must normalize the runtime without mutating
    # the saved scene during preflight.
    settings.open_boundaries = False
    settings.domain_sdf_walls = False
    bpy.ops.mesh.primitive_cube_add(
        size=0.5, location=(0.0, 0.0, -0.25))
    bpy.context.object.name = "Admission Collider"
    check(
        "tag collider",
        bpy.ops.fluidsim.tag_collider() == {"FINISHED"},
    )
    seed = make_particle_seed(
        scene, ((0.0, 0.0, 0.8),), "Admission Seed")
    seed.fluidsim.enabled = True
    seed.fluidsim.auto_inject_seed = True

    # A Blender-authored analytic force and height source must be present in
    # the same admission capture as the setup source/collider and seed.
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 1.0), scale=(0.2,) * 3)
    force = bpy.context.object
    force.name = "Admission Force"
    force.fluidsim.role = "FORCE"
    force.fluidsim.force_type = "DIRECTIONAL"
    force.fluidsim.force_strength = 0.1
    settings.height_enabled = True
    settings.height_base = 0.2
    check("add custom effect",
          bpy.ops.fluidsim.effect_add(
              effect_id="fluidsim.vortex_force") == {"FINISHED"})
    check(
        "compile and apply custom effect",
        bpy.ops.fluidsim.effect_apply_stack(device="cuda:0") == {"FINISHED"},
    )

    from fluidSim_warp_blender51.blender import simulation_admission, sources
    from fluidSim_warp_blender51.blender.scene_bridge import session
    from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs
    from fluidSim_warp_blender51.sim import memory_safety
    from fluidSim_warp_blender51.sim.enums import SourceShapeSourceMode

    owner_type = simulation_admission.SimulationAdmission
    original_preflight = owner_type._preflight
    original_allocate = owner_type._allocate_solver
    original_configure = owner_type._configure_candidate
    original_seeds = sources.commit_prepared_seeds
    original_outputs = owner_type._prepare_outputs
    original_seed_gather = sources.prepare_auto_seeds
    original_release_cuda_memory = memory_safety.release_cuda_memory
    gather_methods = (
        "gather_sources", "gather_source_plans", "gather_colliders",
        "gather_force_fields", "gather_height_source",
    )
    original_gatherers = {
        name: getattr(SceneInputs, name) for name in gather_methods}
    installed = {}
    gathered_depsgraphs = []
    materialized_meshes = []
    cleanup_flushes = []

    def recording_gatherer(name, original):
        def gather(owner, scene_arg, depsgraph=None, *args, **kwargs):
            gathered_depsgraphs.append((name, depsgraph))
            return original(
                owner, scene_arg, depsgraph, *args, **kwargs)
        return gather

    for name, original in original_gatherers.items():
        setattr(SceneInputs, name, recording_gatherer(name, original))

    def gather_seeds(scene_arg, depsgraph=None):
        gathered_depsgraphs.append(("seeds", depsgraph))
        return original_seed_gather(scene_arg, depsgraph)

    sources.prepare_auto_seeds = gather_seeds

    def release_cuda_memory(device):
        cleanup_flushes.append(str(device))
        return original_release_cuda_memory(device)

    memory_safety.release_cuda_memory = release_cuda_memory

    def observe_configuration(owner, candidate, snapshot, scene_arg):
        installed.update({
            "sources": len(snapshot.descriptors) + len(snapshot.extra_plans),
            "colliders": len(snapshot.colliders),
            "forces": len(snapshot.force_fields),
            "analytic_forces": sum(
                getattr(descriptor, "mode", None)
                == SourceShapeSourceMode.FIELD
                for descriptor in snapshot.descriptors),
            "height": bool(getattr(snapshot.height_source, "enabled", False)),
            "seeds": len(snapshot.seeds),
            "effects": len(snapshot.effects),
        })
        result = original_configure(owner, candidate, snapshot, scene_arg)
        materialized_meshes.append(sum(
            1 for entry in snapshot.scene_inputs._evaluated_mesh_cache.values()
            if str(entry.get("device", "")).startswith("cuda")
            and entry.get("mesh") is not None))
        return result

    owner_type._configure_candidate = observe_configuration
    try:
        check("successful admission",
              bpy.ops.fluidsim.initialize() == {"FINISHED"}, fs.status)
    finally:
        owner_type._configure_candidate = original_configure
    sess = session()
    initial_gathers = gathered_depsgraphs[:6]
    check("all admission gatherers observed", len(initial_gathers) == 6,
          repr(initial_gathers))
    check("shared depsgraph coherence",
          initial_gathers[0][1] is not None
          and all(item[1] is initial_gathers[0][1]
                  for item in initial_gathers), repr(initial_gathers))
    check("active CUDA solver", sess.active and str(sess.solver.device).startswith("cuda"))
    check(
        "closed-domain wall invariant",
        not settings.open_boundaries
        and not settings.domain_sdf_walls
        and sess.solver.settings.domain_sdf_walls,
    )
    check("sources installed", installed.get("sources", 0) >= 1)
    check("collider installed", installed.get("colliders", 0) >= 1)
    check("force installed", installed.get("analytic_forces", 0) >= 1)
    check("height installed", installed.get("height") is True)
    check("seed committed", installed.get("seeds", 0) >= 1
          and sess.solver.particles.refresh_host_count() >= 1)
    check("custom effect captured", installed.get("effects", 0) >= 1)
    check("outputs prepared", any(obj.type == "POINTCLOUD" for obj in scene.objects))
    check("CUDA meshes materialized only after preflight on candidate",
          materialized_meshes and materialized_meshes[-1] >= 1,
          repr(materialized_meshes))

    # Both capacity and memory rejection happen before Replacement Cutover and
    # preserve the exact active solver object.
    for label in ("capacity", "memory"):
        old = sess.solver
        def reject(*_args, _label=label, **_kwargs):
            raise RuntimeError(f"injected {_label} preflight rejection")
        owner_type._preflight = reject
        try:
            try:
                sess.initialize(scene)
            except RuntimeError as exc:
                check(f"{label} rejection surfaced", label in str(exc), repr(exc))
            else:
                raise RuntimeError(f"{label} rejection did not fail")
            check(f"{label} preserves old solver", sess.solver is old and sess.active)
        finally:
            owner_type._preflight = original_preflight

    # Allocation, configuration, and seed failures occur while the candidate
    # is staged and must preserve the authoritative solver. Output preparation
    # follows the authority swap, so its failure leaves a clean inactive
    # session. A successful admission between faults proves recovery.
    phase_methods = (
        ("allocate", owner_type, "_allocate_solver", original_allocate, False),
        ("configure", owner_type, "_configure_candidate", original_configure, False),
        ("seed", sources, "commit_prepared_seeds", original_seeds, False),
        ("output", owner_type, "_prepare_outputs", original_outputs, True),
    )
    for label, target, method_name, original, post_cutover in phase_methods:
        flush_count = len(cleanup_flushes)
        old = sess.solver

        def fail(*args, _label=label, _method=method_name, **kwargs):
            if _method == "_configure_candidate":
                original_configure(*args, **kwargs)
            raise RuntimeError(f"injected {_label} failure")
        setattr(target, method_name, fail)
        try:
            try:
                sess.initialize(scene)
            except RuntimeError as exc:
                check(f"{label} fault surfaced", label in str(exc), repr(exc))
            else:
                raise RuntimeError(f"{label} fault did not fail")
            if post_cutover:
                check(f"{label} leaves inactive session", not sess.active)
            else:
                check(f"{label} preserves authoritative solver",
                      sess.active and sess.solver is old)
            check(f"{label} flushes released CUDA ownership",
                  len(cleanup_flushes) > flush_count,
                  repr(cleanup_flushes[flush_count:]))
            wp.synchronize()
        finally:
            setattr(target, method_name, original)
        check(f"recovery after {label}",
              bpy.ops.fluidsim.initialize() == {"FINISHED"}, fs.status)

    sess.release(scene)
    check("final cleanup", not sess.active)
    check("final CUDA pool flush observed", bool(cleanup_flushes),
          repr(cleanup_flushes))
    wp.synchronize()
    for name, original in original_gatherers.items():
        setattr(SceneInputs, name, original)
    sources.prepare_auto_seeds = original_seed_gather
    memory_safety.release_cuda_memory = original_release_cuda_memory
    print("SIMULATION-ADMISSION-ACCEPTANCE-PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SIMULATION-ADMISSION-ACCEPTANCE-FAIL: {exc!r}")
        raise
