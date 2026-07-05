"""Profile and validate the interactive Blender user frame path.

This protects one narrow path: repo add-on load, Quick Setup, Initialize
Solver, then the same per-frame scene-sync, solver-step, viewport-output update,
and optional cache-write calls used by live viewport playback. It is intended to
catch regressions from solver, scene-bridge, viewport, and cache changes that
still pass bpy-free solver profiling.

Times each segment separately so display and cache overhead can be compared
against pure solver cost:

- ``sync``: ``sync_scene_to_solver`` (sources + collider gather)
- ``solve``: ``FluidSolver.step_frame``
- ``view``: ``viewport.update_outputs`` (readback + datablock update)
- ``cache``: ``io.cache.write_frame`` (only when ``--cache``)

Run:
    blender --background --python tools/profile_user_path.py -- \
        [--frames N] [--voxel-size V] [--particles] [--cache] \
        [--collider-subdiv K] [--json-out PATH] [--no-validate]
"""

from __future__ import annotations

import argparse
import json
import statistics
import os
import sys
import time


USER_PATH_CONTRACT = {
    "name": "interactive_viewport_user_path",
    "protected_path": (
        "repo add-on load -> fluidsim.setup_scene -> fluidsim.initialize -> "
        "sync_scene_to_solver -> FluidSolver.step_frame -> "
        "viewport.update_outputs -> optional cache.write_frame"
    ),
    "parameter_surface": [
        "--frames",
        "--voxel-size",
        "--max-particles",
        "--particles",
        "--cache",
        "--collider-subdiv",
        "--max-mean-view-ms",
        "--max-mean-total-ms",
        "--min-particles",
        "--min-voxels",
    ],
    "inspection_surface": [
        "result",
        "checks",
        "means",
        "frames",
        "viewport_counts",
        "scene_stats_probe",
        "particles_mean",
        "voxels_mean",
        "surface_verts_last",
    ],
    "regression_focus": [
        "CUDA session remains active",
        "solver frame and scene stats advance across playback frames",
        "viewport datablocks receive non-empty changing output",
        "collider gather, viewport readback, and optional cache writes stay bounded",
    ],
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--max-particles", type=int, default=2_000_000)
    ap.add_argument("--particles", action="store_true",
                    help="show particle point cloud instead of surface mesh")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--collider-subdiv", type=int, default=5,
                    help="icosphere subdivision for a static high-poly collider")
    ap.add_argument("--max-mean-view-ms", type=float, default=80.0,
                    help="validation ceiling for warm-frame viewport update mean")
    ap.add_argument("--max-mean-total-ms", type=float, default=250.0,
                    help="validation ceiling for warm-frame sync+solve+view(+cache) mean")
    ap.add_argument("--min-particles", type=int, default=10_000,
                    help="minimum mean active particles required by validation")
    ap.add_argument("--min-voxels", type=int, default=1_000,
                    help="minimum mean active voxels required by validation")
    ap.add_argument("--no-validate", action="store_true",
                    help="write timing data without failing on regression checks")
    ap.add_argument("--json-out", default="")
    return ap.parse_args(argv)


def _viewport_counts(bpy, particles_mode):
    surf = bpy.data.objects.get("FluidSim Surface")
    parts = bpy.data.objects.get("FluidSim Particles")
    surface_verts = len(surf.data.vertices) if surf and surf.type == "MESH" else 0
    particle_points = 0
    if parts is not None and hasattr(parts.data, "attributes"):
        attr = parts.data.attributes.get("position")
        particle_points = len(attr.data) if attr is not None else 0
    elif parts is not None and hasattr(parts.data, "vertices"):
        particle_points = len(parts.data.vertices)
    primary_count = particle_points if particles_mode else surface_verts
    return {
        "surface_verts": int(surface_verts),
        "particle_points": int(particle_points),
        "primary_count": int(primary_count),
    }


def _check(checks, name, condition, detail=""):
    ok = bool(condition)
    checks.append({"name": name, "ok": ok, "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def _validate_contract(args, summary, sess, device, output_counts):
    checks = []
    frames = summary["frames"]
    means = summary["means"]
    first = frames[0] if frames else {}
    last = frames[-1] if frames else {}
    primary_counts = [c["primary_count"] for c in output_counts]
    particle_points = [c["particle_points"] for c in output_counts]
    scene_probe = summary["scene_stats_probe"]

    _check(checks, "contract path", summary["contract"]["name"] == USER_PATH_CONTRACT["name"])
    _check(checks, "cuda session", sess.active and str(device).startswith("cuda"), device)
    _check(checks, "frames measured", len(frames) == args.frames, len(frames))
    _check(checks, "solver advanced", sess.solver.frame >= args.frames, sess.solver.frame)
    _check(checks, "particles active", summary["particles_mean"] >= args.min_particles,
           f"{summary['particles_mean']:.0f} >= {args.min_particles}")
    _check(checks, "voxels active", summary["voxels_mean"] >= args.min_voxels,
           f"{summary['voxels_mean']:.0f} >= {args.min_voxels}")
    _check(checks, "viewport primary output nonempty", max(primary_counts or [0]) > 0,
           max(primary_counts or [0]))
    _check(checks, "viewport output changes", len(set(primary_counts)) > 1,
           f"{primary_counts[0] if primary_counts else 0} -> {primary_counts[-1] if primary_counts else 0}")
    if args.particles:
        _check(checks, "particle output available", max(particle_points or [0]) > 0,
               max(particle_points or [0]))
    _check(checks, "scene stats updated",
           scene_probe["particles"] == scene_probe["step_particles"] and
           scene_probe["voxels"] == scene_probe["step_voxels"] and
           scene_probe["particles"] > 0 and scene_probe["voxels"] > 0,
           f"{scene_probe['particles']} particles, {scene_probe['voxels']} voxels")
    _check(checks, "scene stats history recorded",
           scene_probe["frame_stats_len"] > 0 and scene_probe["status"].startswith("Frame "),
           f"{scene_probe['frame_stats_len']} frames, status={scene_probe['status']!r}")
    _check(checks, "viewport mean bounded", means["view_ms"] <= args.max_mean_view_ms,
           f"{means['view_ms']:.3f} <= {args.max_mean_view_ms:.3f} ms")
    _check(checks, "total mean bounded", means["total_ms"] <= args.max_mean_total_ms,
           f"{means['total_ms']:.3f} <= {args.max_mean_total_ms:.3f} ms")
    if args.cache:
        _check(checks, "cache writes timed", means["cache_ms"] > 0.0,
               f"{means['cache_ms']:.3f} ms")

    return checks


def main():
    args = parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    import addon_utils
    import bmesh
    import bpy

    # Measure the repo checkout, not a possibly stale installed addon copy:
    # repo root precedes Blender's addon dirs on sys.path so the package
    # import inside addon_utils.enable resolves here.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    sys.modules.pop("fluidSim_warp_blender51", None)

    addon_utils.enable("fluidSim_warp_blender51", default_set=True, persistent=True)
    import fluidSim_warp_blender51 as _fs_mod
    assert os.path.abspath(_fs_mod.__file__).startswith(repo_root), (
        f"addon loaded from {_fs_mod.__file__}, expected repo {repo_root}")
    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.voxel_size = args.voxel_size
    fs.max_particles = args.max_particles
    fs.max_whitewater = 500_000
    fs.whitewater_enabled = True
    fs.surface_enabled = True
    if args.particles:
        fs.liquid_view_mode = "PARTICLES"
    else:
        fs.liquid_view_mode = "SURFACE"
    fs.particle_display_fraction = 1.0
    fs.cache_enabled = False  # timed manually below

    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}

    # Static high-poly collider to exercise the per-frame collider gather.
    if args.collider_subdiv > 0:
        mesh = bpy.data.meshes.new("BenchCollider")
        obj = bpy.data.objects.new("BenchCollider", mesh)
        scene.collection.objects.link(obj)
        bm = bmesh.new()
        bmesh.ops.create_icosphere(
            bm, subdivisions=args.collider_subdiv, radius=0.8)
        bm.to_mesh(mesh)
        bm.free()
        obj.location = (0.0, 0.0, 0.8)
        obj.fluidsim.role = "COLLIDER"
        obj.fluidsim.enabled = True

    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, fs.status

    from fluidSim_warp_blender51.blender import scene_bridge, viewport
    from fluidSim_warp_blender51.io import cache as cache_io

    sess = scene_bridge.session()
    assert sess.active
    if args.cache:
        fs.cache_enabled = True
        fs.cache_path = "//_bench_user_cache/"

    rows = []
    output_counts = []
    for f in range(args.frames):
        scene.frame_set(f + 1)
        t0 = time.perf_counter()
        dg = bpy.context.evaluated_depsgraph_get()
        scene_bridge.sync_scene_to_solver(scene, sess, dg)
        t1 = time.perf_counter()
        stats = sess.solver.step_frame()
        t2 = time.perf_counter()
        viewport.update_outputs(scene, sess)
        t3 = time.perf_counter()
        counts = _viewport_counts(bpy, args.particles)
        output_counts.append(counts)
        cache_ms = 0.0
        if args.cache:
            tc = time.perf_counter()
            cache_io.write_frame(scene, sess)
            cache_ms = (time.perf_counter() - tc) * 1000.0
        rows.append({
            "frame": f,
            "sync_ms": (t1 - t0) * 1000.0,
            "solve_ms": (t2 - t1) * 1000.0,
            "view_ms": (t3 - t2) * 1000.0,
            "cache_ms": cache_ms,
            "total_ms": (t3 - t0) * 1000.0 + cache_ms,
            "particles": int(stats.active_particles),
            "voxels": int(stats.active_voxels),
            "whitewater": int(stats.whitewater_particles),
            "surface_verts": counts["surface_verts"],
            "particle_points": counts["particle_points"],
        })

    warm = rows[3:] if len(rows) > 6 else rows

    def mean(key):
        return statistics.fmean(r[key] for r in warm)

    # Untimed wrapper-path probe: the segmented timings above isolate sync,
    # solve, and viewport cost; this call verifies the live scene-bridge surface
    # still publishes UI/debug stats from the same playback path.
    probe_stats = scene_bridge.step_frame(scene, sess)
    if probe_stats is None:
        raise RuntimeError("scene_bridge.step_frame returned no stats")

    surf = bpy.data.objects.get("FluidSim Surface")
    n_surf_verts = len(surf.data.vertices) if surf and surf.type == "MESH" else 0
    scene_stats_probe = {
        "particles": int(fs.stat_particles),
        "voxels": int(fs.stat_voxels),
        "whitewater": int(fs.stat_whitewater),
        "step_ms": float(fs.stat_step_ms),
        "substeps": int(fs.stat_substeps),
        "status": str(fs.status),
        "frame_stats_len": len(sess.frame_stats),
        "solver_frame": int(sess.solver.frame),
        "step_particles": int(probe_stats.active_particles),
        "step_voxels": int(probe_stats.active_voxels),
        "step_whitewater": int(probe_stats.whitewater_particles),
    }
    summary = {
        "result": "USER-PATH-PROFILE-PASS",
        "contract": USER_PATH_CONTRACT,
        "config": vars(args),
        "means": {k: mean(k) for k in
                  ("sync_ms", "solve_ms", "view_ms", "cache_ms", "total_ms")},
        "particles_mean": mean("particles"),
        "voxels_mean": mean("voxels"),
        "whitewater_mean": mean("whitewater"),
        "surface_verts_last": n_surf_verts,
        "viewport_counts": output_counts,
        "scene_stats_probe": scene_stats_probe,
        "frames": rows,
    }
    if not args.no_validate:
        checks = _validate_contract(args, summary, sess, sess.solver.device, output_counts)
        summary["checks"] = checks
        failed = [check for check in checks if not check["ok"]]
        if failed:
            summary["result"] = "USER-PATH-REGRESSION-FAIL"
    else:
        summary["checks"] = []
    print("USER-PATH-PROFILE " + json.dumps(summary["means"]))
    print(f"scene: voxels={summary['voxels_mean']:.0f} "
          f"particles={summary['particles_mean']:.0f} "
          f"whitewater={summary['whitewater_mean']:.0f} "
          f"surface_verts={n_surf_verts}")
    if args.json_out:
        parent = os.path.dirname(os.path.abspath(args.json_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=1)
    if summary["result"] != "USER-PATH-PROFILE-PASS":
        print(summary["result"])
        sys.exit(1)
    print(summary["result"])
    sys.exit(0)


if __name__ == "__main__":
    main()
