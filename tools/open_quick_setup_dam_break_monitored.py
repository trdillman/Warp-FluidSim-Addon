"""Open the Quick Setup dam-break scene and log live playback timings.

This launcher is intended for visible Blender GUI review. It enables the
worktree add-on through Blender's add-on system, runs the same Quick Setup
operator exposed in the panel, initializes the solver, starts playback, and
records one CSV row per frame-change update.
"""

from __future__ import annotations

import csv
import os
import sys
import time

import addon_utils
import bpy


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDON_NAME = "fluidSim_warp_blender51"
LOG_PATH = os.path.join(REPO, "bench", "live_quick_setup_dam_break_timings.csv")


def _enable_worktree_addon():
    addon_utils.enable(ADDON_NAME, default_set=True, persistent=True)
    mod = __import__(ADDON_NAME)
    addon_path = os.path.realpath(mod.__file__)
    repo_path = os.path.realpath(REPO)
    if not addon_path.startswith(repo_path):
        raise RuntimeError(
            f"Add-on loaded from {addon_path}, expected worktree {repo_path}")
    if not addon_utils.check(ADDON_NAME)[1]:
        raise RuntimeError(f"{ADDON_NAME} is not enabled through addon_utils")
    if ADDON_NAME not in bpy.context.preferences.addons:
        raise RuntimeError(f"{ADDON_NAME} missing from Blender preferences")
    return mod


def _write_header():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "wall_time",
            "frame",
            "stat_step_ms",
            "particles",
            "voxels",
            "whitewater",
            "substeps",
            "pressure_iterations",
            "pressure_residual",
            "surface_dense_dim",
            "surface_verts",
        ])


def _surface_dense_dim(sess):
    dims = getattr(sess.solver.surface_extractor, "_dense_dim", None)
    if dims is None:
        return "0x0x0"
    return "x".join(str(int(d)) for d in dims)


def _surface_verts():
    obj = bpy.data.objects.get("FluidSim Surface")
    if obj is None or obj.type != "MESH":
        return 0
    return len(obj.data.vertices)


def _install_logger():
    from fluidSim_warp_blender51.blender import scene_bridge

    def log_frame(scene, _depsgraph):
        fs = scene.fluidsim
        if int(fs.stat_particles) <= 0:
            return
        sess = scene_bridge.session()
        stats = sess.solver.stats
        with open(LOG_PATH, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                f"{time.time():.6f}",
                int(scene.frame_current),
                f"{float(fs.stat_step_ms):.6f}",
                int(fs.stat_particles),
                int(fs.stat_voxels),
                int(fs.stat_whitewater),
                int(fs.stat_substeps),
                int(stats.pressure_iterations),
                f"{float(stats.pressure_residual):.9g}",
                _surface_dense_dim(sess),
                _surface_verts(),
            ])

    marker = "_fluidsim_live_timing_logger"
    bpy.app.handlers.frame_change_post[:] = [
        h for h in bpy.app.handlers.frame_change_post
        if getattr(h, "__name__", "") != marker
    ]
    log_frame.__name__ = marker
    bpy.app.handlers.frame_change_post.append(log_frame)


def main():
    mod = _enable_worktree_addon()
    scene = bpy.context.scene
    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    fs = scene.fluidsim
    domain = fs.domain_object
    if domain is None:
        raise RuntimeError("Quick Setup did not create a domain object")
    ds = domain.fluidsim_domain
    fs.live = True
    ds.surface_enabled = True
    ds.whitewater_enabled = True
    fs.show_surface = True
    fs.show_particles = True
    fs.show_whitewater = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 1.0
    ds.profiling_enabled = True
    domain_bounds = tuple(round(float(v), 4) for v in (*domain.location, *domain.scale))

    _write_header()
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, fs.status
    _install_logger()
    scene.frame_set(scene.frame_start + 1)
    try:
        bpy.ops.screen.animation_play()
    except Exception as exc:  # noqa: BLE001 - keep GUI open even if playback op fails
        print(f"FLUIDSIM-LIVE-PLAYBACK-START-FAILED {exc!r}", flush=True)
    print(
        "FLUIDSIM-LIVE-DAM-BREAK-OPEN "
        f"addon={mod.__file__} log={LOG_PATH} "
        f"domain_location_scale={domain_bounds}",
        flush=True,
    )


if __name__ == "__main__":
    main()
