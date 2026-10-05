"""Timeline/viewport acceptance for ST-FLIP Blender integration.

Runs in a real Blender process. It verifies that loading a .blend with the
add-on enabled leaves a persistent frame-change handler in place, and that
advancing the timeline creates/updates visible FluidSim output objects.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import bpy
import numpy as np


def _ensure_package_on_path() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    root = here.parents[2]
    package_dir = root / "fluidSim_warp_blender51"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return package_dir


def _enable_addon(package_dir: pathlib.Path) -> None:
    import addon_utils

    addon_utils.modules_refresh()
    addon_utils.enable("fluidSim_warp_blender51", default_set=True, persistent=True)
    if not hasattr(bpy.context.scene, "fluidsim"):
        import fluidSim_warp_blender51 as addon
        addon.register()


def _count_object(name: str) -> int:
    # Output clouds carry spare capacity, so len(data.points) is not the count.
    from fluidSim_warp_blender51.blender import output_publication

    return output_publication.live_point_count(bpy.data.objects.get(name))



def _initial_particles(particle_count: int = 500_000, speed: float = 22.0, seed: int = 17):
    target = max(500_000, int(particle_count))
    rng = np.random.default_rng(seed)
    nx = max(8, int(round((target * 2.5) ** (1.0 / 3.0))))
    ny = max(8, int(round(nx * 0.64)))
    nz = max(4, int(math.ceil(target / max(nx * ny, 1))))
    xs = np.linspace(-1.55, -0.65, nx, dtype=np.float32)
    ys = np.linspace(-0.42, 0.42, ny, dtype=np.float32)
    zs = np.linspace(0.20, 1.20, nz, dtype=np.float32)
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)[:target].astype(np.float32, copy=True)
    pts += rng.uniform(-0.0045, 0.0045, pts.shape).astype(np.float32)
    phase = pts[:, 1] * 9.0 + pts[:, 2] * 5.0
    vels = np.empty_like(pts, dtype=np.float32)
    vels[:, 0] = speed + 0.8 * np.sin(phase)
    vels[:, 1] = 0.55 * np.sin(phase * 1.7)
    vels[:, 2] = 0.35 * np.cos(phase * 1.3)
    return pts, vels

def main(argv: list[str] | None = None) -> int:
    package_dir = _ensure_package_on_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(package_dir.parent / "demos" / "dam_break.blend"))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--particles", type=int, default=500_000)
    parser.add_argument("--output", default=str(package_dir / "demo_outputs" / "stflip_timeline_viewport_acceptance.json"))
    args, _ = parser.parse_known_args(argv)

    scene_path = pathlib.Path(args.scene)
    if not scene_path.exists():
        raise RuntimeError(f"Timeline acceptance scene not found: {scene_path}")

    _enable_addon(package_dir)
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    _enable_addon(package_dir)
    from fluidSim_warp_blender51.blender import playback, scene_bridge

    scene = bpy.context.scene
    fs = scene.fluidsim
    ds = scene_bridge.active_domain_settings(scene)
    if ds is None:
        raise RuntimeError(f"Timeline acceptance scene has no designated domain: {scene_path}")
    fs.live = True
    ds.spacetime_enabled = True
    fs.cache_playback_enabled = False
    playback.register_handlers()

    handlers = [getattr(h, "__name__", "") for h in bpy.app.handlers.frame_change_post]
    frames = max(24, int(args.frames))
    particles = max(500_000, int(args.particles))
    ds.max_particles = max(int(ds.max_particles), particles + 16_384)
    ds.surface_enabled = True
    ds.surface_resolution_mode = "MULTIPLIER"
    ds.surface_resolution_multiplier = 2.0
    ds.surface_leaf_capacity_scale = 8.0
    ds.surface_max_dense_dim = 512
    ds.surface_update_cadence = "FRAME"
    ds.surface_build_lowres_sdf = True
    ds.surface_filter_iterations = 0
    fs.show_surface = True
    fs.show_particles = True
    fs.particle_display_fraction = 1.0
    start = scene.frame_start
    scene.frame_set(start)
    scene_bridge.release_session()
    sess = scene_bridge.initialize_session(scene)
    solver = sess.solver
    pts, vels = _initial_particles(particle_count=particles)
    solver.inject_particles(pts, velocities=vels)
    # First viewport update is setup only.  Actual acceptance motion below is
    # driven by Blender frame changes and the persistent timeline handler.
    from fluidSim_warp_blender51.blender import output_publication
    output_publication.update_outputs(scene, sess)
    p1 = solver.particle_positions().copy()
    v1, _t1 = solver.surface_mesh()
    out_p1 = _count_object("FluidSim Particles")
    out_s1 = _count_object("FluidSim Surface")
    for f in range(start + 1, start + frames + 1):
        scene.frame_set(f)
    sess = scene_bridge.session()
    if not sess.active:
        raise RuntimeError("Timeline frame-change did not keep the solver active")
    solver = sess.solver
    p2 = solver.particle_positions().copy()
    v2, _t2 = solver.surface_mesh()
    out_p2 = _count_object("FluidSim Particles")
    out_s2 = _count_object("FluidSim Surface")
    n = min(len(p1), len(p2))
    max_motion = float(np.max(np.linalg.norm(p2[:n] - p1[:n], axis=1))) if n else 0.0
    offsets = solver.particles.time_offset[:solver.particles.host_count].numpy() if solver.particles.host_count else np.zeros(0)
    payload = {
        "scene": str(scene_path),
        "handler_present": "_on_frame_change" in handlers or any("_on_frame_change" == h for h in handlers),
        "live": bool(fs.live),
        "spacetime_enabled": bool(ds.spacetime_enabled),
        "requested_frames": int(frames),
        "requested_particles": int(particles),
        "solver_frame": int(solver.frame),
        "particles_frame1": int(len(p1)),
        "particles_frame2": int(len(p2)),
        "surface_vertices_frame1": int(len(v1)),
        "surface_vertices_frame2": int(len(v2)),
        "surface_voxel_size": float(solver.surface.metadata()["voxel_size"]),
        "surface_active_leaves": int(getattr(solver.render_surface_grid, "active_leaf_count", 0)),
        "surface_grid_is_sim_grid": bool(solver.render_surface_grid is solver.grid),
        "viewport_particle_object_count_frame1": int(out_p1),
        "viewport_surface_object_count_frame1": int(out_s1),
        "viewport_particle_object_count_frame2": int(out_p2),
        "viewport_surface_object_count_frame2": int(out_s2),
        "max_particle_motion": max_motion,
        "time_offset_abs_max": float(np.max(np.abs(offsets))) if offsets.size else 0.0,
        "time_offset_std": float(np.std(offsets)) if offsets.size else 0.0,
        "status": fs.status,
    }
    checks = [
        ("frame handler registered", payload["handler_present"]),
        ("live enabled", payload["live"]),
        ("ST-FLIP enabled", payload["spacetime_enabled"]),
        ("solver advanced from timeline", payload["solver_frame"] >= frames),
        ("visible particle object populated", payload["viewport_particle_object_count_frame2"] > 0),
        ("visible surface object populated", payload["viewport_surface_object_count_frame2"] > 0 and payload["surface_vertices_frame2"] > 0),
        ("high resolution surface grid active", payload["surface_voxel_size"] < ds.voxel_size and not payload["surface_grid_is_sim_grid"]),
        ("particles moved", payload["max_particle_motion"] > 1.0e-4),
        ("ST residual active", payload["time_offset_std"] > 1.0e-5),
        ("particle count requirement met", payload["particles_frame1"] >= 500_000 and payload["particles_frame2"] >= 500_000),
        ("frame count requirement met", payload["solver_frame"] >= 24),
    ]
    payload["validation"] = {"passed": all(ok for _, ok in checks), "checks": [{"name": n, "pass": bool(ok)} for n, ok in checks]}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ST-FLIP timeline viewport acceptance: {'PASS' if payload['validation']['passed'] else 'FAIL'}")
    print(f"  particles={payload['particles_frame1']}->{payload['particles_frame2']}, surface={payload['surface_vertices_frame1']}->{payload['surface_vertices_frame2']}, motion={payload['max_particle_motion']:.4f}")
    print(f"  output={output}")
    return 0 if payload["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
