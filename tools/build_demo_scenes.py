"""Build the three canonical FluidSim Warp demo scenes and export particle data.

Run:  blender --background --python tools/build_demo_scenes.py

Scenes (each saved as demos/<name>.blend with per-frame particle exports in
demos/<name>/frame_####.npz and per-frame aggregate metrics in
demos/<name>/series.json):

1. dam_break     - classic Martin & Moses dam-break column collapse.
2. crown_splash  - droplet impact into a shallow pool (Worthington crown).
3. slosh_tank    - offset water column sloshing in a closed tank; first-mode
                   period is analytically known (w^2 = g k tanh(k h)).

The exported data is analyzed offline by tools/analyze_demos.py.
"""

import json
import math
import os
import sys
import time

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMOS = os.path.join(REPO, "demos")
FPS = 24


def log(msg):
    print(f"[demo] {msg}", flush=True)


def lattice(lo, hi, spacing):
    """Regular particle lattice filling [lo, hi) with given spacing."""
    axes = [np.arange(lo[i] + 0.5 * spacing, hi[i], spacing, dtype=np.float32)
            for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def sphere_lattice(center, radius, spacing):
    lo = [center[i] - radius for i in range(3)]
    hi = [center[i] + radius for i in range(3)]
    pts = lattice(lo, hi, spacing)
    d = np.linalg.norm(pts - np.asarray(center, dtype=np.float32), axis=1)
    return pts[d <= radius]


def fresh_scene():
    # Clear the default scene in place; read_homefile() crashes in
    # background mode with heavy third-party startup add-ons installed.
    import addon_utils

    addon_utils.enable("fluidSim_warp_blender51", default_set=True,
                       persistent=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=True)
    for block in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights,
                  bpy.data.pointclouds):
        for datablock in list(block):
            if datablock.users == 0:
                block.remove(datablock)
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.render.fps_base = 1.0
    return scene


def make_domain(scene, lo, hi, name):
    bpy.ops.mesh.primitive_cube_add()
    dom = bpy.context.active_object
    dom.name = name
    center = [(lo[i] + hi[i]) * 0.5 for i in range(3)]
    dom.location = center
    dom.scale = [(hi[i] - lo[i]) * 0.5 for i in range(3)]
    dom.display_type = "WIRE"
    dom.hide_render = True
    bpy.context.view_layer.update()
    scene.fluidsim.domain_object = dom
    return dom


def add_camera_light(scene, cam_loc, cam_rot):
    cam_data = bpy.data.cameras.new("Demo Cam")
    cam = bpy.data.objects.new("Demo Cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = cam_loc
    cam.rotation_euler = cam_rot
    scene.camera = cam
    sun_data = bpy.data.lights.new("Demo Sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Demo Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.7, 0.15, 0.4)


def run_scene(name, frames, dump_velocities=True):
    """Step the live session; export per-frame particle data + aggregates."""
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = bpy.context.scene
    sess = session()
    out_dir = os.path.join(DEMOS, name)
    os.makedirs(out_dir, exist_ok=True)
    series = {
        "fps": FPS,
        "frames": [],
        "particle_count": [],
        "whitewater_count": [],
        "active_voxels": [],
        "step_ms": [],
        "substeps": [],
    }
    t0 = time.perf_counter()
    for f in range(1, frames + 1):
        bpy.ops.fluidsim.step(frames=1)
        solver = sess.solver
        pos = solver.particle_positions()
        vel = solver.particle_velocities()
        ww = solver.whitewater_positions()
        payload = {"positions": pos.astype(np.float16)}
        if dump_velocities:
            payload["velocities"] = vel.astype(np.float16)
        if len(ww):
            payload["whitewater"] = ww.astype(np.float16)
        np.savez_compressed(os.path.join(out_dir, f"frame_{f:04d}.npz"),
                            **payload)
        fs = scene.fluidsim
        series["frames"].append(f)
        series["particle_count"].append(int(len(pos)))
        series["whitewater_count"].append(int(len(ww)))
        series["active_voxels"].append(int(fs.stat_voxels))
        series["step_ms"].append(float(fs.stat_step_ms))
        series["substeps"].append(int(fs.stat_substeps))
        if f % 12 == 0 or f == 1:
            log(f"{name} frame {f}/{frames}: {len(pos):,} particles, "
                f"{len(ww):,} ww, {series['step_ms'][-1]:.0f} ms")
    series["wall_time_s"] = time.perf_counter() - t0
    with open(os.path.join(out_dir, "series.json"), "w",
              encoding="utf-8") as fh:
        json.dump(series, fh)
    log(f"{name} simulated {frames} frames in {series['wall_time_s']:.1f}s")


def save_scene(name, preview=True):
    path = os.path.join(DEMOS, f"{name}.blend")
    if preview:
        try:
            scene = bpy.context.scene
            scene.render.resolution_x = 640
            scene.render.resolution_y = 420
            scene.render.filepath = os.path.join(DEMOS, f"{name}_preview.png")
            bpy.ops.render.render(write_still=True)
            log(f"{name} preview rendered")
        except Exception as exc:  # noqa: BLE001
            log(f"{name} preview render skipped: {exc!r}")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    log(f"saved {path}")


# ---------------------------------------------------------------------------
# Scene 1: dam break
# ---------------------------------------------------------------------------

def build_dam_break():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = fresh_scene()
    fs = scene.fluidsim
    lo, hi = (0.0, 0.0, 0.0), (3.2, 0.6, 1.6)
    make_domain(scene, lo, hi, "DamBreak Domain")
    fs.voxel_size = 0.03
    fs.max_particles = 600_000
    fs.min_substeps = 1
    fs.max_substeps = 16
    fs.whitewater_enabled = True
    fs.profiling_enabled = False
    fs.show_particles = True
    fs.show_surface = True
    bpy.ops.fluidsim.initialize()
    sess = session()
    # H = 1.2 m column, a = 0.6 m wide, spanning the tank in y.
    pts = lattice((0.0, 0.0, 0.0), (0.6, 0.6, 1.2), 0.015)
    n = sess.solver.inject_particles(pts)
    log(f"dam_break: injected {n:,} particles (H=1.2, a=0.6)")
    add_camera_light(scene, (1.6, -4.4, 1.9), (1.30, 0.0, 0.0))
    run_scene("dam_break", 72)
    save_scene("dam_break")
    meta = {
        "scene": "dam_break",
        "gravity": 9.81,
        "column_height": 1.2,
        "column_width": 0.6,
        "domain": [list(lo), list(hi)],
        "voxel_size": 0.03,
        "fps": FPS,
        "spacing": 0.015,
        "injected": int(n),
    }
    with open(os.path.join(DEMOS, "dam_break", "meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    bpy.ops.fluidsim.release()


# ---------------------------------------------------------------------------
# Scene 2: crown splash
# ---------------------------------------------------------------------------

def build_crown_splash():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = fresh_scene()
    fs = scene.fluidsim
    lo, hi = (0.0, 0.0, 0.0), (0.8, 0.8, 0.5)
    make_domain(scene, lo, hi, "CrownSplash Domain")
    fs.voxel_size = 0.01
    fs.max_particles = 600_000
    fs.min_substeps = 1
    fs.max_substeps = 24
    fs.flip_ratio = 0.97
    fs.whitewater_enabled = True
    fs.ww_churn_gain = 2.0
    fs.profiling_enabled = False
    fs.show_particles = True
    fs.particle_display_fraction = 1.0
    bpy.ops.fluidsim.initialize()
    sess = session()
    pool_depth = 0.05
    pool = lattice((0.0, 0.0, 0.0), (0.8, 0.8, pool_depth), 0.005)
    n_pool = sess.solver.inject_particles(pool)
    drop_center = (0.4, 0.4, 0.2)
    drop_radius = 0.045
    drop_speed = 3.5
    drop = sphere_lattice(drop_center, drop_radius, 0.005)
    n_all = sess.solver.inject_particles(drop, velocity=(0.0, 0.0, -drop_speed))
    log(f"crown_splash: pool {n_pool:,} + droplet {n_all - n_pool:,} particles")
    add_camera_light(scene, (0.4, -1.05, 0.42), (1.35, 0.0, 0.0))
    run_scene("crown_splash", 48)
    save_scene("crown_splash")
    meta = {
        "scene": "crown_splash",
        "gravity": 9.81,
        "pool_depth": pool_depth,
        "drop_center": list(drop_center),
        "drop_radius": drop_radius,
        "drop_speed": drop_speed,
        "domain": [list(lo), list(hi)],
        "voxel_size": 0.01,
        "fps": FPS,
        "pool_particles": int(n_pool),
        "total_particles": int(n_all),
    }
    with open(os.path.join(DEMOS, "crown_splash", "meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    bpy.ops.fluidsim.release()


# ---------------------------------------------------------------------------
# Scene 3: sloshing tank
# ---------------------------------------------------------------------------

def build_slosh_tank():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = fresh_scene()
    fs = scene.fluidsim
    lo, hi = (0.0, 0.0, 0.0), (1.0, 0.3, 0.6)
    make_domain(scene, lo, hi, "Slosh Domain")
    fs.voxel_size = 0.02
    fs.max_particles = 300_000
    fs.min_substeps = 1
    fs.max_substeps = 12
    fs.whitewater_enabled = False
    fs.profiling_enabled = False
    fs.show_particles = True
    fs.particle_display_fraction = 1.0
    bpy.ops.fluidsim.initialize()
    sess = session()
    sess.solver.settings.domain_sdf_walls = True
    # First-mode initial surface: z(x) = h + A*cos(pi*x/L) with h = 0.25,
    # A = 0.06 (gentle, near-linear). Analytic first-mode period:
    # omega = sqrt(g * k * tanh(k*h)), k = pi/L  ->  T = 1.40 s.
    pts = lattice((0.0, 0.0, 0.0), (1.0, 0.3, 0.32), 0.01)
    surface = 0.25 + 0.06 * np.cos(math.pi * pts[:, 0] / 1.0)
    pts = pts[pts[:, 2] < surface]
    n = sess.solver.inject_particles(pts)
    log(f"slosh_tank: injected {n:,} particles (h=0.25, A=0.06 first mode)")
    add_camera_light(scene, (0.5, -1.7, 0.5), (1.35, 0.0, 0.0))
    run_scene("slosh_tank", 144)
    save_scene("slosh_tank")
    meta = {
        "scene": "slosh_tank",
        "gravity": 9.81,
        "tank_length": 1.0,
        "mean_depth": 0.25,
        "domain": [list(lo), list(hi)],
        "voxel_size": 0.02,
        "fps": FPS,
        "injected": int(n),
    }
    with open(os.path.join(DEMOS, "slosh_tank", "meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    bpy.ops.fluidsim.release()


def main():
    os.makedirs(DEMOS, exist_ok=True)
    builders = {
        "dam_break": build_dam_break,
        "crown_splash": build_crown_splash,
        "slosh_tank": build_slosh_tank,
    }
    # One scene per Blender process:  ... --python this.py -- <scene>
    which = None
    if "--" in sys.argv:
        rest = sys.argv[sys.argv.index("--") + 1:]
        if rest:
            which = rest[0]
    targets = [which] if which in builders else list(builders)
    t0 = time.perf_counter()
    try:
        for key in targets:
            builders[key]()
    except Exception:
        import traceback

        traceback.print_exc()
        print("DEMO-BUILD-FAIL", flush=True)
        sys.exit(1)
    print(f"total {time.perf_counter() - t0:.0f}s", flush=True)
    print("DEMO-BUILD-PASS", flush=True)


main()
