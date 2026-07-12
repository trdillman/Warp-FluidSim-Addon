"""Build the canonical FluidSim Warp demo scenes and export particle data.

Run:
  blender --background --python tools/build_demo_scenes.py
  blender --background --python tools/build_demo_scenes.py -- --list-scenes
  blender --background --python tools/build_demo_scenes.py -- --standard-cache-scenes
  blender --background --python tools/build_demo_scenes.py -- dam_break slosh_tank

Scenes are saved as demos/<name>.blend. Large per-frame particle exports and
aggregate metrics are generated under artifacts/demo_cache/<name>/ so they do
not become source-control assets.

1. dam_break       - classic Martin & Moses dam-break column collapse.
2. crown_splash    - droplet impact into a shallow pool (Worthington crown).
3. slosh_tank      - offset water column sloshing in a closed tank; first-mode
                     period is analytically known (w^2 = g k tanh(k h)).
4. tidal_lock      - opposed columns colliding through offset lock gates.
5. material_response - four-lane friction/restitution collision showcase.
6. collision_update_proof - single-emitter swept-collision review scene.
7. maelstrom_basin - high-particle vortex basin with live playback metadata.

The exported data is analyzed offline by tools/analyze_demos.py.
"""

import json
import math
import os
import sys
import time
from types import SimpleNamespace

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon

DEMOS = os.path.join(REPO, "demos")
CACHE = os.path.join(REPO, "artifacts", "demo_cache")
FPS = 24
STANDARD_CACHE_SCENES = ("dam_break", "crown_splash", "slosh_tank",
                         "tidal_lock")


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
    enable_repo_addon(REPO, default_set=True, persistent=True)
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


def make_particle_seed(scene, points, name):
    """Create a hidden-in-render mesh whose vertices can seed a live run."""
    pts = np.asarray(points, dtype=np.float32)
    mesh = bpy.data.meshes.new(name)
    if len(pts):
        mesh.vertices.add(len(pts) - 1)
        mesh.vertices.foreach_set("co", pts.ravel())
        mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.display_type = "BOUNDS"
    obj.hide_render = True
    obj.show_name = True
    return obj


def make_source_box(scene, lo, hi, name, emit_rate=8.0):
    """Create a tagged analytic source so reopened scenes are playable."""
    from fluidSim_warp_blender51.blender import sources as _sources

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    src = bpy.context.active_object
    src.name = name
    src.location = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    src.scale = tuple((hi[i] - lo[i]) * 0.5 for i in range(3))
    _sources.tag_source(src, shape="BOX", mode="PARTICLES")
    src.fluidsim.emit_rate = emit_rate
    src.fluidsim.use_object_velocity = False
    src.show_name = True
    return src


def make_collider_box(scene, lo, hi, name, friction=0.05):
    """Create a tagged box collider for demo-scene flow shaping."""
    from fluidSim_warp_blender51.blender import colliders as _colliders

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    col = bpy.context.active_object
    col.name = name
    col.location = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    col.scale = tuple((hi[i] - lo[i]) * 0.5 for i in range(3))
    col.display_type = "WIRE"
    col.hide_render = True
    col.show_name = True
    _colliders.tag_collider(col, friction=friction, restitution=0.0)
    return col


def make_material_collider(name, location, scale, friction, restitution, color,
                           rotation=(0.0, 0.0, 0.0), sphere=False):
    """Create one visible, individually configured showcase collider."""
    from fluidSim_warp_blender51.blender import colliders as _colliders

    if sphere:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32, ring_count=16, radius=scale[0], location=location)
    else:
        bpy.ops.mesh.primitive_cube_add(
            size=1.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    if not sphere:
        obj.scale = scale
    obj.show_name = True
    material = bpy.data.materials.new(name + " Material")
    material.diffuse_color = (*color, 1.0)
    material.metallic = 0.25
    material.roughness = max(0.08, 1.0 - friction * 0.75)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = material.metallic
    bsdf.inputs["Roughness"].default_value = material.roughness
    obj.data.materials.append(material)
    _colliders.tag_collider(
        obj, friction=friction, restitution=restitution)
    return obj


def add_text_block(name, body):
    text = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    text.clear()
    text.write(body)
    return text


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
    out_dir = os.path.join(CACHE, name)
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
    scene = fresh_scene()
    from fluidSim_warp_blender51.blender.scene_bridge import session

    fs = scene.fluidsim
    lo, hi = (0.0, -1.00146496, 0.0), (5.6, 1.00146496, 1.6)
    make_domain(scene, lo, hi, "DamBreak Domain")
    fs.voxel_size = 0.03
    fs.max_particles = 600_000
    fs.min_substeps = 1
    fs.max_substeps = 16
    fs.whitewater_enabled = True
    fs.profiling_enabled = False
    fs.show_particles = True
    fs.show_surface = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 1.0
    fs.live = True
    scene.frame_start = 1
    scene.frame_end = 120
    scene.frame_set(1)
    # Keep the historical 128k preview seed while matching Tyler's reviewed
    # live emitter transform for future playback.
    pts = lattice((0.0, 0.0, 0.0), (0.6, 0.6, 1.2), 0.015)
    src = make_source_box(
        scene,
        (0.0, 0.0, 0.0),
        (0.6, 0.6, 1.2),
        "DamBreak Source Column",
        emit_rate=32.0,
    )
    src.location = (0.47105896, 0.0, 1.06185913)
    src.scale = (0.73273778, 1.70176971, 0.99442428)
    src.display_type = "WIRE"
    src.hide_render = True
    bpy.ops.mesh.primitive_uv_sphere_add(
        location=(2.41923952, 0.0, 0.0), radius=0.39267302)
    collider = bpy.context.active_object
    collider.name = "Sphere"
    from fluidSim_warp_blender51.blender import colliders as _colliders
    _colliders.tag_collider(collider, friction=1.0, restitution=0.0)
    src.fluidsim.enabled = True
    src.fluidsim.enabled = False  # preview bake uses direct particle injection
    bpy.ops.fluidsim.initialize()
    sess = session()
    n = sess.solver.inject_particles(pts)
    log(f"dam_break: injected {n:,} particles (H=1.2, a=0.6)")
    add_camera_light(scene, (1.6, -4.4, 1.9), (1.30, 0.0, 0.0))
    run_scene("dam_break", 72)
    src.fluidsim.enabled = True
    add_text_block(
        "README_Dam_Break_Playable",
        "Dam Break live playback notes\n"
        "=============================\n\n"
        "This file contains Tyler's canonical wide-domain dam-break setup: "
        "a raised source column and a spherical flow collider.\n\n"
        "Best path: open with blender/open_dam_break_playable.py. The script enables "
        "the FluidSim add-on, initializes the CUDA solver, advances to frame 2 "
        "using the 'DamBreak Source Column', enables Live Simulation, "
        "and starts playback.\n\n"
        "Manual path: press Initialize Solver, enable Live Simulation, then "
        "press Play. The 'DamBreak Source Column' stays enabled so Reset or "
        "Reinitialize can create visible fluid from any timeline frame.\n",
    )
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
        "source_object": src.name,
        "source_location": list(src.location),
        "source_scale": list(src.scale),
        "collider_object": collider.name,
        "playable_live_setup": True,
    }
    with open(os.path.join(DEMOS, "dam_break", "meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    bpy.ops.fluidsim.release()


# ---------------------------------------------------------------------------
# Scene 2: crown splash
# ---------------------------------------------------------------------------

def build_crown_splash():
    scene = fresh_scene()
    from fluidSim_warp_blender51.blender.scene_bridge import session

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
    scene = fresh_scene()
    from fluidSim_warp_blender51.blender.scene_bridge import session

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


# ---------------------------------------------------------------------------
# Scene 4: tidal lock
# ---------------------------------------------------------------------------

def build_tidal_lock():
    scene = fresh_scene()
    from fluidSim_warp_blender51.blender.scene_bridge import session

    fs = scene.fluidsim
    lo, hi = (0.0, 0.0, 0.0), (2.4, 0.8, 0.7)
    make_domain(scene, lo, hi, "TidalLock Domain")
    fs.voxel_size = 0.02
    fs.max_particles = 320_000
    fs.min_substeps = 1
    fs.max_substeps = 18
    fs.flip_ratio = 0.96
    fs.whitewater_enabled = True
    fs.max_whitewater = 120_000
    fs.ww_churn_gain = 1.8
    fs.ww_churn_threshold = 0.02
    fs.ww_emission_rate = 120.0
    fs.ww_min_speed = 0.10
    fs.profiling_enabled = False
    fs.show_particles = True
    fs.show_surface = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 1.0
    fs.live = False
    scene.frame_start = 1
    scene.frame_end = 96
    scene.frame_set(1)

    left_src = make_source_box(
        scene,
        (0.08, 0.08, 0.0),
        (0.62, 0.36, 0.32),
        "TidalLock West Source",
        emit_rate=18.0,
    )
    right_src = make_source_box(
        scene,
        (1.78, 0.44, 0.0),
        (2.32, 0.72, 0.32),
        "TidalLock East Source",
        emit_rate=18.0,
    )
    left_src.fluidsim.enabled = False
    right_src.fluidsim.enabled = False
    north_gate = make_collider_box(
        scene,
        (0.92, 0.00, 0.0),
        (1.08, 0.40, 0.46),
        "TidalLock North Gate",
    )
    south_gate = make_collider_box(
        scene,
        (1.32, 0.40, 0.0),
        (1.48, 0.80, 0.46),
        "TidalLock South Gate",
    )

    bpy.ops.fluidsim.initialize()
    sess = session()
    sess.solver.settings.domain_sdf_walls = True
    left = lattice((0.08, 0.08, 0.0), (0.62, 0.36, 0.32), 0.012)
    right = lattice((1.78, 0.44, 0.0), (2.32, 0.72, 0.32), 0.012)
    n_left = sess.solver.inject_particles(
        left,
        velocity=(2.2, 0.55, 0.0),
        temperature=0.2,
    )
    n_total = sess.solver.inject_particles(
        right,
        velocity=(-2.2, -0.55, 0.0),
        temperature=0.8,
    )
    log(f"tidal_lock: west {n_left:,} + east {n_total - n_left:,} particles")
    add_camera_light(scene, (1.2, -2.55, 0.95), (1.25, 0.0, 0.0))
    run_scene("tidal_lock", 72)
    from fluidSim_warp_blender51.blender import viewport

    viewport.update_outputs(scene, sess)
    bpy.context.view_layer.update()
    surface = bpy.data.objects.get("FluidSim Surface")
    particles = bpy.data.objects.get("FluidSim Particles")
    whitewater = bpy.data.objects.get("FluidSim Whitewater")
    log(
        "tidal_lock output counts before save: "
        f"surface={len(surface.data.vertices) if surface else 0}, "
        "particles="
        f"{len(particles.data.points) if particles and particles.type == 'POINTCLOUD' else 0}, "
        "whitewater="
        f"{len(whitewater.data.points) if whitewater and whitewater.type == 'POINTCLOUD' else 0}"
    )
    add_text_block(
        "README_Tidal_Lock",
        "Tidal Lock live playback notes\n"
        "==============================\n\n"
        "Two offset particle columns rush through staggered lock gates. The "
        "saved preview shows the collision braid; the source boxes remain in "
        "the scene and can be enabled for live replay after Initialize Solver.\n",
    )
    save_scene("tidal_lock")
    os.makedirs(os.path.join(DEMOS, "tidal_lock"), exist_ok=True)
    meta = {
        "scene": "tidal_lock",
        "gravity": 9.81,
        "domain": [list(lo), list(hi)],
        "voxel_size": 0.02,
        "fps": FPS,
        "spacing": 0.012,
        "injected": int(n_total),
        "west_particles": int(n_left),
        "east_particles": int(n_total - n_left),
        "source_objects": [left_src.name, right_src.name],
        "gate_objects": [north_gate.name, south_gate.name],
        "playable_live_setup": True,
        "initial_height": 0.32,
    }
    with open(os.path.join(DEMOS, "tidal_lock", "meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    bpy.ops.fluidsim.release()


# ---------------------------------------------------------------------------
# Scene 5: collider material response lab
# ---------------------------------------------------------------------------

def build_material_response():
    scene = fresh_scene()
    from mathutils import Vector
    from fluidSim_warp_blender51.blender.scene_bridge import session

    fs = scene.fluidsim
    lo, hi = (0.0, -2.4, 0.0), (7.2, 2.4, 3.2)
    make_domain(scene, lo, hi, "Material Response Domain")
    fs.voxel_size = 0.035
    fs.leaf_capacity = 16_384
    fs.max_particles = 400_000
    fs.max_whitewater = 180_000
    fs.min_substeps = 1
    fs.max_substeps = 16
    fs.flip_ratio = 0.97
    fs.trace_mode = "NONE"
    fs.relax_enabled = False
    fs.whitewater_enabled = True
    fs.show_particles = True
    fs.show_surface = True
    fs.show_whitewater = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_attribute_mode = "TEMPERATURE"
    fs.particle_display_fraction = 1.0
    fs.profiling_enabled = False
    fs.live = False
    scene.frame_start = 1
    scene.frame_end = 250
    scene.frame_set(1)

    lanes = (
        (-1.8, 0.0, 0.0, "SLIDE", (0.05, 0.75, 1.0)),
        (-0.6, 0.9, 0.0, "GRIP", (1.0, 0.25, 0.05)),
        (0.6, 0.0, 0.9, "BOUNCE", (0.9, 0.05, 0.8)),
        (1.8, 0.75, 0.75, "GRIP + BOUNCE", (0.45, 1.0, 0.12)),
    )
    # Keep these literal tuples visible as the durable demo contract.
    combinations = (
        (0.0, 0.0, "SLIDE"),
        (0.9, 0.0, "GRIP"),
        (0.0, 0.9, "BOUNCE"),
        (0.75, 0.75, "GRIP + BOUNCE"),
    )
    colliders = []
    sources = []
    batches = []
    for lane_index, (y, friction, restitution, label, color) in enumerate(lanes):
        ramp = make_material_collider(
            f"{label} Ramp F{friction:.2f} R{restitution:.2f}",
            (3.25, y, 0.82), (5.2, 0.92, 0.16),
            friction, restitution, color,
            rotation=(0.0, math.radians(11.0), 0.0))
        target = make_material_collider(
            f"{label} Impact Wall F{friction:.2f} R{restitution:.2f}",
            (4.55, y, 1.60), (0.20, 1.05, 3.20),
            friction, restitution, color)
        colliders.extend((ramp, target))

        src_lo = (0.45, y - 0.32, 1.85)
        src_hi = (1.65, y + 0.32, 2.55)
        source = make_source_box(
            scene, src_lo, src_hi, f"{label} Particle Source", emit_rate=12.0)
        # make_source_box uses a unit cube; match its visible/emission bounds.
        source.scale = tuple(src_hi[i] - src_lo[i] for i in range(3))
        source.display_type = "WIRE"
        source.fluidsim.enabled = False
        source.fluidsim.initial_velocity = (5.5, 0.0, -0.15)
        source.fluidsim.temperature = lane_index / 3.0
        sources.append(source)
        batches.append(lattice(src_lo, src_hi, 0.018))

    bpy.ops.fluidsim.initialize()
    sess = session()
    injected = 0
    for lane_index, points in enumerate(batches):
        injected = sess.solver.inject_particles(
            points, velocity=(5.5, 0.0, -0.15),
            temperature=lane_index / 3.0)
    if injected < 250_000:
        raise RuntimeError(
            f"material response demo requires >=250,000 particles, got {injected}")
    log(f"material_response: injected {injected:,} particles across four lanes")

    add_camera_light(scene, (8.8, -10.5, 6.5), (1.1, 0.0, 0.65))
    camera = scene.camera
    camera.rotation_euler = (
        Vector((3.6, 0.0, 1.1)) - camera.location
    ).to_track_quat("-Z", "Y").to_euler()
    camera.data.clip_start = 0.05
    camera.data.clip_end = 100.0
    label_material = bpy.data.materials.new("Material Response Labels")
    label_material.diffuse_color = (0.01, 0.01, 0.01, 1.0)
    label_material.use_nodes = True
    label_bsdf = label_material.node_tree.nodes.get("Principled BSDF")
    label_bsdf.inputs["Base Color"].default_value = (0.01, 0.01, 0.01, 1.0)
    label_bsdf.inputs["Roughness"].default_value = 0.8
    for y, friction, restitution, label, _color in lanes:
        bpy.ops.object.text_add(location=(2.35, y, 1.1))
        text = bpy.context.active_object
        text.name = label + " Material Label"
        text.data.body = f"{label}\nF {friction:.2f}  R {restitution:.2f}"
        text.data.align_x = "CENTER"
        text.data.align_y = "CENTER"
        text.data.size = 0.17
        text.data.extrude = 0.004
        text.rotation_euler = (0.0, math.radians(11.0), 0.0)
        text.data.materials.append(label_material)
    bpy.ops.object.text_add(location=(3.6, -2.2, 3.0))
    legend = bpy.context.active_object
    legend.name = "Material Response Legend"
    legend.data.body = (
        "PER-COLLIDER MATERIAL LAB\n"
        "BLUE  SLIDE   F 0.00 / R 0.00\n"
        "ORANGE  GRIP   F 0.90 / R 0.00\n"
        "MAGENTA  BOUNCE   F 0.00 / R 0.90\n"
        "GREEN  GRIP + BOUNCE   F 0.75 / R 0.75"
    )
    legend.data.align_x = "CENTER"
    legend.data.align_y = "CENTER"
    legend.data.size = 0.19
    legend.data.extrude = 0.006
    legend.rotation_euler = (
        camera.location - legend.location
    ).to_track_quat("Z", "Y").to_euler()
    legend_material = bpy.data.materials.new("Material Response Legend Material")
    legend_material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    legend_material.use_nodes = True
    legend_bsdf = legend_material.node_tree.nodes.get("Principled BSDF")
    legend_bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    legend_bsdf.inputs["Roughness"].default_value = 0.6
    legend.data.materials.append(legend_material)
    run_scene("material_response", 250)
    from fluidSim_warp_blender51.blender import viewport

    # Preserve the full 250-frame cache, then save a readable mid-run review
    # state instead of the settled frame-250 apparatus.
    sess.reset(scene)
    for lane_index, points in enumerate(batches):
        sess.solver.inject_particles(
            points, velocity=(5.5, 0.0, -0.15),
            temperature=lane_index / 3.0)
    for _ in range(24):
        sess.step(scene, write_cache=False)
    viewport.update_outputs(scene, sess)
    log("material_response: restored visible frame-24 review state")
    for source in sources:
        source.fluidsim.enabled = True
    add_text_block(
        "README_Material_Response",
        "Per-Collider Material Response Lab\n"
        "==================================\n\n"
        "Four identical particle releases cross colored ramps and impact "
        "impact walls with independent friction/restitution combinations. "
        "The generated simulation and timeline both span 250 frames.\n\n"
        "Blue: slide (F0/R0)\nOrange: grip (F0.9/R0)\n"
        "Magenta: bounce (F0/R0.9)\nGreen: grip + bounce (F0.75/R0.75).\n",
    )
    fs.live = False
    scene.render.resolution_x = 640
    scene.render.resolution_y = 420
    scene.render.filepath = os.path.join(DEMOS, "material_response_preview.png")
    bpy.ops.render.render(write_still=True)
    fs.live = True
    save_scene("material_response", preview=False)
    os.makedirs(os.path.join(DEMOS, "material_response"), exist_ok=True)
    meta = {
        "scene": "material_response",
        "domain": [list(lo), list(hi)],
        "voxel_size": 0.035,
        "fps": FPS,
        "frame_end": 250,
        "preview_frame": 24,
        "injected": int(injected),
        "source_objects": [source.name for source in sources],
        "collider_objects": [collider.name for collider in colliders],
        "material_combinations": [
            {"friction": f, "restitution": r, "label": label}
            for f, r, label in combinations
        ],
        "playable_live_setup": True,
    }
    with open(os.path.join(DEMOS, "material_response", "meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    bpy.ops.fluidsim.release()


# ---------------------------------------------------------------------------
# Scene 6: single-emitter collision update proof
# ---------------------------------------------------------------------------

def build_collision_update_proof():
    scene = fresh_scene()
    from mathutils import Vector
    from fluidSim_warp_blender51.blender.scene_bridge import session
    from fluidSim_warp_blender51.blender import viewport

    fs = scene.fluidsim
    lo, hi = (0.0, -1.2, 0.0), (6.0, 1.2, 3.0)
    make_domain(scene, lo, hi, "Collision Proof Domain")
    fs.voxel_size = 0.04
    fs.leaf_capacity = 4_096
    fs.max_particles = 160_000
    fs.max_whitewater = 40_000
    fs.min_substeps = 1
    fs.max_substeps = 16
    fs.flip_ratio = 0.97
    fs.trace_mode = "NONE"
    fs.relax_enabled = False
    fs.whitewater_enabled = False
    fs.show_particles = True
    fs.show_surface = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 1.0
    fs.live = False
    scene.frame_start = 1
    scene.frame_end = 72
    scene.frame_set(1)

    source_lo = (0.35, -0.36, 1.45)
    source_hi = (1.25, 0.36, 2.25)
    source = make_source_box(
        scene, source_lo, source_hi, "Collision Proof Emitter", emit_rate=8.0)
    source.scale = tuple(source_hi[i] - source_lo[i] for i in range(3))
    source.display_type = "WIRE"
    source.fluidsim.enabled = False
    source.fluidsim.initial_velocity = (4.5, 0.0, -0.25)

    collider = make_material_collider(
        "Collision Proof Ramp", (3.0, 0.0, 0.78), (2.8, 0.78, 0.14),
        0.35, 0.55, (0.08, 0.75, 0.32),
        rotation=(0.0, math.radians(13.0), 0.0))

    points = lattice(source_lo, source_hi, 0.022)
    bpy.ops.fluidsim.initialize()
    sess = session()
    injected = sess.solver.inject_particles(
        points, velocity=(4.5, 0.0, -0.25), temperature=0.5)
    log(f"collision_update_proof: injected {injected:,} particles from one emitter")

    add_camera_light(scene, (7.4, -7.8, 5.4), (1.1, 0.0, 0.7))
    camera = scene.camera
    camera.rotation_euler = (
        Vector((3.0, 0.0, 1.0)) - camera.location
    ).to_track_quat("-Z", "Y").to_euler()
    camera.data.clip_start = 0.05
    camera.data.clip_end = 100.0

    run_scene("collision_update_proof", 72)
    sess.reset(scene)
    sess.solver.inject_particles(
        points, velocity=(4.5, 0.0, -0.25), temperature=0.5)
    for _ in range(18):
        sess.step(scene, write_cache=False)
    viewport.update_outputs(scene, sess)
    source.fluidsim.enabled = True
    add_text_block(
        "README_Collision_Update_Proof",
        "Collision Update Proof\n"
        "======================\n\n"
        "One simultaneous particle batch from one emitter strikes one oblique "
        "collider. Relaxation is disabled so the saved frame exposes collision "
        "projection quality directly. Collider: friction 0.35, restitution 0.55.\n",
    )
    fs.live = False
    scene.render.resolution_x = 640
    scene.render.resolution_y = 420
    scene.render.filepath = os.path.join(
        DEMOS, "collision_update_proof_preview.png")
    bpy.ops.render.render(write_still=True)
    fs.live = True
    save_scene("collision_update_proof", preview=False)
    os.makedirs(os.path.join(DEMOS, "collision_update_proof"), exist_ok=True)
    with open(os.path.join(DEMOS, "collision_update_proof", "meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "scene": "collision_update_proof",
            "voxel_size": fs.voxel_size,
            "frame_end": 72,
            "preview_frame": 18,
            "injected": int(injected),
            "source_objects": [source.name],
            "collider_objects": [collider.name],
            "friction": 0.35,
            "restitution": 0.55,
            "relax_enabled": False,
            "playable_live_setup": True,
        }, fh, indent=2)
    bpy.ops.fluidsim.release()


# ---------------------------------------------------------------------------
# Scene 7: maelstrom basin
# ---------------------------------------------------------------------------

def build_maelstrom_basin():
    """Build the maelstrom basin through its dedicated playable-demo builder."""
    from tools import build_maelstrom_demo as maelstrom

    args = SimpleNamespace(
        output=os.path.join(DEMOS, "maelstrom_basin.blend"),
        preview=os.path.join(DEMOS, "maelstrom_basin_preview.png"),
        voxel_size=0.05,
        fill_depth=0.92,
        spin=7.0,
        particle_spacing=0.0106,
        max_particles=2_000_000,
        frames=48,
        warmup=6,
        fps_target=10.0,
        no_whitewater=False,
    )
    result = maelstrom.build(args)
    measurement = maelstrom.measure(args, result)
    maelstrom.save_demo(args, result, measurement)


def main():
    os.makedirs(DEMOS, exist_ok=True)
    builders = {
        "dam_break": build_dam_break,
        "crown_splash": build_crown_splash,
        "slosh_tank": build_slosh_tank,
        "tidal_lock": build_tidal_lock,
        "material_response": build_material_response,
        "collision_update_proof": build_collision_update_proof,
        "maelstrom_basin": build_maelstrom_basin,
    }
    targets = list(builders)
    if "--" in sys.argv:
        rest = sys.argv[sys.argv.index("--") + 1:]
        if "--list-scenes" in rest:
            print("available scenes:", " ".join(builders), flush=True)
            print("standard cache scenes:", " ".join(STANDARD_CACHE_SCENES), flush=True)
            return
        if "--standard-cache-scenes" in rest:
            targets = list(STANDARD_CACHE_SCENES)
        elif rest:
            targets = []
            for name in rest:
                if name not in builders:
                    raise SystemExit(
                        f"unknown demo scene '{name}'. Use -- --list-scenes."
                    )
                targets.append(name)
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


if __name__ == "__main__":
    main()
