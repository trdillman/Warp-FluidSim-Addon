"""Build the canonical FluidSim Warp demo scenes and export particle data.

Run:
  blender --background --python tools/build_demo_scenes.py
  blender --background --python tools/build_demo_scenes.py -- --list-scenes
  blender --background --python tools/build_demo_scenes.py -- --standard-cache-scenes
  blender --background --python tools/build_demo_scenes.py -- dam_break crown_splash

Scenes are saved as demos/<name>.blend. Large per-frame particle exports and
aggregate metrics are generated under artifacts/demo_cache/<name>/ so they do
not become source-control assets.

1. dam_break       - classic Martin & Moses dam-break column collapse.
2. crown_splash    - droplet impact into a shallow pool (Worthington crown).
3. force_fields    - all 13 native Blender force-field GPU proxies.
4. adhesive_calender - twin-roll production shear cell with viscosity A/B lanes.
5. paddle_stir      - keyframed rotating paddle in a cylindrical tank; the only
                      scene exercising the moving-collider velocity-coupling path
                      (build explicitly with `-- paddle_stir`).

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
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon

DEMOS = os.path.join(REPO, "demos")
CACHE = os.path.join(REPO, "artifacts", "demo_cache")
FPS = 24
STANDARD_CACHE_SCENES = ("dam_break", "crown_splash")
DEFAULT_SCENES = (
    "dam_break", "crown_splash", "force_fields",
    "cache_export_playback_review",
    "mesh_solid_source", "mesh_shell_source", "point_source",
    "volume_source", "liquid_sink", "height_fill",
)
CALENDER_LOW_VISCOSITY = 0.002
CALENDER_HIGH_VISCOSITY = 0.5
DAM_BREAK_RELEASE_VELOCITY = (0.7, 0.0, 0.0)
DAM_BREAK_WHITEWATER_CHURN_GAIN = 2.0


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


def shear_cell_particles(count):
    """Return exactly ``count`` evenly distributed points for one resin lane."""
    dimensions = np.asarray((3.2, 1.3, 0.7), dtype=np.float64)
    spacing = float(np.cbrt(np.prod(dimensions) / count))
    resolution = np.maximum(np.ceil(dimensions / spacing).astype(int), 1)
    axes = [
        (np.arange(size, dtype=np.float32) + 0.5) / size * dimension
        - dimension * 0.5
        for size, dimension in zip(resolution, dimensions)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    indices = np.floor(
        np.arange(count, dtype=np.float64) * len(grid) / count).astype(int)
    return np.ascontiguousarray(grid[indices], dtype=np.float32)


def fresh_scene():
    # Clear the default scene in place; read_homefile() crashes in
    # background mode with heavy third-party startup add-ons installed.
    if not hasattr(bpy.context.scene, "fluidsim"):
        enable_repo_addon(REPO, default_set=False, persistent=False)
    from fluidSim_warp_blender51.blender.scene_bridge import release_session

    scene = bpy.context.scene
    release_session(scene)
    # The scene datablock is intentionally reused, so reset the full pointer
    # group to RNA defaults before the next builder authors its settings.
    scene.property_unset("fluidsim")
    # Operator deletion skips hidden objects. Demo playback seeds are hidden,
    # so leaving them behind silently injects one demo's particles into every
    # later scene built in the same Blender process.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights,
                  bpy.data.pointclouds):
        for datablock in list(block):
            if datablock.users == 0:
                block.remove(datablock)
    while bpy.data.libraries:
        bpy.data.libraries.remove(bpy.data.libraries[0])
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
        mesh.vertices.add(len(pts))
        mesh.vertices.foreach_set("co", pts.ravel())
        mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.display_type = "BOUNDS"
    obj.hide_render = True
    obj.show_name = True
    return obj


def make_source_box(scene, lo, hi, name, *, emit_rate, emission_mode):
    """Create a tagged analytic source with an explicit truthful lifecycle."""
    from fluidSim_warp_blender51.blender import sources as _sources

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    src = bpy.context.active_object
    src.name = name
    src.location = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
    src.dimensions = tuple(hi[i] - lo[i] for i in range(3))
    _sources.tag_source(src, shape="BOX", mode="PARTICLES")
    src.fluidsim.emit_rate = emit_rate
    src.fluidsim.emission_mode = emission_mode
    src.fluidsim.source_region_kind = "ANALYTIC"
    src.fluidsim.use_object_velocity = False
    src.show_name = True
    return src




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


def run_scene(name, frames, dump_velocities=True, advance_timeline=False):
    """Step the live session; export per-frame particle data + aggregates.

    ``advance_timeline`` moves ``scene.frame_current`` in lockstep with the
    solver. Scenes with keyframed (animated) colliders need this: without an
    advancing frame the depsgraph never re-evaluates the animation, so the
    collider sits frozen, and ``object_angular_velocity`` (scene_inputs.py)
    returns zero every step because of its ``frame <= previous`` guard. Left
    off by default so the static-collider scenes stay byte-identical.
    """
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
        if advance_timeline:
            scene.frame_set(f)
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
    return series


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


def save_live_playback_ready(name, *, preview=False):
    """Save a source-driven demo at frame 1 with real live output ready."""
    from fluidSim_warp_blender51.blender import output_publication
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.live = False
    fs.cache_playback_enabled = False
    scene.frame_set(scene.frame_start)
    if bpy.ops.fluidsim.initialize() != {"FINISHED"}:
        raise RuntimeError(f"{name}: failed to initialize live playback solver")
    sess = session()
    count = sess.solver.particles.refresh_host_count()
    for _ in range(3):
        if count > 0:
            break
        if bpy.ops.fluidsim.step(frames=1) != {"FINISHED"}:
            raise RuntimeError(f"{name}: failed to seed live playback output")
        count = sess.solver.particles.refresh_host_count()
    if count <= 0:
        raise RuntimeError(f"{name}: live playback starts with zero particles")
    output_publication.update_outputs(scene, sess, force_surface=True)
    fs.live = True
    scene["fluidsim_playback_ready"] = True
    save_scene(name, preview=preview)
    log(f"{name}: saved live-playback-ready frame 1 with {count:,} particles")
    bpy.ops.fluidsim.release()


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
    fs.ww_churn_gain = DAM_BREAK_WHITEWATER_CHURN_GAIN
    fs.profiling_enabled = False
    fs.show_particles = True
    fs.show_surface = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 1.0
    fs.live = True
    scene.frame_start = 1
    scene.frame_end = 120
    scene.frame_set(1)
    src = make_source_box(
        scene,
        (0.0, 0.0, 0.0),
        (0.6, 0.6, 1.2),
        "DamBreak Source Column",
        emit_rate=8.0,
        emission_mode="INITIAL_FILL",
    )
    src.fluidsim.initial_velocity = DAM_BREAK_RELEASE_VELOCITY
    src.display_type = "WIRE"
    src.hide_render = True
    bpy.ops.mesh.primitive_uv_sphere_add(
        location=(2.41923952, 0.0, 0.0), radius=0.39267302)
    collider = bpy.context.active_object
    collider.name = "Sphere"
    from fluidSim_warp_blender51.blender import colliders as _colliders
    _colliders.tag_collider(collider, friction=1.0, restitution=0.0)
    src.fluidsim.enabled = True
    bpy.ops.fluidsim.initialize()
    sess = session()
    log("dam_break: Initial Fill will create liquid through the visible source region")
    add_camera_light(scene, (1.6, -4.4, 1.9), (1.30, 0.0, 0.0))
    series = run_scene("dam_break", 72)
    n = int(series["particle_count"][0]) if series["particle_count"] else 0
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
        "source_emitted_particles": int(n),
        "source_emission_mode": "INITIAL_FILL",
        "source_region_kind": "ANALYTIC",
        "source_target_particles_per_voxel": 8.0,
        "release_velocity": list(DAM_BREAK_RELEASE_VELOCITY),
        "whitewater_churn_gain": DAM_BREAK_WHITEWATER_CHURN_GAIN,
        "source_object": src.name,
        "source_location": list(src.location),
        "source_scale": list(src.scale),
        "collider_object": collider.name,
        "playable_live_setup": True,
    }
    with open(os.path.join(DEMOS, "dam_break", "meta.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    fs.live = False
    scene.frame_set(1)
    sess.reset(scene)
    restored = sess.solver.particles.refresh_host_count()
    if restored == 0:
        if bpy.ops.fluidsim.step(frames=1) != {"FINISHED"}:
            raise RuntimeError("dam_break playback seed step failed")
        restored = sess.solver.particles.refresh_host_count()
    if restored != n:
        raise RuntimeError(
            f"dam_break playback reset restored {restored:,}, expected {n:,}")
    fs.live = True
    scene["fluidsim_playback_ready"] = True
    scene["fluidsim_simulated_frames"] = 72
    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(DEMOS, "dam_break.blend"),
        check_existing=False)
    log(f"dam_break: saved live-playback-ready frame 1 with {restored:,} particles")
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
    scene.frame_start = 1
    scene.frame_end = 120
    scene.frame_set(1)
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
              encoding="utf-8", newline="\n") as fh:
        meta["playable_live_setup"] = True
        meta["seed_objects"] = ["Crown Pool Seed", "Crown Droplet Seed"]
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    pool_seed = make_particle_seed(scene, pool, "Crown Pool Seed")
    drop_seed = make_particle_seed(scene, drop, "Crown Droplet Seed")
    for seed in (pool_seed, drop_seed):
        seed.fluidsim.enabled = True
        seed.fluidsim.auto_inject_seed = True
        seed.hide_set(True)
    drop_seed.fluidsim.initial_velocity = (0.0, 0.0, -drop_speed)
    fs.live = False
    scene.frame_set(1)
    sess.reset(scene)
    restored = sess.solver.particles.refresh_host_count()
    if restored != n_all:
        raise RuntimeError(
            f"crown_splash playback reset restored {restored:,}, "
            f"expected {n_all:,}")
    fs.live = True
    scene["fluidsim_playback_ready"] = True
    scene["fluidsim_simulated_frames"] = 48
    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(DEMOS, "crown_splash.blend"),
        check_existing=False)
    log(
        "crown_splash: saved manual-playback-ready frame 1 with "
        f"{restored:,} particles")
    bpy.ops.fluidsim.release()


# ---------------------------------------------------------------------------
# Scene 3: native Blender force-field proxies
# ---------------------------------------------------------------------------

def build_force_fields():
    scene = fresh_scene()
    from mathutils import Vector
    from fluidSim_warp_blender51.blender import colliders as _colliders
    from fluidSim_warp_blender51.blender.scene_bridge import session

    fs = scene.fluidsim
    lo, hi = (-5.2, -3.2, 0.0), (5.2, 3.2, 3.0)
    make_domain(scene, lo, hi, "Force Fields Domain")
    fs.voxel_size = 0.12
    fs.max_particles = 300_000
    fs.min_substeps = 1
    fs.max_substeps = 8
    fs.gravity = (0.0, 0.0, -1.0)
    fs.whitewater_enabled = False
    fs.show_surface = True
    fs.show_particles = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 0.2
    fs.cache_enabled = False
    fs.live = False
    scene.frame_start = 1
    scene.frame_end = 120
    scene["fluidsim_force_fields_demo"] = True
    scene["fluidsim_force_field_types"] = 13

    src = make_source_box(
        scene, (-5.0, -3.0, 0.15), (-4.6, -2.6, 0.55),
        "Optional Live Emitter", emit_rate=0.0,
        emission_mode="CONTINUOUS_INFLOW")
    src.fluidsim.initial_velocity = (0.6, 0.0, 0.0)
    src.display_type = "WIRE"
    src.hide_render = True

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, -0.12),
                                    scale=(5.2, 3.2, 0.12))
    floor = bpy.context.active_object
    floor.name = "Force Demo Floor"
    _colliders.tag_collider(floor, friction=0.15, restitution=0.0)
    floor_material = bpy.data.materials.new("Force Demo Floor Material")
    floor_material.diffuse_color = (0.035, 0.055, 0.085, 1.0)
    floor_material.metallic = 0.35
    floor_material.roughness = 0.3
    floor.data.materials.append(floor_material)

    water_material = bpy.data.materials.new("Force Demo Water")
    water_material.diffuse_color = (0.015, 0.28, 0.95, 1.0)
    water_material.metallic = 0.15
    water_material.roughness = 0.12
    particle_material = bpy.data.materials.new("Force Demo Particles")
    particle_material.diffuse_color = (0.02, 0.72, 1.0, 1.0)
    particle_material.metallic = 0.1
    particle_material.roughness = 0.2
    fs.surface_material = water_material
    fs.particle_material = particle_material

    positions = (
        (-4.0, 2.0, 0.9), (-2.0, 2.0, 0.9), (0.0, 2.0, 0.9),
        (2.0, 2.0, 0.9), (4.0, 2.0, 0.9),
        (-4.0, 0.0, 0.9), (-2.0, 0.0, 0.9), (0.0, 0.0, 0.9),
        (2.0, 0.0, 0.9), (4.0, 0.0, 0.9),
        (-3.0, -2.0, 0.9), (0.0, -2.0, 0.9), (3.0, -2.0, 0.9),
    )
    colors = (
        (1.0, 0.15, 0.05, 1.0), (0.05, 0.45, 1.0, 1.0),
        (0.65, 0.1, 1.0, 1.0), (0.95, 0.2, 0.75, 1.0),
        (0.1, 0.85, 0.65, 1.0), (1.0, 0.75, 0.05, 1.0),
        (0.95, 0.45, 0.05, 1.0), (0.15, 0.9, 1.0, 1.0),
        (0.35, 1.0, 0.2, 1.0), (0.9, 0.9, 0.2, 1.0),
        (0.75, 0.35, 1.0, 1.0), (0.2, 0.65, 0.95, 1.0),
        (0.1, 1.0, 0.85, 1.0),
    )
    specs = (
        dict(kind="FORCE", name="Force - Repel", strength=3.0,
             falloff=1.0, minimum=0.2, note="radial"),
        dict(kind="WIND", name="Wind - Drive +X", strength=1.8,
             rotation=(0.0, math.pi * 0.5, 0.0), flow=0.2, note="axis + flow"),
        dict(kind="VORTEX", name="Vortex - Swirl", strength=2.5,
             falloff=0.5, minimum=0.2, note="axis tangent"),
        dict(kind="MAGNET", name="Magnetic - Velocity Cross Field", strength=2.0,
             note="velocity x magnetic"),
        dict(kind="HARMONIC", name="Harmonic - Damped Spring", strength=2.5,
             rest_length=0.38, harmonic_damping=0.35, note="rest 0.38 damp 0.35"),
        dict(kind="CHARGE", name="Charge - Unit Charge Attract", strength=-2.2,
             falloff=2.0, minimum=0.2, note="unit fluid charge"),
        dict(kind="LENNARDJ", name="Lennard-Jones - Bond", strength=1.5,
             minimum=0.38, note="scale 0.38"),
        dict(kind="TEXTURE", name="Texture - Curl Proxy", strength=1.6,
             size=0.55, texture_mode="CURL", note="curl mode size 0.55"),
        dict(kind="GUIDE", name="Curve Guide - Follow Path", strength=1.3,
             guide_minimum=0.8, guide_clump_amount=0.35,
             note="speed 0.8 clump 0.35"),
        dict(kind="BOID", name="Boid - Fluid Goal", strength=1.5,
             note="goal analogue"),
        dict(kind="TURBULENCE", name="Turbulence - Seeded Noise", strength=1.7,
             size=0.55, seed=41, note="size 0.55 seed 41"),
        dict(kind="DRAG", name="Drag - Linear + Quadratic", strength=0.55,
             linear_drag=0.55, quadratic_drag=0.25,
             note="linear 0.55 quadratic 0.25"),
        dict(kind="FLUID", name="Fluid Flow - Target Velocity", strength=1.6,
             rotation=(0.0, math.pi * 0.5, 0.0), flow=0.8,
             note="target +X flow 0.8"),
    )
    effectors = []
    for spec, location, color in zip(specs, positions, colors):
        bpy.ops.object.effector_add(
            type=spec["kind"], location=location,
            rotation=spec.get("rotation", (0.0, 0.0, 0.0)))
        effector = bpy.context.active_object
        effector.name = spec["name"]
        effector.show_name = True
        effector.show_in_front = True
        effector.empty_display_size = 0.45
        field = effector.field
        field.strength = spec["strength"]
        field.falloff_power = spec.get("falloff", 0.0)
        field.use_min_distance = spec.get("minimum", 0.0) > 0.0
        field.distance_min = spec.get("minimum", 0.0)
        effector.field.use_max_distance = True
        effector.field.distance_max = 0.85
        for property_name in (
                "flow", "size", "seed", "rest_length", "harmonic_damping",
                "linear_drag", "quadratic_drag", "texture_mode",
                "guide_minimum", "guide_clump_amount"):
            if property_name in spec:
                setattr(field, property_name, spec[property_name])
        if field.type == "TEXTURE":
            texture = bpy.data.textures.new("Force Demo Curl Texture", type="CLOUDS")
            texture.noise_scale = 0.35
            field.texture = texture
        if field.type == "GUIDE" and effector.type == "CURVE":
            effector.data.splines.clear()
            spline = effector.data.splines.new("POLY")
            spline.points.add(3)
            for point, coordinate in zip(
                    spline.points,
                    ((-0.65, -0.25, -0.2, 1.0), (-0.2, 0.3, 0.0, 1.0),
                     (0.25, -0.2, 0.25, 1.0), (0.65, 0.25, 0.45, 1.0))):
                point.co = coordinate
            effector.data.use_path = True
        effectors.append(effector)

        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.18,
                                             location=location)
        marker = bpy.context.active_object
        marker.name = f"{spec['name']} Marker"
        material = bpy.data.materials.new(f"{spec['name']} Color")
        material.diffuse_color = color
        material.metallic = 0.15
        material.roughness = 0.25
        marker.data.materials.append(material)

    add_camera_light(scene, (11.5, -15.0, 10.5), (0.0, 0.0, 0.8))
    camera = scene.camera
    camera.rotation_euler = (Vector((0.0, 0.0, 0.9)) - camera.location).to_track_quat(
        "-Z", "Y").to_euler()
    camera.data.clip_start = 0.05
    camera.data.clip_end = 100.0
    scene.render.engine = "BLENDER_EEVEE"
    scene.world.color = (0.025, 0.035, 0.055)

    for spec, location in zip(specs, positions):
        name = spec["name"]
        curve = bpy.data.curves.new(f"{name} Label", type="FONT")
        curve.body = (f"{name}\nStrength {spec['strength']:g} | "
                      f"{spec['note']}")
        curve.align_x = "CENTER"
        curve.size = 0.105
        curve.extrude = 0.004
        label = bpy.data.objects.new(f"{name} Label", curve)
        scene.collection.objects.link(label)
        offset = (0.0, 0.0, 0.72)
        label.location = tuple(location[i] + offset[i] for i in range(3))
        label.rotation_euler = (camera.location - label.location).to_track_quat(
            "Z", "Y").to_euler()

    bpy.ops.fluidsim.initialize()
    sess = session()
    if len(sess.solver.force_fields) != len(specs):
        raise RuntimeError(
            f"expected {len(specs)} force proxies, got "
            f"{len(sess.solver.force_fields)}")
    point_sets = [
        lattice((x - 0.3, y - 0.3, 0.45),
                (x + 0.3, y + 0.3, 1.15), 0.1)
        for x, y, _z in positions
    ]
    points = np.concatenate(point_sets, axis=0)
    injected = sess.solver.inject_particles(points, velocity=(0.6, 0.15, 0.1))
    log(f"force_fields: injected {injected:,} particles")
    bpy.ops.fluidsim.step(frames=18)
    scene.frame_set(19)
    src.fluidsim.enabled = True
    add_text_block(
        "README_Force_Fields",
        "All Blender Force Fields - FluidSim GPU Proxy Demo\n"
        "==================================================\n\n"
        "The scene contains every force-field type from Blender's Add > Force "
        "Field menu. Select a named field and open Physics Properties to review "
        "the native settings that drive FluidSim. Force, Wind, Vortex, Magnetic, "
        "Harmonic, Lennard-Jones, Turbulence, and Drag retain their physical "
        "intent. Charge assumes unit fluid charge. Texture uses a deterministic "
        "GPU field seeded by its texture. Curve Guide samples four controls. "
        "Boid is a fluid goal and Fluid Flow is a directed target velocity.\n\n"
        "The saved frame is pre-stepped. Initialize Solver, enable Live, and "
        "press Play to restart interactive CUDA playback.\n",
    )
    bpy.ops.object.select_all(action="DESELECT")
    magnetic = next(obj for obj in effectors if obj.field.type == "MAGNET")
    magnetic.select_set(True)
    bpy.context.view_layer.objects.active = magnetic
    save_scene("force_fields")
    seed = make_particle_seed(scene, points, "Force Fields Particle Seed")
    seed.fluidsim.enabled = True
    seed.fluidsim.auto_inject_seed = True
    seed.fluidsim.initial_velocity = (0.6, 0.15, 0.1)
    seed.hide_set(True)
    fs.live = False
    scene.frame_set(1)
    sess.reset(scene)
    restored = sess.solver.particles.refresh_host_count()
    if restored != injected:
        raise RuntimeError(
            f"force_fields playback reset restored {restored:,}, "
            f"expected {injected:,}")
    fs.live = True
    scene["fluidsim_playback_ready"] = True
    scene["fluidsim_simulated_frames"] = 18
    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(DEMOS, "force_fields.blend"),
        check_existing=False)
    log(f"force_fields: saved live-playback-ready frame 1 with {restored:,} particles")
    bpy.ops.fluidsim.release()


# ---------------------------------------------------------------------------
# Scene 4: 1M-particle twin-roll adhesive calender viscosity A/B
# ---------------------------------------------------------------------------

def _look_at(camera, target=(0.0, 0.0, 0.0)):
    from mathutils import Vector

    camera.rotation_euler = (
        Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def _add_review_camera(scene, name, location, target=(0.0, 0.0, 0.0),
                       lens=50.0):
    camera_data = bpy.data.cameras.new(name)
    camera_data.lens = lens
    camera_data.clip_start = 0.05
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new(name, camera_data)
    scene.collection.objects.link(camera)
    camera.location = location
    _look_at(camera, target)
    return camera


def _lane_velocity_metrics(positions, velocities):
    """Measure the streamwise velocity profile across the roll gap."""
    if len(positions) == 0:
        return {
            "particle_count": 0,
            "mean_abs_streamwise_speed": 0.0,
            "cross_flow_rms": 0.0,
            "core_speed": 0.0,
            "driven_band_speed": 0.0,
            "core_coupling_ratio": 0.0,
            "profile_gradient_rms": 0.0,
            "profile_curvature_rms": 0.0,
            "z_bin_centers": [],
            "mean_vx_profile": [],
        }
    bins = np.linspace(-0.35, 0.35, 15)
    bin_index = np.digitize(positions[:, 2], bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts = np.bincount(
        np.clip(bin_index, 0, len(centers) - 1), minlength=len(centers))
    sums = np.bincount(
        np.clip(bin_index, 0, len(centers) - 1),
        weights=velocities[:, 0], minlength=len(centers))
    valid = counts > 0
    profile = np.zeros_like(centers)
    profile[valid] = sums[valid] / counts[valid]
    valid_profile = profile[valid]
    valid_centers = centers[valid]
    gradient = (
        np.diff(valid_profile) / np.diff(valid_centers)
        if len(valid_profile) > 1 else np.zeros(1))
    curvature = (
        np.diff(valid_profile, n=2)
        if len(valid_profile) > 2 else np.zeros(1))
    core = np.abs(positions[:, 2]) <= 0.10
    driven = np.abs(positions[:, 2]) >= 0.25
    core_speed = float(np.mean(np.abs(velocities[core, 0]))) if np.any(core) else 0.0
    driven_speed = (
        float(np.mean(np.abs(velocities[driven, 0]))) if np.any(driven) else 0.0)
    return {
        "particle_count": int(len(positions)),
        "mean_abs_streamwise_speed": float(np.mean(np.abs(velocities[:, 0]))),
        "cross_flow_rms": float(np.sqrt(np.mean(
            velocities[:, 1] ** 2 + velocities[:, 2] ** 2))),
        "core_speed": core_speed,
        "driven_band_speed": driven_speed,
        "core_coupling_ratio": (
            core_speed / driven_speed if driven_speed > 1.0e-8 else 0.0),
        "profile_gradient_rms": float(np.sqrt(np.mean(gradient ** 2))),
        "profile_curvature_rms": float(np.sqrt(np.mean(curvature ** 2))),
        "z_bin_centers": [float(value) for value in valid_centers],
        "mean_vx_profile": [float(value) for value in valid_profile],
    }


def _calender_metrics(solver, frame):
    positions = solver.particle_positions().astype(np.float64, copy=False)
    velocities = solver.particle_velocities().astype(np.float64, copy=False)
    viscosity = solver.particles.attr("viscosity").numpy()[:len(positions)]
    split = 0.5 * (CALENDER_LOW_VISCOSITY + CALENDER_HIGH_VISCOSITY)
    low_mask = viscosity < split
    high_mask = viscosity >= split
    return {
        "frame": int(frame),
        "particle_count": int(len(positions)),
        "whitewater_count": int(solver.whitewater.refresh_host_count()),
        "low_viscosity": _lane_velocity_metrics(
            positions[low_mask], velocities[low_mask]),
        "high_viscosity": _lane_velocity_metrics(
            positions[high_mask], velocities[high_mask]),
    }


def _write_calender_report(metrics):
    evaluation = next(
        (sample for sample in metrics["samples"] if sample["frame"] >= 25),
        metrics["samples"][-1])
    low = evaluation["low_viscosity"]
    high = evaluation["high_viscosity"]

    def percent_change(control, changed):
        return (100.0 * (changed - control) / control
                if abs(control) > 1.0e-12 else None)

    report = {
        "comparison": "adhesive_calender_per_particle_viscosity_ab",
        "production_use_case": (
            "counter-moving roll surfaces transfer momentum into adhesive; "
            "viscosity controls how quickly shear penetrates from the driven "
            "bands into the material core"),
        "controlled_variables": (
            "both lanes run in one solver step with mirrored particle layouts, "
            "the same FLUID_FLOW and DRAG proxies, identical geometry, and the "
            "same timeline; only per-particle viscosity differs"),
        "viscosity_model": (
            "implicit grid-velocity diffusion before pressure projection"),
        "config": metrics["config"],
        "evaluation_frame": evaluation["frame"],
        "low_viscosity": low,
        "high_viscosity": high,
        "observed_effect": {
            "core_coupling_change_percent": percent_change(
                low["core_coupling_ratio"], high["core_coupling_ratio"]),
            "profile_gradient_change_percent": percent_change(
                low["profile_gradient_rms"], high["profile_gradient_rms"]),
            "profile_curvature_change_percent": percent_change(
                low["profile_curvature_rms"], high["profile_curvature_rms"]),
            "cross_flow_change_percent": percent_change(
                low["cross_flow_rms"], high["cross_flow_rms"]),
        },
        "runtime": metrics["runtime"],
    }
    report["measured_viscosity_effect"] = bool(
        high["core_coupling_ratio"] > low["core_coupling_ratio"]
        and high["profile_gradient_rms"] < low["profile_gradient_rms"])

    report_path = os.path.join(
        CACHE, "adhesive_calender_viscosity_comparison.json")
    os.makedirs(CACHE, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    markdown_path = os.path.join(DEMOS, "ADHESIVE_CALENDER_REPORT.md")
    with open(markdown_path, "w", encoding="utf-8") as fh:
        fh.write(
            "# Twin-Roll Adhesive Calender: viscosity A/B\n\n"
            "Both lanes run simultaneously under identical FLUID_FLOW shear "
            "bands and DRAG. The divider prevents viscosity averaging between "
            "lanes. Only the per-particle viscosity differs.\n\n"
            f"- Low lane viscosity: `{CALENDER_LOW_VISCOSITY:g}`\n"
            f"- High lane viscosity: `{CALENDER_HIGH_VISCOSITY:g}`\n"
            f"- Evaluation frame: `{evaluation['frame']}`\n"
            f"- Low core coupling: `{low['core_coupling_ratio']:.6f}`\n"
            f"- High core coupling: `{high['core_coupling_ratio']:.6f}`\n"
            f"- Low profile-gradient RMS: "
            f"`{low['profile_gradient_rms']:.6f}`\n"
            f"- High profile-gradient RMS: "
            f"`{high['profile_gradient_rms']:.6f}`\n"
            f"- Measured viscosity effect: "
            f"`{report['measured_viscosity_effect']}`\n\n"
            "The viscosity values are solver coefficients in scene units; this "
            "demo is a comparative shear-transfer test, not a calibrated material "
            "certificate.\n")
    log(f"wrote calender viscosity report {report_path}")
    return report


def _simulate_calender(name, frames):
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = bpy.context.scene
    solver = session().solver
    milestones = sorted({1, min(10, frames), min(25, frames), min(50, frames),
                         min(100, frames), min(150, frames),
                         min(200, frames), frames})
    samples = []
    step_ms = []
    substeps = []
    started = time.perf_counter()
    for frame in range(1, frames + 1):
        bpy.ops.fluidsim.step(frames=1)
        step_ms.append(float(scene.fluidsim.stat_step_ms))
        substeps.append(int(scene.fluidsim.stat_substeps))
        if frame in milestones:
            sample = _calender_metrics(solver, frame)
            samples.append(sample)
            log(
                f"{name} frame {frame}/{frames}: "
                f"{sample['particle_count']:,} liquid, "
                f"{sample['whitewater_count']:,} whitewater, "
                f"core coupling low/high "
                f"{sample['low_viscosity']['core_coupling_ratio']:.3f}/"
                f"{sample['high_viscosity']['core_coupling_ratio']:.3f}")
    runtime = {
        "wall_time_s": time.perf_counter() - started,
        "mean_step_ms": float(np.mean(step_ms)),
        "max_step_ms": float(np.max(step_ms)),
        "max_observed_substeps": int(max(substeps)),
    }
    return samples, runtime


def build_adhesive_calender():
    scene = fresh_scene()
    from fluidSim_warp_blender51.blender.scene_bridge import session

    name = "adhesive_calender"
    particle_count = int(os.environ.get(
        "FLUIDSIM_CALENDER_PARTICLES", "1000000"))
    frames = int(os.environ.get("FLUIDSIM_CALENDER_FRAMES", "250"))
    render_preview = os.environ.get(
        "FLUIDSIM_CALENDER_PREVIEW", "1") not in {"0", "false", "False"}
    if particle_count < 2 or particle_count % 2 or frames < 1:
        raise ValueError(
            "calender particle count must be positive and even; frames must be "
            "positive")

    workspace = bpy.context.workspace
    workspace.name = "Adhesive Calender Viscosity Review"
    fs = scene.fluidsim
    make_domain(scene, (-3.5, -2.0, -1.55), (3.5, 2.0, 1.55),
                "Adhesive Calender Domain")
    fs.domain_sdf_walls = True
    # Preserve the calibrated particles-per-voxel density when the lightweight
    # calibration override is used; the production 1M build resolves at ~0.037.
    fs.voxel_size = 0.08 * (100_000.0 / particle_count) ** (1.0 / 3.0)
    fs.max_particles = particle_count
    fs.min_substeps = 1
    fs.max_substeps = 4
    fs.use_scene_gravity = False
    fs.gravity = (0.0, 0.0, 0.0)
    fs.max_speed = 4.0
    fs.flip_ratio = 0.95
    fs.whitewater_enabled = True
    fs.max_whitewater = 500_000
    fs.ww_churn_gain = 1.0
    fs.ww_churn_threshold = 1.0
    fs.ww_emission_rate = 1.0
    fs.show_surface = True
    fs.show_particles = True
    fs.show_whitewater = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 0.12
    fs.particle_attribute_mode = "VISCOSITY"
    fs.surface_attribute_mode = "VISCOSITY"
    fs.cache_enabled = False
    fs.live = False
    fs.viscosity_enabled = True
    fs.viscosity_from_particles = True
    fs.viscosity_constant = 0.0
    fs.viscosity_iterations = 80
    fs.viscosity_tolerance = 1.0e-4
    scene.frame_start = 1
    scene.frame_end = frames
    scene.render.engine = "BLENDER_EEVEE"
    scene.world.color = (0.008, 0.012, 0.028)
    scene["fluidsim_adhesive_calender_demo"] = True
    scene["fluidsim_low_viscosity"] = CALENDER_LOW_VISCOSITY
    scene["fluidsim_high_viscosity"] = CALENDER_HIGH_VISCOSITY
    scene["fluidsim_initial_particle_count"] = particle_count
    scene["fluidsim_simulated_frames"] = frames
    scene["fluidsim_force_field_count"] = 3
    scene["fluidsim_review_clips"] = "Camera Main, Camera Top, Camera Profile"

    water = bpy.data.materials.new("Calender Adhesive Surface")
    water.diffuse_color = (0.92, 0.28, 0.035, 1.0)
    water.metallic = 0.2
    water.roughness = 0.18
    particles = bpy.data.materials.new("Calender Viscosity Particles")
    particles.diffuse_color = (1.0, 0.55, 0.04, 1.0)
    particles.metallic = 0.1
    particles.roughness = 0.16
    foam = bpy.data.materials.new("Calender Entrained Air")
    foam.diffuse_color = (0.95, 0.98, 1.0, 1.0)
    foam.roughness = 0.25
    fs.surface_material = water
    fs.particle_material = particles
    fs.whitewater_material = foam

    from fluidSim_warp_blender51.blender import colliders as _colliders

    roller_material = bpy.data.materials.new("Calender Roll Steel")
    roller_material.diffuse_color = (0.12, 0.16, 0.22, 1.0)
    roller_material.metallic = 0.85
    roller_material.roughness = 0.2
    for lane_name, lane_y in (("Low", -1.0), ("High", 1.0)):
        for roll_name, roll_z in (("Upper", 1.20), ("Lower", -1.20)):
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=48, radius=0.52, depth=1.55,
                location=(0.0, lane_y, roll_z),
                rotation=(math.pi * 0.5, 0.0, 0.0))
            roller = bpy.context.active_object
            roller.name = f"{lane_name} Viscosity {roll_name} Calender Roll"
            roller.data.materials.append(roller_material)
            roller["fluidsim_visual_role"] = "FLUID_FLOW roll proxy"

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0),
                                    scale=(3.4, 0.12, 1.3))
    divider = bpy.context.active_object
    divider.name = "Viscosity Lane Divider"
    divider.data.materials.append(roller_material)
    _colliders.tag_collider(divider, friction=0.0, restitution=0.0)

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.30))
    upper_target = bpy.context.active_object
    upper_target.name = "Upper Roll Target +X"
    upper_target.rotation_euler = (0.0, math.pi * 0.5, 0.0)
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, -0.30))
    lower_target = bpy.context.active_object
    lower_target.name = "Lower Roll Target -X"
    lower_target.rotation_euler = (0.0, -math.pi * 0.5, 0.0)

    force_specs = (
        ("FLUID", "Upper Roll Surface Velocity", (0.0, 0.0, 0.30),
         upper_target, (0.1, 0.7, 1.0, 1.0)),
        ("FLUID", "Lower Roll Surface Velocity", (0.0, 0.0, -0.30),
         lower_target, (1.0, 0.2, 0.08, 1.0)),
        ("DRAG", "Calender Chamber Drag", (0.0, 0.0, 0.0),
         None, (0.35, 1.0, 0.25, 1.0)),
    )
    effectors = []
    for kind, field_name, location, source_object, color in force_specs:
        bpy.ops.object.effector_add(type=kind, location=location)
        effector = bpy.context.active_object
        effector.name = field_name
        effector.show_name = True
        effector.show_in_front = True
        effector.empty_display_size = 0.7
        effector.field.falloff_power = 0.0
        if kind == "FLUID":
            effector.field.strength = 0.08
            effector.field.flow = 0.8
            effector.field.shape = "PLANE"
            effector.field.source_object = source_object
            effector.field.use_max_distance = True
            effector.field.distance_max = 0.20
        else:
            effector.field.strength = 0.0
            effector.field.shape = "POINT"
            effector.field.linear_drag = 0.01
            effector.field.quadratic_drag = 0.002
            effector.field.use_max_distance = False
        effectors.append(effector)

        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=2, radius=0.18,
            location=(1.9, -0.35 + 0.35 * len(effectors), location[2]))
        marker = bpy.context.active_object
        marker.name = f"{field_name} Marker"
        material = bpy.data.materials.new(f"{field_name} Marker Material")
        material.diffuse_color = color
        material.metallic = 0.2
        material.roughness = 0.2
        marker.data.materials.append(material)

    main_camera = _add_review_camera(
        scene, "Camera Main", (6.8, -8.2, 5.2), lens=54.0)
    _add_review_camera(scene, "Camera Top", (0.0, 0.0, 10.0), lens=60.0)
    _add_review_camera(
        scene, "Camera Profile", (7.5, -0.2, 0.3), lens=66.0)
    scene.camera = main_camera
    sun_data = bpy.data.lights.new("Calender Key", type="AREA")
    sun_data.energy = 1800.0
    sun_data.shape = "DISK"
    sun_data.size = 7.0
    sun = bpy.data.objects.new("Calender Key", sun_data)
    scene.collection.objects.link(sun)
    sun.location = (4.0, -5.0, 8.0)
    _look_at(sun)

    add_text_block(
        "README_Adhesive_Calender",
        "Twin-Roll Adhesive Calender - Viscosity Shear Transfer\n"
        "======================================================\n\n"
        f"This production-style shear cell uses exactly {particle_count:,} "
        f"initial particles over {frames} frames, max 4 substeps, and "
        "whitewater. The two lanes run simultaneously with mirrored particle "
        "layouts and identical forces. Native Blender FLUID_FLOW fields impose "
        "opposing roll-surface target velocities; native DRAG models weak "
        "chamber resistance.\n\n"
        f"The left lane uses per-particle viscosity "
        f"{CALENDER_LOW_VISCOSITY:g}; the right lane uses "
        f"{CALENDER_HIGH_VISCOSITY:g}. The central collider is wider than two "
        "voxels so the viscosity field cannot average between lanes. Review "
        "Camera Main, Camera Top, and Camera Profile. Particle and surface "
        "colors visualize the viscosity attribute.\n\n"
        "This is a comparative solver-coefficient study in scene units, not a "
        "calibrated material certificate. The measured velocity-profile report "
        "is embedded on disk beside this scene.\n\n"
        "The saved scene opens at frame 1 with Live Simulation enabled. Press "
        "Play manually; the two hidden lane seeds restore all particles after "
        "reopen or reset. No playback is forced on file open.\n")

    lane_count = particle_count // 2
    lane_points = shear_cell_particles(lane_count)
    low_points = lane_points.copy()
    high_points = lane_points.copy()
    low_points[:, 1] -= 1.0
    high_points[:, 1] += 1.0
    low_seed = make_particle_seed(
        scene, low_points, "Low Viscosity Adhesive Seed")
    high_seed = make_particle_seed(
        scene, high_points, "High Viscosity Adhesive Seed")
    for seed_object, viscosity in (
            (low_seed, CALENDER_LOW_VISCOSITY),
            (high_seed, CALENDER_HIGH_VISCOSITY)):
        # Keep persistence seeds out of the calibration source pipeline. They
        # are enabled only after the measured run, immediately before save.
        seed_object.fluidsim.enabled = False
        seed_object.fluidsim.auto_inject_seed = True
        seed_object.fluidsim.viscosity = viscosity
        seed_object.hide_set(True)
    bpy.ops.fluidsim.initialize()
    sess = session()
    if len(sess.solver.force_fields) != 3:
        raise RuntimeError(
            f"expected 3 force proxies, got {len(sess.solver.force_fields)}")
    low_total = sess.solver.inject_particles(
        low_points, viscosity=CALENDER_LOW_VISCOSITY)
    high_total = sess.solver.inject_particles(
        high_points, viscosity=CALENDER_HIGH_VISCOSITY)
    injected = high_total
    if low_total != lane_count or high_total != particle_count:
        raise RuntimeError(
            f"expected {particle_count:,} injected particles, got {injected:,}")
    seeded_viscosity = sess.solver.particles.attr(
        "viscosity").numpy()[:particle_count]
    low_seeded = int(np.count_nonzero(
        seeded_viscosity < 0.5 * (
            CALENDER_LOW_VISCOSITY + CALENDER_HIGH_VISCOSITY)))
    if low_seeded != lane_count:
        raise RuntimeError(
            "sequential viscosity injection did not preserve both lanes: "
            f"{low_seeded:,} low / {particle_count - low_seeded:,} high")
    del lane_points, low_points, high_points
    log(f"{name}: injected exactly {injected:,} particles")
    samples, runtime = _simulate_calender(name, frames)
    scene.frame_set(frames)
    final = samples[-1]
    scene["fluidsim_final_particle_count"] = final["particle_count"]
    scene["fluidsim_final_whitewater_count"] = final["whitewater_count"]
    scene["fluidsim_final_low_core_coupling"] = final[
        "low_viscosity"]["core_coupling_ratio"]
    scene["fluidsim_final_high_core_coupling"] = final[
        "high_viscosity"]["core_coupling_ratio"]
    scene["fluidsim_max_observed_substeps"] = runtime["max_observed_substeps"]

    metrics = {
        "scene": name,
        "config": {
            "initial_particles": particle_count,
            "frames": frames,
            "voxel_size": float(fs.voxel_size),
            "max_substeps": 4,
            "whitewater_enabled": True,
            "force_fields": ["FLUID_FLOW_UPPER", "FLUID_FLOW_LOWER", "DRAG"],
            "viscosity_enabled": True,
            "viscosity_from_particles": True,
            "low_viscosity": CALENDER_LOW_VISCOSITY,
            "high_viscosity": CALENDER_HIGH_VISCOSITY,
        },
        "samples": samples,
        "runtime": runtime,
    }
    out_dir = os.path.join(CACHE, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    report = _write_calender_report(metrics)
    scene["fluidsim_measured_viscosity_effect"] = report[
        "measured_viscosity_effect"]
    scene["fluidsim_viscosity_evaluation_frame"] = report["evaluation_frame"]
    if not report["measured_viscosity_effect"]:
        raise RuntimeError(
            "calender A/B did not measure the required viscosity shear-transfer "
            "effect")

    low_seed.fluidsim.enabled = True
    high_seed.fluidsim.enabled = True
    bpy.ops.object.select_all(action="DESELECT")
    effectors[0].select_set(True)
    bpy.context.view_layer.objects.active = effectors[0]
    save_scene(name, preview=render_preview)
    fs.live = False
    scene.frame_set(1)
    sess.reset(scene)
    restored = sess.solver.particles.refresh_host_count()
    if restored != particle_count:
        raise RuntimeError(
            f"playback seed restored {restored:,}, expected {particle_count:,}")
    fs.live = True
    scene["fluidsim_playback_ready"] = True
    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(DEMOS, f"{name}.blend"), check_existing=False)
    log(f"{name}: saved manual-playback-ready frame 1 with {restored:,} particles")
    bpy.ops.fluidsim.release()


# ---------------------------------------------------------------------------
# Phase 3 focused workflow configuration demos
# ---------------------------------------------------------------------------


def _workflow_scene(name, title):
    scene = fresh_scene()
    make_domain(scene, (-2.0, -2.0, 0.0), (2.0, 2.0, 2.5), f"{title} Domain")
    scene.fluidsim.voxel_size = 0.1
    scene.fluidsim.max_particles = 100_000
    scene.fluidsim.whitewater_enabled = False
    scene.fluidsim.live = False
    scene.frame_start = 1
    scene.frame_end = 120
    scene.frame_set(1)
    add_camera_light(scene, (4.2, -5.0, 3.4), (1.15, 0.0, 0.7))
    scene["fluidsim_phase3_workflow"] = name
    return scene


def build_mesh_solid_source():
    from fluidSim_warp_blender51.blender import sources as _sources
    scene = _workflow_scene("mesh_solid_source", "Mesh Solid Source")
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.75, location=(0,0,1.0))
    source = bpy.context.object; source.name = "Watertight Mesh Solid Emitter"
    _sources.tag_source(source); source.fluidsim.source_region_kind = "MESH_SOLID"
    source.fluidsim.mesh_object = source; source.fluidsim.emission_mode = "INITIAL_FILL"
    add_text_block("README_Mesh_Solid_Source", "Watertight Mesh Solid Volume Initial Fill. No shell fallback.")
    save_live_playback_ready("mesh_solid_source", preview=False)


def build_mesh_shell_source():
    from fluidSim_warp_blender51.blender import sources as _sources
    scene = _workflow_scene("mesh_shell_source", "Mesh Shell Source")
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=1.5, location=(0,0,1.0))
    source = bpy.context.object; source.name = "Open Mesh Surface Shell Emitter"
    _sources.tag_source(source); source.fluidsim.source_region_kind = "MESH_SHELL"
    source.fluidsim.mesh_object = source; source.fluidsim.mesh_surface_thickness = 0.12
    source.fluidsim.emission_mode = "CONTINUOUS_INFLOW"
    add_text_block("README_Mesh_Shell_Source", "Open mesh Surface Shell emitter with explicit half-width.")
    save_live_playback_ready("mesh_shell_source", preview=False)


def build_point_source():
    from fluidSim_warp_blender51.blender import sources as _sources
    scene = _workflow_scene("point_source", "Point Source")
    data = bpy.data.pointclouds.new("Rasterized Source Points"); data.resize(5)
    data.attributes["position"].data.foreach_set(
        "vector", (-.5,0,1, -.25,0,1.2, 0,0,1.4, .25,0,1.2, .5,0,1))
    source = bpy.data.objects.new("Rasterized Point Emitter", data); scene.collection.objects.link(source)
    _sources.tag_source(source); source.fluidsim.source_region_kind = "POINTS"
    source.fluidsim.point_input_object = source
    source.fluidsim.point_source_representation = "RASTERIZED_REGION"
    source.fluidsim.point_rasterize_radius = 0.18
    add_text_block("README_Point_Source", "Rasterized point top-up. Direct injection remains a separate compatibility utility.")
    save_live_playback_ready("point_source", preview=False)


def build_volume_source():
    import openvdb
    from fluidSim_warp_blender51.blender import sources as _sources
    scene = _workflow_scene("volume_source", "Volume Source")
    path = os.path.join(DEMOS, "volume_source_density.vdb")
    grid = openvdb.FloatGrid(); grid.name = "density"; accessor = grid.getAccessor()
    for x in range(8):
        for y in range(8):
            for z in range(8):
                if (x-3.5)**2 + (y-3.5)**2 + (z-3.5)**2 <= 12.0:
                    accessor.setValueOn((x,y,z), 1.0)
    openvdb.write(path, grids=[grid])
    data = bpy.data.volumes.new("Volume Source Data"); data.filepath = path
    source = bpy.data.objects.new("Fog Grid Liquid Emitter", data); scene.collection.objects.link(source)
    source.scale = (0.15, 0.15, 0.15); source.location = (-0.6,-0.6,0.4)
    _sources.tag_source(source); source.fluidsim.source_region_kind = "VOLUME_GRID"
    source.fluidsim.volume_input_object = source; source.fluidsim.volume_mask_grid = "density"
    add_text_block("README_Volume_Source", "Blender Volume density grid flattened into source leaves for Initial Fill.")
    source.fluidsim.emission_mode = "INITIAL_FILL"
    save_live_playback_ready("volume_source", preview=False)


def build_liquid_sink():
    from fluidSim_warp_blender51.blender import sources as _sources
    scene = _workflow_scene("liquid_sink", "Liquid Sink")
    make_source_box(scene, (-1,-.5,.2), (-.3,.5,1.2), "Continuous Analytic Emitter",
                    emit_rate=8.0, emission_mode="CONTINUOUS_INFLOW")
    bpy.ops.mesh.primitive_cube_add(size=0.8, location=(0.7,0,0.7))
    sink = bpy.context.object; sink.name = "Mesh Solid Liquid Sink"
    _sources.tag_sink(sink); sink.fluidsim.sink_region_kind = "MESH_SOLID"; sink.fluidsim.mesh_object = sink
    add_text_block("README_Liquid_Sink", "Mesh Solid Interior sink; no emitter lifecycle or payload controls.")
    save_live_playback_ready("liquid_sink", preview=False)


def build_height_fill():
    scene = _workflow_scene("height_fill", "Height Fill")
    image = bpy.data.images.new("Height Profile Image", width=4, height=4)
    values = []
    for y in range(4):
        for x in range(4):
            value = (x+y)/6.0; values.extend((value,value,value,1.0))
    image.pixels[:] = values
    scene.fluidsim.height_enabled = True; scene.fluidsim.height_back_mode = "PROFILE"
    scene.fluidsim.height_profile_image = image; scene.fluidsim.height_profile_scale = 0.6
    scene.fluidsim.height_emission_mode = "INITIAL_FILL"
    add_text_block("README_Height_Fill", "Profile Image height Initial Fill. Infinite mode is unsupported and absent.")
    save_live_playback_ready("height_fill", preview=False)


def build_cache_export_playback_review():
    from fluidSim_warp_blender51.blender import colliders as _colliders

    scene = _workflow_scene(
        "cache_export_playback_review", "Cache Export Playback Review")
    fs = scene.fluidsim
    fs.cache_enabled = False
    fs.cache_playback_enabled = False
    fs.cache_path = os.path.join(DEMOS, "cache_export_playback_review_cache")
    fs.cache_use_custom_range = True
    fs.cache_frame_start = 1
    fs.cache_frame_end = 48
    source = make_source_box(
        scene, (-0.8, -0.5, 0.2), (-0.2, 0.5, 1.4),
        "Cache Review Initial Fill", emit_rate=8.0,
        emission_mode="INITIAL_FILL")
    source.fluidsim.initial_velocity = (0.7, 0.0, 0.0)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(0.55, 0.0, 0.55))
    collider = bpy.context.object
    collider.name = "Cache Review Collider"
    _colliders.tag_collider(collider, friction=0.2, restitution=0.0)
    add_text_block(
        "README_Cache_Export_Playback_Review",
        "Live playback is ready at frame 1. Use Production Cache Bake to "
        "capture frames, then Use Baked Cache to review the authoritative "
        "sequence without stepping the live solver.")
    save_live_playback_ready("cache_export_playback_review", preview=False)


# ---------------------------------------------------------------------------
# Scene 5: paddle stir (moving-collider velocity coupling)
# ---------------------------------------------------------------------------

def build_paddle_stir():
    """Keyframed rotating paddle in a cylindrical tank.

    The only scene that drives the moving-collider path: the paddle's
    rotation_euler.z is keyframed, run_scene advances the timeline so the
    depsgraph re-evaluates the animation, and object_angular_velocity feeds
    collision_vel = linear + cross(angular, p - center) into the solver.
    Blades pierce the 0.32 m waterline so churn reaches the free-surface band
    and emits whitewater.
    """
    import bmesh
    from fluidSim_warp_blender51.blender import colliders as _colliders
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = fresh_scene()
    fs = scene.fluidsim
    lo, hi = (-0.60, -0.60, 0.0), (0.60, 0.60, 0.90)
    make_domain(scene, lo, hi, "PaddleStir Domain")
    fs.voxel_size = 0.012
    fs.max_particles = 900_000
    fs.min_substeps = 1
    fs.max_substeps = 24
    fs.flip_ratio = 0.97
    fs.whitewater_enabled = True
    fs.ww_churn_gain = 2.5
    fs.profiling_enabled = False
    fs.show_particles = True
    fs.particle_display_fraction = 1.0
    fs.show_whitewater = True
    scene.frame_start = 1
    scene.frame_end = 96

    # -- Tank: closed manifold tube, outward normals, open top, cavity R=0.45.
    # SOLIDIFY offset=-1 keeps the original radius-0.49 surface as the outer
    # wall and grows the 0.04 m shell inward to a 0.45 m inner radius (F2 fix:
    # a plain solid cylinder is a plug that expels the pool on substep 1).
    tank_inner_radius = 0.45
    wall_thickness = 0.04
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64, radius=tank_inner_radius + wall_thickness, depth=0.80,
        location=(0.0, 0.0, 0.40))
    tank = bpy.context.active_object
    tank.name = "PaddleStir Tank"
    bm = bmesh.new()
    bm.from_mesh(tank.data)
    bm.faces.ensure_lookup_table()
    top_face = max(bm.faces, key=lambda fc: fc.calc_center_median().z)
    bmesh.ops.delete(bm, geom=[top_face], context="FACES")
    bm.to_mesh(tank.data)
    bm.free()
    tank.data.update()
    wall = tank.modifiers.new("Tank Wall", type="SOLIDIFY")
    wall.thickness = wall_thickness
    wall.offset = -1.0
    _colliders.tag_collider(tank, friction=0.3, restitution=0.0)

    # -- Paddle: hub on the Z axis + 4 radial blade plates, joined, origin on
    # the rotation axis (center = matrix_world.translation, so an off-axis
    # origin would inject a spurious orbital term into cross(angular, p-center)).
    blade_inner, blade_outer = 0.18, 0.38
    blade_z_min, blade_z_max = 0.10, 0.42
    blade_thickness = 0.03
    blade_count = 4
    r_mid = 0.5 * (blade_inner + blade_outer)
    z_mid = 0.5 * (blade_z_min + blade_z_max)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24, radius=0.05, depth=0.50, location=(0.0, 0.0, z_mid))
    paddle = bpy.context.active_object
    paddle.name = "PaddleStir Paddle"
    blades = []
    for k in range(blade_count):
        ang = k * (2.0 * math.pi / blade_count)
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(r_mid * math.cos(ang), r_mid * math.sin(ang), z_mid))
        blade = bpy.context.active_object
        blade.dimensions = (blade_outer - blade_inner, blade_thickness,
                            blade_z_max - blade_z_min)
        bpy.context.view_layer.update()
        blade.rotation_euler = (0.0, 0.0, ang)
        blades.append(blade)
    bpy.ops.object.select_all(action="DESELECT")
    paddle.select_set(True)
    for blade in blades:
        blade.select_set(True)
    bpy.context.view_layer.objects.active = paddle
    bpy.ops.object.join()
    scene.cursor.location = (0.0, 0.0, 0.0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    bpy.context.view_layer.update()
    _colliders.tag_collider(paddle, friction=0.5, restitution=0.0)

    # -- Keyframe rotation_euler.z: 0 -> ~100 RPM by f24, held to f72, back to
    # 0 by f96. Every frame is a keyframe so the integer-frame solver samples
    # equal the analyzer's np.interp reconstruction exactly; LINEAR keeps the
    # per-frame angular velocity clean.
    omega_peak = 2.0 * math.pi * 100.0 / 60.0
    spinup_end, steady_end, total = 24, 72, 96
    dt = 1.0 / FPS
    prefs = bpy.context.preferences.edit
    old_interp = prefs.keyframe_new_interpolation_type
    prefs.keyframe_new_interpolation_type = "LINEAR"
    angle = 0.0
    rotation_keyframes = []
    paddle.rotation_euler = (0.0, 0.0, 0.0)
    try:
        for f in range(1, total + 1):
            if f <= spinup_end:
                rate = omega_peak * (f - 1) / (spinup_end - 1)
            elif f <= steady_end:
                rate = omega_peak
            else:
                rate = omega_peak * (total - f) / (total - steady_end)
            if f > 1:
                angle += rate * dt
            paddle.rotation_euler[2] = angle
            paddle.keyframe_insert("rotation_euler", index=2, frame=f)
            rotation_keyframes.append([f, angle])
    finally:
        prefs.keyframe_new_interpolation_type = old_interp

    bpy.ops.fluidsim.initialize()
    sess = session()
    pool_depth = 0.32
    pool = lattice((-tank_inner_radius, -tank_inner_radius, 0.05),
                   (tank_inner_radius, tank_inner_radius, pool_depth), 0.010)
    radial = np.hypot(pool[:, 0], pool[:, 1])
    pool = pool[radial < (tank_inner_radius - 1.5 * fs.voxel_size)]
    n_pool = sess.solver.inject_particles(pool)
    log(f"paddle_stir: pool {n_pool:,} particles, peak {omega_peak:.2f} rad/s")
    add_camera_light(scene, (0.95, -1.30, 1.05), (1.10, 0.0, 0.62))
    series = run_scene("paddle_stir", total, advance_timeline=True)
    if not series["particle_count"] or min(series["particle_count"]) <= 0:
        raise RuntimeError("paddle_stir measured run lost its seeded pool")
    add_text_block(
        "README_Paddle_Stir",
        "Keyframed rotating paddle (moving collider). run_scene advances the "
        "timeline so object_angular_velocity is nonzero; blades pierce the "
        "waterline to emit whitewater. Scrub 1-96 to see the paddle turn.")
    save_scene("paddle_stir")
    os.makedirs(os.path.join(DEMOS, "paddle_stir"), exist_ok=True)
    meta = {
        "scene": "paddle_stir",
        "gravity": 9.81,
        "fps": FPS,
        "domain": [list(lo), list(hi)],
        "voxel_size": 0.012,
        "tank_center": [0.0, 0.0, 0.40],
        "tank_inner_radius": tank_inner_radius,
        "tank_height": 0.80,
        "tank_wall_thickness": wall_thickness,
        "pool_depth": pool_depth,
        "pool_particles": int(n_pool),
        "paddle_origin": [0.0, 0.0, 0.0],
        "paddle_axis": [0.0, 0.0, 1.0],
        "blade_count": blade_count,
        "blade_inner_radius": blade_inner,
        "blade_outer_radius": blade_outer,
        "blade_z_min": blade_z_min,
        "blade_z_max": blade_z_max,
        "blade_thickness": blade_thickness,
        "rotation_keyframes": rotation_keyframes,
        "peak_rad_per_s": omega_peak,
        "spinup_end_frame": spinup_end,
        "steady_end_frame": steady_end,
        "whitewater_churn_gain": 2.5,
        "collider_objects": ["PaddleStir Tank", "PaddleStir Paddle"],
    }
    with open(os.path.join(DEMOS, "paddle_stir", "meta.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        meta["playable_live_setup"] = True
        meta["seed_objects"] = ["Paddle Stir Pool Seed"]
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    seed = make_particle_seed(scene, pool, "Paddle Stir Pool Seed")
    seed.fluidsim.enabled = True
    seed.fluidsim.auto_inject_seed = True
    seed.hide_set(True)
    fs.live = False
    scene.frame_set(1)
    sess.reset(scene)
    restored = sess.solver.particles.refresh_host_count()
    if restored != n_pool:
        raise RuntimeError(
            f"paddle_stir playback reset restored {restored:,}, "
            f"expected {n_pool:,}")
    fs.live = True
    scene["fluidsim_playback_ready"] = True
    scene["fluidsim_simulated_frames"] = total
    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(DEMOS, "paddle_stir.blend"),
        check_existing=False)
    log(f"paddle_stir: saved live-playback-ready frame 1 with {restored:,} particles")
    bpy.ops.fluidsim.release()


# ---------------------------------------------------------------------------

def main():
    os.makedirs(DEMOS, exist_ok=True)
    enable_repo_addon(REPO, default_set=False, persistent=False)
    builders = {
        "dam_break": build_dam_break,
        "crown_splash": build_crown_splash,
        "force_fields": build_force_fields,
        "cache_export_playback_review": build_cache_export_playback_review,
        "adhesive_calender": build_adhesive_calender,
        "mesh_solid_source": build_mesh_solid_source,
        "mesh_shell_source": build_mesh_shell_source,
        "point_source": build_point_source,
        "volume_source": build_volume_source,
        "liquid_sink": build_liquid_sink,
        "height_fill": build_height_fill,
        "paddle_stir": build_paddle_stir,
    }
    targets = list(DEFAULT_SCENES)
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
