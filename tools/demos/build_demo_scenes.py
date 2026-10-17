"""Build the canonical FluidSim Warp demo scenes and export particle data.

Run:
  blender --background --factory-startup --python tools/demos/build_demo_scenes.py
  blender --background --factory-startup --python tools/demos/build_demo_scenes.py -- --list-scenes
  blender --background --factory-startup --python tools/demos/build_demo_scenes.py -- --standard-cache-scenes
  blender --background --factory-startup --python tools/demos/build_demo_scenes.py -- dam_break crown_splash

Scenes are saved as demos/<name>.blend. Large per-frame particle exports and
aggregate metrics are generated under artifacts/demo_cache/<name>/ so they do
not become source-control assets.

1. dam_break       - a reservoir held by dam walls releases at once onto a
                     rocky cove (Martin & Moses front benchmark on the apron).
2. crown_splash    - droplet impact into a shallow pool (Worthington crown).
3. paddle_stir     - keyframed rotating paddle in a cylindrical tank; the only
                     scene exercising the moving-collider velocity-coupling path
                     (build explicitly with `-- paddle_stir`).

The exported data is analyzed offline by tools/demos/analyze_demos.py.
"""

import json
import math
import os
import sys
import time

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

DEMOS = os.path.join(REPO, "demos")
CACHE = os.path.join(REPO, "artifacts", "demo_cache")
FPS = 24
STANDARD_CACHE_SCENES = ("dam_break", "crown_splash")
DEFAULT_SCENES = ("dam_break", "crown_splash")
DAM_BREAK_WHITEWATER_CHURN_GAIN = 2.0
CROWN_SPLASH_WW_CHURN_THRESHOLD = 0.05
CROWN_SPLASH_WW_EMISSION_RATE = 200.0
# 4.0 two-way rigid coupling: a cork (density ratio 0.3) floats in the crown
# pool away from the impact and is pushed by the splash. Its centre sits so a
# 30% draft starts at rest. FLUIDSIM_CROWN_CORK=0 builds without it (the
# per-frame budget A/B only; the shipped scene always has the cork).
CROWN_CORK_NAME = "Crown Cork"
CROWN_CORK_SIZE = 0.06
CROWN_CORK_DENSITY = 300.0
CROWN_CORK_CENTER = (0.62, 0.40, 0.05 + CROWN_CORK_SIZE * (0.5 - 0.3))


def log(msg):
    print(f"[demo] {msg}", flush=True)


def admitted_total(result, *, label):
    """Return the cumulative count from a successful structured injection."""
    if not result.admitted:
        raise RuntimeError(
            f"{label} particle injection rejected: required={result.required:,}, "
            f"available={result.available:,}, growth={result.growth_status or 'none'}"
        )
    return int(result.total)


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


def set_resolution(ds, lo, hi, voxel_size):
    """Author the grid as Resolution Divisions along the longest domain axis.

    ``voxel_size`` is the cell size the demo was tuned at; it is converted to the
    nearest whole divisions so the saved scene states its resolution explicitly.
    """
    from fluidSim_warp_blender51.sim.quality_contract import divisions_for_voxel_size

    ds.resolution_divisions = divisions_for_voxel_size(
        voxel_size, max(hi[i] - lo[i] for i in range(3)))


def domain_settings(scene):
    """Return the designated domain object's settings group.

    Domain settings live only on ``Object.fluidsim_domain`` (FLUIDSIM_PG_domain);
    ``scene.fluidsim`` holds the scene-level group and has none of them.
    """
    dom = scene.fluidsim.domain_object
    if dom is None:
        raise RuntimeError("demo builder: scene has no designated domain object")
    return dom.fluidsim_domain


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


def run_scene(name, frames, dump_velocities=True, advance_timeline=False,
              on_frame=None):
    """Step the live session; export per-frame particle data + aggregates.

    ``advance_timeline`` moves ``scene.frame_current`` in lockstep with the
    solver. Scenes with keyframed (animated) colliders need this: without an
    advancing frame the depsgraph never re-evaluates the animation, so the
    collider sits frozen, and ``object_angular_velocity`` (scene_inputs.py)
    returns zero every step because of its ``frame <= previous`` guard. Left
    off by default so the static-collider scenes stay byte-identical.

    ``on_frame(f)`` runs after frame ``f`` is stepped and exported.
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
        if on_frame is not None:
            on_frame(f)
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


DAM_BREAK_PREVIEW_FRAME = 24
DAM_BREAK_MATERIAL_COLORS = {
    "DamBreak Concrete": (0.55, 0.55, 0.53, 1.0),
    "DamBreak Sand": (0.76, 0.66, 0.47, 1.0),
    "DamBreak Rock": (0.33, 0.31, 0.29, 1.0),
    "DamBreak Timber": (0.45, 0.30, 0.18, 1.0),
}
DAM_BREAK_WATER_COLOR = (0.04, 0.24, 0.50, 1.0)
DAM_BREAK_SPRAY_COLOR = (0.18, 0.45, 0.75, 1.0)
DAM_BREAK_FOAM_COLOR = (0.95, 0.97, 1.0, 1.0)
DAM_BREAK_CAMERA = ((7.4, -3.1, 3.0), (2.5, 0.0, 0.3), 40.0)
# Share of preview pixels that must read as water (blue well above red).
DAM_BREAK_PREVIEW_MIN_WATER = 0.02


def demo_material(name, color, *, roughness, transmission=0.0):
    """A node material whose viewport color matches its render base color."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        if transmission and "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = transmission
            bsdf.inputs["IOR"].default_value = 1.33
    mat.diffuse_color = color
    mat.roughness = roughness
    return mat


def make_mesh_object(scene, data, materials):
    """Link a closed mesh built by tools/demos/dam_break_cove.py."""
    mesh = bpy.data.meshes.new(data.name)
    mesh.from_pydata(data.verts.tolist(), [], [list(f) for f in data.faces])
    if mesh.validate(verbose=True):
        raise RuntimeError(f"{data.name}: generated mesh needed repair")
    for mat_name in data.materials:
        mesh.materials.append(materials[mat_name])
    mesh.polygons.foreach_set("material_index", data.face_materials)
    mesh.update()
    obj = bpy.data.objects.new(data.name, mesh)
    scene.collection.objects.link(obj)
    return obj


def look_at(obj, target):
    from mathutils import Vector

    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def use_material_shading(scene):
    """Solid/Workbench shading by material color, saved in every 3D view."""
    spaces = [space
              for screen in bpy.data.screens for area in screen.areas
              if area.type == "VIEW_3D" for space in area.spaces
              if space.type == "VIEW_3D"]
    for shading in [scene.display.shading] + [s.shading for s in spaces]:
        shading.light = "STUDIO"
        shading.color_type = "MATERIAL"
        shading.show_shadows = True
        shading.show_cavity = True
    for space in spaces:
        space.shading.type = "SOLID"
        space.region_3d.view_perspective = "CAMERA"
    scene.render.engine = "BLENDER_WORKBENCH"
    if scene.world is not None:
        scene.world.color = (0.66, 0.70, 0.73)


def render_preview(name, *, min_water_fraction):
    """Render the current solver state and require visible water in it."""
    from fluidSim_warp_blender51.blender import playback

    scene = bpy.context.scene
    scene.render.resolution_x = 640
    scene.render.resolution_y = 420
    path = os.path.join(DEMOS, f"{name}_preview.png")
    scene.render.filepath = path
    # A still render re-sends the current frame; while live that would reset
    # the running simulation back to frame 1.
    with playback.suppress_frame_handlers():
        bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(path, check_existing=False)
    try:
        px = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(image)
    rgb = px.reshape(-1, 4)[:, :3]
    water = float(np.mean((rgb[:, 2] - rgb[:, 0] > 0.2) & (rgb[:, 2] > rgb[:, 1])))
    log(f"{name} preview rendered: {water:.1%} water pixels")
    if water < min_water_fraction:
        raise RuntimeError(
            f"{name} preview shows {water:.2%} water pixels, "
            f"need {min_water_fraction:.0%}")
    return water


# ---------------------------------------------------------------------------
# Scene 1: dam break
# ---------------------------------------------------------------------------

def build_dam_break():
    scene = fresh_scene()
    from fluidSim_warp_blender51.blender import colliders as _colliders
    from fluidSim_warp_blender51.blender.scene_bridge import session
    from tools.demos import dam_break_cove as cove

    fs = scene.fluidsim
    lo, hi = cove.DOMAIN_LO, cove.DOMAIN_HI
    make_domain(scene, lo, hi, "DamBreak Domain")
    ds = domain_settings(scene)
    set_resolution(ds, lo, hi, cove.VOXEL)
    ds.max_particles = 600_000
    ds.min_substeps = 1
    ds.max_substeps = 16
    ds.whitewater_enabled = True
    ds.ww_churn_gain = DAM_BREAK_WHITEWATER_CHURN_GAIN
    ds.profiling_enabled = False
    fs.show_particles = True
    fs.show_surface = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 1.0
    fs.surface_material = demo_material(
        "DamBreak Water", DAM_BREAK_WATER_COLOR, roughness=0.05, transmission=0.6)
    fs.particle_material = demo_material(
        "DamBreak Spray", DAM_BREAK_SPRAY_COLOR, roughness=0.2)
    fs.whitewater_material = demo_material(
        "DamBreak Foam", DAM_BREAK_FOAM_COLOR, roughness=0.6)
    fs.live = True
    scene.frame_start = 1
    scene.frame_end = 120
    scene.frame_set(1)
    src = make_source_box(
        scene,
        cove.RESERVOIR_LO,
        cove.RESERVOIR_HI,
        "DamBreak Source Column",
        emit_rate=8.0,
        emission_mode="INITIAL_FILL",
    )
    src.display_type = "WIRE"
    src.hide_render = True
    materials = {name: demo_material(name, color, roughness=0.85)
                 for name, color in DAM_BREAK_MATERIAL_COLORS.items()}
    colliders = []
    for data in cove.collider_meshes():
        obj = make_mesh_object(scene, data, materials)
        _colliders.tag_collider(obj, friction=data.friction, restitution=0.0)
        colliders.append(obj)
        if data.name in ("DamBreak Cove", "DamBreak Sea Stack"):
            # Natural rock reads smooth; the box sides and base stay sharp.
            obj.data.shade_smooth()
            obj.data.set_sharp_from_angle(angle=math.radians(40.0))
    make_mesh_object(scene, cove.apron_mesh(), materials)
    make_mesh_object(scene, cove.wall_mesh(), materials)
    src.fluidsim.enabled = True
    bpy.ops.fluidsim.initialize()
    sess = session()
    log("dam_break: Initial Fill will create liquid through the visible source region")
    cam_loc, cam_target, cam_lens = DAM_BREAK_CAMERA
    add_camera_light(scene, cam_loc, (0.0, 0.0, 0.0))
    look_at(scene.camera, cam_target)
    scene.camera.data.lens = cam_lens
    use_material_shading(scene)

    def preview(frame):
        if frame == DAM_BREAK_PREVIEW_FRAME:
            render_preview("dam_break",
                           min_water_fraction=DAM_BREAK_PREVIEW_MIN_WATER)

    series = run_scene("dam_break", 72, on_frame=preview)
    n = int(series["particle_count"][0]) if series["particle_count"] else 0
    src.fluidsim.enabled = True
    add_text_block(
        "README_Dam_Break_Playable",
        "Dam Break live playback notes\n"
        "=============================\n\n"
        "A reservoir held by two concrete dam walls releases at once. The "
        "water runs across the apron into a rocky cove: a sand beach between "
        "two headlands, a sea stack and a timber jetty. Each of those is a "
        "mesh collider; the concrete training walls and apron slab only show "
        "the domain boundary.\n\n"
        "Press Play: Live Simulation is on and frame 1 holds the full "
        "reservoir. The 'DamBreak Source Column' stays enabled, so Reset or "
        "Reinitialize refills the reservoir from any timeline frame.\n",
    )
    save_scene("dam_break", preview=False)
    meta = {
        "scene": "dam_break",
        "gravity": 9.81,
        "column_height": cove.RESERVOIR_HI[2] - cove.RESERVOIR_LO[2],
        "column_width": cove.RESERVOIR_HI[0] - cove.RESERVOIR_LO[0],
        "reservoir_width": cove.RESERVOIR_HI[1] - cove.RESERVOIR_LO[1],
        "apron_end_x": cove.APRON_END_X,
        "domain": [list(lo), list(hi)],
        "voxel_size": cove.VOXEL,
        "fps": FPS,
        "spacing": cove.SPACING,
        "source_emitted_particles": int(n),
        "source_emission_mode": "INITIAL_FILL",
        "source_region_kind": "ANALYTIC",
        "source_target_particles_per_voxel": 8.0,
        "whitewater_churn_gain": DAM_BREAK_WHITEWATER_CHURN_GAIN,
        "source_object": src.name,
        "source_location": list(src.location),
        "source_scale": list(src.scale),
        "collider_objects": [obj.name for obj in colliders],
        "preview_frame": DAM_BREAK_PREVIEW_FRAME,
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

def add_crown_cork(scene):
    """Author the crown_splash Dynamic collider (None when opted out)."""
    if os.environ.get("FLUIDSIM_CROWN_CORK", "1").strip() == "0":
        log("crown_splash: building WITHOUT the cork (budget A/B)")
        return None
    from fluidSim_warp_blender51.blender import colliders as _colliders

    bpy.ops.mesh.primitive_cube_add(size=CROWN_CORK_SIZE, location=CROWN_CORK_CENTER)
    cork = bpy.context.active_object
    cork.name = CROWN_CORK_NAME
    _colliders.tag_collider(cork, friction=0.0, restitution=0.0,
                            transfer_velocity=False)
    cork.fluidsim.enabled = True
    cork.fluidsim.collider_dynamic = True
    cork.fluidsim.collider_density = CROWN_CORK_DENSITY
    bpy.context.view_layer.update()
    return cork


def build_crown_splash():
    scene = fresh_scene()
    from fluidSim_warp_blender51.blender.scene_bridge import session

    fs = scene.fluidsim
    lo, hi = (0.0, 0.0, 0.0), (0.8, 0.8, 0.5)
    make_domain(scene, lo, hi, "CrownSplash Domain")
    ds = domain_settings(scene)
    set_resolution(ds, lo, hi, 0.01)
    ds.max_particles = 600_000
    ds.min_substeps = 1
    ds.max_substeps = 24
    ds.flip_ratio = 0.97
    ds.whitewater_enabled = True
    ds.ww_churn_gain = 2.0
    # Measured on the saved scene: peak churn is 0.25-0.35 (run-to-run GPU
    # variation), p99.9 ~0.19, so the 0.5 default never emitted. 0.05 keeps a
    # ~5x margin under the lowest observed peak and emits from frame 4-5 in
    # every run; ww_min_speed still keeps the resting pool foam-free. Rate 200
    # gives ~5k live whitewater without saturating the per-substep emission
    # probability (peak ~0.18 of the clamp).
    ds.ww_churn_threshold = CROWN_SPLASH_WW_CHURN_THRESHOLD
    ds.ww_emission_rate = CROWN_SPLASH_WW_EMISSION_RATE
    ds.profiling_enabled = False
    fs.show_particles = True
    fs.particle_display_fraction = 1.0
    scene.frame_start = 1
    scene.frame_end = 120
    cork = add_crown_cork(scene)
    scene.frame_set(1)
    bpy.ops.fluidsim.initialize()
    sess = session()
    pool_depth = 0.05
    pool = lattice((0.0, 0.0, 0.0), (0.8, 0.8, pool_depth), 0.005)
    if cork is not None:
        # No liquid seeded inside the cork's authored volume.
        half = CROWN_CORK_SIZE * 0.5
        inside = np.all(
            np.abs(pool - np.asarray(CROWN_CORK_CENTER, dtype=pool.dtype)) < half,
            axis=1)
        pool = pool[~inside]
    n_pool = admitted_total(
        sess.solver.inject_particles(pool), label="crown_splash pool")
    drop_center = (0.4, 0.4, 0.2)
    drop_radius = 0.045
    drop_speed = 3.5
    drop = sphere_lattice(drop_center, drop_radius, 0.005)
    n_all = admitted_total(
        sess.solver.inject_particles(drop, velocity=(0.0, 0.0, -drop_speed)),
        label="crown_splash droplet",
    )
    log(f"crown_splash: pool {n_pool:,} + droplet {n_all - n_pool:,} particles")
    add_camera_light(scene, (0.4, -1.05, 0.42), (1.35, 0.0, 0.0))
    series = run_scene("crown_splash", 48)
    peak_whitewater = max(series["whitewater_count"])
    if peak_whitewater == 0:
        raise RuntimeError(
            "crown_splash: whitewater is enabled but emitted nothing in 48 "
            "frames; the churn threshold is above the scene's reachable churn")
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
        "dynamic_collider": None if cork is None else {
            "name": CROWN_CORK_NAME,
            "size": CROWN_CORK_SIZE,
            "density": CROWN_CORK_DENSITY,
            "density_ratio": CROWN_CORK_DENSITY / 1000.0,
            "center": list(CROWN_CORK_CENTER),
        },
        "whitewater": {
            "churn_gain": 2.0,
            "churn_threshold": CROWN_SPLASH_WW_CHURN_THRESHOLD,
            "emission_rate": CROWN_SPLASH_WW_EMISSION_RATE,
            "build_peak_count": int(peak_whitewater),
            "build_first_frame": next(
                f for f, n in zip(series["frames"], series["whitewater_count"])
                if n > 0),
        },
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
# Scene 3: paddle stir (moving-collider velocity coupling)
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

    # Calibration envelope overrides. Defaults reproduce the shipped bake
    # exactly; a non-empty FLUIDSIM_PADDLE_VARIANT redirects every output so
    # variant bakes never touch the shipped demo .blend/meta.json. Used by the
    # spin-up threshold calibration (TASK-POLICY-DRIFT-016) to sweep
    # resolution, particle count, duration and paddle speed.
    variant = os.environ.get("FLUIDSIM_PADDLE_VARIANT", "").strip()
    voxel = float(os.environ.get("FLUIDSIM_PADDLE_VOXEL", "0.012"))
    seed_spacing = float(os.environ.get("FLUIDSIM_PADDLE_SEED_SPACING", "0.010"))
    rpm = float(os.environ.get("FLUIDSIM_PADDLE_RPM", "100.0"))
    frame_scale = float(os.environ.get("FLUIDSIM_PADDLE_FRAME_SCALE", "1.0"))
    scene_name = f"paddle_stir__{variant}" if variant else "paddle_stir"

    scene = fresh_scene()
    fs = scene.fluidsim
    lo, hi = (-0.60, -0.60, 0.0), (0.60, 0.60, 0.90)
    make_domain(scene, lo, hi, "PaddleStir Domain")
    ds = domain_settings(scene)
    set_resolution(ds, lo, hi, voxel)
    ds.max_particles = 900_000
    ds.min_substeps = 1
    ds.max_substeps = 24
    ds.flip_ratio = 0.97
    ds.whitewater_enabled = True
    ds.ww_churn_gain = 2.5
    ds.profiling_enabled = False
    fs.show_particles = True
    fs.particle_display_fraction = 1.0
    fs.show_whitewater = True
    scene.frame_start = 1
    scene.frame_end = max(8, int(round(96 * frame_scale)))

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
    omega_peak = 2.0 * math.pi * rpm / 60.0
    spinup_end, steady_end, total = (max(2, int(round(24 * frame_scale))),
                                     max(4, int(round(72 * frame_scale))),
                                     max(8, int(round(96 * frame_scale))))
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
                   (tank_inner_radius, tank_inner_radius, pool_depth),
                   seed_spacing)
    radial = np.hypot(pool[:, 0], pool[:, 1])
    pool = pool[radial < (tank_inner_radius - 1.5 * ds.voxel_size)]
    n_pool = admitted_total(
        sess.solver.inject_particles(pool), label=f"{scene_name} pool")
    log(f"{scene_name}: pool {n_pool:,} particles, peak {omega_peak:.2f} rad/s")
    add_camera_light(scene, (0.95, -1.30, 1.05), (1.10, 0.0, 0.62))
    series = run_scene(scene_name, total, advance_timeline=True)
    if not series["particle_count"] or min(series["particle_count"]) <= 0:
        raise RuntimeError(f"{scene_name} measured run lost its seeded pool")
    add_text_block(
        "README_Paddle_Stir",
        "Keyframed rotating paddle (moving collider). run_scene advances the "
        "timeline so object_angular_velocity is nonzero; blades pierce the "
        "waterline to emit whitewater. Scrub 1-96 to see the paddle turn.")
    if not variant:
        save_scene("paddle_stir")
    os.makedirs(os.path.join(DEMOS, scene_name), exist_ok=True)
    meta = {
        "scene": scene_name,
        "gravity": 9.81,
        "fps": FPS,
        "domain": [list(lo), list(hi)],
        "voxel_size": voxel,
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
    with open(os.path.join(DEMOS, scene_name, "meta.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        meta["playable_live_setup"] = True
        meta["seed_objects"] = ["Paddle Stir Pool Seed"]
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    if variant:
        # Calibration bake: the cache and meta are the whole deliverable. The
        # live-playback packaging below rewrites the shipped demo .blend, so a
        # variant stops here and leaves the shipped assets untouched.
        log(f"{scene_name}: variant bake complete ({n_pool:,} particles, "
            f"{total} frames, voxel {voxel}, {rpm:.1f} RPM)")
        bpy.ops.fluidsim.release()
        return
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
# Scenes 4-5: two-way rigid coupling (4.0 Dynamic colliders)
# ---------------------------------------------------------------------------

def add_dynamic_collider(obj, *, density, drag=1.0, lock_axes=None):
    """Tag ``obj`` as a Dynamic collider the way the Object panel does."""
    from fluidSim_warp_blender51.blender import colliders as _colliders

    _colliders.tag_collider(obj, friction=0.0, restitution=0.0,
                            transfer_velocity=False)
    obj.fluidsim.enabled = True
    obj.fluidsim.collider_dynamic = True
    obj.fluidsim.collider_density = density
    obj.fluidsim.collider_drag = drag
    if lock_axes is not None:
        obj.fluidsim.collider_lock_axes = lock_axes
    bpy.context.view_layer.update()
    return obj


def body_states(sess):
    """{object name: state record} of the live solver's Dynamic colliders."""
    from fluidSim_warp_blender51.blender.scene_inputs import persistent_source_key
    from fluidSim_warp_blender51.sim import rigid_scene

    by_uid = {persistent_source_key(obj): obj.name
              for obj in rigid_scene.dynamic_collider_objects(
                  bpy.context.scene.objects)}
    states = {}
    for record in sess.solver.rigid_body_states():
        states[by_uid.get(str(record["object_uid"]), str(record["object_uid"]))] = record
    return states


def save_playable(name, scene, frames):
    """Reset to frame 1 (clears the published poses) and save live-ready."""
    from fluidSim_warp_blender51.blender.scene_bridge import session

    fs = scene.fluidsim
    fs.live = False
    scene.frame_set(1)
    session().reset(scene)
    fs.live = True
    scene["fluidsim_playback_ready"] = True
    scene["fluidsim_simulated_frames"] = frames
    bpy.ops.wm.save_as_mainfile(
        filepath=os.path.join(DEMOS, f"{name}.blend"), check_existing=False)
    log(f"{name}: saved live-playback-ready frame 1")
    bpy.ops.fluidsim.release()


def write_meta(name, meta):
    os.makedirs(os.path.join(DEMOS, name), exist_ok=True)
    meta["playable_live_setup"] = True
    meta["seed_objects"] = []
    with open(os.path.join(DEMOS, name, "meta.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")


# Density Ladder: four flat blocks of rising density dropped onto a still
# pool. Archimedes says each settles with draft = (density / 1000) x height
# and the 1500 block sinks to the floor. Width/height 2 keeps every block
# stable flat side up (a box floats upright when width/height > 1.23). The
# sinker's rotation is locked: a falling plate flutters, and 4.0 floor contact
# only stops a body (it has no tipping torque), so an unlocked sinker can come
# to rest balanced on a corner.
LADDER_DOMAIN = ((0.0, 0.0, 0.0), (0.96, 0.26, 0.36))
LADDER_VOXEL = 0.01
LADDER_POOL_DEPTH = 0.18
LADDER_BLOCK = (0.12, 0.12, 0.06)
LADDER_DROP_GAP = 0.01        # block bottoms start this far above the pool
LADDER_DENSITIES = (250.0, 500.0, 750.0, 1500.0)
LADDER_SINKER_LOCK = (False, False, False, True, True, True)
LADDER_FRAMES = 96
LADDER_COLORS = ((0.86, 0.74, 0.52, 1.0), (0.66, 0.47, 0.27, 1.0),
                 (0.42, 0.26, 0.15, 1.0), (0.35, 0.36, 0.40, 1.0))


def ladder_block_name(density):
    return f"Ladder Block {int(density)}"


def ladder_expected_drafts():
    """Settled draft per block (height submerged), heavier-than-water = None."""
    height = LADDER_BLOCK[2]
    return {ladder_block_name(d): (d / 1000.0 * height if d < 1000.0 else None)
            for d in LADDER_DENSITIES}


def author_buoyancy_ladder():
    scene = fresh_scene()
    fs = scene.fluidsim
    lo, hi = LADDER_DOMAIN
    make_domain(scene, lo, hi, "Ladder Domain")
    ds = domain_settings(scene)
    set_resolution(ds, lo, hi, LADDER_VOXEL)
    ds.max_particles = 700_000
    ds.min_substeps = 1
    ds.max_substeps = 16
    ds.flip_ratio = 0.95
    ds.profiling_enabled = False
    fs.show_particles = True
    fs.particle_display_fraction = 1.0
    scene.frame_start = 1
    scene.frame_end = LADDER_FRAMES
    make_source_box(scene, lo, (hi[0], hi[1], LADDER_POOL_DEPTH),
                    "Ladder Pool Fill", emit_rate=8.0,
                    emission_mode="INITIAL_FILL")
    pitch = (hi[0] - lo[0]) / len(LADDER_DENSITIES)
    sx, sy, sz = LADDER_BLOCK
    for i, (density, color) in enumerate(zip(LADDER_DENSITIES, LADDER_COLORS)):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(
            lo[0] + pitch * (i + 0.5), 0.5 * (lo[1] + hi[1]),
            LADDER_POOL_DEPTH + LADDER_DROP_GAP + 0.5 * sz))
        block = bpy.context.active_object
        block.name = ladder_block_name(density)
        block.scale = (sx, sy, sz)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        block.data.materials.append(demo_material(
            block.name, color, roughness=0.6))
        block.color = color
        block.show_name = True
        add_dynamic_collider(
            block, density=density,
            lock_axes=LADDER_SINKER_LOCK if density >= 1000.0 else None)
    add_camera_light(scene, (0.48, -0.78, 0.62), (0.0, 0.0, 0.0))
    look_at(scene.camera, (0.48, 0.13, 0.14))
    scene.camera.data.lens = 30.0
    add_text_block("Density Ladder", (
        "Density Ladder (FluidSim 4.0 Dynamic colliders)\n\n"
        "Four 12 x 12 x 6 cm blocks of density 250, 500, 750 and 1500 kg/m^3\n"
        "drop onto a still pool. Each is a Collider with Dynamic on\n"
        "(Object Properties > FluidSim > Collider). The liquid floats them at\n"
        "a quarter, half and three quarters of their height; the 1500 block\n"
        "sinks and rests on the domain floor; its rotation is locked\n"
        "(Lock Rotation X/Y/Z) so it lands flat. Play the timeline to watch;\n"
        "Reset clears the solved keyframes.\n"))
    return scene


def build_buoyancy_ladder():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = author_buoyancy_ladder()
    scene.frame_set(1)
    bpy.ops.fluidsim.initialize()
    sess = session()
    tracks = {ladder_block_name(d): [] for d in LADDER_DENSITIES}

    def record(_frame):
        for block_name, state in body_states(sess).items():
            if block_name in tracks:
                tracks[block_name].append(list(state["pos"]))

    series = run_scene("buoyancy_ladder", LADDER_FRAMES, dump_velocities=False,
                       on_frame=record)
    positions = sess.solver.particle_positions()
    surface = surface_height(positions)
    settled = {}
    half = 0.5 * LADDER_BLOCK[2]
    for block_name, track in tracks.items():
        if not track:
            raise RuntimeError(f"buoyancy_ladder: no solved pose for {block_name}")
        z = float(np.mean(np.asarray(track[-12:])[:, 2]))
        settled[block_name] = {"center_z": z, "bottom_z": z - half,
                               "draft": surface - (z - half)}
        log(f"buoyancy_ladder: {block_name} settled bottom {z - half:.4f} "
            f"draft {surface - (z - half):.4f}")
    scene.frame_set(LADDER_FRAMES)
    save_scene("buoyancy_ladder")
    write_meta("buoyancy_ladder", {
        "scene": "buoyancy_ladder",
        "domain": [list(LADDER_DOMAIN[0]), list(LADDER_DOMAIN[1])],
        "voxel_size": LADDER_VOXEL,
        "fps": FPS,
        "pool_depth": LADDER_POOL_DEPTH,
        "block_size": list(LADDER_BLOCK),
        "densities": list(LADDER_DENSITIES),
        "expected_drafts": ladder_expected_drafts(),
        "build_surface_z": surface,
        "build_settled": settled,
        "build_frames": LADDER_FRAMES,
        "build_peak_particles": int(max(series["particle_count"])),
    })
    save_playable("buoyancy_ladder", scene, LADDER_FRAMES)


def surface_height(positions, quantile=0.98):
    """Free-surface height of a resting pool from its particle heights.

    Particles fill the liquid uniformly, so the surface sits about half a
    particle spacing above a high quantile of their z.
    """
    z = np.asarray(positions, dtype=np.float64)[:, 2]
    if not len(z):
        return 0.0
    return float(np.quantile(z, quantile)) + 0.25 * LADDER_VOXEL


# Water Wheel: a light paddle wheel on a fixed axle. Its translation and its
# rotation about X and Z are locked, so the only freedom left is the spin about
# Y; a falling stream on one side of the axle turns it and a floor sink keeps
# the pool from filling up.
WHEEL_DOMAIN = ((0.0, 0.0, 0.0), (0.6, 0.2, 0.6))
WHEEL_VOXEL = 0.01
WHEEL_NAME = "Water Wheel"
WHEEL_AXLE = (0.3, 0.1, 0.3)
WHEEL_RADIUS = 0.14
WHEEL_HUB_RADIUS = 0.035
WHEEL_PADDLES = 6
WHEEL_PADDLE_THICKNESS = 0.02
WHEEL_WIDTH = 0.14
WHEEL_DENSITY = 300.0
WHEEL_DRAG = 1.0
WHEEL_LOCK = (True, True, True, True, False, True)
WHEEL_STREAM_LO = (0.37, 0.06, 0.50)
WHEEL_STREAM_HI = (0.45, 0.14, 0.55)
WHEEL_STREAM_SPEED = 0.5
WHEEL_FRAMES = 96


def wheel_profile():
    """Star outline (XZ plane, about the axle) of the hub and paddles."""
    half = 0.5 * WHEEL_PADDLE_THICKNESS
    outline = []
    for k in range(WHEEL_PADDLES):
        theta = 2.0 * math.pi * k / WHEEL_PADDLES
        c, s = math.cos(theta), math.sin(theta)
        for r, t in ((WHEEL_HUB_RADIUS, -half), (WHEEL_RADIUS, -half),
                     (WHEEL_RADIUS, half), (WHEEL_HUB_RADIUS, half)):
            outline.append((r * c - t * s, r * s + t * c))
    return outline


def make_wheel_mesh(scene):
    """One closed mesh: the star profile extruded along Y, fanned from the
    axle so every cap triangle is convex."""
    outline = wheel_profile()
    n = len(outline)
    y0, y1 = -0.5 * WHEEL_WIDTH, 0.5 * WHEEL_WIDTH
    verts = [(x, y0, z) for x, z in outline] + [(x, y1, z) for x, z in outline]
    verts += [(0.0, y0, 0.0), (0.0, y1, 0.0)]
    c0, c1 = 2 * n, 2 * n + 1
    faces = []
    # The outline turns from +X toward +Z, a negative turn about +Y, so these
    # windings give outward normals.
    for i in range(n):
        j = (i + 1) % n
        faces.append((n + i, n + j, j, i))       # side wall
        faces.append((c0, i, j))                 # front cap (-Y)
        faces.append((c1, n + j, n + i))         # back cap (+Y)
    mesh = bpy.data.meshes.new(WHEEL_NAME)
    mesh.from_pydata(verts, [], faces)
    if mesh.validate(verbose=True):
        raise RuntimeError("water wheel mesh needed repair")
    mesh.update()
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(mesh)
    volume = bm.calc_volume(signed=True)
    bm.free()
    if volume <= 0.0:
        # An inside-out collider makes everything outside it solid.
        raise RuntimeError(f"water wheel mesh is inside out (volume {volume:.3e})")
    obj = bpy.data.objects.new(WHEEL_NAME, mesh)
    scene.collection.objects.link(obj)
    obj.location = WHEEL_AXLE
    return obj


def wheel_spin_angle(quat):
    """Signed rotation about +Y of a (w, x, y, z) quaternion, in radians."""
    w, _x, y, _z = (float(v) for v in quat)
    return 2.0 * math.atan2(y, w)


def author_water_wheel():
    scene = fresh_scene()
    fs = scene.fluidsim
    lo, hi = WHEEL_DOMAIN
    make_domain(scene, lo, hi, "Wheel Domain")
    ds = domain_settings(scene)
    set_resolution(ds, lo, hi, WHEEL_VOXEL)
    ds.max_particles = 600_000
    ds.min_substeps = 1
    ds.max_substeps = 16
    ds.flip_ratio = 0.95
    ds.profiling_enabled = False
    fs.show_particles = True
    fs.particle_display_fraction = 1.0
    scene.frame_start = 1
    scene.frame_end = WHEEL_FRAMES
    stream = make_source_box(scene, WHEEL_STREAM_LO, WHEEL_STREAM_HI,
                             "Wheel Stream", emit_rate=8.0,
                             emission_mode="CONTINUOUS_INFLOW")
    stream.fluidsim.initial_velocity = (0.0, 0.0, -WHEEL_STREAM_SPEED)
    from fluidSim_warp_blender51.blender import sources as _sources

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    drain = bpy.context.active_object
    drain.name = "Wheel Floor Drain"
    drain.location = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.02)
    drain.dimensions = (hi[0] - lo[0], hi[1] - lo[1], 0.04)
    _sources.tag_sink(drain, shape="BOX")
    drain.show_name = True
    wheel = make_wheel_mesh(scene)
    wheel.data.materials.append(demo_material(
        "Wheel Timber", (0.45, 0.30, 0.18, 1.0), roughness=0.7))
    wheel.show_name = True
    add_dynamic_collider(wheel, density=WHEEL_DENSITY, drag=WHEEL_DRAG,
                         lock_axes=WHEEL_LOCK)
    add_camera_light(scene, (0.3, -1.05, 0.32), (math.pi / 2, 0.0, 0.0))
    add_text_block("Water Wheel", (
        "Water Wheel (FluidSim 4.0 Dynamic colliders)\n\n"
        "A light (300 kg/m^3) paddle wheel is a Collider with Dynamic on.\n"
        "Its Lock toggles hold X, Y, Z movement and X, Z rotation, so it can\n"
        "only spin about its Y axle. The stream falls on the paddles right\n"
        "of the axle and turns the wheel; the floor sink drains the water.\n"
        "Play the timeline to watch; Reset clears the solved keyframes.\n"))
    return scene


def build_water_wheel():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = author_water_wheel()
    scene.frame_set(1)
    bpy.ops.fluidsim.initialize()
    sess = session()
    angles, offsets = [], []

    def record(_frame):
        state = body_states(sess).get(WHEEL_NAME)
        if state is None:
            return
        angles.append(wheel_spin_angle(state["quat"]))
        offsets.append(float(np.linalg.norm(
            np.subtract(state["pos"], WHEEL_AXLE))))

    series = run_scene("water_wheel", WHEEL_FRAMES, dump_velocities=False,
                       on_frame=record)
    if not angles:
        raise RuntimeError("water_wheel: no solved pose for the wheel")
    turned = float(np.unwrap(np.asarray(angles))[-1])
    log(f"water_wheel: turned {math.degrees(turned):.1f} deg in "
        f"{WHEEL_FRAMES} frames; axle drift {max(offsets):.2e} m")
    scene.frame_set(WHEEL_FRAMES)
    save_scene("water_wheel")
    write_meta("water_wheel", {
        "scene": "water_wheel",
        "domain": [list(WHEEL_DOMAIN[0]), list(WHEEL_DOMAIN[1])],
        "voxel_size": WHEEL_VOXEL,
        "fps": FPS,
        "wheel": {
            "name": WHEEL_NAME, "axle": list(WHEEL_AXLE),
            "radius": WHEEL_RADIUS, "paddles": WHEEL_PADDLES,
            "density": WHEEL_DENSITY, "lock_axes": list(WHEEL_LOCK),
        },
        "stream": {"lo": list(WHEEL_STREAM_LO), "hi": list(WHEEL_STREAM_HI),
                   "speed": WHEEL_STREAM_SPEED},
        "build_frames": WHEEL_FRAMES,
        "build_turned_rad": turned,
        "build_axle_drift": max(offsets),
        "build_peak_particles": int(max(series["particle_count"])),
    })
    save_playable("water_wheel", scene, WHEEL_FRAMES)


# ---------------------------------------------------------------------------
# Feature demos: viscosity, surface tension, narrow-band particles
# ---------------------------------------------------------------------------

def show_liquid(scene, prefix, color, view="SURFACE_PARTICLES"):
    """Liquid display with one water-like material."""
    fs = scene.fluidsim
    fs.show_particles = view in {"PARTICLES", "SURFACE_PARTICLES"}
    fs.show_surface = view in {"SURFACE", "SURFACE_PARTICLES"}
    fs.liquid_view_mode = view
    fs.particle_display_fraction = 1.0
    fs.surface_material = demo_material(
        f"{prefix} Liquid", color, roughness=0.1, transmission=0.4)
    fs.particle_material = demo_material(
        f"{prefix} Particles", color, roughness=0.3)


def add_label(scene, text, location, size, align="CENTER"):
    """A flat 3D text label lying face-up in the XY plane."""
    curve = bpy.data.curves.new(text, type="FONT")
    curve.body = text
    curve.size = size
    curve.align_x = align
    obj = bpy.data.objects.new(f"Label {text}", curve)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(demo_material(
        "Label Ink", (0.08, 0.08, 0.09, 1.0), roughness=0.8))
    return obj


# Viscosity Race: one long channel split into three lanes by thin static
# walls. Each lane starts with the same column of liquid; only the source's
# Viscosity differs (water, syrup, tar). With Viscosity on and From Particles
# on, every column carries its own value, so the three fronts run out at
# visibly different speeds from the same release.
RACE_DOMAIN = ((0.0, 0.0, 0.0), (1.6, 0.48, 0.30))
RACE_VOXEL = 0.01
RACE_WALL = 0.02
RACE_FLOOR = 0.02
RACE_COLUMN = (0.20, 0.20)              # x length, height of each column
RACE_LANES = (("Water", 0.0), ("Syrup", 0.004), ("Tar", 0.3))
RACE_FRAMES = 72
RACE_PREVIEW_FRAME = 24                 # fronts furthest apart in proportion
RACE_COLOR = (0.10, 0.36, 0.62, 1.0)


def race_lane_bounds():
    """(y_lo, y_hi) of the liquid in each lane, inside the walls."""
    lo, hi = RACE_DOMAIN
    pitch = (hi[1] - lo[1]) / len(RACE_LANES)
    half = 0.5 * RACE_WALL
    out = []
    for i in range(len(RACE_LANES)):
        y0 = lo[1] + pitch * i + (half if i > 0 else 0.0)
        y1 = lo[1] + pitch * (i + 1) - (half if i < len(RACE_LANES) - 1 else 0.0)
        out.append((y0, y1))
    return out


def race_fronts(positions):
    """Leading-edge x (98th percentile) of the liquid in each lane."""
    pos = np.asarray(positions, dtype=np.float64)
    fronts = {}
    for (lane, _nu), (y0, y1) in zip(RACE_LANES, race_lane_bounds()):
        x = pos[(pos[:, 1] >= y0) & (pos[:, 1] < y1), 0]
        fronts[lane] = float(np.quantile(x, 0.98)) if len(x) else 0.0
    return fronts


def author_viscosity_race():
    from fluidSim_warp_blender51.blender import colliders as _colliders

    scene = fresh_scene()
    lo, hi = RACE_DOMAIN
    make_domain(scene, lo, hi, "Race Domain")
    ds = domain_settings(scene)
    set_resolution(ds, lo, hi, RACE_VOXEL)
    ds.max_particles = 400_000
    ds.min_substeps = 1
    ds.max_substeps = 16
    ds.viscosity_enabled = True
    ds.viscosity_from_particles = True
    ds.profiling_enabled = False
    show_liquid(scene, "Race", RACE_COLOR)
    scene.frame_start = 1
    scene.frame_end = RACE_FRAMES
    # A grippy (no-slip) floor: viscosity resists shear, and on the domain's
    # free-slip floor a slumping column would slide out almost as a plug.
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    floor = bpy.context.active_object
    floor.name = "Race Floor"
    floor.location = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]),
                      lo[2] + 0.5 * RACE_FLOOR)
    floor.dimensions = (hi[0] - lo[0], hi[1] - lo[1], RACE_FLOOR)
    bpy.context.view_layer.update()
    floor.data.materials.append(demo_material(
        "Race Floor", (0.55, 0.55, 0.53, 1.0), roughness=0.8))
    _colliders.tag_collider(floor, friction=1.0, restitution=0.0)
    pitch = (hi[1] - lo[1]) / len(RACE_LANES)
    for i in range(1, len(RACE_LANES)):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        wall = bpy.context.active_object
        wall.name = f"Race Wall {i}"
        wall.location = (0.5 * (lo[0] + hi[0]), lo[1] + pitch * i,
                         0.5 * (lo[2] + hi[2]))
        wall.dimensions = (hi[0] - lo[0], RACE_WALL, hi[2] - lo[2])
        bpy.context.view_layer.update()
        wall.data.materials.append(demo_material(
            "Race Wall", (0.7, 0.7, 0.68, 1.0), roughness=0.6))
        _colliders.tag_collider(wall, friction=0.0, restitution=0.0)
    length, height = RACE_COLUMN
    for (lane, nu), (y0, y1) in zip(RACE_LANES, race_lane_bounds()):
        src = make_source_box(scene, (lo[0], y0, lo[2] + RACE_FLOOR),
                              (lo[0] + length, y1, lo[2] + RACE_FLOOR + height),
                              f"Race {lane}", emit_rate=8.0,
                              emission_mode="INITIAL_FILL")
        src.fluidsim.viscosity = nu
        # Labels sit off the release end of each lane, read from above.
        add_label(scene, f"{lane}  {nu:g} m²/s",
                  (lo[0] - 0.03, 0.5 * (y0 + y1) - 0.016, lo[2]), 0.045,
                  align="RIGHT")
    add_camera_light(scene, (0.60, 0.12, 2.00), (0.0, 0.0, 0.0))
    look_at(scene.camera, (0.60, 0.24, 0.0))
    scene.camera.data.lens = 34.0
    add_text_block("Viscosity Race", (
        "Viscosity Race (FluidSim viscosity)\n\n"
        "Three lanes, three identical 20 cm columns of liquid released at\n"
        "once. The only difference is each source's Viscosity (Object\n"
        "Properties > FluidSim > Source): water 0, syrup 0.004 and tar\n"
        "0.3 m^2/s. The domain has Viscosity on with From Particles on\n"
        "(Domain > Simulate > Viscosity), so every column keeps its own\n"
        "value. The floor is a grippy collider (Friction 1) for the\n"
        "viscosity to shear against. Press Play: water races to the far\n"
        "wall, the syrup spreads about half as far, the tar slumps and\n"
        "creeps.\n"
        "Try: change a column's Viscosity, or turn From Particles off to\n"
        "give every lane the domain's single Viscosity value.\n"))
    return scene


def build_viscosity_race():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = author_viscosity_race()
    scene.frame_set(1)
    bpy.ops.fluidsim.initialize()
    sess = session()
    checkpoints = {}

    def record(frame):
        if frame == RACE_PREVIEW_FRAME:
            use_material_shading(scene)
            render_preview("viscosity_race", min_water_fraction=0.05)
        if frame in (6, 12, 24, 48, RACE_FRAMES):
            checkpoints[frame] = race_fronts(sess.solver.particle_positions())
            log(f"viscosity_race frame {frame} fronts: " + ", ".join(
                f"{k} {v:.3f}" for k, v in checkpoints[frame].items()))

    series = run_scene("viscosity_race", RACE_FRAMES, dump_velocities=False,
                       on_frame=record)
    order = [lane for lane, _ in RACE_LANES]
    at24 = checkpoints[24]
    if not all(at24[a] > at24[b] + 5 * RACE_VOXEL
               for a, b in zip(order, order[1:])):
        raise RuntimeError(f"viscosity_race fronts not ordered at frame 24: {at24}")
    use_material_shading(scene)
    save_scene("viscosity_race", preview=False)
    write_meta("viscosity_race", {
        "scene": "viscosity_race",
        "domain": [list(RACE_DOMAIN[0]), list(RACE_DOMAIN[1])],
        "voxel_size": RACE_VOXEL,
        "fps": FPS,
        "column": list(RACE_COLUMN),
        "lanes": {lane: nu for lane, nu in RACE_LANES},
        "lane_bounds": race_lane_bounds(),
        "build_fronts": {str(k): v for k, v in sorted(checkpoints.items())},
        "build_frames": RACE_FRAMES,
        "build_peak_particles": int(max(series["particle_count"])),
    })
    save_playable("viscosity_race", scene, RACE_FRAMES)


# Zero-G Droplet: a 6 x 2 x 2 cm bar of water floats with gravity off.
# Without surface tension nothing acts on it and it stays a bar; with it the
# Laplace pressure pulls the ends in and the bar contracts and wobbles into a
# ball (capillary time sqrt(rho R^3 / sigma) is about a fifth of a second).
DROP_DOMAIN = ((0.0, 0.0, 0.0), (0.10, 0.10, 0.10))
DROP_VOXEL = 0.002
DROP_BAR = (0.06, 0.02, 0.02)
DROP_FRAMES = 48
DROP_COLOR = (0.12, 0.42, 0.70, 1.0)


def blob_aspect(positions):
    """sqrt(largest / smallest principal variance): 3 for the bar, 1 for a ball."""
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) < 4:
        return 0.0
    eig = np.linalg.eigvalsh(np.cov(pos.T))
    return float(math.sqrt(max(eig[-1], 1e-30) / max(eig[0], 1e-30)))


def author_capillary_droplet(surface_tension=True):
    scene = fresh_scene()
    lo, hi = DROP_DOMAIN
    make_domain(scene, lo, hi, "Droplet Domain")
    ds = domain_settings(scene)
    set_resolution(ds, lo, hi, DROP_VOXEL)
    ds.max_particles = 400_000
    ds.min_substeps = 1
    # The capillary step limit at 2 mm needs ~28 substeps per frame;
    # a lower cap clamps it and the drop shakes itself apart.
    ds.max_substeps = 48
    ds.use_scene_gravity = False
    ds.gravity = (0.0, 0.0, 0.0)
    ds.surface_tension_enabled = surface_tension
    ds.surface_tension_coefficient = 0.072
    ds.surface_tension_density = 1000.0
    ds.profiling_enabled = False
    show_liquid(scene, "Droplet", DROP_COLOR)
    scene.frame_start = 1
    scene.frame_end = DROP_FRAMES
    c = [0.5 * (lo[i] + hi[i]) for i in range(3)]
    make_source_box(scene, [c[i] - 0.5 * DROP_BAR[i] for i in range(3)],
                    [c[i] + 0.5 * DROP_BAR[i] for i in range(3)],
                    "Droplet Bar", emit_rate=8.0, emission_mode="INITIAL_FILL")
    add_camera_light(scene, (0.05, -0.17, 0.11), (0.0, 0.0, 0.0))
    look_at(scene.camera, c)
    scene.camera.data.lens = 50.0
    add_text_block("Zero-G Droplet", (
        "Zero-G Droplet (FluidSim surface tension)\n\n"
        "A 6 x 2 x 2 cm bar of water floats with gravity off (Domain >\n"
        "Setup > Use Scene Gravity off, Gravity 0). Surface Tension is on\n"
        "with water's 0.072 N/m (Domain > Simulate > Surface Tension).\n"
        "Press Play: the bar's ends pull in and it wobbles into a round\n"
        "drop within about a second.\n"
        "Try: turn Surface Tension off and Reset; the bar just hangs there\n"
        "as a bar. Raise the coefficient for a faster, stiffer wobble.\n"))
    return scene


def build_capillary_droplet():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    # Reference: the same bar with surface tension off keeps its shape.
    author_capillary_droplet(surface_tension=False)
    bpy.ops.fluidsim.initialize()
    run_scene("capillary_droplet__off", DROP_FRAMES, dump_velocities=False)
    aspect_off = blob_aspect(session().solver.particle_positions())
    bpy.ops.fluidsim.release()

    scene = author_capillary_droplet(surface_tension=True)
    scene.frame_set(1)
    bpy.ops.fluidsim.initialize()
    sess = session()
    aspects = {}

    def record(frame):
        pos = sess.solver.particle_positions()
        aspects[frame] = blob_aspect(pos)
        if frame % 4 == 0:
            fs = bpy.context.scene.fluidsim
            log(f"capillary_droplet frame {frame}: aspect {aspects[frame]:.2f} "
                f"extent {np.ptp(pos, axis=0).round(4).tolist()} "
                f"st {fs.stat_surface_tension_status} "
                f"kmax {fs.stat_surface_tension_max_curvature:.0f} "
                f"amax {fs.stat_surface_tension_max_acceleration:.1f}")

    series = run_scene("capillary_droplet", DROP_FRAMES, dump_velocities=False,
                       on_frame=record)
    start = aspects[1]
    late = float(np.mean([aspects[f] for f in range(DROP_FRAMES - 11,
                                                    DROP_FRAMES + 1)]))
    log(f"capillary_droplet aspect: start {start:.2f}, late mean {late:.2f}, "
        f"min {min(aspects.values()):.2f}, surface tension off {aspect_off:.2f}")
    if not late < 0.6 * aspect_off:
        raise RuntimeError(
            f"capillary_droplet did not round up: aspect {late:.2f} vs "
            f"{aspect_off:.2f} with surface tension off")
    use_material_shading(scene)
    save_scene("capillary_droplet")
    write_meta("capillary_droplet", {
        "scene": "capillary_droplet",
        "domain": [list(DROP_DOMAIN[0]), list(DROP_DOMAIN[1])],
        "voxel_size": DROP_VOXEL,
        "fps": FPS,
        "bar_size": list(DROP_BAR),
        "surface_tension_coefficient": 0.072,
        "build_aspect": {str(k): v for k, v in sorted(aspects.items())},
        "build_late_aspect": late,
        "build_aspect_surface_tension_off": aspect_off,
        "build_frames": DROP_FRAMES,
        "build_peak_particles": int(max(series["particle_count"])),
    })
    save_playable("capillary_droplet", scene, DROP_FRAMES)


# Deep Plunge: a deep, calm tank with a block of water dropped into it.
# Narrow-band FLIP keeps particles only within a few cells of the surface and
# treats the still interior as grid liquid, so the tank needs a small fraction
# of the particles a full-band run uses. The builder runs both and records the
# particle counts and the resting water level of each.
PLUNGE_DOMAIN = ((0.0, 0.0, 0.0), (1.0, 0.5, 1.0))
PLUNGE_VOXEL = 0.02
PLUNGE_DEPTH = 0.70
PLUNGE_BLOCK = ((0.40, 0.15, 0.82), (0.60, 0.35, 0.97))
PLUNGE_FRAMES = 72
PLUNGE_PREVIEW_FRAME = 14               # block just through the surface
PLUNGE_COLOR = (0.06, 0.30, 0.55, 1.0)


def water_level(positions, domain=PLUNGE_DOMAIN):
    """Mean height of the highest particle in each occupied XY column."""
    pos = np.asarray(positions, dtype=np.float64)
    if not len(pos):
        return 0.0
    lo = domain[0]
    ij = np.floor((pos[:, :2] - lo[:2]) / PLUNGE_VOXEL).astype(np.int64)
    key = ij[:, 0] * 100_000 + ij[:, 1]
    order = np.lexsort((pos[:, 2], key))
    last = np.r_[key[order][1:] != key[order][:-1], True]
    return float(np.mean(pos[order][last, 2]))


def author_deep_plunge(narrow_band=True):
    scene = fresh_scene()
    lo, hi = PLUNGE_DOMAIN
    make_domain(scene, lo, hi, "Plunge Domain")
    ds = domain_settings(scene)
    set_resolution(ds, lo, hi, PLUNGE_VOXEL)
    ds.max_particles = 900_000
    ds.min_substeps = 1
    ds.max_substeps = 16
    ds.pressure_preconditioner = "MULTIGRID"
    ds.narrow_band_enabled = narrow_band
    ds.narrow_band_voxels = 3.0
    ds.profiling_enabled = False
    # Particles only: the point is where the particles are (a thin skin).
    show_liquid(scene, "Plunge", PLUNGE_COLOR, view="PARTICLES")
    scene.frame_start = 1
    scene.frame_end = PLUNGE_FRAMES
    make_source_box(scene, lo, (hi[0], hi[1], PLUNGE_DEPTH), "Plunge Tank Fill",
                    emit_rate=8.0, emission_mode="INITIAL_FILL")
    make_source_box(scene, *PLUNGE_BLOCK, "Plunge Falling Block",
                    emit_rate=8.0, emission_mode="INITIAL_FILL")
    # Visual-only tank (floor, back and left walls, no collider tag) so the
    # particle skin reads as the top of a deep, mostly particle-free tank.
    t = 0.02
    for name, a, b in (
            ("Plunge Tank Floor", (lo[0], lo[1], lo[2] - t), (hi[0], hi[1] + t, lo[2])),
            ("Plunge Tank Back", (lo[0] - t, hi[1], lo[2] - t), (hi[0], hi[1] + t, hi[2])),
            ("Plunge Tank Side", (lo[0] - t, lo[1], lo[2] - t), (lo[0], hi[1] + t, hi[2]))):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        slab = bpy.context.active_object
        slab.name = name
        slab.location = [0.5 * (a[i] + b[i]) for i in range(3)]
        slab.dimensions = [b[i] - a[i] for i in range(3)]
        slab.data.materials.append(demo_material(
            "Plunge Tank", (0.62, 0.64, 0.66, 1.0), roughness=0.6))
    bpy.context.view_layer.update()
    add_camera_light(scene, (1.75, -1.55, 1.05), (0.0, 0.0, 0.0))
    look_at(scene.camera, (0.5, 0.25, 0.55))
    scene.camera.data.lens = 35.0
    add_text_block("Deep Plunge", (
        "Deep Plunge (FluidSim narrow-band particles)\n\n"
        "A 70 cm deep tank with a block of water dropped into it. Narrow\n"
        "Band is on (Domain > Simulate > Particles > Narrow Band, 3 voxels):\n"
        "particles live only near the surface and the calm interior is\n"
        "carried by the grid, so the tank runs on a fraction of the\n"
        "particles. Watch the particle count in the Simulate stage.\n"
        "Try: turn Narrow Band off and Reset; the splash looks the same\n"
        "but the count jumps several times. Narrow band pays off on deep,\n"
        "calm volumes; a shallow, splashy scene saves little. It cannot\n"
        "run with color mixing, viscosity From Particles, a scalar field\n"
        "source or attribute export.\n"))
    return scene


def build_deep_plunge():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    # Reference: the same tank as a full-band (every-cell) particle run.
    author_deep_plunge(narrow_band=False)
    bpy.ops.fluidsim.initialize()
    full = run_scene("deep_plunge__full", PLUNGE_FRAMES, dump_velocities=False)
    full_level = water_level(session().solver.particle_positions())
    bpy.ops.fluidsim.release()

    scene = author_deep_plunge(narrow_band=True)
    scene.frame_set(1)
    bpy.ops.fluidsim.initialize()
    sess = session()

    def preview(frame):
        if frame == PLUNGE_PREVIEW_FRAME:
            use_material_shading(scene)
            render_preview("deep_plunge", min_water_fraction=0.02)

    series = run_scene("deep_plunge", PLUNGE_FRAMES, dump_velocities=False,
                       on_frame=preview)
    level = water_level(sess.solver.particle_positions())
    full_mean = float(np.mean(full["particle_count"]))
    band_mean = float(np.mean(series["particle_count"]))
    ratio = full_mean / max(band_mean, 1.0)
    expected_level = PLUNGE_DEPTH + (
        np.prod(np.subtract(PLUNGE_BLOCK[1], PLUNGE_BLOCK[0]))
        / ((PLUNGE_DOMAIN[1][0] - PLUNGE_DOMAIN[0][0])
           * (PLUNGE_DOMAIN[1][1] - PLUNGE_DOMAIN[0][1])))
    log(f"deep_plunge particles: full {full_mean:,.0f}, narrow band "
        f"{band_mean:,.0f} ({ratio:.1f}x fewer); level full {full_level:.3f}, "
        f"narrow {level:.3f}, expected ~{expected_level:.3f}")
    if ratio < 3.0:
        raise RuntimeError(f"deep_plunge narrow band saved only {ratio:.2f}x")
    if abs(level - full_level) > 2 * PLUNGE_VOXEL:
        raise RuntimeError(
            f"deep_plunge narrow-band level {level:.3f} drifted from the "
            f"full-band {full_level:.3f}")
    use_material_shading(scene)
    save_scene("deep_plunge", preview=False)
    write_meta("deep_plunge", {
        "scene": "deep_plunge",
        "domain": [list(PLUNGE_DOMAIN[0]), list(PLUNGE_DOMAIN[1])],
        "voxel_size": PLUNGE_VOXEL,
        "fps": FPS,
        "tank_depth": PLUNGE_DEPTH,
        "falling_block": [list(PLUNGE_BLOCK[0]), list(PLUNGE_BLOCK[1])],
        "narrow_band_voxels": 3.0,
        "build_mean_particles_narrow_band": band_mean,
        "build_mean_particles_full_band": full_mean,
        "build_particle_ratio": ratio,
        "build_water_level_narrow_band": level,
        "build_water_level_full_band": full_level,
        "build_frames": PLUNGE_FRAMES,
    })
    save_playable("deep_plunge", scene, PLUNGE_FRAMES)


# ---------------------------------------------------------------------------

def main():
    os.makedirs(DEMOS, exist_ok=True)
    enable_repo_addon(REPO, default_set=False, persistent=False)
    builders = {
        "dam_break": build_dam_break,
        "crown_splash": build_crown_splash,
        "paddle_stir": build_paddle_stir,
        "buoyancy_ladder": build_buoyancy_ladder,
        "water_wheel": build_water_wheel,
        "viscosity_race": build_viscosity_race,
        "capillary_droplet": build_capillary_droplet,
        "deep_plunge": build_deep_plunge,
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
