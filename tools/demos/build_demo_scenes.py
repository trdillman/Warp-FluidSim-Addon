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

def main():
    os.makedirs(DEMOS, exist_ok=True)
    enable_repo_addon(REPO, default_set=False, persistent=False)
    builders = {
        "dam_break": build_dam_break,
        "crown_splash": build_crown_splash,
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
