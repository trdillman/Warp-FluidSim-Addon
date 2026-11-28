"""Play and render the Nets Showcase demo in two stages.

    blender --background --factory-startup --python tools/demos/render_nets_showcase.py -- sim
    blender --background --factory-startup --python tools/demos/render_nets_showcase.py -- render

``sim`` opens the saved demo (``artifacts/review_demos/nets_showcase.blend``,
built by ``build_demo_scenes.py -- nets_showcase``), plays it through the real
frame handler with the Surface Nets backend, and writes each frame's liquid
mesh to ``artifacts/nets_showcase/meshes/``. It refuses to continue unless the
render surface really is the ``nets`` backend, and records the backend, the
particle count, the step time and the triangle count of every frame.

``render`` runs afterwards in a fresh Blender (never in the same process: the
simulation holds most of the card), so the VRAM is free for Cycles. It builds the studio look, loads one mesh per frame, renders
PNGs and encodes ``nets_showcase.mp4``.

Options (after ``--``): ``--frames A B`` (default: the scene's range),
``--samples N`` (default 128), ``--size W H`` (default 1920 1080), ``--engine
CYCLES|EEVEE``, ``--still FRAME`` (render one frame and stop), ``--resume`` (keep PNGs
already rendered with the same size, samples, engine and meshes; without it
every frame is rendered again), ``--out DIR`` (default ``artifacts/nets_showcase``), ``--blend PATH``.
``--look-test`` renders a stand-in splash so the lighting can be checked
without a simulation.
"""

from __future__ import annotations

import json
import os
import sys
import time

import bpy
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.demos import nets_showcase as ns  # noqa: E402

DEFAULT_BLEND = os.path.join(REPO, "artifacts", "review_demos", "nets_showcase.blend")
DEFAULT_OUT = os.path.join(REPO, "artifacts", "nets_showcase")
SURFACE_OBJECT = "FluidSim Surface"
WATER_OBJECT = "Nets Water"


def log(msg):
    print(f"[nets_showcase] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {"stage": None, "frames": None, "samples": 128, "size": (1920, 1080),
            "engine": "CYCLES", "still": None, "look_test": False, "resume": False,
            "out": DEFAULT_OUT, "blend": DEFAULT_BLEND}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("sim", "render"):
            opts["stage"] = a
        elif a == "--frames":
            opts["frames"] = (int(argv[i + 1]), int(argv[i + 2]))
            i += 2
        elif a == "--samples":
            opts["samples"] = int(argv[i + 1])
            i += 1
        elif a == "--size":
            opts["size"] = (int(argv[i + 1]), int(argv[i + 2]))
            i += 2
        elif a == "--engine":
            opts["engine"] = argv[i + 1].upper()
            i += 1
        elif a == "--still":
            opts["still"] = int(argv[i + 1])
            i += 1
        elif a == "--out":
            opts["out"] = os.path.abspath(argv[i + 1])
            i += 1
        elif a == "--blend":
            opts["blend"] = os.path.abspath(argv[i + 1])
            i += 1
        elif a == "--resume":
            opts["resume"] = True
        elif a == "--look-test":
            opts["look_test"] = True
        else:
            raise SystemExit(f"unknown option {a!r}")
        i += 1
    if opts["stage"] is None and not opts["look_test"]:
        raise SystemExit("name a stage: sim or render")
    return opts


# ---------------------------------------------------------------------------
# Stage 1: play the demo, save the liquid mesh of every frame


def read_surface_mesh():
    """(verts float32 (N,3), tris int32 (T,3)) of the published surface, or None."""
    obj = bpy.data.objects.get(SURFACE_OBJECT)
    if obj is None or obj.type != "MESH" or len(obj.data.vertices) == 0:
        return None
    mesh = obj.data
    mesh.calc_loop_triangles()
    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts)
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris)
    world = np.array(obj.matrix_world, dtype=np.float32)
    verts = verts.reshape(-1, 3)
    if not np.allclose(world, np.eye(4), atol=1e-6):
        verts = verts @ world[:3, :3].T + world[:3, 3]
    return verts, tris.reshape(-1, 3)


def run_sim(opts):
    os.environ[ns.BACKEND_ENV] = ns.BACKEND
    from tools.repo_addon_loader import enable_repo_addon

    if not os.path.exists(opts["blend"]):
        raise SystemExit(
            f"{opts['blend']} is missing; build it first:\n"
            "  blender --background --factory-startup --python "
            "tools/demos/build_demo_scenes.py -- nets_showcase")
    bpy.ops.wm.open_mainfile(filepath=opts["blend"])
    enable_repo_addon(REPO, default_set=False, persistent=False)
    # Import only after enable_repo_addon, which re-imports the add-on.
    from fluidSim_warp_blender51.blender import scene_bridge

    scene = bpy.context.scene
    if not scene.fluidsim.live:
        raise SystemExit("the saved scene is not in Live mode")
    # Start the solver here, once. The first frame_set would otherwise start it
    # and then Reset it for the start frame, which plans the old and the new
    # solver together and is refused at hero resolution on a 16 GB card.
    first = scene_bridge.session()
    if not first.active:
        if bpy.ops.fluidsim.initialize() != {"FINISHED"}:
            raise RuntimeError(
                f"Initialize was refused: {scene.fluidsim.status!r}")
        first.last_frame = scene.frame_start
    start, end = scene.frame_start, scene.frame_end
    if opts["frames"]:
        start, end = max(start, opts["frames"][0]), min(end, opts["frames"][1])
    mesh_dir = os.path.join(opts["out"], "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    # Meshes and the manifest of an earlier run (another preset or frame range)
    # must not leak into this one.
    for name in os.listdir(mesh_dir):
        if name.endswith(".npy"):
            os.remove(os.path.join(mesh_dir, name))
    manifest = os.path.join(opts["out"], "sim_series.json")
    if os.path.exists(manifest):
        os.remove(manifest)
    series = {"frames": [], "particles": [], "step_ms": [], "substeps": [],
              "verts": [], "tris": [], "backend": [], "wall_s": [], "leaves": []}
    t_all = time.perf_counter()
    scene.frame_set(scene.frame_start)
    # The solver is sequential: step from the first frame, keep what is asked for.
    for frame in range(scene.frame_start, end + 1):
        t0 = time.perf_counter()
        if frame != scene.frame_start:
            scene.frame_set(frame)
        sess = scene_bridge.session()
        solver = sess.solver if sess.active else None
        if solver is None:
            if frame == scene.frame_start:
                continue
            raise RuntimeError(f"frame {frame}: no active solver")
        # A failed step pauses Live and keeps the old solver and surface, which
        # would be saved again under this frame number. Stop instead.
        if frame != scene.frame_start:
            error = str(getattr(sess, "error", "") or "")
            if error or not scene.fluidsim.live or sess.last_frame != frame:
                raise RuntimeError(
                    f"frame {frame}: the simulation did not advance "
                    f"(live={scene.fluidsim.live}, last_frame={sess.last_frame}, "
                    f"error={error or scene.fluidsim.status!r})")
        backend = solver.surface.render_extractor.backend
        if backend != ns.BACKEND:
            raise RuntimeError(
                f"frame {frame}: the render surface is {backend!r}, not "
                f"{ns.BACKEND!r}; {ns.BACKEND_ENV} did not take effect")
        if frame < start:
            continue
        got = read_surface_mesh()
        fs = scene.fluidsim
        if got is None:
            log(f"frame {frame}: no surface mesh published yet")
            continue
        verts, tris = got
        np.save(os.path.join(mesh_dir, f"verts_{frame:04d}.npy"), verts)
        np.save(os.path.join(mesh_dir, f"tris_{frame:04d}.npy"), tris)
        series["frames"].append(frame)
        series["particles"].append(int(solver.particles.refresh_host_count()))
        series["step_ms"].append(float(fs.stat_step_ms))
        series["substeps"].append(int(fs.stat_substeps))
        series["leaves"].append(int(fs.stat_voxels) // 512)
        series["verts"].append(int(len(verts)))
        series["tris"].append(int(len(tris)))
        series["backend"].append(backend)
        series["wall_s"].append(time.perf_counter() - t0)
        log(f"frame {frame}/{end}: {series['particles'][-1]:,} particles, "
            f"{len(tris):,} tris, {series['leaves'][-1]:,} leaves, "
            f"step {series['step_ms'][-1]:.0f} ms, "
            f"{series['wall_s'][-1]:.1f} s, backend {backend}")
    series["total_s"] = time.perf_counter() - t_all
    series["preset"] = ns.preset().name
    with open(os.path.join(opts["out"], "sim_series.json"), "w", encoding="utf-8") as fh:
        json.dump(series, fh)
    if not series["frames"]:
        raise RuntimeError("no frames were captured")
    log(f"simulated {len(series['frames'])} frames in {series['total_s']:.0f} s; "
        f"backend {sorted(set(series['backend']))}")
    try:
        bpy.ops.fluidsim.release()
    except Exception as exc:  # noqa: BLE001
        log(f"release skipped: {exc!r}")


# ---------------------------------------------------------------------------
# Stage 2: the look, then render


def _set_input(node, name, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def make_water_material():
    mat = bpy.data.materials.new("Nets Water")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    glass = nt.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = (0.90, 0.97, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.0
    glass.inputs["IOR"].default_value = 1.33
    absorb = nt.nodes.new("ShaderNodeVolumeAbsorption")
    absorb.inputs["Color"].default_value = (0.55, 0.85, 0.95, 1.0)
    absorb.inputs["Density"].default_value = 18.0
    nt.links.new(glass.outputs["BSDF"], out.inputs["Surface"])
    nt.links.new(absorb.outputs["Volume"], out.inputs["Volume"])
    mat.diffuse_color = ns.WATER_COLOR
    return mat


def make_stone_material(name, color, roughness, specular=0.5, coat=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    for key in ("Specular IOR Level", "Specular"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = specular
            break
    for key in ("Coat Weight", "Clearcoat"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = coat
            break
    mat.diffuse_color = color
    return mat


def make_glass_material(name):
    """Clear glass without refraction or shadow: transparent, with Fresnel reflections.

    Real glass would bend the view of the water and fill the shadows with caustic
    noise; this only adds the faint reflections that make the wall read as glass.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    clear = tree.nodes.new("ShaderNodeBsdfTransparent")
    gloss = tree.nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Roughness"].default_value = 0.03
    fresnel = tree.nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.45
    mix = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(fresnel.outputs["Fac"], mix.inputs["Fac"])
    tree.links.new(clear.outputs["BSDF"], mix.inputs[1])
    tree.links.new(gloss.outputs["BSDF"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    mat.diffuse_color = (0.8, 0.9, 1.0, 0.1)
    return mat


def add_softbox(name, location, size, strength, target, color=(1.0, 1.0, 1.0)):
    """A big emissive card that lights the splash and shows up in its reflections."""
    from mathutils import Vector

    data = bpy.data.lights.new(name, type="AREA")
    data.shape = "RECTANGLE"
    data.size, data.size_y = size
    data.energy = strength
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    direction = Vector(target) - Vector(location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    # Visible in the water's reflections, not to the camera.
    obj.visible_camera = False
    return obj


def set_cycles_device(scene):
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        return "CPU"
    cp = prefs.preferences
    for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            cp.compute_device_type = kind
            cp.get_devices()
        except Exception:  # noqa: BLE001
            continue
        gpus = [d for d in cp.devices if d.type == kind]
        if gpus:
            for d in cp.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            return kind
    scene.cycles.device = "CPU"
    return "CPU"


def setup_look(scene, opts):
    """Dark studio: black floor slab, white marble basin, soft cards, blue water."""
    scene.render.engine = "CYCLES" if opts["engine"] == "CYCLES" else "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = opts["size"]
    scene.render.resolution_percentage = 100
    scene.render.fps = ns.FPS
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = opts["samples"]
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 12
        scene.cycles.transmission_bounces = 12
        scene.cycles.glossy_bounces = 6
        scene.cycles.volume_bounces = 2
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False
        scene.cycles.sample_clamp_indirect = 8.0
        scene.cycles.sample_clamp_direct = 0.0
        log(f"cycles device: {set_cycles_device(scene)}")
    scene.view_settings.view_transform = "Filmic" if "Filmic" in \
        scene.view_settings.bl_rna.properties["view_transform"].enum_items.keys() else "AgX"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    # World: a deep graded backdrop. Everything bright in the reflections is a card.
    world = bpy.data.worlds.new("Nets Studio") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(coord.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    ramp.color_ramp.elements[0].color = (0.010, 0.014, 0.022, 1.0)
    ramp.color_ramp.elements[1].color = (0.060, 0.085, 0.120, 1.0)
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    # Lights.
    for obj in [o for o in scene.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    target = (0.0, 0.0, 0.20)
    add_softbox("Nets Key", (-0.95, -0.55, 0.75), (0.9, 0.6), 120.0, target, (1.0, 0.97, 0.92))
    add_softbox("Nets Rim Right", (0.95, 0.35, 0.45), (0.25, 0.9), 80.0, target, (0.80, 0.90, 1.0))
    add_softbox("Nets Rim Back", (0.0, 0.95, 0.60), (1.0, 0.25), 70.0, target, (0.85, 0.92, 1.0))
    add_softbox("Nets Top", (0.0, 0.0, 1.30), (0.7, 0.7), 40.0, (0.0, 0.0, 0.1))
    # Materials on the collider meshes.
    marble = make_stone_material("Nets Marble", ns.MATERIAL_COLORS["Nets Marble"], 0.18, 0.6, 0.3)
    glass = make_glass_material("Nets Glass")
    slab = make_stone_material("Nets Slab", ns.MATERIAL_COLORS["Nets Slab"], 0.06, 0.8, 0.0)
    for obj in scene.objects:
        if obj.type != "MESH" or not obj.name.startswith("Nets "):
            continue
        if obj.name in ns.GLASS_WALLS:
            obj.data.materials.clear()
            obj.data.materials.append(glass)
            obj.hide_render = False
            obj.visible_shadow = False
        elif obj.name.startswith("Nets Trough"):
            obj.data.materials.clear()
            obj.data.materials.append(marble)
        elif obj.name == "Nets Slab" or obj.name.startswith("Nets Rim"):
            obj.data.materials.clear()
            obj.data.materials.append(slab)
    if scene.camera is not None:
        scene.camera.data.lens = ns.CAMERA_LENS
        scene.camera.data.sensor_width = 36.0
        scene.camera.data.dof.use_dof = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"


def make_water_object(scene, material):
    mesh = bpy.data.meshes.new(WATER_OBJECT)
    obj = bpy.data.objects.new(WATER_OBJECT, mesh)
    scene.collection.objects.link(obj)
    mesh.materials.append(material)
    return obj


def set_water_mesh(obj, verts, tris):
    old = obj.data
    mesh = bpy.data.meshes.new(WATER_OBJECT)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", np.ascontiguousarray(verts, dtype=np.float32).ravel())
    n = len(tris)
    mesh.loops.add(n * 3)
    mesh.loops.foreach_set("vertex_index", np.ascontiguousarray(tris, dtype=np.int32).ravel())
    mesh.polygons.add(n)
    mesh.polygons.foreach_set("loop_start", np.arange(0, n * 3, 3, dtype=np.int32))
    if hasattr(mesh.polygons[0] if n else None, "loop_total"):
        mesh.polygons.foreach_set("loop_total", np.full(n, 3, dtype=np.int32))
    mesh.update()
    mesh.validate()
    mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
    for mat in old.materials:
        mesh.materials.append(mat)
    obj.data = mesh
    bpy.data.meshes.remove(old)
    return mesh


def synthetic_splash(n=520):
    """A stand-in crown on the pool, so the look can be tuned without a simulation."""
    x, y = np.meshgrid(np.linspace(-ns.POOL_HX, ns.POOL_HX, n),
                       np.linspace(-ns.POOL_HY, ns.POOL_HY, max(8, int(n * ns.POOL_HY / ns.POOL_HX))),
                       indexing="ij")
    z = np.full_like(x, ns.POOL_TOP)
    for cx, cy, h, r0 in ((-0.13, 0.0, 0.11, 0.05), (0.0, 0.015, 0.07, 0.035), (0.13, -0.02, 0.04, 0.025)):
        r = np.hypot(x - cx, y - cy)
        ring = h * np.exp(-((r - r0) / (r0 * 0.28)) ** 2) * (1 + 0.35 * np.cos(14 * np.arctan2(y - cy, x - cx)))
        z = z + ring + 0.004 * np.cos(r * 160.0) * np.exp(-r * 14.0)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    ny = x.shape[1]
    idx = np.arange(x.size).reshape(x.shape)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.int32)
    del ny
    return verts, tris


def make_encode_scene(frames_dir, frames, out_path, size):
    """A scene whose sequencer plays the PNGs, rendered to H.264."""
    scene = bpy.data.scenes.new("Nets Encode")
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.fps = ns.FPS
    scene.frame_start, scene.frame_end = 1, len(frames)
    if scene.sequence_editor is None:
        scene.sequence_editor_create()
    seq = scene.sequence_editor
    strips = seq.strips if hasattr(seq, "strips") else seq.sequences
    first = os.path.join(frames_dir, f"frame_{frames[0]:04d}.png")
    strip = strips.new_image("nets", first, 1, 1)
    for f in frames[1:]:
        strip.elements.append(f"frame_{f:04d}.png")
    strip.frame_final_duration = len(frames)
    r = scene.render
    # Blender 5 splits "media type" (image or video) from the file format.
    if hasattr(r.image_settings, "media_type"):
        r.image_settings.media_type = "VIDEO"
    r.image_settings.file_format = "FFMPEG"
    r.ffmpeg.format = "MPEG4"
    r.ffmpeg.codec = "H264"
    r.ffmpeg.constant_rate_factor = "HIGH"
    r.ffmpeg.ffmpeg_preset = "GOOD"
    r.filepath = out_path
    return scene


def encode_with_ffmpeg(frames_dir, frames, out_path):
    """Fallback: H.264 from the PNGs with an ffmpeg binary on PATH."""
    import shutil
    import subprocess

    exe = shutil.which("ffmpeg")
    if exe is None:
        log(f"no ffmpeg; the PNG frames in {frames_dir} are the result")
        return
    listing = os.path.join(frames_dir, "frames.txt")
    with open(listing, "w", encoding="utf-8") as fh:
        for f in frames:
            fh.write(f"file 'frame_{f:04d}.png'\nduration {1.0 / ns.FPS:.6f}\n")
    subprocess.run([exe, "-y", "-f", "concat", "-safe", "0", "-i", listing,
                    "-vf", f"fps={ns.FPS},format=yuv420p", "-c:v", "libx264",
                    "-crf", "16", "-preset", "slow", out_path], check=True)


def mesh_stamp(mesh_dir):
    """Changes whenever the saved meshes do (names, sizes and times)."""
    names = sorted(n for n in os.listdir(mesh_dir) if n.endswith(".npy"))
    return [[n, os.path.getsize(os.path.join(mesh_dir, n)),
             int(os.path.getmtime(os.path.join(mesh_dir, n)))] for n in names]


def run_render(opts):
    out = opts["out"]
    mesh_dir = os.path.join(out, "meshes")
    frames_dir = os.path.join(out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    blend = opts["blend"]
    if opts["look_test"] or not os.path.exists(blend):
        if not opts["look_test"]:
            log(f"{blend} is missing; rendering on the plain basin from nets_showcase.py")
        scene = bpy.context.scene
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        from mathutils import Vector

        for data in ns.collider_meshes():
            mesh = bpy.data.meshes.new(data.name)
            mesh.from_pydata(data.verts.tolist(), [], [list(f) for f in data.faces])
            mesh.update()
            obj = bpy.data.objects.new(data.name, mesh)
            scene.collection.objects.link(obj)
        cam = bpy.data.objects.new("Demo Cam", bpy.data.cameras.new("Demo Cam"))
        scene.collection.objects.link(cam)
        cam.location = ns.CAMERA_LOCATION
        cam.rotation_euler = (Vector(ns.CAMERA_TARGET) - cam.location).to_track_quat("-Z", "Y").to_euler()
        scene.camera = cam
    else:
        bpy.ops.wm.open_mainfile(filepath=blend)
        scene = bpy.context.scene
    # The saved demo is a live FluidSim scene. Nothing here enables the add-on,
    # so no handler advances a solver while frames render.
    for obj in scene.objects:
        if obj.name == SURFACE_OBJECT or obj.name.startswith("FluidSim "):
            obj.hide_render = True
            obj.hide_viewport = True
    setup_look(scene, opts)
    water_mat = make_water_material()
    water = make_water_object(scene, water_mat)

    if opts["look_test"]:
        set_water_mesh(water, *synthetic_splash())
        scene.render.filepath = os.path.join(out, "look_test.png")
        bpy.ops.render.render(write_still=True)
        log(f"look test written to {scene.render.filepath}")
        return

    frames = sorted(int(f[6:10]) for f in os.listdir(mesh_dir) if f.startswith("verts_"))
    manifest = os.path.join(out, "sim_series.json")
    if os.path.exists(manifest):
        # Only the frames the last simulation recorded, never stray older meshes.
        with open(manifest, encoding="utf-8") as fh:
            recorded = set(json.load(fh)["frames"])
        frames = [f for f in frames if f in recorded]
    if opts["frames"]:
        frames = [f for f in frames if opts["frames"][0] <= f <= opts["frames"][1]]
    if opts["still"] is not None:
        frames = [opts["still"]]
    if not frames:
        raise SystemExit(f"no meshes in {mesh_dir}; run the sim stage first")
    t0 = time.perf_counter()
    rendered = []
    config = {"size": list(opts["size"]), "samples": opts["samples"],
              "engine": opts["engine"], "preset": ns.preset().name,
              "mesh_stamp": mesh_stamp(mesh_dir)}
    config_path = os.path.join(frames_dir, "render_config.json")
    keep = False
    if opts["resume"] and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as fh:
            keep = json.load(fh) == config
        if not keep:
            log("render settings or meshes changed since the last run; rendering every frame again")
    if opts["still"] is None:
        if not keep:
            # PNGs from another configuration must not survive next to the new
            # one, or a later --resume would certify them.
            for name in os.listdir(frames_dir):
                if name.startswith("frame_") and name.endswith(".png"):
                    os.remove(os.path.join(frames_dir, name))
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh)
    for f in frames:
        # A still goes beside the sequence, never over a frame the cache vouches for.
        name = f"still_{f:04d}.png" if opts["still"] is not None else f"frame_{f:04d}.png"
        path = os.path.join(frames_dir, name)
        if not (keep and os.path.exists(path)) or opts["still"] is not None:
            verts = np.load(os.path.join(mesh_dir, f"verts_{f:04d}.npy"))
            tris = np.load(os.path.join(mesh_dir, f"tris_{f:04d}.npy"))
            set_water_mesh(water, verts, tris)
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            log(f"frame {f}: {len(tris):,} tris -> {path} "
                f"({time.perf_counter() - t0:.0f} s elapsed)")
        rendered.append(f)
    if opts["still"] is not None:
        return
    video = os.path.join(out, "nets_showcase.mp4")
    try:
        enc = make_encode_scene(frames_dir, rendered, video, opts["size"])
        bpy.ops.render.render(animation=True, scene=enc.name)
    except Exception as exc:  # noqa: BLE001
        log(f"Blender could not encode the video ({exc!r}); trying ffmpeg on PATH")
        encode_with_ffmpeg(frames_dir, rendered, video)
    log(f"video written to {video}")


def main():
    opts = parse_args()
    os.makedirs(opts["out"], exist_ok=True)
    if opts["stage"] == "sim":
        run_sim(opts)
    if opts["look_test"] or opts["stage"] == "render":
        run_render(opts)
    print("NETS-SHOWCASE-PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("NETS-SHOWCASE-FAIL", flush=True)
        sys.exit(1)
