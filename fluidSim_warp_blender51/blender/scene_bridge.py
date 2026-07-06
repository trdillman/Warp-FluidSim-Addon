"""Scene <-> solver bridge.

Owns the live solver session (module-level, cannot live in ID properties),
translates Blender scene/object data into solver descriptors, and steps the
simulation during playback via frame-change handlers.
"""

from __future__ import annotations

import bpy
import numpy as np
from mathutils import Vector

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class Session:
    """Live GPU solver session for one scene."""

    def __init__(self):
        self.solver = None
        self.profiler = None
        self.last_frame = -1
        self.start_frame = 1
        self.frame_stats = []
        self.validation_results = []
        self.benchmark_result = None
        self.benchmark_compare = None
        self.error = ""
        self._collider_prev = {}  # object name -> (matrix, frame)
        self._collider_mesh_cache = {}
        self.collider_mesh_cache_stats = {"rebuilds": 0, "reuses": 0}

    @property
    def active(self):
        return self.solver is not None


_SESSION = Session()
ERROR_SCENE_KEY = "fluidsim_last_error"


def session() -> Session:
    return _SESSION


def _scene_get(scene, key: str, default=""):
    try:
        return scene.get(key, default)
    except AttributeError:
        return getattr(scene, key, default)


def _scene_set(scene, key: str, value):
    try:
        scene[key] = value
    except TypeError:
        setattr(scene, key, value)


def set_error(scene, message: str):
    """Persist the latest simulation error for panel-visible recovery."""
    text = str(message)
    session().error = text
    _scene_set(scene, ERROR_SCENE_KEY, text)


def clear_error(scene):
    session().error = ""
    _scene_set(scene, ERROR_SCENE_KEY, "")


def current_error(scene) -> str:
    return str(_scene_get(scene, ERROR_SCENE_KEY, "") or session().error or "")


# ---------------------------------------------------------------------------
# Settings translation
# ---------------------------------------------------------------------------


def _domain_bounds(scene) -> tuple:
    fs = scene.fluidsim
    obj = fs.domain_object
    if obj is not None:
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        lo = (min(c.x for c in corners), min(c.y for c in corners),
              min(c.z for c in corners))
        hi = (max(c.x for c in corners), max(c.y for c in corners),
              max(c.z for c in corners))
        return lo, hi
    return (-2.0, -2.0, 0.0), (2.0, 2.0, 4.0)


def build_settings(scene):
    """Create SolverSettings from scene properties (imports Warp lazily)."""
    from ..sim.enums import (
        LimitBehavior,
        MultiGrid,
        ParticleSampleType,
        ProjectNonDivergentSolver,
        RelaxMode,
        SdfFilterType,
        TraceMode,
        ViscositySolver,
    )
    from ..sim.solver import SolverSettings

    fs = scene.fluidsim
    s = SolverSettings()
    try:
        prefs = bpy.context.preferences.addons["fluidSim_warp_blender51"].preferences
        if prefs.device:
            s.device = prefs.device
    except (KeyError, AttributeError):
        pass
    lo, hi = _domain_bounds(scene)
    s.domain_min = lo
    s.domain_max = hi
    s.voxel_size = fs.voxel_size
    s.leaf_capacity = fs.leaf_capacity
    s.topology_expand = fs.topology_expand
    s.open_boundaries = fs.open_boundaries
    s.max_particles = fs.max_particles
    s.max_whitewater = fs.max_whitewater
    s.frame_rate = scene.render.fps / scene.render.fps_base
    s.time_scale = fs.time_scale
    s.cfl = fs.cfl
    s.min_substeps = fs.min_substeps
    s.max_substeps = fs.max_substeps
    if fs.use_scene_gravity and scene.use_gravity:
        s.gravity = tuple(scene.gravity)
    else:
        s.gravity = tuple(fs.gravity)
    s.sample_type = ParticleSampleType[fs.sample_type]
    s.flip_ratio = fs.flip_ratio
    s.trace_mode = TraceMode[fs.trace_mode]
    s.max_speed = fs.max_speed
    s.limit_behavior = LimitBehavior[fs.limit_behavior]
    s.limit_percent = fs.limit_percent
    s.particles_per_voxel = fs.particles_per_voxel
    s.count_stiffness = fs.count_stiffness
    s.relax_enabled = fs.relax_enabled
    s.relax_strength = fs.relax_strength
    s.relax_mode = RelaxMode[fs.relax_mode]
    s.restitution = fs.restitution
    s.friction = fs.friction
    s.pressure_solver = ProjectNonDivergentSolver[fs.pressure_solver]
    s.pressure_iterations = fs.pressure_iterations
    s.pressure_tolerance = fs.pressure_tolerance
    s.pressure_multigrid = MultiGrid[fs.multigrid]
    s.pressure_threshold = fs.pressure_threshold
    s.viscosity_enabled = fs.viscosity_enabled
    s.viscosity_solver = ViscositySolver[fs.viscosity_solver]
    s.viscosity_constant = fs.viscosity_constant
    s.viscosity_from_particles = fs.viscosity_from_particles
    s.viscosity_iterations = fs.viscosity_iterations
    s.viscosity_tolerance = fs.viscosity_tolerance
    s.surface_enabled = fs.surface_enabled
    s.particle_radius_scale = fs.particle_radius_scale
    s.surface_filter_type = SdfFilterType[fs.surface_filter_type]
    s.surface_filter_iterations = fs.surface_filter_iterations
    s.surface_filter_strength = fs.surface_filter_strength
    s.whitewater_enabled = fs.whitewater_enabled
    s.ww_churn_decay = fs.ww_churn_decay
    s.ww_churn_gain = fs.ww_churn_gain
    s.ww_churn_threshold = fs.ww_churn_threshold
    s.ww_emission_rate = fs.ww_emission_rate
    s.ww_min_speed = fs.ww_min_speed
    s.ww_life_min = fs.ww_life_min
    s.ww_life_max = fs.ww_life_max
    s.ww_bubble_drag = fs.ww_bubble_drag
    s.ww_spray_drag = fs.ww_spray_drag
    return s


def snippet_argument_names(pg):
    """Names a snippet's declared arguments expose to its body."""
    names = []
    for apg in pg.arguments:
        names.append(apg.field if apg.mode == "FIELD" else apg.constant_name)
    return names


def validate_snippet_pg(pg):
    """Statically validate one snippet property group and mirror the result.

    Updates ``is_valid`` / ``validated_hash`` / ``validation_report`` /
    ``last_error`` on the property group and returns the list of issues.
    """
    from ..sim import snippet_library

    code = pg.text_block.as_string() if pg.text_block else ""
    issues = snippet_library.validate_snippet(
        code, pg.run_over, snippet_argument_names(pg))
    errors = [i for i in issues if i.level == "ERROR"]
    pg.is_valid = not errors
    pg.validated_hash = snippet_library.code_hash(code)
    pg.validation_report = "\n".join(str(i) for i in issues)[:1024]
    if errors:
        pg.last_error = str(errors[0])[:256]
    elif pg.last_error.startswith("ERROR:"):
        pg.last_error = ""
    return issues


def build_snippets(scene):
    """Convert snippet property groups into sim CustomSnippet objects.

    This is the single choke point through which snippets reach the solver
    (both solver init and Apply). Enabled snippets are statically validated
    here; snippets with errors are gated out so only valid code can run in
    the scene.
    """
    result = []
    for pg in scene.fluidsim.snippets:
        if pg.enabled:
            validate_snippet_pg(pg)
            if not pg.is_valid:
                continue
        result.append(snippet_from_pg(pg))
    return result


def snippet_from_pg(pg):
    """Convert one snippet property group into a sim CustomSnippet."""
    from ..sim.custom import CustomArgument, CustomSnippet
    from ..sim.enums import (
        CustomKernelArgumentMode,
        CustomKernelConstantType,
        CustomKernelOrder,
        CustomKernelRunOver,
    )

    code = pg.text_block.as_string() if pg.text_block else ""
    args = []
    for apg in pg.arguments:
        args.append(CustomArgument(
            mode=CustomKernelArgumentMode[apg.mode],
            field=apg.field,
            constant_type=CustomKernelConstantType[apg.constant_type],
            constant_name=apg.constant_name,
            constant_value=apg.constant_value,
        ))
    return CustomSnippet(
        enabled=pg.enabled,
        kernel_name=pg.kernel_name,
        order=CustomKernelOrder[pg.order],
        run_over=CustomKernelRunOver[pg.run_over],
        code=code,
        arguments=args,
        frame_start=pg.frame_start,
        frame_end=pg.frame_end,
        frame_step=pg.frame_step,
    )


# ---------------------------------------------------------------------------
# Object -> descriptor translation
# ---------------------------------------------------------------------------


def _object_velocity(obj, scene, session_obj):
    """Estimate rigid velocity of an animated object by frame differencing."""
    fps = scene.render.fps / scene.render.fps_base
    cur = np.array(obj.matrix_world.translation)
    key = obj.name
    prev = session_obj._collider_prev.get(key)
    frame = scene.frame_current
    session_obj._collider_prev[key] = (cur, frame)
    if prev is None or frame == prev[1]:
        return (0.0, 0.0, 0.0)
    dt = (frame - prev[1]) / fps
    if dt == 0:
        return (0.0, 0.0, 0.0)
    v = (cur - prev[0]) / dt
    return tuple(float(x) for x in v)


def _object_identity(obj):
    as_pointer = getattr(obj, "as_pointer", None)
    if as_pointer is not None:
        return int(as_pointer())
    return id(obj)


def _array_signature(array):
    contiguous = np.ascontiguousarray(array)
    return (contiguous.shape, str(contiguous.dtype), contiguous.tobytes())


def _collider_cache_signature(object_id, device, verts, tris, matrix_world):
    matrix = np.ascontiguousarray(matrix_world, dtype=np.float32)
    return (
        object_id,
        device,
        _array_signature(verts),
        _array_signature(tris),
        _array_signature(matrix),
    )


def gather_sources(scene, sess: Session):
    """Build SourceShapeDesc list from tagged objects + height source."""
    from ..sim.enums import (
        BlendMode,
        SourceShapeForceType,
        SourceShapeShape,
        SourceShapeSourceMode,
        SourceShapeType,
    )
    from ..sim.solver import SourceShapeDesc

    sources = []
    for obj in scene.objects:
        pg = getattr(obj, "fluidsim", None)
        if pg is None or pg.role not in {"SOURCE", "SINK", "FORCE"} or not pg.enabled:
            continue
        if obj.hide_viewport:
            continue
        desc = SourceShapeDesc()
        desc.shape = SourceShapeShape[pg.shape]
        if pg.role == "SOURCE":
            desc.stype = SourceShapeType.SOURCE
        elif pg.role == "SINK":
            desc.stype = SourceShapeType.SINK
        else:
            desc.stype = SourceShapeType.SOURCE
        desc.mode = SourceShapeSourceMode[pg.source_mode]
        mw = obj.matrix_world
        loc = mw.to_translation()
        quat = mw.to_quaternion()
        scale = mw.to_scale()
        desc.position = (loc.x, loc.y, loc.z)
        desc.rotation = (quat.x, quat.y, quat.z, quat.w)
        # Analytic shapes scale via half extents / radius from object scale
        # and the object's bounding box (unit primitives -> world size).
        dims = obj.dimensions
        desc.half_extents = (max(dims.x, 1e-4) / 2,
                             max(dims.y, 1e-4) / 2,
                             max(dims.z, 1e-4) / 2)
        desc.radius = max(dims.x, dims.y, dims.z) / 2
        desc.height = max(dims.z, 1e-4) / 2
        desc.feather = pg.feather
        vel = tuple(pg.initial_velocity)
        if pg.use_object_velocity:
            ov = _object_velocity(obj, scene, sess)
            vel = (vel[0] + ov[0], vel[1] + ov[1], vel[2] + ov[2])
        desc.velocity = vel
        desc.blend_mode = BlendMode[pg.blend_mode]
        desc.emit_rate = pg.emit_rate
        desc.temperature = pg.temperature
        desc.viscosity = pg.viscosity
        desc.density = pg.density
        desc.use_lifespan = pg.use_lifespan
        desc.life_min = pg.life_min
        desc.life_max = pg.life_max
        if pg.role == "FORCE":
            desc.stype = SourceShapeType.SOURCE
            desc.mode = SourceShapeSourceMode.FIELD
            desc.emit_rate = 0.0
        desc.force_type = SourceShapeForceType[pg.force_type]
        desc.force_strength = pg.force_strength
        desc.force_dir = tuple(pg.force_direction)
        sources.append(desc)
    return sources


def gather_colliders(scene, sess: Session, depsgraph=None):
    """Build ColliderDesc list (world-space wp.Mesh) from tagged objects."""
    import warp as wp

    from ..sim.solver import ColliderDesc

    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    colliders = []
    seen_cache_keys = set()
    for obj in scene.objects:
        pg = getattr(obj, "fluidsim", None)
        if pg is None or pg.role != "COLLIDER" or not pg.enabled:
            continue
        if obj.type != "MESH" or obj.hide_viewport:
            continue
        object_id = _object_identity(obj)
        seen_cache_keys.add(object_id)
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None or len(mesh.polygons) == 0:
            eval_obj.to_mesh_clear()
            sess._collider_mesh_cache.pop(object_id, None)
            continue
        mesh.calc_loop_triangles()
        nv = len(mesh.vertices)
        nt = len(mesh.loop_triangles)
        verts = np.empty(nv * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts)
        verts = verts.reshape(nv, 3)
        tris = np.empty(nt * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
        mw = np.array(obj.matrix_world, dtype=np.float32)
        verts_w = verts @ mw[:3, :3].T + mw[:3, 3]
        eval_obj.to_mesh_clear()

        device = sess.solver.device if sess.solver else "cuda:0"
        signature = _collider_cache_signature(object_id, device, verts, tris, mw)
        entry = sess._collider_mesh_cache.get(object_id)
        if entry is not None and entry["signature"] == signature:
            wmesh = entry["mesh"]
            lo = entry["aabb_min"]
            hi = entry["aabb_max"]
            sess.collider_mesh_cache_stats["reuses"] += 1
        else:
            wmesh = wp.Mesh(
                points=wp.array(verts_w, dtype=wp.vec3, device=device),
                indices=wp.array(tris, dtype=wp.int32, device=device),
            )
            lo = verts_w.min(axis=0)
            hi = verts_w.max(axis=0)
            sess._collider_mesh_cache[object_id] = {
                "signature": signature,
                "mesh": wmesh,
                "aabb_min": lo,
                "aabb_max": hi,
            }
            sess.collider_mesh_cache_stats["rebuilds"] += 1
        desc = ColliderDesc()
        desc.mesh = wmesh
        desc.aabb_min = tuple(float(x) for x in lo)
        desc.aabb_max = tuple(float(x) for x in hi)
        c = obj.matrix_world.translation
        desc.center = (c.x, c.y, c.z)
        if pg.transfer_velocity:
            desc.linear_velocity = _object_velocity(obj, scene, sess)
        colliders.append(desc)
    for cache_key in list(sess._collider_mesh_cache):
        if cache_key not in seen_cache_keys:
            del sess._collider_mesh_cache[cache_key]
    return colliders


def gather_height_source(scene):
    from ..sim.enums import SourceHeightBackMode
    from ..sim.solver import HeightSourceDesc

    fs = scene.fluidsim
    if not fs.height_enabled:
        return None
    desc = HeightSourceDesc()
    desc.enabled = True
    desc.base_height = fs.height_base
    desc.back_mode = SourceHeightBackMode[fs.height_back_mode]
    desc.velocity = tuple(fs.height_velocity)
    desc.emit_particles = fs.height_emit_particles
    desc.emit_rate = fs.particles_per_voxel
    return desc


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def initialize_session(scene) -> Session:
    """(Re)create the solver from current scene state."""
    from ..diagnostics.profiler import Profiler
    from ..sim.solver import FluidSolver

    sess = session()
    release_session()
    fs = scene.fluidsim
    settings = build_settings(scene)
    profiler = None
    if fs.profiling_enabled:
        profiler = Profiler(device=settings.device, sync=fs.profiling_sync)
    solver = FluidSolver(settings, profiler=profiler)
    sess.solver = solver
    sess.profiler = profiler
    sess.start_frame = scene.frame_start
    sess.last_frame = -1
    sess.frame_stats = []
    clear_error(scene)
    sess._collider_prev = {}
    sess._collider_mesh_cache = {}
    sess.collider_mesh_cache_stats = {"rebuilds": 0, "reuses": 0}
    sync_scene_to_solver(scene, sess)
    errors = solver.set_custom_snippets(build_snippets(scene))
    for name, err in errors:
        for pg in fs.snippets:
            if pg.kernel_name == name:
                pg.last_error = str(err)[:256]
    fs.status = "Solver ready"
    return sess


def release_session():
    sess = session()
    sess.solver = None
    sess.profiler = None
    sess.last_frame = -1
    sess.frame_stats = []
    sess._collider_prev = {}
    sess._collider_mesh_cache = {}
    sess.collider_mesh_cache_stats = {"rebuilds": 0, "reuses": 0}


def sync_scene_to_solver(scene, sess: Session | None = None, depsgraph=None):
    """Push current scene objects (sources/colliders/height) to the solver."""
    sess = sess or session()
    if not sess.active:
        return
    sess.solver.set_sources(gather_sources(scene, sess))
    sess.solver.set_colliders(gather_colliders(scene, sess, depsgraph))
    sess.solver.set_height_source(gather_height_source(scene))


def step_frame(scene, sess: Session | None = None, *, write_cache: bool = True):
    """Advance the solver one frame and update output objects."""
    from . import viewport

    sess = sess or session()
    if not sess.active:
        return None
    fs = scene.fluidsim
    stats = sess.solver.step_frame()
    sess.frame_stats.append(stats.as_dict())
    if len(sess.frame_stats) > 1000:
        sess.frame_stats.pop(0)
    fs.stat_particles = stats.active_particles
    fs.stat_voxels = stats.active_voxels
    fs.stat_whitewater = stats.whitewater_particles
    fs.stat_step_ms = stats.step_time * 1000.0
    fs.stat_substeps = stats.substeps
    fs.status = (
        f"Frame {sess.solver.frame}: {stats.active_particles:,} particles, "
        f"{stats.active_voxels:,} voxels, {stats.step_time * 1000.0:.1f} ms"
    )
    viewport.update_outputs(scene, sess)
    if write_cache and fs.cache_enabled:
        from ..io import cache

        try:
            cache.write_frame(scene, sess)
        except Exception as exc:  # noqa: BLE001 - cache is supplemental
            print(f"fluidSim cache write failed: {exc!r}")
    return stats


def play_cache_frame(scene, frame: int | None = None):
    """Load one cached timeline frame into the regular output objects."""
    from . import viewport
    from ..io import cache

    fs = scene.fluidsim
    frame = int(scene.frame_current if frame is None else frame)
    if getattr(fs, "cache_use_custom_range", False):
        start, end = cache.resolve_frame_range(
            scene,
            use_custom=True,
            frame_start=fs.cache_frame_start,
            frame_end=fs.cache_frame_end,
        )
        if frame < start or frame > end:
            raise FileNotFoundError(
                f"Cache frame {frame} is outside playback range {start}-{end}"
            )
    base = bpy.path.abspath(fs.cache_path)
    frame_data = cache.read_frame(base, frame)
    if frame_data is None:
        raise FileNotFoundError(f"Cache frame {frame} is missing or corrupt in {base}")
    viewport.update_outputs_from_cache_frame(scene, frame_data)
    fs.stat_particles = len(frame_data.get("positions", ()))
    fs.stat_whitewater = len(frame_data.get("whitewater", ()))
    fs.stat_voxels = 0
    fs.stat_step_ms = 0.0
    fs.stat_substeps = 0
    fs.status = f"Cache frame {frame}: {fs.stat_particles:,} particles"
    clear_error(scene)
    return frame_data


def reset_simulation(scene):
    """Reset solver state and clear outputs (used at playback start)."""
    from . import viewport

    sess = session()
    if not sess.active:
        return
    sess.solver.reset()
    sess.last_frame = -1
    sess.frame_stats = []
    clear_error(scene)
    sess._collider_prev = {}
    sess._collider_mesh_cache = {}
    sess.collider_mesh_cache_stats = {"rebuilds": 0, "reuses": 0}
    sync_scene_to_solver(scene, sess)
    viewport.update_outputs(scene, sess)
    scene.fluidsim.status = "Simulation reset"


# ---------------------------------------------------------------------------
# Handlers (interactive playback)
# ---------------------------------------------------------------------------


def _on_frame_change(scene, depsgraph):
    sess = session()
    fs = getattr(scene, "fluidsim", None)
    if fs is None or not fs.live:
        return
    frame = scene.frame_current
    if frame == sess.last_frame:
        return
    if getattr(fs, "cache_playback_enabled", False):
        sess.last_frame = frame
        try:
            play_cache_frame(scene, frame)
        except Exception as exc:  # noqa: BLE001 - never break playback
            set_error(scene, repr(exc))
            fs.status = f"Cache playback error: {exc}"
            print(f"fluidSim cache playback failed: {exc!r}")
        return
    if not sess.active:
        return
    if frame <= sess.start_frame or frame < sess.last_frame:
        # Rewind / loop restart -> reset the simulation.
        reset_simulation(scene)
        sess.last_frame = frame
        return
    # Step once per new frame (frame skips advance a single step to stay
    # interactive rather than simulating every skipped frame).
    sess.last_frame = frame
    try:
        sync_scene_to_solver(scene, sess, depsgraph)
        step_frame(scene, sess)
    except Exception as exc:  # noqa: BLE001 - never break playback
        set_error(scene, repr(exc))
        fs.status = f"Solver error: {exc}"
        print(f"fluidSim step failed: {exc!r}")


def register_handlers():
    unregister_handlers()
    bpy.app.handlers.frame_change_post.append(_on_frame_change)


def unregister_handlers():
    handlers = bpy.app.handlers.frame_change_post
    for fn in list(handlers):
        if getattr(fn, "__module__", "").startswith("fluidSim_warp_blender51"):
            handlers.remove(fn)
