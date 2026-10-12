"""Source tagging, solver descriptor translation, and particle injection."""

from __future__ import annotations

from dataclasses import dataclass

import bpy
import numpy as np

from ..sim.attributes import stable_source_id
from .scene_inputs import (
    CURRENT_SOURCE_UI_SCHEMA,
    legacy_region_kinds,
    persistent_source_key,
    uses_native_source_plan,
)

# Recorded in every cache-v3 manifest. Bump it when what a source or collider
# contributes to a frame's identity changes, so older caches stop resuming.
SOURCE_SCHEMA_VERSION = 1

_handlers =getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)


@dataclass(frozen=True)
class PreparedParticleSeed:
    """Owned host payload prepared without touching solver particle buffers."""

    stable_id: str
    positions: np.ndarray
    velocity: tuple[float, float, float]
    temperature: float
    viscosity: float
    density: float
    color: tuple[float, float, float, float]
    source_id: int
    life: float
    use_lifespan: bool

    def __post_init__(self):
        positions = np.array(self.positions, dtype=np.float32, copy=True, order="C")
        positions = positions.reshape((-1, 3))
        positions.setflags(write=False)
        object.__setattr__(self, "positions", positions)


def migrate_source_sampling_schema(scene=None):
    """Persist old source occupancy as LEGACY_ABSOLUTE without guessing intent."""
    from contextlib import ExitStack

    data = getattr(bpy, "data", None)
    objects = (getattr(scene, "objects", ()) if scene is not None
               else getattr(data, "objects", ()))
    scenes = ((scene,) if scene is not None
              else tuple(getattr(data, "scenes", ()) or ()))
    pending = []
    for obj in objects:
        pg = getattr(obj, "fluidsim", None)
        if pg is None or getattr(pg, "role", "NONE") != "SOURCE":
            continue
        schema = int(getattr(pg, "source_ui_schema_version", 0) or 0)
        if schema >= 4 or getattr(obj, "library", None) is not None:
            continue
        pending.append((pg, schema))
    with ExitStack() as stack:
        if pending and scenes:
            # Restoring saved behaviour is not an edit: no pending reset.
            from . import pending_settings

            for owner in scenes:
                stack.enter_context(pending_settings.suppress_callbacks(owner))
        for pg, schema in pending:
            if schema < 3:
                # Lifted past schema 3 here, so pin the region kinds the
                # legacy geometry mode implied before they start being read.
                pg.source_region_kind, pg.sink_region_kind = legacy_region_kinds(pg)
            pg.source_sampling_mode = "LEGACY_ABSOLUTE"
            pg.source_sampling_multiplier = 1.0
            pg.source_ui_schema_version = 4
    return len(pending)


def migrate_native_source_evaluation(scene=None):
    """Keep pre-B47 point/volume/field sources animating, once per object.

    Before schema 5, point, volume and field inputs were re-gathered every
    frame even when ``mesh_evaluation_mode`` said First Sim Frame (the
    default), so they followed animation. First Sim Frame now really freezes
    them; a saved source that stored First Sim Frame is switched to Every Sim
    Frame, which is what it did. Mesh sources always froze and keep their mode.
    Every source, sink and field object is then stamped with the current
    schema, so a later deliberate choice of First Sim Frame survives reloads;
    below schema 3 the region kinds were ignored, so they are pinned to the
    kinds the legacy geometry mode implied rather than revived by the stamp
    (the sampling step already pinned liquid sources it lifted). Run after
    :func:`migrate_source_sampling_schema`. Returns the number switched.
    """
    from contextlib import ExitStack

    data = getattr(bpy, "data", None)
    objects = (getattr(scene, "objects", ()) if scene is not None
               else getattr(data, "objects", ()))
    scenes = ((scene,) if scene is not None
              else tuple(getattr(data, "scenes", ()) or ()))
    pending = []
    for obj in objects:
        pg = getattr(obj, "fluidsim", None)
        if pg is None or getattr(pg, "role", "NONE") not in {"SOURCE", "SINK", "FIELD"}:
            continue
        schema = int(getattr(pg, "source_ui_schema_version", 0) or 0)
        if schema >= CURRENT_SOURCE_UI_SCHEMA or getattr(obj, "library", None) is not None:
            continue
        pending.append((pg, schema, uses_native_source_plan(pg)))
    switched = 0
    with ExitStack() as stack:
        if pending and scenes:
            # Restoring saved behaviour is not an edit: no pending reset.
            from . import pending_settings

            for owner in scenes:
                stack.enter_context(pending_settings.suppress_callbacks(owner))
        for pg, schema, native in pending:
            if (native and getattr(pg, "mesh_evaluation_mode", "FIRST_SIM_FRAME")
                    != "EVERY_SIM_FRAME"):
                pg.mesh_evaluation_mode = "EVERY_SIM_FRAME"
                switched += 1
            if schema < 3:
                pg.source_region_kind, pg.sink_region_kind = legacy_region_kinds(pg)
            pg.source_ui_schema_version = CURRENT_SOURCE_UI_SCHEMA
    return switched


def migrate_source_schema(scene=None):
    """Bring saved sources to the current schema; see the two steps."""
    migrate_source_sampling_schema(scene)
    return migrate_native_source_evaluation(scene)


@persistent
def _on_load_post(_filepath):
    migrate_source_schema()


def register_handlers():
    unregister_handlers()
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    load_post = getattr(handlers, "load_post", None)
    if load_post is not None:
        load_post.append(_on_load_post)
    migrate_source_schema(getattr(bpy.context, "scene", None))


def unregister_handlers():
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    load_post = getattr(handlers, "load_post", None)
    if load_post is None:
        return
    while _on_load_post in load_post:
        load_post.remove(_on_load_post)


def tag_source(obj, shape="BOX", mode="PARTICLES"):
    """Create a new P3 liquid emitter without rewriting legacy RNA defaults."""
    obj.fluidsim.role = "SOURCE"
    obj.fluidsim.shape = shape
    obj.fluidsim.source_mode = mode
    obj.fluidsim.source_ui_schema_version = CURRENT_SOURCE_UI_SCHEMA
    obj.fluidsim.source_sampling_mode = "INHERIT"
    obj.fluidsim.source_sampling_multiplier = 1.0
    obj.fluidsim.source_region_kind = "ANALYTIC"
    obj.fluidsim.emission_mode = "INITIAL_FILL"
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def tag_sink(obj, shape="BOX"):
    obj.fluidsim.role = "SINK"
    obj.fluidsim.shape = shape
    obj.fluidsim.source_ui_schema_version = CURRENT_SOURCE_UI_SCHEMA
    obj.fluidsim.sink_region_kind = "ANALYTIC"
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def tag_field(obj, shape="BOX"):
    """Tag a non-liquid Advanced Field Source."""
    obj.fluidsim.role = "FIELD"
    obj.fluidsim.shape = shape
    obj.fluidsim.source_mode = "FIELD"
    obj.fluidsim.source_ui_schema_version = CURRENT_SOURCE_UI_SCHEMA
    obj.fluidsim.field_operation = "PRESSURE"
    obj.fluidsim.field_region_kind = "ANALYTIC"
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def tag_force(obj, shape="BOX"):
    """Tag an object as an analytic force region without sourcing liquid."""
    obj.fluidsim.role = "FORCE"
    obj.fluidsim.shape = shape
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def gather_sources(scene, session, depsgraph=None, *, capture_first_mesh=True):
    """Compatibility wrapper for the SceneInputs owner."""
    return getattr(session, "scene_inputs", session).gather_sources(
        scene, depsgraph, capture_first_mesh=capture_first_mesh)


def gather_height_source(scene):
    """Compatibility wrapper for callers without a live Session."""
    from .scene_inputs import SceneInputs
    return SceneInputs().gather_height_source(scene)


def object_points_world(obj, depsgraph=None):
    """Extract world-space points from a mesh / point cloud / particle system.

    Used for direct particle injection (fluidSimSourceParticles path).
    """
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mw = np.array(obj.matrix_world, dtype=np.float32)

    # Particle system on the object takes priority.
    if eval_obj.particle_systems:
        psys = eval_obj.particle_systems[0]
        n = len(psys.particles)
        if n:
            pts = np.empty(n * 3, dtype=np.float32)
            psys.particles.foreach_get("location", pts)
            return pts.reshape(n, 3)  # already world space

    if obj.type == "POINTCLOUD":
        pc = eval_obj.data
        n = len(pc.points)
        if n:
            pts = np.empty(n * 3, dtype=np.float32)
            pc.attributes["position"].data.foreach_get("vector", pts)
            pts = pts.reshape(n, 3)
            return pts @ mw[:3, :3].T + mw[:3, 3]

    if obj.type == "MESH":
        mesh = eval_obj.to_mesh()
        try:
            n = len(mesh.vertices)
            pts = np.empty(n * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", pts)
        finally:
            # A raise between to_mesh and here would otherwise leak the
            # evaluated mesh until the object is freed (REVIEW-041).
            eval_obj.to_mesh_clear()
        return pts.reshape(n, 3) @ mw[:3, :3].T + mw[:3, 3]

    return np.zeros((0, 3), dtype=np.float32)


def prepare_object_particles(scene, obj, velocity=(0.0, 0.0, 0.0),
                             depsgraph=None):
    """Prepare one immutable direct-seed payload without solver mutation."""
    points = object_points_world(obj, depsgraph)
    if not len(points):
        return None
    pg = obj.fluidsim
    identity = persistent_source_key(obj)
    return PreparedParticleSeed(
        stable_id=f"blender-seed:{identity}",
        positions=points,
        velocity=(tuple(velocity) if any(velocity)
                  else tuple(float(value) for value in pg.initial_velocity)),
        temperature=float(pg.temperature),
        viscosity=float(pg.viscosity),
        density=float(pg.density),
        color=tuple(float(value) for value in getattr(
            pg, "source_color", (0.1, 0.45, 1.0, 1.0))),
        source_id=stable_source_id(f"blender-seed:{identity}"),
        life=((float(pg.life_min) + float(pg.life_max)) * 0.5
              if pg.use_lifespan else 0.0),
        use_lifespan=bool(pg.use_lifespan),
    )


def prepare_auto_seeds(scene, depsgraph=None):
    """Gather all enabled automatic seeds without touching a live solver."""
    prepared = []
    for obj in scene.objects:
        pg = getattr(obj, "fluidsim", None)
        if (pg is not None and pg.enabled
                and getattr(pg, "auto_inject_seed", False)):
            seed = prepare_object_particles(scene, obj, depsgraph=depsgraph)
            if seed is not None:
                prepared.append(seed)
    return tuple(prepared)


def commit_prepared_seeds(sess, prepared):
    """Commit every prepared payload under one device admission gate."""
    result = sess.solver.inject_particle_packets_exact(tuple(prepared))
    return int(result.emitted) if result.admitted else 0


def inject_object_particles(scene, obj, sess, velocity=(0.0, 0.0, 0.0)):
    """Compatibility one-shot using the prepare/commit transaction seam."""
    seed = prepare_object_particles(scene, obj, velocity)
    emitted = commit_prepared_seeds(sess, () if seed is None else (seed,))
    # commit_prepared_seeds reports a rejected batch through last_apply_result
    # instead of raising, so initialize()/reset() can apply their own deferral
    # policy. This one-shot has no such policy: surface it as an error rather
    # than reporting "Injected 0 particles" as a success.
    _raise_if_seed_batch_rejected(sess)
    return emitted


def _raise_if_seed_batch_rejected(sess):
    """Raise the shared capacity message when the last seed commit was refused."""
    result = getattr(
        getattr(sess.solver, "sourcing", None), "last_apply_result", {})
    if not result or result.get("admitted", True):
        return
    from types import SimpleNamespace

    from .scene_bridge import capacity_blocking_message

    raise RuntimeError(capacity_blocking_message((SimpleNamespace(
        code="PARTICLE_CAPACITY",
        required=result.get("required"),
        available=result.get("available"),
    ),)))


def inject_auto_seeds(scene, sess):
    """Restore enabled particle seeds after solver initialize/reset."""
    return commit_prepared_seeds(sess, prepare_auto_seeds(scene))
