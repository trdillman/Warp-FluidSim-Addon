"""Source tagging, solver descriptor translation, and particle injection."""

from __future__ import annotations

from dataclasses import dataclass

import bpy
import numpy as np


@dataclass(frozen=True)
class PreparedParticleSeed:
    """Owned host payload prepared without touching solver particle buffers."""

    stable_id: str
    positions: np.ndarray
    velocity: tuple[float, float, float]
    temperature: float
    viscosity: float
    density: float
    life: float
    use_lifespan: bool

    def __post_init__(self):
        positions = np.array(self.positions, dtype=np.float32, copy=True, order="C")
        positions = positions.reshape((-1, 3))
        positions.setflags(write=False)
        object.__setattr__(self, "positions", positions)


def tag_source(obj, shape="BOX", mode="PARTICLES"):
    """Create a new P3 liquid emitter without rewriting legacy RNA defaults."""
    obj.fluidsim.role = "SOURCE"
    obj.fluidsim.shape = shape
    obj.fluidsim.source_mode = mode
    obj.fluidsim.source_ui_schema_version = 3
    obj.fluidsim.source_region_kind = "ANALYTIC"
    obj.fluidsim.emission_mode = "INITIAL_FILL"
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def tag_sink(obj, shape="BOX"):
    obj.fluidsim.role = "SINK"
    obj.fluidsim.shape = shape
    obj.fluidsim.source_ui_schema_version = 3
    obj.fluidsim.sink_region_kind = "ANALYTIC"
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def tag_field(obj, shape="BOX"):
    """Tag a non-liquid Advanced Field Source."""
    obj.fluidsim.role = "FIELD"
    obj.fluidsim.shape = shape
    obj.fluidsim.source_mode = "FIELD"
    obj.fluidsim.source_ui_schema_version = 3
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
        n = len(mesh.vertices)
        pts = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", pts)
        pts = pts.reshape(n, 3)
        out = pts @ mw[:3, :3].T + mw[:3, 3]
        eval_obj.to_mesh_clear()
        return out

    return np.zeros((0, 3), dtype=np.float32)


def prepare_object_particles(scene, obj, velocity=(0.0, 0.0, 0.0),
                             depsgraph=None):
    """Prepare one immutable direct-seed payload without solver mutation."""
    points = object_points_world(obj, depsgraph)
    if not len(points):
        return None
    pg = obj.fluidsim
    pointer = getattr(obj, "as_pointer", None)
    identity = int(pointer()) if callable(pointer) else id(obj)
    return PreparedParticleSeed(
        stable_id=f"blender-seed:{identity}",
        positions=points,
        velocity=(tuple(velocity) if any(velocity)
                  else tuple(float(value) for value in pg.initial_velocity)),
        temperature=float(pg.temperature),
        viscosity=float(pg.viscosity),
        density=float(pg.density),
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
    """Commit an already-admitted seed batch atomically with respect to capacity."""
    prepared = tuple(prepared)
    solver = sess.solver
    particles = solver.particles
    required = sum(len(seed.positions) for seed in prepared)
    available = max(0, int(particles.capacity) - int(particles.host_count))
    if required > available:
        grow = getattr(solver, "ensure_particle_capacity", None)
        if callable(grow):
            growth = grow(int(particles.host_count) + required)
        else:
            from ..sim.particle_capacity import ParticleGrowthResult

            current = int(particles.capacity)
            growth = ParticleGrowthResult(
                "DISABLED", current, current,
                int(particles.host_count) + required, current,
                reason="solver does not expose transactional particle growth")
        particles = solver.particles
        available = max(
            0, int(particles.capacity) - int(particles.host_count))
        if required > available:
            policy = str(getattr(
                getattr(solver, "settings", None),
                "particle_growth_failure_policy", "DEFER",
            )).upper()
            solver.sourcing.last_apply_result = {
                "admitted": False,
                "required": required,
                "available": available,
                "emitted": 0,
                "capacity_limited": True,
                "deferred": True,
                "pause_requested": policy == "PAUSE",
                "growth_status": growth.status,
                "growth_previous": growth.previous,
                "growth_current": growth.current,
                "growth_target": growth.target,
                "growth_reason": growth.reason,
            }
            return 0
    emitted = 0
    for seed in prepared:
        before = int(particles.host_count)
        solver.inject_particles(
            seed.positions,
            velocity=seed.velocity,
            temperature=seed.temperature,
            viscosity=seed.viscosity,
            density=seed.density,
            life=seed.life,
            use_lifespan=seed.use_lifespan,
        )
        # The public solver seam historically returns the final total, not the
        # append delta. Count from authoritative host state so multiple seed
        # objects cannot be double-counted.
        emitted += int(particles.host_count) - before
    if emitted != required:
        raise RuntimeError(
            "automatic seed transaction violated its admitted particle count "
            f"(required {required}, emitted {emitted})")
    return emitted


def inject_object_particles(scene, obj, sess, velocity=(0.0, 0.0, 0.0)):
    """Compatibility one-shot using the prepare/commit transaction seam."""
    seed = prepare_object_particles(scene, obj, velocity)
    return 0 if seed is None else commit_prepared_seeds(sess, (seed,))


def inject_auto_seeds(scene, sess):
    """Restore enabled particle seeds after solver initialize/reset."""
    return commit_prepared_seeds(sess, prepare_auto_seeds(scene))
