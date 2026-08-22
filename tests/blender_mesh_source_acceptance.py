"""Blender/Warp acceptance for modifier-evaluated mesh surface sourcing."""

from __future__ import annotations

import os
import sys

import bpy
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon


FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


def point_triangle_distance(point, a, b, c):
    point = np.asarray(point, dtype=np.float64)
    a, b, c = (np.asarray(value, dtype=np.float64) for value in (a, b, c))
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return np.linalg.norm(ap)
    bp = point - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return np.linalg.norm(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return np.linalg.norm(point - (a + ab * (d1 / (d1 - d3))))
    cp = point - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return np.linalg.norm(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return np.linalg.norm(point - (a + ac * (d2 / (d2 - d6))))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 >= d3 and d5 >= d6:
        weight = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.linalg.norm(point - (b + weight * (c - b)))
    normal = np.cross(ab, ac)
    return abs(np.dot(point - a, normal)) / np.linalg.norm(normal)


def evaluated_triangles(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        local = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float32)
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float32)
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        triangles = np.asarray(
            [triangle.vertices[:] for triangle in mesh.loop_triangles],
            dtype=np.int32,
        )
        return world, triangles
    finally:
        evaluated.to_mesh_clear()


def main():
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.voxel_size = 0.12
    fs.max_particles = 100_000
    fs.whitewater_enabled = False
    fs.liquid_view_mode = "PARTICLES"
    bpy.ops.fluidsim.setup_scene()

    default_source = next(
        (
            obj for obj in scene.objects
            if getattr(getattr(obj, "fluidsim", None), "role", "NONE")
            == "SOURCE"
        ),
        None,
    )
    if default_source is not None:
        default_source.fluidsim.role = "NONE"

    mesh = bpy.data.meshes.new("AsymmetricEvaluatedSourceMesh")
    vertices = [
        (-0.85, -0.45, 0.65), (0.95, -0.25, 0.72),
        (0.35, 0.82, 0.96), (-0.55, 0.38, 0.78),
    ]
    faces = [(0, 1, 2), (0, 2, 3)]
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    source = bpy.data.objects.new("Asymmetric Modifier Mesh Source", mesh)
    scene.collection.objects.link(source)
    source.scale = (1.15, 0.8, 1.25)
    solidify = source.modifiers.new("Acceptance Solidify", "SOLIDIFY")
    solidify.thickness = 0.16
    solidify.offset = 0.35
    source.fluidsim.role = "SOURCE"
    source.fluidsim.geometry_mode = "MESH"
    source.fluidsim.mesh_evaluation_mode = "EVERY_SIM_FRAME"
    source.fluidsim.mesh_surface_thickness = 0.0
    source.fluidsim.source_mode = "PARTICLES"
    source.fluidsim.emit_rate = 1.0
    source.fluidsim.initial_velocity = (0.0, 0.0, 0.0)
    source.fluidsim.use_object_velocity = False

    depsgraph = bpy.context.evaluated_depsgraph_get()
    world_vertices, triangles = evaluated_triangles(source, depsgraph)
    check("modifier changes source topology", len(triangles) > len(faces),
          f"base={len(faces)} evaluated={len(triangles)}")
    check("acceptance mesh is asymmetric",
          not np.allclose(np.abs(world_vertices.min(axis=0)),
                          np.abs(world_vertices.max(axis=0)), atol=1.0e-4))

    result = bpy.ops.fluidsim.initialize()
    check("initialize", result == {"FINISHED"}, fs.status)
    from fluidSim_warp_blender51.blender.scene_bridge import session

    sess = session()
    check("CUDA session", sess.active and str(sess.solver.device).startswith("cuda"),
          str(sess.solver.device if sess.active else "inactive"))
    check("mesh descriptor accepted",
          len(sess.solver.sourcing.mesh_sources) == 1,
          source.fluidsim.mesh_error)
    check("analytic ABI remains separate",
          len(sess.solver.sourcing.analytic_sources) == 0)

    solver = sess.solver
    solver._rebuild_topology()
    solver._clear_grid_fields()
    solver._index_particles()
    emitted = solver.sourcing.apply(1.0 / scene.render.fps)
    count = solver.particles.refresh_host_count()
    check("GPU mesh source emitted", emitted and count > 0, f"particles={count}")
    if count:
        positions = solver.particles.position.numpy()[:count]
        distances = [
            min(point_triangle_distance(point, world_vertices[a], world_vertices[b],
                                        world_vertices[c])
                for a, b, c in triangles)
            for point in positions
        ]
        diagonal = np.linalg.norm(
            world_vertices.max(axis=0) - world_vertices.min(axis=0))
        tolerance = 1.0e-5 * max(1.0, diagonal)
        check("particles lie on evaluated triangles",
              max(distances) <= tolerance,
              f"max_distance={max(distances):.8g} tolerance={tolerance:.8g}")

    previous_frame = solver.frame
    stats = sess.step(scene, write_cache=False)
    check("live solver frame advances", stats is not None and solver.frame > previous_frame,
          f"frame={solver.frame}")

    source.fluidsim.role = "SINK"
    source.fluidsim.sink_region_mode = "SOLID"
    source.modifiers.remove(solidify)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    sess.sync_scene(scene, depsgraph)
    check("open solid sink rejected without fallback",
          len(sess.solver.sourcing.mesh_sources) == 0
          and "watertight" in source.fluidsim.mesh_error.lower(),
          source.fluidsim.mesh_error)

    if FAILURES:
        print("MESH-SOURCE-ACCEPTANCE-FAIL " + ", ".join(FAILURES))
        raise SystemExit(1)
    print("MESH-SOURCE-ACCEPTANCE-PASS")


if __name__ == "__main__":
    main()
