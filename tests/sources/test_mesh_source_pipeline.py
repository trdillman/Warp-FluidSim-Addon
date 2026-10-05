"""GPU behavior checks for evaluated-triangle sources and sinks."""

from __future__ import annotations

import numpy as np
import pytest

from fluidSim_warp_blender51.sim import constants as C
from fluidSim_warp_blender51.sim import sourcing
from fluidSim_warp_blender51.sim.kernels_fluid import k_particle_in_sink_check
from fluidSim_warp_blender51.sim.surface import k_sdf_to_axis_collision
from fluidSim_warp_blender51.sim.sourcing import MeshSourceDesc


def _mesh(points, triangles, device):
    points_dev = sourcing.wp.array(
        np.asarray(points, dtype=np.float32),
        dtype=sourcing.wp.vec3,
        device=device,
    )
    triangles_dev = sourcing.wp.array(
        np.asarray(triangles, dtype=np.int32).reshape(-1),
        dtype=sourcing.wp.int32,
        device=device,
    )
    return sourcing.wp.Mesh(
        points=points_dev,
        indices=triangles_dev,
        support_winding_number=True,
    )


def _triangle_geometry():
    return (
        [(0.0, 0.0, 0.5), (2.0, 0.0, 0.5), (0.0, 2.0, 0.5)],
        [(0, 1, 2)],
    )


def _cube_geometry():
    points = [
        (0.5, 0.5, 0.5), (1.5, 0.5, 0.5),
        (1.5, 1.5, 0.5), (0.5, 1.5, 0.5),
        (0.5, 0.5, 1.5), (1.5, 0.5, 1.5),
        (1.5, 1.5, 1.5), (0.5, 1.5, 1.5),
    ]
    triangles = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return points, triangles


def _grid_arrays(device):
    leaves = sourcing.wp.array(
        np.asarray([(0, 0, 0)], dtype=np.int32),
        dtype=sourcing.wp.vec3i,
        device=device,
    )
    count = sourcing.wp.array(
        np.asarray([1], dtype=np.int32),
        dtype=sourcing.wp.int32,
        device=device,
    )
    return leaves, count


def _point_triangle_distance(point, a, b, c):
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


def _particle_outputs(capacity, device):
    return {
        "position": sourcing.wp.empty(capacity, dtype=sourcing.wp.vec3, device=device),
        "velocity": sourcing.wp.empty(capacity, dtype=sourcing.wp.vec3, device=device),
        "age": sourcing.wp.empty(capacity, dtype=float, device=device),
        "life": sourcing.wp.empty(capacity, dtype=float, device=device),
        "temperature": sourcing.wp.empty(capacity, dtype=float, device=device),
        "viscosity": sourcing.wp.empty(capacity, dtype=float, device=device),
        "density": sourcing.wp.empty(capacity, dtype=float, device=device),
        "color": sourcing.wp.empty(capacity, dtype=sourcing.wp.vec4, device=device),
        "source_id": sourcing.wp.empty(
            capacity, dtype=sourcing.wp.int32, device=device),
        "time_offset": sourcing.wp.empty(capacity, dtype=float, device=device),
        "alive": sourcing.wp.zeros(capacity, dtype=sourcing.wp.int32, device=device),
        "count": sourcing.wp.zeros(1, dtype=sourcing.wp.int32, device=device),
        "emission_state": sourcing.wp.array(
            [0, 1, 0, 0], dtype=sourcing.wp.int32, device=device),
    }


@pytest.mark.parametrize("geometry", (_triangle_geometry(), _cube_geometry()),
                         ids=("open", "closed"))
def test_mesh_particles_are_projected_to_actual_triangles_with_attributes(geometry):
    device = sourcing.wp.get_preferred_device()
    points, triangles = geometry
    mesh = _mesh(points, triangles, device)
    leaves, num_leaves = _grid_arrays(device)
    outputs = _particle_outputs(1024, device)
    desc = MeshSourceDesc(
        mesh=mesh,
        aabb_min=tuple(np.min(points, axis=0)),
        aabb_max=tuple(np.max(points, axis=0)),
        mode=1,
        stype=0,
        emit_rate=1.0,
        velocity=(1.0, 2.0, 3.0),
        temperature=4.0,
        viscosity=5.0,
        density=6.0,
        use_lifespan=True,
        life_min=2.0,
        life_max=7.0,
    )
    sourcing.wp.launch(
        sourcing.k_mesh_source_particles,
        dim=C.LEAF_VOLUME,
        inputs=[
            mesh.id, desc.to_struct(), desc.query_radius(), leaves,
            num_leaves, sourcing.wp.vec3(0.0, 0.0, 0.0), 0.25,
            sourcing.wp.zeros(C.LEAF_VOLUME, dtype=sourcing.wp.int32, device=device),
            1234, 0, 0, 1024,
        ],
        outputs=list(outputs.values()),
        device=device,
    )
    count = int(outputs["count"].numpy()[0])
    assert count > 0
    positions = outputs["position"].numpy()[:count]
    diagonal = np.linalg.norm(
        np.max(points, axis=0) - np.min(points, axis=0))
    tolerance = 1.0e-5 * max(1.0, diagonal)
    distances = [
        min(_point_triangle_distance(point, points[a], points[b], points[c])
            for a, b, c in triangles)
        for point in positions
    ]
    assert max(distances) <= tolerance
    np.testing.assert_allclose(
        outputs["velocity"].numpy()[:count],
        np.tile((1.0, 2.0, 3.0), (count, 1)))
    np.testing.assert_allclose(outputs["temperature"].numpy()[:count], 4.0)
    np.testing.assert_allclose(outputs["viscosity"].numpy()[:count], 5.0)
    np.testing.assert_allclose(outputs["density"].numpy()[:count], 6.0)
    assert ((outputs["life"].numpy()[:count] >= 2.0)
            & (outputs["life"].numpy()[:count] <= 7.0)).all()


def test_mesh_source_force_velocity_reaches_emitted_particles():
    device = sourcing.wp.get_preferred_device()
    points, triangles = _triangle_geometry()
    mesh = _mesh(points, triangles, device)
    leaves, num_leaves = _grid_arrays(device)
    outputs = _particle_outputs(1024, device)
    desc = MeshSourceDesc(
        mesh=mesh,
        aabb_min=tuple(np.min(points, axis=0)),
        aabb_max=tuple(np.max(points, axis=0)),
        mode=1,
        stype=0,
        emit_rate=1.0,
        velocity=(1.0, 2.0, 3.0),
        force_type=1,
        force_strength=4.0,
        force_dir=(0.0, 1.0, 0.0),
        force_center=(0.0, 0.0, 0.0),
    )
    sourcing.wp.launch(
        sourcing.k_mesh_source_particles,
        dim=C.LEAF_VOLUME,
        inputs=[
            mesh.id, desc.to_struct(), desc.query_radius(), leaves,
            num_leaves, sourcing.wp.vec3(0.0, 0.0, 0.0), 0.25,
            sourcing.wp.zeros(C.LEAF_VOLUME, dtype=sourcing.wp.int32,
                             device=device),
            9876, 0, 0, 1024,
        ],
        outputs=list(outputs.values()),
        device=device,
    )
    count = int(outputs["count"].numpy()[0])

    assert count > 0
    np.testing.assert_allclose(
        outputs["velocity"].numpy()[:count],
        np.tile((1.0, 6.0, 3.0), (count, 1)),
    )


def test_mesh_particle_emission_respects_existing_voxel_top_up_counts():
    device = sourcing.wp.get_preferred_device()
    points, triangles = _triangle_geometry()
    mesh = _mesh(points, triangles, device)
    leaves, num_leaves = _grid_arrays(device)
    outputs = _particle_outputs(64, device)
    desc = MeshSourceDesc(
        mesh=mesh, aabb_min=(0.0, 0.0, 0.5), aabb_max=(2.0, 2.0, 0.5),
        stype=0, mode=1, emit_rate=1.0)
    existing_counts = sourcing.wp.array(
        np.ones(C.LEAF_VOLUME, dtype=np.int32),
        dtype=sourcing.wp.int32,
        device=device)
    sourcing.wp.launch(
        sourcing.k_mesh_source_particles,
        dim=C.LEAF_VOLUME,
        inputs=[
            mesh.id, desc.to_struct(), desc.query_radius(), leaves,
            num_leaves, sourcing.wp.vec3(0.0, 0.0, 0.0), 0.25,
            existing_counts, 5678, 0, 0, 64,
        ],
        outputs=list(outputs.values()),
        device=device)
    assert int(outputs["count"].numpy()[0]) == 0


def test_subvoxel_mesh_particle_source_still_reaches_top_up_target():
    device = sourcing.wp.get_preferred_device()
    points = [
        (0.495, 0.495, 0.5),
        (0.505, 0.495, 0.5),
        (0.495, 0.505, 0.5),
    ]
    triangles = [(0, 1, 2)]
    mesh = _mesh(points, triangles, device)
    leaves, num_leaves = _grid_arrays(device)
    outputs = _particle_outputs(32, device)
    desc = MeshSourceDesc(
        mesh=mesh,
        aabb_min=tuple(np.min(points, axis=0)),
        aabb_max=tuple(np.max(points, axis=0)),
        stype=0,
        mode=1,
        emit_rate=8.0,
    )
    sourcing.wp.launch(
        sourcing.k_mesh_source_particles,
        dim=C.LEAF_VOLUME,
        inputs=[
            mesh.id, desc.to_struct(), desc.query_radius(), leaves,
            num_leaves, sourcing.wp.vec3(0.0, 0.0, 0.0), 1.0,
            sourcing.wp.zeros(C.LEAF_VOLUME, dtype=sourcing.wp.int32,
                              device=device),
            2468, 0, 0, 32,
        ],
        outputs=list(outputs.values()),
        device=device)
    count = int(outputs["count"].numpy()[0])
    assert count == 8
    diagonal = np.linalg.norm(
        np.max(points, axis=0) - np.min(points, axis=0))
    tolerance = 1.0e-5 * max(1.0, diagonal)
    distances = [
        _point_triangle_distance(point, *points)
        for point in outputs["position"].numpy()[:count]
    ]
    assert max(distances) <= tolerance


def _launch_mesh_field(desc, device, voxel_size=0.25):
    leaves, num_leaves = _grid_arrays(device)
    arrays = {
        "voxel_type": sourcing.wp.zeros(
            C.LEAF_VOLUME, dtype=sourcing.wp.int32, device=device),
        "u": sourcing.wp.zeros(C.LEAF_VOLUME, dtype=float, device=device),
        "v": sourcing.wp.zeros(C.LEAF_VOLUME, dtype=float, device=device),
        "w": sourcing.wp.zeros(C.LEAF_VOLUME, dtype=float, device=device),
        "temperature": sourcing.wp.zeros(
            C.LEAF_VOLUME, dtype=float, device=device),
        "viscosity": sourcing.wp.zeros(
            C.LEAF_VOLUME, dtype=float, device=device),
    }
    sourcing.wp.launch(
        sourcing.k_mesh_source_field,
        dim=C.LEAF_VOLUME,
        inputs=[
            desc.mesh.id, desc.to_struct(), desc.query_radius(), leaves,
            num_leaves, sourcing.wp.vec3(0.0, 0.0, 0.0), voxel_size,
        ],
        outputs=list(arrays.values()),
        device=device,
    )
    return {name: value.numpy() for name, value in arrays.items()}


def test_mesh_field_compatibility_mode_does_not_create_liquid_or_mutate_fields():
    """Mesh FIELD was a port-only proxy with no fluidSim.cl counterpart.

    Keep the ABI value loadable for saved files, but never manufacture liquid
    or write unweighted MAC/attributes from the sourcing phase.
    """
    device = sourcing.wp.get_preferred_device()
    points, triangles = _triangle_geometry()
    desc = MeshSourceDesc(
        mesh=_mesh(points, triangles, device),
        aabb_min=(0.0, 0.0, 0.5),
        aabb_max=(2.0, 2.0, 0.5),
        stype=0,
        mode=0,
        velocity=(1.0, 2.0, 3.0),
        force_type=1,
        force_strength=4.0,
        force_dir=(0.0, 1.0, 0.0),
        temperature=4.0,
        viscosity=5.0,
    )
    result = _launch_mesh_field(desc, device)
    assert np.count_nonzero(result["voxel_type"]) == 0
    for name in ("u", "v", "w", "temperature", "viscosity"):
        np.testing.assert_allclose(result[name], 0.0)


def test_mesh_sink_solid_and_surface_regions_are_distinct():
    device = sourcing.wp.get_preferred_device()
    points, triangles = _cube_geometry()
    mesh = _mesh(points, triangles, device)
    center = 3 + 3 * 8 + 3 * 64
    boundary = 2 + 3 * 8 + 3 * 64
    outside = 0

    surface = MeshSourceDesc(
        mesh=mesh, aabb_min=(0.5, 0.5, 0.5), aabb_max=(1.5, 1.5, 1.5),
        stype=2, sink_region_mode=1)
    surface_result = _launch_mesh_field(surface, device)["voxel_type"]
    assert not surface_result[center] & C.VOXEL_SINK
    assert surface_result[boundary] & C.VOXEL_SINK
    assert not surface_result[outside] & C.VOXEL_SINK

    solid = MeshSourceDesc(
        mesh=mesh, aabb_min=(0.5, 0.5, 0.5), aabb_max=(1.5, 1.5, 1.5),
        stype=2, sink_region_mode=0)
    solid_result = _launch_mesh_field(solid, device)["voxel_type"]
    assert solid_result[center] & C.VOXEL_SINK
    assert not solid_result[outside] & C.VOXEL_SINK

    positions = sourcing.wp.array(
        np.asarray([(0.875, 0.875, 0.875), (0.125, 0.125, 0.125)],
                   dtype=np.float32),
        dtype=sourcing.wp.vec3,
        device=device)
    alive = sourcing.wp.array(
        np.asarray([1, 1], dtype=np.int32),
        dtype=sourcing.wp.int32,
        device=device)
    sourcing.wp.launch(
        k_particle_in_sink_check,
        dim=2,
        inputs=[
            positions,
            sourcing.wp.array(np.asarray([2], dtype=np.int32),
                              dtype=sourcing.wp.int32, device=device),
            sourcing.wp.array(solid_result, dtype=sourcing.wp.int32, device=device),
            sourcing.wp.array(np.asarray([0], dtype=np.int32),
                              dtype=sourcing.wp.int32, device=device),
            sourcing.wp.vec3i(8, 8, 8), sourcing.wp.vec3i(1, 1, 1),
            sourcing.wp.vec3(0.0, 0.0, 0.0), 0.25,
        ],
        outputs=[alive],
        device=device)
    np.testing.assert_array_equal(alive.numpy(), (0, 1))


def test_late_collider_classification_clears_overlapping_sink_and_fluid_bits():
    device = sourcing.wp.get_preferred_device()
    leaves, num_leaves = _grid_arrays(device)
    sdf = np.ones(C.LEAF_VOLUME, dtype=np.float32)
    sdf[0] = -1.0
    voxel_type = np.zeros(C.LEAF_VOLUME, dtype=np.int32)
    voxel_type[0] = (
        C.VOXEL_FLUID | C.VOXEL_SINK | C.VOXEL_SOURCE_REGION_TRACKING)
    voxel_type_dev = sourcing.wp.array(
        voxel_type, dtype=sourcing.wp.int32, device=device)
    sourcing.wp.launch(
        k_sdf_to_axis_collision,
        dim=C.LEAF_VOLUME,
        inputs=[
            sourcing.wp.array(sdf, dtype=float, device=device),
            sourcing.wp.array(np.asarray([0], dtype=np.int32),
                              dtype=sourcing.wp.int32, device=device),
            leaves, sourcing.wp.vec3i(8, 8, 8), sourcing.wp.vec3i(1, 1, 1),
            num_leaves, 0.0,
        ],
        outputs=[voxel_type_dev],
        device=device)
    resolved = int(voxel_type_dev.numpy()[0])
    assert resolved & C.VOXEL_COLLISION
    assert not resolved & C.VOXEL_FLUID
    assert not resolved & C.VOXEL_SINK
    assert not resolved & C.VOXEL_SOURCE_REGION_TRACKING
