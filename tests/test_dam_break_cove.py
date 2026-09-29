"""Contracts for the dam_break rocky-cove collider geometry."""

from collections import Counter

import numpy as np
import pytest

from tools import dam_break_cove as cove


def _shells(mesh):
    """Split faces into connected components (shared vertices)."""
    parent = list(range(len(mesh.verts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for face in mesh.faces:
        for v in face[1:]:
            parent[find(v)] = find(face[0])
    groups = {}
    for face in mesh.faces:
        groups.setdefault(find(face[0]), []).append(face)
    return list(groups.values())


def _signed_volume(verts, faces):
    vol = 0.0
    for face in faces:
        a = verts[face[0]]
        for k in range(1, len(face) - 1):
            vol += np.dot(a, np.cross(verts[face[k]], verts[face[k + 1]])) / 6.0
    return vol


@pytest.fixture(scope="module")
def meshes():
    return cove.collider_meshes() + [cove.apron_mesh()]


def test_every_mesh_is_closed_and_consistently_wound(meshes):
    for mesh in meshes:
        directed = Counter()
        for face in mesh.faces:
            for k, a in enumerate(face):
                directed[(a, face[(k + 1) % len(face)])] += 1
        assert max(directed.values()) == 1, f"{mesh.name}: repeated directed edge"
        for a, b in directed:
            assert directed[(b, a)] == 1, f"{mesh.name}: open edge {a}-{b}"


def test_every_shell_faces_outward(meshes):
    for mesh in meshes:
        for shell in _shells(mesh):
            assert _signed_volume(mesh.verts, shell) > 0.0, mesh.name


def test_shells_within_one_mesh_do_not_overlap(meshes):
    for mesh in meshes:
        boxes = []
        for shell in _shells(mesh):
            idx = sorted({v for face in shell for v in face})
            pts = mesh.verts[idx]
            boxes.append((pts.min(axis=0), pts.max(axis=0)))
        for i, (lo_a, hi_a) in enumerate(boxes):
            for lo_b, hi_b in boxes[i + 1:]:
                overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
                assert (overlap <= 0.0).any(), mesh.name


def test_every_solid_is_at_least_three_voxels_thick():
    for lo, hi in cove.dam_boxes() + cove.jetty_boxes():
        assert (np.subtract(hi, lo) >= cove.MIN_THICKNESS - 1e-9).all()
    xs, ys = cove.cove_grid()
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    assert cove.cove_height(gx, gy).min() - cove.BASE_Z >= cove.MIN_THICKNESS
    _zs, scale = cove.stack_rings()
    assert 2 * min(cove.STACK_AXES) * scale.min() >= cove.MIN_THICKNESS


def test_geometry_is_deterministic():
    first, second = cove.collider_meshes(), cove.collider_meshes()
    for a, b in zip(first, second):
        assert a.faces == b.faces
        assert np.array_equal(a.verts, b.verts)
        assert np.array_equal(a.face_materials, b.face_materials)


def test_apron_is_clear_and_the_cove_meets_the_floor():
    x = np.linspace(cove.DAM_X + 0.01, cove.APRON_END_X - 0.01, 60)
    y = np.linspace(-cove.HALF_WIDTH + 0.01, cove.HALF_WIDTH - 0.01, 40)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.005)], axis=1)
    assert (cove.solid_depth(pts) < 0.0).all()
    edge_y = np.linspace(-cove.HALF_WIDTH, cove.HALF_WIDTH, 51)
    assert np.allclose(cove.cove_height(np.full_like(edge_y, cove.APRON_END_X), edge_y), 0.0)


def test_headlands_leave_a_one_metre_gap():
    x = np.linspace(2.75, 3.1, 30)
    for y in (-0.49, 0.0, 0.49):
        assert np.allclose(cove.cove_height(x, np.full_like(x, y)),
                           cove.sand_height(x, np.full_like(x, y)))
    for y in (-0.7, 0.7):
        assert cove.cove_height(x, np.full_like(x, y)).min() > 0.8
    assert 2 * cove.GAP_HALF_WIDTH == pytest.approx(1.0)


def test_reservoir_starts_clear_of_every_collider():
    lo, hi, h = cove.RESERVOIR_LO, cove.RESERVOIR_HI, cove.SPACING
    axes = [np.arange(lo[i] + 0.5 * h, hi[i], h) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    assert len(grid) == 40 * 80 * 80
    assert (cove.solid_depth(grid) < 0.0).all()


def test_solid_depth_finds_points_inside_each_collider():
    inside = np.array([
        [0.3, 0.8, 0.7],                    # dam abutment
        [3.0, 0.8, 0.4],                    # headland
        [5.0, 0.0, 0.1],                    # under the beach
        [*cove.STACK_CENTER, 0.4],          # sea stack
        [4.5, -0.55, 0.71],                 # jetty deck
        [4.3, -0.47, 0.5],                  # jetty post
    ])
    outside = np.array([
        [1.5, 0.0, 0.3],                    # over the apron
        [3.0, 0.0, 0.6],                    # above the gap
        [*cove.STACK_CENTER, 1.0],          # above the stack
        [4.5, -0.55, 0.8],                  # above the deck
    ])
    assert (cove.solid_depth(inside) > 0.02).all()
    assert (cove.solid_depth(outside) < 0.0).all()


def test_cove_mesh_top_matches_the_height_function():
    mesh = cove.cove_mesh()
    top = mesh.verts[mesh.verts[:, 2] > cove.BASE_Z]
    assert np.allclose(top[:, 2], cove.cove_height(top[:, 0], top[:, 1]))
    assert set(np.unique(mesh.face_materials)) == {cove.SAND, cove.ROCK}


def test_runup_and_downstream_measures():
    xs = np.array([4.0, 4.5])
    sand = cove.sand_height(xs, np.zeros(2))
    wet = np.stack([xs, np.zeros(2), sand + 0.01], axis=1)
    assert cove.beach_runup(wet) == pytest.approx(sand.max())
    assert cove.beach_runup(np.array([[1.0, 0.0, 0.01]])) == 0.0
    pts = np.array([[1.0, 0, 0], [3.7, 0, 0], [4.0, 0, 0], [2.0, 0, 0]])
    assert cove.downstream_fraction(pts) == pytest.approx(0.5)
