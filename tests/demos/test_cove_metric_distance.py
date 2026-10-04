"""CPU-only physical-distance and near-threshold cove regressions."""
import math

import numpy as np
import pytest

from tools.demos import dam_break_cove as cove


def test_stack_penetration_rejects_known_45mm_counterexample():
    point = np.array([[4.024827122500947, .13, .35]])
    # Independent distances to the canonical fan and alternate-diagonal surfaces.
    assert cove.solid_depth(point)[0] == pytest.approx(.04515793584036025, abs=1e-10)
    assert cove.solid_depth(point)[0] > 1.5*cove.VOXEL

@pytest.fixture
def regular_stack(monkeypatch):
    monkeypatch.setattr(cove, "stack_rings",
        lambda: (np.array([-.1,.82]), np.ones((2,cove.STACK_SIDES))))
    if hasattr(cove, "_stack_triangles"):
        cove._stack_triangles.cache_clear()
    yield
    if hasattr(cove, "_stack_triangles"):
        cove._stack_triangles.cache_clear()

@pytest.mark.parametrize("inward", [.04,.05])
def test_regular_elliptical_stack_matches_exact_face_normal_distance(regular_stack, inward):
    theta = 2*math.pi/cove.STACK_SIDES
    edge = np.array([.225*(math.cos(theta)-1), .20*math.sin(theta)])
    expected = inward*abs(edge[1])/np.linalg.norm(edge)
    point = np.array([[cove.STACK_CENTER[0]+.225-inward, cove.STACK_CENTER[1], .4]])
    actual = cove._stack_depth(point)[0]
    assert actual == pytest.approx(expected, abs=1e-10)
    assert (actual <= 1.5*cove.VOXEL) == (inward == .04)

def test_stack_caps_and_outside_sign(regular_stack):
    cx,cy = cove.STACK_CENTER
    points = np.array([[cx,cy,.80],[cx,cy,-.08],[cx,cy,.85],
                       [cx+.3,cy,.4],[cx,cy,.4]])
    depth = cove._stack_depth(points)
    assert depth[:2] == pytest.approx([.02,.02],abs=1e-10)
    assert (depth[2:4] < 0).all()
    assert depth[4] > 1.5*cove.VOXEL


def test_stack_empty_boundary_and_nonfinite_inputs(regular_stack):
    assert cove.solid_depth(np.empty((0, 3))).shape == (0,)
    cx, cy = cove.STACK_CENTER
    boundary = np.array([[cx + .225, cy, .4], [cx, cy, -.1]])
    assert cove._stack_depth(boundary) == pytest.approx([0., 0.], abs=1e-10)
    with pytest.raises(ValueError, match="Non-finite"):
        cove._stack_depth(np.array([[np.nan, cy, .4]]))


def test_signed_triangle_distance_handles_edges_vertices_and_outside():
    # Outward-oriented tetrahedron: exact nearest plane and endpoint controls.
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    triangles = vertices[[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]
    points = np.array([[.1, .1, .1], [2., 0., 0.], [.5, -.2, 0.], [0., 0., 0.]])
    assert cove._triangle_surface_depth(points, triangles) == pytest.approx(
        [.1, -1., -.2, 0.], abs=1e-10)


@pytest.mark.parametrize("verts, expected_indices", [
    ([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)],
     [[0, 1, 2], [0, 2, 3]]),
    ([(0., 0., 0.), (.25, .25, 0.), (1., 0., 0.), (0., 1., 0.)],
     [[0, 1, 3], [1, 2, 3]]),
])
def test_stack_quad_tessellation_normal_and_concave_flip(monkeypatch, verts, expected_indices):
    points = np.array(verts)
    mesh = cove.MeshData("quad", points, ((0, 1, 2, 3),), np.array([0]), (), 0.)
    monkeypatch.setattr(cove, "stack_mesh", lambda: mesh)
    cove._stack_triangles.cache_clear()
    try:
        assert np.array_equal(cove._stack_triangles(), points[expected_indices])
    finally:
        cove._stack_triangles.cache_clear()
