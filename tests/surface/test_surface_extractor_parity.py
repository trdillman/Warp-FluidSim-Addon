"""4.0 lane W: the sparse-leaf marching-cubes backend matches the dense one.

Warp 1.18 deprecates ``wp.MarchingCubes`` and ships
``warp.geometry.sparse_marching_cubes_from_cells``.  ``SurfaceExtractor``
picks the sparse backend when that exists and keeps the dense sub-grid path
for Warp 1.17.  Both must produce the same mesh for the same SDF, otherwise a
user's surface changes with the Warp version they happen to have installed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import surface as SURF  # noqa: E402
from fluidSim_warp_blender51.sim.fields import (  # noqa: E402
    ScalarField,
    SparseGrid,
    voxel_center_world,
    voxel_coord_of_flat,
)

needs_nets = pytest.mark.skipif(
    not SURF.SURFACE_NETS_AVAILABLE,
    reason="warp.geometry.IsoSurfaceNets needs Warp >= 1.19",
)

needs_sparse = pytest.mark.skipif(
    not SURF.SPARSE_MARCHING_CUBES_AVAILABLE,
    reason="warp.geometry.sparse_marching_cubes_from_cells needs Warp >= 1.18",
)

VOXEL = 0.02


@wp.kernel
def _k_sphere_sdf(sdf: wp.array(dtype=float),
                  leaf_coords: wp.array(dtype=wp.vec3i),
                  origin: wp.vec3, voxel_size: float,
                  centre: wp.vec3, radius: float):
    tid = wp.tid()
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    p = voxel_center_world(ijk, origin, voxel_size)
    sdf[tid] = wp.length(p - centre) - radius


def _sphere_field(device, centre, radius, box_lo, box_hi):
    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(0.64, 0.64, 0.64),
                      voxel_size=VOXEL, leaf_capacity=512, device=device)
    grid.begin_topology()
    grid.mark_world_box(box_lo, box_hi)
    grid.finalize_topology()
    field = ScalarField(grid, name="sdf")
    wp.launch(_k_sphere_sdf, dim=grid.voxel_capacity, device=device,
              inputs=[field.data, grid.leaf_coords, grid.origin, VOXEL,
                      wp.vec3(*centre), float(radius)])
    return grid, field


CASES = {
    # Fully interior sphere: the closed-surface case.
    "interior": ((0.32, 0.32, 0.32), 0.15, (0.12, 0.12, 0.12),
                 (0.52, 0.52, 0.52)),
    # Sphere cut by the domain corner: the dense path clips its box here and
    # the sparse path must drop the same out-of-domain cells.
    "domain_corner": ((0.0, 0.0, 0.0), 0.2, (0.0, 0.0, 0.0),
                      (0.3, 0.3, 0.3)),
    # Liquid that reaches the edge of the active leaves: the surface meets
    # the background value there, as it does when the band is clipped.
    "clipped_band": ((0.32, 0.32, 0.32), 0.2, (0.2, 0.2, 0.2),
                     (0.45, 0.45, 0.45)),
}


def _oriented_triangles_in(a_verts, a_tris, b_verts):
    """A's triangles renumbered into B's vertex ids, rotated to a canonical
    start so winding is kept: an emission-order-free, orientation-aware key.

    Every A vertex must coincide with a B vertex to within 1e-5.
    """
    b_verts = np.asarray(b_verts, np.float64)
    remap = np.empty(len(a_verts), dtype=np.int64)
    for i, v in enumerate(np.asarray(a_verts, np.float64)):
        d = np.abs(b_verts - v).max(axis=1)
        j = int(d.argmin())
        assert d[j] < 1e-5, (i, v, b_verts[j])
        remap[i] = j
    tris = remap[np.asarray(a_tris)]
    return {tuple(np.roll(t, -int(np.argmin(t)))) for t in tris.tolist()}


@needs_sparse
@pytest.mark.parametrize("case", sorted(CASES))
def test_sparse_and_dense_backends_produce_the_same_mesh(cuda_device, case):
    centre, radius, lo, hi = CASES[case]
    grid, field = _sphere_field(cuda_device, centre, radius, lo, hi)

    dense = SURF.SurfaceExtractor(grid, cuda_device, backend="dense")
    sparse = SURF.SurfaceExtractor(grid, cuda_device, backend="sparse")
    dv, dt = dense.extract(field)
    sv, st = sparse.extract(field)

    assert len(dt) > 0
    assert sv.dtype == np.float32 and st.dtype == np.int32
    assert sv.shape[1] == 3 and st.shape[1] == 3
    assert len(sv) == len(dv)
    assert len(st) == len(dt)
    np.testing.assert_allclose(sv.min(axis=0), dv.min(axis=0), atol=1e-5)
    np.testing.assert_allclose(sv.max(axis=0), dv.max(axis=0), atol=1e-5)
    np.testing.assert_allclose(np.sort(sv, axis=0), np.sort(dv, axis=0),
                               atol=1e-5)
    # Same triangles, independent of emission order.
    dense_tris = {tuple(np.roll(t, -int(np.argmin(t)))) for t in dt.tolist()}
    assert _oriented_triangles_in(sv, st, dv) == dense_tris


@needs_sparse
def test_sparse_backend_mesh_is_watertight_on_a_closed_sphere(cuda_device):
    centre, radius, lo, hi = CASES["interior"]
    grid, field = _sphere_field(cuda_device, centre, radius, lo, hi)
    verts, tris = SURF.SurfaceExtractor(
        grid, cuda_device, backend="sparse").extract(field)
    edges = np.sort(np.concatenate(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    # Every vertex lies on the sphere to within marching-cubes error.
    dist = np.linalg.norm(verts - np.asarray(centre, np.float32), axis=1)
    assert np.max(np.abs(dist - radius)) < 0.25 * VOXEL


def test_auto_backend_follows_the_installed_warp(cuda_device):
    grid, _field = _sphere_field(cuda_device, *CASES["interior"])
    ext = SURF.SurfaceExtractor(grid, cuda_device)
    expected = "sparse" if SURF.SPARSE_MARCHING_CUBES_AVAILABLE else "dense"
    assert ext.backend == expected


def test_empty_grid_returns_an_empty_mesh_on_both_backends(cuda_device):
    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(0.32, 0.32, 0.32),
                      voxel_size=VOXEL, leaf_capacity=8, device=cuda_device)
    grid.begin_topology()
    grid.finalize_topology()
    field = ScalarField(grid, name="sdf")
    backends = ["dense"] + (["sparse"] if SURF.SPARSE_MARCHING_CUBES_AVAILABLE
                            else []) + (["nets"] if SURF.SURFACE_NETS_AVAILABLE
                                        else [])
    for backend in backends:
        v, t = SURF.SurfaceExtractor(grid, cuda_device,
                                     backend=backend).extract(field)
        assert v.shape == (0, 3) and t.shape == (0, 3)


# --- Surface Nets backend (Warp >= 1.19, opt-in) -----------------------------

def _triangle_quality(verts, tris):
    """Per-triangle quality 4*sqrt(3)*area / sum(edge^2): 1 equilateral, 0 flat."""
    v = np.asarray(verts, np.float64)[np.asarray(tris)]
    e0, e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(e0, -e2), axis=1)
    denom = (e0 ** 2).sum(1) + (e1 ** 2).sum(1) + (e2 ** 2).sum(1)
    return 4.0 * np.sqrt(3.0) * area / np.maximum(denom, 1e-30)


@needs_nets
def test_nets_backend_mesh_is_closed_outward_and_on_the_sphere(cuda_device):
    centre, radius, lo, hi = CASES["interior"]
    grid, field = _sphere_field(cuda_device, centre, radius, lo, hi)
    verts, tris = SURF.SurfaceExtractor(
        grid, cuda_device, backend="nets").extract(field)
    assert verts.dtype == np.float32 and tris.dtype == np.int32
    assert len(tris) > 0 and tris.max() < len(verts)
    edges = np.sort(np.concatenate(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    # Vertices sit on the sphere to within a fraction of a voxel.
    c = np.asarray(centre, np.float32)
    dist = np.linalg.norm(verts - c, axis=1)
    assert np.max(np.abs(dist - radius)) < 0.5 * VOXEL
    # Faces wind outward (normal points away from the centre), so the mesh
    # keeps the orientation the marching-cubes backends give the renderer.
    v = verts.astype(np.float64)[tris]
    normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    outward = np.einsum("ij,ij->i", normal, v.mean(axis=1) - c)
    assert np.all(outward > 0.0)
    # Signed volume agrees with the marching-cubes mesh of the same field.
    def volume(vs, ts):
        p = vs.astype(np.float64)[ts]
        return float(np.einsum("ij,ij->i", p[:, 0],
                               np.cross(p[:, 1], p[:, 2])).sum() / 6.0)
    dv, dt = SURF.SurfaceExtractor(
        grid, cuda_device, backend="dense").extract(field)
    assert volume(verts, tris) == pytest.approx(volume(dv, dt), rel=0.02)


@needs_nets
def test_nets_triangles_are_more_even_than_marching_cubes(cuda_device):
    grid, field = _sphere_field(cuda_device, *CASES["interior"])
    nv, nt = SURF.SurfaceExtractor(
        grid, cuda_device, backend="nets").extract(field)
    dv, dt = SURF.SurfaceExtractor(
        grid, cuda_device, backend="dense").extract(field)
    nq, dq = _triangle_quality(nv, nt), _triangle_quality(dv, dt)
    assert nq.mean() > dq.mean()
    assert (nq < 0.1).mean() < (dq < 0.1).mean() + 1e-12


@needs_nets
@pytest.mark.parametrize("case", ["domain_corner", "clipped_band"])
def test_nets_backend_handles_clipped_surfaces(cuda_device, case):
    grid, field = _sphere_field(cuda_device, *CASES[case])
    verts, tris = SURF.SurfaceExtractor(
        grid, cuda_device, backend="nets").extract(field)
    assert len(tris) > 0 and tris.max() < len(verts)
    assert np.all(np.isfinite(verts))


@needs_nets
def test_env_override_selects_nets_for_auto(cuda_device, monkeypatch):
    grid, _field = _sphere_field(cuda_device, *CASES["interior"])
    monkeypatch.setenv(SURF.SURFACE_BACKEND_ENV, "nets")
    assert SURF.SurfaceExtractor(grid, cuda_device).backend == "nets"
    monkeypatch.setenv(SURF.SURFACE_BACKEND_ENV, "nonsense")
    expected = "sparse" if SURF.SPARSE_MARCHING_CUBES_AVAILABLE else "dense"
    assert SURF.SurfaceExtractor(grid, cuda_device).backend == expected
    monkeypatch.delenv(SURF.SURFACE_BACKEND_ENV)
    assert SURF.SurfaceExtractor(grid, cuda_device).backend == expected


def test_nets_backend_errors_clearly_without_warp_1_19(cuda_device, monkeypatch):
    grid, _field = _sphere_field(cuda_device, *CASES["interior"])
    monkeypatch.setattr(SURF, "SURFACE_NETS_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="Warp >= 1.19"):
        SURF.SurfaceExtractor(grid, cuda_device, backend="nets")
