"""Sharp surfacing: anisotropic (ellipsoid) particle splat and splash-preserving
smoothing for the render surface.

Run on CPU with ``FLUIDSIM_TEST_DEVICE=cpu``; on a GPU host they use CUDA.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import surface as SURF
from fluidSim_warp_blender51.sim import surface_anisotropy as ANISO
from fluidSim_warp_blender51.sim.fields import (
    ScalarField,
    SparseGrid,
    voxel_coord_of_flat,
)

H = 0.02
SIZE = 0.64
FAR = 4.0 * H


# --- host maths ---------------------------------------------------------------

def test_axis_scales_keep_volume_and_cap_stretch():
    # A flat neighbourhood: wide in x and y, almost nothing in z.
    a = ANISO.ellipsoid_axis_scales([1.0, 1.0, 1.0e-6], max_stretch=4.0)
    assert np.prod(a) == pytest.approx(1.0, rel=1e-9)
    assert max(a) / min(a) == pytest.approx(4.0, rel=1e-9)
    # A disc's wide axes grow by stretch^(1/3); a needle's long axis reaches
    # the stretch^(2/3) bound the splat reach is sized from.
    assert max(a) == pytest.approx(4.0 ** (1.0 / 3.0), rel=1e-9)
    needle = ANISO.ellipsoid_axis_scales([1.0, 1.0e-6, 1.0e-6], max_stretch=4.0)
    assert max(needle) == pytest.approx(ANISO.max_axis_multiplier(4.0), rel=1e-9)
    # Bulk liquid stays round.
    assert ANISO.ellipsoid_axis_scales([2.0, 2.0, 2.0], 4.0) == pytest.approx(
        [1.0, 1.0, 1.0])
    # Max Stretch 1 means plain balls.
    assert ANISO.ellipsoid_axis_scales([3.0, 1.0, 0.1], 1.0) == pytest.approx(
        [1.0, 1.0, 1.0])


# --- device helpers -------------------------------------------------------------

@wp.kernel
def _k_sdf_sample_points(
    leaf_coords: wp.array(dtype=wp.vec3i),
    origin: wp.vec3,
    h: float,
    out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    out[tid] = origin + h * wp.vec3(float(ijk[0]) + 0.5, float(ijk[1]) + 0.5,
                                    float(ijk[2]) + 0.5)


def _grid(device, lo, hi):
    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(SIZE, SIZE, SIZE),
                      voxel_size=H, leaf_capacity=512, device=device)
    grid.begin_topology()
    grid.mark_world_box(lo, hi)
    grid.finalize_topology()
    return grid


def _reset(grid, field, device):
    wp.launch(SURF.k_surface_sdf_reset, dim=grid.voxel_capacity, device=device,
              inputs=[field.data, grid.num_leaves, FAR])


def _ball_sdf(grid, positions, count, radius, device):
    field = ScalarField(grid, name="ball")
    _reset(grid, field, device)
    splat = int(np.ceil(radius / H + 1.0))
    wp.launch(SURF.k_build_surface_sdf, dim=positions.shape[0], device=device,
              inputs=[positions, count, grid.leaf_table, grid.voxels_dim,
                      grid.leaves_dim, grid.origin, H, radius, splat, FAR],
              outputs=[field.data])
    return field


def _aniso_sdf(grid, positions, count, radius, device, *, stretch=4.0,
               flatten=0.0, neighbor_radius_voxels=3.0):
    field = ScalarField(grid, name="aniso")
    _reset(grid, field, device)
    surfacer = ANISO.AnisotropicSurfacer(device)
    surfacer.splat(grid, field, positions, count, positions.shape[0],
                   voxel_size=H, sim_voxel_size=H, particle_radius=radius,
                   neighbor_radius_voxels=neighbor_radius_voxels,
                   max_stretch=stretch, flatten=flatten, max_distance=FAR)
    return field, surfacer


def _samples(grid, field, device):
    n = int(grid.active_voxel_count)
    pts = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(_k_sdf_sample_points, dim=n, device=device,
              inputs=[grid.leaf_coords, grid.origin, H], outputs=[pts])
    return pts.numpy(), field.data.numpy()[:n]


def _sheet(device, spacing, jitter=0.15, seed=3):
    """One layer of particles in the plane z = 0.33 (a voxel-centre plane)."""
    rng = np.random.default_rng(seed)
    xs = np.arange(0.18, 0.46 + 1e-9, spacing)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.33)], axis=1)
    pts[:, :2] += rng.uniform(-jitter, jitter, (len(pts), 2)) * spacing
    positions = wp.array(pts.astype(np.float32), dtype=wp.vec3, device=device)
    count = wp.array([len(pts)], dtype=wp.int32, device=device)
    return positions, count


# --- behaviour ------------------------------------------------------------------

def test_isolated_particle_is_drawn_as_the_same_ball(cuda_device):
    dev = cuda_device
    grid = _grid(dev, (0.2, 0.2, 0.2), (0.44, 0.44, 0.44))
    positions = wp.array([[0.321, 0.317, 0.329]], dtype=wp.vec3, device=dev)
    count = wp.array([1], dtype=wp.int32, device=dev)
    radius = 0.6 * H * 2.0
    ball = _ball_sdf(grid, positions, count, radius, dev)
    aniso, _ = _aniso_sdf(grid, positions, count, radius, dev)
    b, a = ball.data.numpy(), aniso.data.numpy()
    # Same distances wherever the ball splat wrote one; the ellipsoid splat
    # may reach a few more far cells, always with the true ball distance.
    written = b < FAR - 1e-6
    assert written.sum() > 100
    np.testing.assert_allclose(a[written], b[written], atol=1e-6)
    assert np.all(a <= b + 1e-6)


def test_thin_sheet_stays_closed_with_sharp_surfacing(cuda_device):
    """The core claim: a one-particle-thick sheet beads apart as balls but
    stays a continuous sheet as ellipsoids, from the same particles."""
    dev = cuda_device
    spacing = 1.6 * H
    radius = 0.6 * H  # the default Particle Radius 1.2 x half a voxel
    positions, count = _sheet(dev, spacing)
    grid = _grid(dev, (0.12, 0.12, 0.26), (0.52, 0.52, 0.40))

    ball = _ball_sdf(grid, positions, count, radius, dev)
    aniso, _ = _aniso_sdf(grid, positions, count, radius, dev)

    pts, ball_d = _samples(grid, ball, dev)
    _, aniso_d = _samples(grid, aniso, dev)
    # Voxel centres in the sheet's plane, well inside its footprint.
    inside = ((np.abs(pts[:, 2] - 0.33) < 0.25 * H)
              & (pts[:, 0] > 0.21) & (pts[:, 0] < 0.43)
              & (pts[:, 1] > 0.21) & (pts[:, 1] < 0.43))
    assert inside.sum() > 50
    ball_cover = float((ball_d[inside] < 0.0).mean())
    aniso_cover = float((aniso_d[inside] < 0.0).mean())
    assert ball_cover < 0.85, ball_cover
    assert aniso_cover > 0.9, aniso_cover
    assert aniso_cover > ball_cover + 0.1
    # And the sheet does not swell: no thicker than about one voxel.
    off_plane = ((np.abs(pts[:, 2] - 0.33) > 0.9 * H)
                 & (pts[:, 0] > 0.21) & (pts[:, 0] < 0.43)
                 & (pts[:, 1] > 0.21) & (pts[:, 1] < 0.43))
    assert float((aniso_d[off_plane] < 0.0).mean()) < 0.02


@pytest.mark.parametrize("backend", ["sparse", "dense", "nets"])
def test_sharp_surface_meshes_on_every_backend(cuda_device, backend):
    if backend == "sparse" and not SURF.SPARSE_MARCHING_CUBES_AVAILABLE:
        pytest.skip("sparse backend needs Warp >= 1.18")
    if backend == "nets" and not getattr(SURF, "SURFACE_NETS_AVAILABLE", False):
        pytest.skip("nets backend needs Warp >= 1.19 and the nets extractor")
    dev = cuda_device
    positions, count = _sheet(dev, 1.2 * H)
    grid = _grid(dev, (0.12, 0.12, 0.26), (0.52, 0.52, 0.40))
    aniso, _ = _aniso_sdf(grid, positions, count, 0.6 * H, dev)
    verts, tris = SURF.SurfaceExtractor(grid, dev, backend=backend).extract(aniso)
    assert len(tris) > 100 and tris.max() < len(verts)
    assert np.all(np.isfinite(verts))
    # The mesh hugs the sheet: no vertex more than ~1.5 voxels off its plane.
    assert np.max(np.abs(verts[:, 2] - 0.33)) < 1.5 * H


def test_flatten_pulls_bumpy_particles_toward_their_neighbours(cuda_device):
    dev = cuda_device
    rng = np.random.default_rng(7)
    xs = np.arange(0.2, 0.44, 0.8 * H)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    z = 0.33 + rng.uniform(-0.4, 0.4, gx.size) * H
    pts = np.stack([gx.ravel(), gy.ravel(), z], axis=1).astype(np.float32)
    positions = wp.array(pts, dtype=wp.vec3, device=dev)
    count = wp.array([len(pts)], dtype=wp.int32, device=dev)
    grid = _grid(dev, (0.12, 0.12, 0.26), (0.52, 0.52, 0.40))
    _, flat0 = _aniso_sdf(grid, positions, count, 0.6 * H, dev, flatten=0.0)
    c0 = flat0.centers.numpy()[:len(pts)].copy()
    _, flat1 = _aniso_sdf(grid, positions, count, 0.6 * H, dev, flatten=0.9)
    c1 = flat1.centers.numpy()[:len(pts)]
    np.testing.assert_allclose(c0, pts, atol=1e-6)
    assert np.std(c1[:, 2]) < 0.5 * np.std(pts[:, 2])


def test_splash_weight_is_full_in_dense_liquid_and_low_in_spray(cuda_device):
    dev = cuda_device
    rng = np.random.default_rng(11)
    # A dense block (8 particles per voxel) and one lone drop far away.
    block = rng.uniform((0.20, 0.20, 0.20), (0.30, 0.30, 0.30), (1000, 3))
    drop = np.array([[0.45, 0.45, 0.45]])
    pts = np.concatenate([block, drop]).astype(np.float32)
    positions = wp.array(pts, dtype=wp.vec3, device=dev)
    count = wp.array([len(pts)], dtype=wp.int32, device=dev)
    grid = _grid(dev, (0.16, 0.16, 0.16), (0.50, 0.50, 0.50))
    surfacer = ANISO.AnisotropicSurfacer(dev)
    weight = surfacer.smoothing_weight(
        grid, grid.voxel_capacity, positions, count, len(pts), voxel_size=H,
        sim_voxel_size=H, particles_per_voxel=8.0, preserve=0.8)
    sample_pts, w = _samples(grid, ScalarField(grid), dev)
    w = weight.numpy()[:len(w)]

    def nearest(p):
        return w[np.argmin(np.linalg.norm(sample_pts - np.asarray(p), axis=1))]

    assert nearest((0.25, 0.25, 0.25)) == pytest.approx(1.0)
    assert nearest((0.45, 0.45, 0.45)) < 0.4
    # Far from any particle the weight falls to 1 - preserve.
    assert nearest((0.40, 0.21, 0.45)) == pytest.approx(0.2, abs=1e-6)


def test_render_detail_flags_follow_the_settings():
    from types import SimpleNamespace

    pipeline = object.__new__(SURF.SurfacePipeline)
    pipeline.solver = SimpleNamespace(settings=SimpleNamespace())
    assert not pipeline.render_detail_active()
    pipeline.solver.settings.surface_smooth_preserve_splashes = 0.5
    assert pipeline.render_detail_active() and not pipeline.detail_enabled()
    pipeline.solver.settings.surface_smooth_preserve_splashes = 0.0
    pipeline.solver.settings.surface_detail_enabled = True
    assert pipeline.render_detail_active() and pipeline.detail_enabled()
    # Every input is part of the render signature, so a change rebuilds.
    before = pipeline._render_detail_signature()
    pipeline.solver.settings.surface_detail_max_stretch = 2.0
    assert pipeline._render_detail_signature() != before
