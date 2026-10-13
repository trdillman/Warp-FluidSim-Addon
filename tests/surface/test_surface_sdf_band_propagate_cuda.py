"""Error bounds for the closest-point band extension of the particle SDF.

``SDF_BAND_MODE_PROPAGATE`` seeds a small splat, records the winning particle
per seeded voxel, and hands those closest points outward one cell at a time
instead of splatting the full capillary support radius.  Unlike the splat
cull, this is *not* output-identical to the full splat, so the claims it
does make have to be pinned:

1. **Never an underestimate.** Every value the propagation writes is
   ``|c - p| - r`` recomputed from the receiving voxel's own centre to a real
   particle, so it is achievable by construction and can only sit at or above
   the true minimum over all particles.  It never interpolates, so it cannot
   manufacture a distance no particle supports.

2. **Exact inside a provable window.** A voxel is seeded by its *true* nearest
   particle whenever that particle lies within Chebyshev radius ``seed`` cells
   of it.  Chebyshev distance never exceeds Euclidean, so every voxel whose
   exact value is at most ``seed * h - r`` was seeded by the minimiser and
   carries the exact value.  ``band_seed_splat_radius_voxels`` picks the seed
   so that window covers the capillary force band and the one further cell its
   normal and curvature stencil reads.

3. **Bounded, and monotone in passes, outside it.** Each pass takes a min
   against another real particle distance, so a voxel's value can only fall,
   and it is bounded below by the exact field -- pass count is an accuracy dial
   with no correctness risk in either direction.

The derived quantities are what surface tension actually reads, so the normal
direction and the curvature are checked too, not just the field: differentiation
amplifies error, and propagation error is spatially clustered rather than
smooth, which is the shape that hurts a second derivative most.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import constants as C  # noqa: E402
from fluidSim_warp_blender51.sim import surface as SURF  # noqa: E402
from fluidSim_warp_blender51.sim.fields import ScalarField, SparseGrid  # noqa: E402

VOXEL = 0.02
RADIUS_SCALE = 1.2
SEEDS = (2, 3, 4, 5)


# --------------------------------------------------------------------------
# Pure arithmetic: no device needed.
# --------------------------------------------------------------------------


def test_passes_cover_the_gap_between_seed_and_splat():
    assert SURF.band_propagation_passes(6, 4) == 2
    assert SURF.band_propagation_passes(7, 2) == 5
    # A seed at or past the splat radius leaves nothing to propagate, and must
    # not ask for a negative number of passes.
    assert SURF.band_propagation_passes(4, 4) == 0
    assert SURF.band_propagation_passes(3, 6) == 0


def test_seed_radius_covers_the_force_band_and_its_stencil():
    """seed * h - r must reach band + 1 voxels, which is what the rule buys."""

    for r_vox in (0.5, 0.6, 0.9, 1.4):
        for band in (0.5, 1.0, 1.5, 3.0):
            seed = SURF.band_seed_splat_radius_voxels(r_vox, band, 1)
            exact_to_voxels = seed - r_vox
            assert exact_to_voxels >= band + 1.0 - 1e-9, (r_vox, band, seed)
    # Never below the splat radius the particle support already required.
    assert SURF.band_seed_splat_radius_voxels(0.6, 1.5, 9) == 9


def test_propagate_is_the_default():
    """Propagation ships on; the splat stays reachable as the reference path.

    This assertion is the reverse of what it was.  Propagation is not
    bit-identical to the splat, so it was held opt-in until the device tests
    below had measured what "not identical" actually costs: 3.0e-5 degrees of
    normal direction and 2e-6 relative curvature, on a field that provably
    never underestimates.  The seed radius stays *derived* (``None``) rather
    than pinned to a constant, so the exactness window keeps following the
    band contract instead of drifting away from it.
    """

    assert SURF.default_sdf_band_mode() == SURF.SDF_BAND_MODE_PROPAGATE
    assert SURF.default_sdf_band_seed_radius() is None
    assert SURF.SDF_BAND_MODE_SPLAT in SURF.SDF_BAND_MODES


# --------------------------------------------------------------------------
# Device tests.
# --------------------------------------------------------------------------


def _scene(device, blob=True, n=60_000):
    """A cloud with a real free surface, not a filled box.

    A uniformly filled box has almost no band: every voxel is trivially seeded
    and the propagation never has to travel, so it would pass this file while
    saying nothing. The blob adds a concave seam between two spheres and a thin
    sheet, which is where closest-point propagation can actually go wrong.
    """

    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0),
                      voxel_size=VOXEL, leaf_capacity=8192, device=device)
    rng = np.random.default_rng(20260903)
    if blob:
        parts = []
        for centre, rad in (((0.35, 0.5, 0.5), 0.16), ((0.58, 0.5, 0.5), 0.13)):
            m = int(n * 0.42)
            u = rng.normal(size=(m, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            rr = rad * rng.uniform(0.0, 1.0, size=(m, 1)) ** (1.0 / 3.0)
            parts.append(np.asarray(centre) + u * rr)
        m = n - sum(len(p) for p in parts)
        parts.append(np.stack([rng.uniform(0.25, 0.75, m),
                               rng.uniform(0.70, 0.74, m),
                               rng.uniform(0.30, 0.70, m)], axis=1))
        pos = np.concatenate(parts)
    else:
        pos = rng.uniform(0.2, 0.8, size=(n, 3))
    pos = np.clip(pos, 0.03, 0.97).astype(np.float32)

    positions = wp.array(pos, dtype=wp.vec3, device=device)
    count = wp.array(np.array([len(pos)], dtype=np.int32), dtype=wp.int32,
                     device=device)
    grid.begin_topology()
    grid.mark_points(positions, count, len(pos))
    grid.expand(2)
    assert grid.finalize_topology(sync=True) >= 1
    return grid, positions, count, len(pos)


def _exact(grid, device, positions, count, npart, radius, splat, far):
    """The shipped splat, which another test already pins bit-for-bit."""

    sdf = ScalarField(grid)
    wp.launch(SURF.k_surface_sdf_reset, dim=grid.voxel_capacity,
              inputs=[sdf.data, grid.num_leaves, far], device=device)
    wp.launch(SURF.k_build_surface_sdf, dim=npart,
              inputs=[positions, count, grid.leaf_table, grid.voxels_dim,
                      grid.leaves_dim, grid.origin, float(grid.voxel_size),
                      radius, splat, far],
              outputs=[sdf.data], device=device)
    wp.synchronize_device(device)
    return sdf.data.numpy().copy()


def _propagated(grid, device, positions, count, npart, radius, seed, passes,
                far):
    nvox = grid.voxel_capacity
    h = float(grid.voxel_size)
    a, b = ScalarField(grid), ScalarField(grid)
    closest = wp.zeros(nvox, dtype=wp.int32, device=device)
    points = wp.zeros(nvox, dtype=wp.vec3, device=device)
    points_tmp = wp.zeros(nvox, dtype=wp.vec3, device=device)

    wp.launch(SURF.k_surface_sdf_reset, dim=nvox,
              inputs=[a.data, grid.num_leaves, SURF.SDF_BAND_FAR],
              device=device)
    wp.launch(SURF.k_surface_band_index_reset, dim=nvox,
              inputs=[grid.num_leaves], outputs=[closest], device=device)
    wp.launch(SURF.k_build_surface_sdf, dim=npart,
              inputs=[positions, count, grid.leaf_table, grid.voxels_dim,
                      grid.leaves_dim, grid.origin, h, radius, seed,
                      SURF.SDF_SPLAT_NO_CULL],
              outputs=[a.data], device=device)
    wp.launch(SURF.k_seed_surface_sdf_closest, dim=npart,
              inputs=[positions, count, grid.leaf_table, grid.voxels_dim,
                      grid.leaves_dim, grid.origin, h, radius, seed, a.data],
              outputs=[closest], device=device)
    wp.launch(SURF.k_gather_surface_band_points, dim=nvox,
              inputs=[closest, positions, grid.num_leaves], outputs=[points],
              device=device)

    src, dst = a.data, b.data
    for _ in range(int(passes)):
        wp.launch(SURF.k_propagate_surface_sdf_band, dim=nvox,
                  inputs=[src, points, grid.leaf_table, grid.leaf_coords,
                          grid.voxels_dim, grid.leaves_dim, grid.num_leaves,
                          grid.origin, h, radius],
                  outputs=[dst, points_tmp], device=device)
        src, dst = dst, src
        points, points_tmp = points_tmp, points
    wp.launch(SURF.k_clamp_surface_sdf, dim=nvox,
              inputs=[grid.num_leaves, far], outputs=[src], device=device)
    wp.synchronize_device(device)
    return src.numpy().copy()


def _active(grid):
    n = int(grid.num_leaves.numpy()[0])
    mask = np.zeros(grid.voxel_capacity, dtype=bool)
    mask[: n * C.LEAF_VOLUME] = True
    return mask


def _dense(grid, flat):
    """Lift the sparse field into (i, j, k) so derivatives are a stencil away.

    Sized by the leaf lattice rather than ``voxels_dim``: the last leaf row can
    extend past the domain extent, and indexing that would wrap. Cells outside
    the allocated leaves stay NaN and are dropped from every statistic, so a
    stencil straddling the sparse boundary is excluded rather than silently
    reading a fill value.
    """

    n = int(grid.num_leaves.numpy()[0])
    coords = grid.leaf_coords.numpy()[:n]
    ld = grid.leaves_dim
    out = np.full((int(ld[0]) * C.LEAF_DIM, int(ld[1]) * C.LEAF_DIM,
                   int(ld[2]) * C.LEAF_DIM), np.nan, dtype=np.float64)
    local = np.arange(C.LEAF_VOLUME)
    li = local & (C.LEAF_DIM - 1)
    lj = (local >> 3) & (C.LEAF_DIM - 1)
    lk = (local >> 6) & (C.LEAF_DIM - 1)
    for slot in range(n):
        c = coords[slot]
        base = slot * C.LEAF_VOLUME
        out[(c[0] << 3) + li, (c[1] << 3) + lj, (c[2] << 3) + lk] = \
            flat[base: base + C.LEAF_VOLUME]
    return out


def _derivatives(field, h):
    grad = np.stack(np.gradient(field, h, edge_order=1), axis=-1)
    lap = np.zeros_like(field)
    for axis in range(3):
        lap += (np.roll(field, -1, axis) - 2.0 * field
                + np.roll(field, 1, axis)) / (h * h)
    return grad, lap


def _config(filters):
    """The (splat, far, seed) the solver would derive, from the shipped rules."""

    from fluidSim_warp_blender51.sim.surface_tension_math import (
        normal_support_voxels, required_sdf_splat_radius_voxels)

    band, smoothing = 1.5, 1
    r_vox = 0.5 * RADIUS_SCALE
    support = normal_support_voxels(band, smoothing) + filters
    splat = required_sdf_splat_radius_voxels(r_vox, band, smoothing, filters)
    far = (support + 1.0) * VOXEL
    seed = SURF.band_seed_splat_radius_voxels(
        r_vox, band, max(1, int(np.ceil(r_vox + 1.0))), smoothing, filters)
    return splat, far, seed


@pytest.fixture(scope="module")
def scene(cuda_device):
    return _scene(cuda_device)


@pytest.mark.parametrize("filters", [0, 1])
@pytest.mark.parametrize("seed", SEEDS)
def test_propagation_never_underestimates(cuda_device, scene, filters, seed):
    """The whole correctness argument in one assertion.

    A capillary force computed from a field that dips *below* the true
    distance would pull on cells the fluid does not reach. Every value here is
    a real particle distance, so the field can only ever sit above the exact
    one; the tolerance is a float32 ulp at these magnitudes, not slack.
    """

    grid, positions, count, npart = scene
    splat, far, _ = _config(filters)
    radius = 0.5 * VOXEL * RADIUS_SCALE
    passes = SURF.band_propagation_passes(splat, seed)

    exact = _exact(grid, cuda_device, positions, count, npart, radius, splat,
                   far)
    prop = _propagated(grid, cuda_device, positions, count, npart, radius,
                       seed, passes, far)
    band = _active(grid) & (exact < far - 1e-9)
    assert band.sum() > 1000, "geometry produced no band to test"

    # 1e-5 voxels is ~27 float32 ulps at these magnitudes -- loose enough that
    # a differently-rounded GPU cannot flake it, and still three orders below
    # the ~1e-2 voxel error the propagation legitimately carries outside the
    # seeded window, so a real underestimate cannot hide under it.
    deficit = (exact - prop)[band]
    assert deficit.max() <= 1e-5 * VOXEL, (
        f"propagated field dips {deficit.max() / VOXEL:.3e} voxels below the "
        f"exact field at seed={seed}, filters={filters}")


@pytest.mark.parametrize("filters", [0, 1])
@pytest.mark.parametrize("seed", SEEDS)
def test_exact_inside_the_seeded_window(cuda_device, scene, filters, seed):
    """Every voxel with ``sdf <= seed * h - r`` matches the splat exactly."""

    grid, positions, count, npart = scene
    splat, far, _ = _config(filters)
    radius = 0.5 * VOXEL * RADIUS_SCALE
    passes = SURF.band_propagation_passes(splat, seed)

    exact = _exact(grid, cuda_device, positions, count, npart, radius, splat,
                   far)
    prop = _propagated(grid, cuda_device, positions, count, npart, radius,
                       seed, passes, far)
    window = _active(grid) & (exact <= seed * VOXEL - radius)
    assert window.sum() > 1000, "seeded window is empty; nothing was checked"

    err = np.abs(prop - exact)[window]
    assert err.max() <= 1e-5 * VOXEL, (
        f"seeded window is not exact at seed={seed}, filters={filters}: "
        f"max error {err.max() / VOXEL:.3e} voxels over {int(window.sum())} "
        f"voxels")


@pytest.mark.parametrize("filters", [0, 1])
def test_derived_seed_leaves_the_capillary_stencil_unchanged(
        cuda_device, scene, filters):
    """Normals and curvature at the interface match the exact splat.

    The seed radius the solver derives puts the whole force band and its
    stencil inside the exact window, so the quantities surface tension reads
    should be indistinguishable from the full splat -- not merely close.
    """

    grid, positions, count, npart = scene
    splat, far, seed = _config(filters)
    radius = 0.5 * VOXEL * RADIUS_SCALE
    passes = SURF.band_propagation_passes(splat, seed)
    assert passes > 0, "derived seed left nothing to propagate"

    exact = _dense(grid, _exact(grid, cuda_device, positions, count, npart,
                                radius, splat, far))
    prop = _dense(grid, _propagated(grid, cuda_device, positions, count, npart,
                                    radius, seed, passes, far))

    ok = np.isfinite(exact) & np.isfinite(prop) & (exact < far - 1e-9)
    for axis in range(3):
        ok &= (np.roll(np.isfinite(exact), 1, axis)
               & np.roll(np.isfinite(exact), -1, axis))
    shell = ok & (np.abs(exact) < 2.0 * VOXEL)
    assert shell.sum() > 500, "no near-interface shell to compare"

    ge, le = _derivatives(exact, VOXEL)
    gp, lp = _derivatives(prop, VOXEL)
    ne = ge / np.maximum(np.linalg.norm(ge, axis=-1, keepdims=True), 1e-12)
    npd = gp / np.maximum(np.linalg.norm(gp, axis=-1, keepdims=True), 1e-12)
    angle = np.degrees(np.arccos(
        np.clip((ne * npd).sum(axis=-1), -1.0, 1.0)))[shell]

    scale = np.percentile(np.abs(le[shell]), 95)
    curvature = np.abs(lp - le)[shell] / max(scale, 1e-12)

    assert angle.max() < 1.0e-3, (
        f"normal direction moved {angle.max():.4f} deg at filters={filters}")
    assert curvature.max() < 1.0e-3, (
        f"curvature moved {curvature.max():.5f} relative at "
        f"filters={filters}")


@pytest.mark.parametrize("filters", [0, 1])
def test_extra_passes_only_ever_lower_the_field(cuda_device, scene, filters):
    """Pass count is a pure accuracy dial: monotone down, floored by exact.

    This is what makes a cheaper seed safe to offer. If a pass could ever
    *raise* a value, the seed radius would be a correctness knob rather than a
    performance one, and no error bound derived at one pass count would carry
    to another.
    """

    grid, positions, count, npart = scene
    splat, far, _ = _config(filters)
    radius = 0.5 * VOXEL * RADIUS_SCALE
    seed = 2
    mask = _active(grid)

    previous = None
    for passes in range(SURF.band_propagation_passes(splat, seed) + 3):
        field = _propagated(grid, cuda_device, positions, count, npart, radius,
                            seed, passes, far)
        if previous is not None:
            rose = (field[mask] > previous[mask] + 1e-5 * VOXEL).sum()
            assert rose == 0, (
                f"{rose} voxels rose going to {passes} passes at "
                f"filters={filters}")
        previous = field
