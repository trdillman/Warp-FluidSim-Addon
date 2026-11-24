"""Guided shell upres: the guide kernels on a known coarse liquid.

Run on CPU with ``FLUIDSIM_TEST_DEVICE=cpu``; on a GPU host they use CUDA.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import guided_upres as GU
from fluidSim_warp_blender51.sim.fields import (
    ScalarField,
    SparseGrid,
    voxel_coord_of_flat,
)

SIZE = (0.64, 0.64, 0.64)
COARSE_H = 0.04
FINE_H = 0.02
CENTRE = (0.32, 0.32, 0.32)
RADIUS = 0.16
VEL = (0.5, -0.25, 1.0)


@wp.kernel
def _k_sphere(sdf: wp.array(dtype=float), leaf_coords: wp.array(dtype=wp.vec3i),
              origin: wp.vec3, h: float, c: wp.vec3, r: float):
    tid = wp.tid()
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    x = origin + h * wp.vec3(float(ijk[0]) + 0.5, float(ijk[1]) + 0.5,
                             float(ijk[2]) + 0.5)
    sdf[tid] = wp.length(x - c) - r


@wp.kernel
def _k_centres(leaf_coords: wp.array(dtype=wp.vec3i), origin: wp.vec3,
               h: float, out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    out[tid] = origin + h * wp.vec3(float(ijk[0]) + 0.5, float(ijk[1]) + 0.5,
                                    float(ijk[2]) + 0.5)


def _grid(device, h, lo=(0.08, 0.08, 0.08), hi=(0.56, 0.56, 0.56)):
    g = SparseGrid(origin=(0.0, 0.0, 0.0), size=SIZE, voxel_size=h,
                   leaf_capacity=512, device=device)
    g.begin_topology()
    g.mark_world_box(lo, hi)
    g.finalize_topology()
    return g


def _coarse_frame(device, time, velocity=VEL, radius=RADIUS):
    g = _grid(device, COARSE_H)
    sdf = ScalarField(g, name="coarse_sdf")
    wp.launch(_k_sphere, dim=g.voxel_capacity, device=device,
              inputs=[sdf.data, g.leaf_coords, g.origin, COARSE_H,
                      wp.vec3(*CENTRE), radius])
    n = g.voxel_capacity

    def full(value):
        return wp.full(n, float(value), dtype=float, device=device)

    vel = SimpleNamespace(u=full(velocity[0]), v=full(velocity[1]),
                          w=full(velocity[2]), wu=full(1.0), wv=full(1.0),
                          ww=full(1.0))
    return GU.GuideFrame(g, sdf, vel, time)


def _centres(grid, device):
    n = int(grid.active_voxel_count)
    out = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(_k_centres, dim=n, device=device,
              inputs=[grid.leaf_coords, grid.origin, FINE_H], outputs=[out])
    return out.numpy()


def test_phi_takes_the_coarse_liquid_eroded_by_the_shell(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=3)
    frame = _coarse_frame(dev, 0.0)
    guide.set_frames(frame, None)
    fine = _grid(dev, FINE_H)
    limit = 8.0 * FINE_H
    phi = wp.full(fine.voxel_capacity, limit, dtype=float, device=dev)
    guide.apply_phi(fine, phi, fine.active_voxel_count, limit, dev)

    n = int(fine.active_voxel_count)
    x = _centres(fine, dev)
    p = phi.numpy()[:n]
    d = np.linalg.norm(x - np.asarray(CENTRE), axis=1) - RADIUS
    shell = 3 * FINE_H
    # Deep coarse liquid is liquid in the fine solve...
    deep = d < -shell - COARSE_H
    assert deep.sum() > 100 and np.all(p[deep] < 0.0)
    # ...and the shell itself is left to the fine particles.
    band = d > -shell + COARSE_H
    assert np.all(p[band] >= 0.0)


def test_deep_fine_liquid_cannot_drift_outside_the_coarse_surface(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=2)
    guide.set_frames(_coarse_frame(dev, 0.0), None)
    fine = _grid(dev, FINE_H)
    limit = 8.0 * FINE_H
    # Fine thinks everything is deep liquid.
    phi = wp.full(fine.voxel_capacity, -limit, dtype=float, device=dev)
    guide.apply_phi(fine, phi, fine.active_voxel_count, limit, dev)
    n = int(fine.active_voxel_count)
    x = _centres(fine, dev)
    p = phi.numpy()[:n]
    d = np.linalg.norm(x - np.asarray(CENTRE), axis=1) - RADIUS
    far_out = d > 3 * 2 * FINE_H + COARSE_H
    assert far_out.sum() > 100
    assert np.all(p[far_out] > -2 * FINE_H)


def test_interior_velocity_follows_the_guide_and_blends_in_time(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=2)
    a = _coarse_frame(dev, 0.0, velocity=(0.0, 0.0, 0.0))
    b = _coarse_frame(dev, 1.0, velocity=(1.0, -2.0, 4.0))
    guide.set_frames(a, b)
    guide.set_time(0.25)
    assert guide.alpha == pytest.approx(0.25)
    fine = _grid(dev, FINE_H)
    n = fine.voxel_capacity
    limit = 8.0 * FINE_H
    phi = wp.full(n, limit, dtype=float, device=dev)
    guide.apply_phi(fine, phi, fine.active_voxel_count, limit, dev)
    nb = [wp.full(n, 9.0, dtype=float, device=dev) for _ in range(3)]
    guide.apply_velocity(fine, phi, *nb, fine.active_voxel_count, dev)
    m = int(fine.active_voxel_count)
    p = phi.numpy()[:m]
    deep = p < -2 * FINE_H
    shallow = p >= -2 * FINE_H
    assert deep.sum() > 50
    for arr, expected in zip(nb, (0.25, -0.5, 1.0)):
        values = arr.numpy()[:m]
        np.testing.assert_allclose(values[deep], expected, atol=1e-5)
        # Shell and air keep their own velocity.
        assert np.all(values[shallow] == 9.0)


def test_mark_leaves_covers_the_coarse_liquid(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=2)
    guide.set_frames(_coarse_frame(dev, 0.0), None)
    fine = SparseGrid(origin=(0.0, 0.0, 0.0), size=SIZE, voxel_size=FINE_H,
                      leaf_capacity=512, device=dev)
    fine.begin_topology()
    guide.mark_leaves(fine, dev)
    fine.finalize_topology()
    x = _centres(fine, dev)
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    assert np.all(lo <= np.asarray(CENTRE) - RADIUS)
    assert np.all(hi >= np.asarray(CENTRE) + RADIUS)


@wp.kernel
def _k_ball_union(sdf: wp.array(dtype=float), leaf_coords: wp.array(dtype=wp.vec3i),
                  origin: wp.vec3, h: float, c: wp.vec3, r: float, floor: float):
    """A sphere as a particle splat sees it: never deeper than ``floor``."""
    tid = wp.tid()
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    x = origin + h * wp.vec3(float(ijk[0]) + 0.5, float(ijk[1]) + 0.5,
                             float(ijk[2]) + 0.5)
    sdf[tid] = wp.max(wp.length(x - c) - r, floor)


def _splat_like_frame(device, time=0.0, radius=RADIUS, velocity=VEL):
    frame = _coarse_frame(device, time, velocity=velocity, radius=radius)
    g = frame
    wp.launch(_k_ball_union, dim=int(g.sdf.shape[0]), device=device,
              inputs=[g.sdf, g.leaf_coords, g.origin, COARSE_H,
                      wp.vec3(*CENTRE), radius, -0.6 * COARSE_H])
    return frame


def _coarse_centres(frame, device):
    n = frame.active_voxels
    out = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(_k_centres, dim=n, device=device,
              inputs=[frame.leaf_coords, frame.origin, COARSE_H], outputs=[out])
    return out.numpy()


def test_reconstruct_depth_turns_a_splat_sdf_into_interior_distance(cuda_device):
    dev = cuda_device
    frame = _splat_like_frame(dev)
    n = frame.active_voxels
    before = frame.sdf.numpy()[:n].copy()
    assert before.min() >= -0.6 * COARSE_H - 1e-6
    depth = 4.0 * COARSE_H
    frame.reconstruct_depth(depth)
    after = frame.sdf.numpy()[:n]
    d = np.linalg.norm(_coarse_centres(frame, dev) - np.asarray(CENTRE), axis=1) - RADIUS
    inside = d < -COARSE_H
    # Down to the requested depth it tracks the true distance within a voxel.
    np.testing.assert_allclose(np.maximum(after[inside], -depth),
                               np.maximum(d[inside], -depth), atol=COARSE_H)
    assert after.min() < -2.5 * COARSE_H
    # Air and the surface layer are untouched.
    np.testing.assert_array_equal(after[before >= 0.0], before[before >= 0.0])


@wp.kernel
def _k_wall(sdf: wp.array(dtype=float), collision: wp.array(dtype=float),
            leaf_coords: wp.array(dtype=wp.vec3i), origin: wp.vec3, h: float,
            wall_x: float):
    """Solid below ``wall_x``; a splat SDF finds no particles inside it."""
    tid = wp.tid()
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    x = origin[0] + h * (float(ijk[0]) + 0.5)
    collision[tid] = x - wall_x
    if x < wall_x:
        sdf[tid] = h


def test_reconstruct_depth_keeps_liquid_on_a_wall_deep(cuda_device):
    dev = cuda_device
    frame = _splat_like_frame(dev)
    n = int(frame.sdf.shape[0])
    collision = wp.zeros(n, dtype=float, device=dev)
    wall_x = CENTRE[0]
    wp.launch(_k_wall, dim=n, device=dev,
              inputs=[frame.sdf, collision, frame.leaf_coords, frame.origin,
                      COARSE_H, wall_x])
    frame.reconstruct_depth(4.0 * COARSE_H, collision)
    m = frame.active_voxels
    after = frame.sdf.numpy()[:m]
    x = _coarse_centres(frame, dev)
    r = np.linalg.norm(x[:, 1:] - np.asarray(CENTRE[1:]), axis=1)
    # Liquid against the wall, near the sphere's middle, is the deep interior.
    on_wall = (x[:, 0] > wall_x) & (x[:, 0] < wall_x + COARSE_H) & (r < 0.5 * RADIUS)
    assert on_wall.sum() > 0
    assert np.all(after[on_wall] < -2.0 * COARSE_H)


def test_shell_guidance_works_from_a_splat_sdf_once_deepened(cuda_device):
    dev = cuda_device
    fine = _grid(dev, FINE_H)
    limit = 8.0 * FINE_H

    def deep_cells(frame):
        guide = GU.ShellGuide(shell_voxels=3)
        guide.set_frames(frame, None)
        phi = wp.full(fine.voxel_capacity, limit, dtype=float, device=dev)
        guide.apply_phi(fine, phi, fine.active_voxel_count, limit, dev)
        return int((phi.numpy()[:int(fine.active_voxel_count)] < 0.0).sum())

    # A union of balls is never deeper than the shell: no guided interior.
    assert deep_cells(_splat_like_frame(dev)) == 0
    frame = _splat_like_frame(dev)
    frame.reconstruct_depth(3 * 3 * FINE_H + 2 * COARSE_H)
    assert deep_cells(frame) > 100


def _seed_inputs(dev, fine, phi_value=None):
    n = fine.voxel_capacity
    limit = 8.0 * FINE_H
    phi = wp.full(n, limit if phi_value is None else phi_value, dtype=float, device=dev)
    zeros = wp.zeros(n, dtype=wp.int32, device=dev)
    return {"phi": phi, "cls": wp.zeros(n, dtype=wp.int32, device=dev),
            "collision": wp.full(n, 1.0, dtype=float, device=dev),
            "vtype": wp.zeros(n, dtype=wp.int32, device=dev),
            "counts": zeros, "band": wp.zeros(n, dtype=wp.int32, device=dev),
            "demand": wp.zeros(n, dtype=wp.int32, device=dev),
            "band_offsets": wp.zeros(n, dtype=wp.int32, device=dev),
            "band_admitted": wp.zeros(1, dtype=wp.int32, device=dev)}


def _demand(guide, fine, inp, dev, fill=4):
    a, b, common = guide._guide_inputs()
    h = FINE_H
    wp.launch(GU.k_guide_seed_demand, dim=fine.voxel_capacity, device=dev,
              inputs=[inp["phi"], inp["cls"], inp["collision"], inp["vtype"],
                      inp["counts"], inp["band"], inp["band_offsets"],
                      inp["band_admitted"], fine.leaf_coords, fine.num_leaves, fine.origin,
                      h, guide.shell_voxels * h, a.sdf, a.leaf_table, b.sdf,
                      b.leaf_table, *common, float(guide.alpha),
                      1 if guide.bootstrap else 0, fill],
              outputs=[inp["demand"]])
    return inp["demand"].numpy()[:int(fine.active_voxel_count)]


def test_seed_fills_the_whole_coarse_shell_on_the_first_frame(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=3)
    guide.set_frames(_coarse_frame(dev, 0.0), None)
    fine = _grid(dev, FINE_H)
    dem = _demand(guide, fine, _seed_inputs(dev, fine), dev)
    d = np.linalg.norm(_centres(fine, dev) - np.asarray(CENTRE), axis=1) - RADIUS
    shell = 3 * FINE_H
    # Well inside the eligible band (-(shell + h), -h/2), clear of sampling error.
    inner = (d < -0.5 * FINE_H - 0.25 * COARSE_H) & (d > -(shell + FINE_H) + 0.25 * COARSE_H)
    assert inner.sum() > 100
    assert np.all(dem[inner] == 4)
    assert np.all(dem[d > COARSE_H] == 0)            # air
    assert np.all(dem[d < -shell - FINE_H - COARSE_H] == 0)   # deep interior


def test_seed_after_the_first_frame_only_fills_new_coarse_liquid(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=3)
    small, big = 0.10, RADIUS
    guide.set_frames(_coarse_frame(dev, 0.0, radius=small),
                     _coarse_frame(dev, 1.0, radius=big))
    guide.set_time(1.0)
    guide.bootstrap = False
    fine = _grid(dev, FINE_H)
    dem = _demand(guide, fine, _seed_inputs(dev, fine), dev)
    r = np.linalg.norm(_centres(fine, dev) - np.asarray(CENTRE), axis=1)
    assert dem.sum() > 0
    # Nothing where frame a already had liquid.
    assert np.all(dem[r < small - COARSE_H] == 0)
    # Occupied cells are never topped up.
    guide.set_frames(_coarse_frame(dev, 0.0), None)
    guide.bootstrap = True
    inp = _seed_inputs(dev, fine)
    inp["counts"] = wp.full(fine.voxel_capacity, 1, dtype=wp.int32, device=dev)
    assert _demand(guide, fine, inp, dev).sum() == 0
    # Fine sink cells are never seeded either.
    inp = _seed_inputs(dev, fine)
    inp["vtype"] = wp.full(fine.voxel_capacity, 2, dtype=wp.int32, device=dev)  # VOXEL_SINK
    assert _demand(guide, fine, inp, dev).sum() == 0


def _empty_frame(device, time):
    g = SparseGrid(origin=(0.0, 0.0, 0.0), size=SIZE, voxel_size=COARSE_H,
                   leaf_capacity=512, device=device)
    g.begin_topology()
    g.finalize_topology()
    sdf = ScalarField(g, name="coarse_sdf")
    n = g.voxel_capacity
    zero = wp.zeros(n, dtype=float, device=device)
    vel = SimpleNamespace(u=zero, v=zero, w=zero, wu=zero, wv=zero, ww=zero)
    return GU.GuideFrame(g, sdf, vel, time)


def test_seed_fills_liquid_missing_from_an_empty_previous_frame(cuda_device):
    dev = cuda_device
    fine = _grid(dev, FINE_H)
    first = GU.ShellGuide(shell_voxels=3)
    first.set_frames(_coarse_frame(dev, 1.0), None)
    expected = _demand(first, fine, _seed_inputs(dev, fine), dev)
    # The coarse solve emitted its liquid during this frame: frame a is empty.
    guide = GU.ShellGuide(shell_voxels=3)
    guide.set_frames(_empty_frame(dev, 0.0), _coarse_frame(dev, 1.0))
    guide.set_time(0.5)
    for bootstrap in (True, False):
        guide.bootstrap = bootstrap
        dem = _demand(guide, fine, _seed_inputs(dev, fine), dev)
        assert dem.sum() > 0
        np.testing.assert_array_equal(dem, expected)
    # The fine level set follows frame b alone there.
    limit = 8.0 * FINE_H
    phi = wp.full(fine.voxel_capacity, limit, dtype=float, device=dev)
    guide.apply_phi(fine, phi, fine.active_voxel_count, limit, dev)
    assert int((phi.numpy()[:int(fine.active_voxel_count)] < 0.0).sum()) > 0


@pytest.mark.parametrize("a_faces_valid", [True, False])
def test_seeded_particles_carry_the_coarse_velocity(cuda_device, a_faces_valid):
    from types import SimpleNamespace

    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=3)
    if a_faces_valid:
        guide.set_frames(_coarse_frame(dev, 0.0), None)
    else:
        # Frame a is allocated but has no valid face: seeds take frame b's
        # velocity whole, not a blend toward 0.
        a = _coarse_frame(dev, 0.0, velocity=(0.0, 0.0, 0.0))
        for weight in (a.wu, a.wv, a.ww):
            weight.zero_()
        guide.set_frames(a, _coarse_frame(dev, 1.0))
        guide.set_time(0.25)
    fine = _grid(dev, FINE_H)
    inp = _seed_inputs(dev, fine)
    cap = 200000
    attrs = {k: wp.zeros(cap, dtype=float, device=dev)
             for k in ("age", "life", "temperature", "viscosity", "density",
                       "time_offset")}
    particles = SimpleNamespace(
        capacity=cap, count=wp.zeros(1, dtype=wp.int32, device=dev),
        position=wp.zeros(cap, dtype=wp.vec3, device=dev),
        velocity=wp.zeros(cap, dtype=wp.vec3, device=dev),
        alive=wp.zeros(cap, dtype=wp.int32, device=dev),
        attr=lambda name: attrs[name], refresh_host_count=lambda: None)
    n = fine.voxel_capacity
    scratch = [wp.zeros(n, dtype=wp.int32, device=dev)] + [
        wp.zeros(1, dtype=wp.int32, device=dev) for _ in range(3)]
    guide.seed(fine, fine.active_voxel_count, inp["phi"], inp["cls"],
               inp["collision"], inp["vtype"], inp["counts"], inp["band"], *scratch,
               particles, 4, 123, dev)
    count = int(particles.count.numpy()[0])
    assert count > 1000
    v = particles.velocity.numpy()[:count]
    np.testing.assert_allclose(v, np.tile(VEL, (count, 1)), atol=1e-5)
    x = particles.position.numpy()[:count]
    d = np.linalg.norm(x - np.asarray(CENTRE), axis=1) - RADIUS
    assert d.max() < FINE_H and d.min() > -(3 + 2) * FINE_H - COARSE_H


def test_fine_budgets_scale_leaves_with_volume_and_particles_with_area():
    assert GU.fine_budget_scales(2) == (4.0, 8.0)
    assert GU.fine_budget_scales(3) == (9.0, 27.0)


def test_fine_colliders_are_private_copies_and_dynamic_ones_are_refused():
    from types import SimpleNamespace

    static = SimpleNamespace(mesh=object(), dynamic=False, base_points=None)
    copies = GU.fine_colliders([static])
    assert copies[0] is not static and copies[0].mesh is static.mesh
    copies[0].base_points = "fine"
    assert static.base_points is None
    with pytest.raises(NotImplementedError):
        GU.fine_colliders([SimpleNamespace(mesh=object(), dynamic=True)])


def test_guide_phi_leaves_fine_solid_and_sink_cells_as_air(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=3)
    frame = _coarse_frame(dev, 0.0)
    frame.reconstruct_depth(3 * 3 * FINE_H + 2 * COARSE_H)
    guide.set_frames(frame, None)
    fine = _grid(dev, FINE_H)
    n = fine.voxel_capacity
    m = int(fine.active_voxel_count)
    limit = 8.0 * FINE_H
    x = _centres(fine, dev)
    solid = np.ones(n, dtype=np.float32)
    solid[:m][x[:, 0] < CENTRE[0]] = -1.0
    sink = np.zeros(n, dtype=np.int32)
    sink[:m][(x[:, 0] >= CENTRE[0]) & (x[:, 1] < CENTRE[1])] = 2  # VOXEL_SINK
    phi = wp.full(n, -limit, dtype=float, device=dev)
    guide.apply_phi(fine, phi, m, limit, dev,
                    wp.array(solid, dtype=float, device=dev),
                    wp.array(sink, dtype=wp.int32, device=dev))
    out = phi.numpy()[:m]
    blocked = (solid[:m] < 0.0) | (sink[:m] != 0)
    assert np.all(out[blocked] == limit)
    assert np.any(out[~blocked] < 0.0)


def test_fine_sources_keep_sinks_and_drop_liquid_emission():
    from fluidSim_warp_blender51.sim.enums import SourceShapeType
    from fluidSim_warp_blender51.sim.sourcing import SourceBatch, SourceShapeDesc

    liquid = SourceShapeDesc()
    liquid.stype = SourceShapeType.SOURCE
    sink = SourceShapeDesc()
    sink.stype = SourceShapeType.SINK
    plans = (SimpleNamespace(operation=0), SimpleNamespace(operation=1))  # LIQUID, SINK
    descriptors, kept = GU.fine_source_batch(SimpleNamespace(
        legacy_descriptors=(liquid, sink), native_plans=plans))
    assert descriptors == (sink,)
    assert kept == (plans[1],)
    assert GU.fine_source_batch(SourceBatch()) == ((), ())


def test_scene_sync_copies_sources_and_force_fields_only_when_replaced():
    from fluidSim_warp_blender51.sim.sourcing import SourceBatch

    calls = []
    fine = SimpleNamespace(
        set_source_batch=lambda d, p: calls.append(("sources", d, p)),
        set_force_fields=lambda f: calls.append(("fields", f)))
    coarse = SimpleNamespace(sourcing=SimpleNamespace(batch=SourceBatch()),
                             force_fields=["wind"])
    upres = GU.GuidedUpres.__new__(GU.GuidedUpres)
    upres.coarse, upres.fine = coarse, fine
    upres._sources = upres._force_fields = None
    upres._sync_scene()
    assert calls == [("sources", (), ()), ("fields", ["wind"])]
    upres._sync_scene()
    assert len(calls) == 2          # unchanged: nothing reinstalled
    coarse.force_fields = ["vortex"]
    upres._sync_scene()
    assert calls[-1] == ("fields", ["vortex"]) and len(calls) == 3


def test_seed_skips_only_cells_the_band_reseed_actually_filled(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=3)
    guide.set_frames(_coarse_frame(dev, 0.0), None)
    fine = _grid(dev, FINE_H)
    expected = _demand(guide, fine, _seed_inputs(dev, fine), dev)
    inp = _seed_inputs(dev, fine)
    inp["band"] = wp.full(fine.voxel_capacity, 1, dtype=wp.int32, device=dev)
    # The band asked everywhere but got no budget: the guide still fills.
    np.testing.assert_array_equal(_demand(guide, fine, inp, dev), expected)
    # The band got everything it asked for: nothing left for the guide.
    inp["band_admitted"] = wp.full(1, 1 << 30, dtype=wp.int32, device=dev)
    assert _demand(guide, fine, inp, dev).sum() == 0


def test_interior_velocity_uses_the_only_frame_that_has_liquid(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=2)
    guide.set_frames(_empty_frame(dev, 0.0),
                     _coarse_frame(dev, 1.0, velocity=(1.0, -2.0, 4.0)))
    guide.set_time(0.25)
    fine = _grid(dev, FINE_H)
    n = fine.voxel_capacity
    limit = 8.0 * FINE_H
    phi = wp.full(n, limit, dtype=float, device=dev)
    guide.apply_phi(fine, phi, fine.active_voxel_count, limit, dev)
    nb = [wp.full(n, 9.0, dtype=float, device=dev) for _ in range(3)]
    guide.apply_velocity(fine, phi, *nb, fine.active_voxel_count, dev)
    m = int(fine.active_voxel_count)
    deep = phi.numpy()[:m] < -2 * FINE_H
    assert deep.sum() > 50
    for arr, expected in zip(nb, (1.0, -2.0, 4.0)):
        np.testing.assert_allclose(arr.numpy()[:m][deep], expected, atol=1e-5)


def test_moving_colliders_get_a_private_mesh_kept_in_step(cuda_device):
    dev = cuda_device
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    src = wp.Mesh(points=wp.array(pts, dtype=wp.vec3, device=dev),
                  indices=wp.array([0, 1, 2], dtype=wp.int32, device=dev))
    moving = SimpleNamespace(mesh=src, dynamic=False, linear_velocity=(1.0, 0.0, 0.0),
                             angular_velocity=(0.0, 0.0, 0.0), reposed=False)
    still = SimpleNamespace(mesh=src, dynamic=False, linear_velocity=(0.0, 0.0, 0.0),
                            angular_velocity=(0.0, 0.0, 0.0))
    cache = {}
    first, kept = GU.fine_colliders([moving, still], cache)
    assert kept.mesh is src                       # static: shared, never written
    assert first.mesh is not src
    first.mesh.points.assign(pts + 5.0)           # a fine re-pose
    np.testing.assert_array_equal(src.points.numpy(), pts)
    src.points.assign(pts + 1.0)                  # the scene moves the source
    again, _ = GU.fine_colliders([moving, still], cache)
    assert again.mesh is first.mesh               # reused, refreshed
    np.testing.assert_array_equal(again.mesh.points.numpy(), pts + 1.0)
    replacement = wp.Mesh(points=wp.array(pts + 2.0, dtype=wp.vec3, device=dev),
                          indices=src.indices)
    moving.mesh = replacement
    (updated,) = GU.fine_colliders([moving], cache)
    assert updated.mesh is not first.mesh
    assert set(cache) == {id(replacement)}
    np.testing.assert_array_equal(updated.mesh.points.numpy(), pts + 2.0)
    assert GU.fine_colliders([], cache) == []
    assert not cache


def test_factors_beyond_one_leaf_per_coarse_voxel_are_refused():
    with pytest.raises(ValueError):
        GU.GuidedUpres(None, factor=16)


def test_interior_velocity_ignores_a_frame_with_leaves_but_no_valid_faces(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=2)
    a = _coarse_frame(dev, 0.0, velocity=(0.0, 0.0, 0.0))
    for weight in (a.wu, a.wv, a.ww):
        weight.zero_()                     # allocated, but no valid face
    guide.set_frames(a, _coarse_frame(dev, 1.0, velocity=(1.0, -2.0, 4.0)))
    guide.set_time(0.25)
    fine = _grid(dev, FINE_H)
    n = fine.voxel_capacity
    limit = 8.0 * FINE_H
    phi = wp.full(n, limit, dtype=float, device=dev)
    guide.apply_phi(fine, phi, fine.active_voxel_count, limit, dev)
    nb = [wp.full(n, 9.0, dtype=float, device=dev) for _ in range(3)]
    guide.apply_velocity(fine, phi, *nb, fine.active_voxel_count, dev)
    m = int(fine.active_voxel_count)
    deep = phi.numpy()[:m] < -2 * FINE_H
    assert deep.sum() > 50
    for arr, expected in zip(nb, (1.0, -2.0, 4.0)):
        np.testing.assert_allclose(arr.numpy()[:m][deep], expected, atol=1e-5)
