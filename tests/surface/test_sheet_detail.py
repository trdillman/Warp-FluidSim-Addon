"""Sheet Detail: fine particles in the thin parts of a coarse liquid.

After Roy, Paquette and Poulin 2020 (particle upsampling for splashing
liquids). Run on CPU with ``FLUIDSIM_TEST_DEVICE=cpu``; on a GPU host they use
CUDA.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import sheet_detail as SD
from fluidSim_warp_blender51.sim.fields import (
    MacVelocityField,
    SparseGrid,
    voxel_flat_index,
)

H = 0.05
PPV = 8.0
DT = 1.0 / 24.0
GRAVITY = (0.0, 0.0, -9.81)


def _block(n=8, origin=(0.3, 0.3, 0.3)):
    """Full liquid: a 2x2x2 lattice per voxel over n^3 voxels."""
    axis = (np.arange(2 * n) + 0.5) * (H / 2)
    g = np.meshgrid(axis, axis, axis, indexing="ij")
    return (np.stack([a.ravel() for a in g], 1) + np.asarray(origin)).astype(np.float32)


def _sheet(n=16, z=0.5, origin=(0.3, 0.3)):
    """One particle layer, two particles per voxel side: a thin sheet."""
    axis = (np.arange(2 * n) + 0.5) * (H / 2)
    x, y = np.meshgrid(axis + origin[0], axis + origin[1], indexing="ij")
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], 1).astype(np.float32)


def _particles(device, pos, vel=None):
    pos = np.asarray(pos, np.float32)
    vel = np.zeros_like(pos) if vel is None else np.asarray(vel, np.float32)
    return (wp.array(pos, dtype=wp.vec3, device=device),
            wp.array(vel, dtype=wp.vec3, device=device),
            wp.array(np.array([len(pos)], np.int32), dtype=wp.int32, device=device),
            len(pos))


def _empty_grid(device):
    """No leaves anywhere: every fine particle falls ballistically."""
    grid = SimpleNamespace(
        origin=wp.vec3(0.0, 0.0, 0.0),
        leaf_table=wp.full(8, -1, dtype=wp.int32, device=device),
        voxels_dim=wp.vec3i(16, 16, 16), leaves_dim=wp.vec3i(2, 2, 2))
    z = wp.zeros(1, dtype=float, device=device)
    vel = SimpleNamespace(u=z, v=z, w=z, wu=z, wv=z, ww=z)
    return grid, vel


@wp.kernel
def _k_stretch_x(leaf_table: wp.array(dtype=wp.int32), voxels_dim: wp.vec3i,
                 leaves_dim: wp.vec3i, voxel_size: float, centre: float, rate: float,
                 u: wp.array(dtype=float), wu: wp.array(dtype=float),
                 wv: wp.array(dtype=float), ww: wp.array(dtype=float)):
    i, j, k = wp.tid()
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, wp.vec3i(i, j, k))
    if flat >= 0:
        # u lives on the voxel's lower x face.
        u[flat] = rate * (float(i) * voxel_size - centre)
        wu[flat] = 1.0
        wv[flat] = 1.0
        ww[flat] = 1.0


def _stretch_grid(device, points, centre, rate):
    """Leaves around ``points`` carrying u = rate * (x - centre), v = w = 0."""
    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), voxel_size=H,
                      leaf_capacity=64, device=device)
    pos, _vel, count, n = _particles(device, points)
    grid.begin_topology()
    grid.mark_points(pos, count, n)
    grid.expand(1)
    grid.finalize_topology(sync=True)
    vel = MacVelocityField(grid)
    dims = tuple(int(d) for d in grid.voxels_dim)
    wp.launch(_k_stretch_x, dim=dims,
              inputs=[grid.leaf_table, grid.voxels_dim, grid.leaves_dim, H, centre, rate,
                      vel.u, vel.wu, vel.wv, vel.ww], device=device)
    return grid, vel


def _step(sd, device, pos, *, grid=None, vel=None, dt=DT, pvel=None):
    ppos, pv, pcount, n = _particles(device, pos, pvel)
    if grid is None:
        grid, vel = _empty_grid(device)
    sd.step(ppos, pv, pcount, n, grid=grid, vel=vel, voxel_size=H,
            particles_per_voxel=PPV, gravity=GRAVITY, dt=dt)
    return sd.stats


def _thin(device, pos):
    sd = SD.SheetDetail(device=device)
    ppos, _pv, pcount, n = _particles(device, pos)
    return sd.classify(ppos, pcount, n, voxel_size=H,
                       particles_per_voxel=PPV).numpy()[:n]


# --- host maths ---------------------------------------------------------------

def test_a_fine_particle_has_one_split_of_the_coarse_volume():
    r = SD.fine_radius(1.0, 8)
    assert r == pytest.approx(0.5)
    assert 8 * r ** 3 == pytest.approx(1.0)


def test_full_liquid_neighbour_count_is_density_times_ball_volume():
    assert SD.full_neighbour_count(8.0, 2.0) == pytest.approx(8.0 * 4.0 / 3.0 * np.pi * 8.0)


# --- thin detection -----------------------------------------------------------

def test_a_single_layer_sheet_is_thin_and_the_inside_of_a_block_is_not(cuda_device):
    assert _thin(cuda_device, _sheet()).all()
    block = _block()
    thin = _thin(cuda_device, block)
    centre = np.abs(block - block.mean(0)).max(1) < 2 * H
    assert centre.any() and not thin[centre].any()


# --- spawning -----------------------------------------------------------------

def test_every_thin_particle_spawns_split_children_close_by(cuda_device):
    sd = SD.SheetDetail(SD.SheetDetailSettings(split=4), device=cuda_device)
    sheet = _sheet(n=6)
    stats = _step(sd, cuda_device, sheet)
    assert stats["added"] == 4 * len(sheet) == sd.count
    fine = sd.pos.numpy()
    # Children start within the jitter ball of a parent.
    d = np.linalg.norm(fine[:, None, :] - sheet[None, :, :], axis=2).min(1)
    assert d.max() <= 0.5 * H + 1e-6
    assert np.all(sd.age.numpy() == 0.0)


def test_children_of_a_sheet_stay_in_its_plane(cuda_device):
    sd = SD.SheetDetail(SD.SheetDetailSettings(split=8), device=cuda_device)
    sheet = _sheet(n=6, z=0.5)
    _step(sd, cuda_device, sheet)
    fine = sd.pos.numpy()
    # A thin disk: in-plane spread up to the jitter, out of plane a tenth.
    assert np.abs(fine[:, 2] - 0.5).max() <= 0.1 * H + 1e-6
    assert np.abs(fine[:, :2] - sheet[:, :2].mean(0)).max() > 0.5 * H


def test_full_liquid_spawns_nothing(cuda_device):
    sd = SD.SheetDetail(device=cuda_device)
    block = _block()
    stats = _step(sd, cuda_device, block)
    # Only the block's skin can be thin; its core never spawns.
    fine = sd.pos.numpy()
    if len(fine):
        assert np.abs(fine - block.mean(0)).max(1).min() > 1.5 * H
    assert stats["added"] < len(block)


def test_covered_particles_do_not_spawn_again(cuda_device):
    sd = SD.SheetDetail(SD.SheetDetailSettings(split=4), device=cuda_device)
    sheet = _sheet(n=6)
    grid, vel = _stretch_grid(cuda_device, sheet, centre=0.45, rate=0.0)
    _step(sd, cuda_device, sheet, grid=grid, vel=vel)
    first = sd.count
    stats = _step(sd, cuda_device, sheet, grid=grid, vel=vel)
    assert stats["kept"] == first
    assert stats["added"] == 0


def test_the_budget_is_a_hard_cap(cuda_device):
    sd = SD.SheetDetail(SD.SheetDetailSettings(split=8, max_particles=50),
                        device=cuda_device)
    stats = _step(sd, cuda_device, _sheet(n=6))
    assert sd.count == 50 and stats["budget_hit"]


# --- motion ---------------------------------------------------------------------

def test_a_stretching_flow_pulls_the_children_apart(cuda_device):
    sd = SD.SheetDetail(SD.SheetDetailSettings(split=8, max_age=10.0),
                        device=cuda_device)
    sheet = _sheet(n=6)
    centre = float(sheet[:, 0].mean())
    grid, vel = _stretch_grid(cuda_device, sheet, centre=centre, rate=0.6)
    _step(sd, cuda_device, sheet, grid=grid, vel=vel)
    spread0 = float(sd.pos.numpy()[:, 0].std())
    for _ in range(3):
        _step(sd, cuda_device, sheet, grid=grid, vel=vel)
    fine = sd.pos.numpy()
    # The original children move with u = rate * (x - centre): outward.
    spread = float(fine[:, 0].std())
    assert spread > spread0 * 1.02
    assert np.abs(fine[:, 1:] - sheet[:, 1:].mean(0)).max() < 0.25  # v = w = 0


def test_children_with_no_liquid_grid_fall_under_gravity(cuda_device):
    sd = SD.SheetDetail(device=cuda_device)
    sheet = _sheet(n=4)
    _step(sd, cuda_device, sheet)
    z0 = sd.pos.numpy()[:, 2].mean()
    # A short frame so the children stay near their parents.
    _step(sd, cuda_device, sheet, dt=0.02)
    old = sd.age.numpy() > 0.0
    z1 = sd.pos.numpy()[old, 2].mean()
    # Two symplectic Euler substeps fall 3/4 g dt^2 (the exact 1/2 needs more).
    assert z1 == pytest.approx(z0 - 0.75 * 9.81 * 0.02 ** 2, abs=2e-4)
    assert sd.vel.numpy()[old, 2].mean() == pytest.approx(-9.81 * 0.02, rel=1e-3)


def test_leaves_with_no_valid_faces_do_not_stop_the_children(cuda_device):
    # A leaf whose faces carry no weight gives no velocity sample; the
    # children must fall, not freeze in place.
    sd = SD.SheetDetail(device=cuda_device)
    sheet = _sheet(n=4)
    grid, vel = _stretch_grid(cuda_device, sheet, centre=0.45, rate=0.0)
    for a in (vel.wu, vel.wv, vel.ww):
        a.zero_()
    _step(sd, cuda_device, sheet, grid=grid, vel=vel)
    _step(sd, cuda_device, sheet, grid=grid, vel=vel, dt=0.02)
    old = sd.age.numpy() > 0.0
    assert sd.vel.numpy()[old, 2].mean() == pytest.approx(-9.81 * 0.02, rel=1e-3)


def test_valid_zero_flow_faces_hold_the_children_still(cuda_device):
    # Weighted faces with zero velocity are a real sample: no fall.
    sd = SD.SheetDetail(device=cuda_device)
    sheet = _sheet(n=4)
    grid, vel = _stretch_grid(cuda_device, sheet, centre=0.45, rate=0.0)
    _step(sd, cuda_device, sheet, grid=grid, vel=vel)
    _step(sd, cuda_device, sheet, grid=grid, vel=vel, dt=0.02)
    old = sd.age.numpy() > 0.0
    assert np.abs(sd.vel.numpy()[old]).max() < 1e-6


# --- removal ----------------------------------------------------------------------

def test_children_older_than_max_age_die(cuda_device):
    sd = SD.SheetDetail(SD.SheetDetailSettings(max_age=0.01), device=cuda_device)
    sheet = _sheet(n=4)
    _step(sd, cuda_device, sheet)
    stats = _step(sd, cuda_device, sheet)
    assert stats["kept"] == 0
    assert stats["added"] == sd.count > 0  # the parents are bare again


def test_children_the_liquid_left_behind_die(cuda_device):
    sd = SD.SheetDetail(device=cuda_device)
    sheet = _sheet(n=4)
    _step(sd, cuda_device, sheet)
    moved = sheet + np.float32([0.0, 0.0, 0.3])
    stats = _step(sd, cuda_device, moved)
    assert stats["kept"] == 0


def test_children_back_inside_full_liquid_die(cuda_device):
    sd = SD.SheetDetail(device=cuda_device)
    block = _block(n=8, origin=(0.3, 0.3, 0.3))
    sheet = _sheet(n=4, z=0.5, origin=(0.4, 0.4))  # inside the block
    _step(sd, cuda_device, sheet)
    assert sd.count
    stats = _step(sd, cuda_device, block)
    assert stats["kept"] == 0


# --- render inputs ------------------------------------------------------------------

def test_render_drops_only_the_thin_particles_children_stand_in_for(cuda_device):
    sheet = _sheet(n=6)
    ppos, _pv, pcount, n = _particles(cuda_device, sheet)
    covered = SD.SheetDetail(device=cuda_device)
    _step(covered, cuda_device, sheet)
    covered.classify(ppos, pcount, n, voxel_size=H, particles_per_voxel=PPV)
    _pos, count, _dim = covered.render_inputs(ppos, pcount, n)
    assert int(count.numpy()[0]) == 0
    # No budget: no children, so every coarse particle still renders.
    bare = SD.SheetDetail(SD.SheetDetailSettings(max_particles=0), device=cuda_device)
    _step(bare, cuda_device, sheet)
    pos, count, _dim = bare.render_inputs(ppos, pcount, n)
    assert int(count.numpy()[0]) == n
    # ... at their own positions, not uninitialised slots.
    got = np.sort(pos.numpy()[:n].view([("", np.float32)] * 3), axis=0)
    want = np.sort(sheet.view([("", np.float32)] * 3), axis=0)
    np.testing.assert_array_equal(got, want)


def test_render_before_any_step_passes_the_coarse_particles_through(cuda_device):
    sd = SD.SheetDetail(device=cuda_device)
    ppos, _pv, pcount, n = _particles(cuda_device, _sheet(n=4))
    pos, count, dim = sd.render_inputs(ppos, pcount, n)
    assert pos is ppos and count is pcount and dim == n
    _fine_pos, fine_count, fine_n = sd.fine_inputs()
    assert fine_n == 0 and int(fine_count.numpy()[0]) == 0


# --- state ------------------------------------------------------------------------

def test_state_survives_a_checkpoint_round_trip(cuda_device):
    sd = SD.SheetDetail(device=cuda_device)
    sheet = _sheet(n=4)
    _step(sd, cuda_device, sheet)
    _step(sd, cuda_device, sheet)
    arrays, scalars = sd.export_state()
    back = SD.SheetDetail(device=cuda_device)
    back.import_state(arrays, scalars)
    assert back.count == sd.count and back.frame == sd.frame
    np.testing.assert_array_equal(back.pos.numpy(), sd.pos.numpy())
    np.testing.assert_array_equal(back.age.numpy(), sd.age.numpy())
    with pytest.raises(ValueError):
        back.import_state({**arrays, "age": arrays["age"][:-1]}, scalars)


def test_the_same_inputs_give_the_same_children(cuda_device):
    def run():
        sd = SD.SheetDetail(device=cuda_device)
        _step(sd, cuda_device, _sheet(n=4))
        return np.sort(sd.pos.numpy(), axis=0)

    np.testing.assert_array_equal(run(), run())


def test_reset_forgets_the_children(cuda_device):
    sd = SD.SheetDetail(device=cuda_device)
    _step(sd, cuda_device, _sheet(n=4))
    sd.reset()
    assert sd.count == 0 and sd.frame == 0 and sd.thin is None


# --- the surface pipeline -----------------------------------------------------------

def _solver_with_a_sheet(device, **extra):
    if not wp.get_device(device).is_cuda:
        pytest.skip("the full solver needs CUDA")
    from fluidSim_warp_blender51.sim.enums import SurfaceResolutionMode
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(0.5, 0.5, 0.5), voxel_size=1.0 / 32.0,
        leaf_capacity=256, max_particles=60_000, surface_enabled=True,
        whitewater_enabled=False, narrow_band_enabled=False,
        surface_resolution_mode=int(SurfaceResolutionMode.MULTIPLIER),
        surface_resolution_multiplier=2.0, device=device, **extra)
    solver = FluidSolver(settings)
    h = 1.0 / 32.0
    axis = (np.arange(16) + 0.5) * (h / 2) + 0.17
    x, y = np.meshgrid(axis, axis, indexing="ij")
    sheet = np.stack([x.ravel(), y.ravel(), np.full(x.size, 0.3)], 1).astype(np.float32)
    solver.inject_particles(sheet)
    return solver


def test_the_render_surface_splats_the_fine_particles_and_the_solver_sdf_does_not(
        cuda_device):
    solver = _solver_with_a_sheet(cuda_device, surface_sheet_detail_enabled=True)
    try:
        solver.step_frame()
        surface = solver.surface
        assert surface.sheet is not None and surface.sheet.count > 0
        stats = surface.last_sdf_splat["sheetDetail"]
        assert stats["fineParticles"] == surface.sheet.count
        assert stats["fineRadius"] < 0.5 * float(solver.settings.voxel_size)
        verts, tris = surface.mesh()
        assert len(verts) and len(tris)
        # Render-only: the shared solver SDF never sees the fine particles.
        assert surface.render_sdf is not surface.sdf
        # A checkpoint keeps the fine particles.
        arrays, _scalars = surface.export_sheet_state()
        assert arrays["sheet_detail_pos"].shape == (surface.sheet.count, 3)
    finally:
        solver.release()


def test_sheet_detail_off_changes_nothing(cuda_device):
    solver = _solver_with_a_sheet(cuda_device)
    try:
        solver.step_frame()
        assert solver.surface.sheet is None
        assert "sheetDetail" not in solver.surface.last_sdf_splat
        assert solver.surface.export_sheet_state() == ({}, {})
    finally:
        solver.release()


def _sdf_pipeline(device, positions):
    """A real render grid and build_sdf, with a stand-in solver."""
    from fluidSim_warp_blender51.sim import surface as SURF
    from fluidSim_warp_blender51.sim.fields import ScalarField
    from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings

    settings = SolverSettings(domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
                              voxel_size=2 * H, device=device,
                              surface_sheet_detail_enabled=True,
                              # Filters off, so the test reads the raw splat.
                              surface_filter_iterations=0, surface_dilate_iterations=0,
                              surface_erode_iterations=0, surface_smooth_iterations=0)
    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), voxel_size=H,
                      leaf_capacity=64, device=device)
    pos, _vel, count, n = _particles(device, positions)
    grid.begin_topology()
    grid.mark_points(pos, count, n)
    grid.expand(1)
    grid.finalize_topology(sync=True)
    solver = SimpleNamespace(settings=settings, device=device, grid=object(),
                             particles=SimpleNamespace(count=count),
                             _np_dim=lambda: n)
    pipe = SURF.SurfacePipeline.__new__(SURF.SurfacePipeline)
    pipe.solver = solver
    pipe.sheet = None
    sdf, tmp = ScalarField(grid, name="sdf"), ScalarField(grid, name="tmp")
    return pipe, grid, sdf, tmp, pos


@wp.kernel
def _k_read_voxel(table: wp.array(dtype=wp.int32), vd: wp.vec3i, ld: wp.vec3i,
                  ijk: wp.vec3i, field: wp.array(dtype=float),
                  out: wp.array(dtype=float)):
    out[0] = field[voxel_flat_index(table, vd, ld, ijk)]


def _sdf_at(grid, sdf, point):
    ijk = wp.vec3i(*[int(np.floor(c / H)) for c in point])
    out = wp.zeros(1, dtype=float, device=grid.device)
    wp.launch(_k_read_voxel, dim=1, inputs=[grid.leaf_table, grid.voxels_dim, grid.leaves_dim, ijk,
                                sdf.data, out], device=grid.device)
    return float(out.numpy()[0])


def test_build_sdf_splats_fine_particles_at_their_own_radius(cuda_device):
    point = np.float32([[0.5 + H / 2, 0.5 + H / 2, 0.5 + H / 2]])  # a voxel centre
    pipe, grid, sdf, tmp, pos = _sdf_pipeline(cuda_device, point)
    nvox = max(1, grid.active_voxel_count)
    none = wp.zeros(1, dtype=wp.int32, device=cuda_device)
    fine_count = wp.array(np.array([1], np.int32), dtype=wp.int32, device=cuda_device)
    fine_r = SD.fine_radius(0.5 * 2 * H * float(pipe.solver.settings.particle_radius_scale), 8)
    inputs = {"coarse": (pos, none, 1), "fine": (pos, fine_count, 1), "radius": fine_r}
    pipe.build_sdf(grid, sdf, tmp, nvox, pos, H, render_detail=True, sheet_inputs=inputs)
    # A lone fine particle is a ball, never thinner than the render grid holds.
    from fluidSim_warp_blender51.sim.surface import (
        SHEET_FILM_MAX_STRETCH,
        SHEET_FILM_MIN_VOXELS,
    )
    film_r = max(fine_r, SHEET_FILM_MIN_VOXELS * H * SHEET_FILM_MAX_STRETCH ** (2 / 3))
    assert _sdf_at(grid, sdf, point[0]) == pytest.approx(-film_r, abs=1e-6)
    assert pipe.last_sdf_splat["sheetDetail"]["fineParticles"] == 1
    # The same particle as a coarse one splats at the coarse radius instead.
    inputs = {"coarse": (pos, fine_count, 1),
              "fine": (pos, none, 0), "radius": fine_r}
    pipe.build_sdf(grid, sdf, tmp, nvox, pos, H, render_detail=True, sheet_inputs=inputs)
    coarse_r = pipe.last_sdf_splat["particleRadius"]
    assert _sdf_at(grid, sdf, point[0]) == pytest.approx(-coarse_r, abs=1e-6)


def test_after_frame_steps_the_sheet_before_it_builds(cuda_device, monkeypatch):
    from fluidSim_warp_blender51.sim import surface as SURF

    settings = SimpleNamespace(surface_enabled=True, surface_sheet_detail_enabled=True,
                               surface_waves_enabled=False, surface_update_cadence=1,
                               surface_build_lowres_sdf=False)
    pipe = SURF.SurfacePipeline.__new__(SURF.SurfacePipeline)
    events = []
    pipe.solver = SimpleNamespace(settings=settings, _phase=lambda name: _Null(name))
    pipe.sheet = None
    monkeypatch.setattr(pipe, "step_sheet", lambda: events.append("sheet"), raising=False)
    monkeypatch.setattr(pipe, "build", lambda **kw: events.append("build"), raising=False)
    pipe.after_frame()
    assert events == ["sheet", "build"]
    # Switched off, the fine particles are dropped, not kept on the GPU.
    settings.surface_sheet_detail_enabled = False
    pipe.sheet = object()
    events.clear()
    pipe.after_frame()
    assert events == ["build"] and pipe.sheet is None


class _Null:
    def __init__(self, name):
        from fluidSim_warp_blender51.sim.constants import TIMER_PHASE_SLOTS

        assert name in TIMER_PHASE_SLOTS, f"unknown profiler phase: {name}"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_sheet_of_fine_particles_meshes_as_one_unbroken_film(cuda_device):
    # Fine particles a little over a render voxel apart, in one plane: every
    # render voxel on that plane inside the patch must be liquid.
    spacing = 1.2 * H
    axis = 0.3 + np.arange(12) * spacing
    z0 = 0.5 + H / 2
    film_values = []
    for z_plane in (z0, 0.5 + H):  # on a sample plane, and halfway between two
        film_values.append(_film_max(cuda_device, axis, spacing, z_plane))
    assert max(film_values) < 0.0


def _film_max(cuda_device, axis, spacing, z_plane):
    z0 = 0.5 + H / 2  # the sample plane nearest the film
    x, y = np.meshgrid(axis, axis, indexing="ij")
    film = np.stack([x.ravel(), y.ravel(), np.full(x.size, z_plane)], 1).astype(np.float32)
    pipe, grid, sdf, tmp, pos = _sdf_pipeline(cuda_device, film)
    nvox = max(1, grid.active_voxel_count)
    none = wp.zeros(1, dtype=wp.int32, device=cuda_device)
    count = wp.array(np.array([len(film)], np.int32), dtype=wp.int32, device=cuda_device)
    inputs = {"coarse": (pos, none, 1), "fine": (pos, count, len(film)),
              "radius": SD.fine_radius(0.5 * 2 * H, 8)}
    pipe.build_sdf(grid, sdf, tmp, nvox, pos, H, render_detail=True, sheet_inputs=inputs)
    inner = (axis[2:-2, None] + 0.5 * spacing)
    values = [_sdf_at(grid, sdf, (float(a), float(b), z0))
              for a in inner[:, 0] for b in inner[:, 0]]
    return max(values)
