"""Surface waves: fine wave detail on a coarse liquid surface (Mercier et al. 2015).

Run on CPU with ``FLUIDSIM_TEST_DEVICE=cpu``; on a GPU host they use CUDA.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import surface_waves as SW

DT = 1.0 / 24.0


def _plane(n=41, size=0.8, bump=0.0, width=0.03):
    xs = np.linspace(0.0, size, n)
    x, y = np.meshgrid(xs, xs)
    z = bump * np.exp(-((x - size / 2) ** 2 + (y - size / 2) ** 2) / width ** 2)
    verts = np.stack([x.ravel(), y.ravel(), z.ravel()], 1).astype(np.float32)
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            tris += [a, a + 1, a + n, a + 1, a + n + 1, a + n]
    return verts, np.array(tris, np.int32)


def _waves(device, **kw):
    kw.setdefault("spacing", 0.01)
    kw.setdefault("coarse_voxel", 0.02)
    return SW.SurfaceWaves(SW.SurfaceWaveSettings(**kw), device=device)


# --- host maths ---------------------------------------------------------------

def test_wave_speed_follows_capillary_gravity_dispersion():
    # Long waves are gravity waves: c = sqrt(g * lambda / 2 pi).
    assert SW.wave_speed(10.0) == pytest.approx(np.sqrt(9.81 * 10.0 / (2 * np.pi)),
                                                rel=1e-3)
    # Short ripples are capillary and speed up again below ~1.7 cm.
    assert SW.wave_speed(0.004) > SW.wave_speed(0.017) < SW.wave_speed(0.1)


def test_substeps_keep_the_explicit_wave_step_stable():
    assert SW.substep_count(0.0, 0.01, 0.3) == 0
    assert SW.substep_count(DT, 0.01, 0.0) == 1
    n = SW.substep_count(DT, 0.01, 0.3)
    # The stiffest mode: two points half a spacing apart, swinging apart.
    omega_max = np.sqrt(32.0) * 0.3 / 0.01
    assert omega_max * DT / n <= 1.0 + 1e-12
    assert SW.substep_count(DT, 1.0, 0.0, damping=20.0) * 0.5 >= 20.0 * DT


def test_two_close_points_alone_do_not_blow_up(cuda_device):
    # The worst case for the explicit step: one isolated pair at the
    # thinning distance, with a tiny opposite perturbation.
    spacing = 0.01
    waves = _waves(cuda_device, spacing=spacing, strength=0.0)
    dev = waves.device
    pos = wp.array(np.array([[0, 0, 0], [0.51 * spacing, 0, 0]], np.float32),
                   dtype=wp.vec3, device=dev)
    nrm = wp.array(np.array([[0, 0, 1], [0, 0, 1]], np.float32), dtype=wp.vec3,
                   device=dev)
    h = wp.array(np.array([1e-6, -1e-6], np.float32), dtype=float, device=dev)
    hv = wp.zeros(2, dtype=float, device=dev)
    zeros = wp.zeros(2, dtype=float, device=dev)
    h_tmp, hv_tmp = wp.empty_like(h), wp.empty_like(hv)
    grid = waves._grid(pos, 2)
    steps = SW.substep_count(DT, spacing, waves.speed, 1.5)
    sub_dt = DT / steps
    for _ in range(24 * steps):
        wp.launch(SW.k_wave_step, dim=2,
                  inputs=[grid.id, pos, nrm, zeros, zeros, h, hv, waves.radius,
                          waves.speed ** 2, 1.5, 0.0, 1e9, 0.0, 0.0, sub_dt, 1.0,
                          (SW.MIN_SEPARATION * spacing) ** 2, h_tmp, hv_tmp],
                  device=dev)
        h, h_tmp = h_tmp, h
        hv, hv_tmp = hv_tmp, hv
    assert np.abs(h.numpy()).max() <= 1e-6


def test_candidates_never_exceed_the_point_budget(cuda_device, monkeypatch):
    verts, tris = _plane()
    sizes = []
    real_empty = SW.wp.empty

    def spy(n, *a, **kw):
        if kw.get("dtype") is SW.wp.vec3:
            sizes.append(int(n))
        return real_empty(n, *a, **kw)

    waves = _waves(cuda_device, spacing=0.002, max_points=500)
    monkeypatch.setattr(SW.wp, "empty", spy)
    waves.step(verts, tris, np.zeros_like(verts), DT)
    monkeypatch.undo()
    assert max(sizes) <= 500 and 0 < waves.count <= 500


# --- the frame step -------------------------------------------------------------

def test_a_flat_surface_gets_no_waves(cuda_device):
    verts, tris = _plane()
    waves = _waves(cuda_device)
    for _ in range(4):
        out = waves.step(verts, tris, np.zeros_like(verts), DT)
    np.testing.assert_allclose(out, verts, atol=1e-6)
    # The layer is about one point per spacing square, never denser than
    # the half-spacing thinning allows.
    area = 0.8 * 0.8
    assert 0.5 * area / 0.01 ** 2 < waves.count < 4.0 * area / (0.5 * 0.01) ** 2


def test_a_bend_sharper_than_the_grid_seeds_bounded_waves(cuda_device):
    verts, tris = _plane(bump=0.03, width=0.02)
    waves = _waves(cuda_device, strength=4.0)
    for _ in range(12):
        out = waves.step(verts, tris, np.zeros_like(verts), DT)
    moved = out - verts
    assert np.abs(moved).max() > 1e-4
    # Displacement is along the surface normal: on the flat rim, straight up.
    rim = np.hypot(verts[:, 0] - 0.4, verts[:, 1] - 0.4) > 0.3
    np.testing.assert_allclose(moved[rim, :2], 0.0, atol=1e-6)
    # And it never exceeds the amplitude cap (half a coarse voxel).
    assert np.abs(waves.heights()).max() <= 0.5 * 0.02 + 1e-7


def test_a_resolved_bend_seeds_nothing(cuda_device):
    # A bump many coarse voxels wide is detail the coarse sim already has.
    verts, tris = _plane(bump=0.02, width=0.3)
    waves = _waves(cuda_device)
    for _ in range(4):
        out = waves.step(verts, tris, np.zeros_like(verts), DT)
    np.testing.assert_allclose(out, verts, atol=1e-6)


def test_the_far_side_of_a_thin_sheet_is_not_a_bend(cuda_device):
    # Two flat faces 1 cm apart, facing away from each other: each is flat,
    # so neither may read the other as curvature and seed waves.
    top, tris = _plane()
    bottom = top - np.array([0.0, 0.0, 0.01], np.float32)
    flipped = tris.reshape(-1, 3)[:, ::-1].reshape(-1)
    verts = np.concatenate([top, bottom])
    both = np.concatenate([tris, flipped + len(top)]).astype(np.int32)
    waves = _waves(cuda_device, strength=4.0)
    for _ in range(4):
        out = waves.step(verts, both, np.zeros_like(verts), DT)
    np.testing.assert_allclose(out, verts, atol=1e-6)


def test_a_surface_far_beyond_the_budget_does_not_overflow(cuda_device):
    # 100 m square at 0.1 mm spacing asks for 10^12 points: far past int32.
    verts, tris = _plane(n=3, size=100.0)
    waves = _waves(cuda_device, spacing=1e-4, max_points=1000)
    waves.step(verts, tris, np.zeros_like(verts), DT)
    assert 0 < waves.count <= 1000


def test_points_ride_the_surface_and_keep_their_waves(cuda_device):
    verts, tris = _plane()
    waves = _waves(cuda_device)
    vel = np.zeros_like(verts)
    vel[:, 0] = 0.24  # one centimetre per frame
    waves.step(verts, tris, vel, DT)
    first = waves.count
    waves.h.assign(np.full(waves.count, 1e-3, np.float32))
    shift = np.array([0.24 * DT, 0.0, 0.0], np.float32)
    waves.step(verts + shift, tris, vel, DT)
    # Nearly everything was carried along; only the new leading edge is new.
    assert waves.stats["kept"] > 0.9 * first
    assert waves.stats["added"] < 0.1 * first
    # Carried points keep their height (minus one frame of damping).
    kept = waves.heights()[: waves.stats["kept"]]
    assert np.median(kept) > 0.8e-3


def test_points_die_with_the_surface(cuda_device):
    verts, tris = _plane()
    waves = _waves(cuda_device)
    waves.step(verts, tris, np.zeros_like(verts), DT)
    out = waves.step(np.zeros((0, 3), np.float32), np.zeros(0, np.int32),
                     np.zeros((0, 3), np.float32), DT)
    assert len(out) == 0 and waves.count == 0
    # A surface that jumped far away is reseeded, not dragged along.
    far = verts + np.array([0.0, 0.0, 1.0], np.float32)
    waves.step(verts, tris, np.zeros_like(verts), DT)
    waves.step(far, tris, np.zeros_like(verts), DT)
    assert waves.stats["kept"] == 0 and waves.stats["added"] > 0


def test_the_point_budget_is_a_hard_cap(cuda_device):
    verts, tris = _plane()
    waves = _waves(cuda_device, max_points=500)
    waves.step(verts, tris, np.zeros_like(verts), DT)
    assert 0 < waves.count <= 500


def test_a_bump_of_wave_height_spreads_and_decays(cuda_device):
    verts, tris = _plane()
    waves = _waves(cuda_device, damping=3.0)
    waves.step(verts, tris, np.zeros_like(verts), DT)
    pos = waves.pos.numpy()[: waves.count]
    r = np.hypot(pos[:, 0] - 0.4, pos[:, 1] - 0.4)
    h0 = np.where(r < 0.02, 4e-3, 0.0).astype(np.float32)
    waves.h.assign(h0)
    for _ in range(3):
        waves.step(verts, tris, np.zeros_like(verts), DT)
    pos = waves.pos.numpy()[: waves.count]
    h = waves.heights()
    r = np.hypot(pos[:, 0] - 0.4, pos[:, 1] - 0.4)
    # The disturbance left the centre and reached the ring around it.
    assert np.abs(h[r < 0.02]).max() < 4e-3
    assert np.abs(h[(r > 0.03) & (r < 0.15)]).max() > 1e-5
    # Damping takes energy out (the potential part of it, sum of h squared).
    assert np.square(h).sum() < np.square(h0).sum()


def test_the_same_inputs_give_the_same_waves(cuda_device):
    verts, tris = _plane(bump=0.03, width=0.02)
    outs = []
    for _ in range(2):
        waves = _waves(cuda_device, strength=4.0)
        for _ in range(3):
            out = waves.step(verts, tris, np.zeros_like(verts), DT)
        outs.append(out)
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-6)


def test_wave_state_survives_a_checkpoint_round_trip(cuda_device):
    verts, tris = _plane(bump=0.03, width=0.02)
    a = _waves(cuda_device, strength=4.0)
    b = _waves(cuda_device, strength=4.0)
    for _ in range(3):
        a.step(verts, tris, np.zeros_like(verts), DT)
    arrays, scalars = a.export_state()
    b.import_state(arrays, scalars)
    assert b.count == a.count and b.frame == 3
    out_a = a.step(verts, tris, np.zeros_like(verts), DT)
    out_b = b.step(verts, tris, np.zeros_like(verts), DT)
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)


def test_reset_forgets_the_waves(cuda_device):
    verts, tris = _plane(bump=0.03, width=0.02)
    waves = _waves(cuda_device, strength=4.0)
    for _ in range(3):
        waves.step(verts, tris, np.zeros_like(verts), DT)
    waves.reset()
    assert waves.count == 0 and waves.frame == 0 and waves.time == 0.0


# --- the surface pipeline hook ------------------------------------------------

def test_the_pipeline_steps_waves_once_per_frame_and_serves_that_mesh(
        cuda_device, monkeypatch):
    from fluidSim_warp_blender51.sim import surface as SURF

    verts, tris = _plane(bump=0.03, width=0.02)
    settings = SimpleNamespace(
        surface_enabled=True, surface_waves_enabled=True, surface_waves_detail=0.5,
        surface_waves_strength=4.0, surface_waves_damping=1.5,
        surface_update_cadence=0, time_scale=1.0, frame_rate=24.0)
    solver = SimpleNamespace(settings=settings, device=cuda_device,
                             _phase=lambda name: _Null())
    pipe = SURF.SurfacePipeline.__new__(SURF.SurfacePipeline)
    pipe.solver = solver
    pipe.waves = None
    pipe.render_sdf = object()
    pipe.built_key = ("k",)
    calls = []
    pipe.render_extractor = SimpleNamespace(
        extract=lambda sdf: (calls.append(sdf), (verts, tris))[1])
    monkeypatch.setattr(pipe, "voxel_size", lambda: 0.02, raising=False)
    monkeypatch.setattr(pipe, "build_key", lambda: ("k",), raising=False)
    monkeypatch.setattr(pipe, "_vertex_velocity", lambda v: np.zeros_like(v),
                        raising=False)
    for _ in range(3):
        pipe.after_frame()
    assert len(calls) == 3 and pipe.waves.frame == 3
    assert pipe.waves.settings.spacing == pytest.approx(0.01)
    assert pipe.mesh_key == ("k",)
    out, out_tris = pipe.mesh_cache
    assert out_tris is tris and out.shape == verts.shape
    assert np.abs(out - verts).max() > 0.0


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_waves_build_the_surface_first_when_the_cadence_did_not(
        cuda_device, monkeypatch):
    from fluidSim_warp_blender51.sim import surface as SURF

    verts, tris = _plane()
    settings = SimpleNamespace(
        surface_enabled=True, surface_waves_enabled=True, surface_waves_detail=0.5,
        surface_waves_strength=1.0, surface_waves_damping=1.5,
        surface_update_cadence=2, time_scale=1.0, frame_rate=24.0)
    solver = SimpleNamespace(settings=settings, device=cuda_device,
                             _phase=lambda name: _Null())
    pipe = SURF.SurfacePipeline.__new__(SURF.SurfacePipeline)
    pipe.solver = solver
    pipe.waves = None
    pipe.render_sdf = object()
    pipe.built_key = None
    events = []

    def build(build_render=True, build_lowres=None):
        events.append("build")
        pipe.built_key = ("k",)
        return True

    pipe.render_extractor = SimpleNamespace(
        extract=lambda sdf: (events.append("extract"), (verts, tris))[1])
    monkeypatch.setattr(pipe, "build", build, raising=False)
    monkeypatch.setattr(pipe, "voxel_size", lambda: 0.02, raising=False)
    monkeypatch.setattr(pipe, "build_key", lambda: ("k",), raising=False)
    monkeypatch.setattr(pipe, "_vertex_velocity", lambda v: np.zeros_like(v),
                        raising=False)
    pipe.step_waves()
    pipe.step_waves()
    assert events == ["build", "extract", "extract"]
    assert pipe.mesh_key == pipe.built_key == ("k",)


def test_a_rebuilt_surface_keeps_the_current_waves(cuda_device, monkeypatch):
    from fluidSim_warp_blender51.sim import surface as SURF

    verts, tris = _plane(bump=0.03, width=0.02)
    settings = SimpleNamespace(
        surface_enabled=True, surface_waves_enabled=True, surface_waves_detail=0.5,
        surface_waves_strength=4.0, surface_waves_damping=1.5,
        surface_update_cadence=1, time_scale=1.0, frame_rate=24.0)
    solver = SimpleNamespace(settings=settings, device=cuda_device,
                             _phase=lambda name: _Null())
    pipe = SURF.SurfacePipeline.__new__(SURF.SurfacePipeline)
    pipe.solver = solver
    pipe.waves = None
    pipe.render_sdf = object()
    pipe.built_key = ("k",)
    pipe.render_extractor = SimpleNamespace(extract=lambda sdf: (verts, tris))
    monkeypatch.setattr(pipe, "voxel_size", lambda: 0.02, raising=False)
    monkeypatch.setattr(pipe, "build_key", lambda: ("k",), raising=False)
    monkeypatch.setattr(pipe, "_vertex_velocity", lambda v: np.zeros_like(v),
                        raising=False)
    for _ in range(3):
        stepped, _ = pipe.step_waves()
    frame, heights = pipe.waves.frame, pipe.waves.heights().copy()
    pipe.mesh_cache = None  # what a paused rebuild does
    again, again_tris = pipe.mesh()
    assert again_tris is tris and pipe.waves.frame == frame
    np.testing.assert_array_equal(pipe.waves.heights(), heights)
    np.testing.assert_allclose(again, stepped, atol=1e-6)
