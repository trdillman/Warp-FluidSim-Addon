"""Contracts for the nets_showcase scene: its layout, meshes and the 2x resolution claim."""

import glob
import json
import os
from collections import Counter

import numpy as np
import pytest

from tools.demos import nets_showcase as ns
from tools.demos import upres_cove as uc

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GIB = 1024 ** 3


def _edges_used_twice(mesh):
    counts = Counter()
    for face in mesh.faces:
        for a, b in zip(face, face[1:] + face[:1], strict=True):
            counts[(min(a, b), max(a, b))] += 1
    return all(n == 2 for n in counts.values())


@pytest.mark.parametrize("mesh", ns.collider_meshes(), ids=lambda m: m.name)
def test_collider_meshes_are_closed_and_outward(mesh):
    assert _edges_used_twice(mesh)
    assert ns.signed_volume(mesh) > 0.0


def test_collider_parts_are_at_least_three_hero_voxels_thick():
    p = ns.PRESETS[ns.HERO]
    assert ns.WALL_THICKNESS >= 3 * p.voxel
    assert ns.SLAB_TOP + ns.SLAB_BELOW >= 3 * p.voxel


def test_no_liquid_can_touch_a_domain_wall_at_rest():
    p = ns.PRESETS[ns.HERO]
    # The pool is held by the trough wall, well inside the domain sides.
    assert ns.DOMAIN_HX - (ns.POOL_HX + ns.WALL_THICKNESS) >= 30 * p.voxel
    assert ns.DOMAIN_HY - (ns.POOL_HY + ns.WALL_THICKNESS) >= 30 * p.voxel
    # It stands on the slab, not on the domain floor.
    assert ns.SLAB_TOP >= 10 * p.voxel
    for drop in ns.DROPS:
        assert ns.DOMAIN_HI[2] - drop.top >= 30 * p.voxel
        assert abs(drop.center_xy[0]) + drop.radius < ns.POOL_HX
        assert abs(drop.center_xy[1]) + drop.radius < ns.POOL_HY


def test_slab_covers_the_trough_but_not_the_whole_floor():
    slab = ns.slab_mesh().verts
    outer_x, outer_y = ns.POOL_HX + ns.WALL_THICKNESS, ns.POOL_HY + ns.WALL_THICKNESS
    assert slab[:, 0].max() > outer_x and slab[:, 0].min() < -outer_x
    assert slab[:, 1].max() > outer_y and slab[:, 1].min() < -outer_y
    assert slab[:, 2].min() < 0.0 < ns.SLAB_TOP
    # Inside the domain, so its bounding box does not book the whole floor.
    assert slab[:, 0].max() < ns.DOMAIN_HX and slab[:, 1].max() < ns.DOMAIN_HY


def _builder_source():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "demos",
                        "build_demo_scenes.py")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_trough_walls_are_taller_than_the_splash_and_the_front_is_glass():
    # GPU runs saw spray up to 0.28 m with low walls; the walls must clear it.
    assert ns.TROUGH_TOP_FRONT == ns.TROUGH_TOP_BACK == ns.TROUGH_TOP_END
    assert ns.TROUGH_TOP_FRONT >= 0.30
    assert ns.TROUGH_TOP_FRONT < ns.DOMAIN_HI[2] - 0.05
    names = [m.name for m in ns.trough_wall_meshes()]
    assert names == ["Nets Trough Front", "Nets Trough Back", "Nets Trough Left",
                     "Nets Trough Right"]
    # The camera-side wall is clear glass in the render, not hidden.
    assert ns.GLASS_WALLS == ("Nets Trough Front",)
    assert "hide_render" not in _builder_source().split("def build_nets_showcase")[0].split(
        "def author_nets_showcase")[1].split("make_temple_object")[1].split("add_camera_light")[0]


def test_trough_keeps_water_far_from_the_domain_walls():
    p = ns.PRESETS[ns.HERO]
    clearance = ns.DOMAIN_HX - (ns.POOL_HX + ns.WALL_THICKNESS)
    assert clearance >= 20 * p.voxel
    assert ns.DOMAIN_HY - (ns.POOL_HY + ns.WALL_THICKNESS) >= 20 * p.voxel
    assert ns.WALL_THICKNESS >= 3 * p.voxel


def test_inside_wall_check_counts_particles_in_the_wall_only():
    mid = 0.5 * (ns.SLAB_TOP + ns.TROUGH_TOP_BACK)
    in_wall = np.array([[0.0, ns.POOL_HY + 0.5 * ns.WALL_THICKNESS, mid]])
    above = np.array([[0.0, ns.POOL_HY + 0.5 * ns.WALL_THICKNESS, ns.TROUGH_TOP_BACK + 0.01]])
    in_pool = np.array([[0.0, 0.0, mid]])
    assert ns.inside_wall_fraction(in_wall) == 1.0
    assert ns.inside_wall_fraction(above) == 0.0
    assert ns.inside_wall_fraction(in_pool) == 0.0


def test_trough_wall_parts_do_not_overlap_inside_one_mesh():
    for sides, ends in (ns.wall_boxes(),):
        for group in (sides, ends):
            (lo0, hi0), (lo1, hi1) = group
            assert any(hi0[i] <= lo1[i] or hi1[i] <= lo0[i] for i in range(3))


def test_drops_are_big_enough_for_surface_nets_at_hero_resolution():
    for drop in ns.DROPS:
        assert ns.drop_voxels(drop, ns.PRESETS[ns.HERO]) >= ns.MIN_DROP_VOXELS


def test_impacts_are_gentle():
    for d in ns.DROPS:
        assert d.speed + ns.G * d.impact_time < 2.5  # m/s at the pool


def test_drops_land_one_after_another_inside_the_clip():
    frames = [ns.drop_fall_frame(d) for d in ns.DROPS]
    assert frames == sorted(frames)
    assert frames[1] - frames[0] >= 4 and frames[2] - frames[1] >= 3
    assert frames[-1] + 60 <= ns.FRAMES  # the splash gets room to play out


def test_drops_fall_where_the_launch_speed_says():
    for drop in ns.DROPS:
        t = drop.impact_time
        assert drop.speed * t + 0.5 * ns.G * t * t == pytest.approx(drop.gap)


def test_hero_is_at_least_twice_the_finest_demo_scene():
    f = ns.fidelity(ns.PRESETS[ns.HERO])
    assert f["longest_axis_cells"] == 600
    assert f["cells_per_axis_x"] >= 2.0
    assert f["domain_cells_x"] >= 8.0
    assert f["particles_x"] >= 2.0
    assert f["voxel_mm"] <= f["baseline_voxel_mm"] / 2.0


def test_baseline_matches_upres_cove():
    assert ns.BASELINE["voxel"] == uc.VOXEL
    nx, ny, nz = (round((hi - lo) / uc.VOXEL)
                  for lo, hi in zip(uc.DOMAIN_LO, uc.DOMAIN_HI, strict=True))
    assert max(nx, ny, nz) == ns.BASELINE["longest_axis_cells"]
    assert nx * ny * nz == pytest.approx(ns.BASELINE["domain_cells"], rel=0.02)
    assert uc.expected_particles() == pytest.approx(ns.BASELINE["particles"], rel=0.01)


def test_no_shipped_demo_beats_the_baseline():
    best_cells = 0
    for path in glob.glob(os.path.join(REPO, "demos", "*", "meta.json")):
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if "domain" not in meta or "voxel_size" not in meta:
            continue
        lo, hi = meta["domain"]
        longest = max(hi_i - lo_i for lo_i, hi_i in zip(lo, hi, strict=True))
        best_cells = max(best_cells, round(longest / meta["voxel_size"]))
    # capillary_droplet is a 10 cm bar at 2 mm; everything shipped is coarser than
    # the opt-in Upres Cove baseline.
    assert 0 < best_cells <= ns.BASELINE["longest_axis_cells"]


def test_hero_startup_demand_fits_the_reserve_and_a_16gb_card():
    p = ns.PRESETS[ns.HERO]
    # A source's startup demand is its whole volume (no narrow-band culling is
    # credited), so the reserve has to cover the full fill, with headroom.
    assert ns.expected_particles(p) * 1.15 < p.max_particles
    # Rough per-particle cost (position, velocity, APIC rows, bins) well under 200 B.
    assert p.max_particles * 200 < 2 * GIB


def test_pool_is_a_thin_film_so_the_full_fill_stays_affordable():
    p = ns.PRESETS[ns.HERO]
    pool_particles = ns.pool_volume() / p.voxel ** 3 * ns.PARTICLES_PER_VOXEL
    assert pool_particles < 6_000_000
    assert ns.POOL_DEPTH >= 4 * p.voxel  # still several voxels deep


def test_presets_do_not_depend_on_narrow_band():
    assert not hasattr(ns, "NARROW_BAND_VOXELS")


def test_surface_box_fits_max_dense_dim():
    for p in ns.PRESETS.values():
        # Padded to 32 by the extractor; the splash fills the domain at most.
        padded = -(-max(ns.divisions(p)) // 32) * 32
        assert p.max_dense_dim >= padded


def test_hero_fits_a_16gb_card_by_the_addons_own_estimate():
    pytest.importorskip("warp")
    import dataclasses

    from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes
    from tools.profiling import profile_gpu_solver as P

    p = ns.PRESETS[ns.HERO]
    import argparse

    args = argparse.Namespace(
        device="cpu", voxel_size=p.voxel, max_particles=p.max_particles,
        min_substeps=None, max_substeps=None, max_whitewater=0, no_surface=False,
        whitewater=False, scene_preset="dam_break", no_whitewater=True,
        viscosity=False, surface_tension=False)
    settings = dataclasses.replace(
        P._build_settings(args), domain_min=ns.DOMAIN_LO, domain_max=ns.DOMAIN_HI,
        leaf_capacity=p.leaf_capacity, narrow_band_enabled=False,
        surface_max_dense_dim=p.max_dense_dim)
    assert estimate_solver_bytes(settings) < 12 * GIB


def test_draft_is_the_same_scene_at_half_resolution():
    hero, draft = ns.PRESETS[ns.HERO], ns.PRESETS[ns.DRAFT]
    assert draft.voxel == pytest.approx(2 * hero.voxel)
    assert ns.expected_particles(draft) == pytest.approx(ns.expected_particles(hero) / 8, rel=1e-6)


def test_preset_lookup(monkeypatch):
    monkeypatch.delenv(ns.PRESET_ENV, raising=False)
    assert ns.preset().name == ns.HERO
    monkeypatch.setenv(ns.PRESET_ENV, "Draft")
    assert ns.preset().name == ns.DRAFT
    monkeypatch.setenv(ns.PRESET_ENV, "huge")
    with pytest.raises(ValueError):
        ns.preset()


def test_camera_frames_every_drop():
    cam = np.array(ns.CAMERA_LOCATION)
    forward = np.array(ns.CAMERA_TARGET) - cam
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    half_h = 18.0 / ns.CAMERA_LENS   # 36 mm sensor, horizontal
    half_v = half_h * 9 / 16
    for drop in ns.DROPS:
        for point in (drop.center, (drop.center[0], drop.center[1], drop.top)):
            rel = np.array(point) - cam
            depth = rel @ forward
            assert depth > 0
            assert abs(rel @ right) / depth < half_h
            assert abs(rel @ up) / depth < half_v


def test_particle_checks():
    inside = np.array([[0.0, 0.0, 0.10], [0.1, 0.0, 0.12]])
    assert ns.wall_clearance(inside) == pytest.approx(ns.DOMAIN_HY)
    assert ns.inside_wall_fraction(inside) == 0.0
    in_wall = np.array([[ns.POOL_HX + ns.WALL_THICKNESS / 2, 0.0, ns.POOL_TOP]])
    assert ns.inside_wall_fraction(in_wall) == 1.0
    near_side = np.array([[ns.DOMAIN_HX - 0.005, 0.0, 0.2]])
    assert ns.wall_clearance(near_side) == pytest.approx(0.005)
    swarm = np.concatenate([np.zeros((999, 3)) + [0, 0, 0.1], near_side])
    assert ns.wall_contact_fraction(swarm, 0.01) == pytest.approx(1 / 1000)
    assert ns.wall_contact_fraction(inside, 0.01) == 0.0


def test_builder_and_renderer_are_wired_to_the_shared_numbers():
    with open(os.path.join(REPO, "tools", "demos", "build_demo_scenes.py"), encoding="utf-8") as fh:
        builder = fh.read()
    assert '"nets_showcase": build_nets_showcase' in builder
    # The extractor reads the backend when the solver is created.
    assert builder.index("os.environ[ns.BACKEND_ENV] = ns.BACKEND") < builder.index(
        "scene = author_nets_showcase()")
    with open(os.path.join(REPO, "tools", "demos", "render_nets_showcase.py"), encoding="utf-8") as fh:
        renderer = fh.read()
    assert "render_extractor.backend" in renderer
    # A paused or failed step must stop the capture, not save the old surface again.
    assert "sess.last_frame != frame" in renderer
    # Meshes from an earlier run must not reach the render.
    assert 'os.remove(os.path.join(mesh_dir, name))' in renderer
    assert 'json.load(fh)["frames"]' in renderer
    assert "FLUIDSIM_SURFACE_BACKEND" in renderer or "ns.BACKEND_ENV" in renderer


def test_pool_tiles_cover_the_pool_once_and_keep_each_source_small():
    for p in ns.PRESETS.values():
        tiles = ns.pool_tiles(p)
        v = p.voxel
        volume = sum(np.prod(np.array(hi) - np.array(lo)) for lo, hi in tiles)
        inner = (2 * ns.POOL_HX - 2 * v) * (2 * ns.POOL_HY - 2 * v) * ns.POOL_DEPTH
        assert volume == pytest.approx(inner, rel=1e-6)
        for lo, hi in tiles:
            size = np.array(hi) - np.array(lo)
            # The planner books a cube of the longest side: keep it small.
            assert size.max() <= ns.MAX_TILE + 1e-9
            # Tile edges sit on voxel faces (no voxel centre on a shared edge).
            for value in (lo[0] - (-ns.DOMAIN_HX), lo[1] - (-ns.DOMAIN_HY),
                          hi[0] - (-ns.DOMAIN_HX), hi[1] - (-ns.DOMAIN_HY)):
                assert abs(value / v - round(value / v)) < 1e-6


def test_build_stops_when_initialize_is_refused():
    src = _builder_source()
    assert 'bpy.ops.fluidsim.initialize() != {"FINISHED"}' in src


def test_authoring_does_not_start_a_solver_before_the_build_initializes():
    src = _builder_source()
    author = src.split("def author_nets_showcase")[1].split("def build_nets_showcase")[0]
    assert "frame_set(" not in author and "fs.live = True" not in author
    assert "bpy.ops.fluidsim.release()" in src.split("def build_nets_showcase")[1]


def test_playback_save_rebuilds_the_solver_instead_of_resetting_it():
    # A Reset needs the old and the new solver in memory together.
    assert "fresh_solver=True" in _builder_source().split("def build_nets_showcase")[1]
