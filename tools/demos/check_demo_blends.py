"""Open generated demo .blend files in Blender and verify saved scene contents.

Run with Blender:
    blender --background --factory-startup --python tools/demos/check_demo_blends.py
"""

import json
import math
import os
import sys

import bpy
import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

DEMOS = os.path.join(REPO, "demos")
SCENES = ("dam_break", "crown_splash", "chromatic_vortex", "paddle_stir",
          "buoyancy_ladder", "water_wheel",
          "viscosity_race", "deep_plunge")
EXPECTED_AUTO_SEEDS = {
    "crown_splash": {"Crown Pool Seed", "Crown Droplet Seed"},
    "paddle_stir": {"Paddle Stir Pool Seed"},
}

# An enabled effect whose own coefficients zero it out is a UI-only claim
# (DEMO_SCENE_CONTRACT.md, "Enabled effects are reachable"). Each entry is a
# necessary condition for the effect to change the solve at all.
EFFECT_PRECONDITIONS = {
    "whitewater_enabled": (
        ("ww_churn_gain > 0", lambda s: s.ww_churn_gain > 0.0),
        ("ww_emission_rate > 0", lambda s: s.ww_emission_rate > 0.0),
    ),
    "viscosity_enabled": (
        ("viscosity_from_particles or viscosity_constant > 0",
         lambda s: s.viscosity_from_particles or s.viscosity_constant > 0.0),
    ),
    "surface_tension_enabled": (
        ("surface_tension_coefficient > 0",
         lambda s: s.surface_tension_coefficient > 0.0),
    ),
    "species_enabled": (
        ("species_mixing_rate > 0", lambda s: s.species_mixing_rate > 0.0),
    ),
}
# Whitewater also has a runtime-only gate: a candidate emits only where churn
# reaches ww_churn_threshold, and churn depends on the flow. No static bound
# exists, so every whitewater demo declares the timeline window (frames after
# frame_start) within which live whitewater must appear on a fresh reopen.
# Each value is the measured first-emission frame plus margin.
WHITEWATER_ACTIVATION_FRAMES = {
    "dam_break": 24,          # frame 11 on reopen
    "paddle_stir": 32,        # frame 15 (needs the keyframed paddle moving)
    "crown_splash": 12,       # frame 3 on reopen (4-5 in the build run)
}


def open_demo_blend(path):
    """Open ``path`` as an add-on session loads it, and refuse a pre-1.1 file.

    A demo saved before FluidSim 1.1 would open with default domain settings,
    so the check would measure the wrong scene. The checker never saves, so
    the demo on disk is untouched.
    """
    bpy.ops.wm.open_mainfile(filepath=path)
    enable_repo_addon(REPO, default_set=False, persistent=False)
    # Import only after enable_repo_addon, which re-imports the add-on.
    from fluidSim_warp_blender51.blender import file_compat

    stale = [scene.name for scene in bpy.data.scenes
             if file_compat.is_pre_bridge_scene(scene)]
    if stale:
        raise RuntimeError(f"{path}: saved before FluidSim 1.1 (scenes {stale}); "
                           "rebuild it with tools/demos/build_demo_scenes.py")


def effect_precondition_failures(settings):
    """Map each enabled effect to the preconditions its own settings violate."""
    failures = {}
    for flag, preconditions in EFFECT_PRECONDITIONS.items():
        if getattr(settings, flag, False):
            failed = [label for label, holds in preconditions if not holds(settings)]
            if failed:
                failures[flag] = failed
    return failures


def check_whitewater_reachable(name):
    """Reopen ``name`` and prove its enabled whitewater emits during playback.

    Drives the saved live timeline through ``scene.frame_set`` -- the
    production frame handler, with keyframed colliders evaluated -- and reads
    the live count from ``frame_stats``. The ``FluidSim Whitewater`` point
    cloud keeps a spare-capacity floor, so its point count proves nothing.
    """
    open_demo_blend(os.path.join(DEMOS, name + ".blend"))
    # Import only after enable_repo_addon: it re-imports the add-on, so an
    # earlier module reference would hold a dead session the handler never uses.
    from fluidSim_warp_blender51.blender import scene_bridge

    scene = bpy.context.scene
    settings = scene_bridge.active_domain_settings(scene)
    window = WHITEWATER_ACTIVATION_FRAMES.get(name)
    report = {
        "window_frames": window,
        "churn_threshold": float(settings.ww_churn_threshold),
        "emission_rate": float(settings.ww_emission_rate),
        "peak_churn": 0.0,
        "first_live_frame": None,
        "peak_live_count": 0,
        "liquid_count_at_peak": 0,
        "max_dropped": 0,
    }
    if window is None:
        report["error"] = ("whitewater_enabled but no WHITEWATER_ACTIVATION_FRAMES "
                           "entry declares when it must emit")
        return False, report
    if not scene.fluidsim.live:
        report["error"] = "saved scene is not in Live mode"
        return False, report
    applied_ok = False
    try:
        for frame in range(scene.frame_start + 1, scene.frame_start + window + 1):
            scene.frame_set(frame)
            sess = scene_bridge.session()
            if not (sess.active and sess.frame_stats):
                report["error"] = f"solver inactive at frame {frame}"
                break
            report["peak_churn"] = max(
                report["peak_churn"], float(sess.solver.churn.data.numpy().max()))
            stats = sess.frame_stats[-1]
            whitewater_count = int(stats.get("whitewaterParticles") or 0)
            liquid_count = int(stats.get("activeParticles") or 0)
            report["max_dropped"] = max(
                report["max_dropped"],
                int(stats.get("whitewaterDropped") or 0),
            )
            if whitewater_count > report["peak_live_count"]:
                report["peak_live_count"] = whitewater_count
                report["liquid_count_at_peak"] = liquid_count
            if (report["first_live_frame"] is None
                    and whitewater_count > 0):
                report["first_live_frame"] = frame
        sess = scene_bridge.session()
        if sess.active:
            applied = sess.solver.settings
            applied_ok = bool(
                applied.whitewater_enabled
                and abs(applied.ww_churn_threshold - settings.ww_churn_threshold) < 1.0e-6
                and abs(applied.ww_emission_rate - settings.ww_emission_rate) < 1.0e-6
                and abs(applied.ww_min_speed - settings.ww_min_speed) < 1.0e-6
                and abs(applied.churn_gain - settings.ww_churn_gain) < 1.0e-6)
            report["applied_matches_authored"] = applied_ok
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)
    finally:
        scene.fluidsim.live = False
        bpy.ops.fluidsim.release()
    return bool(
        applied_ok
        and report["first_live_frame"] is not None
        and report["max_dropped"] == 0
    ), report


CORK_NAME = "Crown Cork"
CORK_LIVE_FRAMES = 12   # live frames stepped before the pose track is read
CORK_BAKE_FRAMES = 6    # baked frames exported to Alembic


def _cork_translation(obj):
    return np.asarray(obj.matrix_world, dtype=np.float64)[:3, 3].copy()


def check_cork_motion(name):
    """Reopen ``name``; prove the Dynamic cork's pose track moves and that an
    Alembic export of a baked cache carries it (4.0 two-way rigid coupling).
    """
    import shutil
    import tempfile

    open_demo_blend(os.path.join(DEMOS, name + ".blend"))
    from fluidSim_warp_blender51.blender import playback, scene_bridge
    from fluidSim_warp_blender51.sim import rigid_scene

    scene = bpy.context.scene
    fs = scene.fluidsim
    report = {"live_track_range": 0.0, "alembic_range": 0.0,
              "alembic_max_error": None}
    cork = bpy.data.objects.get(CORK_NAME)
    contract = bool(
        cork and cork.fluidsim.role == "COLLIDER" and cork.fluidsim.enabled
        and cork.fluidsim.collider_dynamic
        and abs(float(cork.fluidsim.collider_density) - 300.0) < 1.0e-6)
    checks = {"cork_contract": contract, "cork_pose_track_moves": False,
              "alembic_carries_cork": False}
    if not contract:
        return checks, report
    work = tempfile.mkdtemp(prefix="fluidsim_cork_check_")
    try:
        start = scene.frame_start
        last = start + CORK_LIVE_FRAMES
        for frame in range(start + 1, last + 1):
            scene.frame_set(frame)
        fs.live = False
        bpy.ops.fluidsim.release()
        action = getattr(cork.animation_data, "action", None)
        owned = bool(action is not None and action.get(rigid_scene.OWNED_ACTION_TAG))
        # Replay the published keyframes with no solver attached.
        track = []
        for frame in range(start + 1, last + 1):
            scene.frame_set(frame)
            track.append(_cork_translation(cork))
        track = np.asarray(track)
        report["live_track_range"] = float(np.ptp(track, axis=0).max())
        report["owned_action"] = owned
        checks["cork_pose_track_moves"] = bool(
            owned and report["live_track_range"] > 1.0e-4)

        fs.cache_path = os.path.join(work, "cache")
        fs.cache_enabled = False
        fs.cache_use_custom_range = True
        fs.cache_frame_start = start
        fs.cache_frame_end = start + CORK_BAKE_FRAMES - 1
        scene.frame_set(start)
        from fluidSim_warp_blender51.blender import cache_bake

        cache_bake.bake(scene, scene_bridge.session(), start,
                        start + CORK_BAKE_FRAMES - 1)
        playback.set_cache_playback_enabled(scene, True)
        fs.cache_playback_enabled = True
        abc = os.path.join(work, "cork.abc")
        result = bpy.ops.fluidsim.export_alembic(filepath=abc)
        report["export_result"] = sorted(result)
        expected = {}
        for frame in range(start, start + CORK_BAKE_FRAMES):
            scene.frame_set(frame)
            expected[frame] = _cork_translation(cork)
        fresh = bpy.data.scenes.new("CorkAlembicCheck")
        bpy.context.window.scene = fresh
        before = set(fresh.objects)
        bpy.ops.wm.alembic_import(filepath=abc, as_background_job=False)
        added = [o for o in fresh.objects if o not in before]
        report["imported_all"] = sorted(o.name for o in added)
        # Alembic object names replace spaces with underscores.
        imported = [o for o in added
                    if o.name.replace("_", " ").startswith(CORK_NAME)]
        report["imported"] = [o.name for o in imported]
        if len(imported) == 1:
            carried, errors = [], []
            for frame in sorted(expected):
                fresh.frame_set(frame)
                depsgraph = bpy.context.evaluated_depsgraph_get()
                world = np.asarray(
                    imported[0].evaluated_get(depsgraph).matrix_world)[:3, 3]
                carried.append(world)
                errors.append(float(np.abs(world - expected[frame]).max()))
            report["alembic_range"] = float(np.ptp(np.asarray(carried), axis=0).max())
            report["alembic_max_error"] = max(errors)
            checks["alembic_carries_cork"] = bool(
                "FINISHED" in result and report["alembic_range"] > 1.0e-5
                and report["alembic_max_error"] < 1.0e-4)
        bpy.context.window.scene = scene
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)
    finally:
        fs.live = False
        if bpy.ops.fluidsim.release.poll():
            bpy.ops.fluidsim.release()
        shutil.rmtree(work, ignore_errors=True)
    return checks, report


LADDER_DENSITIES = (250.0, 500.0, 750.0, 1500.0)
LADDER_HALF_HEIGHT = 0.03   # 6 cm blocks, origin at the centre
LADDER_LIVE_FRAMES = 72     # drafts settle by ~frame 48
WHEEL_NAME = "Water Wheel"
WHEEL_LOCK = (True, True, True, True, False, True)
WHEEL_LIVE_FRAMES = 24


def _play_live_then_replay(scene, bodies, frames):
    """Play ``frames`` live frames, release the solver, then replay the
    published keyframes with no solver attached.

    Returns ``(owned, matrices)``: whether every body's action carries the
    FluidSim ownership tag, and each body's replayed world matrices.
    """
    from fluidSim_warp_blender51.sim import rigid_scene

    fs = scene.fluidsim
    start = scene.frame_start
    for frame in range(start + 1, start + frames + 1):
        scene.frame_set(frame)
    fs.live = False
    bpy.ops.fluidsim.release()
    owned = all(
        obj.animation_data is not None and obj.animation_data.action is not None
        and obj.animation_data.action.get(rigid_scene.OWNED_ACTION_TAG)
        for obj in bodies)
    matrices = {obj.name: [] for obj in bodies}
    for frame in range(start, start + frames + 1):
        scene.frame_set(frame)
        for obj in bodies:
            matrices[obj.name].append(np.asarray(obj.matrix_world, dtype=np.float64))
    return owned, {key: np.asarray(value) for key, value in matrices.items()}


def _release(scene):
    scene.fluidsim.live = False
    if bpy.ops.fluidsim.release.poll():
        bpy.ops.fluidsim.release()


def check_buoyancy_ladder(name):
    """Every block is a Dynamic collider of its labelled density; live play
    floats the three lighter blocks in density order and sinks the heaviest."""
    open_demo_blend(os.path.join(DEMOS, name + ".blend"))
    scene = bpy.context.scene
    blocks = [bpy.data.objects.get(f"Ladder Block {int(d)}") for d in LADDER_DENSITIES]
    contract = all(
        obj is not None and obj.fluidsim.role == "COLLIDER" and obj.fluidsim.enabled
        and obj.fluidsim.collider_dynamic
        and abs(float(obj.fluidsim.collider_density) - density) < 1.0e-6
        for obj, density in zip(blocks, LADDER_DENSITIES))
    checks = {"ladder_contract": contract, "ladder_pose_tracks_owned": False,
              "ladder_floats_in_density_order": False,
              "ladder_heaviest_sinks": False}
    report = {}
    if not contract:
        return checks, report
    try:
        owned, matrices = _play_live_then_replay(scene, blocks, LADDER_LIVE_FRAMES)
        bottoms = [float(matrices[obj.name][-1][2, 3]) - LADDER_HALF_HEIGHT
                   for obj in blocks]
        travel = [float(np.ptp(matrices[obj.name][:, 2, 3])) for obj in blocks]
        report.update(bottoms=bottoms, vertical_travel=travel)
        checks["ladder_pose_tracks_owned"] = bool(owned and min(travel) > 1.0e-3)
        # Lighter blocks ride higher; the floaters stay well off the floor.
        floaters = bottoms[:3]
        checks["ladder_floats_in_density_order"] = bool(
            floaters[0] > floaters[1] > floaters[2] > 0.08)
        checks["ladder_heaviest_sinks"] = bottoms[3] < 0.01
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)
    finally:
        _release(scene)
    return checks, report


def check_water_wheel(name):
    """The wheel is a Dynamic collider pinned to its axle; live play turns it
    about Y while the locked axes stay still."""
    open_demo_blend(os.path.join(DEMOS, name + ".blend"))
    scene = bpy.context.scene
    wheel = bpy.data.objects.get(WHEEL_NAME)
    contract = bool(
        wheel and wheel.fluidsim.role == "COLLIDER" and wheel.fluidsim.enabled
        and wheel.fluidsim.collider_dynamic
        and tuple(bool(v) for v in wheel.fluidsim.collider_lock_axes) == WHEEL_LOCK)
    checks = {"wheel_contract": contract, "wheel_pose_track_owned": False,
              "wheel_turns": False, "wheel_axle_held": False}
    report = {}
    if not contract:
        return checks, report
    try:
        owned, matrices = _play_live_then_replay(scene, [wheel], WHEEL_LIVE_FRAMES)
        mats = matrices[wheel.name]
        angle = np.unwrap(np.arctan2(mats[:, 0, 2], mats[:, 0, 0]))
        turned = float(abs(angle[-1] - angle[0]))
        drift = float(np.ptp(mats[:, :3, 3], axis=0).max())
        # The Y axis of a pure Y rotation never moves.
        tilt = float(np.abs(mats[:, :3, 1] - np.array([0.0, 1.0, 0.0])).max())
        report.update(turned_rad=turned, axle_drift=drift, axis_tilt=tilt)
        checks["wheel_pose_track_owned"] = bool(owned)
        checks["wheel_turns"] = turned > math.pi / 2
        checks["wheel_axle_held"] = drift < 1.0e-4 and tilt < 1.0e-4
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)
    finally:
        _release(scene)
    return checks, report


RIGID_DEMO_CHECKS = {
    "buoyancy_ladder": check_buoyancy_ladder,
    "water_wheel": check_water_wheel,
}


def point_count(obj):
    # Output clouds carry spare capacity, so len(data.points) is not the count.
    from fluidSim_warp_blender51.blender import output_publication

    return output_publication.live_point_count(obj)


def output_snapshot(limit=32):
    """Return a bounded sample of whichever owned viewport outputs exist."""
    snapshot = []
    visible_count = 0
    for name in ("FluidSim Particles", "FluidSim Whitewater", "FluidSim Surface"):
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        if obj.type == "POINTCLOUD":
            count = point_count(obj)
            position = obj.data.attributes.get("position")
            values = (() if position is None else tuple(
                round(float(value), 6)
                for item in position.data[:min(limit, count)]
                for value in item.vector
            ))
        elif obj.type == "MESH":
            count = len(obj.data.vertices)
            values = tuple(
                round(float(value), 6)
                for vertex in obj.data.vertices[:min(limit, count)]
                for value in vertex.co
            )
        else:
            continue
        visible_count += count
        snapshot.append((name, count, values))
    return visible_count, tuple(snapshot)


def check_playback_contract(scene, fs, *, cache_mode=False):
    """Exercise the saved timeline contract through real frame handlers."""
    result = {"ready": False, "advances": False}
    if fs is None:
        return result
    expected_mode = bool(fs.live and bool(fs.cache_playback_enabled) == cache_mode)
    result["ready"] = bool(
        expected_mode
        and scene.frame_current == scene.frame_start
        and scene.get("fluidsim_playback_ready") is True)
    if not result["ready"]:
        return result

    from fluidSim_warp_blender51.blender.scene_bridge import session

    _before_count, before = output_snapshot()
    try:
        scene.frame_set(scene.frame_start + 1)
        middle_count, middle = output_snapshot()
        scene.frame_set(scene.frame_start + 2)
        after_count, after = output_snapshot()
        sess = session()
        if cache_mode:
            state_valid = not sess.active
        else:
            state_valid = bool(
                sess.active
                and sess.solver.particles.refresh_host_count() > 0)
        result["advances"] = bool(
            state_valid
            and max(middle_count, after_count) > 0
            and (middle != before or after != middle))
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    return result


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


RACE_LIVE_FRAMES = 24       # the build checks the front order at frame 24


def load_meta(name):
    with open(os.path.join(DEMOS, name, "meta.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _release_live(scene):
    scene.fluidsim.live = False
    if bpy.ops.fluidsim.release.poll():
        bpy.ops.fluidsim.release()


def check_viscosity_race(scene, ds):
    """Three lane sources carry their own viscosity over a grippy floor; live
    play keeps the fronts in viscosity order (thinnest furthest)."""
    meta = load_meta("viscosity_race")
    lanes = meta["lanes"]
    sources = [bpy.data.objects.get(f"Race {lane}") for lane in lanes]
    floor = bpy.data.objects.get("Race Floor")
    checks = {
        "per_source_viscosity": bool(
            ds and ds.viscosity_enabled and ds.viscosity_from_particles
            and all(obj is not None and obj.fluidsim.role == "SOURCE"
                    and abs(float(obj.fluidsim.viscosity) - nu) <= 1.0e-6 * max(nu, 1.0e-3)
                    for nu, obj in zip(lanes.values(), sources))),
        "grippy_floor": bool(
            floor and floor.fluidsim.role == "COLLIDER"
            and abs(float(floor.fluidsim.collider_friction) - 1.0) < 1.0e-6),
        "fronts_ordered_by_viscosity": False,
    }
    report = {}
    try:
        from fluidSim_warp_blender51.blender.scene_bridge import session

        start = scene.frame_start
        for frame in range(scene.frame_current + 1, start + RACE_LIVE_FRAMES + 1):
            scene.frame_set(frame)
        pos = session().solver.particle_positions()
        fronts = []
        for y0, y1 in meta["lane_bounds"]:
            x = pos[(pos[:, 1] >= y0) & (pos[:, 1] < y1), 0]
            fronts.append(float(np.quantile(x, 0.98)) if len(x) else 0.0)
        report["fronts"] = dict(zip(lanes, fronts))
        gap = 5 * meta["voxel_size"]
        checks["fronts_ordered_by_viscosity"] = all(
            a > b + gap for a, b in zip(fronts, fronts[1:]))
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)
    finally:
        _release_live(scene)
    return checks, report


def check_deep_plunge(scene, ds):
    """Narrow band is on with none of its blockers, and the live tank runs on
    a fraction of the full-band particle count the build measured."""
    meta = load_meta("deep_plunge")
    checks = {
        "narrow_band_enabled": bool(
            ds and ds.narrow_band_enabled and ds.narrow_band_voxels > 0.0
            and not ds.species_enabled and not ds.export_attributes
            and not (ds.viscosity_enabled and ds.viscosity_from_particles)),
        "live_particles_narrow_band": False,
    }
    report = {}
    try:
        from fluidSim_warp_blender51.blender.scene_bridge import session

        count = int(session().solver.particles.refresh_host_count())
        report["live_particles"] = count
        report["build_full_band_particles"] = meta["build_mean_particles_full_band"]
        checks["live_particles_narrow_band"] = bool(
            0 < count < meta["build_mean_particles_full_band"] / 3.0)
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)
    finally:
        _release_live(scene)
    return checks, report


FEATURE_DEMO_CHECKS = {
    "viscosity_race": check_viscosity_race,
    "deep_plunge": check_deep_plunge,
}


def is_zstd_blend(path):
    with open(path, "rb") as handle:
        return handle.read(4) == ZSTD_MAGIC


def check_scene(name):
    path = os.path.join(DEMOS, name + ".blend")
    result = {
        "scene": name,
        "path": path,
        "exists": os.path.exists(path),
        "opened": False,
        "checks": {},
    }
    if not result["exists"]:
        return result

    open_demo_blend(path)
    scene = bpy.context.scene
    saved_frame = scene.frame_current
    result["opened"] = True

    fs = getattr(scene, "fluidsim", None)
    # Domain settings live on the designated domain object, never on
    # scene.fluidsim (FLUIDSIM_PG_scene declares none of them).
    from fluidSim_warp_blender51.blender.scene_bridge import active_domain_settings

    ds = active_domain_settings(scene)
    surface = bpy.data.objects.get("FluidSim Surface")
    particles = bpy.data.objects.get("FluidSim Particles")
    whitewater = bpy.data.objects.get("FluidSim Whitewater")

    checks = {
        # The builder saves Zstandard-compressed; an uncompressed file means a
        # hand save replaced the builder output (TASK-POLICY-DRIFT-018).
        "saved_compressed": is_zstd_blend(path),
        "fluidsim_settings": fs is not None,
        "domain_object": bool(fs and fs.domain_object),
        "camera": bool(scene.camera),
        "valid_domain_boundary_policy": bool(
            ds and (ds.open_boundaries or ds.domain_sdf_walls)),
    }
    actual_auto_seeds = {
        obj.name for obj in scene.objects
        if getattr(getattr(obj, "fluidsim", None), "auto_inject_seed", False)
    }
    checks["owned_auto_seeds"] = (
        actual_auto_seeds == EXPECTED_AUTO_SEEDS.get(name, set()))
    if not checks["owned_auto_seeds"]:
        result["actual_auto_seeds"] = sorted(actual_auto_seeds)
    playback = check_playback_contract(scene, fs, cache_mode=False)
    checks["playback_ready"] = playback["ready"]
    checks["playback_advances"] = playback["advances"]
    if playback.get("error"):
        result["playback_contract_error"] = playback["error"]
    if name in FEATURE_DEMO_CHECKS:
        # Continues the live session the playback check left running.
        feature_checks, result["feature"] = FEATURE_DEMO_CHECKS[name](scene, ds)
        checks.update(feature_checks)
    if name == "dam_break":
        from tools.demos import dam_break_cove as cove

        domain = bpy.data.objects.get("DamBreak Domain")
        source = bpy.data.objects.get("DamBreak Source Column")
        def near(actual, expected):
            return all(
                abs(float(a) - float(e)) < 1.0e-4
                for a, e in zip(actual, expected)
            )
        reservoir_lo = np.asarray(cove.RESERVOIR_LO)
        reservoir_hi = np.asarray(cove.RESERVOIR_HI)
        # The apron slab and training walls only show the domain boundary; a
        # collider role on either would change the solve.
        expected_frictions = {m.name: m.friction for m in cove.collider_meshes()}
        colliders = {obj.name: obj for obj in scene.objects
                     if getattr(obj, "fluidsim", None) is not None
                     and obj.fluidsim.role == "COLLIDER"}
        checks.update({
            "canonical_domain_scale": bool(
                domain and near(domain.dimensions,
                                np.subtract(cove.DOMAIN_HI, cove.DOMAIN_LO))),
            "canonical_source_transform": bool(
                source
                and near(source.location, 0.5 * (reservoir_lo + reservoir_hi))
                and near(source.dimensions, reservoir_hi - reservoir_lo)
                and source.fluidsim.emission_mode == "INITIAL_FILL"
                and source.fluidsim.source_region_kind == "ANALYTIC"),
            "cove_colliders": bool(
                set(colliders) == set(expected_frictions)
                and all(obj.fluidsim.enabled
                        and abs(float(obj.fluidsim.collider_friction)
                                - expected_frictions[obj_name]) < 1.0e-6
                        for obj_name, obj in colliders.items())),
            "no_redundant_empties": not any(
                obj.type == "EMPTY" for obj in scene.objects),
        })
        if not checks["cove_colliders"]:
            result["actual_colliders"] = {
                obj_name: float(obj.fluidsim.collider_friction)
                for obj_name, obj in colliders.items()}
    elif name == "crown_splash":
        pool_seed = bpy.data.objects.get("Crown Pool Seed")
        drop_seed = bpy.data.objects.get("Crown Droplet Seed")
        seed_contract = bool(
            pool_seed and drop_seed
            and pool_seed.type == drop_seed.type == "MESH"
            and pool_seed.fluidsim.enabled
            and drop_seed.fluidsim.enabled
            and pool_seed.fluidsim.auto_inject_seed
            and drop_seed.fluidsim.auto_inject_seed
            and abs(float(pool_seed.fluidsim.initial_velocity.z)) < 1.0e-6
            and abs(float(drop_seed.fluidsim.initial_velocity.z) + 3.5) < 1.0e-6
        )
        checks.update({
            "restartable_seed_contract": seed_contract,
            "live_playback_saved": bool(
                fs and fs.live and saved_frame == 1
                and scene.get("fluidsim_playback_ready") is True),
        })
        if seed_contract and checks["live_playback_saved"]:
            try:
                from fluidSim_warp_blender51.blender.scene_bridge import session

                scene.frame_set(2)
                sess = session()
                before = sess.solver.particles.position[:1024].numpy().copy()
                scene.frame_set(3)
                after = sess.solver.particles.position[:1024].numpy()
                delta = float(np.max(np.linalg.norm(after - before, axis=1)))
                checks["viewport_playback_advances"] = bool(
                    sess.active
                    and sess.solver.particles.refresh_host_count() > 0
                    and point_count(bpy.data.objects.get("FluidSim Particles")) > 0
                    and delta > 1.0e-6)
                result["playback_position_delta"] = delta
            except Exception as exc:  # noqa: BLE001
                checks["viewport_playback_advances"] = False
                result["playback_error"] = repr(exc)
            finally:
                fs.live = False
                bpy.ops.fluidsim.release()
        else:
            checks["viewport_playback_advances"] = False
    elif name == "chromatic_vortex":
        red_source = bpy.data.objects.get("Red Dye Initial Fill")
        blue_source = bpy.data.objects.get("Blue Dye Initial Fill")
        vortex = bpy.data.objects.get("Chromatic Mixing Vortex Force")
        live_surface = bpy.data.objects.get("FluidSim Surface")
        color_attr = (
            live_surface.data.color_attributes.get("display_color")
            if live_surface and live_surface.type == "MESH" else None)
        live_colors = np.asarray(
            [item.color[:3] for item in color_attr.data], dtype=np.float32
        ) if color_attr and len(color_attr.data) else np.empty((0, 3))
        checks.update({
            "color_mixing_enabled": bool(
                ds and ds.species_enabled and ds.species_mixing_rate > 0.0),
            "two_colored_initial_fills": bool(
                red_source and blue_source
                and red_source.fluidsim.role == "SOURCE"
                and blue_source.fluidsim.role == "SOURCE"
                and red_source.fluidsim.emission_mode == "INITIAL_FILL"
                and blue_source.fluidsim.emission_mode == "INITIAL_FILL"
                and red_source.fluidsim.source_color[0]
                > red_source.fluidsim.source_color[2]
                and blue_source.fluidsim.source_color[2]
                > blue_source.fluidsim.source_color[0]),
            "owned_spin_force": bool(
                vortex and vortex.fluidsim.role == "FORCE"
                and vortex.fluidsim.force_type == "SPIN"
                and vortex.fluidsim.force_strength > 0.0),
            "live_two_color_output": bool(
                len(live_colors)
                and np.any(live_colors[:, 0] > live_colors[:, 2] + 0.15)
                and np.any(live_colors[:, 2] > live_colors[:, 0] + 0.15)),
        })
    result["checks"] = checks
    surface = bpy.data.objects.get("FluidSim Surface")
    particles = bpy.data.objects.get("FluidSim Particles")
    whitewater = bpy.data.objects.get("FluidSim Whitewater")
    result["surface_vertices"] = len(surface.data.vertices) if surface and surface.type == "MESH" else 0
    result["particle_points"] = point_count(particles)
    result["whitewater_points"] = point_count(whitewater)

    settings = ds
    unmet = effect_precondition_failures(settings) if settings is not None else {}
    checks["enabled_effects_have_driving_settings"] = not unmet
    if unmet:
        result["unreachable_effects"] = unmet
    if settings is not None and settings.whitewater_enabled:
        reachable, report = check_whitewater_reachable(name)
        checks["whitewater_emits_in_declared_window"] = reachable
        result["whitewater_reachability"] = report
    if name == "crown_splash":
        # Last: it reopens the .blend, which stales every RNA reference above.
        cork_checks, result["cork_motion"] = check_cork_motion(name)
        checks.update(cork_checks)
    if name in RIGID_DEMO_CHECKS:
        # Last for the same reason: it reopens the .blend.
        rigid_checks, result["rigid_motion"] = RIGID_DEMO_CHECKS[name](name)
        checks.update(rigid_checks)
    return result


def main():
    disk_scenes = {
        os.path.splitext(filename)[0]
        for filename in os.listdir(DEMOS)
        if filename.endswith(".blend")
    }
    declared_scenes = set(SCENES)
    if disk_scenes != declared_scenes:
        print(json.dumps({
            "undeclared_demo_blends": sorted(disk_scenes - declared_scenes),
            "missing_demo_blends": sorted(declared_scenes - disk_scenes),
        }, indent=2))
        raise SystemExit("DEMO-BLEND-CHECK-FAIL (scene inventory mismatch)")
    requested = tuple(sys.argv[sys.argv.index("--") + 1:]) if "--" in sys.argv else ()
    unknown = sorted(set(requested) - declared_scenes)
    if unknown:
        raise SystemExit(f"DEMO-BLEND-CHECK-FAIL (unknown scenes: {', '.join(unknown)})")
    results = [check_scene(name) for name in (requested or SCENES)]
    passed = all(
        item["exists"] and item["opened"] and all(item["checks"].values())
        for item in results
    )
    print(json.dumps(results, indent=2))
    print("DEMO-BLEND-CHECK-" + ("PASS" if passed else "FAIL"))
    sys.exit(0 if passed else 1)


main()
