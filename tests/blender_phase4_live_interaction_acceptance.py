"""Blender/CUDA acceptance for Phase-4 live and pending interaction.

Run:
  blender --background --factory-startup --python tests/blender_phase4_live_interaction_acceptance.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import bpy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon

FAILURES = []
REPORT = {"schema": "fluidsim-phase4-live-interaction-v1", "checks": []}


def check(name, condition, detail=""):
    passed = bool(condition)
    REPORT["checks"].append({"name": name, "passed": passed, "detail": str(detail)})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")
    if not passed:
        FAILURES.append(name)


def main():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    from fluidSim_warp_blender51.blender import scene_bridge, viewport
    from fluidSim_warp_blender51.io import cache

    scene = bpy.context.scene
    fs = scene.fluidsim
    bpy.ops.fluidsim.setup_scene()
    fs.voxel_size = 0.12
    fs.leaf_capacity = 256
    fs.max_particles = 80_000
    fs.max_whitewater = 10_000
    fs.whitewater_enabled = False
    fs.surface_enabled = False
    fs.show_surface = False
    fs.show_particles = False
    fs.spacetime_enabled = True
    fs.spacetime_reference_mass = 0.0
    fs.particles_per_voxel = 8

    result = bpy.ops.fluidsim.initialize()
    sess = scene_bridge.session()
    check("CUDA solver initialized", result == {"FINISHED"} and sess.active
          and str(sess.solver.device).startswith("cuda"), fs.status)
    if not sess.active:
        return

    solver = sess.solver
    particle_state = solver.particles
    initial_reference = solver._st_reference_mass_cached
    t0 = time.perf_counter()
    fs.flip_ratio = 0.35
    fs.gravity = (0.0, 0.0, -3.0)
    fs.use_scene_gravity = False
    fs.pressure_iterations = 18
    fs.viscosity_enabled = True
    fs.viscosity_constant = 0.02
    fs.particles_per_voxel = 12
    live_apply_ms = (time.perf_counter() - t0) * 1000.0
    hot_values = (solver.settings.flip_ratio, tuple(solver.settings.gravity),
                  solver.settings.pressure_iterations,
                  solver.settings.viscosity_enabled,
                  solver.settings.viscosity_constant)
    check("hot fields reach active solver", abs(solver.settings.flip_ratio - 0.35) < 1e-6
          and all(abs(a - b) < 1e-6 for a, b in zip(
              solver.settings.gravity, (0.0, 0.0, -3.0)))
          and solver.settings.pressure_iterations == 18
          and solver.settings.viscosity_enabled
          and abs(solver.settings.viscosity_constant - 0.02) < 1.0e-6,
          f"{hot_values}; {live_apply_ms:.2f} ms")
    check("live edits preserve particle allocation", sess.solver is solver
          and solver.particles is particle_state)
    check("ST-FLIP automatic mass rescales", initial_reference > 0.0
          and abs(solver._st_reference_mass_cached / initial_reference - 1.5) < 1.0e-5,
          f"{initial_reference} -> {solver._st_reference_mass_cached}")
    check("live refresh is bounded", live_apply_ms < 250.0, f"{live_apply_ms:.2f} ms")

    # Keyframed RNA is evaluated before each authoritative step seam.
    fs.cfl = 2.0
    fs.flip_ratio = 0.8
    fs.keyframe_insert("cfl", frame=1)
    fs.keyframe_insert("flip_ratio", frame=1)
    fs.cfl = 1.25
    fs.flip_ratio = 0.4
    fs.keyframe_insert("cfl", frame=2)
    fs.keyframe_insert("flip_ratio", frame=2)
    manual_values = []
    for frame in (1, 2):
        scene.frame_set(frame)
        scene_bridge.step_frame(scene, sess, write_cache=False)
        manual_values.append((sess.solver.settings.cfl, sess.solver.settings.flip_ratio))
    check("manual step evaluates animated live values",
          all(abs(actual - expected) < 1e-6
              for pair, expected_pair in zip(
                  manual_values, ((2.0, 0.8), (1.25, 0.4)))
              for actual, expected in zip(pair, expected_pair)), manual_values)

    with tempfile.TemporaryDirectory(prefix="fluidsim_phase4_cache_") as cache_dir:
        fs.cache_path = cache_dir
        fs.cache_enabled = False
        fs.cache_use_custom_range = True
        fs.cache_frame_start = 1
        fs.cache_frame_end = 2
        sess.export_cache_sequence(scene, 1, 2)
        metadata = cache.read_sequence_metadata(cache_dir)
        frame_one = cache.read_frame(cache_dir, 1)
        frame_two = cache.read_frame(cache_dir, 2)
        check("cache records settings on every frame", frame_one is not None
              and frame_two is not None and "settings_signature" in frame_one
              and "settings_signature" in frame_two)
        check("cache reports animated settings boundary", metadata is not None
              and metadata.get("mixed_settings") is True
              and len(metadata.get("settings_segments", ())) == 2,
              json.dumps(metadata, default=str)[:500])

    # Interactive frame-change playback uses the same pre-step seam.
    scene.frame_set(1)
    sess.last_frame = 1
    fs.live = True
    scene.frame_set(2)
    check("playback applies animated live values", abs(sess.solver.settings.cfl - 1.25) < 1e-6
          and abs(sess.solver.settings.flip_ratio - 0.4) < 1e-6)

    advanced_solver = sess.solver
    advanced_particles = advanced_solver.particles
    sess.last_frame = 2
    advanced_solver.frame = max(2, int(advanced_solver.frame))
    fs.voxel_size = 0.1
    pending = scene_bridge.pending_presentation(scene)
    check("mid-run allocation edit stays pending", pending["action"] == "REINIT_PENDING"
          and sess.solver is advanced_solver and sess.solver.particles is advanced_particles,
          pending)
    scene.frame_set(scene.frame_start)
    check("timeline rewind applies pending reinitialize", sess.active
          and sess.solver is not advanced_solver
          and abs(sess.solver.settings.voxel_size - 0.1) < 1e-8
          and scene_bridge.pending_presentation(scene)["action"] == "NONE",
          scene_bridge.pending_presentation(scene))

    # Real Blender RNA/Scene instances keep independent cached pending state.
    first = scene
    first.fluidsim.leaf_capacity += 64
    scene_bridge.request_scene_update(first, scene_bridge.SceneUpdate.REBUILD)
    second = first.copy()
    second.name = "Phase4 Pending Isolation"
    scene_bridge._capture_applied_state(second, sess)
    second.fluidsim.voxel_size = 0.09
    scene_bridge.request_scene_update(second, scene_bridge.SceneUpdate.REBUILD)
    first_pending = scene_bridge.pending_presentation(first)
    second_pending = scene_bridge.pending_presentation(second)
    check("multiple-scene pending state is isolated",
          first_pending["action"] == "REINIT_PENDING"
          and second_pending["action"] == "REINIT_PENDING"
          and first_pending["reasons"] != second_pending["reasons"],
          f"first={first_pending}; second={second_pending}")

    check("fixed header popover registered",
          hasattr(bpy.types, "FLUIDSIM_PT_interaction_popover"))
    check("viewport pending overlay registered", viewport._OVERLAY_HANDLE is not None)


try:
    main()
except Exception as exc:  # noqa: BLE001
    FAILURES.append(type(exc).__name__)
    REPORT["traceback"] = traceback.format_exc()
    print(REPORT["traceback"])
finally:
    try:
        from fluidSim_warp_blender51.blender import scene_bridge
        scene_bridge.release_session(bpy.context.scene)
    except Exception:
        pass
    REPORT["ok"] = not FAILURES
    report_path = Path(os.environ.get(
        "FLUIDSIM_PHASE4_REPORT", REPO_ROOT / "artifacts" / "phase4_live_interaction.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
    marker = "PHASE4-LIVE-INTERACTION-PASS" if not FAILURES else "PHASE4-LIVE-INTERACTION-FAIL"
    print(marker)
    if FAILURES:
        raise SystemExit(1)
