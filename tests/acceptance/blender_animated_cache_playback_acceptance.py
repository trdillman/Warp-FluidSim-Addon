"""Blender/CUDA acceptance: a bake with an animated collider plays and resumes.

REVIEW-022 follow-up. The cache scene key hashes source transforms and
evaluated geometry, so with a keyframed collider every frame after the first
of a finished bake was refused as "baked from different scene settings", and
resume depended on where the playhead happened to be.

Run:
  blender --background --factory-startup --python tests/acceptance/blender_animated_cache_playback_acceptance.py
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import traceback

import bpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

FAILURES = []
START, PARTIAL_END, END = 1, 3, 6


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def _animated_collider(scene, domain_object):
    centre = domain_object.matrix_world.translation.copy()
    half = domain_object.dimensions.x * 0.25
    bpy.ops.mesh.primitive_cube_add(size=domain_object.dimensions.x * 0.15,
                                    location=centre)
    collider = bpy.context.active_object
    collider.name = "Animated Collider"
    collider.fluidsim.role = "COLLIDER"
    collider.location.x = centre.x - half
    collider.keyframe_insert("location", frame=START)
    collider.location.x = centre.x + half
    collider.keyframe_insert("location", frame=END)
    scene.frame_set(START)
    return collider


def _play(scene, frame):
    from fluidSim_warp_blender51.blender import playback, scene_bridge

    with playback.suppress_frame_handlers():
        scene.frame_set(frame)
    playback.play_cache_frame(scene_bridge.session(), scene, frame)


def main(cache_root):
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)
    from fluidSim_warp_blender51.blender import (
        cache_identity, production_bake, scene_bridge)

    scene = bpy.context.scene
    bpy.ops.fluidsim.setup_scene()
    fs = scene.fluidsim
    domain = scene_bridge.active_domain_settings(scene)
    check("domain settings resolved", domain is not None)
    if domain is None:
        return
    domain.voxel_size = 0.12
    domain.whitewater_enabled = False
    domain.surface_enabled = False
    fs.cache_path = str(cache_root)
    fs.live = False
    _animated_collider(scene, fs.domain_object)

    depsgraph = bpy.context.evaluated_depsgraph_get
    first = cache_identity.signature_bundle(scene, depsgraph())
    scene.frame_set(END)
    last = cache_identity.signature_bundle(scene, depsgraph())
    scene.frame_set(START)
    check("collider animation changes the source signature",
          first["source_signature"] != last["source_signature"])
    check("settings signature is frame-invariant",
          first["settings_signature"] == last["settings_signature"]
          == cache_identity.settings_signature(scene))

    # partial bake: frames START..PARTIAL_END of START..END
    backend = production_bake.ProductionBakeBackend()
    next_frame = backend.initialize(scene, START, END)
    for frame in range(next_frame, PARTIAL_END + 1):
        backend.step_and_commit(scene, frame)
    backend.finish(scene)

    with_playhead = int(END - 1)
    scene.frame_set(with_playhead)
    decision = production_bake.ProductionBakeBackend().resume_decision(scene)
    check("partial bake resumes with the playhead elsewhere",
          decision.allowed and decision.code == "RESUME_READY"
          and decision.resume_frame == PARTIAL_END + 1, decision)
    check("resume check leaves the playhead where it was",
          scene.frame_current == with_playhead, scene.frame_current)

    result = production_bake.run_synchronous(scene, end=END, resume=True)
    check("resume completes the bake",
          result.status is production_bake.BakeStatus.COMPLETE
          and result.last_committed_frame == END, result)

    played = []
    for frame in range(START, END + 1):
        try:
            _play(scene, frame)
            played.append(frame)
        except Exception as exc:
            check(f"frame {frame} plays back", False, f"{type(exc).__name__}: {exc}")
    check("every baked frame plays back", played == list(range(START, END + 1)),
          played)

    domain.voxel_size = 0.1
    try:
        _play(scene, START + 1)
        refused = ""
    except ValueError as exc:
        refused = str(exc)
    check("a settings edit is still refused", "different scene settings" in refused,
          refused)
    decision = production_bake.ProductionBakeBackend().resume_decision(scene)
    check("a settings edit blocks resume", not decision.allowed, decision)

    if scene_bridge.session().active:
        scene_bridge.session().release(scene)


if __name__ == "__main__":
    root = Path(tempfile.mkdtemp(prefix="fluidsim-animated-cache-"))
    try:
        main(root / "cache")
    except Exception:
        FAILURES.append("unhandled exception")
        traceback.print_exc()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if FAILURES:
        print(f"ANIMATED-CACHE-PLAYBACK-ACCEPTANCE-FAIL: {FAILURES}", flush=True)
        sys.exit(1)
    print("ANIMATED-CACHE-PLAYBACK-ACCEPTANCE-PASS", flush=True)
