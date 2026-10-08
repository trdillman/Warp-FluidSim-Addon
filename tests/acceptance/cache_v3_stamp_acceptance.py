"""Blender/CUDA acceptance: cache-v3 stamps every frame's sources and colliders.

SPEC-cache-v3. Uses the shipped ``demos/paddle_stir.blend`` (a keyframed
paddle in a tank), coarsened so a short bake is cheap. The blend is never
saved and every bake goes to a temporary directory.

  1. Animated scene: bake frames 1-30 of 1-40. The stamp of frame 1 differs
     from frame 30, every manifest stamp is what the scene evaluates at that
     frame, and the stamp inside each frame file equals the manifest's.
     Resume is allowed while the animation is unchanged.
  2. Shift the paddle keyframes from frame 20 on by 5 frames. Resume is refused
     with SOURCE_ANIMATION_CHANGED naming the first frame whose sources differ
     (computed independently here from stamps taken before and after the edit).
     "Re-bake from frame 20" is allowed, re-baking from after the change is
     refused, and the re-bake itself completes with frames 1-19 untouched.
  3. Static scene (paddle animation removed): every stamp is identical and a
     partial bake resumes to completion.
  4. A 2.x manifest is refused with the 3.0 message and never read as empty.

Run:
  blender --background --factory-startup --python-use-system-env --python tests/acceptance/cache_v3_stamp_acceptance.py
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import traceback

import bpy
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

FAILURES = []
BLEND = REPO_ROOT / "demos" / "paddle_stir.blend"
PADDLE = "PaddleStir Paddle"
START, BAKED, END = 1, 30, 40
MOVE_FROM, SHIFT = 20, 5
STATIC_START, STATIC_PARTIAL, STATIC_END = 1, 6, 10


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def _release_session():
    from fluidSim_warp_blender51.blender import scene_bridge

    if scene_bridge.session().active:
        scene_bridge.session().release(bpy.context.scene)


def _open_coarse_scene(cache_dir):
    """Open the paddle demo with a cheap configuration and an empty cache."""
    _release_session()
    bpy.ops.wm.open_mainfile(filepath=str(BLEND))
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)
    from fluidSim_warp_blender51.blender import scene_bridge

    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.live = False
    fs.cache_path = str(cache_dir)
    domain = scene_bridge.active_domain_settings(scene)
    domain.voxel_size = 0.06
    domain.whitewater_enabled = False
    domain.surface_enabled = False
    domain.profiling_enabled = False
    session = scene_bridge.session()
    if session.active:
        session.release(scene)
    return scene


def _stamp_at(scene, frame):
    from fluidSim_warp_blender51.blender import cache_identity, playback

    with playback.suppress_frame_handlers():
        scene.frame_set(frame)
        return cache_identity.source_frame_stamp(
            scene, bpy.context.evaluated_depsgraph_get(), frame)


def _stamps(scene, frames):
    stamps = {frame: _stamp_at(scene, frame) for frame in frames}
    _stamp_at(scene, START)
    return stamps


def _manifest(cache_dir):
    from fluidSim_warp_blender51.io.cache_schema import load_manifest, manifest_path

    return load_manifest(manifest_path(cache_dir))


def _manifest_stamps(cache_dir):
    manifest = _manifest(cache_dir)
    return {frame: manifest.frames[frame].source_frame_stamp
            for frame in manifest.committed_frames}


def _payload_stamps(cache_dir):
    manifest = _manifest(cache_dir)
    found = {}
    for frame in manifest.committed_frames:
        with np.load(Path(cache_dir) / manifest.frames[frame].frame_file) as data:
            found[frame] = str(np.asarray(data["source_frame_stamp"]).item())
    return found


def _bake_partial(scene, start, end, last):
    from fluidSim_warp_blender51.blender import production_bake

    backend = production_bake.ProductionBakeBackend()
    next_frame = backend.initialize(scene, start, end)
    try:
        for frame in range(next_frame, last + 1):
            backend.step_and_commit(scene, frame)
    finally:
        backend.finish(scene)


def _decision(scene, **kwargs):
    from fluidSim_warp_blender51.blender import production_bake

    return production_bake.ProductionBakeBackend().resume_decision(scene, **kwargs)


def _shift_paddle_keys(scene, first, shift):
    """Drag the paddle's rotation keys from ``first`` onward ``shift`` frames later."""
    paddle = bpy.data.objects[PADDLE]
    last_key = int(paddle.animation_data.action.frame_range[1])
    values = {}
    for frame in range(first, last_key + 1):
        scene.frame_set(frame)
        values[frame] = float(paddle.rotation_euler[2])
    prefs = bpy.context.preferences.edit
    interpolation = prefs.keyframe_new_interpolation_type
    prefs.keyframe_new_interpolation_type = "LINEAR"
    try:
        for frame in range(last_key, first - 1, -1):
            paddle.rotation_euler[2] = values[frame]
            paddle.keyframe_insert("rotation_euler", index=2, frame=frame + shift)
        for frame in range(first, first + shift):
            paddle.keyframe_delete("rotation_euler", index=2, frame=frame)
    finally:
        prefs.keyframe_new_interpolation_type = interpolation
    scene.frame_set(START)


def animated_scene(root):
    cache_dir = root / "animated"
    scene = _open_coarse_scene(cache_dir)
    # Imported after the blend opens: enabling the add-on again reloads its
    # modules, and an earlier import would hold the previous copy.
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.io import cache

    pristine = _stamps(scene, range(START, END + 1))

    _bake_partial(scene, START, END, BAKED)
    manifest = _manifest(cache_dir)
    check("manifest is cache-v3 with frames 1-30",
          manifest.format_version == 3
          and manifest.committed_frames == tuple(range(START, BAKED + 1)),
          manifest.committed_frames[-3:])
    baked = _manifest_stamps(cache_dir)
    check("stamp of frame 1 differs from frame 30",
          baked[START] != baked[BAKED],
          f"{baked[START][:12]} vs {baked[BAKED][:12]}")
    check("every stamp is what the scene evaluates at that frame",
          all(baked[frame] == pristine[frame] for frame in baked))
    check("the paddle gives a new stamp on every moving frame",
          len(set(baked.values())) == len(baked), len(set(baked.values())))
    check("the stamp stored in each frame file equals the manifest's",
          _payload_stamps(cache_dir) == baked)
    played = cache.read_frame(str(cache_dir), BAKED)
    check("a baked frame reads back with its stamp",
          played is not None
          and str(played["source_frame_stamp"]) == baked[BAKED])

    decision = _decision(scene)
    check("resume is allowed while the animation is unchanged",
          decision.allowed and decision.code == "RESUME_READY"
          and decision.resume_frame == BAKED + 1, decision)

    _shift_paddle_keys(scene, MOVE_FROM, SHIFT)
    edited = _stamps(scene, range(START, BAKED + 1))
    expected = next(frame for frame in range(START, BAKED + 1)
                    if edited[frame] != pristine[frame])
    check("the keyframe shift changes the sources at frame 20 and not before",
          expected == MOVE_FROM
          and all(edited[frame] == pristine[frame]
                  for frame in range(START, MOVE_FROM)),
          f"first differing frame {expected}")

    decision = _decision(scene)
    check("resume is refused at the first changed frame",
          not decision.allowed and decision.code == "SOURCE_ANIMATION_CHANGED"
          and decision.first_changed_frame == expected
          and str(expected) in decision.message, decision)
    check("the refusal leaves the playhead where it was",
          bpy.context.scene.frame_current == START, scene.frame_current)

    late = _decision(scene, rebake_from=BAKED + 1)
    check("re-baking from after the change is refused",
          not late.allowed and late.code == "SOURCE_ANIMATION_CHANGED"
          and late.first_changed_frame == expected, late)
    early = _decision(scene, rebake_from=expected)
    check("re-baking from the first changed frame is allowed",
          early.allowed and early.code == "REBAKE_READY"
          and early.resume_frame == expected, early)

    before = _manifest_stamps(cache_dir)
    result = production_bake.run_synchronous(
        scene, end=END, resume=True, rebake_from=expected)
    check("the re-bake completes",
          result.status is production_bake.BakeStatus.COMPLETE
          and result.last_committed_frame == END, result)
    after = _manifest_stamps(cache_dir)
    check("frames before the change are untouched",
          all(after[frame] == before[frame] for frame in range(START, expected)))
    check("frames from the change on carry the new animation",
          all(after[frame] == _stamp_at(scene, frame)
              for frame in range(expected, END + 1))
          and after[expected] != before[expected])
    check("the frame files agree with the manifest after the re-bake",
          _payload_stamps(cache_dir) == after)
    decision = _decision(scene)
    check("the finished cache matches the edited animation",
          decision.code == "ALREADY_COMPLETE", decision)


def static_scene(root):
    cache_dir = root / "static"
    scene = _open_coarse_scene(cache_dir)
    from fluidSim_warp_blender51.blender import production_bake

    paddle = bpy.data.objects[PADDLE]
    paddle.animation_data_clear()
    paddle.rotation_euler = (0.0, 0.0, 0.0)

    _bake_partial(scene, STATIC_START, STATIC_END, STATIC_PARTIAL)
    stamps = _manifest_stamps(cache_dir)
    check("a static scene stamps every frame the same",
          len(stamps) == STATIC_PARTIAL and len(set(stamps.values())) == 1,
          len(set(stamps.values())))
    decision = _decision(scene)
    check("a static scene resumes",
          decision.allowed and decision.code == "RESUME_READY"
          and decision.resume_frame == STATIC_PARTIAL + 1, decision)
    result = production_bake.run_synchronous(scene, end=STATIC_END, resume=True)
    stamps = _manifest_stamps(cache_dir)
    check("the resumed static bake completes with one stamp throughout",
          result.status is production_bake.BakeStatus.COMPLETE
          and sorted(stamps) == list(range(STATIC_START, STATIC_END + 1))
          and len(set(stamps.values())) == 1, result)


def legacy_cache(root):
    cache_dir = root / "legacy"
    scene = _open_coarse_scene(cache_dir)
    from fluidSim_warp_blender51.io import cache
    from fluidSim_warp_blender51.io.cache_schema import (
        LEGACY_2X_CACHE_MESSAGE, LegacyCacheFormatError, manifest_path)

    cache_dir.mkdir(parents=True)
    manifest_path(cache_dir).write_text('{"format_version": 2}', encoding="utf-8")
    decision = _decision(scene)
    check("a 2.x cache is refused with the 3.0 message",
          not decision.allowed and decision.code == "CACHE_FORMAT_2X"
          and decision.message == LEGACY_2X_CACHE_MESSAGE, decision)
    try:
        cache.read_frame(str(cache_dir), START)
        refused = False
    except LegacyCacheFormatError:
        refused = True
    check("a 2.x cache is never read as empty", refused)


def main(root):
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)
    animated_scene(root)
    static_scene(root)
    legacy_cache(root)
    _release_session()


if __name__ == "__main__":
    work = Path(tempfile.mkdtemp(prefix="fluidsim-cache-v3-"))
    try:
        main(work)
    except Exception:
        FAILURES.append("unhandled exception")
        traceback.print_exc()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if FAILURES:
        print(f"CACHE-V3-STAMP-ACCEPTANCE-FAIL: {FAILURES}", flush=True)
        sys.exit(1)
    print("CACHE-V3-STAMP-ACCEPTANCE-PASS", flush=True)
