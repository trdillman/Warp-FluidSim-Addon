"""Blender acceptance: export a baked cache's liquid surface as an Alembic sequence.

Solver-free. The script hides every CUDA device from Warp
(``CUDA_VISIBLE_DEVICES=-1``), enables the add-on from this checkout (so it also
proves registration, the playback-only status and cache playback work without
CUDA), writes a tiny cache-v3 sequence from numpy data, exports it with
``fluidsim.export_alembic``, re-imports the file into a fresh scene and checks,
per frame, that the vertex count equals ``len(surface_verts)`` and the bounding
box matches to 1e-4. It also checks the refusals and that the scene state is
restored. Prints ``ALEMBIC_EXPORT_OK`` on success.

Run (no arguments needed; needs Warp importable, not a CUDA device):
  blender -b --factory-startup --python-use-system-env \\
      --python tests/acceptance/blender_alembic_export_acceptance.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import bpy
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "tests" / "cache_io" / "fixtures"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

FAILURES = []
FRAMES = 4
TOLERANCE = 1e-4


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def _refused(scene, **kwargs):
    """Run the operator and return its error report text, or None when it ran."""
    from fluidSim_warp_blender51.blender import interchange

    try:
        interchange.plan_export(scene, **kwargs)
    except interchange.ExportRefused as exc:
        return str(exc)
    return None


def main(work):
    from tools.repo_addon_loader import enable_repo_addon

    # No CUDA device is visible to Warp: the machine this lane targets.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)

    from tiny_cache import build_tiny_cache

    from fluidSim_warp_blender51.blender import (
        cache_identity, interchange, output_publication, playback)
    from fluidSim_warp_blender51.io import cache
    from fluidSim_warp_blender51.panels import PLAYBACK_ONLY_TEXT, runtime_ui_state

    check("operator registered", hasattr(bpy.ops.fluidsim, "export_alembic"))
    ready, message = runtime_ui_state()
    check("status strip says playback only",
          (not ready) and message == PLAYBACK_ONLY_TEXT, repr(message))

    scene = bpy.context.scene
    bpy.ops.fluidsim.setup_scene()
    fs = scene.fluidsim
    fs.cache_path = str(work / "empty")
    check("refused with no cache", "no baked cache" in (_refused(scene) or ""),
          repr(_refused(scene)))

    signature = cache_identity.settings_signature(scene)
    base = build_tiny_cache(work / "cache", frame_count=FRAMES,
                            settings_signature=signature)
    fs.cache_path = str(base)

    surface = output_publication.output_object(scene, output_publication.SURFACE_NAME)
    if surface is None:
        # Publishing a cached frame creates the output objects.
        playback.set_cache_playback_enabled(scene, True)
        playback.play_cache_frame(None, scene, 1)
        surface = output_publication.output_object(
            scene, output_publication.SURFACE_NAME)
    check("surface output object exists", surface is not None)

    # Refusals.
    check("refused outside cached range",
          "outside the cached frames" in (_refused(scene, frame_end=FRAMES + 5) or ""))
    from fluidSim_warp_blender51.blender import production_bake
    real_active = production_bake.bake_is_active
    production_bake.bake_is_active = lambda _scene: True
    try:
        check("refused during a production bake",
              "production bake" in (_refused(scene) or ""))
    finally:
        production_bake.bake_is_active = real_active
    frame_two = base / "frames" / "frame_000002.npz"
    saved = frame_two.read_bytes()
    frame_two.write_bytes(saved[:-1] + bytes([saved[-1] ^ 0xFF]))
    check("refused for an unverified cache",
          "not verified" in (_refused(scene) or ""), repr(_refused(scene)))
    frame_two.write_bytes(saved)

    # State to be restored.
    scene.frame_set(1)
    other = bpy.data.objects.new("Bystander", None)
    scene.collection.objects.link(other)
    for obj in scene.objects:
        obj.select_set(False)
    other.select_set(True)
    bpy.context.view_layer.objects.active = other
    fs.live = False
    fs.cache_playback_enabled = False
    before = (bool(fs.live), bool(fs.cache_playback_enabled), scene.frame_current)

    abc = work / "surface.abc"
    result = bpy.ops.fluidsim.export_alembic(filepath=str(abc))
    check("export operator finished", "FINISHED" in result, str(result))
    check("alembic file written", abc.is_file() and abc.stat().st_size > 0)

    after = (bool(fs.live), bool(fs.cache_playback_enabled), scene.frame_current)
    check("live / cache playback / frame restored", before == after,
          f"{before} -> {after}")
    check("selection restored",
          [o.name for o in scene.objects if o.select_get()] == ["Bystander"]
          and bpy.context.view_layer.objects.active is other)

    # Expected per-frame geometry, straight from the cache.
    expected = {}
    for frame in range(1, FRAMES + 1):
        verts = np.asarray(cache.read_frame(str(base), frame)["surface_verts"])
        expected[frame] = (len(verts), verts.min(axis=0), verts.max(axis=0))
    check("vertex count varies across frames",
          len({count for count, _lo, _hi in expected.values()}) > 1)

    # Re-import into a fresh scene and compare.
    fresh = bpy.data.scenes.new("ImportCheck")
    bpy.context.window.scene = fresh
    before_objects = set(fresh.objects)
    bpy.ops.wm.alembic_import(filepath=str(abc), as_background_job=False)
    imported = [o for o in fresh.objects if o not in before_objects and o.type == "MESH"]
    check("one mesh imported", len(imported) == 1, str([o.name for o in imported]))
    if imported:
        mesh_object = imported[0]
        for frame in range(1, FRAMES + 1):
            fresh.frame_set(frame)
            evaluated = mesh_object.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", coords)
            coords = coords.reshape(-1, 3)
            evaluated.to_mesh_clear()
            count, low, high = expected[frame]
            check(f"frame {frame} vertex count", len(coords) == count,
                  f"{len(coords)} vs {count}")
            if len(coords):
                error = max(float(np.abs(coords.min(axis=0) - low).max()),
                            float(np.abs(coords.max(axis=0) - high).max()))
                check(f"frame {frame} bounding box", error <= TOLERANCE,
                      f"max error {error:.2e}")


if __name__ == "__main__":
    work = Path(tempfile.mkdtemp(prefix="fsabc_"))
    try:
        main(work)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        FAILURES.append("exception")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if FAILURES:
        print(f"ALEMBIC_EXPORT_FAILED {FAILURES}", flush=True)
        sys.exit(1)
    print("ALEMBIC_EXPORT_OK", flush=True)
