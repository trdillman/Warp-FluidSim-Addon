"""Blender acceptance: export a baked cache's surface and particles as animated USD.

Solver-free, like the Alembic acceptance: CUDA is hidden from Warp, the add-on
is enabled from this checkout, and a tiny cache-v3 sequence is written from
numpy data. Its particle count grows past the starting point-cloud capacity
and then shrinks, which would replace the datablock mid-range without the
exporter's reservation (and USD would then write separate prims instead of
time samples). ``fluidsim.export_usd`` writes a .usda that is read back with
pxr and checked per frame: particle count, the first point, the cached
``speed`` primvar, no zero-width padding, and the surface vertex count. Scene
state (selection, frame, frame range, modes, modifiers, node groups) must be
restored. Prints ``USD_EXPORT_OK`` on success.

Run (no arguments needed; needs Warp importable, not a CUDA device):
  blender -b --factory-startup --python-use-system-env \\
      --python tests/acceptance/blender_usd_export_acceptance.py
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
# Grows past the 1024-point starting capacity, then drops below half of it.
PARTICLE_COUNTS = (300, 2600, 40, 1800)
TOLERANCE = 1e-5


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def main(work):
    from tools.repo_addon_loader import enable_repo_addon

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)

    from tiny_cache import build_tiny_cache, particle_line, particle_speed, sphere_mesh

    from fluidSim_warp_blender51.blender import (
        cache_identity, interchange, output_publication, playback, scene_bridge)

    check("operator registered", hasattr(bpy.ops.fluidsim, "export_usd"))
    scene = bpy.context.scene
    bpy.ops.fluidsim.setup_scene()
    fs = scene.fluidsim
    fs.show_particles = True
    fs.cache_path = str(work / "empty")
    try:
        interchange.plan_usd_export(scene)
        refused = None
    except interchange.ExportRefused as exc:
        refused = str(exc)
    check("refused with no cache", "no baked cache" in (refused or ""), repr(refused))

    frames = range(1, len(PARTICLE_COUNTS) + 1)
    base = build_tiny_cache(
        work / "cache", frame_count=len(PARTICLE_COUNTS),
        settings_signature=cache_identity.settings_signature(scene),
        particle_counts=PARTICLE_COUNTS)
    fs.cache_path = str(base)
    # Publishing a cached frame creates the output objects.
    playback.set_cache_playback_enabled(scene, True)
    playback.play_cache_frame(scene_bridge.session(), scene, 1)
    particles = output_publication.output_object(
        scene, output_publication.PARTICLES_NAME)
    check("particle output exists", particles is not None)
    plan = interchange.plan_usd_export(scene)
    check("plan sizes particles for the range's peak",
          [(o.name, peak) for o, peak in plan.points]
          == [(particles.name, max(PARTICLE_COUNTS))], str(plan.points))

    # State to be restored.
    scene.frame_set(2)
    scene.frame_start, scene.frame_end = 1, 250
    other = bpy.data.objects.new("Bystander", None)
    scene.collection.objects.link(other)
    for obj in scene.objects:
        obj.select_set(False)
    other.select_set(True)
    bpy.context.view_layer.objects.active = other
    fs.live = False
    fs.cache_playback_enabled = False

    def snapshot():
        return (bool(fs.live), bool(fs.cache_playback_enabled), scene.frame_current,
                scene.frame_start, scene.frame_end, len(bpy.data.node_groups),
                {o.name: [m.show_render for m in o.modifiers] for o in plan.objects})

    before = snapshot()
    usda = work / "liquid.usda"
    result = bpy.ops.fluidsim.export_usd(filepath=str(usda))
    check("export operator finished", "FINISHED" in result, str(result))
    check("usd file written", usda.is_file() and usda.stat().st_size > 0)
    after = snapshot()
    check("modes / frame / range / node groups / modifiers restored",
          before == after, f"{before} -> {after}")
    check("selection restored",
          [o.name for o in scene.objects if o.select_get()] == ["Bystander"]
          and bpy.context.view_layer.objects.active is other)

    from pxr import Usd

    stage = Usd.Stage.Open(str(usda))
    points = [p for p in stage.Traverse() if p.GetTypeName() == "Points"]
    meshes = [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
    check("one Points prim (no prim per replaced datablock)", len(points) == 1,
          str([str(p.GetPath()) for p in points]))
    check("one Mesh prim", len(meshes) == 1, str([str(p.GetPath()) for p in meshes]))
    if points:
        prim = points[0]
        positions = prim.GetAttribute("points")
        widths = prim.GetAttribute("widths")
        speed = prim.GetAttribute("primvars:fluidsim_speed")
        check("speed primvar exported", bool(speed))
        for frame, count in zip(frames, PARTICLE_COUNTS):
            got = np.asarray(positions.Get(frame), dtype=np.float32)
            check(f"frame {frame} particle count", len(got) == count,
                  f"{len(got)} vs {count}")
            if len(got) == count:
                error = float(np.abs(got - particle_line(frame, count)).max())
                check(f"frame {frame} positions", error <= TOLERANCE, f"{error:.2e}")
            check(f"frame {frame} no zero-width padding",
                  min(widths.Get(frame) or [0.0]) > 0.0)
            if speed:
                values = np.asarray(speed.Get(frame), dtype=np.float32)
                check(f"frame {frame} speed primvar",
                      values.shape == (count,) and np.allclose(
                          values, particle_speed(frame, count)),
                      f"{values[:1]} x {len(values)}")
    if meshes:
        positions = meshes[0].GetAttribute("points")
        for frame in frames:
            expected = len(sphere_mesh(frame)[0])
            got = len(positions.Get(frame) or ())
            check(f"frame {frame} surface vertex count", got == expected,
                  f"{got} vs {expected}")


if __name__ == "__main__":
    work = Path(tempfile.mkdtemp(prefix="fsusd_"))
    hidden = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        main(work)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        FAILURES.append("exception")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if hidden is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = hidden
    if FAILURES:
        print(f"USD_EXPORT_FAILED {FAILURES}", flush=True)
        sys.exit(1)
    print("USD_EXPORT_OK", flush=True)
