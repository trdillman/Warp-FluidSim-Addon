"""Blender --background gate for cache sequence export and playback.

Run:
  blender --background --factory-startup --python tests/blender_cache_export_pipeline.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import bpy
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon

FAILURES = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def point_count(obj):
    # Output clouds carry spare capacity, so len(data.points) is not the count.
    from fluidSim_warp_blender51.blender import viewport

    return viewport.live_point_count(obj)


def point_sample(obj, limit=32):
    count = point_count(obj)
    if count == 0:
        return np.zeros((0, 3), dtype=np.float32)
    n = min(count, limit)
    if obj.type == "POINTCLOUD":
        attr = obj.data.attributes.get("position")
        values = np.empty(len(attr.data) * 3, dtype=np.float32)
        attr.data.foreach_get("vector", values)
    else:
        values = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", values)
    # Live points occupy the head of the buffer; the tail is padding.
    return values.reshape(-1, 3)[:n].copy()


def main():
    try:
        enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
        check("addon enable", True)
    except Exception as exc:  # noqa: BLE001
        check("addon enable", False, repr(exc))
        return

    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.voxel_size = 0.12
    fs.max_particles = 80_000
    fs.max_whitewater = 5_000
    fs.whitewater_enabled = False
    fs.surface_enabled = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 1.0
    fs.cache_enabled = False
    fs.cache_playback_enabled = False
    fs.cache_use_custom_range = True
    fs.cache_frame_start = 1
    fs.cache_frame_end = 3

    cache_dir = tempfile.mkdtemp(prefix="fluidsim_cache_pipeline_")
    fs.cache_path = cache_dir
    try:
        check("setup_scene", bpy.ops.fluidsim.setup_scene() == {"FINISHED"})
        check(
            "export cache sequence",
            bpy.ops.fluidsim.export_cache_sequence() == {"FINISHED"},
            fs.status,
        )

        metadata_path = os.path.join(cache_dir, "fluidsim_sequence.json")
        check("sequence metadata exists", os.path.exists(metadata_path), metadata_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            check(
                "sequence metadata range",
                (
                    metadata.get("format_version") == 1
                    and metadata.get("frame_start") == 1
                    and metadata.get("frame_end") == 3
                    and metadata.get("frame_count") == 3
                ),
                str(metadata),
            )

        frame_paths = [
            os.path.join(cache_dir, f"fluidsim_{frame:05d}.npz")
            for frame in range(1, 4)
        ]
        check(
            "cache frame files exist",
            all(os.path.exists(path) for path in frame_paths),
            ", ".join(os.path.basename(path) for path in frame_paths),
        )

        frame_data = []
        for frame, path in enumerate(frame_paths, start=1):
            if not os.path.exists(path):
                continue
            with np.load(path, allow_pickle=False) as data:
                positions = data["positions"]
                surface_verts = data["surface_verts"]
                frame_data.append(
                    {
                        "frame": int(np.asarray(data["frame"]).item()),
                        "particles": int(len(positions)),
                        "surface_verts": int(len(surface_verts)),
                        "position_sample": positions[:32].copy(),
                    }
                )
                check(
                    f"frame {frame} snapshot payload",
                    (
                        int(np.asarray(data["format_version"]).item()) == 1
                        and int(np.asarray(data["frame"]).item()) == frame
                        and len(positions) > 1_000
                        and len(data["velocities"]) == len(positions)
                        and "voxel_size" in data.files
                    ),
                    f"{len(positions)} particles, {len(surface_verts)} verts",
                )

        if len(frame_data) >= 2:
            changed = any(
                not np.allclose(frame_data[0]["position_sample"], item["position_sample"])
                for item in frame_data[1:]
                if item["position_sample"].shape == frame_data[0]["position_sample"].shape
            )
            check("cache frames differ over time", changed)

        from fluidSim_warp_blender51.blender.scene_bridge import session

        bpy.ops.fluidsim.release()
        check("solver released before cache playback", not session().active)
        fs.live = True
        fs.cache_playback_enabled = True

        session().last_frame = -1
        scene.frame_set(1)
        parts_1 = bpy.data.objects.get("FluidSim Particles")
        sample_1 = point_sample(parts_1)
        count_1 = point_count(parts_1)
        check("cache playback frame 1 output", count_1 > 1_000, f"{count_1} points")
        check("cache playback does not require live solver", not session().active)

        scene.frame_set(3)
        parts_3 = bpy.data.objects.get("FluidSim Particles")
        sample_3 = point_sample(parts_3)
        count_3 = point_count(parts_3)
        check("cache playback frame 3 output", count_3 > 1_000, f"{count_3} points")
        if sample_1.shape == sample_3.shape and len(sample_1):
            check("cache playback changes visible output", not np.allclose(sample_1, sample_3))
        else:
            check("cache playback changes visible output", count_1 != count_3)
        check("cache playback status", fs.status.startswith("Cache frame 3"), fs.status)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


main()
print()
if FAILURES:
    print(f"CACHE-EXPORT-PIPELINE-FAIL ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("CACHE-EXPORT-PIPELINE-PASS")
sys.exit(0)
