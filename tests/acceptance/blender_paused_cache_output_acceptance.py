"""Real Blender RNA edits republish a verified paused cache without Warp.

Synthetic committed cache fixture; real cache reader, output IDs and callbacks.
This does not qualify solver/GPU output or framebuffer rendering.
"""
from __future__ import annotations

import argparse
import importlib.abc
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class NoWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            raise ModuleNotFoundError("Warp deliberately unavailable in cache acceptance")


def main(report_path=None):
    if "warp" in sys.modules:
        raise RuntimeError("Cache-only acceptance requires a fresh process without Warp")
    blocker = NoWarp()
    sys.meta_path.insert(0, blocker)
    from tools.repo_addon_loader import enable_repo_addon
    addon = enable_repo_addon(ROOT, default_set=False, persistent=False)
    from fluidSim_warp_blender51.blender import cache_identity, playback
    from fluidSim_warp_blender51.blender import output_publication as output
    from fluidSim_warp_blender51.blender.menus import create_domain_object
    from fluidSim_warp_blender51.blender.scene_bridge import session
    from fluidSim_warp_blender51.io import cache, cache_schema
    from tests.cache_io.fixtures.tiny_cache import _payload, build_tiny_cache

    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)
        print(f"PASS: {name}", flush=True)

    scene = bpy.context.scene
    fs = scene.fluidsim
    domain = create_domain_object(scene)
    domain.fluidsim_domain.whitewater_enabled = True
    domain.fluidsim_domain.surface_enabled = True
    fs.show_surface = False
    fs.show_particles = False
    fs.show_whitewater = True
    fs.liquid_display_mode = "OFF"
    scene.frame_set(7)
    bpy.context.view_layer.update()
    signature = cache_identity.settings_signature(scene)
    with tempfile.TemporaryDirectory(prefix="fluidsim-paused-cache-") as directory:
        base = build_tiny_cache(directory, frame_start=7, frame_count=1,
                                settings_signature=signature, particle_counts=[6])
        manifest = cache_schema.load_manifest(cache_schema.manifest_path(base))
        entry = manifest.frames[7]
        payload = _payload(7, signature, 6)
        payload.positions = np.arange(18, dtype=np.float32).reshape(6, 3)
        payload.output_attributes = {}
        payload.attribute_inventory["particles"] = ()
        payload.whitewater = np.ones((3, 3), np.float32)
        payload.whitewater_attributes = {
            "velocity": np.zeros((3, 3), np.float32), "age": np.zeros(3, np.float32),
            "life": np.ones(3, np.float32), "type": np.array([0, 1, 2], np.int32)}
        _, digest, size = cache.write_frame_v3_atomic(
            base, payload, 7, scene_key_value=manifest.scene_key,
            settings_signature=signature, source_signature=manifest.source_signature,
            source_frame_stamp=entry.source_frame_stamp, simulation_time=7 / 24)
        manifest = replace(manifest, frames={7: replace(
            entry, frame_sha256=digest, frame_size=size, whitewater_count=3)})
        cache_schema.write_manifest_atomic(cache_schema.manifest_path(base), manifest)
        fs.cache_path = str(base)
        fs.cache_playback_enabled = True
        fs.live = True
        playback.handle_frame_change(session(), scene, bpy.context.evaluated_depsgraph_get())
        check("six cached particles published while liquid hidden",
              output.live_point_count(output.output_object(scene, output.PARTICLES_NAME)) == 6)
        original_read = cache.read_frame
        reads = []

        def counted_read(*args, **kwargs):
            reads.append(args)
            return original_read(*args, **kwargs)

        cache.read_frame = counted_read
        try:
            fs.liquid_display_mode = "SURFACE"
            check("retained mesh appears without re-reading cache",
                  not output.output_object(scene, output.SURFACE_NAME).hide_get() and not reads)
            fs.liquid_display_mode = "PREVIEW"
            check("preview retains particle handoff without re-reading cache",
                  output.latest_published_points().particle_count == 6 and not reads)
            for mode in ("SEPARATE", "BOTH"):
                fs.whitewater_output_mode = mode
                check(f"{mode} immediately publishes all three types", all(
                    output.live_point_count(output.output_object(scene, name)) == 1
                    for name in (output.FOAM_NAME, output.BUBBLES_NAME, output.SPRAY_NAME)))
            fs.particle_display_fraction = .5
            particles = output.output_object(scene, output.PARTICLES_NAME)
            check("fraction immediately publishes three particles", output.live_point_count(particles) == 3)
            fs.particle_attribute_mode = "HEIGHT"
            colors = particles.data.attributes.get("display_color")
            check("attribute edit publishes display color", colors is not None)
            values = np.zeros(len(colors.data) * 4, np.float32)
            colors.data.foreach_get("color", values)
            check("height colors vary on current cached points",
                  np.ptp(values.reshape(-1, 4)[:3, :3], axis=0).max() > .1)
            before = len(reads)
            fs.particle_display_scale = 2
            check("radius-only edit uses no cache read", len(reads) == before)
            check("timeline remains paused on frame seven", scene.frame_current == 7)
            check("no solver allocated", not session().active)
            check("Warp was never imported", "warp" not in sys.modules)
        finally:
            cache.read_frame = original_read
    import addon_utils
    addon_utils.disable(addon.__name__, default_set=False)
    sys.meta_path.remove(blocker)
    result = {"passed": True, "blender": bpy.app.version_string, "checks": checks,
              "fixture": "synthetic committed cache, real Blender callbacks, Warp unavailable"}
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PAUSED-CACHE-OUTPUT-ACCEPTANCE-PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    main(args.json_out)
