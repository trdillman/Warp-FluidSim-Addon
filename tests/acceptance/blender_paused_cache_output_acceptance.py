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
from types import SimpleNamespace

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class NoWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            raise ModuleNotFoundError("Warp deliberately unavailable in cache acceptance")


class RecordingLayout:
    """Draw adapters against real RNA; no claim of native pixels or mouse input."""

    def __init__(self, records=None):
        self.records = [] if records is None else records

    def label(self, **kwargs):
        self.records.append({"kind": "label", **kwargs})

    def prop(self, owner, name, **kwargs):
        assert hasattr(owner, name), name
        self.records.append({"kind": "prop", "name": name, **kwargs})

    def operator(self, identifier, **kwargs):
        group, name = identifier.split(".")
        getattr(getattr(bpy.ops, group), name).get_rna_type()
        self.records.append({"kind": "operator", "id": identifier, **kwargs})
        return SimpleNamespace()

    def __getattr__(self, name):
        return lambda *args, **kwargs: RecordingLayout(self.records)


def recorded_draw(cls):
    layout = RecordingLayout()
    cls.draw(SimpleNamespace(layout=layout), bpy.context)
    return layout.records


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

        from fluidSim_warp_blender51 import panels
        from fluidSim_warp_blender51.blender import (
            liquid_display,
            scene_bridge,
            ui_header,
            ui_scene,
        )
        from fluidSim_warp_blender51.blender.deck import controls, wheel

        # The render-output authoring controls never silently become viewport controls.
        check("render output RNA describes its separate scope",
              fs.bl_rna.properties["liquid_view_mode"].name == "Render Outputs"
              and "Independent of Liquid Display" in
              fs.bl_rna.properties["liquid_view_mode"].description)
        for authored, display, visible, label in (
                ("HIDDEN", "SURFACE", True, "Liquid Viewport: Surface Mesh"),
                ("PARTICLES", "SURFACE", True, "Liquid Viewport: Surface Mesh"),
                ("SURFACE", "OFF", False, "Liquid Viewport: Off"),
                ("HIDDEN", "PREVIEW", False, "Liquid Viewport: Liquid Preview")):
            fs.liquid_view_mode = authored
            fs.liquid_display_mode = display
            liquid_display.apply_viewport(scene, deferred=False)
            check(f"{authored}/{display} preserves independent viewport policy",
                  output.output_object(scene, output.SURFACE_NAME).hide_get() != visible
                  and fs.liquid_view_mode == authored)
            check(f"{authored}/{display} summary follows viewport policy",
                  any(row.get("text") == label for row in
                      recorded_draw(ui_scene.FLUIDSIM_PT_scene_outputs)))

        fs.liquid_display_mode = "SURFACE"
        output.output_object(scene, output.SURFACE_NAME).data.clear_geometry()
        liquid_display.apply_viewport(scene, deferred=False)
        check("empty surface uses the same particle fallback in summary and visibility",
              liquid_display.viewport_summary(scene) == "Particles (surface unavailable)"
              and not output.output_object(scene, output.PARTICLES_NAME).hide_get())
        playback.play_cache_frame(session(), scene, 7)

        # A UI lookup must neither create/link a collection nor adopt a legacy object.
        lookup_scene = bpy.data.scenes.new("ReadOnlyOutputLookup")
        lookup_scene.fluidsim.output_collection_name = "ReadOnlyOutputCollection"
        before = (len(bpy.data.collections), len(bpy.data.objects))
        check("missing output lookup is read-only",
              output.output_object(lookup_scene, output.SURFACE_NAME, ensure=False) is None
              and (len(bpy.data.collections), len(bpy.data.objects)) == before)
        coll = bpy.data.collections.new("ReadOnlyOutputCollection")
        legacy = bpy.data.objects.new("ReadOnlyLegacyOutput", None)
        coll.objects.link(legacy)
        check("legacy lookup does not adopt or link",
              output.output_object(lookup_scene, legacy.name, ensure=False) == legacy
              and not legacy.get(output.OUTPUT_TAG)
              and coll.name not in lookup_scene.collection.children)
        check("normal output lookup still adopts and links",
              output.output_object(lookup_scene, legacy.name) == legacy
              and legacy.get(output.OUTPUT_TAG) == legacy.name
              and coll.name in lookup_scene.collection.children)

        for cls in (panels.FLUIDSIM_PT_viewport_status,
                    ui_header.FLUIDSIM_PT_header_popover,
                    ui_scene.FLUIDSIM_PT_scene_cache_bake):
            check(f"{cls.__name__} exposes cache exit without CUDA",
                  any(row.get("id") == "fluidsim.cache_playback_stop"
                      for row in recorded_draw(cls)))
        check("Deck transport exposes cache exit",
              "fluidsim.cache_playback_stop" in {
                  item.target for item in controls.transport_controls("CACHE_PLAYBACK", False, False)})
        check("Bake wheel changes Play Cache into cache exit",
              "fluidsim.cache_playback_stop" in {
                  item.target for item in wheel.stage_controls(bpy.context, "BAKE")})
        manifest_before = cache_schema.manifest_path(base).read_bytes()
        check("public cache exit preserves committed data",
              bpy.ops.fluidsim.cache_playback_stop() == {"FINISHED"}
              and not fs.cache_playback_enabled and not fs.live
              and cache_schema.manifest_path(base).read_bytes() == manifest_before)

        # Real cache error/recovery, using the handler that timeline playback calls.
        bpy.ops.fluidsim.cache_playback_start()
        scene.frame_set(8)
        playback.handle_frame_change(session(), scene, bpy.context.evaluated_depsgraph_get())
        check("missing frame records an actual error", "missing or corrupt" in fs.status)
        for cls in (panels.FLUIDSIM_PT_viewport_status, ui_header.FLUIDSIM_PT_header_popover):
            records = recorded_draw(cls)
            text = " ".join(row.get("text", "") for row in records)
            check(f"{cls.__name__} retains actionable cache error",
                  "Cache Playback Error" in text and "Cache frame 8" in text
                  and any(row.get("id") == "fluidsim.open_scene_properties" for row in records))
        check("error presentation leaves retry armed", fs.live and fs.cache_playback_enabled)
        scene.frame_set(7)
        playback.handle_frame_change(session(), scene, bpy.context.evaluated_depsgraph_get())
        check("valid frame clears error and restores ready presentation",
              not scene_bridge.current_error(scene)
              and any(row.get("text") == "Cache Playback Ready"
                      for row in recorded_draw(panels.FLUIDSIM_PT_viewport_status)))
    import addon_utils
    addon_utils.disable(addon.__name__, default_set=False)
    sys.meta_path.remove(blocker)
    result = {"passed": True, "blender": bpy.app.version_string, "checks": checks,
              "fixture": "synthetic committed cache, real Blender callbacks, Warp unavailable",
              "ui_evidence": "recording layout with native RNA; no framebuffer or mouse input"}
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PAUSED-CACHE-OUTPUT-ACCEPTANCE-PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    main(args.json_out)
