"""Windowed acceptance of the evaluated-mesh FSPT bridge on a saved FluidSim demo."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P1 = Path(__file__).with_name("blender_fspt_acceptance.py")
LIVE_STATES = 24
PAUSE_AFTER = 8
REFINED_SAMPLES = 8


class FrozenSurfaceError(AssertionError):
    """Fresh solver metadata accompanied geometry that did not advance."""


def surface_digest(vertices, triangles):
    """Normalize concatenation offsets while retaining position and topology."""
    assert len(triangles), "The evaluated fluid surface has no triangles"
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(triangles - triangles.min(), dtype="<i4").tobytes())
    digest.update(np.ascontiguousarray(vertices[triangles], dtype="<f4").tobytes())
    return digest.hexdigest()


def verify_advance(prior, current, prior_pixels, pixels):
    assert current["epoch"] == prior["epoch"], "The live sequence crossed a solver reset"
    for field in ("timeline_frame", "solver_frame", "generation", "state_revision"):
        assert current[field] > prior[field], f"Non-advancing {field}: {prior[field]} to {current[field]}"
    if current["geometry_sha256"] == prior["geometry_sha256"]:
        raise FrozenSurfaceError("solved state advanced but fluid geometry did not")
    assert pixels.shape == prior_pixels.shape, "Trace size changed within the live sequence"
    delta = float(np.mean(np.abs(pixels[..., :3] - prior_pixels[..., :3])))
    assert delta > 1e-6, f"The completed image remained unchanged: {delta:g}"
    return delta


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--blend", type=Path, default=ROOT / "demos/dam_break.blend")
    parser.add_argument("--addon-root", type=Path, default=ROOT)
    parser.add_argument("--addon-module", default="fluidSim_warp_blender51")
    parser.add_argument("--deadline", type=float, default=360)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if not 0 < args.deadline <= 900:
        parser.error("--deadline must be positive and no greater than 900 seconds")
    return args


def _fixture_helpers():
    spec = importlib.util.spec_from_file_location("fspt_live_base_fixture", P1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_fixture(bpy, base, scene):
    """Keep the original demo's domain, simulation values and source objects."""
    fs = scene.fluidsim
    fs.liquid_view_mode = "SURFACE"
    fs.view_surface = True
    fs.view_particles = False
    fs.show_whitewater = False
    fs.surface_attribute_mode = "NONE"
    fs.liquid_preview_enabled = False
    glass, node = base.material("FSPT cove water", "ShaderNodeBsdfGlass", (0.9, 0.98, 1.0))
    node.inputs["IOR"].default_value = 1.333
    fs.surface_material = glass

    terrain, _ = base.material("FSPT cove stone", "ShaderNodeBsdfDiffuse", (0.46, 0.31, 0.19))
    apron, _ = base.material("FSPT cove apron", "ShaderNodeBsdfDiffuse", (0.30, 0.40, 0.40))
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.hide_render = obj.hide_viewport = True
        elif obj.type == "MESH" and not obj.name.startswith("FluidSim "):
            role = getattr(getattr(obj, "fluidsim", None), "role", "NONE")
            if role not in {"DOMAIN", "SOURCE", "SINK", "FORCE"}:
                # Retain owned mesh and collider IDs while replacing shader slots.
                mat = apron if "Apron" in obj.name else terrain
                if not len(obj.material_slots):
                    obj.data.materials.append(mat)
                for slot in obj.material_slots:
                    slot.link = "OBJECT"
                    slot.material = mat

    world = bpy.data.worlds.new("FSPT cove constant world")
    world.use_nodes = True
    sky = world.node_tree.nodes["Background"]
    sky.inputs["Color"].default_value = (0.23, 0.35, 0.60, 1)
    sky.inputs["Strength"].default_value = 1
    scene.world = world

    # Out of the camera view, a mesh emitter lights the rocks and water.
    emitter, node = base.material("FSPT cove soft light", "ShaderNodeEmission", (1.0, 0.79, 0.55))
    node.inputs["Strength"].default_value = 3.0
    mesh = bpy.data.meshes.new("FSPT cove soft light")
    mesh.from_pydata([(-1, -2, 4), (-1, 2, 4), (5, 2, 4), (5, -2, 4)], [], [(0, 1, 2, 3)])
    mesh.update()
    mesh.materials.append(emitter)
    scene.collection.objects.link(bpy.data.objects.new(mesh.name, mesh))

    assert scene.camera is not None, "The saved demo must supply its authored camera"
    scene.camera.data.clip_start = 0.1
    scene.camera.data.clip_end = 100
    scene.render.engine = base.ENGINE
    scene.sync_mode = "NONE"
    settings = scene.fspt
    settings.device = "cuda:0"
    settings.viewport_samples = 1
    settings.resolution_percentage = 25
    settings.max_bounces = 6
    settings.seed = 31


def _mesh_arrays(mesh):
    mesh.calc_loop_triangles()
    points = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", points.ravel())
    triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", triangles.ravel())
    return points, triangles


def _evaluated_geometry(bpy, surface):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = surface.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        points, triangles = _mesh_arrays(mesh)
        transform = np.asarray(evaluated.matrix_world, dtype=np.float64)
        points = (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
        if np.linalg.det(transform[:3, :3]) < 0:
            triangles = triangles[:, [0, 2, 1]]
        return surface_digest(points, triangles), len(points), len(triangles)
    finally:
        evaluated.to_mesh_clear()


class LiveAcceptance:
    def __init__(self, bpy, base, args):
        self.bpy, self.base, self.args = bpy, base, args
        self.path = args.json_out.resolve()
        self.folder = self.path.with_name(self.path.stem + "-originals")
        self.started = time.monotonic()
        self.report = {"status": "RUNNING", "stage": "load", "frames": [], "controls": {}}
        self.latest = None
        self.prior_pixels = None
        self.frozen_payload = None
        self.stage = "ready"
        self.playing_states = set()
        self.target_reused_frame = None
        self.last_observation_issue = None

    def prepare(self):
        bpy = self.bpy
        assert not bpy.app.background, "Requires a real Blender window"
        assert self.args.blend.is_file(), f"Missing authored demo: {self.args.blend}"
        self.folder.mkdir(parents=True, exist_ok=False)
        self.report.update(verifier_sha256=self.base.sha(Path(__file__)),
                           fixture_helper_sha256=self.base.sha(P1),
                           input_blend=str(self.args.blend.resolve()),
                           input_blend_sha256=self.base.sha(self.args.blend))
        self.addon = self.base.load_addon(self.args, self.report)

        def module(name):
            return importlib.import_module(f"{self.addon.__name__}.{name}")

        self.bridge = module("blender.scene_bridge")
        self.surface = module("blender.surface_publication")
        self.output = module("blender.output_publication")
        self.geometry_nodes = module("blender.geometry_nodes")
        self.pending_settings = module("blender.pending_settings")
        self.engine = module("blender.fspt_engine")
        self.backend = module("fspt.backend")
        self.report["runtime"] = self.base.runtime_info(self.addon)
        assert self.report["runtime"]["warp"] == "1.19.0", self.report["runtime"]
        bpy.ops.wm.open_mainfile(filepath=str(self.args.blend.resolve()), load_ui=False)
        scene = bpy.context.scene
        assert scene.fluidsim.live and scene.frame_current == scene.frame_start
        assert not scene.fluidsim.cache_playback_enabled
        assert self.bridge.active_domain_settings(scene) is not None
        sources = [obj for obj in scene.objects if obj.fluidsim.role == "SOURCE" and obj.fluidsim.enabled]
        assert sources and any("DamBreak Source" in obj.name for obj in sources), "No enabled dam-break source"
        domain = self.bridge.active_domain_settings(scene)
        assert domain.surface_enabled and domain.live_surface_publish_cadence == 1
        self.report["demo"] = {"scene": scene.name, "start": scene.frame_start,
                               "sources": sorted(obj.name for obj in sources),
                               "voxel_size": domain.voxel_size,
                               "whitewater_simulation_enabled": domain.whitewater_enabled,
                               "hidden_presentation": ["points", "whitewater", "Blender light objects"]}
        _prepare_fixture(bpy, self.base, scene)
        self.output.refresh_existing_output_settings(scene)
        self.prepared = self.folder / "dam_break-fspt-live.blend"
        assert bpy.ops.wm.save_as_mainfile(filepath=str(self.prepared), copy=True) == {"FINISHED"}
        self.report["prepared_blend"] = str(self.prepared)
        self.report["prepared_blend_sha256"] = self.base.sha(self.prepared)
        self.report["viewport"] = {"resolution_percentage": scene.fspt.resolution_percentage,
                                   "seed": scene.fspt.seed, "max_bounces": scene.fspt.max_bounces,
                                   "live_samples": 1, "paused_samples": REFINED_SAMPLES,
                                   "route": "EVALUATED_MESH"}
        self.render_spy = patch.object(self.backend.CudaBackend, "render", self.observe_render())
        self.render_spy.start()
        self.base.checkpoint(self.path, self.report, "viewport_setup")
        self.viewport()

    @property
    def scene(self):
        return self.bpy.context.scene

    def viewport(self):
        bpy = self.bpy
        self.window = bpy.context.window_manager.windows[0]
        self.area = next(area for area in self.window.screen.areas if area.type == "VIEW_3D")
        self.region = next(region for region in self.area.regions if region.type == "WINDOW")
        space = self.area.spaces.active
        space.overlay.show_overlays = False
        space.clip_start, space.clip_end = 0.1, 100
        space.region_3d.view_perspective = "CAMERA"
        space.shading.type = "RENDERED"
        self.area.tag_redraw()

    def observe_render(self):
        original = self.backend.CudaBackend.render

        def render(backend, snapshot, camera, width, height, **kwargs):
            frame = original(backend, snapshot, camera, width, height, **kwargs)
            if len(frame.publications) == 1:
                # Only the demo's fluid has glass. Select the actual submitted
                # face rows, including connectivity and their world positions.
                glass = [i for i, material in enumerate(snapshot.materials) if material.kind == "glass"]
                assert len(glass) == 1, f"Expected one identifiable fluid material, found {glass}"
                faces = snapshot.triangles[snapshot.material_indices == glass[0]]
                digest = surface_digest(snapshot.vertices, faces)
                self.latest = (frame, digest, len(faces))
            return frame

        return render

    def override(self):
        return self.bpy.context.temp_override(window=self.window, screen=self.window.screen,
                                              area=self.area, region=self.region)

    def play(self):
        assert not self.window.screen.is_animation_playing
        with self.override():
            assert self.bpy.ops.screen.animation_play() == {"FINISHED"}
        assert self.window.screen.is_animation_playing

    def pause(self):
        assert self.window.screen.is_animation_playing
        with self.override():
            assert self.bpy.ops.screen.animation_cancel(restore_frame=False) == {"FINISHED"}
        assert not self.window.screen.is_animation_playing

    def observe(self, *, samples=1):
        scene, sess = self.scene, self.bridge.session()
        publication = self.surface.current_surface_publication(scene)
        if not publication or not publication.visible or publication.triangle_count == 0:
            return None
        info = dict(self.engine.LAST_VIEWPORT_INFO)
        if not self.latest or info.get("status") != "COMPLETE" or info.get("samples") != samples:
            return None
        frame, renderer_geometry, renderer_triangles = self.latest
        if tuple(info["key"]) != frame.key or frame.samples != samples:
            return None
        if frame.publications != (publication,) or frame.scene_frame != scene.frame_current:
            return None
        if not sess.owns_scene(scene) or publication.solver_frame != sess.solver.frame:
            return None
        self.base.validate_info(info)
        assert info["scene_frame"] == publication.timeline_frame == scene.frame_current
        assert publication.channel == "EVALUATED_MESH" and publication.source == "LIVE"
        sources = info.get("sources")
        assert sources and len(sources) == 1, "Presented telemetry has no submitted fluid source"
        for name in ("scene_token", "owner_token", "epoch", "channel", "generation", "source",
                     "timeline_frame", "solver_frame", "state_revision"):
            assert json.dumps(sources[0][name]) == json.dumps(getattr(publication, name)), (
                f"Presented {name} did not match the submitted source")
        assert self.window.scene == scene
        surface = self.output.output_object(scene, self.output.SURFACE_NAME, ensure=False)
        geometry, vertices, triangles = _evaluated_geometry(self.bpy, surface)
        assert geometry == renderer_geometry, "Submitted fluid is different from the current evaluated surface"
        assert triangles == renderer_triangles == publication.triangle_count
        pixels = np.array(frame.pixels, copy=True)
        assert pixels.ndim == 3 and pixels.shape[-1] == 4 and np.isfinite(pixels).all()
        row = {key: getattr(publication, key) for key in (
            "timeline_frame", "solver_frame", "state_revision", "simulation_time", "epoch",
            "generation", "build_key", "channel", "source", "scene_token", "owner_token")}
        row.update(geometry_sha256=geometry, rendered_surface_sha256=renderer_geometry,
                   image_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                   evaluated_vertex_count=vertices, evaluated_triangle_count=triangles,
                   animation_playing=bool(self.window.screen.is_animation_playing), telemetry=info)
        return row, pixels

    def save_observation(self, label, row, pixels, *, screenshot=False):
        path = self.folder / f"{label}.npy"
        np.save(path, pixels, allow_pickle=False)
        row.update(raw_pixels=str(path), raw_pixels_sha256=self.base.sha(path))
        if screenshot:
            path = self.folder / f"{label}-screenshot.png"
            with self.override():
                assert self.bpy.ops.screen.screenshot(filepath=str(path)) == {"FINISHED"}
            assert path.stat().st_size > 0
            row.update(screenshot=str(path), screenshot_sha256=self.base.sha(path))

    def accept_live(self):
        result = self.observe()
        if result is None:
            self.area.tag_redraw()
            return
        row, pixels = result
        frames = self.report["frames"]
        if frames and row["state_revision"] <= frames[-1]["state_revision"]:
            return
        assert row["animation_playing"], "The state did not come from ordinary Blender GUI playback"
        if frames:
            assert row["telemetry"]["key"][1:] == frames[-1]["telemetry"]["key"][1:], (
                "Visual proof must keep camera, resolution and trace settings fixed")
            row["image_mean_abs_delta"] = verify_advance(frames[-1], row, self.prior_pixels, pixels)
        identity = (row["epoch"], row["state_revision"])
        assert identity not in self.playing_states, "A solved state was counted twice"
        self.playing_states.add(identity)
        number = len(frames) + 1
        self.save_observation(f"live-{number:02d}", row, pixels, screenshot=number in {1, LIVE_STATES})
        self.prior_pixels = pixels
        frames.append(row)
        self.base.checkpoint(self.path, self.report, f"live_{number:02d}")

        if number == PAUSE_AFTER:
            self.pause()
            self.scene.fspt.viewport_samples = REFINED_SAMPLES
            self.stage = "refine"
            self.area.tag_redraw()
        elif number == LIVE_STATES:
            self.pause()
            assert len({item["geometry_sha256"] for item in frames}) == LIVE_STATES
            assert len({item["image_sha256"] for item in frames}) == LIVE_STATES
            self.start_freeze()

    def refine(self):
        result = self.observe(samples=REFINED_SAMPLES)
        if result is None:
            self.area.tag_redraw()
            return
        row, pixels = result
        live = self.report["frames"][-1]
        assert not row["animation_playing"]
        for key in ("timeline_frame", "solver_frame", "state_revision", "epoch", "generation",
                    "geometry_sha256"):
            assert row[key] == live[key], f"Paused refinement changed {key}"
        assert row["telemetry"]["key"] == live["telemetry"]["key"]
        assert row["telemetry"]["draw_count"] > live["telemetry"]["draw_count"]
        assert row["image_sha256"] != live["image_sha256"]
        self.save_observation("paused-refinement", row, pixels, screenshot=True)
        self.report["controls"]["pause_refine"] = row
        self.scene.fspt.viewport_samples = 1
        self.stage = "live"
        self.play()

    def start_freeze(self):
        original = self.output.prepare_live_payload
        obj = self.output.output_object(self.scene, self.output.SURFACE_NAME, ensure=False)
        points, triangles = _mesh_arrays(obj.data)
        frozen_digest = surface_digest(points, triangles)
        self.fresh_payloads = []

        def freeze(*args, **kwargs):
            payload = original(*args, **kwargs)
            if not payload.publish_surface or not len(payload.surface_tris):
                return payload
            incoming = surface_digest(payload.surface_verts, payload.surface_tris)
            assert incoming != frozen_digest, "The injected fault must mask an actual surface change"
            self.fresh_payloads.append(incoming)
            # Real publication, real eligibility metadata, and a real FSPT
            # draw follow, but they carry the prior geometry and its counts.
            return replace(payload, surface_verts=points, surface_tris=triangles,
                           surface_attributes={})

        self.frozen_payload = patch.object(self.output, "prepare_live_payload", freeze)
        self.frozen_payload.start()
        self.stage = "freeze"
        self.play()

    def reject_frozen(self):
        if not self.fresh_payloads:
            return
        row_pixels = self.observe()
        if row_pixels is None:
            self.area.tag_redraw()
            return
        row, pixels = row_pixels
        prior = self.report["frames"][-1]
        if row["state_revision"] <= prior["state_revision"]:
            return
        self.pause()
        self.frozen_payload.stop()
        self.frozen_payload = None
        try:
            verify_advance(prior, row, self.prior_pixels, pixels)
        except FrozenSurfaceError as error:
            assert row["image_sha256"] == prior["image_sha256"], "Frozen geometry changed fixed-seed pixels"
            self.save_observation("frozen-control", row, pixels)
            self.report["controls"]["freeze"] = {
                "rejected": str(error), "fault_hits": len(self.fresh_payloads),
                "suppressed_payload_geometry_sha256": self.fresh_payloads,
                "prior": prior, "observed": row}
        else:
            raise AssertionError("Accepted a surface after its publication geometry was frozen")
        self.stage = "publication_controls"

    def publication_controls(self):
        scene, sess = self.scene, self.bridge.session()
        domain = self.bridge.active_domain_settings(scene)
        solver_frame = sess.solver.frame
        cadence = next(value for value in range(2, 9) if solver_frame % value)
        # Setup for the control only. The accepted 24 frames ran with the
        # demo's original cadence. Use public RNA, then verify the real policy.
        with self.pending_settings.suppress_callbacks(scene):
            domain.live_surface_publish_cadence = cadence
        assert not sess.solver.host_frame().should_publish_surface(cadence)
        original = self.output._swap_mesh
        target = self.output.output_object(scene, self.output.SURFACE_NAME, ensure=False)
        previous_geometry = surface_digest(*_mesh_arrays(target.data))
        observed_geometry = None

        class InjectedSurfaceMutation(RuntimeError):
            pass

        def fail_after_swap(obj, *args, **kwargs):
            nonlocal observed_geometry
            result = original(obj, *args, **kwargs)
            if obj == target:
                observed_geometry = surface_digest(*_mesh_arrays(obj.data))
                assert observed_geometry != previous_geometry, "The real mesh content did not change"
                raise InjectedSurfaceMutation("FSPT live acceptance after real surface mutation")
            return result

        with patch.object(self.output, "_swap_mesh", fail_after_swap):
            try:
                self.output.update_outputs(scene, sess, force_surface=True)
            except InjectedSurfaceMutation:
                pass
            else:
                raise AssertionError("Post-mutation fault did not reach the real mesh replacement")
        failed = self.surface.get_surface_state(scene)
        assert failed.quarantined and not failed.eligible
        assert self.surface.current_surface_publication(scene) is None
        self.output.update_outputs(scene, sess)
        skipped = self.surface.get_surface_state(scene)
        assert skipped.quarantined and not skipped.eligible
        assert skipped.committed == failed.committed
        assert self.surface.current_surface_publication(scene) is None
        self.output.update_outputs(scene, sess, force_surface=True)
        recovered = self.surface.current_surface_publication(scene)
        assert recovered is not None and not self.surface.get_surface_state(scene).quarantined
        assert recovered.generation > failed.committed.generation
        with self.pending_settings.suppress_callbacks(scene):
            domain.live_surface_publish_cadence = 1
        self.report["controls"]["mutation_cadence"] = {
            "fault": "InjectedSurfaceMutation", "solver_frame": solver_frame, "cadence": cadence,
            "failed": failed.status, "failed_reason": failed.reason, "skip": skipped.status,
            "skip_reason": skipped.reason, "prior_generation": failed.committed.generation,
            "previous_geometry": previous_geometry, "mutated_geometry": observed_geometry,
            "recovered_generation": recovered.generation}
        self.base.checkpoint(self.path, self.report, "sync_warning")

        def fail_sync(_scene):
            raise RuntimeError("FSPT live sync control")

        with patch.object(self.geometry_nodes, "sync_render_inputs", fail_sync):
            self.output.update_outputs(scene, sess, force_surface=True)
        warning = self.surface.get_surface_state(scene)
        assert warning.pending is not None and not warning.quarantined
        assert not warning.eligible and warning.status == "PENDING"
        assert "FSPT live sync control" in warning.reason
        assert self.surface.current_surface_publication(scene) is None
        self.output.refresh_existing_output_settings(scene)
        synced = self.surface.current_surface_publication(scene)
        assert synced == warning.pending
        assert sess.solver.frame == solver_frame, "Output controls advanced the simulation"
        self.report["controls"]["sync_warning"] = {
            "warning": warning.reason, "attempt": warning.attempt, "generation": synced.generation,
            "solver_frame": solver_frame}
        self.base.checkpoint(self.path, self.report, "rewind")
        self.target_reused_frame = self.report["frames"][0]
        scene.frame_set(scene.frame_start)
        current = self.surface.current_surface_publication(scene)
        assert current is None or current.epoch != synced.epoch, "Rewind retained the retired solver epoch"
        for frame in range(scene.frame_start + 1, self.target_reused_frame["timeline_frame"] + 1):
            scene.frame_set(frame)
        self.stage = "reused_frame"
        self.area.tag_redraw()

    def reject_reused_frame(self):
        result = self.observe()
        if result is None:
            self.area.tag_redraw()
            return
        row, pixels = result
        prior = self.target_reused_frame
        assert not row["animation_playing"]
        assert row["epoch"] != prior["epoch"], "Reset reused the prior solver epoch"
        assert row["timeline_frame"] == prior["timeline_frame"], "The repeated timeline label did not match"
        assert row["telemetry"]["key"] != prior["telemetry"]["key"], "Accepted an image from the old epoch"
        self.save_observation("reset-reused-frame", row, pixels, screenshot=True)
        self.report["controls"]["reset"] = {"prior": prior, "observed": row}
        self.base.checkpoint(self.path, self.report, "reopen")
        self.latest = None
        self.engine.LAST_VIEWPORT_INFO = {}
        assert self.bpy.ops.wm.open_mainfile(filepath=str(self.prepared), load_ui=False) == {"FINISHED"}
        assert self.scene.frame_current == self.scene.frame_start and self.scene.fluidsim.live
        self.viewport()
        self.stage = "reopen_ready"

    def reopen(self):
        result = self.observe()
        if result is None:
            self.area.tag_redraw()
            return
        row, pixels = result
        sess = self.bridge.session()
        particle_count = sess.solver.particles.refresh_host_count()
        assert row["animation_playing"] and sess.owns_scene(self.scene) and particle_count > 0
        prior = self.report["controls"]["reset"]["observed"]
        assert row["epoch"] != prior["epoch"] or row["scene_token"] != prior["scene_token"]
        self.pause()
        self.save_observation("reopen-live", row, pixels, screenshot=True)
        self.report["controls"]["reopen"] = {"particle_count": particle_count, "observed": row}
        assert len(self.report["frames"]) == LIVE_STATES
        self.finish()

    def tick(self):
        try:
            if time.monotonic() - self.started > self.args.deadline:
                raise TimeoutError(f"Timed out in {self.stage}: {self.engine.LAST_VIEWPORT_INFO!r}")
            if self.stage in {"live", "freeze", "reopen"}:
                assert self.scene.fluidsim.live, self.scene.fluidsim.status
                status = self.engine.LAST_VIEWPORT_INFO.get("status")
                if status == "ERROR":
                    issue = self.engine.LAST_VIEWPORT_INFO.get("error")
                    if issue != self.last_observation_issue:
                        self.last_observation_issue = issue
                        self.report["last_viewport_error"] = issue
                        self.base.checkpoint(self.path, self.report, self.stage)
            if self.stage in {"ready", "reopen_ready"}:
                self.stage = "live" if self.stage == "ready" else "reopen"
                self.play()
            elif self.stage == "live":
                self.accept_live()
            elif self.stage == "refine":
                self.refine()
            elif self.stage == "freeze":
                self.reject_frozen()
            elif self.stage == "publication_controls":
                self.publication_controls()
            elif self.stage == "reused_frame":
                self.reject_reused_frame()
            elif self.stage == "reopen":
                self.reopen()
            if self.stage != "complete":
                return 0.005
        except BaseException:  # noqa: BLE001 - record failure and close this owned GUI process
            self.fail()
        return None

    def finish(self):
        self.render_spy.stop()
        self.area.spaces.active.shading.type = "SOLID"
        self.scene.render.engine = "BLENDER_WORKBENCH"
        self.addon.unregister()
        self.stage = "complete"
        self.report.update(status="VERIFIED", stage=self.stage,
                           elapsed_seconds=time.monotonic() - self.started)
        self.base.write_report(self.path, self.report)
        print("FSPT-LIVE-ACCEPTANCE-PASS", self.path, flush=True)
        self.bpy.ops.wm.quit_blender()

    def fail(self):
        if self.frozen_payload is not None:
            self.frozen_payload.stop()
        self.report.update(status="FAILED", stage=self.stage, error=traceback.format_exc())
        self.base.write_report(self.path, self.report)
        print("FSPT-LIVE-ACCEPTANCE-FAIL", self.report["error"], flush=True)
        os._exit(1)


def main():
    import bpy

    args = _arguments()
    harness = LiveAcceptance(bpy, _fixture_helpers(), args)
    try:
        harness.prepare()
        bpy.app.timers.register(harness.tick, first_interval=0.05, persistent=True)
    except BaseException:  # noqa: BLE001 - record setup failure with an owned nonzero exit
        harness.fail()


if __name__ == "__main__":
    main()
