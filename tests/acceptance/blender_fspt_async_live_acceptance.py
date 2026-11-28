"""Asynchronous live FSPT proof joined to independently audited capture geometry.

The original reference verifier remains strict about the current draw-time
generation. This verifier checks actual evaluated geometry at capture, retains
only its immutable audit receipt, and joins that receipt to the snapshot that
produced an actually presented image. The image's source labels never advance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import time
import traceback
from collections import Counter, OrderedDict
from pathlib import Path
from threading import local
from unittest.mock import patch

import numpy as np

REFERENCE = Path(__file__).with_name("blender_fspt_live_acceptance.py")
spec = importlib.util.spec_from_file_location("fspt_live_reference", REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--blend", type=Path, default=reference.ROOT / "demos/dam_break.blend")
    parser.add_argument("--addon-root", type=Path, default=reference.ROOT)
    parser.add_argument("--addon-module", default="fluidSim_warp_blender51")
    parser.add_argument("--states", type=int, default=240)
    parser.add_argument("--soak-seconds", type=float, default=0)
    parser.add_argument("--deadline", type=float, default=900)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if not 24 <= args.states <= 600 or not 0 <= args.soak_seconds <= 600 or not 0 < args.deadline <= 3600:
        parser.error("states must be 24..600, soak 0..600 seconds and deadline 0..3600 seconds")
    if args.soak_seconds and args.deadline <= args.soak_seconds:
        parser.error("deadline must leave time for the live controls before the soak")
    return args


def fluid_geometry(snapshot):
    glass = [i for i, material in enumerate(snapshot.materials) if material.kind == "glass"]
    assert len(glass) == 1, f"Expected exactly one identifiable fluid material, found {glass}"
    faces = snapshot.triangles[snapshot.material_indices == glass[0]]
    return reference.surface_digest(snapshot.vertices, faces), len(faces)


def join_geometry_audit(snapshot, audits):
    audit = audits.get(snapshot.digest)
    assert audit is not None, "Presented snapshot has no independent evaluated-geometry capture audit"
    actual, triangles = fluid_geometry(snapshot)
    assert actual == audit["geometry_sha256"], "Presented fluid differs from independently audited evaluated geometry"
    assert triangles == audit["evaluated_triangle_count"], "Presented triangle count differs from capture audit"
    return audit


def verify_soak_advance(prior, current, prior_pixels, pixels):
    assert current["epoch"] == prior["epoch"], "Soak crossed a solver epoch"
    for name in ("timeline_frame", "solver_frame", "state_revision", "generation"):
        assert current[name] > prior[name], f"Soak did not advance {name}"
    assert current["telemetry"]["key"][1:] == prior["telemetry"]["key"][1:], "Soak changed its ray/quality settings"
    assert pixels.shape == prior_pixels.shape and np.isfinite(pixels).all()
    delta = float(np.mean(np.abs(pixels[..., :3] - prior_pixels[..., :3])))
    if current["geometry_sha256"] != prior["geometry_sha256"]:
        assert delta > 1e-6, "Soak changed audited geometry but completed pixels remained frozen"
    return delta


class AsyncLiveAcceptance(reference.LiveAcceptance):
    def __init__(self, bpy, base, args):
        super().__init__(bpy, base, args)
        self.audits = OrderedDict()
        self.audit_count = 0
        self.guards = []
        self.callback = local()
        self.boundary_counts = Counter()
        self.soak_started = None
        self.soak_complete = False
        self.soak_last = None
        self.soak_last_draw = None
        self.soak_next_sample = 0

    def prepare(self):
        super().prepare()
        assert hasattr(self.engine, "viewport_observations"), "Asynchronous adapter is missing"
        self.report.update(verifier_sha256=self.base.sha(Path(__file__)),
                           reference_verifier_sha256=self.base.sha(REFERENCE),
                           requested_states=self.args.states,
                           requested_soak_seconds=self.args.soak_seconds,
                           proof_boundary="independent evaluated geometry at capture joined to completed draw",
                           performance_scope="instrumented functional proof; authoritative solver-ready age unmeasured")
        self.report["demo"]["authored_frame_end"] = int(self.scene.frame_end)
        # Extend only this private fixture's playback range to avoid an
        # unintended timeline rewind during the long ordinary-playback gate.
        self.scene.frame_end = max(self.scene.frame_end, self.args.states * 10,
                                   100000 if self.args.soak_seconds else 0)
        self.report["demo"]["acceptance_frame_end"] = int(self.scene.frame_end)
        adapter = importlib.import_module(self.addon.__name__ + ".blender.fspt_scene")
        original = adapter.extract_scene
        def capture(depsgraph, for_viewport=False):
            assert not getattr(self.callback, "drawing", False), "Drawing extracted scene geometry"
            snapshot = original(depsgraph, for_viewport=for_viewport)
            if for_viewport and snapshot.publications:
                try:
                    self.audit_capture(depsgraph, snapshot)
                except BaseException:
                    self.observer_error = traceback.format_exc()
                    raise
            return snapshot
        guard = patch.object(adapter, "extract_scene", capture)
        guard.start()
        self.guards.append(guard)
        backend = importlib.import_module(self.addon.__name__ + ".fspt.async_backend").AsyncCudaBackend
        for name in ("submit", "collect", "close"):
            original_method = getattr(backend, name)
            def checked(instance, *args, _name=name, _method=original_method, **kwargs):
                assert not getattr(self.callback, "drawing", False), f"Drawing called CUDA backend {_name}"
                if _name == "collect":
                    assert not kwargs.get("wait", False), "Ordinary live viewport waited for CUDA"
                self.boundary_counts[_name] += 1
                return _method(instance, *args, **kwargs)
            guard = patch.object(backend, name, checked)
            guard.start()
            self.guards.append(guard)
        self.base.checkpoint(self.path, self.report, "async_observer_ready")

    def audit_capture(self, depsgraph, snapshot):
        assert len(snapshot.publications) == 1
        stamp = snapshot.publications[0]
        assert stamp.channel == "EVALUATED_MESH" and stamp.source == "LIVE"
        scene = getattr(depsgraph.scene, "original", depsgraph.scene)
        assert self.surface.current_surface_publication(scene) == stamp
        session = self.bridge.session()
        assert session.owns_scene(scene) and session.surface_epoch == stamp.epoch
        surface = self.output.output_object(scene, self.output.SURFACE_NAME, ensure=False)
        geometry, vertices, triangles = reference._evaluated_geometry(depsgraph, surface)
        submitted, submitted_triangles = fluid_geometry(snapshot)
        assert geometry == submitted, "Captured FSPT fluid differs from independent evaluated mesh read"
        assert triangles == submitted_triangles == stamp.triangle_count
        solver = session.solver
        audit = {"snapshot_digest": snapshot.digest, "geometry_sha256": geometry,
                 "evaluated_vertex_count": vertices, "evaluated_triangle_count": triangles,
                 "captured_at_observer": time.monotonic(), "source_generation": stamp.generation,
                 "source_epoch": stamp.epoch, "source_scene": stamp.scene_token,
                 "solver_frame_at_capture": int(solver.frame),
                 "state_revision_at_capture": int(solver.liquid_state_revision)}
        if snapshot.digest in self.audits:
            assert self.audits[snapshot.digest]["geometry_sha256"] == geometry
        else:
            self.audits[snapshot.digest] = audit
            self.audit_count += 1
        while len(self.audits) > 64:
            self.audits.popitem(last=False)

    def observe_draw(self):
        original = self.engine.FSPTRenderEngine.view_draw
        def draw(renderer, context, depsgraph):
            before = self.engine._DRAW_COUNT
            draw_started = time.monotonic()
            self.callback.drawing = True
            try:
                original(renderer, context, depsgraph)
            finally:
                self.callback.drawing = False
            if context.window != self.window or context.area != self.area or context.region != self.region:
                return
            if (self.stage not in {"live", "pause_base", "refine", "freeze_base", "freeze",
                                  "reused_frame", "reopen", "soak"} or self.observer_error):
                return
            self.report["draw_observer"]["entered"] += 1
            try:
                info = dict(renderer._view_info)
                view = renderer._view
                if info.get("status") != "COMPLETE" or self.engine._DRAW_COUNT <= before or view is None:
                    self.drop("draw_not_complete", info.get("error", info.get("reason")))
                    return
                frame, snapshot = view.presentable, view.presented_snapshot
                if frame is None or snapshot is None or renderer._texture_frame is not frame:
                    self.drop("draw_missing_owned_frame")
                    return
                identity = self.stage, frame.key, frame.samples
                if identity == self.last_draw_identity:
                    return
                observation = self.capture_owned_draw(context, depsgraph, frame, snapshot, info, draw_started)
                if observation is not None:
                    self.last_draw_identity = identity
                    if len(self.pending) == self.pending.maxlen:
                        self.drop("observer_backlog")
                    self.pending.append(observation)
                    self.report["draw_observer"]["matched"] += 1
            except BaseException:
                self.observer_error = traceback.format_exc()
        return draw

    def capture_owned_draw(self, context, depsgraph, frame, snapshot, info, draw_started):
        if not frame.publications:
            self.drop("no_visible_publication")
            return None
        assert len(frame.publications) == 1 and snapshot.publications == frame.publications
        stamp = frame.publications[0]
        audit = join_geometry_audit(snapshot, self.audits)
        assert (audit["source_generation"], audit["source_epoch"], audit["source_scene"]) == (
            stamp.generation, stamp.epoch, stamp.scene_token)
        scene = getattr(depsgraph.scene, "original", depsgraph.scene)
        assert context.window.scene == scene and self.surface.id_token(scene) == stamp.scene_token
        bridge = importlib.import_module(self.addon.__name__ + ".blender.fspt_bridge")
        # Validate the independent start of the observed draw, rather than
        # moving its clock forward by this verifier's image/geometry hashing.
        presentation = bridge.validate_viewport_publications(
            scene, frame.publications, view_layer=depsgraph.view_layer,
            clock=lambda: draw_started)
        self.base.validate_info(info)
        assert info["backend"] == "warp-cuda-async-v1" and info["transport"] == "host-pinned-rgba32f"
        assert tuple(info["key"]) == frame.key and info["samples"] == frame.samples
        assert info["scene_frame"] == frame.scene_frame == snapshot.frame == stamp.timeline_frame
        sources = info.get("sources")
        assert sources and len(sources) == 1
        for name in ("scene_token", "owner_token", "epoch", "channel", "generation", "source",
                     "timeline_frame", "source_timeline_frame", "solver_frame", "state_revision"):
            assert json.dumps(sources[0][name]) == json.dumps(getattr(stamp, name)), name
        pixels = np.array(frame.pixels, copy=True)
        assert pixels.ndim == 3 and pixels.shape[-1] == 4 and np.isfinite(pixels).all()
        row = {name: getattr(stamp, name) for name in (
            "timeline_frame", "source_timeline_frame", "solver_frame", "state_revision", "simulation_time", "epoch",
            "generation", "build_key", "channel", "source", "scene_token", "owner_token")}
        row.update(geometry_sha256=audit["geometry_sha256"], rendered_surface_sha256=audit["geometry_sha256"],
                   image_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                   evaluated_vertex_count=audit["evaluated_vertex_count"],
                   evaluated_triangle_count=audit["evaluated_triangle_count"], capture_audit=dict(audit),
                   presented_at_observer=draw_started, presentation_status=presentation["status"],
                   publication_to_draw_seconds=presentation["publication_age_seconds"],
                   requested_frame_at_draw=int(scene.frame_current),
                   animation_playing=bool(context.window.screen.is_animation_playing), telemetry=info)
        if self.stage == "reopen":
            row["solver_particle_count"] = self.bridge.session().solver.particles.refresh_host_count()
        pixels.setflags(write=False)
        return row, pixels

    def observe(self, *, samples=1):
        while True:
            result = super().observe(samples=samples)
            if result is None:
                return None
            if self.stage in {"pause_base", "refine", "freeze_base", "reused_frame"} and (
                    result[0]["presentation_status"] != "CURRENT"
                    or result[0]["timeline_frame"] != self.scene.frame_current):
                self.drop("paused_control_requires_current_source")
                continue
            return result

    def diagnostic(self):
        self.report["capture_audit_count"] = self.audit_count
        self.report["async_boundary_calls"] = dict(self.boundary_counts)
        super().diagnostic()

    def finish(self):
        if self.args.soak_seconds and not self.soak_complete:
            self.soak_started = self.soak_last_draw = time.monotonic()
            self.soak_next_sample = self.soak_started
            self.report["soak"] = {"status": "RUNNING", "new_states": 0, "changed_images": 0,
                                   "samples": [], "maximum_image_capacity": 0}
            self.scene.frame_end = max(self.scene.frame_end, 100000)
            self.change_stage("soak")
            self.play()
            return
        self.report["capture_audit_count"] = self.audit_count
        self.report["async_boundary_calls"] = dict(self.boundary_counts)
        for guard in reversed(self.guards):
            guard.stop()
        self.draw_spy.stop()
        self.area.spaces.active.shading.type = "SOLID"
        self.scene.render.engine = "BLENDER_WORKBENCH"
        self.addon.unregister()
        scheduler = importlib.import_module(self.addon.__name__ + ".fspt.scheduler")
        arbiter = scheduler.get_arbiter("cuda:0")
        assert arbiter._active is None and not arbiter._viewport_waiters and not arbiter._final_waiters
        self.stage = "complete"
        self.report.update(status="VERIFIED", stage=self.stage,
                           elapsed_seconds=time.monotonic() - self.started)
        self.base.write_report(self.path, self.report)
        print("FSPT-ASYNC-LIVE-ACCEPTANCE-PASS", self.path, flush=True)
        self.bpy.ops.wm.quit_blender()

    def tick(self):
        if self.stage != "soak":
            return super().tick()
        try:
            now = time.monotonic()
            if self.observer_error:
                raise RuntimeError(self.observer_error)
            assert now - self.started <= self.args.deadline, "Soak exceeded the owned deadline"
            assert self.scene.fluidsim.live and self.window.screen.is_animation_playing
            result = self.observe()
            if result is not None:
                row, pixels = result
                identity = row["epoch"], row["state_revision"]
                prior_identity = ((self.soak_last[0]["epoch"], self.soak_last[0]["state_revision"])
                                  if self.soak_last is not None else None)
                if identity != prior_identity:
                    if self.soak_last is not None:
                        prior, prior_pixels = self.soak_last
                        delta = verify_soak_advance(prior, row, prior_pixels, pixels)
                        if row["geometry_sha256"] != prior["geometry_sha256"] and delta > 1e-6:
                            self.report["soak"]["changed_images"] += 1
                    self.soak_last = row, pixels
                    self.soak_last_draw = now
                    self.report["soak"]["new_states"] += 1
                    if now >= self.soak_next_sample:
                        self.save_observation(f"soak-{int(now - self.soak_started):03d}", row, pixels, screenshot=True)
                        self.report["soak"]["samples"].append(row)
                        self.soak_next_sample = now + 60
            assert now - self.soak_last_draw < 10, "Soak made no new completed live-image progress for 10 seconds"
            for renderer in tuple(self.engine._ENGINES):
                view = renderer._view
                if view is not None and view.backend is not None:
                    assert not view.error, view.error
                    capacity = view.backend._capacity
                    self.report["soak"]["maximum_image_capacity"] = max(
                        capacity, self.report["soak"]["maximum_image_capacity"])
            if now >= self.diagnostic_at:
                self.diagnostic_at = now + 1
                self.diagnostic()
            if now - self.soak_started >= self.args.soak_seconds:
                assert self.report["soak"]["new_states"] >= 2 and self.report["soak"]["changed_images"] > 0, (
                    "Soak did not produce independently changing fluid images")
                self.pause()
                self.soak_complete = True
                self.report["soak"].update(status="VERIFIED", elapsed_seconds=now - self.soak_started)
                self.finish()
                return None
            return 0.005
        except BaseException:
            self.fail()


def main():
    import bpy

    args = arguments()
    reference.LIVE_STATES = args.states
    run = AsyncLiveAcceptance(bpy, reference._fixture_helpers(), args)
    try:
        run.prepare()
        bpy.app.timers.register(run.tick, first_interval=0.05, persistent=True)
    except BaseException:
        run.fail()


if __name__ == "__main__":
    main()
