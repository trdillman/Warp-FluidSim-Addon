"""Qualify native GPU publication and the GUI-worker Final ownership handoff.

A controlled ViewportState supplies demand and capture at an explicit,
evaluated Blender boundary. The solver, publication owner, static capture,
broker, CUDA renderer, Blender render operator, and engine's app timer are real.
Ordinary rendered-region GPU eligibility and viewport display are outside this
proof.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from threading import current_thread, main_thread
from time import monotonic
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_fspt_acceptance as base
import blender_fspt_device_bridge_acceptance as producer_fixture
import blender_fspt_device_surface_acceptance as surface_fixture

ENV = "FLUIDSIM_FSPT_DEVICE_SURFACE"
WIDTH, HEIGHT = 128, 96
TIMELINE = (2, 3)
SOURCES = (
    "blender.fspt_engine", "blender.fspt_device_runtime", "blender.fspt_device_bridge",
    "blender.fspt_device_scene", "blender.fspt_bridge", "blender.output_publication",
    "blender.surface_publication", "blender.scene_bridge", "fspt.viewport",
    "fspt.viewport_backend", "fspt.final_handoff", "fspt.device_scene",
    "fspt.device_backend", "fspt.surface_pool", "fspt.async_backend",
)
FINAL_ORDER = (
    "render_enter", "retirement_enter", "retirement_receipt", "final_backend_allocate",
    "final_backend_allocated", "final_backend_close_enter", "final_backend_closed",
    "render_exit",
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--addon-root", type=Path, default=base.ROOT)
    parser.add_argument("--addon-module", default="fluidSim_warp_blender51")
    parser.add_argument("--deadline", type=float, default=240.0)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if not np.isfinite(args.deadline) or args.deadline <= 0:
        parser.error("--deadline must be positive and finite")
    return args


def thread_facts():
    return {"thread": current_thread().ident, "on_main": current_thread() is main_thread()}


def verify_handoff(final, delta):
    ordered = [event for event in final["events"] if event["event"] in FINAL_ORDER]
    assert tuple(event["event"] for event in ordered) == FINAL_ORDER, ordered
    enter, retire, receipt, allocate, allocated, close, closed, leave = ordered
    assert not enter["on_main"] and enter["thread"] is not None, enter
    worker = enter["thread"]
    assert all(not entry["on_main"] and entry["thread"] == worker
               for entry in (allocate, allocated, close, closed, leave)), ordered
    assert retire["on_main"] and receipt["on_main"], ordered
    assert retire["thread"] == receipt["thread"] != worker, ordered
    assert final["viewport_before"]["trace_pending"], final["viewport_before"]
    assert retire["view_owned"] and retire["aggregate_open"] and not retire["trace_pending"], retire
    assert retire["reserved_bytes"] > 0 and retire["generation_count"] > 0, retire
    assert receipt["request_matches"] and receipt["owners_retired"], receipt
    assert receipt["reserved_bytes"] == receipt["generation_count"] == 0, receipt
    assert not receipt["resources_remaining"] and not receipt["view_owned"], receipt
    assert all(entry["hold_active"] and entry["coordinator_blocked"]
               for entry in (retire, receipt, allocate, allocated, close, closed)), ordered
    assert all(not entry["resources_remaining"] for entry in (allocate, allocated)), ordered
    assert not leave["hold_active"] and not leave["coordinator_blocked"], leave
    assert allocated["backend_type"] == "AsyncCudaBackend", allocated
    assert closed["backend_closed"] and closed["render_stream_retirement"] == 0, closed
    assert delta == {"requests": 1, "serviced": 1, "timeouts": 0, "failures": 0}, delta
    events = tuple(event["event"] for event in final["events"])
    assert events.count("render_complete") == 1 and "render_cancel" not in events, events


class Case:
    def __init__(self, args, path, report):
        self.args, self.path, self.report = args, path, report
        self.deadline = monotonic() + args.deadline
        self.previous_flag = os.environ.get(ENV)
        self.addon = self.view = self.withdraw = self.session = None
        self.final_backend = None
        self.patches, self.handlers = [], []
        self.frame_index, self.selection = 0, None
        self.prior_pixels = None
        self.stage = "setup"

    def modules(self, name):
        return importlib.import_module(self.addon.__name__ + "." + name)

    def checkpoint(self, stage):
        self.stage = stage
        base.checkpoint(self.path, self.report, stage)

    def artifact(self, name):
        path = self.path.with_name(f"{self.path.stem}-{name}")
        assert not path.exists(), f"Use a fresh evidence destination: {path}"
        return path

    def setup(self):
        assert not bpy.app.background, "A real Blender GUI is required"
        bpy.context.preferences.view.show_splash = False
        bpy.context.preferences.use_preferences_save = False
        os.environ[ENV] = "1"
        self.addon = base.load_addon(self.args, self.report)
        self.report["runtime"] = base.runtime_info(self.addon)
        assert self.report["runtime"]["warp"] == "1.19.0"
        self.report["loaded_sources"] = {}
        for name in SOURCES:
            source = Path(self.modules(name).__file__).resolve()
            self.report["loaded_sources"][name] = {"path": str(source), "sha256": base.sha(source)}
        self.engine = self.modules("blender.fspt_engine")
        self.runtime = self.modules("blender.fspt_device_runtime")
        self.surface_publication = self.modules("blender.surface_publication")
        self.output = self.modules("blender.output_publication")
        self.bridge = self.modules("blender.fspt_bridge")
        self.adapter = self.modules("blender.fspt_device_scene")
        self.viewport = self.modules("fspt.viewport")
        self.handoff = self.modules("fspt.final_handoff").MAIN_FINAL_HANDOFF
        self.warp = self.modules("runtime.warp_runtime")._import_warp()
        self.device = self.warp.get_device(base.script_device())
        self.alias = str(self.device)
        assert self.device.is_cuda and self.runtime.enabled()
        assert bpy.app.timers.is_registered(self.engine._pump_device_runtime)
        assert not self.runtime.has_resources(self.alias) and not self.handoff.blocked(self.alias)

        origin = (3.0, -2.0, 1.0)
        scene, _ = surface_fixture.make_scene(self.modules, origin)
        self.scene = scene
        world = bpy.data.worlds.new("Integrated FSPT sky")
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.12, 0.22, 0.34, 1)
        scene.world = world
        fluid, _ = base.material("Integrated fluid diffuse", "ShaderNodeBsdfDiffuse", (0.9, 0.04, 0.02))
        scene.fluidsim.surface_material = fluid
        bpy.ops.object.camera_add(location=(3.25, -2.85, 1.73))
        camera = bpy.context.object
        camera.rotation_euler = (Vector((3.25, -1.75, 1.24)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.data.lens = 55
        scene.camera = camera
        # Keep real rendered-region consumers out of the controlled owner fixture.
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.spaces.active.shading.type = "SOLID"
        scene.render.engine = self.engine.ENGINE_ID
        scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.color_mode = "RGBA"
        self.final_path = self.artifact("final.exr")
        scene.render.filepath = str(self.final_path)
        scene.fspt.device = self.alias
        scene.fspt.samples, scene.fspt.max_bounces, scene.fspt.seed = 8, 6, 31
        bpy.context.view_layer.update()

        self.solver = surface_fixture.make_solver(
            self.modules, self.alias, origin, waves=False,
            cadence=self.modules("sim.enums").SurfaceUpdateCadence.FRAME)
        self.session = self.modules("blender.scene_bridge").session()
        self.session.install_solver(
            self.solver, profiler=None, device=self.solver.device,
            scene_inputs=self.modules("blender.scene_inputs").SceneInputs(), solved_timeline_frame=1)
        pointer = self.modules("blender.scene_access").scene_identity(scene)
        self.session.bind_scene(pointer, "fspt-device-viewport-proof")
        self.view = self.viewport.ViewportState(self.engine._COORDINATOR, self.engine._viewport_backend)
        assert self.view in self.engine._COORDINATOR.views
        self.withdraw = self.runtime.subscribe(scene, self.view, device=self.alias)
        self.runtime.set_capability(scene, self.view, self.alias, reason="")
        self.static_cache = self.adapter.StaticSceneCache()
        self.checkpoint("advance")

    def advance(self):
        timeline = TIMELINE[self.frame_index]
        previous = self.selection
        self.selection = None
        self.view.invalidate(surface_only=True)
        self.scene.frame_set(timeline)
        self.session.step(self.scene, write_cache=False, timeline_frame=timeline)
        self.completed = self.session.current_solver_completion(self.scene)
        assert self.completed is not None and self.completed.source_timeline_frame == timeline
        self.host = self.surface_publication.current_surface_publication(self.scene)
        assert self.host is not None and self.host.source_timeline_frame == timeline
        assert self.host.state_ready_at == self.completed.ready_at
        assert self.runtime.selection(self.scene, self.alias) is None, "Unpolled source was eligible"
        assert self.surface_publication.current_surface_publication(
            self.scene, self.surface_publication.SurfaceChannel.GPU_FINAL) is None
        if previous is not None:
            try:
                self.bridge.validate_publications(
                    self.scene, (previous.publication,), for_viewport=True,
                    view_layer=bpy.context.view_layer, device=self.alias)
            except self.bridge.SurfaceNotReadyError:
                pass
            else:
                raise AssertionError("Superseded GPU source survived the native host advance")
        resource = self.runtime.device_resources(self.alias)
        assert resource is not None and resource.ticket is not None and resource.committed, (
            self.runtime.fallback_reason(self.scene, self.alias), self.runtime.resource_error(self.alias))
        self.checkpoint(f"poll-{timeline}")

    def acquire_geometry(self, selection):
        stream = self.modules("fspt.async_backend")._nonblocking_stream(self.warp, self.device)
        retirement = stream.retirement
        lease = None
        try:
            lease = selection.pool.acquire(selection.key, stream)
            assert lease is not None, "Independent reader could not lease the published GPU geometry"
            facts = SimpleNamespace(wp=self.warp, device=self.device)
            return producer_fixture.packet_arrays(facts, lease, stream)
        finally:
            try:
                if lease is not None:
                    lease.finish()
            finally:
                try:
                    self.warp.synchronize_stream(stream)
                    selection.pool.poll()
                    if lease is not None:
                        assert lease.retired
                finally:
                    stream.retire()
                    assert retirement.result == 0, retirement.result

    def capture(self):
        selected = self.runtime.poll(self.scene, self.alias)
        if selected is None:
            failure = (self.runtime.fallback_reason(self.scene, self.alias)
                       or self.runtime.resource_error(self.alias))
            assert not failure, failure
            return
        assert selected.publication.source_timeline_frame == TIMELINE[self.frame_index]
        assert selected.publication.state_ready_at == self.completed.ready_at
        assert selected.publication.state_revision == self.host.state_revision
        assert selected.publication.channel == self.surface_publication.SurfaceChannel.GPU_FINAL
        graph = bpy.context.evaluated_depsgraph_get()
        graph.update()  # Controlled native capture boundary, outside the solver hook.
        reason = self.adapter.device_scene_reason(graph, selected)
        assert not reason, reason
        camera = self.modules("blender.fspt_scene").camera_snapshot(graph, WIDTH, HEIGHT)
        request = self.viewport.ViewRequest(camera, WIDTH, HEIGHT, 4, 6, 31, self.alias)
        self.view.request_update(request)
        if not self.view.capture(lambda: self.adapter.capture_device_scene(
                graph, selected, static_revision=0, cache=self.static_cache)):
            return
        self.selection = selected
        self.source_geometry = self.acquire_geometry(selected)
        surface = self.output.output_object(self.scene, self.output.SURFACE_NAME, ensure=False)
        self.native_geometry = surface_fixture.evaluated_surface(surface)
        surface_fixture.require_parity(self.source_geometry, self.native_geometry)
        self.checkpoint(f"trace-{TIMELINE[self.frame_index]}")

    def eligible(self, publications):
        self.bridge.validate_publications(
            self.scene, publications, for_viewport=True,
            view_layer=bpy.context.view_layer, device=self.alias)
        return True

    def render_view(self):
        self.view.tick(self.eligible)
        assert not self.view.error, self.view.error
        if not self.view.at_goal:
            assert not self.view.dirty, "The controlled device request lost eligibility"
            return
        frame = self.view.presentable
        assert frame.publications == (self.selection.publication,) and frame.device == self.alias
        assert frame.scene_frame == TIMELINE[self.frame_index] and frame.samples == 4
        assert frame.geometry_digest == self.view.presented_snapshot.geometry_digest
        assert dict(frame.statistics)["device_geometry_copies"] == self.frame_index + 1
        pixels = frame.pixels
        assert pixels.shape == (HEIGHT, WIDTH, 4) and np.isfinite(pixels).all()
        assert np.max(pixels[..., :3]) > 0 and np.ptp(pixels[..., :3]) > 0
        timeline = TIMELINE[self.frame_index]
        original = self.artifact(f"frame-{timeline}.npz")
        np.savez(original, pixels=pixels, positions=self.source_geometry[0], triangles=self.source_geometry[1],
                 evaluated_positions=self.native_geometry[0], evaluated_triangles=self.native_geometry[1])
        row = {
            "timeline_frame": timeline,
            "solver_frame": self.completed.solver_frame,
            "state_revision": self.completed.state_revision,
            "publication": asdict(self.selection.publication),
            "source_geometry_sha256": [surface_fixture.digest(array) for array in self.source_geometry],
            "evaluated_geometry_sha256": [surface_fixture.digest(array) for array in self.native_geometry],
            "pixels_sha256": surface_fixture.digest(pixels),
            "original": str(original), "original_sha256": base.sha(original),
            "backend": frame.backend, "transport": frame.transport, "samples": frame.samples,
            "device_geometry_copies": dict(frame.statistics)["device_geometry_copies"],
            "stale_gpu_publication_rejected_before_poll": self.frame_index > 0,
            "publication_clock_order_verified": (self.completed.ready_at <= self.selection.publication.captured_at
                                                 <= self.selection.publication.published_at),
        }
        assert row["publication_clock_order_verified"], row["publication"]
        rows = self.report["frames"]
        if rows:
            previous = rows[-1]
            for key in ("solver_frame", "state_revision"):
                assert row[key] > previous[key], (key, previous, row)
            assert row["publication"]["generation"] > previous["publication"]["generation"]
            assert row["source_geometry_sha256"][0] != previous["source_geometry_sha256"][0]
            difference = float(np.mean(np.abs(pixels[..., :3] - self.prior_pixels[..., :3])))
            assert difference > 1e-6, difference
            row["image_mean_absolute_difference"] = difference
        rows.append(row)
        self.prior_pixels = pixels.copy()
        self.frame_index += 1
        self.checkpoint("advance" if self.frame_index < len(TIMELINE) else "invoke-final")

    def event(self, name, **facts):
        self.report["final"]["events"].append({"event": name, **thread_facts(), **facts})

    def ownership(self):
        return {"hold_active": self.handoff.blocked(self.alias),
                "coordinator_blocked": self.engine._COORDINATOR.blocked(self.alias),
                "resources_remaining": self.runtime.has_resources(self.alias),
                "view_owned": self.view.backend is not None}

    def view_ownership(self):
        delegate = getattr(self.view.backend, "_delegate", None)
        current = getattr(delegate, "_device_current", None)
        snapshot = self.view.presented_snapshot
        open_lease = (current is not None and snapshot is not None
                      and snapshot.reference.key == self.selection.key
                      and current.lease.packet.stamp == self.selection.key.build_stamp)
        return {"aggregate_open": open_lease,
                "trace_pending": delegate is not None and delegate.pending,
                "reserved_bytes": self.final_budget.reserved_bytes,
                "generation_count": self.final_budget.generation_count, **self.ownership()}

    def patch(self, owner, name, value):
        self.patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def observe_final(self):
        engine = self.engine
        render = engine.FSPTRenderEngine.render
        retirement = engine._retire_device_for_final
        backend_factory = engine._backend
        backend_class = self.modules("fspt.async_backend").AsyncCudaBackend
        close = backend_class.close

        def observed_render(instance, depsgraph):
            self.event("render_enter")
            try:
                return render(instance, depsgraph)
            finally:
                self.event("render_exit", **self.ownership())

        def observed_retirement(device, identity):
            self.event("retirement_enter", **self.view_ownership())
            receipt = retirement(device, identity)
            self.event("retirement_receipt", request_matches=receipt.request_id is identity,
                       owners_retired=receipt.owners_retired, reserved_bytes=receipt.reserved_bytes,
                       generation_count=receipt.generation_count, device=receipt.device,
                       **self.ownership())
            return receipt

        def observed_backend(device):
            self.event("final_backend_allocate", **self.ownership())
            instance = backend_factory(device)
            self.final_backend = instance
            self.event("final_backend_allocated", backend_type=type(instance).__name__, **self.ownership())
            return instance

        def observed_close(instance):
            if instance is not self.final_backend:
                return close(instance)
            self.event("final_backend_close_enter", **self.ownership())
            retired_stream = instance._render_stream.retirement
            result = close(instance)
            self.event("final_backend_closed", backend_closed=instance.closed,
                       render_stream_retirement=retired_stream.result, **self.ownership())
            return result

        self.patch(engine.FSPTRenderEngine, "render", observed_render)
        self.patch(engine, "_retire_device_for_final", observed_retirement)
        self.patch(engine, "_backend", observed_backend)
        self.patch(backend_class, "close", observed_close)
        for name in ("render_init", "render_complete", "render_cancel"):
            def handler(*_args, event=name):
                self.event(event)
            callbacks = getattr(bpy.app.handlers, name)
            callbacks.append(handler)
            self.handlers.append((callbacks, handler))

    def invoke_final(self):
        resource = self.runtime.device_resources(self.alias)
        self.final_budget = resource.budget
        self.view.request_update(replace(self.view.request, target_samples=self.view.request.target_samples + 1))
        assert self.view.tick(self.eligible) == "submitted", "Expected one owned quantum before INVOKE"
        self.report["final"]["viewport_before"] = self.view_ownership()
        before = self.report["final"]["viewport_before"]
        assert before["view_owned"] and before["aggregate_open"] and before["reserved_bytes"] > 0
        assert before["generation_count"] > 0 and not before["hold_active"] and before["trace_pending"]
        self.handoff_before = self.handoff.statistics
        self.observe_final()
        result = bpy.ops.render.render("INVOKE_DEFAULT", write_still=True)
        self.report["final"]["operator_result"] = sorted(result)
        assert result == {"RUNNING_MODAL"}, result
        self.checkpoint("wait-final")

    def finish_final(self):
        final = self.report["final"]
        if bpy.app.is_job_running("RENDER"):
            return False
        if not any(event["event"] in {"render_complete", "render_cancel"} for event in final["events"]):
            return False
        info = dict(self.engine.LAST_RENDER_INFO)
        final["render_info"] = info
        assert info["status"] == "COMPLETE" and info["samples"] == self.scene.fspt.samples, info
        assert info["scene_frame"] == TIMELINE[-1] and info["device"] == self.alias, info
        assert len(info["sources"]) == 1 and info["sources"][0]["channel"] == "EVALUATED_MESH", info
        counts = self.handoff.statistics
        delta = {key: counts[key] - self.handoff_before[key] for key in self.handoff_before}
        final["handoff_statistics_delta"] = delta
        verify_handoff(final, delta)
        assert self.final_budget.reserved_bytes == self.final_budget.generation_count == 0
        assert not self.runtime.has_resources(self.alias) and self.view.backend is None
        image = bpy.data.images.load(str(self.final_path), check_existing=False)
        try:
            assert tuple(image.size) == (WIDTH, HEIGHT), tuple(image.size)
            pixels = np.empty(WIDTH * HEIGHT * 4, dtype=np.float32)
            image.pixels.foreach_get(pixels)
            rgb = pixels.reshape(-1, 4)[:, :3]
            assert np.isfinite(pixels).all() and np.max(rgb) > 0 and np.ptp(rgb) > 0
            final.update(original=str(self.final_path), original_sha256=base.sha(self.final_path),
                         pixels_sha256=surface_fixture.digest(pixels), image_size=list(image.size),
                         pixel_range=[float(np.min(rgb)), float(np.max(rgb))], event_order_verified=True)
        finally:
            bpy.data.images.remove(image)
        return True

    def finish(self, error=None):
        cleanup = self.report["cleanup"]
        try:
            if bpy.app.is_job_running("RENDER"):
                raise RuntimeError("Render worker is still active at verifier shutdown")
            for callbacks, handler in self.handlers:
                callbacks.remove(handler)
            for owner, name, value in reversed(self.patches):
                setattr(owner, name, value)
            if self.view is not None:
                self.view.close()
                cleanup["viewport_closed"] = self.view.closed and self.view.backend is None
                assert cleanup["viewport_closed"]
            if self.withdraw is not None:
                self.withdraw()
            if self.session is not None:
                assert self.session.release(self.scene), "Owned Solver Session did not release"
            if self.addon is not None:
                pending = self.runtime.shutdown() if hasattr(self, "runtime") else ()
                cleanup["unretired_devices"] = list(pending)
                cleanup["resources_remaining"] = (bool(pending) or
                    (hasattr(self, "runtime") and self.runtime.has_resources()))
                cleanup["hold_remaining"] = (self.handoff.blocked(self.alias)
                    if hasattr(self, "handoff") and hasattr(self, "alias") else None)
                assert not cleanup["resources_remaining"] and not cleanup["hold_remaining"], cleanup
                self.addon.unregister()
                cleanup["addon_unregistered"] = True
        except BaseException:
            self.report["cleanup_error"] = traceback.format_exc()
            error = error or self.report["cleanup_error"]
        finally:
            if self.previous_flag is None:
                os.environ.pop(ENV, None)
            else:
                os.environ[ENV] = self.previous_flag
            if error:
                self.report.update(status="FAILED", error=error)
            else:
                self.report.update(status="VERIFIED", stage="complete")
            base.write_report(self.path, self.report)
        if error:
            print("FSPT-DEVICE-VIEWPORT-FAIL", self.path, error, flush=True)
            os._exit(1)
        print("FSPT-DEVICE-VIEWPORT-PASS", self.path, flush=True)
        bpy.ops.wm.quit_blender()

    def tick(self):
        try:
            if monotonic() >= self.deadline:
                raise TimeoutError(f"Device viewport proof exceeded its deadline at {self.stage}")
            if self.stage == "setup":
                self.setup()
            elif self.stage == "advance":
                self.advance()
            elif self.stage.startswith("poll-"):
                self.capture()
            elif self.stage.startswith("trace-"):
                self.render_view()
            elif self.stage == "invoke-final":
                self.invoke_final()
            elif self.stage == "wait-final":
                if self.finish_final():
                    self.finish()
                    return None
            else:
                raise RuntimeError(f"Unrecognized verification stage: {self.stage}")
            return 0.025
        except BaseException:
            self.finish(traceback.format_exc())
            return None


def main():
    args = arguments()
    path = args.json_out.resolve()
    assert not path.exists(), f"Use a fresh evidence destination: {path}"
    report = {
        "status": "RUNNING", "stage": "setup",
        "scope": "Native CUDA Session publication, controlled viewport rendering, GUI worker Final ownership",
        "frontend_fixture": "Controlled ViewportState in the real engine coordinator; "
                            "explicit evaluated capture boundary; no region display or ordinary GPU route claim",
        "verifier_sha256": base.sha(Path(__file__).resolve()),
        "fixture_sha256": {Path(mod.__file__).name: base.sha(Path(mod.__file__))
                           for mod in (base, surface_fixture, producer_fixture)},
        "frames": [], "final": {"events": []}, "cleanup": {},
    }
    runner = Case(args, path, report)
    base.write_report(path, report)
    bpy.app.timers.register(runner.tick, first_interval=0.2)


if __name__ == "__main__":
    main()
