"""Real-window asynchronous FSPT boundaries, two views, final admission and teardown.

This static-scene gate verifies callback ownership and resources. Fresh live
simulation, ten-minute soak and fixed-quality performance have separate gates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import sys
import time
import traceback
from collections import Counter, defaultdict, deque
from pathlib import Path
from threading import current_thread, local, main_thread
from unittest.mock import patch
from weakref import WeakSet

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("blender_fspt_acceptance.py")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--addon-root", type=Path, default=ROOT)
    parser.add_argument("--addon-module", default="fluidSim_warp_blender51")
    parser.add_argument("--lifecycle-cycles", dest="cycles", type=int, default=50)
    parser.add_argument("--deadline", type=float, default=360)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if not 1 <= args.cycles <= 200 or not 0 < args.deadline <= 900:
        parser.error("cycles must be 1..200 and deadline must be 0..900 seconds")
    return args


def digest(pixels):
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def compare_view_pixels(first, second):
    """Compare actual RGB at a common normalized grid after physical splitting."""
    def sample(pixels):
        assert pixels.ndim == 3 and pixels.shape[2] == 4 and np.isfinite(pixels).all()
        coords = (np.arange(64) + 0.5) / 64
        y = np.minimum((coords * pixels.shape[0]).astype(int), pixels.shape[0] - 1)
        x = np.minimum((coords * pixels.shape[1]).astype(int), pixels.shape[1] - 1)
        return pixels[y[:, None], x[None, :], :3]
    delta = float(np.mean(np.abs(sample(first) - sample(second))))
    assert delta > 0.001, f"Offset views did not produce different completed pixels: {delta}"
    return delta


class AsyncViewportAcceptance:
    def __init__(self, bpy, args):
        self.bpy, self.args = bpy, args
        self.path = args.json_out.resolve()
        self.folder = self.path.with_name(self.path.stem + "-originals")
        self.folder.mkdir(parents=True, exist_ok=False)
        spec = importlib.util.spec_from_file_location("fspt_async_view_fixture", FIXTURE)
        self.fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.fixture)
        self.started = time.monotonic()
        self.stage = "initial"
        self._phases = local()
        self.patchers = []
        self.backends = WeakSet()
        self.calls = Counter()
        self.times = defaultdict(lambda: deque(maxlen=4096))
        self.histories = defaultdict(lambda: deque(maxlen=32))
        self.retirements = []
        self.maximum_active_backends = self.maximum_pending = self.maximum_capacity = 0
        self.completed_cycles = 0
        self.retiring = []
        self.report = {"status": "RUNNING", "stage": self.stage,
                       "scope": "static native callback/resource qualification; no live-performance claim",
                       "verifier_sha256": self.fixture.sha(Path(__file__)),
                       "fixture_sha256": self.fixture.sha(FIXTURE), "stages": {}, "cycles": []}

    @property
    def phase(self):
        return getattr(self._phases, "phase", None)

    @phase.setter
    def phase(self, value):
        self._phases.phase = value

    def checkpoint(self, stage):
        self.stage = stage
        self.fixture.checkpoint(self.path, self.report, stage)

    def patch(self, owner, name, value):
        guard = patch.object(owner, name, value)
        guard.start()
        self.patchers.append(guard)

    def instrument(self):
        engine, backend = self.engine, self.backend_module.AsyncCudaBackend
        callbacks = {name: getattr(engine.FSPTRenderEngine, name)
                     for name in ("view_draw", "view_update", "render")}
        def callback(name, phase, instance, *args):
            previous, self.phase = self.phase, phase
            started = time.perf_counter()
            try:
                if phase != "final":
                    assert current_thread() is main_thread(), phase
                return callbacks[name](instance, *args)
            finally:
                self.times[phase].append(time.perf_counter() - started)
                self.calls[phase] += 1
                self.phase = previous
        # RNA dispatch derives the supplied argument count from the Python
        # code signature. Keep explicit parameters on every engine callback.
        def view_draw(instance, context, depsgraph):
            return callback("view_draw", "draw", instance, context, depsgraph)
        def view_update(instance, context, depsgraph):
            return callback("view_update", "update", instance, context, depsgraph)
        def render(instance, depsgraph):
            return callback("render", "final", instance, depsgraph)
        for name, wrapper in (("view_draw", view_draw), ("view_update", view_update), ("render", render)):
            self.patch(engine.FSPTRenderEngine, name, wrapper)
        original_init = backend.__init__
        def create(instance, *args, **kwargs):
            assert self.phase != "draw", "Drawing constructed a CUDA backend"
            original_init(instance, *args, **kwargs)
            self.backends.add(instance)
            self.calls["backend_created"] += 1
        self.patch(backend, "__init__", create)
        for name in ("submit", "collect", "close"):
            original = getattr(backend, name)
            def operation(instance, *args, _name=name, _original=original, **kwargs):
                assert self.phase != "draw", f"Drawing reached backend {_name}"
                if _name == "collect" and kwargs.get("wait", False):
                    assert self.phase == "final", "Viewport collection requested a blocking wait"
                    self.calls["final_owned_wait"] += 1
                receipt = (getattr(getattr(instance, "_render_stream", None), "retirement", None)
                           if _name == "close" else None)
                result = _original(instance, *args, **kwargs)
                self.calls[_name] += 1
                if _name == "submit" and result:
                    self.calls["admitted"] += 1
                if _name == "close" and receipt is not None:
                    assert receipt.result == 0, f"CUDA stream retirement: {receipt.result}"
                    self.retirements.append({"device": str(instance.device), "result": receipt.result})
                self.check_resources()
                return result
            self.patch(backend, name, operation)
        extraction = importlib.import_module(self.addon.__name__ + ".blender.fspt_scene")
        original_extract = extraction.extract_scene
        def extract(*args, **kwargs):
            assert self.phase in {"update", "final"}, f"Scene extracted during {self.phase}"
            self.calls["extract"] += 1
            return original_extract(*args, **kwargs)
        self.patch(extraction, "extract_scene", extract)
        wp = importlib.import_module(self.addon.__name__ + ".runtime.warp_runtime")._import_warp()
        def no_device_wait(*args, **kwargs):
            raise AssertionError("FSPT viewport reached a device-wide synchronization")
        self.patch(wp, "synchronize", no_device_wait)
        self.patch(wp, "synchronize_device", no_device_wait)

    def check_resources(self):
        active = [backend for backend in self.backends if not backend.closed]
        pending = sum(backend.pending for backend in active)
        capacity = sum(backend._capacity for backend in active)
        assert pending <= 1, f"Device admitted {pending} simultaneous FSPT jobs"
        self.maximum_pending = max(self.maximum_pending, pending)
        self.maximum_active_backends = max(self.maximum_active_backends, len(active))
        self.maximum_capacity = max(self.maximum_capacity, capacity)

    def prepare(self):
        bpy = self.bpy
        assert not bpy.app.background, "This gate requires a real Blender window"
        bpy.context.preferences.view.show_splash = False
        bpy.context.preferences.use_preferences_save = False
        self.addon = self.fixture.load_addon(self.args, self.report)
        self.report["runtime"] = self.fixture.runtime_info(self.addon)
        self.engine = importlib.import_module(self.addon.__name__ + ".blender.fspt_engine")
        self.backend_module = importlib.import_module(self.addon.__name__ + ".fspt.async_backend")
        assert hasattr(self.engine, "viewport_observations"), "Asynchronous viewport adapter missing"
        self.scene, self.material_node, _ = self.fixture.fixture()
        self.scene.fspt.resolution_percentage = 20
        self.scene.fspt.viewport_samples = 8
        self.window = bpy.context.window_manager.windows[0]
        area = next(area for area in self.window.screen.areas if area.type == "VIEW_3D")
        self.original_area = area.as_pointer()
        self.original_size = (area.width, area.height)
        self.instrument()
        self.configure_area(area)
        self.checkpoint("initial")

    def configure_area(self, area, *, offset=(0.0, 0.0)):
        space = area.spaces.active
        space.region_3d.view_perspective = "CAMERA"
        space.region_3d.view_camera_offset = offset
        space.overlay.show_overlays = False
        space.shading.type = "RENDERED"
        area.tag_redraw()

    def areas(self):
        return [area for area in self.window.screen.areas if area.type == "VIEW_3D"]

    def views(self):
        result = {}
        for instance in tuple(self.engine._ENGINES):
            state = instance._view
            info = dict(instance._view_info)
            if state is None or state.presentable is None or info.get("status") != "COMPLETE":
                continue
            frame = state.presentable
            if instance._texture_frame is not frame:
                continue
            if tuple(info["key"]) != frame.key or info["samples"] != frame.samples:
                continue
            if info["backend"] != "warp-cuda-async-v1" or info["transport"] != "host-pinned-rgba32f":
                raise AssertionError(f"Unexpected presentation route: {info}")
            region_id = info["region_id"]
            history = self.histories[region_id]
            item = {"key": info["key"], "samples": info["samples"], "draw_count": info["draw_count"]}
            if not history or history[-1] != item:
                history.append(item)
            result[region_id] = (info, frame)
        return result

    def ready_views(self, count):
        views = self.views()
        expected = {}
        for area in self.areas():
            if area.spaces.active.shading.type != "RENDERED":
                continue
            region = next(region for region in area.regions if region.type == "WINDOW")
            expected[region.as_pointer()] = (max(1, region.width * self.scene.fspt.resolution_percentage // 100),
                                             max(1, region.height * self.scene.fspt.resolution_percentage // 100))
        if len(expected) != count or set(views) != set(expected):
            return None
        for token, (info, frame) in views.items():
            if tuple(frame.key[2:4]) != expected[token] or frame.samples < self.scene.fspt.viewport_samples:
                return None
            assert np.isfinite(frame.pixels).all()
            assert np.ptp(frame.pixels[..., :3].mean(axis=2)) > 0.05, "No rendered scene contrast"
        return views

    def save_views(self, label, views, *, screenshot=True):
        rows = []
        for number, (token, (info, frame)) in enumerate(sorted(views.items())):
            path = self.folder / f"{label}-{number}.npy"
            np.save(path, frame.pixels, allow_pickle=False)
            rows.append(dict(info, pixels=str(path), pixels_sha256=self.fixture.sha(path),
                             content_sha256=digest(frame.pixels),
                             mean_rgb=frame.pixels[..., :3].mean(axis=(0, 1)).tolist(),
                             sample_observations=list(self.histories[token])))
        output = {"views": rows, "sizes": [[area.width, area.height] for area in self.areas()]}
        if screenshot:
            path = self.folder / f"{label}-screenshot.png"
            assert self.bpy.ops.screen.screenshot(filepath=str(path)) == {"FINISHED"}
            output.update(screenshot=str(path), screenshot_sha256=self.fixture.sha(path))
        self.report["stages"][label] = output
        return output

    def split(self):
        area = self.areas()[0]
        region = next(region for region in area.regions if region.type == "WINDOW")
        with self.bpy.context.temp_override(window=self.window, area=area, region=region):
            result = self.bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
        assert result == {"FINISHED"}, result
        areas = sorted(self.areas(), key=lambda value: value.x)
        assert len(areas) == 2, "The physical viewport split did not create two views"
        self.configure_area(areas[0])
        self.configure_area(areas[1], offset=(0.13, 0.0))
        self.checkpoint("two_views")

    def retire_cycle(self):
        self.retiring = [backend for backend in self.backends if not backend.closed]
        assert self.retiring and any(backend.pending for backend in self.retiring)
        for area in self.areas():
            area.spaces.active.shading.type = "SOLID"
        self.scene.render.engine = "BLENDER_WORKBENCH"
        self.checkpoint("cycle_retire")

    def tick(self):
        try:
            assert time.monotonic() - self.started < self.args.deadline, (
                f"Deadline at {self.stage}; {self.engine.viewport_observations()}")
            self.check_resources()
            for row in self.engine.viewport_observations():
                assert row.get("status") != "ERROR", row
            if self.stage == "initial":
                views = self.ready_views(1)
                if views is None:
                    return 0.01
                result = self.save_views("initial", views)
                history = result["views"][0]["sample_observations"]
                assert len({row["samples"] for row in history}) >= 2, "No progressive display observed"
                self.split()
            elif self.stage == "two_views":
                views = self.ready_views(2)
                if views is None:
                    return 0.01
                assert all(area.width < self.original_size[0] for area in self.areas())
                assert len({frame.key[1] for _, frame in views.values()}) == 2, "View histories share a camera"
                images = [frame.pixels for _, frame in views.values()]
                difference = compare_view_pixels(*images)
                self.save_views("two_views", views)["common_grid_rgb_mean_absolute_delta"] = difference
                self.prior_world = {token: frame.pixels.copy() for token, (_, frame) in views.items()}
                self.prior_keys = {token: frame.key for token, (_, frame) in views.items()}
                world = self.scene.world.copy()
                world.name = "FSPT async replacement sky"
                world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.31, 0.07, 0.12, 1)
                self.scene.world = world
                self.checkpoint("world_reassignment")
            elif self.stage == "world_reassignment":
                views = self.ready_views(2)
                if views is None or any(frame.key == self.prior_keys.get(token)
                                        for token, (_, frame) in views.items()):
                    return 0.01
                deltas = {str(token): float(np.mean(np.abs(frame.pixels[..., :3] -
                                             self.prior_world[token][..., :3])))
                          for token, (_, frame) in views.items()}
                assert all(delta > 0.001 for delta in deltas.values()), deltas
                self.save_views("world_reassignment", views)["mean_absolute_deltas"] = deltas
                self.scene.fspt.viewport_samples = 64
                self.checkpoint("final_pending")
            elif self.stage == "final_pending":
                pending = sum(backend.pending for backend in self.backends if not backend.closed)
                if pending == 0:
                    return 0.002
                before = self.calls["final_owned_wait"]
                self.fixture.render(self.addon, self.scene, self.folder, "final-with-pending-view", self.report)
                assert self.calls["final_owned_wait"] > before
                assert not any(backend.pending for backend in self.backends if not backend.closed)
                self.report["final_admission"] = {"pending_before": pending,
                                                  "owned_waits": self.calls["final_owned_wait"] - before}
                self.scene.fspt.viewport_samples = 2
                self.checkpoint("after_final")
            elif self.stage == "after_final":
                views = self.ready_views(2)
                if views is None:
                    return 0.01
                self.save_views("after_final", views)
                self.scene.fspt.viewport_samples = 64
                self.checkpoint("cycle_pending")
            elif self.stage == "cycle_pending":
                self.views()
                if not any(backend.pending for backend in self.backends if not backend.closed):
                    return 0.002
                self.retire_cycle()
            elif self.stage == "cycle_retire":
                if not all(backend.closed for backend in self.retiring):
                    return 0.01
                assert all(backend._capacity == 0 and not backend.pending for backend in self.retiring)
                arbiters = {backend.arbiter for backend in self.retiring}
                assert all(arbiter._active is None and not arbiter._viewport_waiters
                           and not arbiter._final_waiters for arbiter in arbiters)
                self.report["cycles"].append({"cycle": self.completed_cycles + 1,
                                               "retired_backends": len(self.retiring),
                                               "stream_retirements": len(self.retirements)})
                self.retiring.clear()
                self.scene.render.engine = self.fixture.ENGINE
                self.scene.fspt.viewport_samples = 2
                for number, area in enumerate(sorted(self.areas(), key=lambda value: value.x)):
                    self.configure_area(area, offset=(number * 0.13, 0.0))
                self.checkpoint("cycle_reopen")
            elif self.stage == "cycle_reopen":
                views = self.ready_views(2)
                if views is None:
                    return 0.01
                self.completed_cycles += 1
                if self.completed_cycles == self.args.cycles:
                    self.save_views("last_cycle", views)
                    return self.finish()
                self.scene.fspt.viewport_samples = 64
                self.checkpoint("cycle_pending")
            return 0.01
        except BaseException:
            self.report.update(status="FAILED", error=traceback.format_exc())
            self.evidence()
            self.fixture.write_report(self.path, self.report)
            print("FSPT-ASYNC-VIEWPORT-FAIL", self.report["error"], flush=True)
            os._exit(1)

    def evidence(self):
        self.report.update(calls=dict(self.calls), stream_retirements=self.retirements,
                           completed_cycles=self.completed_cycles,
                           maximum_pending_jobs=self.maximum_pending,
                           maximum_active_backends=self.maximum_active_backends,
                           maximum_combined_image_capacity=self.maximum_capacity,
                           elapsed_seconds=time.monotonic() - self.started,
                           callback_seconds={name: {"count": len(values),
                               "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                               "max": float(max(values))} for name, values in self.times.items() if values})

    def finish(self):
        for area in self.areas():
            area.spaces.active.shading.type = "SOLID"
        self.scene.render.engine = "BLENDER_WORKBENCH"
        self.addon.unregister()
        assert all(backend.closed for backend in self.backends)
        assert not self.bpy.app.timers.is_registered(self.engine._pump_views)
        assert self.maximum_pending == 1
        assert len(self.retirements) >= self.args.cycles * 2 + 1
        self.evidence()
        for guard in reversed(self.patchers):
            guard.stop()
        self.report.update(status="VERIFIED", stage="complete")
        self.fixture.write_report(self.path, self.report)
        print("FSPT-ASYNC-VIEWPORT-PASS", self.path, flush=True)
        self.bpy.ops.wm.quit_blender()
        return None


def main():
    import bpy

    run = AsyncViewportAcceptance(bpy, arguments())
    try:
        run.prepare()
    except BaseException:
        run.report.update(status="FAILED", error=traceback.format_exc())
        run.fixture.write_report(run.path, run.report)
        print("FSPT-ASYNC-VIEWPORT-FAIL", run.report["error"], flush=True)
        os._exit(1)
    bpy.app.timers.register(run.tick, first_interval=0.01)


if __name__ == "__main__":
    main()
