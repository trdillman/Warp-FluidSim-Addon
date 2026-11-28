"""FSPT adapter: capture in updates, submit/poll in timers, present in draws."""
import weakref
from contextlib import closing
from threading import current_thread, main_thread
from time import monotonic

import bpy
import numpy as np
from bpy.props import IntProperty, PointerProperty, StringProperty

from ..fspt.contracts import MAX_BOUNCES, CameraSnapshot, frame_key, require_current
from ..fspt.viewport import ViewportCoordinator, ViewportState, ViewRequest
from ..unregister_guard import class_registration, handler_registration, unregister_classes

ENGINE_ID = "FLUIDSIM_FSPT"
LAST_RENDER_INFO = {}
LAST_VIEWPORT_INFO = {}
_ENGINES = weakref.WeakSet()
_DRAW_COUNT = 0
_REGISTERED = False
_COORDINATOR = ViewportCoordinator()


def _backend(device):
    from ..fspt.async_backend import AsyncCudaBackend

    return AsyncCudaBackend(device)


def _ensure_timer():
    if _REGISTERED and not bpy.app.timers.is_registered(_pump_views):
        bpy.app.timers.register(_pump_views, first_interval=0.001)


def _pump_views():
    if not _REGISTERED:
        return None
    active = False
    for engine in tuple(_ENGINES):
        try:
            active = engine._pump_view() or active
        except Exception as exc:
            engine._timer_failed(exc)
    return 0.012 if active else None


def viewport_observations():
    """Independent presentation receipts for the currently owned viewport regions."""
    return [dict(engine._view_info) for engine in tuple(_ENGINES) if engine._view_info]


@bpy.app.handlers.persistent
def _on_fspt_view_history_pre(*_args):
    for engine in tuple(_ENGINES):
        try:
            engine.close()
        except Exception as exc:
            print(f"FSPT history retirement failed: {exc}")


class FSPTSettings(bpy.types.PropertyGroup):
    samples: IntProperty(name="Final Samples", default=64, min=1, max=4096)
    viewport_samples: IntProperty(name="Viewport Samples", default=64, min=1, max=4096)
    max_bounces: IntProperty(name="Max Bounces", default=6, min=1, max=MAX_BOUNCES)
    seed: IntProperty(name="Seed", default=0, min=0, max=1000000)
    resolution_percentage: IntProperty(name="Viewport Resolution %", default=100, min=10, max=100)
    device: StringProperty(name="CUDA Device", default="cuda:0")


def _frame_info(frame, scene, width, height):
    from .fspt_bridge import frame_info

    return frame_info(frame, width, height)


class FSPTRenderEngine(bpy.types.RenderEngine):
    bl_idname = ENGINE_ID
    bl_label = "FSPT (Experimental)"
    bl_use_preview = False
    bl_use_shading_nodes_custom = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._view = None
        self._unsubscribe = None
        self._scene_layer = self._scene_inputs = self._location = None
        self._texture = self._texture_frame = None
        self._camera_bytes = self._camera = None
        self._view_info = {}
        self._next_update = 0.0
        _ENGINES.add(self)

    def _state(self):
        if self._view is None:
            self._view = ViewportState(_COORDINATOR, _backend)
        return self._view

    def close(self):
        global LAST_VIEWPORT_INFO
        if getattr(self, "_unsubscribe", None) is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if getattr(self, "_view", None) is not None:
            self._view.close()
            self._view = None
        self._scene_layer = self._scene_inputs = self._location = None
        self._texture = self._texture_frame = None
        self._camera_bytes = self._camera = None
        if LAST_VIEWPORT_INFO is getattr(self, "_view_info", None):
            LAST_VIEWPORT_INFO = {"status": "RETIRED"}
        self._view_info = {}

    def __del__(self):
        try:
            self.close()
        finally:
            parent = getattr(super(), "__del__", None)
            if parent is not None:
                parent()

    def render(self, depsgraph):
        from .fspt_bridge import prepare_final_publication, validate_publications
        from .fspt_scene import camera_snapshot, extract_scene

        global LAST_RENDER_INFO
        scene = depsgraph.scene
        result = None
        cancelled = False
        LAST_RENDER_INFO = {"status": "RENDERING"}
        try:
            settings = scene.fspt
            width = max(1, int(scene.render.resolution_x * scene.render.resolution_percentage / 100))
            height = max(1, int(scene.render.resolution_y * scene.render.resolution_percentage / 100))
            if scene.render.use_border:
                raise RuntimeError("FSPT P1 does not yet support render borders; disable Border")
            if scene.render.film_transparent:
                raise RuntimeError("FSPT P1 does not support transparent film")
            if scene.render.use_multiview:
                raise RuntimeError("FSPT P1 does not support multiview/stereo rendering")
            if sum(layer.use for layer in scene.view_layers) != 1:
                raise RuntimeError("FSPT P1 requires exactly one enabled view layer")
            with _COORDINATOR.final(settings.device):
                prepare_final_publication(depsgraph)
                snapshot = extract_scene(depsgraph, for_viewport=False)
                camera = camera_snapshot(depsgraph, width, height)
                # The final job owns its exact snapshot and a fresh history.
                with closing(_backend(settings.device)) as backend:
                    result = self.begin_result(0, 0, width, height, layer=depsgraph.view_layer.name)
                    for _ in range(settings.samples):
                        if self.test_break():
                            cancelled = True
                            break
                        if not backend.submit(snapshot, camera, width, height, final=True,
                                              max_bounces=settings.max_bounces, seed=settings.seed):
                            raise RuntimeError("Final FSPT job could not acquire its CUDA device")
                        frame = backend.collect(wait=True)
                        require_current(frame_key(snapshot, camera, width, height,
                                                  settings.max_bounces, settings.seed), frame.key)
                        validate_publications(scene, frame.publications, for_viewport=False,
                                              exact_frame=True, view_layer=depsgraph.view_layer)
                        result.layers[0].passes["Combined"].rect = frame.pixels.reshape(-1, 4).tolist()
                        self.update_result(result)
                        self.update_progress(frame.samples / settings.samples)
            self.end_result(result, cancel=cancelled)
            result = None
            if cancelled:
                LAST_RENDER_INFO = {"status": "CANCELLED"}
            else:
                LAST_RENDER_INFO = _frame_info(frame, scene, width, height)
        except Exception as exc:
            LAST_RENDER_INFO = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
            # Blender's render pipeline cancels file writing on an ERROR report.
            # error_set alone only changes the displayed message; an exception
            # escaping this callback does not stop write_still either.
            self.report({"ERROR"}, str(exc))
            if result is not None:
                self.end_result(result, cancel=True)
            self.error_set(str(exc))
        finally:
            _ensure_timer()

    def _published(self, channel, state):
        from .fspt_bridge import forward_live_publication
        from .surface_publication import SurfaceChannel

        view = self._view
        if view is None or channel != SurfaceChannel.EVALUATED_MESH:
            return
        if view.snapshot is not None and view.snapshot.publications == (state.eligible,):
            return
        prior = view.snapshot.publications if view.snapshot is not None else ()
        forward = len(prior) == 1 and forward_live_publication(prior[0], state.eligible)
        view.invalidate(surface_only=forward)
        self._next_update = 0.0
        if current_thread() is main_thread():
            self.tag_redraw()
            _ensure_timer()

    def _timer_failed(self, exc):
        if self._view is not None:
            self._view.fail(exc)
        self.tag_redraw()

    def _active_region(self):
        if self._location is None:
            return None
        window_id, area_id, region_id = self._location
        for window in bpy.context.window_manager.windows:
            if window.as_pointer() != window_id:
                continue
            for area in window.screen.areas:
                if (area.as_pointer() == area_id and area.type == "VIEW_3D"
                        and area.spaces.active.shading.type == "RENDERED"
                        and window.scene.render.engine == ENGINE_ID
                        and any(region.as_pointer() == region_id for region in area.regions)):
                    return window
        return None

    def _pump_view(self):
        from .fspt_bridge import (
            SurfaceNotReadyError,
            validate_publications,
            validate_viewport_publications,
        )
        from .surface_publication import id_token

        view = self._view
        if view is None or not view.needs_tick:
            return False
        window = self._active_region()
        if window is None:
            self.close()
            return False
        source = (id_token(window.scene), id_token(window.view_layer))
        if source != self._scene_layer:
            view.rebind()
        def eligible(stamps):
            try:
                validate_publications(window.scene, stamps, for_viewport=True,
                                      view_layer=window.view_layer)
            except SurfaceNotReadyError:
                return False
            return True
        def presentation_eligible(stamps):
            try:
                validate_viewport_publications(window.scene, stamps, view_layer=window.view_layer)
            except SurfaceNotReadyError:
                return False
            return True
        old = view.presentable
        action = view.tick(eligible, presentation_eligible=presentation_eligible)
        if view.presentable is not old:
            self.tag_redraw()
        if action == "capture" and monotonic() >= self._next_update:
            self._next_update = monotonic() + 0.02
            self.tag_update()
            self.tag_redraw()
        return view.needs_tick

    def _request(self, context, scene):
        settings, region = scene.fspt, context.region
        matrix = np.array(context.region_data.perspective_matrix.inverted(), dtype=np.float32)
        encoded = matrix.tobytes()
        if encoded != self._camera_bytes:
            self._camera, self._camera_bytes = CameraSnapshot(matrix), encoded
        return ViewRequest(self._camera,
                           max(1, region.width * settings.resolution_percentage // 100),
                           max(1, region.height * settings.resolution_percentage // 100),
                           settings.viewport_samples, settings.max_bounces, settings.seed, settings.device)

    def _scene_changed(self, depsgraph):
        from .fspt_bridge import forward_live_publication
        from .scene_access import active_domain_object
        from .surface_publication import current_surface_publication, id_token

        prior = self._view.snapshot.publications if self._view.snapshot is not None else ()
        current = current_surface_publication(depsgraph.scene)
        fluid_tokens = set()
        if len(prior) == 1 and forward_live_publication(prior[0], current):
            fluid_tokens = {current.object_token, current.mesh_token}
        domain = id_token(active_domain_object(getattr(depsgraph.scene, "original", depsgraph.scene)))
        for update in depsgraph.updates:
            value = update.id
            # View/camera edits change a small ray request, not scene triangles.
            kind = getattr(getattr(value, "bl_rna", None), "identifier", "")
            if kind == "Camera" or (kind == "Object" and getattr(value, "type", "") == "CAMERA"):
                continue
            # Scene RNA contains FSPT sample/resolution settings and time.
            # Actual object/material/world edits have their own updated IDs;
            # fluid changes arrive through the committed publication callback.
            if kind == "Scene":
                continue
            if id_token(value) in fluid_tokens:
                # The matching immutable publication already marked this
                # surface dirty. Other objects/material IDs remain hard edits.
                continue
            if kind == "Object" and domain is not None and id_token(value) == domain:
                continue
            return True
        return False

    @staticmethod
    def _scene_input_key(scene, layer):
        from .fspt_bridge import _surface_object
        from .surface_publication import id_token, surface_owner_token

        # Reassigning existing datablocks can emit only a Scene update. Keep
        # their identities separately from sample/resolution controls so these
        # changes still invalidate geometry/shading and unsupported overrides.
        output = _surface_object(scene)
        surface_inputs = (() if output is None else (
            np.array(output.matrix_world, dtype=np.float32).tobytes(),
            tuple((slot.link, id_token(slot.material)) for slot in output.material_slots)))
        return (id_token(scene.world), id_token(layer.material_override), surface_inputs,
                id_token(getattr(layer, "world_override", None)),
                surface_owner_token(scene),
                bool(getattr(getattr(scene, "fluidsim", None), "cache_playback_enabled", False)))

    def view_update(self, context, depsgraph):
        from .fspt_scene import extract_scene
        from .surface_publication import id_token, subscribe_surfaces

        view = self._state()
        try:
            scene = getattr(depsgraph.scene, "original", depsgraph.scene)
            source = (id_token(scene), id_token(depsgraph.view_layer))
            if source != self._scene_layer:
                if self._unsubscribe is not None:
                    self._unsubscribe()
                self._unsubscribe = subscribe_surfaces(scene, self._published)
                self._scene_layer = source
                view.rebind()
            self._location = (context.window.as_pointer(), context.area.as_pointer(),
                              context.region.as_pointer())
            view.request_update(self._request(context, scene))
            inputs = self._scene_input_key(scene, depsgraph.view_layer)
            if inputs != self._scene_inputs or self._scene_changed(depsgraph):
                view.invalidate()
            self._scene_inputs = inputs
            view.capture(lambda: extract_scene(depsgraph, for_viewport=True))
        except Exception as exc:
            view.fail(exc)
        self.tag_redraw()
        if view.needs_tick:
            _ensure_timer()

    def _draw_error(self, message):
        import blf
        import gpu

        gpu.state.active_framebuffer_get().clear(color=(0.04, 0.01, 0.04, 1.0))
        blf.position(0, 20, 40, 0)
        blf.size(0, 16)
        blf.color(0, 1.0, 0.55, 0.55, 1.0)
        blf.draw(0, "FSPT: " + message[:180])
        self.update_stats("FSPT Experimental", message)

    def view_draw(self, context, depsgraph):
        import gpu
        from gpu_extras.presets import draw_texture_2d

        from .fspt_bridge import SurfaceNotReadyError, validate_viewport_publications

        global LAST_VIEWPORT_INFO, _DRAW_COUNT
        scene = depsgraph.scene
        region = context.region
        try:
            view = self._state()
            # View navigation can trigger draw without a dependency-graph
            # update. Retain only 16 camera floats and scalar request metadata.
            view.request_update(self._request(context, scene))
            if view.needs_tick:
                _ensure_timer()
            if view.error:
                raise RuntimeError(view.error)
            frame = view.presentable
            if frame is None:
                self._view_info = LAST_VIEWPORT_INFO = {"status": "WAITING", "reason": "Waiting for an eligible completed image"}
                self._draw_error("Waiting for an eligible completed image")
                return
            require_current(view.presentation_key, frame.key)
            presentation = validate_viewport_publications(scene, frame.publications,
                                                          view_layer=depsgraph.view_layer)
            width, height = frame.key[2:4]
            if self._texture_frame is not frame:
                buffer = gpu.types.Buffer("FLOAT", width * height * 4, frame.pixels.ravel())
                self._texture = gpu.types.GPUTexture((width, height), format="RGBA32F", data=buffer)
                self._texture_frame = frame
            previous_blend = gpu.state.blend_get()
            bound = False
            try:
                gpu.state.blend_set("ALPHA_PREMULT")
                self.bind_display_space_shader(scene)
                bound = True
                draw_texture_2d(self._texture, (0, 0), region.width, region.height)
            finally:
                try:
                    if bound:
                        self.unbind_display_space_shader()
                finally:
                    gpu.state.blend_set(previous_blend)
            _DRAW_COUNT += 1
            self._view_info = LAST_VIEWPORT_INFO = _frame_info(frame, scene, width, height)
            self._view_info.update(draw_count=_DRAW_COUNT, region_id=region.as_pointer(),
                                   capture_statistics=dict(view.statistics), presentation=presentation)
            source = frame.publications[0] if frame.publications else None
            label = (f"{source.source} · surface {source.solver_frame if source.source == 'LIVE' else source.cache_frame}"
                     f" · timeline {frame.scene_frame} · " if source is not None else "CUDA · ")
            if presentation["status"] == "RETAINED":
                label += f"retained / update pending · publication age {presentation['publication_age_seconds'] * 1000:.0f} ms · "
            self.update_stats("FSPT Experimental", f"{label}{frame.samples} samples · pinned-host image")
        except SurfaceNotReadyError as exc:
            # A publication may retire after collection but before this draw.
            # Reject only these pixels. A newer eligible job may already be
            # pending; collection and current-capture checks own its validity.
            # Expiring an old display must not invalidate that newer job.
            self._view.presentable = self._view.presented_snapshot = None
            self._texture = self._texture_frame = None
            self._view_info = LAST_VIEWPORT_INFO = {"status": "WAITING", "reason": str(exc)}
            _ensure_timer()
            self._draw_error(str(exc))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            # Drawing cannot wait for tracing or retire CUDA resources. The
            # timer/lifecycle owner handles work; this path only invalidates.
            if self._view is not None:
                self._view.fail(exc)
            self._texture = self._texture_frame = None
            self._view_info = LAST_VIEWPORT_INFO = {"status": "ERROR", "error": message}
            self._draw_error(message)


class FSPTRenderPanel(bpy.types.Panel):
    bl_label = "FSPT Experimental"
    bl_idname = "RENDER_PT_fluidsim_fspt"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.scene is not None and context.scene.render.engine == ENGINE_ID

    def draw(self, context):
        layout = self.layout
        layout.label(text="Asynchronous CUDA · evaluated fluid bridge", icon="EXPERIMENTAL")
        layout.label(text="Pinned-host presentation; experimental")
        layout.label(text="Diffuse, smooth Glass, Emission; constant World")
        layout.label(text="Unsupported graphs produce a diagnostic.")
        for prop in ("device", "viewport_samples", "resolution_percentage", "samples", "max_bounces", "seed"):
            layout.prop(context.scene.fspt, prop)


CLASSES = (FSPTSettings, FSPTRenderEngine, FSPTRenderPanel)


def register():
    global _REGISTERED
    if hasattr(bpy.types.Scene, "fspt"):
        raise RuntimeError("FSPT Scene property already registered")
    with class_registration(CLASSES), handler_registration(tuple(
            (getattr(bpy.app.handlers, name), _on_fspt_view_history_pre)
            for name in ("load_pre", "undo_pre", "redo_pre"))):
        bpy.types.Scene.fspt = PointerProperty(type=FSPTSettings)
    _REGISTERED = True


def unregister():
    global _REGISTERED
    if not _REGISTERED:
        return
    _REGISTERED = False
    if bpy.app.timers.is_registered(_pump_views):
        bpy.app.timers.unregister(_pump_views)
    for name in ("load_pre", "undo_pre", "redo_pre"):
        callbacks = getattr(bpy.app.handlers, name)
        if _on_fspt_view_history_pre in callbacks:
            callbacks.remove(_on_fspt_view_history_pre)
    for engine in tuple(_ENGINES):
        try:
            engine.close()
        except Exception as exc:
            print(f"FSPT backend teardown failed: {exc}")
    if hasattr(bpy.types.Scene, "fspt"):
        del bpy.types.Scene.fspt
    unregister_classes(CLASSES)
    _REGISTERED = False
