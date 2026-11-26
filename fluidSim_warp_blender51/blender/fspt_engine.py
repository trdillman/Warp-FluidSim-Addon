"""Experimental FSPT RenderEngine adapter. Blender owns all callback execution."""
import weakref

import bpy
import numpy as np
from bpy.props import IntProperty, PointerProperty, StringProperty

from ..fspt.contracts import MAX_BOUNCES, CameraSnapshot, frame_key, require_current
from ..unregister_guard import class_registration, unregister_classes

ENGINE_ID = "FLUIDSIM_FSPT"
LAST_RENDER_INFO = {}
LAST_VIEWPORT_INFO = {}
_ENGINES = weakref.WeakSet()
_DRAW_COUNT = 0
_REGISTERED = False


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
        self._backend = None
        self._snapshot = None
        self._snapshot_error = ""
        self._last_frame = None
        self._texture = None
        self._device = None
        _ENGINES.add(self)

    def close(self):
        backend = getattr(self, "_backend", None)
        self._backend = None
        self._snapshot = None
        self._last_frame = None
        self._texture = None
        if backend is not None:
            backend.close()

    def __del__(self):
        try:
            self.close()
        finally:
            parent = getattr(super(), "__del__", None)
            if parent is not None:
                parent()

    def _get_backend(self, settings):
        if self._backend is None or self._device != settings.device:
            if self._backend is not None:
                self._backend.close()
            self._last_frame = self._texture = None
            from ..fspt.backend import CudaBackend

            self._backend = CudaBackend(settings.device)
            self._device = settings.device
        return self._backend

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
            prepare_final_publication(depsgraph)
            snapshot = extract_scene(depsgraph, for_viewport=False)
            camera = camera_snapshot(depsgraph, width, height)
            # A final job starts at sample zero even if this instance is reused.
            self.close()
            backend = self._get_backend(settings)
            result = self.begin_result(0, 0, width, height, layer=depsgraph.view_layer.name)
            for _ in range(settings.samples):
                if self.test_break():
                    cancelled = True
                    break
                frame = backend.render(snapshot, camera, width, height,
                                       max_bounces=settings.max_bounces, seed=settings.seed)
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
            try:
                self.close()
            except Exception as exc:
                message = f"FSPT GPU teardown failed: {exc}"
                LAST_RENDER_INFO["status"] = "ERROR"
                LAST_RENDER_INFO.setdefault("error", message)
                LAST_RENDER_INFO["cleanup_error"] = message
                self.report({"ERROR"}, message)
                self.error_set(LAST_RENDER_INFO["error"])

    def view_update(self, context, depsgraph):
        from .fspt_scene import extract_scene

        try:
            self._snapshot = extract_scene(depsgraph, for_viewport=True)
            self._snapshot_error = ""
        except Exception as exc:
            self._snapshot = None
            self._snapshot_error = f"{type(exc).__name__}: {exc}"
            self._last_frame = self._texture = None
        self.tag_redraw()

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

        from .fspt_bridge import validate_publications

        global LAST_VIEWPORT_INFO, _DRAW_COUNT
        scene = depsgraph.scene
        settings = scene.fspt
        region = context.region
        try:
            if self._snapshot_error:
                raise RuntimeError(self._snapshot_error)
            if self._snapshot is None:
                self.view_update(context, depsgraph)
            if self._snapshot is None:
                raise RuntimeError(self._snapshot_error or "No scene snapshot")
            snapshot = self._snapshot
            validate_publications(scene, snapshot.publications, for_viewport=True,
                                  view_layer=depsgraph.view_layer)
            width = max(1, region.width * settings.resolution_percentage // 100)
            height = max(1, region.height * settings.resolution_percentage // 100)
            matrix = np.array(context.region_data.perspective_matrix.inverted(), dtype=np.float32)
            camera = CameraSnapshot(matrix)
            key = frame_key(snapshot, camera, width, height, settings.max_bounces, settings.seed)
            backend = self._get_backend(settings)
            if (self._last_frame is None or self._last_frame.key != key
                    or self._last_frame.samples < settings.viewport_samples):
                frame = backend.render(snapshot, camera, width, height,
                                       max_bounces=settings.max_bounces, seed=settings.seed)
                require_current(key, frame.key)
                buffer = gpu.types.Buffer("FLOAT", width * height * 4, frame.pixels.ravel())
                self._texture = gpu.types.GPUTexture((width, height), format="RGBA32F", data=buffer)
                self._last_frame = frame
            frame = self._last_frame
            require_current(key, frame.key)
            validate_publications(scene, frame.publications, for_viewport=True,
                                  view_layer=depsgraph.view_layer)
            previous_blend = gpu.state.blend_get()
            gpu.state.blend_set("ALPHA_PREMULT")
            self.bind_display_space_shader(scene)
            try:
                draw_texture_2d(self._texture, (0, 0), region.width, region.height)
            finally:
                self.unbind_display_space_shader()
                gpu.state.blend_set(previous_blend)
            _DRAW_COUNT += 1
            LAST_VIEWPORT_INFO = _frame_info(frame, scene, width, height)
            LAST_VIEWPORT_INFO["draw_count"] = _DRAW_COUNT
            source = frame.publications[0] if frame.publications else None
            label = (f"{source.source} · surface {source.solver_frame if source.source == 'LIVE' else source.cache_frame}"
                     f" · timeline {scene.frame_current} · " if source is not None else "CUDA · ")
            self.update_stats("FSPT Experimental", f"{label}{frame.samples} samples · host-staged image")
            if frame.samples < settings.viewport_samples:
                self.tag_redraw()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            try:
                self.close()
            except Exception as cleanup:
                message += f"; GPU teardown: {cleanup}"
            LAST_VIEWPORT_INFO = {"status": "ERROR", "error": message}
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
        layout.label(text="CUDA reference · evaluated fluid bridge", icon="EXPERIMENTAL")
        layout.label(text="Diffuse, smooth Glass, Emission; constant World")
        layout.label(text="Unsupported graphs produce a diagnostic.")
        for prop in ("device", "viewport_samples", "resolution_percentage", "samples", "max_bounces", "seed"):
            layout.prop(context.scene.fspt, prop)


CLASSES = (FSPTSettings, FSPTRenderEngine, FSPTRenderPanel)


def register():
    global _REGISTERED
    if hasattr(bpy.types.Scene, "fspt"):
        raise RuntimeError("FSPT Scene property already registered")
    with class_registration(CLASSES):
        bpy.types.Scene.fspt = PointerProperty(type=FSPTSettings)
    _REGISTERED = True


def unregister():
    global _REGISTERED
    if not _REGISTERED:
        return
    for engine in tuple(_ENGINES):
        try:
            engine.close()
        except Exception as exc:
            print(f"FSPT backend teardown failed: {exc}")
    if hasattr(bpy.types.Scene, "fspt"):
        del bpy.types.Scene.fspt
    unregister_classes(CLASSES)
    _REGISTERED = False
