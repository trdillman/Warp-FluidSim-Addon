"""Capability-owned Blender operators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from ..blender.scene_access import scene_identity
from . import common as _runtime

_FLUIDSIM_OT_session_base = _runtime._FLUIDSIM_OT_session_base


def _scene_is_available(scene, key: int) -> bool:
    try:
        pointer = getattr(scene, "as_pointer", None)
        if callable(pointer) and int(pointer()) != int(key):
            return False
    except (ReferenceError, TypeError, ValueError):
        return False
    scenes = getattr(getattr(bpy, "data", None), "scenes", None)
    if scenes is None:
        return True
    try:
        return any(scene_identity(candidate) == int(key) for candidate in scenes)
    except (ReferenceError, TypeError):
        return False


def _update_bake_rna(scene, controller) -> bool:
    try:
        fs = scene.fluidsim
        fs.bake_status = controller.status.value
        fs.bake_progress = float(controller.progress)
        fs.bake_message = str(controller.message)
        fs.status = str(controller.message)
    except ReferenceError:
        return False
    return True


_DEFAULT_BAKE_TIMER_INTERVAL = 0.05
_MIN_BAKE_TIMER_INTERVAL = 0.001


class _FLUIDSIM_OT_production_bake_modal:
    resume_existing = False
    restart_existing = False
    rebake = False
    rebake_frame = 0
    _timer = None
    _timer_interval = _DEFAULT_BAKE_TIMER_INTERVAL
    _window = None
    _run = None
    _window_manager = None

    @classmethod
    def poll(cls, context):
        from ..blender import production_bake

        if production_bake.active_bakes():
            cls.poll_message_set("A production bake is already active")
            return False
        return _runtime._runtime_poll(cls)

    def execute(self, context):
        if _runtime._unsaved_relative_cache_path(context.scene):
            self.report({"ERROR"}, _runtime.UNSAVED_RELATIVE_CACHE_MESSAGE)
            return {"CANCELLED"}
        from ..blender.cache_bake import BakeRun

        self._window_manager = context.window_manager
        self._timer = None
        run = None
        try:
            run = BakeRun(
                context.scene,
                resume=self.resume_existing,
                restart=bool(self.restart_existing),
                rebake_from=int(self.rebake_frame) if self.rebake else None,
            )
            self._run = run
            _update_bake_rna(run.scene, run.controller)
            self._window = context.window
            self._timer_interval = _DEFAULT_BAKE_TIMER_INTERVAL
            self._timer = self._window_manager.event_timer_add(
                self._timer_interval, window=self._window
            )
            self._window_manager.modal_handler_add(self)
        except Exception as exc:  # noqa: BLE001
            cleanup_errors = []
            try:
                if self._timer is not None:
                    self._window_manager.event_timer_remove(self._timer)
                    self._timer = None
            except Exception as cleanup_exc:  # noqa: BLE001
                cleanup_errors.append(f"timer cleanup failed: {cleanup_exc}")
            if run is not None:
                try:
                    run.close()
                except Exception as cleanup_exc:  # noqa: BLE001
                    cleanup_errors.append(f"cache restore failed: {cleanup_exc}")
            self._run = None
            suffix = "" if not cleanup_errors else f"; {'; '.join(cleanup_errors)}"
            self.report(
                {"ERROR"}, f"Production bake could not start: {exc}{suffix}"
            )
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _advance(self, failure):
        """Tick the run once; a failure to settle its cache fails the bake.

        Returns the controller's suggested delay before the next tick, or
        ``None`` when it has no suggestion (or the tick failed).
        """
        from ..blender.production_bake import BakeStatus

        try:
            return self._run.tick()
        except Exception as exc:  # noqa: BLE001
            controller = self._run.controller
            controller.status = BakeStatus.FAILED
            controller.error = f"{type(exc).__name__}: {exc}"
            controller.message = f"{failure}: {exc}"
            return None

    def _retime(self, context, delay):
        """Re-arm the modal timer at the controller's suggested delay.

        A fixed 50 ms timer capped light frames at 20 per second although the
        controller asks to be ticked again after 10 ms. The timer is replaced
        only when the suggestion changes.
        """
        if delay is None or self._timer is None:
            return
        try:
            delay = max(_MIN_BAKE_TIMER_INTERVAL, float(delay))
        except (TypeError, ValueError):
            return
        if not math.isfinite(delay) or math.isclose(delay, self._timer_interval):
            return
        window_manager = self._window_manager or context.window_manager
        previous = self._timer
        self._timer = window_manager.event_timer_add(
            delay, window=self._window or context.window)
        self._timer_interval = delay
        window_manager.event_timer_remove(previous)

    def _finish(self, context, *, scene_available=None):
        from ..blender.production_bake import BakeStatus

        run = self._run
        controller = run.controller
        cleanup_error = None
        try:
            if self._timer is not None:
                window_manager = self._window_manager or context.window_manager
                window_manager.event_timer_remove(self._timer)
                self._timer = None
        except Exception as exc:  # noqa: BLE001
            cleanup_error = exc
        finally:
            try:
                run.close()
            except Exception as exc:  # noqa: BLE001
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            controller.status = BakeStatus.FAILED
            controller.error = f"{type(cleanup_error).__name__}: {cleanup_error}"
            controller.message = f"Production bake cleanup failed: {cleanup_error}"
        if scene_available is None:
            scene_available = _scene_is_available(run.scene, run.scene_key)
        if scene_available:
            _update_bake_rna(run.scene, controller)
        if controller.status is BakeStatus.COMPLETE:
            self.report({"INFO"}, controller.message)
            return {"FINISHED"}
        if controller.status is BakeStatus.CANCELLED:
            self.report({"WARNING"}, controller.message)
            return {"CANCELLED"}
        self.report({"ERROR"}, controller.error or controller.message)
        return {"CANCELLED"}

    def _cancel_removed_scene(self, context):
        self._run.abandon("Production bake cancelled because its scene was removed")
        return self._finish(context, scene_available=False)

    def modal(self, context, event):
        run = self._run
        if not _scene_is_available(run.scene, run.scene_key):
            return self._cancel_removed_scene(context)
        # Esc only (owner decision D3): the bake runs in the background, so
        # right-click must reach Blender for selection and context menus.
        if event.type == "ESC" and event.value == "PRESS":
            run.request_cancel()
            _update_bake_rna(run.scene, run.controller)
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        delay = self._advance("Production bake failed")
        _update_bake_rna(run.scene, run.controller)
        if run.finished:
            return self._finish(context)
        self._retime(context, delay)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        # Blender requires Operator.cancel to return None (it calls this when a
        # file is opened mid-bake); a returned set raised inside Blender.
        run = self._run
        if run is None:
            return
        if not _scene_is_available(run.scene, run.scene_key):
            self._cancel_removed_scene(context)
            return
        run.request_cancel()
        self._advance("Production bake cleanup failed")
        self._finish(context)


class FLUIDSIM_OT_bake_start(_FLUIDSIM_OT_production_bake_modal, bpy.types.Operator):
    """Start a fresh one-frame-transaction production bake"""

    bl_idname = "fluidsim.bake_start"
    bl_label = "Start Production Bake"
    bl_options = {"REGISTER"}
    resume_existing = False
    restart_existing: BoolProperty(
        name="Restart Existing Cache",
        default=False,
        options={"SKIP_SAVE"},
        description="Delete only FluidSim cache artifacts before baking",
    )
    expected_cache_path: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, _event):
        if not self.restart_existing:
            return self.execute(context)
        try:
            self.expected_cache_path = str(_runtime._resolved_cache_path(context.scene))
        except Exception as exc:
            self.report({"ERROR"}, f"Cache path could not be resolved: {exc}")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        layout.label(text="Replace Existing Production Cache?", icon="ERROR")
        layout.label(text=self.expected_cache_path, icon="FILE_FOLDER")
        layout.label(text="FluidSim will replace its manifest, frames, checkpoints, and logs.")
        layout.label(text="Unrelated files in the directory are not removed.", icon="INFO")

    def execute(self, context):
        if self.restart_existing and self.expected_cache_path:
            current = str(_runtime._resolved_cache_path(context.scene))
            if current != self.expected_cache_path:
                self.report(
                    {"ERROR"}, "Cache path changed after confirmation; review Restart again"
                )
                return {"CANCELLED"}
        return super().execute(context)


class FLUIDSIM_OT_bake_resume(_FLUIDSIM_OT_production_bake_modal, bpy.types.Operator):
    """Resume from the last verified committed checkpoint, or re-bake from a frame"""

    bl_idname = "fluidsim.bake_resume"
    bl_label = "Resume Production Bake"
    bl_options = {"REGISTER"}
    resume_existing = True
    # "Re-bake from frame N": keep the frames before N, replace the rest. Offered
    # when Resume was refused because a source or collider animation changed.
    rebake: BoolProperty(
        name="Re-bake From Frame",
        default=False,
        options={"SKIP_SAVE"},
        description="Keep the baked frames before the chosen frame and bake from it onward",
    )
    rebake_frame: IntProperty(
        name="Frame",
        default=0,
        min=-1048574,
        max=1048574,
        options={"SKIP_SAVE"},
        description="First frame to bake again",
    )

    def invoke(self, context, _event):
        if not self.rebake:
            return self.execute(context)
        return context.window_manager.invoke_props_dialog(self, width=480)

    def draw(self, _context):
        layout = self.layout
        layout.label(text=f"Re-bake From Frame {int(self.rebake_frame)}?", icon="ERROR")
        layout.label(
            text=f"Frames {int(self.rebake_frame)} onward in this cache are replaced.")
        layout.label(text="Earlier frames are kept.", icon="INFO")


class FLUIDSIM_OT_bake_cancel(bpy.types.Operator):
    """Request cancellation at the next manifest-committed frame boundary"""

    bl_idname = "fluidsim.bake_cancel"
    bl_label = "Cancel Production Bake"

    # Off by default: the button cancels only the bake this scene owns, so a
    # click in one scene never stops a bake started in another.
    all_scenes: BoolProperty(
        name="All Scenes",
        default=False,
        options={"SKIP_SAVE"},
        description="Cancel the running Production Bake even when another scene owns it",
    )

    @classmethod
    def poll(cls, _context):
        from ..blender import production_bake

        if not production_bake.active_bakes():
            cls.poll_message_set("No production bake is active")
            return False
        return True

    def execute(self, context):
        from ..blender import production_bake

        running = production_bake.active_bakes()
        if not bool(getattr(self, "all_scenes", False)):
            key = scene_identity(getattr(context, "scene", None))
            owned = tuple(
                (scene, controller) for scene, controller in running
                if scene_identity(scene) == key)
            if not owned:
                names = ", ".join(
                    f"'{getattr(scene, 'name', '')}'" for scene, _ in running
                    if getattr(scene, "name", ""))
                where = f" in scene {names}" if names else " in another scene"
                self.report(
                    {"WARNING"},
                    f"The Production Bake is running{where}; switch to that "
                    "scene to cancel it")
                return {"CANCELLED"}
            running = owned
        for scene, controller in running:
            controller.request_cancel()
            _update_bake_rna(scene, controller)
        return {"FINISHED"}


class FLUIDSIM_OT_cache_integrity_scan(bpy.types.Operator):
    """Verify the manifest-authoritative cache and report orphan files"""

    bl_idname = "fluidsim.cache_integrity_scan"
    bl_label = "Scan Cache Integrity"

    def execute(self, context):
        from ..blender.cache_manager import inspect_cache

        path = Path(bpy.path.abspath(context.scene.fluidsim.cache_path))
        inspection = inspect_cache(path)
        context.scene.fluidsim.status = inspection.message
        severity = {"INFO"} if inspection.healthy else {"ERROR"}
        detail = inspection.message
        if inspection.orphan_files:
            detail += f"; {len(inspection.orphan_files)} orphan(s)"
        self.report(severity, detail)
        return {"FINISHED"}


class FLUIDSIM_OT_cache_remove_orphans(bpy.types.Operator):
    """Remove only files not referenced by the committed cache manifest"""

    bl_idname = "fluidsim.cache_remove_orphans"
    bl_label = "Remove Cache Orphans"
    bl_options = {"REGISTER"}

    expected_cache_path: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})
    expected_orphans_json: StringProperty(default="[]", options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, _event):
        from ..blender.cache_manager import inspect_cache

        try:
            path = _runtime._resolved_cache_path(context.scene)
            inspection = inspect_cache(path)
        except Exception as exc:
            self.report({"ERROR"}, f"Cache could not be inspected: {exc}")
            return {"CANCELLED"}
        if not inspection.orphan_files:
            self.report({"INFO"}, "No orphan cache files were found")
            return {"CANCELLED"}
        self.expected_cache_path = str(path)
        self.expected_orphans_json = json.dumps(list(inspection.orphan_files))
        return context.window_manager.invoke_props_dialog(self, width=620)

    def draw(self, _context):
        layout = self.layout
        orphans = tuple(json.loads(self.expected_orphans_json or "[]"))
        layout.label(text=f"Remove {len(orphans)} uncommitted/orphan file(s)?", icon="ERROR")
        layout.label(text=self.expected_cache_path, icon="FILE_FOLDER")
        box = layout.box()
        for relative in orphans:
            box.label(text=str(relative), icon="FILE")

    def execute(self, context):
        from ..blender.cache_manager import inspect_cache, remove_orphans

        path = _runtime._resolved_cache_path(context.scene)
        if self.expected_cache_path and str(path) != self.expected_cache_path:
            self.report({"ERROR"}, "Cache path changed after confirmation; inspect again")
            return {"CANCELLED"}
        expected = tuple(json.loads(self.expected_orphans_json or "[]"))
        current = tuple(inspect_cache(path).orphan_files)
        if expected and current != expected:
            self.report(
                {"ERROR"}, "Cache orphan inventory changed after confirmation; inspect again"
            )
            return {"CANCELLED"}
        removed = remove_orphans(path)
        self.report({"INFO"}, f"Removed {len(removed)} orphan cache file(s)")
        return {"FINISHED"}


class FLUIDSIM_OT_cache_free(bpy.types.Operator):
    """Transactionally free the configured FluidSim cache"""

    bl_idname = "fluidsim.cache_free"
    bl_label = "Free Cache"
    bl_options = {"REGISTER"}

    expected_cache_path: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, _event):
        try:
            self.expected_cache_path = str(_runtime._resolved_cache_path(context.scene))
        except Exception as exc:
            self.report({"ERROR"}, f"Cache path could not be resolved: {exc}")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        layout.label(text="Free Entire Cache?", icon="ERROR")
        layout.label(text=self.expected_cache_path, icon="FILE_FOLDER")
        layout.label(text="The directory is transactionally renamed before deletion.")

    def execute(self, context):
        from ..blender.cache_manager import free_cache

        path = _runtime._resolved_cache_path(context.scene)
        if self.expected_cache_path and str(path) != self.expected_cache_path:
            self.report({"ERROR"}, "Cache path changed after confirmation; review Free Cache again")
            return {"CANCELLED"}
        try:
            retained = free_cache(path)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Cache could not be freed: {exc}")
            return {"CANCELLED"}
        if retained is not None:
            self.report({"ERROR"}, f"Cache deletion failed; data remains at {retained}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Cache freed")
        return {"FINISHED"}


class FLUIDSIM_OT_cache_move(bpy.types.Operator):
    """Copy, verify, publish, and then remove a cache"""

    bl_idname = "fluidsim.cache_move"
    bl_label = "Move Cache"
    bl_options = {"REGISTER"}

    destination: StringProperty(name="Destination", subtype="DIR_PATH")
    expected_source: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, _event):
        source = _runtime._resolved_cache_path(context.scene)
        self.expected_source = str(source)
        if not self.destination:
            self.destination = str(source.with_name(source.name + "_moved"))
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        layout.label(text=f"Source: {self.expected_source}", icon="FILE_FOLDER")
        layout.prop(self, "destination")
        layout.label(text="The copy is verified before the source is removed.", icon="INFO")

    def execute(self, context):
        from ..blender.cache_manager import move_cache

        fs = context.scene.fluidsim
        source = _runtime._resolved_cache_path(context.scene)
        if self.expected_source and str(source) != self.expected_source:
            self.report({"ERROR"}, "Cache source changed after confirmation; review Move again")
            return {"CANCELLED"}
        destination = Path(bpy.path.abspath(self.destination)).expanduser().resolve()
        try:
            moved = move_cache(source, destination)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Cache move failed: {exc}")
            return {"CANCELLED"}
        fs.cache_path = str(moved)
        self.report({"INFO"}, f"Cache moved to {moved}")
        return {"FINISHED"}


class FLUIDSIM_OT_cache_rename(bpy.types.Operator):
    """Rename only the cache directory leaf"""

    bl_idname = "fluidsim.cache_rename"
    bl_label = "Rename Cache"
    bl_options = {"REGISTER"}

    name: StringProperty(name="New Directory Name", default="fluidsim_cache")
    expected_source: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, _event):
        source = _runtime._resolved_cache_path(context.scene)
        self.expected_source = str(source)
        self.name = source.name
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        layout.label(text=f"Current: {self.expected_source}", icon="FILE_FOLDER")
        layout.prop(self, "name")

    def execute(self, context):
        from ..blender.cache_manager import rename_cache

        fs = context.scene.fluidsim
        source = _runtime._resolved_cache_path(context.scene)
        if self.expected_source and str(source) != self.expected_source:
            self.report({"ERROR"}, "Cache source changed after confirmation; review Rename again")
            return {"CANCELLED"}
        try:
            renamed = rename_cache(source, self.name)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Cache rename failed: {exc}")
            return {"CANCELLED"}
        fs.cache_path = str(renamed)
        self.report({"INFO"}, f"Cache renamed to {renamed.name}")
        return {"FINISHED"}


class FLUIDSIM_OT_cache_open_folder(bpy.types.Operator):
    """Open the configured cache folder without shell interpolation"""

    bl_idname = "fluidsim.cache_open_folder"
    bl_label = "Open Cache Folder"

    def execute(self, context):
        from ..blender.cache_manager import open_cache_folder

        try:
            open_cache_folder(Path(bpy.path.abspath(context.scene.fluidsim.cache_path)))
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not open cache folder: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}
