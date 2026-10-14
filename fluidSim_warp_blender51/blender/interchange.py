"""Interchange export from a baked cache: Alembic mesh sequence of the liquid surface.

The operator is cache-driven and never touches the solver. Blender's own
Alembic exporter walks the frame range and evaluates the scene per frame; each
evaluation fires ``frame_change_post``, which is exactly how cache playback
publishes a frame (``playback.handle_frame_change`` -> ``play_cache_frame`` ->
``output_publication.update_outputs_from_cache_frame``). So the export is:
cache playback on, live on (the frame handler returns early otherwise), only
the surface object selected, then ``bpy.ops.wm.alembic_export`` over the range.
Every piece of scene state the export changes is restored in a ``finally``.

The export must not run inside ``playback.suppress_frame_handlers``: that would
block the very handler that publishes each frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bpy
from bpy.props import IntProperty, StringProperty

from ..io import cache, cache_schema
from . import output_publication, playback, production_bake

SURFACE_KIND = output_publication.SURFACE_NAME


class ExportRefused(Exception):
    """The export cannot start; the message is shown to the user as-is."""


@dataclass(frozen=True)
class ExportPlan:
    """What an export would write, resolved and validated before any state change."""

    base_dir: str
    frames: tuple[int, ...]
    surface: object

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]


def available_frames(scene) -> tuple[int, ...]:
    """Verified, contiguous cache frames that cache playback can also play.

    Raises :class:`ExportRefused` with a user-facing reason when there are none.
    """
    fs = scene.fluidsim
    base = bpy.path.abspath(fs.cache_path)
    try:
        manifest = cache_schema.load_manifest(cache_schema.manifest_path(base))
    except ValueError as exc:
        raise ExportRefused(f"The cache manifest is unreadable: {exc}") from exc
    if manifest is None:
        try:
            version = cache.detect_cache_format(base)
        except Exception:  # noqa: BLE001 - only used to pick a better message
            version = 0
        if version not in (0, cache.CACHE_FORMAT_VERSION):
            raise ExportRefused(
                "This cache was baked by an older FluidSim. Re-bake it to export.")
        raise ExportRefused("There is no baked cache to export. Bake first.")
    try:
        frames = cache_schema.verified_contiguous_frames(base, manifest)
    except ValueError as exc:
        raise ExportRefused(
            f"The cache is not verified and contiguous: {exc}") from exc
    if not frames:
        raise ExportRefused("The cache has no committed frames.")
    if getattr(fs, "cache_use_custom_range", False):
        low, high = cache.resolve_frame_range(
            scene, use_custom=True,
            frame_start=fs.cache_frame_start, frame_end=fs.cache_frame_end)
        frames = tuple(f for f in frames if low <= f <= high)
        if not frames:
            raise ExportRefused(
                "The Storage & Range playback range contains no cached frames.")
    return frames


def plan_export(scene, frame_start: int = 0, frame_end: int = 0) -> ExportPlan:
    """Validate everything the export needs; ``0`` bounds mean the whole cache."""
    if production_bake.bake_is_active(scene):
        raise ExportRefused("A production bake is running. Wait for it or cancel it.")
    frames = available_frames(scene)
    start = int(frame_start) or frames[0]
    end = int(frame_end) or frames[-1]
    if start > end:
        raise ExportRefused(f"The export range {start}-{end} is empty.")
    if start < frames[0] or end > frames[-1]:
        raise ExportRefused(
            f"The export range {start}-{end} is outside the cached frames "
            f"{frames[0]}-{frames[-1]}.")
    surface = output_publication.output_object(scene, SURFACE_KIND)
    if surface is None:
        raise ExportRefused(
            "There is no FluidSim surface object. Enable the surface output and "
            "play a cached frame first.")
    return ExportPlan(
        base_dir=bpy.path.abspath(scene.fluidsim.cache_path),
        frames=tuple(f for f in frames if start <= f <= end),
        surface=surface,
    )


class _SceneState:
    """Everything the export touches, captured up front and restored on exit."""

    def __init__(self, scene, surface):
        self.scene = scene
        self.surface = surface
        fs = scene.fluidsim
        self.live = bool(fs.live)
        self.cache_playback = bool(getattr(fs, "cache_playback_enabled", False))
        self.frame = int(scene.frame_current)
        self.view_layer = bpy.context.view_layer
        self.selected = [o for o in self.view_layer.objects if o.select_get()]
        self.active = self.view_layer.objects.active
        self.hide_viewport = bool(surface.hide_viewport)
        self.hide_get = bool(surface.hide_get())

    def enter(self, plan: ExportPlan):
        scene = self.scene
        fs = scene.fluidsim
        # Releases the solver session; refused (False) during a production bake.
        if not playback.set_cache_playback_enabled(scene, True):
            raise ExportRefused(str(fs.status))
        with playback.suppress_mode_transitions():
            fs.cache_playback_enabled = True
            # The frame handler returns early unless live is on; with cache
            # playback on, live does not start the solver.
            fs.live = True
        # A stale cursor would skip republishing the current frame.
        playback.clear_cache_cursors()

        surface = self.surface
        surface.hide_viewport = False
        surface.hide_set(False)
        for obj in self.view_layer.objects:
            obj.select_set(False)
        surface.select_set(True)
        self.view_layer.objects.active = surface

    def restore(self):
        scene = self.scene
        fs = scene.fluidsim
        problems = []

        def attempt(label, action):
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - keep restoring the rest
                problems.append(f"{label}: {exc}")

        def restore_selection():
            surface = self.surface
            for obj in self.view_layer.objects:
                obj.select_set(False)
            for obj in self.selected:
                try:
                    obj.select_set(True)
                except (ReferenceError, RuntimeError):
                    pass
            self.view_layer.objects.active = self.active
            surface.hide_viewport = self.hide_viewport
            surface.hide_set(self.hide_get)

        def restore_frame():
            # Still in cache-playback mode, so this republishes the original
            # frame's mesh when the cache holds it.
            playback.clear_cache_cursors()
            if scene.frame_current != self.frame:
                scene.frame_set(self.frame)

        def restore_modes():
            with playback.suppress_mode_transitions():
                fs.live = self.live
                fs.cache_playback_enabled = self.cache_playback
            if not self.cache_playback:
                # Pops cursors and, if live was on, re-enters live simulation.
                playback.set_cache_playback_enabled(scene, False)

        attempt("selection", restore_selection)
        attempt("frame", restore_frame)
        attempt("playback mode", restore_modes)
        return problems


def _normalized_path(filepath: str) -> str:
    path = Path(bpy.path.abspath(filepath))
    if path.suffix.lower() != ".abc":
        path = path.with_suffix(".abc")
    return str(path)


def export_alembic(scene, filepath: str, plan: ExportPlan) -> list[str]:
    """Run the exporter under cache playback; returns restore problems (if any)."""
    state = _SceneState(scene, plan.surface)
    try:
        state.enter(plan)
        result = bpy.ops.wm.alembic_export(
            filepath=filepath,
            start=plan.start,
            end=plan.end,
            selected=True,
            flatten=True,
            export_hair=False,
            export_particles=False,
            uvs=False,
            normals=True,
            vcolors=False,
            evaluation_mode="RENDER",
        )
        if "FINISHED" not in result:
            raise RuntimeError("Blender's Alembic exporter did not finish.")
    finally:
        problems = state.restore()
    return problems


class FLUIDSIM_OT_export_alembic(bpy.types.Operator):
    """Export the baked liquid surface as an Alembic mesh sequence"""

    bl_idname = "fluidsim.export_alembic"
    bl_label = "Export Surface as Alembic"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH", default="")
    filter_glob: StringProperty(default="*.abc", options={"HIDDEN"})
    frame_start: IntProperty(
        name="Start", default=0, min=0,
        description="First frame to export; 0 uses the first cached frame")
    frame_end: IntProperty(
        name="End", default=0, min=0,
        description="Last frame to export; 0 uses the last cached frame")

    @classmethod
    def poll(cls, context):
        return getattr(context, "scene", None) is not None

    def _plan(self, context):
        try:
            return plan_export(context.scene, self.frame_start, self.frame_end)
        except ExportRefused as exc:
            self.report({"ERROR"}, str(exc))
            return None

    def invoke(self, context, event):
        if self._plan(context) is None:
            return {"CANCELLED"}
        if not self.filepath:
            self.filepath = "fluidsim_surface.abc"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if not self.filepath:
            self.report({"ERROR"}, "Choose an output file.")
            return {"CANCELLED"}
        scene = context.scene
        plan = self._plan(context)
        if plan is None:
            return {"CANCELLED"}
        filepath = _normalized_path(self.filepath)
        try:
            problems = export_alembic(scene, filepath, plan)
        except ExportRefused as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:  # noqa: BLE001 - surface any exporter failure
            self.report({"ERROR"}, f"Alembic export failed: {exc}")
            return {"CANCELLED"}
        for problem in problems:
            self.report({"WARNING"}, f"Could not fully restore {problem}")
        self.report(
            {"INFO"},
            f"Exported frames {plan.start}-{plan.end} ({len(plan.frames)}) to {filepath}")
        return {"FINISHED"}


def plan_vdb_export(scene, frame_start=0, frame_end=0):
    """Export the verified bake without touching a live solver or the timeline."""
    from ..io import cache_vdb_export

    if production_bake.bake_is_active(scene):
        raise ExportRefused("A production bake is running. Wait for it or cancel it.")
    frames = available_frames(scene)
    start = int(frame_start) or frames[0]
    end = int(frame_end) or frames[-1]
    if start > end or start < frames[0] or end > frames[-1]:
        raise ExportRefused("The VDB export range is outside the Storage & Range cached frames.")
    base = bpy.path.abspath(scene.fluidsim.cache_path)
    try:
        return cache_vdb_export.plan_export(base, frame_start=start, frame_end=end)
    except cache_vdb_export.LegacyOriginRequired as exc:
        from .scene_access import active_domain_object
        from .solver_settings import build_settings

        if active_domain_object(scene) is None:
            raise ExportRefused(str(exc)) from exc
        for layer in scene.view_layers:
            layer.update()
        try:
            return cache_vdb_export.plan_export(base, build_settings(scene),
                frame_start=start, frame_end=end)
        except (ValueError, OSError) as legacy_exc:
            raise ExportRefused(str(legacy_exc)) from legacy_exc
    except (ValueError, OSError) as exc:
        raise ExportRefused(str(exc)) from exc


class FLUIDSIM_OT_export_vdb(bpy.types.Operator):
    """Export a baked liquid level set to a new OpenVDB sequence folder"""

    bl_idname = "fluidsim.export_vdb"
    bl_label = "Export Liquid Level Set as VDB"
    bl_options = {"REGISTER"}

    directory: StringProperty(name="Parent Folder", subtype="DIR_PATH")
    sequence_name: StringProperty(name="Sequence Folder", default="fluidsim_vdb",
        description="New folder created under the selected parent; existing folders are never replaced")
    frame_start: IntProperty(name="Start", default=0, min=0,
        description="First exported frame; 0 uses Storage & Range")
    frame_end: IntProperty(name="End", default=0, min=0,
        description="Last exported frame; 0 uses Storage & Range")

    @classmethod
    def poll(cls, context):
        return getattr(context, "scene", None) is not None

    def _plan(self, context):
        try:
            return plan_vdb_export(context.scene, self.frame_start, self.frame_end)
        except (ExportRefused, ValueError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return None

    def invoke(self, context, event):
        if self._plan(context) is None:
            return {"CANCELLED"}
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ..io.cache_vdb_export import export_sequence

        name = self.sequence_name.strip()
        if not self.directory or not name or name in {".", ".."} or any(
                c in name for c in ("/", "\\", ":")):
            self.report({"ERROR"}, "Choose a parent folder and a single new sequence folder name.")
            return {"CANCELLED"}
        plan = self._plan(context)
        if plan is None:
            return {"CANCELLED"}
        destination = Path(bpy.path.abspath(self.directory)) / name
        try:
            paths = export_sequence(plan, destination)
        except Exception as exc:  # noqa: BLE001 - present file/validation failures
            self.report({"ERROR"}, f"VDB export failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {len(paths)} liquid level-set frames to {destination / 'frames'}")
        return {"FINISHED"}


CLASSES = (FLUIDSIM_OT_export_alembic, FLUIDSIM_OT_export_vdb)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    from ..unregister_guard import unregister_classes

    unregister_classes(CLASSES)
