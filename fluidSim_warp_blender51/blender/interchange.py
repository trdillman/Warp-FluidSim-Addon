"""Interchange export from a baked cache: Alembic mesh sequence of the liquid surface.

The operator is cache-driven and refuses export while a live solver exists.
Pausing Live does not release that session; exporting must never release it.
Blender's own Alembic exporter walks the frame range and evaluates the scene per frame; each
evaluation fires ``frame_change_post``, which is exactly how cache playback
publishes a frame (``playback.handle_frame_change`` -> ``play_cache_frame`` ->
``output_publication.update_outputs_from_cache_frame``). So the export is:
cache playback on, live on (the frame handler returns early otherwise), only
the surface object selected, then ``bpy.ops.wm.alembic_export`` over the range.
Dynamic colliders (4.0 two-way rigid coupling) ride along: their cached poses
are keyed first, then they are selected with the surface (``ExportPlan.carried``).
Every piece of scene state the export changes is restored in a ``finally``.

The export must not run inside ``playback.suppress_frame_handlers``: that would
block the very handler that publishes each frame.

USD (``export_usd``) rides the same playback, with three additions measured on
Blender 5.2 (4.0 plan section 7): Alembic writes an empty transform for a
PointCloud, while USD writes ``UsdGeom.Points`` with widths, velocities and a
primvar per custom attribute. But the USD exporter writes per-frame time
samples for handler-published geometry only when (a) the handler rewrites the
same datablock in place and (b) the object carries a time-dependent modifier;
a replaced datablock becomes a separate prim. So the export

* reserves each point cloud's capacity for the range's largest frame (from the
  cache manifest) and holds shrinking off, so no cloud is replaced mid-range;
* adds a temporary Geometry Nodes modifier to every exported object: Scene
  Time x 0 as a Set Position offset (makes the object time-dependent) and, on
  point clouds, a Delete Geometry of zero-radius points. Publication pads
  clouds to capacity with zero-radius points; trimming them keeps the file to
  the live points (every published live point has a positive radius);
* walks the scene frame range, which it sets to the export range (the 5.2
  exporter has no range arguments), and restores it afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bpy
from bpy.props import IntProperty, StringProperty

from ..io import cache, cache_schema
from . import output_publication, playback, production_bake, rigid_publication

SURFACE_KIND = output_publication.SURFACE_NAME


class ExportRefused(Exception):
    """The export cannot start; the message is shown to the user as-is."""


@dataclass(frozen=True)
class ExportPlan:
    """What an export would write, resolved and validated before any state change."""

    base_dir: str
    frames: tuple[int, ...]
    surface: object
    # Dynamic colliders whose solved poses the cache holds (cache v3.1).
    carried: tuple = ()

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


def _require_export_idle(scene):
    """Refuse before changing modes: cache playback tears down the singleton."""
    from . import scene_bridge

    if production_bake.bake_is_active(scene):
        raise ExportRefused("A production bake is running. Wait for it or cancel it.")
    sess = scene_bridge.session()
    if sess.active:
        owner = "This scene" if sess.owns_scene(scene) else "Another scene"
        raise ExportRefused(
            f"{owner} has an active live solver session (even if Live is paused). "
            "Alembic/USD export would discard its particle and rigid-body state. "
            "Export from a separate Blender instance, or explicitly switch the "
            "owning scene to Cache Playback first if you intend to discard "
            "that live state.")


def _export_frames(scene, frame_start: int, frame_end: int) -> tuple[int, ...]:
    """Cached frames inside the requested range; ``0`` bounds mean the whole cache."""
    _require_export_idle(scene)
    frames = available_frames(scene)
    start = int(frame_start) or frames[0]
    end = int(frame_end) or frames[-1]
    if start > end:
        raise ExportRefused(f"The export range {start}-{end} is empty.")
    if start < frames[0] or end > frames[-1]:
        raise ExportRefused(
            f"The export range {start}-{end} is outside the cached frames "
            f"{frames[0]}-{frames[-1]}.")
    return tuple(f for f in frames if start <= f <= end)


def plan_export(scene, frame_start: int = 0, frame_end: int = 0) -> ExportPlan:
    """Validate everything the export needs; ``0`` bounds mean the whole cache."""
    frames = _export_frames(scene, frame_start, frame_end)
    surface = output_publication.output_object(scene, SURFACE_KIND)
    if surface is None:
        raise ExportRefused(
            "There is no FluidSim surface object. Enable the surface output and "
            "play a cached frame first.")
    base_dir = bpy.path.abspath(scene.fluidsim.cache_path)
    return ExportPlan(
        base_dir=base_dir,
        frames=frames,
        surface=surface,
        carried=rigid_publication.carried_objects(scene, base_dir, frames),
    )


@dataclass(frozen=True)
class UsdExportPlan:
    """The outputs a USD export writes, with each point output's peak count."""

    base_dir: str
    frames: tuple[int, ...]
    surface: object | None
    points: tuple[tuple[object, int], ...]

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def objects(self) -> tuple[object, ...]:
        head = () if self.surface is None else (self.surface,)
        return head + tuple(obj for obj, _peak in self.points)


def _range_point_counts(base_dir: str, frames) -> tuple[int, int]:
    """Largest particle and whitewater counts the manifest records in ``frames``."""
    manifest = cache_schema.load_manifest(cache_schema.manifest_path(base_dir))
    entries = [manifest.frames[frame] for frame in frames]
    return (max((int(e.particle_count) for e in entries), default=0),
            max((int(e.whitewater_count) for e in entries), default=0))


def plan_usd_export(scene, frame_start: int = 0, frame_end: int = 0) -> UsdExportPlan:
    """Resolve the range and the outputs to write: the surface (when it renders)
    and every rendering point output with points somewhere in the range.

    Each typed whitewater output (foam, bubbles, spray) is sized for the
    combined whitewater count, an upper bound on its own.
    """
    frames = _export_frames(scene, frame_start, frame_end)
    base_dir = bpy.path.abspath(scene.fluidsim.cache_path)
    particles, whitewater = _range_point_counts(base_dir, frames)
    surface = output_publication.output_object(scene, SURFACE_KIND)
    if surface is not None and (surface.type != "MESH" or surface.hide_render):
        surface = None
    points = []
    for kind, peak in (
            (output_publication.PARTICLES_NAME, particles),
            *((name, whitewater)
              for name in output_publication.WHITEWATER_OUTPUT_NAMES)):
        obj = output_publication.output_object(scene, kind)
        if obj is None or obj.hide_render or peak <= 0:
            continue
        points.append((obj, peak))
    if surface is None and not points:
        raise ExportRefused(
            "There are no FluidSim outputs to export. Enable the surface, particle "
            "or whitewater output and play a cached frame first.")
    return UsdExportPlan(
        base_dir=base_dir, frames=frames, surface=surface, points=tuple(points))


def _set_frame_range(scene, start: int, end: int):
    # Assigning a start past the current end drags the end along (and the
    # reverse), so the start is written again once the end is in place.
    scene.frame_start = int(start)
    scene.frame_end = int(end)
    scene.frame_start = int(start)


class _SceneState:
    """Everything the export touches, captured up front and restored on exit."""

    def __init__(self, scene, objects):
        # Recheck after the file picker / for direct callers with a stale plan.
        # Outside the export try/finally: a refusal must not run restore(),
        # whose playback transition could otherwise initialize a new solver.
        _require_export_idle(scene)
        self.scene = scene
        self.objects = tuple(objects)
        fs = scene.fluidsim
        self.live = bool(fs.live)
        self.cache_playback = bool(getattr(fs, "cache_playback_enabled", False))
        self.frame = int(scene.frame_current)
        self.frame_range = (int(scene.frame_start), int(scene.frame_end))
        self.view_layer = bpy.context.view_layer
        self.selected = [o for o in self.view_layer.objects if o.select_get()]
        self.active = self.view_layer.objects.active
        self.hidden = [(obj, bool(obj.hide_viewport), bool(obj.hide_get()))
                       for obj in self.objects]

    def enter(self, plan):
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

        for obj in self.view_layer.objects:
            obj.select_set(False)
        for obj in self.objects:
            obj.hide_viewport = False
            obj.hide_set(False)
            obj.select_set(True)
        self.view_layer.objects.active = self.objects[0]
        carried = getattr(plan, "carried", ())
        if carried:
            # Key every cached pose before the exporter samples transforms.
            rigid_publication.publish_cache_frames(
                scene, plan.base_dir, plan.frames)
            for obj in carried:
                if obj.visible_get(view_layer=self.view_layer):
                    obj.select_set(True)

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
            for obj in self.view_layer.objects:
                obj.select_set(False)
            for obj in self.selected:
                try:
                    obj.select_set(True)
                except (ReferenceError, RuntimeError):
                    pass
            self.view_layer.objects.active = self.active
            for obj, hide_viewport, hide_get in self.hidden:
                obj.hide_viewport = hide_viewport
                obj.hide_set(hide_get)

        def restore_frame():
            if (int(scene.frame_start), int(scene.frame_end)) != self.frame_range:
                _set_frame_range(scene, *self.frame_range)
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


def _normalized_path(filepath: str, suffixes=(".abc",)) -> str:
    """Absolute output path; an unknown suffix becomes the first of ``suffixes``."""
    path = Path(bpy.path.abspath(filepath))
    if path.suffix.lower() not in suffixes:
        path = path.with_suffix(suffixes[0])
    return str(path)


def export_alembic(scene, filepath: str, plan: ExportPlan) -> list[str]:
    """Run the exporter under cache playback; returns restore problems (if any)."""
    state = _SceneState(scene, (plan.surface,))
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


USD_SUFFIXES = (".usd", ".usda", ".usdc")
SAMPLER_NAME = "FluidSim USD Sampler"


def _sampler_group(name: str, trim_padding: bool):
    """A Geometry Nodes group that changes nothing but depends on time.

    Scene Time x 0 drives a Set Position offset, so the USD exporter samples
    the object every frame. With ``trim_padding`` it also deletes zero-radius
    points: the padding publication keeps beyond the live count.
    """
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    group.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    geometry = nodes.new("NodeGroupInput").outputs[0]
    if trim_padding:
        radius = nodes.new("GeometryNodeInputRadius")
        padding = nodes.new("FunctionNodeCompare")
        padding.data_type = "FLOAT"
        padding.operation = "LESS_EQUAL"
        padding.inputs[1].default_value = 0.0
        links.new(radius.outputs[0], padding.inputs[0])
        delete = nodes.new("GeometryNodeDeleteGeometry")
        delete.domain = "POINT"
        links.new(geometry, delete.inputs["Geometry"])
        links.new(padding.outputs[0], delete.inputs["Selection"])
        geometry = delete.outputs[0]
    time = nodes.new("GeometryNodeInputSceneTime")
    zero = nodes.new("ShaderNodeMath")
    zero.operation = "MULTIPLY"
    zero.inputs[1].default_value = 0.0
    offset = nodes.new("ShaderNodeCombineXYZ")
    move = nodes.new("GeometryNodeSetPosition")
    links.new(time.outputs["Frame"], zero.inputs[0])
    links.new(zero.outputs[0], offset.inputs[0])
    links.new(geometry, move.inputs["Geometry"])
    links.new(offset.outputs[0], move.inputs["Offset"])
    links.new(move.outputs[0], nodes.new("NodeGroupOutput").inputs[0])
    return group


class _UsdSamplers:
    """The temporary modifiers and node groups, removed by :meth:`remove`.

    Point outputs export their cached points: the existing modifiers on a
    point cloud (the add-on's render group realizes an ico-sphere per point)
    are muted for render and restored by :meth:`remove`.
    """

    def __init__(self):
        self.groups = []
        self.modifiers = []
        self.muted = []

    def attach(self, objects):
        groups = {}
        for obj in objects:
            trim = obj.type == "POINTCLOUD"
            if trim:
                for existing in obj.modifiers:
                    if existing.show_render:
                        self.muted.append((existing, True))
                        existing.show_render = False
            if trim not in groups:
                groups[trim] = _sampler_group(SAMPLER_NAME, trim)
                self.groups.append(groups[trim])
            modifier = obj.modifiers.new(SAMPLER_NAME, "NODES")
            self.modifiers.append((obj, modifier))
            modifier.node_group = groups[trim]

    def remove(self) -> list[str]:
        problems = []
        for obj, modifier in reversed(self.modifiers):
            try:
                obj.modifiers.remove(modifier)
            except Exception as exc:  # noqa: BLE001 - keep removing the rest
                problems.append(f"modifier on {obj.name}: {exc}")
        self.modifiers.clear()
        for modifier, shown in reversed(self.muted):
            try:
                modifier.show_render = shown
            except Exception as exc:  # noqa: BLE001
                problems.append(f"modifier {modifier.name}: {exc}")
        self.muted.clear()
        for group in self.groups:
            try:
                bpy.data.node_groups.remove(group)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"node group: {exc}")
        self.groups.clear()
        return problems


def export_usd(scene, filepath: str, plan: UsdExportPlan) -> list[str]:
    """Write the plan's outputs as one animated USD stage; returns restore problems.

    Raises when a point cloud was replaced mid-range anyway (its points would
    be split across prims instead of sampled per frame).
    """
    state = _SceneState(scene, plan.objects)
    samplers = _UsdSamplers()
    try:
        state.enter(plan)
        with output_publication.hold_point_capacity():
            for obj, peak in plan.points:
                output_publication.reserve_point_capacity(obj, peak)
            # Reservation leaves placeholders: republish the first frame.
            playback.clear_cache_cursors()
            samplers.attach(plan.objects)
            datablocks = [(obj, obj.data.as_pointer()) for obj in plan.objects]
            _set_frame_range(scene, plan.start, plan.end)
            result = bpy.ops.wm.usd_export(
                filepath=filepath,
                selected_objects_only=True,
                export_animation=True,
                evaluation_mode="RENDER",
                export_materials=False,
                export_uvmaps=False,
                export_custom_properties=False,
            )
            if "FINISHED" not in result:
                raise RuntimeError("Blender's USD exporter did not finish.")
            replaced = [obj.name for obj, pointer in datablocks
                        if obj.data.as_pointer() != pointer]
            if replaced:
                raise RuntimeError(
                    f"{', '.join(replaced)} changed datablock during the export, "
                    "so the file has no per-frame samples for it.")
    finally:
        problems = samplers.remove()
        problems += state.restore()
    return problems


class _CacheRangeExport:
    """File-select operator body shared by the cache-driven exporters.

    Subclasses set ``format_name``, ``default_name``, ``suffixes`` and the
    ``planner``/``exporter`` callables.
    """

    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH", default="")
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
            return self.planner(context.scene, self.frame_start, self.frame_end)
        except ExportRefused as exc:
            self.report({"ERROR"}, str(exc))
            return None

    def invoke(self, context, event):
        if self._plan(context) is None:
            return {"CANCELLED"}
        if not self.filepath:
            self.filepath = self.default_name
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
        filepath = _normalized_path(self.filepath, self.suffixes)
        try:
            problems = self.exporter(scene, filepath, plan)
        except ExportRefused as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:  # noqa: BLE001 - surface any exporter failure
            self.report({"ERROR"}, f"{self.format_name} export failed: {exc}")
            return {"CANCELLED"}
        for problem in problems:
            self.report({"WARNING"}, f"Could not fully restore {problem}")
        self.report(
            {"INFO"},
            f"Exported frames {plan.start}-{plan.end} ({len(plan.frames)}) to {filepath}")
        return {"FINISHED"}


class FLUIDSIM_OT_export_alembic(_CacheRangeExport, bpy.types.Operator):
    """Export the baked liquid surface as an Alembic mesh sequence"""

    bl_idname = "fluidsim.export_alembic"
    bl_label = "Export Surface as Alembic"

    filter_glob: StringProperty(default="*.abc", options={"HIDDEN"})

    format_name = "Alembic"
    default_name = "fluidsim_surface.abc"
    suffixes = (".abc",)
    planner = staticmethod(plan_export)
    exporter = staticmethod(export_alembic)


class FLUIDSIM_OT_export_usd(_CacheRangeExport, bpy.types.Operator):
    """Export the baked liquid surface, particles and whitewater as animated USD, with their attributes"""

    bl_idname = "fluidsim.export_usd"
    bl_label = "Export Liquid as USD"

    filter_glob: StringProperty(default="*.usd;*.usda;*.usdc", options={"HIDDEN"})

    format_name = "USD"
    default_name = "fluidsim_liquid.usd"
    suffixes = USD_SUFFIXES
    planner = staticmethod(plan_usd_export)
    exporter = staticmethod(export_usd)


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


CLASSES = (FLUIDSIM_OT_export_alembic, FLUIDSIM_OT_export_usd, FLUIDSIM_OT_export_vdb)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    from ..unregister_guard import unregister_classes

    unregister_classes(CLASSES)
