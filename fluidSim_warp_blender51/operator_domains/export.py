"""Capability-owned Blender operators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from . import common as _runtime

_FLUIDSIM_OT_session_base = _runtime._FLUIDSIM_OT_session_base


class FLUIDSIM_OT_export_frame(_FLUIDSIM_OT_session_base, bpy.types.Operator):
    """Export the current simulation frame (surface + particles) to disk"""

    bl_idname = "fluidsim.export_frame"
    bl_label = "Export Current Frame"

    filepath: StringProperty(
        subtype="FILE_PATH", default="//fluidsim_frame",
        description="Set the output path prefix for the current surface and particle export; changing it writes the exported frame to a different location")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        bridge = _runtime._bridge()
        sess = bridge.session()
        if not sess.active:
            self.report({"ERROR"}, "Solver not initialized")
            return {"CANCELLED"}
        from ..io import export as export_mod

        scene = context.scene
        base = bpy.path.abspath(self.filepath)
        try:
            paths = export_mod.export_frame(sess.solver, base)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Frame export failed: {exc}")
            bridge.set_error(scene, f"Frame export failed: {exc}")
            scene.fluidsim.status = f"Frame export failed: {exc}"
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported: {', '.join(paths)}")
        return {"FINISHED"}


class FLUIDSIM_OT_export_cache_sequence(bpy.types.Operator):
    """Bake the selected timeline range to FluidSim cache frames"""

    bl_idname = "fluidsim.export_cache_sequence"
    bl_label = "Export Cache Sequence"

    @classmethod
    def poll(cls, context):
        return _runtime._runtime_poll(cls)

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim
        bridge = _runtime._bridge()
        from ..io import cache

        try:
            start, end = cache.resolve_frame_range(
                scene,
                use_custom=fs.cache_use_custom_range,
                frame_start=fs.cache_frame_start,
                frame_end=fs.cache_frame_end,
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            fs.status = f"Cache export failed: {exc}"
            return {"CANCELLED"}

        try:
            from ..blender import cache_bake

            base = cache_bake.bake(scene, bridge.session(), start, end)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Cache sequence export failed: {exc}")
            fs.status = f"Cache export failed: {exc}"
            return {"CANCELLED"}

        count = end - start + 1
        fs.status = f"Exported {count} cache frame(s) to {base}"
        self.report({"INFO"}, fs.status)
        return {"FINISHED"}
