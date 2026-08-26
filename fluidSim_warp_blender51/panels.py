"""Compact 3D Viewport launcher for the FluidSim Properties workflow."""

from __future__ import annotations

import bpy

from .blender.ui_common import FluidSimView3DPanel, activity_ui_state, pending_ui_state
from .preferences import configured_device
from .runtime.warp_runtime import probe_runtime, runtime_status_for_device


class FLUIDSIM_OT_open_scene_properties(bpy.types.Operator):
    """Focus an existing Properties editor on Scene Properties."""

    bl_idname = "fluidsim.open_scene_properties"
    bl_label = "Open FluidSim Scene Properties"
    bl_description = "Focus an existing Properties editor on Scene Properties"

    def execute(self, context):
        screen = getattr(context, "screen", None)
        areas = getattr(screen, "areas", ()) if screen is not None else ()
        for area in areas:
            if area.type == "PROPERTIES":
                area.spaces.active.context = "SCENE"
                self.report({"INFO"}, "FluidSim controls are in Scene Properties")
                return {"FINISHED"}
        self.report(
            {"WARNING"},
            "No Properties editor is open; change any editor to Properties, then choose Scene",
        )
        return {"CANCELLED"}


class FLUIDSIM_PT_viewport_status(FluidSimView3DPanel, bpy.types.Panel):
    """Status and lifecycle shortcuts; authored settings live in Properties."""

    bl_label = "FluidSim"
    bl_idname = "FLUIDSIM_PT_viewport_status"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        state = activity_ui_state(scene)

        status = layout.box()
        status.label(text=state["label"], icon=state["icon"])
        status.label(
            text="This scene owns the solver" if state["owns_scene"] else "Solver not owned by this scene"
        )

        probe = probe_runtime()
        selected = configured_device()
        runtime_ready, runtime_message = runtime_status_for_device(selected, probe=probe)
        runtime = layout.box()
        row = runtime.row(align=True)
        row.label(text=runtime_message, icon="CHECKMARK" if runtime_ready else "ERROR")
        row.operator("fluidsim.refresh_runtime_probe", text="", icon="FILE_REFRESH")

        controls = layout.column(align=True)
        activity = state["id"]
        if activity == "LIVE_PREVIEW":
            controls.operator("fluidsim.toggle_live", text="Stop Live Preview", icon="PAUSE")
        elif activity == "CACHE_PLAYBACK":
            controls.operator(
                "fluidsim.cache_playback_stop", text="Stop Cache Playback", icon="PAUSE"
            )
        elif activity == "PRODUCTION_BAKE":
            controls.operator("fluidsim.bake_cancel", text="Cancel Production Bake", icon="CANCEL")
        else:
            live = controls.column(align=True)
            live.enabled = runtime_ready
            live.operator("fluidsim.toggle_live", text="Start Live Preview", icon="PLAY")

        if state["owns_scene"] and activity not in {"CACHE_PLAYBACK", "PRODUCTION_BAKE"}:
            row = controls.row(align=True)
            row.operator_context = "EXEC_DEFAULT"
            row.operator("fluidsim.step", text="Step One Frame", icon="FRAME_NEXT").frames = 1
            row.operator("fluidsim.release", text="Release Solver", icon="X")
        elif not state["owns_scene"] and activity == "IDLE":
            initialize = controls.column(align=True)
            initialize.enabled = runtime_ready
            initialize.operator("fluidsim.initialize", text="Initialize Solver", icon="MOD_FLUIDSIM")

        pending = pending_ui_state(scene)
        if pending["action"] != "NONE":
            box = layout.box()
            box.label(text=pending["summary"], icon="INFO")
            box.label(text="Review and apply staged values in Scene Properties.")

        layout.separator()
        layout.operator(
            "fluidsim.open_scene_properties",
            text="Open FluidSim Scene Properties",
            icon="PROPERTIES",
        )
        layout.label(text="Object roles are configured in Object Properties.", icon="OBJECT_DATA")


CLASSES = (
    FLUIDSIM_OT_open_scene_properties,
    FLUIDSIM_PT_viewport_status,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass
