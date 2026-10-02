"""Pinned status/transport strip for the FluidSim 3D View sidebar.

The workflow stages (``blender/ui_stages.py``) sit below it; settings and object
roles are authored there and in Properties.
"""

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


def _status_note(state) -> str:
    """One plain-language line about the solver session; empty when nothing to add."""
    from .blender.scene_bridge import session

    if state["id"] in {"PRODUCTION_BAKE", "CACHE_PLAYBACK"}:
        return ""
    if state["owns_scene"]:
        return "Simulation loaded; Resume Preview continues it."
    if session().active:
        return "Another scene is using the solver."
    return "No simulation loaded; Preview starts one."


class FLUIDSIM_PT_viewport_status(FluidSimView3DPanel, bpy.types.Panel):
    """Pinned status and transport strip above the workflow stages."""

    bl_label = "FluidSim"
    bl_idname = "FLUIDSIM_PT_viewport_status"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        state = activity_ui_state(scene)

        status = layout.box()
        status.label(text=state["label"], icon=state["icon"])
        note = _status_note(state)
        if note:
            status.label(text=note)

        probe = probe_runtime()
        selected = configured_device()
        runtime_ready, runtime_message = runtime_status_for_device(selected, probe=probe)
        runtime = layout.box()
        row = runtime.row(align=True)
        row.label(text=runtime_message, icon="CHECKMARK" if runtime_ready else "ERROR")
        row.operator("fluidsim.refresh_runtime_probe", text="", icon="FILE_REFRESH")

        controls = layout.column(align=True)
        activity = state["id"]
        screen = getattr(context, "screen", None)
        playing = bool(
            getattr(scene.fluidsim, "live", False)
            and screen is not None
            and getattr(screen, "is_animation_playing", False)
        )
        if activity == "PRODUCTION_BAKE":
            controls.operator("fluidsim.bake_cancel", text="Cancel Production Bake", icon="CANCEL")
        elif activity == "CACHE_PLAYBACK":
            controls.operator(
                "fluidsim.preview_pause" if playing else "fluidsim.cache_preview_start",
                text="Pause Cache Preview" if playing else "Resume Cache Preview",
                icon="PAUSE" if playing else "PLAY",
            )
        elif playing:
            controls.operator("fluidsim.preview_pause", text="Pause Preview", icon="PAUSE")
        else:
            live = controls.column(align=True)
            live.enabled = runtime_ready
            live.operator(
                "fluidsim.preview_start",
                text="Resume Preview" if state["owns_scene"] else "Preview",
                icon="PLAY",
            )

        if state["owns_scene"] and activity not in {"CACHE_PLAYBACK", "PRODUCTION_BAKE"}:
            controls.operator(
                "fluidsim.restart_simulation",
                text="Restart Simulation",
                icon="FILE_REFRESH",
            )

        pending = pending_ui_state(scene)
        if pending["action"] != "NONE":
            box = layout.box()
            box.label(text=pending["summary"], icon="INFO")
            box.operator("fluidsim.apply_changes", text="Apply Changes", icon="CHECKMARK")


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
