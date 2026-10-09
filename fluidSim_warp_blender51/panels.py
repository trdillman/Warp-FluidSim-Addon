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


PLAYBACK_ONLY_TEXT = "Playback only — CUDA unavailable"


def playback_only() -> bool:
    """True when Warp/CUDA cannot run: caches still load, play and export."""
    probe = probe_runtime()
    return probe is not None and not probe.available


def runtime_ui_state():
    """``(ready, message)`` for the configured CUDA device.

    Without a usable Warp/CUDA runtime the message is the fixed playback-only
    notice; the probe's own reason is drawn beneath it by the status strip.
    """
    probe = probe_runtime()
    if probe is not None and not probe.available:
        return False, PLAYBACK_ONLY_TEXT
    return runtime_status_for_device(configured_device(), probe=probe)


def draw_transport(layout, context, state, runtime_ready):
    """Preview / Pause / Resume / Cancel for the current activity, plus Restart.

    Shared by the sidebar strip and the viewport-header popover so both surfaces
    follow the same activity state machine.
    """
    scene = context.scene
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


def draw_pending_apply(layout, scene):
    """Offer Apply Changes only while an edit is waiting for a reset or re-init."""
    pending = pending_ui_state(scene)
    if pending["action"] != "NONE":
        box = layout.box()
        box.label(text=pending["summary"], icon="INFO")
        box.operator("fluidsim.apply_changes", text="Apply Changes", icon="CHECKMARK")


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

        runtime_ready, runtime_message = runtime_ui_state()
        runtime = layout.box()
        row = runtime.row(align=True)
        row.label(text=runtime_message, icon="CHECKMARK" if runtime_ready else "ERROR")
        row.operator("fluidsim.refresh_runtime_probe", text="", icon="FILE_REFRESH")
        if playback_only():
            runtime.label(text=probe_runtime().message)

        draw_transport(layout, context, state, runtime_ready)
        draw_pending_apply(layout, scene)


CLASSES = (
    FLUIDSIM_OT_open_scene_properties,
    FLUIDSIM_PT_viewport_status,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    # One class that will not unregister must not leave every later class registered.
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:  # noqa: BLE001 - report and keep walking
            print(f"FluidSim: could not unregister {cls.__name__}: {exc!r}")
