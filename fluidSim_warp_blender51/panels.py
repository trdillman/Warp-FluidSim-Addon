"""Pinned status/transport strip for the FluidSim 3D View sidebar.

The workflow stages (``blender/ui_stages.py``) sit below it; settings and object
roles are authored there and in Properties.
"""

from __future__ import annotations

import bpy

from .blender.ui_common import FluidSimView3DPanel, activity_ui_state, pending_ui_state
from .preferences import configured_device
from .runtime.warp_runtime import probe_runtime, runtime_status_for_device


def _properties_target(context) -> str:
    """Properties tab that holds the FluidSim controls for the current selection.

    Domain settings live in the Physics tab, but only when the active object is
    the scene's designated domain; otherwise the scene-owned panels (Scene) are
    the useful target.
    """
    from .blender.scene_access import active_domain_object

    scene = getattr(context, "scene", None)
    active = getattr(context, "active_object", None)
    if scene is not None and active is not None:
        domain = active_domain_object(scene)
        if domain is not None and domain == active:
            return "PHYSICS"
    return "SCENE"


class FLUIDSIM_OT_open_scene_properties(bpy.types.Operator):
    """Focus an existing Properties editor on the FluidSim settings."""

    # The id predates the Physics-tab move; it stays so saved keymaps and UI resolve.
    bl_idname = "fluidsim.open_scene_properties"
    bl_label = "Open Full Settings"
    bl_description = (
        "Focus an existing Properties editor on the FluidSim settings: the Physics tab "
        "when the Fluid Domain is active, otherwise the Scene tab"
    )

    # Automatic picks the tab for the selection; the domain's Physics panel
    # asks for the Scene tab explicitly, since it is already in Physics.
    target: bpy.props.EnumProperty(
        name="Tab",
        description="Which Properties tab to open",
        items=(
            ("AUTO", "Automatic", "Physics for the active Fluid Domain, otherwise Scene"),
            ("SCENE", "Scene", "The scene-level FluidSim settings"),
            ("PHYSICS", "Physics", "The Fluid Domain settings"),
        ),
        default="AUTO",
        options={"SKIP_SAVE"},
    )

    def execute(self, context):
        target = str(getattr(self, "target", "AUTO") or "AUTO")
        if target not in {"SCENE", "PHYSICS"}:
            target = _properties_target(context)
        tab = "Physics" if target == "PHYSICS" else "Scene"
        screen = getattr(context, "screen", None)
        areas = getattr(screen, "areas", ()) if screen is not None else ()
        for area in areas:
            if area.type == "PROPERTIES":
                area.spaces.active.context = target
                self.report({"INFO"}, f"FluidSim controls are in {tab} Properties")
                return {"FINISHED"}
        self.report(
            {"WARNING"},
            f"No Properties editor is open; change any editor to Properties, then choose {tab}",
        )
        return {"CANCELLED"}


def _status_note(state) -> str:
    """One plain-language line about the solver session; empty when nothing to add."""
    from .blender.scene_bridge import session

    if state["id"] in {"PRODUCTION_BAKE", "CACHE_PLAYBACK"}:
        return ""
    if state["owns_scene"]:
        return "Simulation loaded; Play continues it."
    if session().active:
        return "Another scene is using the solver."
    return "No simulation loaded; Play starts one."


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
    """Play / Pause / Cancel for the current activity, plus Restart.

    Play is Blender's own timeline play: the playback handler arms and admits
    the simulation, so there is no separate Preview step to pass through.
    ``runtime_ready`` does not gate Play: cache playback and plain timeline
    playback need no GPU, and a live-admission failure reports in the status.
    Shared by the sidebar strip and the viewport-header popover so both surfaces
    follow the same activity state machine.
    """
    controls = layout.column(align=True)
    activity = state["id"]
    screen = getattr(context, "screen", None)
    playing = bool(screen is not None and getattr(screen, "is_animation_playing", False))
    if activity == "PRODUCTION_BAKE":
        controls.operator("fluidsim.bake_cancel", text="Cancel Production Bake", icon="CANCEL")
    elif playing:
        controls.operator("fluidsim.preview_pause", text="Pause", icon="PAUSE")
    else:
        controls.operator("screen.animation_play", text="Play", icon="PLAY")

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
    from .unregister_guard import register_classes

    register_classes(CLASSES)


def unregister():
    from .unregister_guard import unregister_classes

    # One class that will not unregister must not leave every later class registered.
    unregister_classes(CLASSES)
