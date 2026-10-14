"""Viewport-header surfaces for the FluidSim workflow.

A "FluidSim" popover in the 3D View header gives the live loop (Play / Pause /
Restart, Quality, the physics an artist tunes while watching) without opening the
sidebar, and a button that opens a dedicated FluidSim workspace.

The popover draws the same shared functions as the sidebar (``panels.draw_transport``
and ``ui_stages.draw_simulate_controls``), so it can never drift from the stages.
The workspace is the user's own copy of the current layout with the sidebar open
and a timeline guaranteed; it is saved with the .blend like any other workspace,
and the add-on never deletes it on unregister.
"""

from __future__ import annotations

import bpy

from ..panels import draw_pending_apply, draw_transport, runtime_ui_state
from .scene_access import active_domain_settings
from .ui_common import activity_ui_state, draw_status_message
from .ui_stages import (
    draw_guided_setup_buttons,
    draw_quality,
    draw_simulate_controls,
)

WORKSPACE_NAME = "FluidSim"
# Editors that never serve the viewport + timeline layout a simulation needs.
_KEEP_AREA_TYPES = frozenset({"VIEW_3D", "PROPERTIES"})


def _screens(workspace):
    return tuple(getattr(workspace, "screens", ()))


def _spaces_of(area, space_type):
    return [space for space in area.spaces if space.type == space_type]


def prepare_workspace(workspace) -> None:
    """Shape a duplicated workspace for simulation work.

    Guarantees a 3D Viewport with its sidebar open and a timeline, and leaves the
    mode at Object so opening the workspace never switches the user into Edit or
    Sculpt. Blender gives an add-on no way to pick the active sidebar tab, so the
    FluidSim tab is whichever one the copied viewport already showed.
    """
    workspace.object_mode = "OBJECT"
    for screen in _screens(workspace):
        areas = list(screen.areas)
        if not any(area.type == "VIEW_3D" for area in areas):
            # The largest editor becomes the viewport.
            max(areas, key=lambda area: area.width * area.height).type = "VIEW_3D"
            areas = list(screen.areas)
        for area in areas:
            if area.type == "VIEW_3D":
                for space in _spaces_of(area, "VIEW_3D"):
                    space.show_region_ui = True
        if not any(area.type == "DOPESHEET_EDITOR" and area.ui_type == "TIMELINE"
                   for area in areas):
            spare = [area for area in areas if area.type not in _KEEP_AREA_TYPES]
            if spare:
                timeline = min(spare, key=lambda area: area.width * area.height)
                timeline.type = "DOPESHEET_EDITOR"
                timeline.ui_type = "TIMELINE"


def _draw_deck_toggle(layout, context):
    """The Liquid Deck show/hide toggle and wheel stage; absent when the Deck is Off."""
    from .deck import deck_mode

    settings = getattr(getattr(context, "scene", None), "fluidsim", None)
    if settings is None or deck_mode(context) == "OFF":
        return
    row = layout.row(align=True)
    row.prop(settings, "deck_shown", text="", icon="WINDOW")
    if settings.deck_shown:
        row.prop(settings, "deck_stage", text="")


class FLUIDSIM_OT_open_workspace(bpy.types.Operator):
    """Switch to the FluidSim workspace, creating it from the current layout the first time."""

    bl_idname = "fluidsim.open_workspace"
    bl_label = "Open FluidSim Workspace"
    bl_description = (
        "Switch to the FluidSim workspace: a 3D Viewport with the sidebar open and a "
        "timeline. Created from the current layout the first time"
    )

    @classmethod
    def poll(cls, context):
        return getattr(context, "window", None) is not None

    def execute(self, context):
        window = context.window
        existing = bpy.data.workspaces.get(WORKSPACE_NAME)
        if existing is not None:
            window.workspace = existing
            return {"FINISHED"}

        before = {workspace.name for workspace in bpy.data.workspaces}
        if bpy.ops.workspace.duplicate() != {"FINISHED"}:
            self.report({"WARNING"}, "Could not create the FluidSim workspace")
            return {"CANCELLED"}
        created = [w for w in bpy.data.workspaces if w.name not in before]
        if not created:
            self.report({"WARNING"}, "Could not create the FluidSim workspace")
            return {"CANCELLED"}
        workspace = created[0]
        workspace.name = WORKSPACE_NAME
        prepare_workspace(workspace)
        window.workspace = workspace
        self.report(
            {"INFO"},
            "FluidSim workspace ready; open the FluidSim tab in the viewport sidebar",
        )
        return {"FINISHED"}


class FLUIDSIM_PT_header_popover(bpy.types.Panel):
    """Quick controls for the live loop, opened from the 3D View header."""

    bl_label = "FluidSim"
    bl_idname = "FLUIDSIM_PT_header_popover"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 18

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return scene is not None and getattr(scene, "fluidsim", None) is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene

        state = activity_ui_state(scene)
        runtime_ready, runtime_message = runtime_ui_state()
        layout.label(text=state["label"], icon=state["icon"])
        if not runtime_ready:
            draw_status_message(layout, runtime_message, "ERROR")

        domain = active_domain_settings(scene)
        if domain is None:
            box = layout.box()
            box.label(text="No liquid domain yet", icon="INFO")
            draw_guided_setup_buttons(box)
        else:
            draw_transport(layout, context, state, runtime_ready)
            draw_pending_apply(layout, scene)

            layout.separator()
            draw_quality(layout, scene)
            draw_simulate_controls(layout, scene, domain)

        workspace = getattr(context, "workspace", None)
        if getattr(workspace, "name", "") != WORKSPACE_NAME:
            layout.separator()
            layout.operator(
                "fluidsim.open_workspace", text="FluidSim Workspace", icon="WORKSPACE"
            )


def draw_header_button(self, context):
    """Append the FluidSim popover to the 3D View header."""
    self.layout.popover(
        panel=FLUIDSIM_PT_header_popover.bl_idname, text="FluidSim", icon="MOD_FLUIDSIM"
    )
    _draw_deck_toggle(self.layout, context)


CLASSES = (
    FLUIDSIM_OT_open_workspace,
    FLUIDSIM_PT_header_popover,
)


def _header_type():
    return getattr(bpy.types, "VIEW3D_HT_header", None)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    header = _header_type()
    if header is not None:
        header.append(draw_header_button)


def unregister():
    header = _header_type()
    if header is not None:
        try:
            header.remove(draw_header_button)
        except (ValueError, RuntimeError):
            pass
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:  # noqa: BLE001 - report and keep walking
            print(f"FluidSim: could not unregister {cls.__name__}: {exc!r}")
