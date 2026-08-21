"""Full FluidSim workflow mirrored into the 3D View sidebar.

The authoritative draw implementations remain in the native Scene Properties
modules.  These panels deliberately delegate to those implementations so the
sidebar and Properties editor expose the same controls, conditions, lifecycle
rules, and Custom Effects workflow without maintaining duplicate UI code.
"""

from __future__ import annotations

import bpy

from . import effect_ui, effect_workspace, ui_scene
from .ui_common import FluidSimView3DPanel


class FLUIDSIM_VIEW3D_PT_full_settings(FluidSimView3DPanel, bpy.types.Panel):
    """Collapsed parent for the complete mirrored Scene workflow."""

    bl_label = "Full Settings"
    bl_idname = "FLUIDSIM_VIEW3D_PT_full_settings"
    bl_parent_id = "FLUIDSIM_PT_viewport_status"
    bl_order = 100
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, _context):
        self.layout.label(
            text="These controls edit the same Scene settings as Properties.",
            icon="PROPERTIES",
        )


def _mirror_id(panel_id: str) -> str:
    prefix = "FLUIDSIM_PT_"
    suffix = panel_id[len(prefix):] if panel_id.startswith(prefix) else panel_id
    return f"FLUIDSIM_VIEW3D_PT_{suffix}"


def _make_panel_mirror(source_panel):
    """Create one lightweight panel class that delegates to ``source_panel``."""

    def draw(self, context):
        return source_panel.draw(self, context)

    @classmethod
    def poll(cls, context):
        source_poll = getattr(source_panel, "poll", None)
        return bool(source_panel.poll(context)) if callable(source_poll) else True

    attributes = {
        "__module__": __name__,
        "__doc__": f"3D View mirror of {source_panel.__name__}.",
        "bl_label": str(getattr(source_panel, "bl_label", source_panel.__name__)),
        "bl_idname": _mirror_id(
            str(getattr(source_panel, "bl_idname", source_panel.__name__))
        ),
        "bl_space_type": FluidSimView3DPanel.bl_space_type,
        "bl_region_type": FluidSimView3DPanel.bl_region_type,
        "bl_category": FluidSimView3DPanel.bl_category,
        "bl_order": int(getattr(source_panel, "bl_order", 0)),
        "bl_options": set(getattr(source_panel, "bl_options", set())),
        "draw": draw,
        "poll": poll,
    }
    parent = str(getattr(source_panel, "bl_parent_id", "") or "")
    attributes["bl_parent_id"] = (
        _mirror_id(parent)
        if parent else FLUIDSIM_VIEW3D_PT_full_settings.bl_idname
    )
    class_name = _mirror_id(
        str(getattr(source_panel, "bl_idname", source_panel.__name__))
    )
    return type(
        class_name, (FluidSimView3DPanel, bpy.types.Panel), attributes
    )


VIEW3D_SOURCE_PANELS = (
    *ui_scene.SCENE_PANEL_CLASSES,
    effect_ui.FLUIDSIM_PT_custom_effects,
    effect_ui.FLUIDSIM_PT_effect_packages,
    effect_workspace.FLUIDSIM_PT_effect_warp_code,
)

VIEW3D_PANEL_CLASSES = (
    FLUIDSIM_VIEW3D_PT_full_settings,
    *tuple(
        _make_panel_mirror(source_panel)
        for source_panel in VIEW3D_SOURCE_PANELS
    ),
)


def register():
    for cls in VIEW3D_PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(VIEW3D_PANEL_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass
