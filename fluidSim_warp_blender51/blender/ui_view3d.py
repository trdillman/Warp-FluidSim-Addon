"""Lean FluidSim shelf for the 3D View sidebar.

The full authoring workflow lives in the native Scene/Object Properties editors;
re-hosting all of it in the sidebar produced a deep, redundant panel tree. This
module keeps only a shallow **Live Tuning** shelf under the compact viewport
status launcher: the handful of per-iteration knobs an artist adjusts while
watching a live preview, bound to the same datablock the solver reads
(``active_domain_settings``), plus a jump back to the authoritative Properties
workflow.
"""

from __future__ import annotations

import bpy

from .scene_bridge import active_domain_settings
from .ui_common import FluidSimView3DPanel, draw_property

# The per-iteration knobs an artist scrubs during a live preview. Kept
# deliberately short (D5); the full parameter set stays in Scene Properties.
LIVE_TUNING_PROPS = (
    "voxel_size",
    "particles_per_voxel",
    "flip_ratio",
    "particle_display_fraction",
)


class FLUIDSIM_VIEW3D_PT_live_tuning(FluidSimView3DPanel, bpy.types.Panel):
    """Shallow live-tuning shelf under the compact viewport status launcher."""

    bl_label = "Live Tuning"
    bl_idname = "FLUIDSIM_VIEW3D_PT_live_tuning"
    bl_parent_id = "FLUIDSIM_PT_viewport_status"
    bl_order = 100
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = getattr(context, "scene", None)
        owner = active_domain_settings(scene) if scene is not None else None
        if owner is not None:
            column = layout.column(align=True)
            for name in LIVE_TUNING_PROPS:
                draw_property(column, owner, name)
        layout.separator()
        layout.operator(
            "fluidsim.open_scene_properties",
            text="Open FluidSim Scene Properties",
            icon="PROPERTIES",
        )


VIEW3D_PANEL_CLASSES = (FLUIDSIM_VIEW3D_PT_live_tuning,)


def register():
    for cls in VIEW3D_PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(VIEW3D_PANEL_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass
