"""Staged workflow shelves for the 3D View sidebar: Setup, then Simulate.

The compact status launcher (``FLUIDSIM_PT_viewport_status``) stays pinned at
the top of the FluidSim tab with Preview/Pause and Apply Changes. Below it, one
top-level panel per stage holds the handful of controls an artist reaches for at
that point in the work, so the common loop happens beside the viewport instead of
in the Scene Properties tree. Everything else stays one click away through the
existing Scene Properties panels; nothing is duplicated into a deep sidebar tree.

Every control is bound to the group that actually owns it
(:func:`~.pending_settings.setting_owner`), the same rule the Live Tuning shelf
follows, so an edit made here is the value the solver reads.
"""

from __future__ import annotations

import bpy

from . import file_compat
from .pending_settings import setting_owner
from .scene_bridge import active_domain_settings
from .ui_common import (
    FluidSimView3DPanel,
    draw_property,
    draw_readout_grid,
)

# Stage controls, kept as data so the Blender validator can pin them without
# re-deriving them from draw code.
SETUP_DOMAIN_PROPS = ("voxel_size", "particles_per_voxel")
SIMULATE_PROPS = (
    "flip_ratio",
    "viscosity_enabled",
    "viscosity_constant",
    "surface_tension_enabled",
    "surface_tension_coefficient",
    "use_scene_gravity",
    "gravity",
)
SIMULATE_READOUTS = (
    ("stat_step_ms", "Step (ms)"),
    ("stat_substeps", "Substeps"),
    ("stat_particles", "Particles"),
    ("stat_voxels", "Voxels"),
    ("stat_cfl_clamped", "CFL clamped"),
)


class FluidSimStagePanel(FluidSimView3DPanel):
    """Base for the stage panels: any scene with FluidSim settings."""

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return scene is not None and getattr(scene, "fluidsim", None) is not None


def _layout(panel):
    layout = panel.layout
    layout.use_property_split = True
    layout.use_property_decorate = False
    return layout


def _prop(layout, scene, name, **kwargs):
    """Draw ``name`` on the group that owns it; skip it when that group lacks it."""
    return draw_property(layout, setting_owner(scene, name), name, **kwargs)


def _is_advanced(scene) -> bool:
    """Fail-open Basic/Advanced gate matching the Scene Properties panels."""
    fs = getattr(scene, "fluidsim", None)
    return bool(fs is None or getattr(fs, "ui_mode", "ADVANCED") == "ADVANCED")


def _guided_setup_buttons(layout, *, primary_scale: float = 1.3):
    row = layout.row(align=True)
    row.scale_y = primary_scale
    preview = row.operator("fluidsim.setup_scene", text="Set Up & Preview…", icon="PLAY")
    preview.start_preview = True
    only = layout.operator("fluidsim.setup_scene", text="Set Up Only…", icon="MOD_FLUIDSIM")
    only.start_preview = False


class FLUIDSIM_STAGE_PT_setup(FluidSimStagePanel, bpy.types.Panel):
    """Stage 1: choose quality, size the domain, and run guided setup."""

    bl_label = "1 · Setup"
    bl_idname = "FLUIDSIM_STAGE_PT_setup"
    bl_order = 10

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        fs = scene.fluidsim
        file_compat.draw_pre_bridge_warning(layout, scene)

        domain = active_domain_settings(scene)
        if domain is None:
            # The one obvious first move for an empty scene.
            box = layout.box()
            box.label(text="Start here: no liquid domain yet", icon="INFO")
            _guided_setup_buttons(box)

        layout.prop(fs, "ui_mode", expand=True)
        _prop(layout, scene, "quality")

        preset = layout.row(align=True)
        _prop(preset, scene, "product_preset", text="")
        preset.operator("fluidsim.apply_product_preset", text="Apply", icon="PRESET")

        if domain is None:
            return
        layout.separator()
        draw_property(layout, fs, "domain_object")
        column = layout.column(align=True)
        for name in SETUP_DOMAIN_PROPS:
            _prop(column, scene, name)


class FLUIDSIM_STAGE_PT_setup_advanced(FluidSimStagePanel, bpy.types.Panel):
    """Capacity and re-run controls, shown only in the Advanced interface."""

    bl_label = "More Domain Options"
    bl_idname = "FLUIDSIM_STAGE_PT_setup_advanced"
    bl_parent_id = "FLUIDSIM_STAGE_PT_setup"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return FluidSimStagePanel.poll(context) and _is_advanced(context.scene)

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        if active_domain_settings(scene) is not None:
            column = layout.column(align=True)
            for name in (
                "max_particles",
                "auto_grow_particles",
                "topology_expand",
                "open_boundaries",
            ):
                _prop(column, scene, name)
            layout.separator()
            _guided_setup_buttons(layout, primary_scale=1.0)
        layout.operator(
            "fluidsim.open_scene_properties",
            text="All Domain Settings…",
            icon="PROPERTIES",
        )


class FLUIDSIM_STAGE_PT_simulate(FluidSimStagePanel, bpy.types.Panel):
    """Stage 2: the physics an artist tunes while watching the preview."""

    bl_label = "2 · Simulate"
    bl_idname = "FLUIDSIM_STAGE_PT_simulate"
    bl_order = 20

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        domain = active_domain_settings(scene)
        if domain is None:
            layout.label(text="Set up a liquid domain first.", icon="INFO")
            return

        if getattr(domain, "sample_type", "FLIP") == "FLIP":
            _prop(layout, scene, "flip_ratio")

        viscosity = layout.column(align=True)
        _prop(viscosity, scene, "viscosity_enabled")
        if domain.viscosity_enabled and not getattr(domain, "viscosity_from_particles", False):
            _prop(viscosity, scene, "viscosity_constant")

        tension = layout.column(align=True)
        _prop(tension, scene, "surface_tension_enabled")
        if domain.surface_tension_enabled:
            _prop(tension, scene, "surface_tension_coefficient")

        gravity = layout.column(align=True)
        _prop(gravity, scene, "use_scene_gravity")
        if not domain.use_scene_gravity:
            _prop(gravity, scene, "gravity")

        layout.separator()
        draw_readout_grid(
            layout, scene.fluidsim, SIMULATE_READOUTS, title="Live Readout"
        )
        layout.operator(
            "fluidsim.open_scene_properties",
            text="All Simulation Settings…",
            icon="PROPERTIES",
        )


STAGE_PANEL_CLASSES = (
    FLUIDSIM_STAGE_PT_setup,
    FLUIDSIM_STAGE_PT_setup_advanced,
    FLUIDSIM_STAGE_PT_simulate,
)


def register():
    for cls in STAGE_PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(STAGE_PANEL_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass
