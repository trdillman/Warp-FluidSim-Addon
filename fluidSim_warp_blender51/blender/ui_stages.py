"""Staged workflow shelves for the 3D View sidebar: Setup, Simulate, Look, Bake.

The status/transport strip (``FLUIDSIM_PT_viewport_status``) stays pinned at the
top of the FluidSim tab with Preview/Pause and Apply Changes. Below it, one
top-level panel per stage holds the handful of controls an artist reaches for at
that point in the work, so the common loop happens beside the viewport instead of
in the Scene Properties tree. Everything else stays one click away through the
existing Scene Properties panels; nothing is duplicated into a deep sidebar tree.

Every control is bound to the group that actually owns it
(:func:`~.pending_settings.setting_owner`), so an edit made here is the value the
solver reads.

The Bake stage is the only home of Cache & Production Bake: the Scene tab no
longer carries those panels. It draws the shared cache functions from
``ui_scene`` (the bake start / resume / restart / cancel state machine), plus
Storage & Range and Manage Cache shelves, so every cache control stays reachable.
"""

from __future__ import annotations

import bpy

from . import file_compat
from .pending_settings import setting_owner
from .scene_access import active_domain_settings
from .ui_common import (
    FluidSimView3DPanel,
    draw_property,
    draw_readout_grid,
)
from .ui_scene import (
    cache_manage_available,
    draw_cache_bake,
    draw_cache_manage,
    draw_cache_playback,
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
LOOK_PROPS = (
    "liquid_view_mode",
    "surface_enabled",
    "surface_smooth_iterations",
    "particle_display_fraction",
    "particle_display_scale",
    "whitewater_enabled",
    "ww_emission_rate",
    "show_whitewater",
    "show_overlay",
)
LOOK_MATERIAL_PROPS = ("surface_material", "particle_material", "whitewater_material")
BAKE_PROPS = ("cache_path", "export_attributes")
# Drawn by the shared Production Bake draw; pinned so a rename there is caught.
BAKE_DELEGATED_PROPS = ("bake_start", "bake_end")
BAKE_STORAGE_PROPS = (
    "cache_compress",
    "checkpoint_interval",
    "cache_use_custom_range",
    "cache_frame_start",
    "cache_frame_end",
)
_SURFACE_VIEW_MODES = frozenset({"SURFACE", "SURFACE_PARTICLES"})
_PARTICLE_VIEW_MODES = frozenset({"PARTICLES", "SURFACE_PARTICLES"})


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


def draw_quality(layout, scene):
    """The Quality tier, drawn on the group that owns it."""
    return _prop(layout, scene, "quality")


def draw_guided_setup_buttons(layout, *, primary_scale: float = 1.3):
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
            draw_guided_setup_buttons(box)

        layout.prop(fs, "ui_mode", expand=True)
        draw_quality(layout, scene)

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
            draw_guided_setup_buttons(layout, primary_scale=1.0)
        layout.operator(
            "fluidsim.open_scene_properties",
            text="All Domain Settings…",
            icon="PROPERTIES",
        )


def draw_simulate_controls(layout, scene, domain):
    """The physics an artist tunes while watching the preview.

    One drawing shared by the Simulate stage and the viewport-header popover, so
    both surfaces bind the same properties to the same owner.
    """
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

        draw_simulate_controls(layout, scene, domain)

        layout.separator()
        draw_readout_grid(
            layout, scene.fluidsim, SIMULATE_READOUTS, title="Live Readout"
        )
        layout.operator(
            "fluidsim.open_scene_properties",
            text="All Simulation Settings…",
            icon="PROPERTIES",
        )


class FLUIDSIM_STAGE_PT_look(FluidSimStagePanel, bpy.types.Panel):
    """Stage 3: what the viewport shows, and which surfaces the solver builds."""

    bl_label = "3 · Look"
    bl_idname = "FLUIDSIM_STAGE_PT_look"
    bl_order = 30

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        fs = scene.fluidsim
        domain = active_domain_settings(scene)
        if domain is None:
            layout.label(text="Set up a liquid domain first.", icon="INFO")
            return

        _prop(layout, scene, "liquid_view_mode")

        surface = layout.column(align=True)
        _prop(surface, scene, "surface_enabled")
        if domain.surface_enabled:
            _prop(surface, scene, "surface_smooth_iterations", text="Smoothing")

        if fs.liquid_view_mode in _PARTICLE_VIEW_MODES:
            particles = layout.column(align=True)
            _prop(particles, scene, "particle_display_fraction")
            _prop(particles, scene, "particle_display_scale")

        whitewater = layout.column(align=True)
        _prop(whitewater, scene, "whitewater_enabled")
        if domain.whitewater_enabled:
            _prop(whitewater, scene, "ww_emission_rate", text="Amount")
            _prop(whitewater, scene, "show_whitewater")

        _prop(layout, scene, "show_overlay", text="Viewport HUD")


class FLUIDSIM_STAGE_PT_look_materials(FluidSimStagePanel, bpy.types.Panel):
    """Material pickers for the outputs the current view actually produces."""

    bl_label = "Materials"
    bl_idname = "FLUIDSIM_STAGE_PT_look_materials"
    bl_parent_id = "FLUIDSIM_STAGE_PT_look"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return (
            FluidSimStagePanel.poll(context)
            and active_domain_settings(context.scene) is not None
        )

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        fs = scene.fluidsim
        domain = active_domain_settings(scene)
        if fs.liquid_view_mode in _SURFACE_VIEW_MODES:
            _prop(layout, scene, "surface_material")
        if fs.liquid_view_mode in _PARTICLE_VIEW_MODES:
            _prop(layout, scene, "particle_material")
        if domain is not None and domain.whitewater_enabled:
            _prop(layout, scene, "whitewater_material")
        layout.operator(
            "fluidsim.open_scene_properties",
            text="All Output Settings…",
            icon="PROPERTIES",
        )


class FLUIDSIM_STAGE_PT_bake(FluidSimStagePanel, bpy.types.Panel):
    """Stage 4: bake the cache, play it back, and export from it."""

    bl_label = "4 · Bake & Export"
    bl_idname = "FLUIDSIM_STAGE_PT_bake"
    bl_order = 40

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        # The cache location is scene-level, so it stays editable (and an existing
        # cache stays manageable below) even before a domain exists.
        _prop(layout, scene, "cache_path")
        if active_domain_settings(scene) is None:
            layout.label(text="Set up a liquid domain first to bake.", icon="INFO")
            return

        _prop(layout, scene, "export_attributes")

        # One shared drawing of the bake state machine (start / resume / restart /
        # cancel, progress, runtime readiness).
        layout.separator()
        draw_cache_playback(layout, context)
        layout.separator()
        draw_cache_bake(layout, context)

        layout.separator()
        layout.operator(
            "fluidsim.export_frame", text="Export Current Frame…", icon="EXPORT"
        )


class FLUIDSIM_STAGE_PT_bake_storage(FluidSimStagePanel, bpy.types.Panel):
    """Cache compression, checkpoint spacing and the shared playback/export range."""

    bl_label = "Storage & Range"
    bl_idname = "FLUIDSIM_STAGE_PT_bake_storage"
    bl_parent_id = "FLUIDSIM_STAGE_PT_bake"
    bl_order = 1
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        fs = scene.fluidsim
        _prop(layout, scene, "cache_compress")
        _prop(layout, scene, "checkpoint_interval")
        _prop(layout, scene, "cache_use_custom_range")
        if fs.cache_use_custom_range:
            row = layout.row(align=True)
            _prop(row, scene, "cache_frame_start")
            _prop(row, scene, "cache_frame_end")


class FLUIDSIM_STAGE_PT_bake_manage(FluidSimStagePanel, bpy.types.Panel):
    """Integrity scan, move, rename and free, for a recognized FluidSim cache."""

    bl_label = "Manage Cache"
    bl_idname = "FLUIDSIM_STAGE_PT_bake_manage"
    bl_parent_id = "FLUIDSIM_STAGE_PT_bake"
    bl_order = 2
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return FluidSimStagePanel.poll(context) and cache_manage_available(context)

    def draw(self, context):
        draw_cache_manage(_layout(self), context)


STAGE_PANEL_CLASSES = (
    FLUIDSIM_STAGE_PT_setup,
    FLUIDSIM_STAGE_PT_setup_advanced,
    FLUIDSIM_STAGE_PT_simulate,
    FLUIDSIM_STAGE_PT_look,
    FLUIDSIM_STAGE_PT_look_materials,
    FLUIDSIM_STAGE_PT_bake,
    FLUIDSIM_STAGE_PT_bake_storage,
    FLUIDSIM_STAGE_PT_bake_manage,
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
