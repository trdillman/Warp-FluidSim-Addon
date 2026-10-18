"""Contextual FluidSim controls in Blender's Object Physics Properties.

This module is a presentation adapter.  It reuses the authoritative Scene and
Object panel draw functions, while selecting only the groups relevant to the
active FluidSim domain, output, or input object.  No settings are copied onto
output objects; every control edits the existing ``Scene.fluidsim``,
``Object.fluidsim`` or domain ``Object.fluidsim_domain`` RNA owner.
"""

from __future__ import annotations

import bpy

from . import effect_ui, file_compat, output_publication, ui_object, ui_scene
from .scene_access import active_domain_object
from .ui_common import FluidSimObjectPanel, active_object_role


class FluidSimPhysicsPanel(FluidSimObjectPanel):
    """Base for FluidSim panels in Blender's Physics Properties context."""

    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"


_OUTPUT_KINDS = {
    output_publication.SURFACE_NAME: "SURFACE",
    output_publication.PARTICLES_NAME: "PARTICLES",
    output_publication.WHITEWATER_NAME: "WHITEWATER",
    output_publication.FOAM_NAME: "FOAM",
    output_publication.BUBBLES_NAME: "BUBBLES",
    output_publication.SPRAY_NAME: "SPRAY",
}


def physics_object_kind(context) -> str:
    """Return the active object's FluidSim presentation kind."""
    scene = getattr(context, "scene", None)
    obj = getattr(context, "object", None)
    fs = getattr(scene, "fluidsim", None) if scene is not None else None
    if obj is None or fs is None:
        return "NONE"
    if obj is active_domain_object(scene):
        return "DOMAIN"
    try:
        output_tag = obj.get(output_publication.OUTPUT_TAG)
    except AttributeError:
        output_tag = None
    if output_tag in _OUTPUT_KINDS:
        return _OUTPUT_KINDS[output_tag]
    return "INPUT" if active_object_role(context) != "NONE" else "NONE"


def _layout(panel):
    layout = panel.layout
    layout.use_property_split = True
    layout.use_property_decorate = False
    return layout


def _physics_id(group: str, source_id: str) -> str:
    prefix = "FLUIDSIM_PT_"
    suffix = source_id[len(prefix):] if source_id.startswith(prefix) else source_id
    return f"FLUIDSIM_PT_physics_{group}_{suffix}"


def _make_panel_mirrors(
    source_panels,
    *,
    group: str,
    accepted_kinds: frozenset[str],
    kinds_by_panel=None,
    order_offset: int = 0,
):
    """Mirror existing panels under the Physics root without copying state.

    The source panels resolve their settings group themselves (the domain
    panels through ``ui_scene._domain_owner``), and the DOMAIN kind already
    means a domain object is designated, so a mirror needs no redirect.
    ``kinds_by_panel`` narrows individual source panels to fewer kinds than
    the group accepts. ``order_offset`` shifts the mirrors' ``bl_order`` so a
    group can be placed after another group's top-level panels.
    """
    kinds_by_panel = dict(kinds_by_panel or {})
    source_panels = tuple(source_panels)
    id_map = {
        str(source_panel.bl_idname): _physics_id(group, str(source_panel.bl_idname))
        for source_panel in source_panels
    }
    classes = []

    def draw_for(source_panel):
        def draw(self, context):
            return source_panel.draw(self, context)
        return draw

    def poll_for(source_panel):
        kinds = kinds_by_panel.get(source_panel, accepted_kinds)

        @classmethod
        def poll(cls, context):
            if not FluidSimPhysicsPanel.poll(context):
                return False
            if physics_object_kind(context) not in kinds:
                return False
            source_poll = getattr(source_panel, "poll", None)
            return bool(source_panel.poll(context)) if callable(source_poll) else True
        return poll

    for source_panel in source_panels:
        source_id = str(source_panel.bl_idname)
        attributes = {
            "__module__": __name__,
            "__doc__": f"Physics Properties mirror of {source_panel.__name__}.",
            "bl_label": str(getattr(source_panel, "bl_label", source_panel.__name__)),
            "bl_idname": id_map[source_id],
            "bl_space_type": FluidSimPhysicsPanel.bl_space_type,
            "bl_region_type": FluidSimPhysicsPanel.bl_region_type,
            "bl_context": FluidSimPhysicsPanel.bl_context,
            "bl_order": int(getattr(source_panel, "bl_order", 0)) + order_offset,
            "bl_options": set(getattr(source_panel, "bl_options", set())),
            "draw": draw_for(source_panel),
            "poll": poll_for(source_panel),
        }
        source_parent = str(getattr(source_panel, "bl_parent_id", "") or "")
        attributes["bl_parent_id"] = id_map.get(
            source_parent, "FLUIDSIM_PT_physics_root"
        )
        classes.append(
            type(
                id_map[source_id],
                (FluidSimPhysicsPanel, bpy.types.Panel),
                attributes,
            )
        )
    return tuple(classes)


class FLUIDSIM_PT_physics_root(FluidSimPhysicsPanel, bpy.types.Panel):
    bl_label = "FluidSim"
    bl_idname = "FLUIDSIM_PT_physics_root"
    bl_order = 0

    def draw(self, context):
        layout = _layout(self)
        scene = context.scene
        obj = context.object
        kind = physics_object_kind(context)
        file_compat.draw_pre_bridge_warning(layout, scene)
        labels = {
            "NONE": "No FluidSim role assigned",
            "INPUT": active_object_role(context).replace("_", " ").title(),
            "DOMAIN": "Simulation Domain",
            "SURFACE": "Surface Output",
            "PARTICLES": "Liquid Particle Output",
            "WHITEWATER": "Combined Whitewater Output",
            "FOAM": "Foam Output",
            "BUBBLES": "Bubble Output",
            "SPRAY": "Spray Output",
        }
        layout.label(text=labels.get(kind, kind.title()), icon="MOD_FLUIDSIM")
        layout.label(text=obj.name, icon="OBJECT_DATA")

        layout.operator(
            "fluidsim.open_scene_properties",
            # The Scene tab holds scene-level state only (status, presets,
            # outputs, diagnostics); the domain settings are in this tab.
            text="Open Scene FluidSim Settings",
            icon="PROPERTIES",
        )
        if kind == "DOMAIN" or kind in _OUTPUT_KINDS.values():
            install = layout.operator(
                "fluidsim.install_render_nodes",
                text="Create / Repair Outputs",
                icon="GEOMETRY_NODES",
            )
            install.force = False

        if kind == "DOMAIN":
            layout.operator(
                "fluidsim.clear_domain_assignment",
                text="Remove Domain Assignment",
                icon="X",
            )
        elif kind in _OUTPUT_KINDS.values():
            layout.operator(
                "fluidsim.remove_selected_output",
                text="Remove Selected FluidSim Output…",
                icon="TRASH",
            )
        elif (
            kind == "NONE"
            and getattr(obj, "type", "") == "MESH"
            and obj is not active_domain_object(scene)
        ):
            layout.operator(
                "fluidsim.use_active_as_domain",
                text="Use Active Object as FluidSim Domain",
                icon="CUBE",
            )
        if kind == "INPUT":
            layout.operator(
                "fluidsim.clear_object_role",
                text="Remove FluidSim Role from Selection",
                icon="X",
            )


# Input-role panels are reused verbatim, but hidden for generated domain/output
# objects.  The existing role root retains all add/remove actions.
INPUT_PANEL_CLASSES = _make_panel_mirrors(
    ui_object.OBJECT_PANEL_CLASSES,
    group="input",
    accepted_kinds=frozenset({"NONE", "INPUT"}),
)

# The domain controls, mirrored onto the domain's Physics tab (the Scene tab
# registers the same draw sources, ``ui_scene.SCENE_TAB_PANEL_CLASSES``).
# Persistent Attributes and Profiling are children of scene-owned panels in the
# Scene tab, so the mirror makes them top-level domain panels.
DOMAIN_SOURCE_PANELS = (
    ui_scene.FLUIDSIM_PT_scene_domain,
    ui_scene.FLUIDSIM_PT_scene_capacity,
    ui_scene.FLUIDSIM_PT_scene_height,
    ui_scene.FLUIDSIM_PT_scene_simulation,
    ui_scene.FLUIDSIM_PT_sim_time,
    ui_scene.FLUIDSIM_PT_sim_gravity,
    ui_scene.FLUIDSIM_PT_sim_pressure,
    ui_scene.FLUIDSIM_PT_sim_viscosity,
    ui_scene.FLUIDSIM_PT_sim_surface_tension,
    ui_scene.FLUIDSIM_PT_sim_color_mixing,
    ui_scene.FLUIDSIM_PT_sim_relax,
    ui_scene.FLUIDSIM_PT_sim_limits,
    ui_scene.FLUIDSIM_PT_sim_forces,
    ui_scene.FLUIDSIM_PT_sim_dissipation,
    ui_scene.FLUIDSIM_PT_sim_attraction,
    ui_scene.FLUIDSIM_PT_output_attributes,
    ui_scene.FLUIDSIM_PT_attrs_surface,
    ui_scene.FLUIDSIM_PT_attrs_particles,
    ui_scene.FLUIDSIM_PT_attrs_whitewater,
    ui_scene.FLUIDSIM_PT_diagnostics_profiling,
)
DOMAIN_PANEL_CLASSES = _make_panel_mirrors(
    DOMAIN_SOURCE_PANELS,
    group="domain",
    accepted_kinds=frozenset({"DOMAIN"}),
)

SURFACE_SOURCE_PANELS = (
    ui_scene.FLUIDSIM_PT_scene_surface,
    ui_scene.FLUIDSIM_PT_surface_resolution,
    ui_scene.FLUIDSIM_PT_surface_reconstruct,
    ui_scene.FLUIDSIM_PT_surface_filters,
)
SURFACE_PANEL_CLASSES = _make_panel_mirrors(
    SURFACE_SOURCE_PANELS,
    group="surface",
    accepted_kinds=frozenset({"DOMAIN", "SURFACE"}),
)

# Admission-Status readbacks were consolidated into the scene Diagnostics
# telemetry panel (D4), so the per-object physics mirror no longer carries them;
# the mirror stays edit-only, matching the scene surface it derives from.
WHITEWATER_SOURCE_PANELS = (
    ui_scene.FLUIDSIM_PT_scene_whitewater,
    ui_scene.FLUIDSIM_PT_whitewater_emission,
    ui_scene.FLUIDSIM_PT_whitewater_classification,
    ui_scene.FLUIDSIM_PT_whitewater_life,
    ui_scene.FLUIDSIM_PT_whitewater_dynamics,
)
# Emission, classification and lifetime are common to every whitewater stream,
# so SEPARATE-mode Foam/Bubbles/Spray outputs keep them (the combined shell is
# hidden in that mode). Per-stream dynamics stay on the combined output only:
# each separate output already draws its own stream's dynamics.
WHITEWATER_PANEL_CLASSES = _make_panel_mirrors(
    WHITEWATER_SOURCE_PANELS,
    group="whitewater",
    accepted_kinds=frozenset({"DOMAIN", "WHITEWATER", "FOAM", "BUBBLES", "SPRAY"}),
    kinds_by_panel={
        ui_scene.FLUIDSIM_PT_whitewater_dynamics: frozenset({"DOMAIN", "WHITEWATER"}),
    },
)

# The domain's Physics tab also carries every FluidSim setting: the scene-level
# panels (status, presets, outputs, display, cache & bake, diagnostics) are
# mirrored after the domain's own groups. They still edit ``Scene.fluidsim``.
SCENE_ON_DOMAIN_PANEL_CLASSES = _make_panel_mirrors(
    ui_scene.SCENE_PANEL_CLASSES,
    group="scene",
    accepted_kinds=frozenset({"DOMAIN"}),
    order_offset=200,
)

EFFECTS_ON_DOMAIN_PANEL_CLASSES = _make_panel_mirrors(
    effect_ui.EFFECT_PANEL_CLASSES,
    group="effects",
    accepted_kinds=frozenset({"DOMAIN"}),
    order_offset=200,
)


class _OutputPanel(FluidSimPhysicsPanel):
    accepted_kind = ""

    @classmethod
    def poll(cls, context):
        return (
            FluidSimPhysicsPanel.poll(context)
            and physics_object_kind(context) == cls.accepted_kind
        )


class FLUIDSIM_PT_physics_surface_output(_OutputPanel, bpy.types.Panel):
    bl_label = "Surface Output"
    bl_idname = "FLUIDSIM_PT_physics_surface_output"
    bl_parent_id = "FLUIDSIM_PT_physics_root"
    accepted_kind = "SURFACE"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "liquid_view_mode")
        layout.prop(fs, "surface_material")
        layout.prop(fs, "surface_attribute_mode")
        ui_scene._draw_attribute_group(
            layout, ui_scene._domain_owner(context), "surface")


class FLUIDSIM_PT_physics_particles_output(_OutputPanel, bpy.types.Panel):
    bl_label = "Liquid Particle Output"
    bl_idname = "FLUIDSIM_PT_physics_particles_output"
    bl_parent_id = "FLUIDSIM_PT_physics_root"
    accepted_kind = "PARTICLES"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "liquid_view_mode")
        layout.prop(fs, "particle_display_fraction")
        layout.prop(fs, "particle_display_scale")
        layout.prop(fs, "particle_attribute_mode")
        layout.prop(fs, "particle_material")
        ui_scene._draw_attribute_group(
            layout, ui_scene._domain_owner(context), "particles")


class FLUIDSIM_PT_physics_whitewater_output(_OutputPanel, bpy.types.Panel):
    bl_label = "Combined Whitewater Output"
    bl_idname = "FLUIDSIM_PT_physics_whitewater_output"
    bl_parent_id = "FLUIDSIM_PT_physics_root"
    accepted_kind = "WHITEWATER"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "show_whitewater")
        layout.prop(fs, "whitewater_output_mode")
        layout.prop(fs, "whitewater_display_fraction")
        layout.prop(fs, "whitewater_display_scale")
        layout.prop(fs, "whitewater_attribute_mode")
        layout.prop(fs, "whitewater_material")
        ui_scene._draw_attribute_group(
            layout, ui_scene._domain_owner(context), "whitewater")


class FLUIDSIM_PT_physics_foam_output(_OutputPanel, bpy.types.Panel):
    bl_label = "Foam Output"
    bl_idname = "FLUIDSIM_PT_physics_foam_output"
    bl_parent_id = "FLUIDSIM_PT_physics_root"
    accepted_kind = "FOAM"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "whitewater_output_mode")
        layout.prop(fs, "whitewater_display_fraction")
        layout.prop(fs, "whitewater_display_scale")
        layout.prop(fs, "foam_material")
        # Output mode, display scale and the material picker are scene-scoped;
        # the ww_* dynamics below are solver settings read from the domain
        # group, so that grid has to edit the domain owner instead.
        ui_scene.draw_property_grid(
            layout,
            ui_scene._domain_owner(context),
            (
                "ww_foam_advection",
                "ww_foam_preservation",
                "ww_foam_density_influence",
                "ww_foam_life_multiplier",
            ),
        )


class FLUIDSIM_PT_physics_bubbles_output(_OutputPanel, bpy.types.Panel):
    bl_label = "Bubble Output"
    bl_idname = "FLUIDSIM_PT_physics_bubbles_output"
    bl_parent_id = "FLUIDSIM_PT_physics_root"
    accepted_kind = "BUBBLES"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "whitewater_output_mode")
        layout.prop(fs, "whitewater_display_fraction")
        layout.prop(fs, "whitewater_display_scale")
        layout.prop(fs, "bubble_material")
        # Output mode, display scale and the material picker are scene-scoped;
        # the ww_* dynamics below are solver settings read from the domain
        # group, so that grid has to edit the domain owner instead.
        ui_scene.draw_property_grid(
            layout,
            ui_scene._domain_owner(context),
            (
                "ww_bubble_drag",
                "ww_bubble_buoyancy",
                "ww_bubble_life_multiplier",
            ),
        )


class FLUIDSIM_PT_physics_spray_output(_OutputPanel, bpy.types.Panel):
    bl_label = "Spray Output"
    bl_idname = "FLUIDSIM_PT_physics_spray_output"
    bl_parent_id = "FLUIDSIM_PT_physics_root"
    accepted_kind = "SPRAY"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "whitewater_output_mode")
        layout.prop(fs, "whitewater_display_fraction")
        layout.prop(fs, "whitewater_display_scale")
        layout.prop(fs, "spray_material")
        # Output mode, display scale and the material picker are scene-scoped;
        # the ww_* dynamics below are solver settings read from the domain
        # group, so that grid has to edit the domain owner instead.
        ui_scene.draw_property_grid(
            layout,
            ui_scene._domain_owner(context),
            (
                "ww_spray_drag",
                "ww_spray_gravity_scale",
                "ww_spray_life_multiplier",
            ),
        )


OUTPUT_PANEL_CLASSES = (
    FLUIDSIM_PT_physics_surface_output,
    FLUIDSIM_PT_physics_particles_output,
    FLUIDSIM_PT_physics_whitewater_output,
    FLUIDSIM_PT_physics_foam_output,
    FLUIDSIM_PT_physics_bubbles_output,
    FLUIDSIM_PT_physics_spray_output,
)

PHYSICS_PANEL_CLASSES = (
    FLUIDSIM_PT_physics_root,
    *INPUT_PANEL_CLASSES,
    *DOMAIN_PANEL_CLASSES,
    *SURFACE_PANEL_CLASSES,
    *WHITEWATER_PANEL_CLASSES,
    *SCENE_ON_DOMAIN_PANEL_CLASSES,
    *EFFECTS_ON_DOMAIN_PANEL_CLASSES,
    *OUTPUT_PANEL_CLASSES,
)


def register():
    for cls in PHYSICS_PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    from ..unregister_guard import unregister_classes

    unregister_classes(PHYSICS_PANEL_CLASSES)

