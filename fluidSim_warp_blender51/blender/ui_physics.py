"""Contextual FluidSim controls in Blender's Object Physics Properties.

This module is a presentation adapter.  It reuses the authoritative Scene and
Object panel draw functions, while selecting only the groups relevant to the
active FluidSim domain, output, or input object.  No settings are copied onto
output objects; every control edits the existing ``Scene.fluidsim`` or
``Object.fluidsim`` RNA owner.
"""

from __future__ import annotations

import bpy

from . import output_publication, ui_object, ui_scene
from .scene_bridge import active_domain_settings
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
    if obj is getattr(fs, "domain_object", None):
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


class _OwnerScene:
    """A read-through view of ``scene`` whose ``fluidsim`` is the domain owner.

    Post Phase-3 migration the canonical domain settings live on
    ``obj.fluidsim_domain`` -- ``solver_settings.build_settings`` already reads
    it through ``active_domain_settings``, so a control still bound to
    ``scene.fluidsim`` would edit a datablock the solver no longer consults.
    The domain mirrors reuse the Scene panel draw functions verbatim, and those
    bind every control to ``context.scene.fluidsim``; handing them this view
    repoints the *owner* to the domain datablock -- a genuine ``PropertyGroup``
    struct, so ``layout.prop`` still edits real RNA -- while every other scene
    attribute reads through unchanged.

    (Runtime ``stat_*`` telemetry is written to ``scene.fluidsim`` by the solver
    and so renders from this same owner until Chunk 5 relocates those readouts
    to Diagnostics; that transitional staleness is displayed-only.)
    """

    __slots__ = ("_scene", "_fluidsim")

    def __init__(self, scene, fluidsim):
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_fluidsim", fluidsim)

    @property
    def fluidsim(self):
        return object.__getattribute__(self, "_fluidsim")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_scene"), name)


class _OwnerContext:
    """A read-through view of ``context`` whose ``scene`` is an `_OwnerScene`."""

    __slots__ = ("_context", "_scene")

    def __init__(self, context, scene):
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_scene", scene)

    @property
    def scene(self):
        return object.__getattribute__(self, "_scene")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_context"), name)


def _make_panel_mirrors(source_panels, *, group: str, accepted_kinds: frozenset[str],
                        owner_redirect: bool = False):
    """Mirror existing panels under the Physics root without copying state."""
    source_panels = tuple(source_panels)
    id_map = {
        str(source_panel.bl_idname): _physics_id(group, str(source_panel.bl_idname))
        for source_panel in source_panels
    }
    classes = []

    def draw_for(source_panel):
        if not owner_redirect:
            def draw(self, context):
                return source_panel.draw(self, context)
            return draw

        def draw(self, context):
            scene = getattr(context, "scene", None)
            owner = active_domain_settings(scene) if scene is not None else None
            # Pre-migration (or before a domain object exists) the scene group is
            # still canonical, so redirecting would change nothing; delegate as-is.
            if owner is None or owner is getattr(scene, "fluidsim", None):
                return source_panel.draw(self, context)
            return source_panel.draw(self, _OwnerContext(context, _OwnerScene(scene, owner)))
        return draw

    def poll_for(source_panel):
        @classmethod
        def poll(cls, context):
            if not FluidSimPhysicsPanel.poll(context):
                return False
            if physics_object_kind(context) not in accepted_kinds:
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
            "bl_order": int(getattr(source_panel, "bl_order", 0)),
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
            text="Open Full FluidSim Settings",
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
            and obj is not scene.fluidsim.domain_object
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

DOMAIN_SOURCE_PANELS = (
    ui_scene.FLUIDSIM_PT_scene_domain,
    ui_scene.FLUIDSIM_PT_scene_capacity,
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
    ui_scene.FLUIDSIM_PT_sim_spacetime,
)
DOMAIN_PANEL_CLASSES = _make_panel_mirrors(
    DOMAIN_SOURCE_PANELS,
    group="domain",
    accepted_kinds=frozenset({"DOMAIN"}),
    owner_redirect=True,
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
    accepted_kinds=frozenset({"SURFACE"}),
)

WHITEWATER_SOURCE_PANELS = (
    ui_scene.FLUIDSIM_PT_scene_whitewater,
    ui_scene.FLUIDSIM_PT_whitewater_emission,
    ui_scene.FLUIDSIM_PT_whitewater_classification,
    ui_scene.FLUIDSIM_PT_whitewater_life,
    ui_scene.FLUIDSIM_PT_whitewater_dynamics,
    ui_scene.FLUIDSIM_PT_whitewater_stats,
)
WHITEWATER_PANEL_CLASSES = _make_panel_mirrors(
    WHITEWATER_SOURCE_PANELS,
    group="whitewater",
    accepted_kinds=frozenset({"WHITEWATER"}),
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
        ui_scene._draw_attribute_group(layout, fs, "surface")


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
        ui_scene._draw_attribute_group(layout, fs, "particles")


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
        layout.prop(fs, "whitewater_display_scale")
        layout.prop(fs, "whitewater_attribute_mode")
        layout.prop(fs, "whitewater_material")
        ui_scene._draw_attribute_group(layout, fs, "whitewater")


class FLUIDSIM_PT_physics_foam_output(_OutputPanel, bpy.types.Panel):
    bl_label = "Foam Output"
    bl_idname = "FLUIDSIM_PT_physics_foam_output"
    bl_parent_id = "FLUIDSIM_PT_physics_root"
    accepted_kind = "FOAM"

    def draw(self, context):
        layout = _layout(self)
        fs = context.scene.fluidsim
        layout.prop(fs, "whitewater_output_mode")
        layout.prop(fs, "whitewater_display_scale")
        layout.prop(fs, "foam_material")
        ui_scene.draw_property_grid(
            layout,
            fs,
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
        layout.prop(fs, "whitewater_display_scale")
        layout.prop(fs, "bubble_material")
        ui_scene.draw_property_grid(
            layout,
            fs,
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
        layout.prop(fs, "whitewater_display_scale")
        layout.prop(fs, "spray_material")
        ui_scene.draw_property_grid(
            layout,
            fs,
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
    *OUTPUT_PANEL_CLASSES,
)


def register():
    for cls in PHYSICS_PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(PHYSICS_PANEL_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass

