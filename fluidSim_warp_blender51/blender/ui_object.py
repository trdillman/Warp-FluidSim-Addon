"""Role-aware FluidSim panels for Blender Object Properties."""

from __future__ import annotations

import bpy

from .scene_inputs import uses_native_source_plan
from .ui_common import (
    FluidSimObjectPanel,
    active_object_role,
    canonical_region_kind,
    draw_property_grid,
    draw_status_message,
)


def _layout(panel):
    layout = panel.layout
    layout.use_property_split = True
    layout.use_property_decorate = False
    return layout


def _settings(context):
    obj = getattr(context, "object", None)
    return getattr(obj, "fluidsim", None) if obj is not None else None


def role_poll(context, *roles: str) -> bool:
    return FluidSimObjectPanel.poll(context) and active_object_role(context) in set(roles)


def volume_grid_poll(context) -> bool:
    if not role_poll(context, "SOURCE", "SINK", "FIELD"):
        return False
    return canonical_region_kind(getattr(context, "object", None)) == "VOLUME_GRID"


def _selected_role_counts(context):
    result = {role: 0 for role in ("NONE", "SOURCE", "SINK", "COLLIDER", "FORCE", "FIELD")}
    for obj in tuple(getattr(context, "selected_objects", ()) or ()):
        settings = getattr(obj, "fluidsim", None)
        role = str(getattr(settings, "role", "NONE") or "NONE")
        result[role if role in result else "NONE"] += 1
    return result


class FLUIDSIM_PT_object_role(FluidSimObjectPanel, bpy.types.Panel):
    bl_label = "FluidSim Role — Selected Objects"
    bl_idname = "FLUIDSIM_PT_object_role"
    bl_order = 0

    def draw(self, context):
        layout = _layout(self)
        settings = _settings(context)
        role = active_object_role(context)

        # Primary control: the active object's FluidSim role as a native Type
        # dropdown (drawn first). Editing it retags exactly the active object.
        if settings is not None:
            layout.prop(settings, "role")
            if role != "NONE":
                layout.prop(settings, "enabled")

        # Batch helpers below: apply a role to every selected object at once.
        counts = _selected_role_counts(context)
        selected_count = len(tuple(context.selected_objects or ()))
        tagged = sum(value for key, value in counts.items() if key != "NONE")
        helpers = layout.column(align=True)
        helpers.label(
            text=f"Apply to Selected — {selected_count} selected · {tagged} tagged",
            icon="OBJECT_DATA",
        )
        actions = helpers.column(align=True)
        if counts["SOURCE"] < selected_count:
            actions.operator("fluidsim.tag_liquid_emitter", text="Set Selected as Liquid Sources", icon="ADD")
        if counts["SINK"] < selected_count:
            actions.operator("fluidsim.tag_liquid_sink", text="Set Selected as Liquid Sinks", icon="REMOVE")
        if counts["COLLIDER"] < selected_count:
            actions.operator("fluidsim.tag_collider", text="Set Selected Meshes as Colliders", icon="MOD_PHYSICS")
        if counts["FORCE"] < selected_count:
            actions.operator("fluidsim.tag_force_region", text="Set Selected as Force Regions", icon="FORCE_FORCE")
        if counts["FIELD"] < selected_count:
            actions.operator("fluidsim.tag_field_source", text="Set Selected as Field Sources", icon="MOD_FLUIDSIM")
        if tagged:
            actions.operator("fluidsim.clear_object_role", text="Clear Roles from Selected", icon="X")


class FLUIDSIM_PT_object_status(FluidSimObjectPanel, bpy.types.Panel):
    bl_label = "Active Object Input Status"
    bl_idname = "FLUIDSIM_PT_object_status"
    bl_parent_id = "FLUIDSIM_PT_object_role"

    @classmethod
    def poll(cls, context):
        if not FluidSimObjectPanel.poll(context) or active_object_role(context) == "NONE":
            return False
        settings = _settings(context)
        return bool(
            getattr(settings, "mesh_error", "")
            or getattr(settings, "source_status_message", "")
            or active_object_role(context) == "SOURCE"
        )

    def draw(self, context):
        layout = _layout(self)
        settings = _settings(context)
        if settings.mesh_error:
            draw_status_message(layout, settings.mesh_error, "ERROR")
        if settings.source_status_message:
            severity = str(settings.source_status_severity or "INFO").upper()
            draw_status_message(layout, settings.source_status_message, severity)
        if settings.source_status_code:
            layout.label(text=f"Code: {settings.source_status_code}")
        if active_object_role(context) == "SOURCE":
            layout.label(text=f"Last emitted: {settings.source_status_emitted:,}")
            if settings.source_status_capacity_limited:
                draw_status_message(layout, "Source emission was capacity limited.", "WARNING")


class FLUIDSIM_PT_object_region(FluidSimObjectPanel, bpy.types.Panel):
    bl_label = "Region Geometry"
    bl_idname = "FLUIDSIM_PT_object_region"
    bl_order = 10

    @classmethod
    def poll(cls, context):
        return role_poll(context, "SOURCE", "SINK", "FORCE", "FIELD")

    def draw(self, context):
        layout = _layout(self)
        settings = _settings(context)
        role = active_object_role(context)
        if role == "SOURCE":
            layout.prop(settings, "source_region_kind")
        elif role == "SINK":
            layout.prop(settings, "sink_region_kind")
        elif role == "FIELD":
            layout.prop(settings, "field_region_kind")

        region = canonical_region_kind(context.object)
        if role == "FORCE" or region == "ANALYTIC":
            layout.prop(settings, "shape")
            layout.prop(settings, "feather")
        elif region in {"MESH_SOLID", "MESH_SHELL"}:
            layout.prop(settings, "mesh_object")
            layout.prop(settings, "mesh_evaluation_mode")
            if region == "MESH_SHELL":
                layout.prop(settings, "mesh_surface_thickness")
        elif role == "SOURCE" and region == "POINTS":
            layout.prop(settings, "point_source_representation")
            layout.prop(settings, "point_input_object")
            if settings.point_source_representation == "RASTERIZED_REGION":
                layout.prop(settings, "point_rasterize_radius")
        # Point, volume and field inputs are evaluated on the same First /
        # Every Sim Frame cadence as meshes (B47).
        if uses_native_source_plan(settings):
            layout.prop(settings, "mesh_evaluation_mode")


class FLUIDSIM_PT_object_source(FluidSimObjectPanel, bpy.types.Panel):
    bl_label = "Liquid Source"
    bl_idname = "FLUIDSIM_PT_object_source"
    bl_order = 20

    @classmethod
    def poll(cls, context):
        return role_poll(context, "SOURCE")

    def draw(self, context):
        layout = _layout(self)
        settings = _settings(context)
        layout.prop(settings, "emission_mode")
        layout.prop(settings, "source_sampling_mode")
        if settings.source_sampling_mode == "MULTIPLIER":
            layout.prop(settings, "source_sampling_multiplier")
        elif settings.source_sampling_mode == "LEGACY_ABSOLUTE":
            layout.prop(settings, "emit_rate")
        if canonical_region_kind(context.object) == "POINTS":
            layout.prop(settings, "auto_inject_seed")
        layout.prop(settings, "initial_velocity")
        layout.prop(settings, "use_object_velocity")
        draw_property_grid(
            layout,
            settings,
            ("temperature", "viscosity", "density", "source_color"),
        )
        layout.prop(settings, "use_lifespan")
        if canonical_region_kind(context.object) == "POINTS":
            layout.operator("fluidsim.inject_particles", text="Inject Active Object Now", icon="PARTICLE_DATA")


class FLUIDSIM_PT_object_lifespan(FluidSimObjectPanel, bpy.types.Panel):
    bl_label = "Source Lifespan"
    bl_idname = "FLUIDSIM_PT_object_lifespan"
    bl_parent_id = "FLUIDSIM_PT_object_source"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        settings = _settings(context)
        return role_poll(context, "SOURCE") and bool(getattr(settings, "use_lifespan", False))

    def draw(self, context):
        layout = _layout(self)
        settings = _settings(context)
        layout.prop(settings, "use_lifespan")
        layout.prop(settings, "life_min")
        layout.prop(settings, "life_max")


class FLUIDSIM_PT_object_blending(FluidSimObjectPanel, bpy.types.Panel):
    bl_label = "Field Blending"
    bl_idname = "FLUIDSIM_PT_object_blending"
    bl_order = 30
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return role_poll(context, "FIELD")

    def draw(self, context):
        layout = _layout(self)
        settings = _settings(context)
        layout.prop(settings, "blend_mode")
        if settings.blend_mode in {"ADD", "PULL", "BLEND"}:
            layout.prop(settings, "field_amount")


class FLUIDSIM_PT_object_collider(FluidSimObjectPanel, bpy.types.Panel):
    bl_label = "Collider"
    bl_idname = "FLUIDSIM_PT_object_collider"
    bl_order = 40

    @classmethod
    def poll(cls, context):
        return role_poll(context, "COLLIDER")

    def draw(self, context):
        layout = _layout(self)
        draw_property_grid(
            layout,
            _settings(context),
            ("collider_friction", "collider_restitution", "transfer_velocity"),
        )


class FLUIDSIM_PT_object_force(FluidSimObjectPanel, bpy.types.Panel):
    bl_label = "Force Region"
    bl_idname = "FLUIDSIM_PT_object_force"
    bl_order = 50

    @classmethod
    def poll(cls, context):
        return role_poll(context, "FORCE")

    def draw(self, context):
        layout = _layout(self)
        settings = _settings(context)
        layout.prop(settings, "force_type")
        if settings.force_type != "OFF":
            layout.prop(settings, "force_strength")
        if settings.force_type in {"DIRECTIONAL", "SPIN"}:
            layout.prop(settings, "force_direction")


class FLUIDSIM_PT_object_field(FluidSimObjectPanel, bpy.types.Panel):
    bl_label = "Field Source"
    bl_idname = "FLUIDSIM_PT_object_field"
    bl_order = 60

    @classmethod
    def poll(cls, context):
        return role_poll(context, "FIELD")

    def draw(self, context):
        layout = _layout(self)
        settings = _settings(context)
        layout.prop(settings, "field_operation")
        layout.prop(settings, "field_value_source")
        if settings.field_value_source == "CONSTANT":
            if settings.field_operation in {"PRESSURE", "SCALAR"}:
                layout.prop(settings, "field_scalar_value")
            else:
                layout.prop(settings, "field_vector_value")
        else:
            layout.label(text="Payload is read from the Volume Grid panel.", icon="VOLUME_DATA")


class FLUIDSIM_PT_object_volume(FluidSimObjectPanel, bpy.types.Panel):
    bl_label = "Volume Grid"
    bl_idname = "FLUIDSIM_PT_object_volume"
    bl_order = 70

    @classmethod
    def poll(cls, context):
        return volume_grid_poll(context)

    def draw(self, context):
        layout = _layout(self)
        settings = _settings(context)
        role = active_object_role(context)
        layout.prop(settings, "volume_input_object")
        layout.prop(settings, "volume_grid_type")
        if settings.volume_grid_type == "FOG":
            layout.prop(settings, "volume_mask_grid")
            layout.prop(settings, "volume_fog_threshold")
        else:
            layout.prop(settings, "volume_sdf_grid")
        if role == "FIELD" and settings.field_value_source == "VOLUME_GRID":
            if settings.field_operation in {"PRESSURE", "SCALAR"}:
                layout.prop(settings, "volume_scalar_grid")
            else:
                layout.prop(settings, "volume_vector_grid")
                if not settings.volume_vector_grid:
                    layout.prop(settings, "volume_vector_x_grid")
                    layout.prop(settings, "volume_vector_y_grid")
                    layout.prop(settings, "volume_vector_z_grid")


OBJECT_PANEL_CLASSES = (
    FLUIDSIM_PT_object_role,
    FLUIDSIM_PT_object_status,
    FLUIDSIM_PT_object_region,
    FLUIDSIM_PT_object_source,
    FLUIDSIM_PT_object_lifespan,
    FLUIDSIM_PT_object_blending,
    FLUIDSIM_PT_object_collider,
    FLUIDSIM_PT_object_force,
    FLUIDSIM_PT_object_field,
    FLUIDSIM_PT_object_volume,
)


def register():
    for cls in OBJECT_PANEL_CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(OBJECT_PANEL_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:  # noqa: BLE001 - report and keep walking
            print(f"FluidSim: could not unregister {cls.__name__}: {exc!r}")
