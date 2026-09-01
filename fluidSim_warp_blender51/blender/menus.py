"""``VIEW3D_MT_add`` entry + operator for creating a FluidSim simulation domain.

Spec O4/§7 makes a domain a first-class Add-menu citizen. The operator creates
a fresh domain object (canonical fallback geometry via
:func:`migration.create_domain_object`), designates it the scene's active
``fluidsim.domain_object``, and migrates the scene's settings onto the object
datablock so the solver -- which reads ``active_domain_settings`` -- sees a
consistent home immediately. Multiple domain objects may coexist; the
designated ``domain_object`` is the active one, so adding a domain also makes
it active.
"""

from __future__ import annotations

import bpy

from . import migration


class FLUIDSIM_OT_add_domain(bpy.types.Operator):
    bl_idname = "fluidsim.add_domain"
    bl_label = "Add Fluid Domain"
    bl_description = (
        "Add a FluidSim simulation domain and make it the active domain"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        obj = migration.create_domain_object(scene)
        if obj is None:
            self.report({"ERROR"}, "No FluidSim scene settings to attach a domain to")
            return {"CANCELLED"}
        migration.migrate_domain_settings_schema(scene, obj)

        # Make the freshly added domain the active/selected object, so the
        # Physics tab and the object-role panels immediately target it.
        view_layer = getattr(context, "view_layer", None)
        objects = getattr(view_layer, "objects", None)
        for other in tuple(getattr(objects, "selected", ()) or ()):
            select = getattr(other, "select_set", None)
            if callable(select):
                select(False)
        select = getattr(obj, "select_set", None)
        if callable(select):
            select(True)
        if objects is not None:
            objects.active = obj
        return {"FINISHED"}


def draw_add_menu(self, context):
    """Contribute the FluidSim domain entry to ``VIEW3D_MT_add``."""
    self.layout.operator(
        FLUIDSIM_OT_add_domain.bl_idname,
        text="Fluid Domain (FluidSim)",
        icon="MOD_FLUIDSIM",
    )


def register():
    bpy.utils.register_class(FLUIDSIM_OT_add_domain)
    add_menu = getattr(bpy.types, "VIEW3D_MT_add", None)
    if add_menu is not None:
        add_menu.append(draw_add_menu)


def unregister():
    add_menu = getattr(bpy.types, "VIEW3D_MT_add", None)
    if add_menu is not None:
        add_menu.remove(draw_add_menu)
    bpy.utils.unregister_class(FLUIDSIM_OT_add_domain)
