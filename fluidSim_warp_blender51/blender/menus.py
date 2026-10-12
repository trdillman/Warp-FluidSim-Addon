"""``VIEW3D_MT_add`` entry + operator for creating a FluidSim simulation domain.

Spec O4/§7 makes a domain a first-class Add-menu citizen. The operator creates
a fresh domain object (canonical fallback geometry via
:func:`create_domain_object`) and designates it the scene's active
``fluidsim.domain_object``; its settings live on the object's
``fluidsim_domain`` group. Multiple domain objects may coexist; the
designated ``domain_object`` is the active one, so adding a domain also makes
it active.
"""

from __future__ import annotations

import bpy

# Faces of a closed box over the 8 corners emitted by ``_default_domain_geometry``.
_DEFAULT_DOMAIN_FACES = (
    (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
    (1, 5, 7, 3), (3, 7, 6, 2), (2, 6, 4, 0),
)


def _default_domain_geometry():
    """Corner verts + faces for a box spanning the canonical fallback domain.

    A freshly created domain must reproduce the bounds an unset ``domain_object``
    yields, or it would collapse ``domain_bounds`` to a degenerate AABB and
    break solver initialization.
    """
    from .solver_settings import DEFAULT_DOMAIN_MAX, DEFAULT_DOMAIN_MIN

    lo, hi = DEFAULT_DOMAIN_MIN, DEFAULT_DOMAIN_MAX
    verts = [(x, y, z)
             for x in (lo[0], hi[0])
             for y in (lo[1], hi[1])
             for z in (lo[2], hi[2])]
    return verts, list(_DEFAULT_DOMAIN_FACES)


def create_domain_object(scene):
    """Create, link and designate a fresh FluidSim domain object.

    Always creates a new object and re-designates it, so the Add menu can
    introduce an additional domain and make it the active one. Returns None
    only when ``scene.fluidsim`` is absent.
    """
    fs = getattr(scene, "fluidsim", None)
    if fs is None:
        return None
    mesh = bpy.data.meshes.new("FluidSim Domain")
    verts, faces = _default_domain_geometry()
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("FluidSim Domain", mesh)
    from .setup_wizard import present_as_domain

    present_as_domain(obj)
    scene.collection.objects.link(obj)
    fs.domain_object = obj
    return obj


class FLUIDSIM_OT_add_domain(bpy.types.Operator):
    bl_idname = "fluidsim.add_domain"
    bl_label = "Add Fluid Domain"
    bl_description = (
        "Add a FluidSim simulation domain and make it the active domain"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        obj = create_domain_object(scene)
        if obj is None:
            self.report({"ERROR"}, "No FluidSim scene settings to attach a domain to")
            return {"CANCELLED"}

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
        try:
            add_menu.remove(draw_add_menu)
        except Exception as exc:  # noqa: BLE001 - report and keep walking
            print(f"FluidSim: could not remove the Add menu entry: {exc!r}")
    try:
        bpy.utils.unregister_class(FLUIDSIM_OT_add_domain)
    except Exception as exc:  # noqa: BLE001 - report and keep walking
        print(f"FluidSim: could not unregister FLUIDSIM_OT_add_domain: {exc!r}")
