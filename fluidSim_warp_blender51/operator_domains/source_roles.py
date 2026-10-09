"""Capability-owned Blender operators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from . import common as _runtime

_FLUIDSIM_OT_session_base = _runtime._FLUIDSIM_OT_session_base


class FLUIDSIM_OT_setup_scene(bpy.types.Operator):
    """Turn selected mesh geometry into a runnable FluidSim setup"""

    bl_idname = "fluidsim.setup_scene"
    bl_label = "Run Guided Setup"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        name="Mode",
        items=(
            ("QUICK_LIQUID", "Quick Liquid", "Selected mesh becomes an initial-fill emitter"),
            ("QUICK_CONTAINER", "Quick Container", "Selected mesh becomes a collider with a height fill"),
            ("QUICK_VISCOUS", "Quick Viscous Liquid", "Selected mesh becomes a thick-liquid emitter"),
            ("ADD_COLLIDER", "Add Collider", "Tag selected meshes as colliders"),
            ("ADD_INFLOW", "Add Inflow", "Tag selected meshes as continuous inflows"),
            ("ADD_OUTFLOW", "Add Outflow", "Tag selected meshes as liquid sinks"),
        ),
        default="QUICK_LIQUID",
        options={"SKIP_SAVE"},
    )
    preset_id: StringProperty(default="", options={"SKIP_SAVE"})
    start_preview: BoolProperty(
        name="Start Preview",
        description="Start timeline preview after the scene is configured",
        default=False,
        options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context):
        from ..blender import production_bake

        if production_bake.bake_is_active(context.scene):
            cls.poll_message_set("Production Bake owns this scene")
            return False
        selected = getattr(context, "selected_objects", ())
        if not any(getattr(obj, "type", None) == "MESH" for obj in selected):
            cls.poll_message_set("Select at least one mesh object")
            return False
        return True

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "mode")
        layout.prop(self, "start_preview")

    def execute(self, context):
        from ..blender.setup_wizard import setup_selected

        try:
            result = setup_selected(context, self.mode, self.preset_id)
        except Exception as exc:  # noqa: BLE001
            context.scene.fluidsim.status = f"Guided setup failed: {exc}"
            self.report({"ERROR"}, context.scene.fluidsim.status)
            return {"CANCELLED"}
        context.scene.fluidsim.status = result.message
        for warning in result.warnings[:3]:
            self.report({"WARNING"}, warning)
        if getattr(self, "start_preview", False):
            try:
                preview_result = bpy.ops.fluidsim.preview_start()
                if "FINISHED" not in preview_result:
                    reason = context.scene.fluidsim.status or str(preview_result)
                    prefix = "Preview could not start: "
                    if reason.startswith(prefix):
                        reason = reason[len(prefix):]
                    raise RuntimeError(reason)
            except Exception as exc:  # noqa: BLE001 - setup remains committed
                context.scene.fluidsim.status = (
                    f"Scene configured; preview could not start: {exc}"
                )
                self.report({"WARNING"}, context.scene.fluidsim.status)
                return {"FINISHED"}
        self.report({"INFO"}, result.message)
        return {"FINISHED"}
def _execute_role_tag(operator, context, role):
    from ..blender import colliders as _colliders
    from ..blender import sources as _sources

    count = 0
    skipped = 0
    for obj in context.selected_objects:
        if role == "SOURCE":
            _sources.tag_source(obj)
        elif role == "SINK":
            _sources.tag_sink(obj)
        elif role == "FORCE":
            _sources.tag_force(obj)
        elif role == "FIELD":
            _sources.tag_field(obj)
        elif role == "COLLIDER":
            if obj.type != "MESH":
                skipped += 1
                continue
            _colliders.tag_collider(obj)
        else:
            obj.fluidsim.role = "NONE"
            obj.display_type = "TEXTURED"
            obj.hide_render = False
        count += 1
    if count == 0:
        message = ("Colliders require a selected mesh object"
                   if role == "COLLIDER" else "No supported selected objects")
        operator.report({"WARNING"}, message)
        return {"CANCELLED"}
    detail = f"; skipped {skipped} non-mesh object(s)" if skipped else ""
    operator.report({"INFO"}, f"Tagged {count} object(s) as {role}{detail}")
    return {"FINISHED"}


class FLUIDSIM_OT_tag_object(bpy.types.Operator):
    """Tag the selected objects with a simulation role"""

    bl_idname = "fluidsim.tag_object"
    bl_label = "Tag Object"
    bl_options = {"REGISTER", "UNDO"}

    role: EnumProperty(
        name="Role",
        items=[
            ("SOURCE", "Liquid Emitter", "Emit liquid"),
            ("COLLIDER", "Collider", "Collide with liquid"),
            ("SINK", "Liquid Sink", "Remove liquid"),
            ("FORCE", "Force Region", "Apply an analytic force to liquid"),
            ("FIELD", "Advanced Field Source", "Source non-liquid fields"),
            ("NONE", "Clear", "Remove the FluidSim role"),
        ],
        default="SOURCE",
        description="Choose the FluidSim role assigned to every selected object; changing it changes whether the objects emit, collide, remove, force, source fields, or remain untagged",
    )

    @classmethod
    def poll(cls, context):
        if not getattr(context, "selected_objects", None):
            cls.poll_message_set("Select at least one object to tag")
            return False
        return True

    def execute(self, context):
        return _execute_role_tag(self, context, self.role)


class _FLUIDSIM_OT_role_tag_base:
    role = "NONE"

    @classmethod
    def poll(cls, context):
        selected = getattr(context, "selected_objects", None) or ()
        if not selected:
            cls.poll_message_set("Select at least one object to tag")
            return False
        if cls.role == "COLLIDER" and not any(
                getattr(obj, "type", None) == "MESH" for obj in selected):
            cls.poll_message_set("Select at least one mesh object for a collider")
            return False
        return True

    def execute(self, context):
        return _execute_role_tag(self, context, self.role)


class FLUIDSIM_OT_tag_liquid_emitter(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.tag_liquid_emitter"
    bl_label = "Tag Liquid Emitter"
    bl_options = {"REGISTER", "UNDO"}
    role = "SOURCE"


class FLUIDSIM_OT_tag_liquid_sink(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.tag_liquid_sink"
    bl_label = "Tag Liquid Sink"
    bl_options = {"REGISTER", "UNDO"}
    role = "SINK"


class FLUIDSIM_OT_tag_collider(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.tag_collider"
    bl_label = "Tag Collider"
    bl_options = {"REGISTER", "UNDO"}
    role = "COLLIDER"


class FLUIDSIM_OT_tag_force_region(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.tag_force_region"
    bl_label = "Tag Force Region"
    bl_options = {"REGISTER", "UNDO"}
    role = "FORCE"


class FLUIDSIM_OT_tag_field_source(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.tag_field_source"
    bl_label = "Tag Advanced Field Source"
    bl_options = {"REGISTER", "UNDO"}
    role = "FIELD"


class FLUIDSIM_OT_clear_object_role(_FLUIDSIM_OT_role_tag_base, bpy.types.Operator):
    bl_idname = "fluidsim.clear_object_role"
    bl_label = "Clear FluidSim Role"
    bl_options = {"REGISTER", "UNDO"}
    role = "NONE"

    @classmethod
    def poll(cls, context):
        selected = getattr(context, "selected_objects", None) or ()
        if not selected:
            cls.poll_message_set("Select at least one tagged object")
            return False
        if not any(getattr(getattr(obj, "fluidsim", None), "role", "NONE") != "NONE"
                   for obj in selected):
            cls.poll_message_set("Select at least one object with a FluidSim role")
            return False
        return True


class FLUIDSIM_OT_inject_particles(bpy.types.Operator):
    """Append mesh vertices, point-cloud points, or the first particle system once; this compatibility utility is not a volume fill"""

    bl_idname = "fluidsim.inject_particles"
    bl_label = "Inject Active Object Points Once"

    @classmethod
    def poll(cls, context):
        if not _runtime._bridge().session().active:
            cls.poll_message_set("Initialize the solver before injecting points")
            return False
        obj = getattr(context, "active_object", None)
        if obj is None:
            cls.poll_message_set("Select an active mesh, point cloud, or particle-system object")
            return False
        if (getattr(obj, "type", None) not in {"MESH", "POINTCLOUD"}
                and not getattr(obj, "particle_systems", None)):
            cls.poll_message_set(
                "Active object must provide mesh vertices, point-cloud points, or a particle system")
            return False
        return True

    def execute(self, context):
        bridge = _runtime._bridge()
        sess = bridge.session()
        if not sess.active:
            self.report({"ERROR"}, "Solver not initialized")
            return {"CANCELLED"}
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, "No active object")
            return {"CANCELLED"}
        from ..blender import output_publication
        from ..blender import sources as _sources

        scene = context.scene
        try:
            n = _sources.inject_object_particles(scene, obj, sess)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Particle injection failed: {exc}")
            bridge.set_error(scene, f"Particle injection failed: {exc}")
            scene.fluidsim.status = f"Injection failed: {exc}"
            return {"CANCELLED"}
        result = getattr(sess.solver.sourcing, "last_injection_result", None)
        if result is not None and not result.admitted:
            self.report({"WARNING"}, (
                "Particle injection needs "
                f"{result.required:,} slots but only {result.available:,} "
                "are available. Increase Max Particles or enable automatic "
                "particle-capacity growth, then retry."
            ))
            return {"CANCELLED"}
        output_publication.update_outputs(scene, sess)
        self.report({"INFO"}, f"Injected {n:,} particles from {obj.name}")
        return {"FINISHED"}
