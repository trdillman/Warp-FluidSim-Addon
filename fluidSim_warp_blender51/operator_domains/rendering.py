"""Capability-owned Blender operators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from ..blender.scene_access import active_domain_object
from . import common as _runtime

_FLUIDSIM_OT_session_base = _runtime._FLUIDSIM_OT_session_base


def _bridge():
    return _runtime._bridge()


def _configured_device():
    return _runtime._configured_device()


def _mutation_poll(cls, context):
    return _runtime._mutation_poll(cls, context)


def _owned_capacity_plan(scene):
    return _runtime._owned_capacity_plan(scene)


def _owned_capacity_signature(scene, plan):
    return _runtime._owned_capacity_signature(scene, plan)


def _pending():
    return _runtime._pending()


def _resolved_cache_path(scene):
    return _runtime._resolved_cache_path(scene)


def _runtime_poll(cls):
    return _runtime._runtime_poll(cls)


def _scene_identity_token(scene):
    return _runtime._scene_identity_token(scene)


def _solver_settings():
    return _runtime._solver_settings()


class FLUIDSIM_OT_use_active_as_domain(bpy.types.Operator):
    """Assign the active mesh as the scene's FluidSim domain bounds object."""

    bl_idname = "fluidsim.use_active_as_domain"
    bl_label = "Use Active Object as FluidSim Domain"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "active_object", None)
        scene = getattr(context, "scene", None)
        if scene is None or getattr(scene, "fluidsim", None) is None:
            return False
        if obj is None or getattr(obj, "type", None) != "MESH":
            cls.poll_message_set("Select a mesh object to use as the FluidSim domain")
            return False
        if obj is active_domain_object(scene):
            cls.poll_message_set("The active object is already the FluidSim domain")
            return False
        from ..blender.ui_common import active_object_role

        if active_object_role(context) != "NONE":
            cls.poll_message_set(
                "Clear the object's FluidSim input role before assigning it as the domain"
            )
            return False
        return True

    def execute(self, context):
        obj = getattr(context, "active_object", None)
        scene = getattr(context, "scene", None)
        if (
            scene is None
            or getattr(scene, "fluidsim", None) is None
            or obj is None
            or getattr(obj, "type", None) != "MESH"
        ):
            self.report(
                {"ERROR"},
                "Select a mesh object to use as the FluidSim domain",
            )
            return {"CANCELLED"}
        from ..blender.ui_common import active_object_role

        if active_object_role(context) != "NONE":
            self.report(
                {"ERROR"},
                "Clear the object's FluidSim input role before using it as the domain",
            )
            return {"CANCELLED"}
        scene.fluidsim.domain_object = obj
        scene.fluidsim.status = f"FluidSim domain assigned to {obj.name}"
        self.report({"INFO"}, scene.fluidsim.status)
        return {"FINISHED"}


class FLUIDSIM_OT_clear_domain_assignment(bpy.types.Operator):
    """Clear the selected scene domain reference without deleting the object."""

    bl_idname = "fluidsim.clear_domain_assignment"
    bl_label = "Remove FluidSim Domain Assignment"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        obj = getattr(context, "active_object", None)
        return bool(
            scene is not None
            and getattr(scene, "fluidsim", None) is not None
            and obj is not None
            and obj is active_domain_object(scene)
        )

    def execute(self, context):
        scene = getattr(context, "scene", None)
        fs = getattr(scene, "fluidsim", None) if scene is not None else None
        obj = getattr(context, "active_object", None)
        if fs is None or obj is None or obj is not active_domain_object(scene):
            self.report({"ERROR"}, "The active object is no longer the FluidSim domain")
            return {"CANCELLED"}
        fs.domain_object = None
        fs.status = f"Removed FluidSim domain assignment from {obj.name}"
        self.report({"INFO"}, fs.status)
        return {"FINISHED"}


class FLUIDSIM_OT_remove_selected_output(bpy.types.Operator):
    """Remove one selected add-on-owned output after explicit confirmation."""

    bl_idname = "fluidsim.remove_selected_output"
    bl_label = "Remove Selected FluidSim Output"
    bl_options = {"REGISTER", "UNDO"}

    expected_object: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})
    expected_kind: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        from ..blender import output_publication

        obj = getattr(context, "active_object", None)
        if obj is None:
            return False
        try:
            kind = obj.get(output_publication.OUTPUT_TAG)
            owned = bool(obj.get("fluidsim_owned"))
        except AttributeError:
            return False
        return bool(
            owned
            and kind
            in {
                output_publication.SURFACE_NAME,
                output_publication.PARTICLES_NAME,
                *output_publication.WHITEWATER_OUTPUT_NAMES,
            }
        )

    def invoke(self, context, event):
        from ..blender import output_publication

        obj = getattr(context, "active_object", None)
        if obj is None:
            return {"CANCELLED"}
        self.expected_object = str(obj.name)
        self.expected_kind = str(obj.get(output_publication.OUTPUT_TAG, ""))
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        from ..blender import output_publication

        obj = getattr(context, "active_object", None)
        if obj is None:
            self.report({"ERROR"}, "No active FluidSim output")
            return {"CANCELLED"}
        current_kind = str(obj.get(output_publication.OUTPUT_TAG, ""))
        if (
            obj.name != self.expected_object
            or current_kind != self.expected_kind
            or not bool(obj.get("fluidsim_owned"))
        ):
            self.report(
                {"ERROR"},
                "Selected output changed after confirmation; review removal again",
            )
            return {"CANCELLED"}
        try:
            removed = output_publication.remove_output(
                context.scene,
                current_kind,
                expected_object=obj,
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not remove FluidSim output: {exc}")
            return {"CANCELLED"}
        if not removed:
            self.report({"ERROR"}, "The selected FluidSim output no longer exists")
            return {"CANCELLED"}
        context.scene.fluidsim.status = (
            f"Removed {self.expected_object}; Create / Repair Outputs restores it"
        )
        self.report({"INFO"}, context.scene.fluidsim.status)
        return {"FINISHED"}


class FLUIDSIM_OT_install_render_nodes(bpy.types.Operator):
    """Install or explicitly refresh bundled FluidSim render node groups"""

    bl_idname = "fluidsim.install_render_nodes"
    bl_label = "Install FluidSim Render Nodes"
    bl_options = {"REGISTER", "UNDO"}

    force: BoolProperty(
        name="Update FluidSim-owned Modifiers",
        default=False,
        options={"SKIP_SAVE"},
        description=(
            "Repoint only FluidSim-owned modifiers to the bundled revision; "
            "user node groups and user modifiers are never overwritten"
        ),
    )

    def execute(self, context):
        from ..blender import geometry_nodes, output_publication

        try:
            outputs = output_publication.ensure_output_shells(context.scene)
            result = geometry_nodes.install_render_nodes(
                context.scene,
                force=bool(self.force),
                output_objects={
                    str(obj.get(output_publication.OUTPUT_TAG, obj.name)): obj for obj in outputs
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Render-node installation failed: {exc}")
            return {"CANCELLED"}
        for warning in result.warnings[:3]:
            self.report({"WARNING"}, warning)
        context.scene.fluidsim.status = result.message
        self.report({"INFO"}, result.message)
        return {"FINISHED"}


class FLUIDSIM_OT_copy_bake_command(bpy.types.Operator):
    """Copy a shell-safe headless bake command for the current scene"""

    bl_idname = "fluidsim.copy_bake_command"
    bl_label = "Copy Headless Bake Command"

    def execute(self, context):
        import os
        import shlex
        import subprocess

        blend_path = str(bpy.data.filepath or "")
        if not blend_path:
            self.report({"ERROR"}, "Save the .blend file before copying a bake command")
            return {"CANCELLED"}
        runner = Path(__file__).resolve().parent / "release" / "run_headless_bake.py"
        fs = context.scene.fluidsim
        args = [
            bpy.app.binary_path,
            "--background",
            blend_path,
            "--python",
            str(runner),
            "--",
            "--scene",
            context.scene.name,
            "--start",
            str(fs.bake_start),
            "--end",
            str(fs.bake_end),
            "--cache",
            str(Path(bpy.path.abspath(fs.cache_path)).resolve()),
            "--device",
            _configured_device(),
            "--resume",
            "auto",
            "--checkpoint-interval",
            str(fs.checkpoint_interval),
        ]
        command = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
        context.window_manager.clipboard = command
        self.report({"INFO"}, "Headless production-bake command copied")
        return {"FINISHED"}
