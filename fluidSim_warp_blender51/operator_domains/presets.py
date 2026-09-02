"""Capability-owned Blender operators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

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


def _snippets():
    return _runtime._snippets()


def _solver_settings():
    return _runtime._solver_settings()


class FLUIDSIM_OT_apply_product_preset(bpy.types.Operator):
    """Apply a validated built-in or user product preset"""

    bl_idname = "fluidsim.apply_product_preset"
    bl_label = "Apply FluidSim Preset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..presets import library
        from ..blender.scene_bridge import active_domain_settings

        identifier = str(context.scene.fluidsim.product_preset or "")
        try:
            preset = library.load_preset(identifier)
            # Write preset values to the active domain settings so the solver
            # (which builds from active_domain_settings) actually picks them up;
            # post-migration scene.fluidsim is a stale ghost copy.
            result = library.apply_preset(
                context.scene, preset, target=active_domain_settings(context.scene)
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Preset could not be applied: {exc}")
            return {"CANCELLED"}
        message = result.message
        # A preset marked requires_reset only takes effect after the solver is
        # rebuilt. Reset in place when a session owns this scene; otherwise the
        # values already sit on the domain object and apply at initialize time.
        if result.requires_reset:
            bridge = _bridge()
            sess = bridge.session()
            owns_scene = getattr(sess, "owns_scene", lambda _scene: sess.active)
            if owns_scene(context.scene):
                bridge.reset_simulation(context.scene)
                message = f"{result.message} — solver reset"
        context.scene.fluidsim.status = message
        self.report({"INFO"}, message)
        return {"FINISHED"}


class FLUIDSIM_OT_save_product_preset(bpy.types.Operator):
    """Capture allow-listed scene settings as an atomic user preset"""

    bl_idname = "fluidsim.save_product_preset"
    bl_label = "Save User Preset"
    bl_options = {"REGISTER"}

    preset_id: StringProperty(
        name="Identifier",
        default="my_water",
        description="Lowercase stable identifier using letters, numbers, and underscores",
    )
    preset_name: StringProperty(name="Name", default="My Water")
    description: StringProperty(name="Description", default="Custom FluidSim settings")
    expected_scale: StringProperty(name="Expected Scale", default="1 Blender unit = 1 metre")
    performance_cost: EnumProperty(
        name="Performance Cost",
        items=(("Low", "Low", ""), ("Medium", "Medium", ""),
               ("High", "High", ""), ("Very High", "Very High", "")),
        default="Medium",
    )
    overwrite: BoolProperty(name="Overwrite Existing", default=False)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=520)

    def execute(self, context):
        from ..presets import library
        from ..blender.scene_bridge import active_domain_settings

        try:
            preset = library.capture_scene_preset(
                context.scene,
                identifier=self.preset_id,
                name=self.preset_name,
                description=self.description,
                expected_scale=self.expected_scale,
                performance_cost=self.performance_cost,
                # Snapshot the live domain settings the solver uses, not the
                # stale scene.fluidsim ghost copy.
                source=active_domain_settings(context.scene),
            )
            path = library.save_user_preset(
                preset, overwrite=bool(self.overwrite)
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"User preset could not be saved: {exc}")
            return {"CANCELLED"}
        context.scene.fluidsim.status = f"Saved user preset: {path.name}"
        self.report({"INFO"}, context.scene.fluidsim.status)
        return {"FINISHED"}
