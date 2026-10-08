"""Capability-owned Blender operators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from . import common as _runtime

_FLUIDSIM_OT_session_base = _runtime._FLUIDSIM_OT_session_base


def _preset_field_owner(scene):
    """Resolve each preset field to the settings group that declares it.

    Most preset fields feed the solver and are declared on the active domain
    settings. Output and cache-bake fields (``whitewater_output_mode``,
    ``cache_compress``, ``checkpoint_interval``) are declared on
    ``scene.fluidsim``. Writing or reading those through the domain target
    raises. ``pending_settings.declaring_group()`` holds the partition.
    """
    from ..blender.pending_settings import declaring_group

    return lambda key: declaring_group(scene, key)


class FLUIDSIM_OT_apply_product_preset(bpy.types.Operator):
    """Apply a validated built-in or user product preset"""

    bl_idname = "fluidsim.apply_product_preset"
    bl_label = "Apply FluidSim Preset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..presets import library
        from ..blender import pending_settings
        from ..blender.scene_access import active_domain_settings

        identifier = str(context.scene.fluidsim.product_preset or "")
        target = active_domain_settings(context.scene)
        if target is None:
            self.report({"ERROR"}, "Designate a FluidSim domain object before applying a preset")
            return {"CANCELLED"}
        try:
            preset = library.load_preset(identifier)
            # Write preset values to the active domain settings so the solver
            # (which builds from active_domain_settings) actually picks them up.
            result = library.apply_preset(
                context.scene, preset,
                target=target,
                field_target=_preset_field_owner(context.scene),
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Preset could not be applied: {exc}")
            return {"CANCELLED"}
        message = result.message
        # A preset marked requires_reset only takes effect after the solver
        # picks up the new values. Route through the Pending Settings state
        # machine rather than forcing a plain reset: some preset fields (e.g.
        # voxel_size, leaf_capacity) are REINITIALIZE-tier and need a full
        # rebuild, and only apply_pending_reinitialize()/apply_pending_reset()
        # complete the matching applied-state transition. A plain reset always
        # captures with reset_only=True, leaving the applied reinit signature
        # stale so REINIT_PENDING immediately reappears after a preset that
        # touched a reinit-tier field (see AGENTS.md:L21).
        if result.requires_reset:
            sess = _runtime._bridge().session()
            owns_scene = getattr(sess, "owns_scene", lambda _scene: sess.active)
            if owns_scene(context.scene):
                try:
                    pending_settings.refresh_pending_state(context.scene, sess)
                    action = pending_settings.pending_presentation(context.scene)["action"]
                    if action == "REINIT_PENDING":
                        pending_settings.apply_pending_reinitialize(context.scene)
                        message = f"{result.message} — solver reinitialized"
                    elif action == "RESET_PENDING":
                        pending_settings.apply_pending_reset(context.scene)
                        message = f"{result.message} — solver reset"
                except Exception as exc:  # noqa: BLE001
                    context.scene.fluidsim.status = result.message
                    self.report(
                        {"ERROR"},
                        f"{result.message}, but the solver could not be updated: {exc}",
                    )
                    return {"CANCELLED"}
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
        from ..blender.scene_access import active_domain_settings

        source = active_domain_settings(context.scene)
        if source is None:
            self.report({"ERROR"}, "Designate a FluidSim domain object before saving a preset")
            return {"CANCELLED"}
        try:
            preset = library.capture_scene_preset(
                context.scene,
                identifier=self.preset_id,
                name=self.preset_name,
                description=self.description,
                expected_scale=self.expected_scale,
                performance_cost=self.performance_cost,
                # Snapshot the domain settings the solver uses.
                source=source,
                field_source=_preset_field_owner(context.scene),
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
