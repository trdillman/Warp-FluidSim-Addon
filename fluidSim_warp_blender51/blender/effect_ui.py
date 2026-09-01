"""Artist-facing Custom Effects stack and transactional workflow UI."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from . import effect_library, effects
from ..sim.effects.enums import ParameterKind, TrustState
from ..sim.effects.fingerprint import execution_stack_fingerprint
from ..sim.effects.schema import EffectInstanceV2


_STATUS_ICONS = {
    "APPLIED": "CHECKMARK",
    "TESTED": "FILE_TICK",
    "DRAFT": "GREASEPENCIL",
    "DIRTY": "ERROR",
    "INVALID": "CANCEL",
    "UNTRUSTED": "LOCKED",
    "RUNTIME_ERROR": "ERROR",
    "DISABLED": "HIDE_ON",
}

_LAST_LIBRARY_FATAL = ""


def _active_effect(scene):
    stack = getattr(scene, "fluidsim_effects", None)
    if stack is None or not stack.items:
        return None
    index = max(0, min(int(stack.active_index), len(stack.items) - 1))
    return stack.items[index]


def _normalize_orders(stack) -> None:
    for index, item in enumerate(stack.items):
        item.order = index


def _definition_from_pg(item):
    from ..sim.effects.schema import EffectDefinitionV2

    return EffectDefinitionV2.from_dict(json.loads(item.definition_json))


def _effect_items(_self, _context):
    global _LAST_LIBRARY_FATAL
    try:
        definitions = effect_library.library_snapshot().library.definitions()
        _LAST_LIBRARY_FATAL = ""
    except Exception as exc:
        _LAST_LIBRARY_FATAL = f"{type(exc).__name__}: {exc}"
        definitions = effect_library.builtin_library().definitions()
    return [
        (
            definition.effect_id,
            definition.name,
            f"{definition.summary}  Cost: {definition.cost_hint.title()}",
        )
        for definition in definitions
    ]


def _compile_device(context, explicit: str = "") -> str:
    if explicit:
        return explicit
    override = os.environ.get("FLUIDSIM_EFFECT_VALIDATION_DEVICE", "").strip()
    if override:
        return override
    try:
        from ..preferences import configured_device

        device = str(configured_device() or "").strip()
        if device:
            return device
    except Exception:
        pass
    if os.environ.get("FLUIDSIM_ALLOW_EFFECT_CPU_VALIDATION") == "1":
        return "cpu"
    raise RuntimeError(
        "No CUDA device is selected. Select a CUDA device in FluidSim Preferences. "
        "CPU is available only for explicit development validation."
    )


def _report_candidate(operator, candidate, success_message: str):
    if candidate.success:
        operator.report({"INFO"}, success_message)
        return {"FINISHED"}
    issue = candidate.issues[0] if candidate.issues else None
    message = issue.message if issue is not None else "Effect validation failed"
    if issue is not None and issue.line:
        message = f"Line {issue.line}: {message}"
    operator.report({"ERROR"}, message[:240])
    return {"CANCELLED"}


def _draw_parameter(layout, parameter):
    row = layout.row(align=True)
    label = parameter.label or parameter.key.replace("_", " ").title()
    kind = ParameterKind(parameter.kind)
    if kind is ParameterKind.FLOAT:
        row.prop(parameter, "float_value", text=label)
    elif kind is ParameterKind.INT:
        row.prop(parameter, "int_value", text=label)
    elif kind is ParameterKind.BOOL:
        row.prop(parameter, "bool_value", text=label)
    elif kind is ParameterKind.VECTOR:
        row.prop(parameter, "vector_value", text=label)
    elif kind is ParameterKind.COLOR:
        row.prop(parameter, "color_value", text=label)
    elif kind is ParameterKind.ENUM:
        row.prop(parameter, "enum_value", text=label)
    elif kind is ParameterKind.OBJECT:
        row.prop(parameter, "object_value", text=label)
        if parameter.object_value is None:
            op = row.operator("fluidsim.effect_create_controller", text="", icon="EMPTY_AXIS")
            op.parameter_key = parameter.key
        else:
            op = row.operator("fluidsim.effect_select_controller", text="", icon="RESTRICT_SELECT_OFF")
            op.parameter_key = parameter.key
    if parameter.unit and parameter.unit not in {"NONE", "OBJECT"}:
        row.label(text=parameter.unit)
    if parameter.description:
        help_box = layout.box()
        help_box.label(text=parameter.description[:120], icon="INFO")


def _draw_artist_parameters(layout, item, *, boxed=True):
    """Render artist intent only; raw kernel plumbing lives in Warp Code."""

    definition = _definition_from_pg(item)
    box = layout.box() if boxed else layout
    box.label(text=definition.summary, icon="FORCE_VORTEX")
    if definition.description:
        box.label(text=definition.description[:120])
    for parameter in item.parameters:
        _draw_parameter(box, parameter)
    timing = box.row(align=True)
    timing.prop(item, "frame_start", text="Start")
    timing.prop(item, "frame_end", text="End")
    timing.prop(item, "frame_step", text="Every")


def _draw_status(layout, scene, stack, item):
    box = layout.box()
    icon = _STATUS_ICONS.get(item.status, "INFO")
    box.label(text=f"Status: {item.status.replace('_', ' ').title()}", icon=icon)
    try:
        live_signature, _inventory = execution_stack_fingerprint(
            effects.scene_effect_instances(scene)
        )
        live_error = ""
    except Exception as exc:
        live_signature = ""
        live_error = f"{type(exc).__name__}: {exc}"
    live_dirty = bool(live_error or live_signature != stack.applied_signature)
    if live_dirty:
        box.label(text="Live draft differs from applied effect", icon="ERROR")
    elif item.status == "DIRTY":
        box.label(text="Draft differs from applied effect", icon="ERROR")
    if live_error:
        box.label(text=live_error[:160], icon="CANCEL")
    if item.trust_state not in {
        TrustState.BUILT_IN_TRUSTED.value,
        TrustState.LOCAL_TRUSTED.value,
    }:
        box.label(text="Executable Warp code is disabled until its exact source is trusted.", icon="LOCKED")
        op = box.operator("fluidsim.effect_trust", text="Review & Trust", icon="UNLOCKED")
        op.index = int(stack.active_index)
    elif item.trust_state == TrustState.LOCAL_TRUSTED.value:
        op = box.operator("fluidsim.effect_untrust", text="Revoke Local Trust", icon="LOCKED")
        op.index = int(stack.active_index)
    if stack.blocking_reason:
        box.label(text="Production Bake blocked", icon="CANCEL")
        box.label(text=stack.blocking_reason[:160])
    else:
        box.label(text="Production Bake effect identity is ready", icon="CHECKMARK")
    if stack.applied_signature:
        box.label(text=f"Applied SHA-256: {stack.applied_signature[:16]}…")
    box.operator("fluidsim.effect_refresh_status", text="Refresh Status", icon="FILE_REFRESH")


def _draw_library_diagnostics(layout):
    diagnostics = ()
    fatal = _LAST_LIBRARY_FATAL
    try:
        diagnostics = effect_library.library_diagnostics()
    except Exception as exc:
        fatal = f"{type(exc).__name__}: {exc}"
    if not fatal and not diagnostics:
        return
    box = layout.box()
    box.label(text="Installed package problem", icon="ERROR")
    if fatal:
        box.label(text=fatal[:180])
    for diagnostic in diagnostics[:4]:
        source = Path(diagnostic.source).name or diagnostic.source
        box.label(text=f"{source}: {diagnostic.message}"[:180])
    if len(diagnostics) > 4:
        box.label(text=f"{len(diagnostics) - 4} additional problem(s)")
    box.operator("fluidsim.effect_reveal_library", text="Reveal Effect Library", icon="FILE_FOLDER")


class FLUIDSIM_UL_effect_stack(bpy.types.UIList):
    """Compact ordered Custom Effects stack."""

    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        mute_icon = "HIDE_ON" if item.muted else "HIDE_OFF"
        row.prop(item, "muted", text="", icon=mute_icon, emboss=False)
        solo_icon = "SOLO_ON" if item.solo else "SOLO_OFF"
        row.prop(item, "solo", text="", icon=solo_icon, emboss=False)
        row.label(text=item.display_name, icon=_STATUS_ICONS.get(item.status, "INFO"))
        try:
            cost = _definition_from_pg(item).cost_hint.title()
        except Exception:
            cost = "Unknown"
        row.label(text=cost)


class FLUIDSIM_OT_effect_add(bpy.types.Operator):
    bl_idname = "fluidsim.effect_add"
    bl_label = "Add Custom Effect"
    bl_description = "Add an artist-ready effect from the built-in or installed library"
    bl_options = {"REGISTER", "UNDO"}

    effect_id: EnumProperty(name="Effect", items=_effect_items)

    def invoke(self, context, _event):
        if not self.effect_id:
            self.effect_id = context.scene.fluidsim_effects.selected_library_effect
        return context.window_manager.invoke_search_popup(self)

    def execute(self, context):
        definition = effect_library.combined_library().get(self.effect_id)
        instance = EffectInstanceV2(
            instance_id=f"effect-{uuid.uuid4().hex[:12]}",
            definition=definition,
            order=len(context.scene.fluidsim_effects.items),
            display_name=definition.name,
        )
        effects.add_instance_to_scene(context.scene, instance)
        context.scene.fluidsim_effects.selected_library_effect = definition.effect_id
        self.report({"INFO"}, f"Added {definition.name}. Set parameters, then Compile & Apply.")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_remove(bpy.types.Operator):
    bl_idname = "fluidsim.effect_remove"
    bl_label = "Remove Effect"
    bl_description = "Remove the selected effect from the stack"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(default=-1)

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        stack = getattr(scene, "fluidsim_effects", None)
        if stack is None or not len(stack.items):
            cls.poll_message_set("Select an effect first")
            return False
        return True

    def execute(self, context):
        stack = context.scene.fluidsim_effects
        index = self.index if self.index >= 0 else int(stack.active_index)
        if index < 0 or index >= len(stack.items):
            self.report({"WARNING"}, "Select an effect first")
            return {"CANCELLED"}
        item = stack.items[index]
        text_names = [kernel.source_text.name for kernel in item.kernels if kernel.source_text]
        stack.items.remove(index)
        for name in text_names:
            text = bpy.data.texts.get(name)
            if text is not None and text.users <= 1:
                bpy.data.texts.remove(text)
        stack.active_index = max(0, min(index, len(stack.items) - 1))
        _normalize_orders(stack)
        effects.refresh_stack_state(context.scene)
        return {"FINISHED"}


class FLUIDSIM_OT_effect_duplicate(bpy.types.Operator):
    bl_idname = "fluidsim.effect_duplicate"
    bl_label = "Duplicate Effect"
    bl_description = "Duplicate the active effect as a new entry at the end of the stack"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        item = _active_effect(context.scene)
        if item is None:
            return {"CANCELLED"}
        instance = effects.effect_instance_from_pg(context.scene, item)
        duplicate = replace(
            instance,
            instance_id=f"effect-{uuid.uuid4().hex[:12]}",
            display_name=f"{instance.display_name or instance.definition.name} Copy",
            order=len(context.scene.fluidsim_effects.items),
            solo=False,
        )
        effects.add_instance_to_scene(context.scene, duplicate)
        return {"FINISHED"}


class FLUIDSIM_OT_effect_move(bpy.types.Operator):
    bl_idname = "fluidsim.effect_move"
    bl_label = "Move Effect"
    bl_description = "Move the active effect up or down to change its evaluation order"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(items=(("UP", "Up", ""), ("DOWN", "Down", "")), default="UP")

    def execute(self, context):
        stack = context.scene.fluidsim_effects
        index = int(stack.active_index)
        target = index - 1 if self.direction == "UP" else index + 1
        if target < 0 or target >= len(stack.items):
            return {"CANCELLED"}
        stack.items.move(index, target)
        stack.active_index = target
        _normalize_orders(stack)
        effects.refresh_stack_state(context.scene)
        return {"FINISHED"}


class FLUIDSIM_OT_effect_toggle_mute(bpy.types.Operator):
    bl_idname = "fluidsim.effect_toggle_mute"
    bl_label = "Mute or Unmute Effect"
    bl_description = "Mute or unmute the active effect without removing it from the stack"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        item = _active_effect(context.scene)
        if item is None:
            return {"CANCELLED"}
        item.muted = not item.muted
        effects.refresh_stack_state(context.scene)
        return {"FINISHED"}


class FLUIDSIM_OT_effect_toggle_solo(bpy.types.Operator):
    bl_idname = "fluidsim.effect_toggle_solo"
    bl_label = "Solo or Unsolo Effect"
    bl_description = "Solo the active effect so only it evaluates, or clear solo to evaluate the whole stack"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        item = _active_effect(context.scene)
        if item is None:
            return {"CANCELLED"}
        item.solo = not item.solo
        effects.refresh_stack_state(context.scene)
        return {"FINISHED"}


class _EffectDeviceOperator:
    device: StringProperty(
        name="Validation Device",
        description="Explicit Warp device for automated validation; normal UI uses the selected CUDA device",
        options={"HIDDEN"},
    )

    def effect_device(self, context):
        return _compile_device(context, self.device)


class FLUIDSIM_OT_effect_test_draft(_EffectDeviceOperator, bpy.types.Operator):
    bl_idname = "fluidsim.effect_test_draft"
    bl_label = "Test Draft"
    bl_description = "Validate, JIT compile, and run bounded sandbox launches for the complete enabled stack"

    def execute(self, context):
        try:
            candidate = effects.test_draft(context.scene, device=self.effect_device(context))
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:240])
            return {"CANCELLED"}
        return _report_candidate(self, candidate, "Draft compiled and passed guarded sandbox launches")


class FLUIDSIM_OT_effect_apply_stack(_EffectDeviceOperator, bpy.types.Operator):
    bl_idname = "fluidsim.effect_apply_stack"
    bl_label = "Compile & Apply"
    bl_description = "Compile the complete draft and atomically replace the applied runtime stack"

    def execute(self, context):
        try:
            candidate = effects.apply_stack(context.scene, device=self.effect_device(context))
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:240])
            return {"CANCELLED"}
        return _report_candidate(self, candidate, "Custom Effects compiled, sandboxed, and applied atomically")


class FLUIDSIM_OT_effect_revert_draft(bpy.types.Operator):
    bl_idname = "fluidsim.effect_revert_draft"
    bl_label = "Revert Draft"
    bl_description = "Restore the exact last applied effect stack"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            effects.revert_draft(context.scene)
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:240])
            return {"CANCELLED"}
        self.report({"INFO"}, "Draft restored to the last applied stack")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_preview_one_frame(bpy.types.Operator):
    bl_idname = "fluidsim.effect_preview_one_frame"
    bl_label = "Preview One Frame"
    bl_description = "Advance the live solver by one frame using the exact applied effect stack"

    def execute(self, context):
        reason = effects.production_blocking_reason(context.scene)
        if reason:
            self.report({"ERROR"}, reason[:240])
            return {"CANCELLED"}
        from .scene_bridge import session

        if not session().owns_scene(context.scene):
            self.report({"ERROR"}, "Initialize the solver before previewing an effect")
            return {"CANCELLED"}
        result = bpy.ops.fluidsim.step(frames=1)
        return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}


class FLUIDSIM_OT_effect_create_controller(bpy.types.Operator):
    bl_idname = "fluidsim.effect_create_controller"
    bl_label = "Create Effect Controller"
    bl_description = "Create an empty object to drive this effect's object parameter"
    bl_options = {"REGISTER", "UNDO"}

    parameter_key: StringProperty(default="controller")

    def execute(self, context):
        item = _active_effect(context.scene)
        if item is None:
            return {"CANCELLED"}
        parameter = next((entry for entry in item.parameters if entry.key == self.parameter_key), None)
        if parameter is None or parameter.kind != ParameterKind.OBJECT.value:
            self.report({"ERROR"}, "The selected effect has no matching object controller")
            return {"CANCELLED"}
        name = f"FluidSim {item.display_name} Controller"
        controller = bpy.data.objects.new(name, None)
        controller.empty_display_type = "CIRCLE" if "vortex" in item.effect_id else "SPHERE"
        controller.empty_display_size = 1.0
        controller["fluidsim_owned"] = True
        controller["fluidsim_effect_controller"] = item.instance_id
        controller["fluidsim_effect_parameter"] = parameter.key
        target_collection = context.collection or context.scene.collection
        target_collection.objects.link(controller)
        if context.object is not None:
            controller.matrix_world.translation = context.object.matrix_world.translation
        parameter.object_value = controller
        bpy.ops.object.select_all(action="DESELECT")
        controller.select_set(True)
        context.view_layer.objects.active = controller
        effects.refresh_stack_state(context.scene)
        self.report({"INFO"}, f"Created {controller.name}; move or animate it in the scene")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_select_controller(bpy.types.Operator):
    bl_idname = "fluidsim.effect_select_controller"
    bl_label = "Select Effect Controller"
    bl_description = "Select the empty object that drives this effect's object parameter"

    parameter_key: StringProperty(default="controller")

    def execute(self, context):
        item = _active_effect(context.scene)
        if item is None:
            return {"CANCELLED"}
        parameter = next((entry for entry in item.parameters if entry.key == self.parameter_key), None)
        controller = parameter.object_value if parameter is not None else None
        if controller is None:
            self.report({"ERROR"}, "No controller is assigned")
            return {"CANCELLED"}
        bpy.ops.object.select_all(action="DESELECT")
        controller.hide_set(False)
        controller.select_set(True)
        context.view_layer.objects.active = controller
        return {"FINISHED"}


class FLUIDSIM_OT_effect_trust(bpy.types.Operator):
    bl_idname = "fluidsim.effect_trust"
    bl_label = "Review & Trust Custom Code"
    bl_description = "Trust this exact SHA-256 source content on this machine"

    index: IntProperty(default=-1)
    acknowledge: BoolProperty(
        name="I understand this Warp code executes with my Blender process",
        default=False,
    )

    def invoke(self, context, _event):
        if self.index < 0:
            self.index = int(context.scene.fluidsim_effects.active_index)
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, _context):
        layout = self.layout
        layout.label(text="Custom Warp code is trusted local executable code.", icon="ERROR")
        layout.label(text="Trust is bound to the exact SHA-256 content and is revoked after edits.")
        layout.prop(self, "acknowledge")

    def execute(self, context):
        if not self.acknowledge:
            self.report({"ERROR"}, "Review and acknowledge the executable-code warning")
            return {"CANCELLED"}
        try:
            effects.trust_effect(context.scene, self.index)
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:240])
            return {"CANCELLED"}
        self.report({"INFO"}, "Trusted this exact source digest locally; compile before use")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_untrust(bpy.types.Operator):
    bl_idname = "fluidsim.effect_untrust"
    bl_label = "Revoke Effect Trust"

    index: IntProperty(default=-1)

    def execute(self, context):
        index = self.index if self.index >= 0 else int(context.scene.fluidsim_effects.active_index)
        effects.untrust_effect(context.scene, index)
        self.report({"INFO"}, "Local trust revoked; effect execution is blocked")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_refresh_status(bpy.types.Operator):
    bl_idname = "fluidsim.effect_refresh_status"
    bl_label = "Refresh Custom Effect Status"
    bl_description = "Recompute draft, controller, trust, and applied identity state"

    def execute(self, context):
        try:
            effects.refresh_stack_state(context.scene)
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:240])
            return {"CANCELLED"}
        self.report({"INFO"}, "Custom Effect status refreshed")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_reveal_library(bpy.types.Operator):
    bl_idname = "fluidsim.effect_reveal_library"
    bl_label = "Reveal Effect Library"
    bl_description = "Open the extension-managed Custom Effect package directory"

    def execute(self, _context):
        path = effect_library.reveal_path()
        result = bpy.ops.wm.path_open(filepath=str(path))
        return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}


class FLUIDSIM_OT_effect_delete_installed(bpy.types.Operator):
    bl_idname = "fluidsim.effect_delete_installed"
    bl_label = "Remove Installed Effect Package"
    bl_description = "Remove this effect from the local package library; scene instances remain embedded"

    def invoke(self, context, _event):
        return context.window_manager.invoke_confirm(self, _event)

    def execute(self, context):
        item = _active_effect(context.scene)
        if item is None:
            return {"CANCELLED"}
        try:
            definition = _definition_from_pg(item)
            if definition.built_in:
                raise ValueError("built-in effects cannot be removed from the package library")
            removed = effect_library.delete_definition(definition.effect_id)
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:240])
            return {"CANCELLED"}
        self.report({"INFO"}, f"Removed {len(removed)} installed package(s); scene copy retained")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_import_package(bpy.types.Operator):
    bl_idname = "fluidsim.effect_import_package"
    bl_label = "Import Effect Package"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.fseffect", options={"HIDDEN"})
    conflict: EnumProperty(
        items=(("cancel", "Cancel on Conflict", ""), ("keep_both", "Keep Both", ""), ("replace", "Replace Installed", "")),
        default="cancel",
    )

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            report = effect_library.install_package(self.filepath, conflict=self.conflict)
            from ..sim.effects.package import read_effect_package
            definition = read_effect_package(report.path)
            instance = EffectInstanceV2(
                instance_id=f"effect-{uuid.uuid4().hex[:12]}",
                definition=definition,
                enabled=False,
                order=len(context.scene.fluidsim_effects.items),
                display_name=definition.name,
            )
            effects.add_instance_to_scene(context.scene, instance)
        except Exception as exc:
            self.report({"ERROR"}, f"Package import failed: {exc}"[:240])
            return {"CANCELLED"}
        self.report({"INFO"}, "Imported package as disabled, untrusted code; review before enabling")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_compare_update(bpy.types.Operator):
    bl_idname = "fluidsim.effect_compare_update"
    bl_label = "Compare Update"
    bl_description = "Compare an Effect Package with the installed package of the same identity"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.fseffect", options={"HIDDEN"})

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, _context):
        try:
            comparison = effect_library.compare_package_update(self.filepath)
        except Exception as exc:
            self.report({"ERROR"}, f"Update comparison failed: {exc}"[:240])
            return {"CANCELLED"}
        versions = comparison.candidate_version
        if comparison.installed_version:
            versions = f"installed {comparison.installed_version}, candidate {versions}"
        self.report(
            ({"WARNING"} if comparison.status in {
                "MODIFIED", "OLDER", "AMBIGUOUS", "BUILTIN_CONFLICT"
            } else {"INFO"}),
            f"{comparison.status}: {versions}. {comparison.message}"[:240],
        )
        return {"FINISHED"}


class FLUIDSIM_OT_effect_rename_installed(bpy.types.Operator):
    bl_idname = "fluidsim.effect_rename_installed"
    bl_label = "Rename Package"
    bl_description = (
        "Give the installed package a new effect identity and display name; "
        "the embedded scene instance remains unchanged"
    )

    new_effect_id: StringProperty(name="Effect ID")
    new_name: StringProperty(name="Display Name")

    def invoke(self, context, _event):
        item = _active_effect(context.scene)
        if item is None:
            return {"CANCELLED"}
        try:
            definition = _definition_from_pg(item)
            if definition.built_in:
                raise ValueError("built-in effects cannot be renamed")
            effect_library.installed_package_for_effect(definition.effect_id)
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:240])
            return {"CANCELLED"}
        self.new_effect_id = definition.effect_id
        self.new_name = definition.name
        return context.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "new_effect_id")
        layout.prop(self, "new_name")
        box = layout.box()
        box.label(text="Only the installed library package is renamed.", icon="INFO")
        box.label(text="Existing scene instances remain embedded under their current identity.")

    def execute(self, context):
        item = _active_effect(context.scene)
        if item is None:
            return {"CANCELLED"}
        try:
            definition = _definition_from_pg(item)
            if definition.built_in:
                raise ValueError("built-in effects cannot be renamed")
            installed = effect_library.installed_package_for_effect(definition.effect_id)
            renamed = effect_library.rename_installed_package(
                installed.path,
                new_effect_id=self.new_effect_id,
                new_name=self.new_name,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Package rename failed: {exc}"[:240])
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Renamed installed package to {renamed.definition.name}; scene copy remains unchanged",
        )
        return {"FINISHED"}


class FLUIDSIM_OT_effect_export_package(bpy.types.Operator):
    bl_idname = "fluidsim.effect_export_package"
    bl_label = "Export Effect Package"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.fseffect", options={"HIDDEN"})

    def invoke(self, context, _event):
        item = _active_effect(context.scene)
        if item is not None and not self.filepath:
            safe = item.effect_id.replace(".", "_")
            self.filepath = f"{safe}-{item.effect_version}.fseffect"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        item = _active_effect(context.scene)
        if item is None:
            return {"CANCELLED"}
        try:
            definition = effects.effect_instance_from_pg(context.scene, item).definition
            path = Path(self.filepath)
            if path.suffix != ".fseffect":
                path = path.with_suffix(".fseffect")
            effect_library.export_definition(path, replace(definition, built_in=False))
        except Exception as exc:
            self.report({"ERROR"}, f"Package export failed: {exc}"[:240])
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {path.name}")
        return {"FINISHED"}


def _draw_effect_stack(layout, stack) -> None:
    box = layout.box()
    header = box.row(align=True)
    header.label(text="Effect Stack", icon="MODIFIER")
    header.operator("fluidsim.effect_add", text="Add Effect", icon="ADD")

    row = box.row()
    row.template_list(
        "FLUIDSIM_UL_effect_stack", "", stack, "items", stack, "active_index", rows=5
    )
    actions = row.column(align=True)
    actions.operator("fluidsim.effect_remove", text="", icon="REMOVE")
    actions.operator("fluidsim.effect_duplicate", text="", icon="DUPLICATE")
    up = actions.operator("fluidsim.effect_move", text="", icon="TRIA_UP")
    up.direction = "UP"
    down = actions.operator("fluidsim.effect_move", text="", icon="TRIA_DOWN")
    down.direction = "DOWN"
    actions.operator("fluidsim.effect_toggle_mute", text="", icon="HIDE_ON")
    actions.operator("fluidsim.effect_toggle_solo", text="", icon="SOLO_ON")


def _draw_selected_effect(layout, item) -> None:
    box = layout.box()
    box.label(text="Selected Effect", icon="FORCE_VORTEX")
    try:
        _draw_artist_parameters(box, item, boxed=False)
    except Exception as exc:
        error = box.box()
        error.label(text="Embedded effect definition is invalid", icon="CANCEL")
        error.label(text=f"{type(exc).__name__}: {exc}"[:180])


def _draw_effect_workflow(layout) -> None:
    box = layout.box()
    box.label(text="Workflow", icon="PLAY")
    row = box.row(align=True)
    row.operator("fluidsim.effect_test_draft", text="Test Draft", icon="EXPERIMENTAL")
    row.operator("fluidsim.effect_apply_stack", text="Compile & Apply", icon="CHECKMARK")
    row.operator("fluidsim.effect_revert_draft", text="Revert Draft", icon="LOOP_BACK")
    box.operator("fluidsim.effect_preview_one_frame", text="Preview One Frame", icon="PLAY")


class FLUIDSIM_PT_custom_effects(bpy.types.Panel):
    bl_label = "Custom Effects"
    bl_idname = "FLUIDSIM_PT_custom_effects"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_order = 50
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return getattr(getattr(context, "scene", None), "fluidsim_effects", None) is not None

    def draw(self, context):
        layout = self.layout
        stack = context.scene.fluidsim_effects
        _draw_effect_stack(layout, stack)

        item = _active_effect(context.scene)
        if item is None:
            box = layout.box()
            box.label(text="Add an effect, adjust its controls, then Test Draft and Compile & Apply.", icon="INFO")
            return

        _draw_selected_effect(layout, item)
        _draw_effect_workflow(layout)
        _draw_status(layout, context.scene, stack, item)


class FLUIDSIM_PT_effect_packages(bpy.types.Panel):
    bl_label = "Effect Packages"
    bl_idname = "FLUIDSIM_PT_effect_packages"
    bl_parent_id = "FLUIDSIM_PT_custom_effects"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return getattr(getattr(context, "scene", None), "fluidsim_effects", None) is not None

    def draw(self, context):
        layout = self.layout
        _draw_library_diagnostics(layout)

        row = layout.row(align=True)
        row.operator("fluidsim.effect_import_package", text="Import Package", icon="IMPORT")
        row.operator("fluidsim.effect_compare_update", text="Compare Update", icon="FILE_REFRESH")
        layout.operator("fluidsim.effect_reveal_library", text="Reveal Effect Library", icon="FILE_FOLDER")

        item = _active_effect(context.scene)
        if item is None:
            layout.label(text="Select an effect to export or manage its installed package.", icon="INFO")
            return

        layout.operator("fluidsim.effect_export_package", text="Export Selected Effect", icon="EXPORT")
        try:
            definition = _definition_from_pg(item)
        except Exception:
            definition = None
        if definition is not None and not definition.built_in:
            try:
                effect_library.installed_package_for_effect(definition.effect_id)
                installed = True
            except (FileNotFoundError, ValueError):
                installed = False
            if installed:
                package_actions = layout.row(align=True)
                package_actions.operator(
                    "fluidsim.effect_rename_installed",
                    text="Rename Package",
                    icon="GREASEPENCIL",
                )
                package_actions.operator(
                    "fluidsim.effect_delete_installed",
                    text="Remove Package",
                    icon="TRASH",
                )
            else:
                layout.label(
                    text="Scene effect is embedded; no installed library package",
                    icon="INFO",
                )


CLASSES = (
    FLUIDSIM_UL_effect_stack,
    FLUIDSIM_OT_effect_add,
    FLUIDSIM_OT_effect_remove,
    FLUIDSIM_OT_effect_duplicate,
    FLUIDSIM_OT_effect_move,
    FLUIDSIM_OT_effect_toggle_mute,
    FLUIDSIM_OT_effect_toggle_solo,
    FLUIDSIM_OT_effect_test_draft,
    FLUIDSIM_OT_effect_apply_stack,
    FLUIDSIM_OT_effect_revert_draft,
    FLUIDSIM_OT_effect_preview_one_frame,
    FLUIDSIM_OT_effect_create_controller,
    FLUIDSIM_OT_effect_select_controller,
    FLUIDSIM_OT_effect_trust,
    FLUIDSIM_OT_effect_untrust,
    FLUIDSIM_OT_effect_refresh_status,
    FLUIDSIM_OT_effect_reveal_library,
    FLUIDSIM_OT_effect_delete_installed,
    FLUIDSIM_OT_effect_import_package,
    FLUIDSIM_OT_effect_compare_update,
    FLUIDSIM_OT_effect_rename_installed,
    FLUIDSIM_OT_effect_export_package,
    FLUIDSIM_PT_custom_effects,
    FLUIDSIM_PT_effect_packages,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError):
            pass
