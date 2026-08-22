"""Advanced Warp Code authoring workspace for Custom Effects."""

from __future__ import annotations

import json

import bpy
from bpy.props import IntProperty

from . import effects
from ..sim.effects.fields import FIELD_SPECS
from ..sim.effects.source import generate_kernel_source


def _active(scene):
    stack = scene.fluidsim_effects
    if not stack.items:
        return None, None
    index = max(0, min(int(stack.active_index), len(stack.items) - 1))
    item = stack.items[index]
    kernel = item.kernels[0] if item.kernels else None
    return item, kernel


def _text_area(context):
    """Prefer an existing Text Editor before repurposing the current area."""
    screen = getattr(context, "screen", None)
    if screen is not None:
        existing = next(
            (candidate for candidate in screen.areas if candidate.type == "TEXT_EDITOR"),
            None,
        )
        if existing is not None:
            return existing
    return getattr(context, "area", None)


class FLUIDSIM_OT_effect_open_text_editor(bpy.types.Operator):
    bl_idname = "fluidsim.effect_open_text_editor"
    bl_label = "Open in Text Editor"

    kernel_index: IntProperty(default=0, min=0)

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None or self.kernel_index >= len(item.kernels):
            return {"CANCELLED"}
        kernel = item.kernels[self.kernel_index]
        if kernel.source_text is None:
            self.report({"ERROR"}, "This kernel has no source Text datablock")
            return {"CANCELLED"}
        area = _text_area(context)
        if area is None:
            self.report({"ERROR"}, "No Blender editor area is available")
            return {"CANCELLED"}
        area.type = "TEXT_EDITOR"
        area.spaces.active.text = kernel.source_text
        return {"FINISHED"}


class FLUIDSIM_OT_effect_go_to_error(bpy.types.Operator):
    bl_idname = "fluidsim.effect_go_to_error"
    bl_label = "Go to Error Line"

    kernel_index: IntProperty(default=0, min=0)

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None or self.kernel_index >= len(item.kernels):
            return {"CANCELLED"}
        kernel = item.kernels[self.kernel_index]
        text = kernel.source_text
        if text is None:
            return {"CANCELLED"}
        area = _text_area(context)
        if area is None:
            return {"CANCELLED"}
        area.type = "TEXT_EDITOR"
        area.spaces.active.text = text
        line = max(0, int(kernel.error_line) - 1)
        column = max(0, int(kernel.error_column) - 1)
        text.cursor_set(line, character=column)
        return {"FINISHED"}


class FLUIDSIM_OT_effect_show_generated_source(bpy.types.Operator):
    bl_idname = "fluidsim.effect_show_generated_source"
    bl_label = "Show Generated Source"

    kernel_index: IntProperty(default=0, min=0)

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None or self.kernel_index >= len(item.kernels):
            return {"CANCELLED"}
        instance = effects.effect_instance_from_pg(context.scene, item)
        generated = generate_kernel_source(instance, self.kernel_index)
        name = f"FluidSim Generated Source - {instance.instance_id} - {generated.kernel_id}.py"
        text = bpy.data.texts.get(name) or bpy.data.texts.new(name)
        text.clear()
        text.write(generated.source)
        text["fluidsim_generated_source"] = True
        text["fluidsim_source_digest"] = generated.source_digest
        item.kernels[self.kernel_index].generated_text_name = text.name
        area = _text_area(context)
        if area is not None:
            area.type = "TEXT_EDITOR"
            area.spaces.active.text = text
        self.report({"INFO"}, f"Generated Source SHA-256: {generated.source_digest[:16]}…")
        return {"FINISHED"}


class FLUIDSIM_PT_effect_warp_code(bpy.types.Panel):
    bl_label = "Warp Code"
    bl_idname = "FLUIDSIM_PT_effect_warp_code"
    bl_parent_id = "FLUIDSIM_PT_custom_effects"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        stack = getattr(getattr(context, "scene", None), "fluidsim_effects", None)
        if stack is None or not bool(stack.items):
            return False
        index = max(0, min(int(stack.active_index), len(stack.items) - 1))
        item = stack.items[index]
        return bool(item.kernels)

    def draw(self, context):
        layout = self.layout
        stack = context.scene.fluidsim_effects
        item, _kernel = _active(context.scene)
        if item is None:
            return
        trust = layout.box()
        trust.label(text="Warp Code is trusted local executable code.", icon="ERROR")
        trust.label(text=f"Trust: {item.trust_state.replace('_', ' ').title()}")
        trust.prop(item, "safe_mode", text="Safe Mode")
        expert = trust.column(align=True)
        expert.enabled = not item.safe_mode
        expert.prop(item, "expert_acknowledged", text="Expert Access Acknowledged")

        for index, kernel in enumerate(item.kernels):
            box = layout.box()
            box.label(text=kernel.label or kernel.kernel_id, icon="TEXT")
            row = box.row(align=True)
            row.label(text="Domain")
            row.prop(kernel, "domain", text="")
            row.label(text="Stage")
            row.prop(kernel, "stage", text="")
            row = box.row(align=True)
            row.label(text="Cadence")
            row.prop(kernel, "cadence", text="")
            row.prop(kernel, "safe_mode", text="Safe Mode")
            source_row = box.row(align=True)
            source_row.prop(kernel, "source_text", text="Source")
            op = source_row.operator("fluidsim.effect_open_text_editor", text="Open in Text Editor", icon="TEXT")
            op.kernel_index = index
            if kernel.error_line:
                op = box.operator("fluidsim.effect_go_to_error", text=f"Go to Error Line {kernel.error_line}", icon="ERROR")
                op.kernel_index = index
            op = box.operator("fluidsim.effect_show_generated_source", text="Show Generated Source", icon="DOCUMENTS")
            op.kernel_index = index
            permissions = box.box()
            permissions.label(text="Data Access", icon="LOCKED")
            for binding in kernel.fields:
                spec = FIELD_SPECS.get(binding.field)
                row = permissions.row(align=True)
                row.label(text=binding.field)
                row.prop(binding, "access", text="")
                if spec is not None:
                    row.label(text=f"{spec.dtype} · {spec.units}")
                    if spec.prerequisites:
                        permissions.label(text="Requires: " + ", ".join(spec.prerequisites))

        api = layout.box()
        api.label(text="Field API Browser", icon="VIEWZOOM")
        domains = {kernel.domain for kernel in item.kernels}
        for name, spec in sorted(FIELD_SPECS.items()):
            if spec.domain.value not in domains:
                continue
            row = api.row(align=True)
            row.label(text=name)
            row.label(text=f"{spec.dtype} · {spec.units}")
            row.label(text="Persistent" if spec.persistent else "Transient")

        profile = layout.box()
        profile.label(text="Compile and Runtime Evidence", icon="TIME")
        profile.label(text=f"Warp: {stack.last_warp_version or 'Not compiled'}")
        profile.label(text=f"Compile: {stack.last_compile_seconds * 1000.0:.2f} ms")
        profile.label(text=f"Sandbox: {stack.last_sandbox_seconds * 1000.0:.2f} ms")
        profile.label(text=f"Calls: {item.calls:,}")
        profile.label(text=f"Elements: {item.elements:,}")
        profile.label(text=f"Submission: {item.submission_ms:.3f} ms")
        source_digest = item.kernels[0].generated_source_digest if item.kernels else ""
        profile.label(text=f"Source SHA-256: {source_digest[:16] + '…' if source_digest else 'Not compiled'}")
        profile.label(text=f"Last Runtime Error: {item.runtime_error or 'None'}")
        if item.compile_report_json and item.compile_report_json != "{}":
            try:
                reports = json.loads(item.compile_report_json)
                if reports:
                    profile.label(text=f"Target device: {reports[0].get('device', 'unknown')}")
            except Exception:
                pass


CLASSES = (
    FLUIDSIM_OT_effect_open_text_editor,
    FLUIDSIM_OT_effect_go_to_error,
    FLUIDSIM_OT_effect_show_generated_source,
    FLUIDSIM_PT_effect_warp_code,
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
