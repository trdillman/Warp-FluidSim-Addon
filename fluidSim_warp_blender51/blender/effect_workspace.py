"""Advanced Warp Code authoring workspace for Custom Effects."""

from __future__ import annotations

import json

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from . import effects
from ..sim.effects.enums import TrustState
from ..sim.effects.fields import FIELD_SPECS
from ..sim.effects.source import generate_kernel_source
from ..sim.effects.templates import starter_template_items


_TRUSTED_STATES = frozenset(
    {TrustState.BUILT_IN_TRUSTED.value, TrustState.LOCAL_TRUSTED.value}
)

# Instance ids the developer has opted to keep trusted across edits *this
# session only*.  This is process-global and is cleared on ``unregister`` (add-on
# reload / Blender restart), so it can never outlive the session or be persisted
# into a shipped .blend.  It never bypasses the exact-SHA trust gate: it only
# spares an opted-in author the Review & Trust dialog on every edit; the drifted
# digest is still re-trusted through the real ``trust_effect`` path.
_SESSION_TRUST: set[str] = set()


def set_session_trust(instance_id: str, enabled: bool) -> None:
    """Opt an effect instance into (or out of) session-scoped edit trust."""
    if enabled:
        _SESSION_TRUST.add(str(instance_id))
    else:
        _SESSION_TRUST.discard(str(instance_id))


def session_trust_enabled(instance_id: str) -> bool:
    """Whether this instance is currently opted into session edit trust."""
    return str(instance_id) in _SESSION_TRUST


def reprompt_trust_required(item) -> bool:
    """Whether the Review & Trust dialog must re-prompt for this effect."""
    if item.trust_state in _TRUSTED_STATES:
        return False
    return not session_trust_enabled(item.instance_id)


def session_retrust_pending(item) -> bool:
    """A session-trusted instance whose digest drifted untrusted after an edit."""
    return (
        session_trust_enabled(item.instance_id)
        and item.trust_state not in _TRUSTED_STATES
    )


def apply_session_trust(scene) -> int:
    """Re-establish SHA-bound trust for session-trusted effects that drifted."""
    stack = getattr(scene, "fluidsim_effects", None)
    if stack is None:
        return 0
    count = 0
    for index, item in enumerate(stack.items):
        if session_retrust_pending(item):
            effects.trust_effect(scene, index)
            count += 1
    return count


def _active(scene):
    stack = getattr(scene, "fluidsim_effects", None)
    if stack is None or not stack.items:
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


def _generate_kernel_boilerplate(kernel, item) -> str:
    # Same resolver as execution, so the header lists what the kernel binds.
    param_keys = effects.bound_parameter_keys(item, kernel)
    param_map = {p.key: p for p in item.parameters}

    header = [
        "# ==============================================================================",
        f"# FluidSim Warp Kernel: {kernel.kernel_id} ({kernel.label or 'Custom'})",
        f"# Domain: {kernel.domain} | Stage: {kernel.stage} | Cadence: {kernel.cadence}",
        "#",
        "# AVAILABLE CONSTANTS:",
        "#   voxel_size (float), frame (int), time (float), real_time_step (float),",
        "#   solver_time_step (float), time_scale (float), substeps (int)",
        "#   real_time_step = simulated seconds per frame (time_scale / fps);",
        "#   solver_time_step = simulated seconds per substep.",
        "#",
    ]
    if kernel.domain == "PARTICLES":
        header.extend([
            "# BUILT-IN THREAD VARIABLES:",
            "#   particle_index (int), position (wp.vec3), velocity (wp.vec3)",
            "#",
        ])
    else:
        header.extend([
            "# BUILT-IN THREAD VARIABLES:",
            "#   flat_voxel_index (int), leaf_index (int), global_x/y/z (int),",
            "#   local_x/y/z (int), world_position (wp.vec3)",
            "#",
        ])

    header.append("# BOUND FIELDS:")
    if kernel.fields:
        for f in kernel.fields:
            spec = FIELD_SPECS.get(f.field)
            dtype = spec.dtype if spec else "unknown"
            header.append(f"#   {f.field} ({dtype}) -> {f.access}")
    else:
        header.append("#   (None bound)")
    header.append("#")

    header.append("# BOUND PARAMETERS:")
    if param_keys:
        for k in param_keys:
            p = param_map.get(k)
            if p:
                header.append(f"#   {p.key} ({p.kind}): {p.label} [{p.unit or 'NONE'}] - {p.description or ''}")
            else:
                header.append(f"#   {k}")
    else:
        header.append("#   (None bound)")
    header.append("# ==============================================================================")
    header.append("")
    return "\n".join(header)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

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
        if hasattr(area, "spaces") and hasattr(area.spaces, "active"):
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
        if hasattr(area, "spaces") and hasattr(area.spaces, "active"):
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
            if hasattr(area, "spaces") and hasattr(area.spaces, "active"):
                area.spaces.active.text = text
        self.report({"INFO"}, f"Generated Source SHA-256: {generated.source_digest[:16]}…")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_session_trust(bpy.types.Operator):
    """Trust this effect's edits for the rest of this session (developer)."""

    bl_idname = "fluidsim.effect_session_trust"
    bl_label = "Trust While Editing (Session)"
    bl_options = {"INTERNAL"}

    enable: BoolProperty(default=True)
    acknowledged: BoolProperty(
        name="I author this Warp code and accept running my local edits",
        default=False,
    )

    def invoke(self, context, event):
        if not self.enable:
            return self.execute(context)
        self.acknowledged = False
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.label(
            text="Session trust runs your local edits without re-prompting.",
            icon="ERROR",
        )
        layout.label(text="Trust is still SHA-bound; it clears when the add-on reloads.")
        layout.prop(self, "acknowledged")

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None:
            return {"CANCELLED"}
        if self.enable and not self.acknowledged:
            self.report({"WARNING"}, "Acknowledge authorship to enable session trust")
            return {"CANCELLED"}
        set_session_trust(item.instance_id, self.enable)
        if self.enable:
            apply_session_trust(context.scene)
            self.report({"INFO"}, "Editing this effect is trusted for this session")
        else:
            self.report({"INFO"}, "Session trust cleared for this effect")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_create_custom(bpy.types.Operator):
    """Create a new custom effect from a starter template."""

    bl_idname = "fluidsim.effect_create_custom"
    bl_label = "Create Custom Effect"
    bl_options = {"REGISTER", "UNDO"}

    template: EnumProperty(
        name="Template",
        items=starter_template_items(),
        default="PARTICLE_FORCE",
    )
    effect_name: StringProperty(name="Name", default="New Custom Effect")

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        pg = effects.add_custom_effect_from_template(
            context.scene, self.template, effect_name=self.effect_name
        )
        if pg.kernels and pg.kernels[0].source_text:
            area = _text_area(context)
            if area is not None:
                area.type = "TEXT_EDITOR"
                if hasattr(area, "spaces") and hasattr(area.spaces, "active"):
                    area.spaces.active.text = pg.kernels[0].source_text
        self.report({"INFO"}, f"Created custom effect {self.effect_name}")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_convert_to_custom(bpy.types.Operator):
    """Detach this built-in or package effect into an editable custom effect."""

    bl_idname = "fluidsim.effect_convert_to_custom"
    bl_label = "Convert to Editable Custom Effect"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None:
            return {"CANCELLED"}
        effects.convert_instance_to_custom(context.scene, item)
        self.report({"INFO"}, "Converted to editable custom effect")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_parameter_add(bpy.types.Operator):
    """Add a new artist-facing parameter to this custom effect."""

    bl_idname = "fluidsim.effect_parameter_add"
    bl_label = "Add Parameter"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None:
            return {"CANCELLED"}
        key = effects.unique_id("param", (q.key for q in item.parameters))
        p = item.parameters.add()
        p.key = key
        p.label = key.replace("_", " ").title()
        p.kind = "FLOAT"
        p.float_value = 1.0
        p.minimum = 0.0
        p.maximum = 100.0
        p.has_minimum = True
        p.has_maximum = True
        effects.rebuild_definition_json(context.scene, item)
        self.report({"INFO"}, f"Added parameter {p.key}")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_parameter_remove(bpy.types.Operator):
    """Remove a parameter from this custom effect."""

    bl_idname = "fluidsim.effect_parameter_remove"
    bl_label = "Remove Parameter"
    bl_options = {"REGISTER", "UNDO"}

    parameter_index: IntProperty(default=0, min=0)

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None or self.parameter_index >= len(item.parameters):
            return {"CANCELLED"}
        # Unbind the key first, or every kernel that used it fails validation.
        effects.unbind_parameter_key(item, item.parameters[self.parameter_index].key)
        item.parameters.remove(self.parameter_index)
        effects.rebuild_definition_json(context.scene, item)
        return {"FINISHED"}


class FLUIDSIM_OT_effect_parameter_move(bpy.types.Operator):
    """Reorder a parameter up or down."""

    bl_idname = "fluidsim.effect_parameter_move"
    bl_label = "Move Parameter"
    bl_options = {"REGISTER", "UNDO"}

    parameter_index: IntProperty(default=0, min=0)
    direction: EnumProperty(items=[("UP", "Up", ""), ("DOWN", "Down", "")], default="UP")

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None or self.parameter_index >= len(item.parameters):
            return {"CANCELLED"}
        target = self.parameter_index - 1 if self.direction == "UP" else self.parameter_index + 1
        if 0 <= target < len(item.parameters):
            item.parameters.move(self.parameter_index, target)
            effects.rebuild_definition_json(context.scene, item)
        return {"FINISHED"}


class FLUIDSIM_OT_effect_kernel_add(bpy.types.Operator):
    """Add an additional simulation kernel to this effect."""

    bl_idname = "fluidsim.effect_kernel_add"
    bl_label = "Add Kernel"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None:
            return {"CANCELLED"}
        kid = effects.unique_id("kernel", (q.kernel_id for q in item.kernels))
        k = item.kernels.add()
        k.kernel_id = kid
        k.parameter_keys_authored = True
        k.label = kid.replace("_", " ").title()
        k.domain = "PARTICLES"
        k.stage = "AFTER_PARTICLE_TRANSFER"
        k.cadence = "EVERY_SUBSTEP"
        k.safe_mode = True
        text_name = f"FluidSim Effect - {item.instance_id} - {kid}.py"
        text = bpy.data.texts.get(text_name) or bpy.data.texts.new(text_name)
        text.clear()
        text.write("# Custom kernel body\npass\n")
        k.source_text = text
        effects.rebuild_definition_json(context.scene, item)
        self.report({"INFO"}, f"Added kernel {kid}")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_kernel_remove(bpy.types.Operator):
    """Remove a kernel from this effect."""

    bl_idname = "fluidsim.effect_kernel_remove"
    bl_label = "Remove Kernel"
    bl_options = {"REGISTER", "UNDO"}

    kernel_index: IntProperty(default=0, min=0)

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None or self.kernel_index >= len(item.kernels):
            return {"CANCELLED"}
        if len(item.kernels) <= 1:
            self.report({"WARNING"}, "An effect must have at least one kernel")
            return {"CANCELLED"}
        item.kernels.remove(self.kernel_index)
        effects.rebuild_definition_json(context.scene, item)
        return {"FINISHED"}


_NO_FIELD_ITEMS = [("none", "None Available", "")]
# Blender keeps no reference to dynamic enum strings, so the item lists must outlive the callback.
_FIELD_ITEMS_BY_DOMAIN: dict[str, list] = {}


def _domain_field_items_callback(self, context):
    item, _k = _active(context.scene)
    if item is None or not item.kernels:
        return _NO_FIELD_ITEMS
    k_idx = min(getattr(self, "kernel_index", 0), len(item.kernels) - 1)
    domain = item.kernels[k_idx].domain
    items = _FIELD_ITEMS_BY_DOMAIN.get(domain)
    if items is None:
        items = [
            (name, f"{name} ({spec.dtype})", spec.description or "")
            for name, spec in sorted(FIELD_SPECS.items())
            if spec.domain.value == domain
        ] or _NO_FIELD_ITEMS
        _FIELD_ITEMS_BY_DOMAIN[domain] = items
    return items


class FLUIDSIM_OT_effect_field_add(bpy.types.Operator):
    """Bind a solver data field to this kernel."""

    bl_idname = "fluidsim.effect_field_add"
    bl_label = "Add Field Binding"
    bl_options = {"REGISTER", "UNDO"}

    kernel_index: IntProperty(default=0, min=0)
    field_name: EnumProperty(name="Field", items=_domain_field_items_callback)
    access: EnumProperty(name="Access", items=effects._ACCESS_ITEMS, default="READ_WRITE_CURRENT")

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None or self.kernel_index >= len(item.kernels):
            return {"CANCELLED"}
        if self.field_name == "none":
            return {"CANCELLED"}
        kernel = item.kernels[self.kernel_index]
        existing = next((f for f in kernel.fields if f.field == self.field_name), None)
        if existing is not None:
            existing.access = self.access
        else:
            fb = kernel.fields.add()
            fb.field = self.field_name
            fb.access = self.access
        effects.rebuild_definition_json(context.scene, item)
        self.report({"INFO"}, f"Bound field {self.field_name} ({self.access})")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_field_remove(bpy.types.Operator):
    """Remove a field binding from this kernel."""

    bl_idname = "fluidsim.effect_field_remove"
    bl_label = "Remove Field Binding"
    bl_options = {"REGISTER", "UNDO"}

    kernel_index: IntProperty(default=0, min=0)
    field_index: IntProperty(default=0, min=0)

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None or self.kernel_index >= len(item.kernels):
            return {"CANCELLED"}
        kernel = item.kernels[self.kernel_index]
        if self.field_index < len(kernel.fields):
            kernel.fields.remove(self.field_index)
            effects.rebuild_definition_json(context.scene, item)
        return {"FINISHED"}


class FLUIDSIM_OT_effect_generate_boilerplate(bpy.types.Operator):
    """Generate or update structured cheatsheet docstrings in the kernel Text datablock."""

    bl_idname = "fluidsim.effect_generate_boilerplate"
    bl_label = "Update Header Docstrings"
    bl_options = {"REGISTER", "UNDO"}

    kernel_index: IntProperty(default=0, min=0)

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None or self.kernel_index >= len(item.kernels):
            return {"CANCELLED"}
        kernel = item.kernels[self.kernel_index]
        if kernel.source_text is None:
            return {"CANCELLED"}
        text = kernel.source_text
        content = text.as_string()
        # Strip old header banner if present
        if content.startswith("# =============================================================================="):
            end_marker = "# ==============================================================================\n"
            parts = content.split(end_marker, 2)
            user_body = parts[-1].lstrip("\n") if len(parts) >= 3 else content
        else:
            user_body = content
        new_header = _generate_kernel_boilerplate(kernel, item)
        text.clear()
        text.write(new_header + "\n" + user_body)
        self.report({"INFO"}, "Updated kernel header comments in Text Editor")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_validate_kernel(bpy.types.Operator):
    """Validate AST syntax, parameter bindings, and field stage legality."""

    bl_idname = "fluidsim.effect_validate_kernel"
    bl_label = "Validate Code"

    kernel_index: IntProperty(default=0, min=0)

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None:
            return {"CANCELLED"}
        effects.rebuild_definition_json(context.scene, item)
        instance = effects.effect_instance_from_pg(context.scene, item)
        from ..sim.effects.validation import validate_effect
        res = validate_effect(instance, safe_mode=item.safe_mode)
        for kernel_pg in item.kernels:
            kernel_pg.error_line = 0
            kernel_pg.error_column = 0
        if not res.valid:
            # Pin each located issue to the kernel that produced it, not the clicked one.
            known = {kernel_pg.kernel_id: kernel_pg for kernel_pg in item.kernels}
            for found in res.issues:
                kernel_pg = known.get(getattr(found, "kernel_id", ""))
                if kernel_pg is not None and not kernel_pg.error_line:
                    kernel_pg.error_line = int(getattr(found, "line", 0) or 0)
                    kernel_pg.error_column = int(getattr(found, "column", 0) or 0)
            issue = res.issues[0]
            self.report({"ERROR"}, f"Validation failed ({issue.code}): {issue.message}")
            return {"FINISHED"}  # FINISHED so the panel redraws with the error line
        self.report({"INFO"}, "Kernel validation passed. AST and field bindings are valid.")
        return {"FINISHED"}


class FLUIDSIM_OT_effect_test_sandbox(bpy.types.Operator):
    """Compile with Warp and execute 1 test step on synthetic buffers."""

    bl_idname = "fluidsim.effect_test_sandbox"
    bl_label = "Test in Sandbox"

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None:
            return {"CANCELLED"}
        effects.rebuild_definition_json(context.scene, item)
        instance = effects.effect_instance_from_pg(context.scene, item)
        from ..sim.effects.compiler import compile_effect
        from ..sim.effects.sandbox import sandbox_test
        from ..preferences import configured_device
        device = configured_device() or "cpu"
        try:
            compiled = compile_effect(instance, device=device)
            report = sandbox_test(compiled, device=device)
            if not report.passed:
                issue = report.issues[0] if report.issues else None
                msg = f"{issue.code}: {issue.message}" if issue is not None else "Sandbox execution failed"
                self.report({"ERROR"}, f"Sandbox test failed: {msg}")
                return {"CANCELLED"}
            compile_seconds = sum(k.compile_report.duration_seconds for k in compiled.kernels)
            exec_seconds = sum(k.duration_seconds for k in report.kernels)
            self.report(
                {"INFO"},
                f"Sandbox passed! Compile: {compile_seconds * 1000.0:.1f}ms, "
                f"Exec: {exec_seconds * 1000.0:.2f}ms",
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Sandbox compilation failed: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class FLUIDSIM_OT_effect_install_to_library(bpy.types.Operator):
    """Package this effect and save it to your user effect library."""

    bl_idname = "fluidsim.effect_install_to_library"
    bl_label = "Save to User Library"

    @staticmethod
    def _package_path(item):
        from .effect_library import user_library_path

        # Same folder and naming as effect_library.install_package, so the
        # library browser lists what was saved here.
        return user_library_path() / f"{item.effect_id}-{item.effect_version}.fseffect"

    def invoke(self, context, event):
        item, _kernel = _active(context.scene)
        if item is not None and self._package_path(item).exists():
            return context.window_manager.invoke_confirm(
                self,
                event,
                title="Overwrite Library Package?",
                message=f"{self._package_path(item).name} already exists and will be replaced.",
            )
        return self.execute(context)

    def execute(self, context):
        item, _kernel = _active(context.scene)
        if item is None:
            return {"CANCELLED"}
        effects.rebuild_definition_json(context.scene, item)
        definition = effects.exportable_definition(context.scene, item)
        pkg_path = self._package_path(item)
        from ..sim.effects.package import write_effect_package
        try:
            write_effect_package(pkg_path, definition)
            # No trust is recorded here: an installed package is reviewed through
            # Review & Trust (or session trust) like any other, never auto-trusted.
            self.report({"INFO"}, f"Saved to library: {pkg_path.name}")
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to save package: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class FLUIDSIM_OT_open_effect_authoring_workspace(bpy.types.Operator):
    """Switch or arrange workspace for custom effect authoring."""

    bl_idname = "fluidsim.open_effect_authoring_workspace"
    bl_label = "Open Authoring Layout"

    def execute(self, context):
        item, kernel = _active(context.scene)
        if kernel and kernel.source_text:
            area = _text_area(context)
            if area is not None:
                area.type = "TEXT_EDITOR"
                if hasattr(area, "spaces") and hasattr(area.spaces, "active"):
                    area.spaces.active.text = kernel.source_text
                self.report({"INFO"}, f"Authoring layout ready; editing {kernel.source_text.name}")
                return {"FINISHED"}
        self.report({"INFO"}, "No active kernel source text found")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# UI Panel
# ---------------------------------------------------------------------------

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
        from ..preferences import effect_authoring_enabled

        # D9: the Warp-code authoring workspace is a developer surface, gated on
        # the add-on preference.  The artist-facing Custom Effects stack panel is
        # never gated (see FLUIDSIM_PT_custom_effects.poll).
        if not effect_authoring_enabled(context):
            return False
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

        # --- Authoring Actions Bar ---
        top_bar = layout.row(align=True)
        top_bar.operator("fluidsim.effect_create_custom", text="New Custom Effect", icon="ADD")
        top_bar.operator("fluidsim.open_effect_authoring_workspace", text="Authoring Layout", icon="WORKSPACE")
        if not item.effect_id.startswith("custom."):
            top_bar.operator("fluidsim.effect_convert_to_custom", text="Convert to Custom", icon="DUPLICATE")

        # --- Effect Metadata & Library Box ---
        meta_box = layout.box()
        meta_row = meta_box.row(align=True)
        meta_row.prop(item, "display_name", text="Effect")
        pub_row = meta_box.row(align=True)
        pub_row.operator("fluidsim.effect_install_to_library", text="Save to User Library", icon="EXPORT")
        pub_row.operator("fluidsim.effect_export_package", text="Export Package", icon="PACKAGE")

        # --- Trust & Safety Box ---
        trust = layout.box()
        trust.label(text="Warp Code is trusted local executable code.", icon="ERROR")
        trust.label(text=f"Trust: {item.trust_state.replace('_', ' ').title()}")
        trust.prop(item, "safe_mode", text="Safe Mode")
        expert = trust.column(align=True)
        expert.enabled = not item.safe_mode
        expert.prop(item, "expert_acknowledged", text="Expert Access Acknowledged")

        # Scoped "trust while editing"
        session = trust.row(align=True)
        if session_trust_enabled(item.instance_id):
            if session_retrust_pending(item):
                session.label(text="Edit changed the digest — re-trust to run.", icon="FILE_REFRESH")
            else:
                session.label(text="Session trust: editing this effect is trusted.", icon="UNLOCKED")
            op = session.operator("fluidsim.effect_session_trust", text="Stop Trusting Edits")
            op.enable = False
        else:
            op = session.operator("fluidsim.effect_session_trust", text="Trust While Editing", icon="UNLOCKED")
            op.enable = True

        # --- Parameters Designer Box ---
        param_box = layout.box()
        param_head = param_box.row(align=True)
        param_head.label(text="Parameters (Artist Interface)", icon="PROPERTIES")
        param_head.operator("fluidsim.effect_parameter_add", text="Add Parameter", icon="ADD")
        for p_idx, param in enumerate(item.parameters):
            p_box = param_box.box()
            row = p_box.row(align=True)
            row.prop(param, "label", text="")
            row.prop(param, "key", text="Key")
            row.prop(param, "kind", text="")
            up_op = row.operator("fluidsim.effect_parameter_move", text="", icon="TRIA_UP")
            up_op.parameter_index = p_idx
            up_op.direction = "UP"
            down_op = row.operator("fluidsim.effect_parameter_move", text="", icon="TRIA_DOWN")
            down_op.parameter_index = p_idx
            down_op.direction = "DOWN"
            rm_op = row.operator("fluidsim.effect_parameter_remove", text="", icon="X")
            rm_op.parameter_index = p_idx

            if param.kind in {"FLOAT", "INT"}:
                r_val = p_box.row(align=True)
                r_val.prop(param, "float_value" if param.kind == "FLOAT" else "int_value", text="Default")
                r_val.prop(param, "minimum", text="Min")
                r_val.prop(param, "maximum", text="Max")
                r_val.prop(param, "unit", text="Unit")
            elif param.kind == "BOOL":
                p_box.prop(param, "bool_value", text="Default")
            elif param.kind == "VECTOR":
                r_vec = p_box.row(align=True)
                r_vec.prop(param, "vector_value", text="Default")
                r_vec.prop(param, "unit", text="Unit")
            elif param.kind == "COLOR":
                p_box.prop(param, "color_value", text="Default")
            elif param.kind == "ENUM":
                p_box.prop(param, "enum_items_json", text="Items")
                p_box.prop(param, "enum_value", text="Default")
            elif param.kind == "OBJECT":
                r_obj = p_box.row(align=True)
                r_obj.prop(param, "object_value", text="Object")
                c_op = r_obj.operator("fluidsim.effect_create_controller", text="Create Empty", icon="EMPTY_AXIS")
                c_op.parameter_key = param.key

        # --- Kernels Manager Box ---
        kernel_box = layout.box()
        k_head = kernel_box.row(align=True)
        k_head.label(text="Kernels", icon="TEXT")
        k_head.operator("fluidsim.effect_kernel_add", text="Add Kernel", icon="ADD")

        for index, kernel in enumerate(item.kernels):
            box = kernel_box.box()
            k_title = box.row(align=True)
            k_title.prop(kernel, "label", text="")
            k_title.label(text=f"ID: {kernel.kernel_id}")
            if len(item.kernels) > 1:
                rm_k = k_title.operator("fluidsim.effect_kernel_remove", text="", icon="X")
                rm_k.kernel_index = index

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
            op_open = source_row.operator("fluidsim.effect_open_text_editor", text="Open in Text Editor", icon="TEXT")
            op_open.kernel_index = index
            op_bp = source_row.operator("fluidsim.effect_generate_boilerplate", text="Docstrings", icon="FILE_TEXT")
            op_bp.kernel_index = index

            tools_row = box.row(align=True)
            v_op = tools_row.operator("fluidsim.effect_validate_kernel", text="Validate Code", icon="CHECKMARK")
            v_op.kernel_index = index
            tools_row.operator("fluidsim.effect_test_sandbox", text="Test Sandbox", icon="PLAY")
            op_show = tools_row.operator("fluidsim.effect_show_generated_source", text="Show Generated Source", icon="DOCUMENTS")
            op_show.kernel_index = index

            if kernel.error_line:
                op_err = box.operator("fluidsim.effect_go_to_error", text=f"Go to Error Line {kernel.error_line}", icon="ERROR")
                op_err.kernel_index = index

            # Field Bindings
            permissions = box.box()
            perm_head = permissions.row(align=True)
            perm_head.label(text="Data Access (Field Bindings)", icon="LOCKED")
            add_f = perm_head.operator("fluidsim.effect_field_add", text="Add Field", icon="ADD")
            add_f.kernel_index = index

            for f_idx, binding in enumerate(kernel.fields):
                spec = FIELD_SPECS.get(binding.field)
                r_field = permissions.row(align=True)
                r_field.label(text=binding.field)
                r_field.prop(binding, "access", text="")
                if spec is not None:
                    r_field.label(text=f"{spec.dtype} · {spec.units}")
                    if spec.prerequisites:
                        permissions.label(text="Requires: " + ", ".join(spec.prerequisites))
                rm_f = r_field.operator("fluidsim.effect_field_remove", text="", icon="X")
                rm_f.kernel_index = index
                rm_f.field_index = f_idx

            # Parameter Bindings Exposure
            param_row = box.row(align=True)
            param_row.prop(kernel, "parameter_keys_csv", text="Bound Parameters")

        # --- Field API Browser ---
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

        # --- Profile Box ---
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
    FLUIDSIM_OT_effect_session_trust,
    FLUIDSIM_OT_effect_create_custom,
    FLUIDSIM_OT_effect_convert_to_custom,
    FLUIDSIM_OT_effect_parameter_add,
    FLUIDSIM_OT_effect_parameter_remove,
    FLUIDSIM_OT_effect_parameter_move,
    FLUIDSIM_OT_effect_kernel_add,
    FLUIDSIM_OT_effect_kernel_remove,
    FLUIDSIM_OT_effect_field_add,
    FLUIDSIM_OT_effect_field_remove,
    FLUIDSIM_OT_effect_generate_boilerplate,
    FLUIDSIM_OT_effect_validate_kernel,
    FLUIDSIM_OT_effect_test_sandbox,
    FLUIDSIM_OT_effect_install_to_library,
    FLUIDSIM_OT_open_effect_authoring_workspace,
    FLUIDSIM_PT_effect_warp_code,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    # D9: session edit-trust never outlives the add-on; drop it on reload.
    _SESSION_TRUST.clear()
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:  # noqa: BLE001 - report and keep walking
            print(f"FluidSim: could not unregister {cls.__name__}: {exc!r}")
