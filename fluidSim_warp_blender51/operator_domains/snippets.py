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


class FLUIDSIM_OT_snippet_add(bpy.types.Operator):
    """deprecated compatibility adapter that adds a Custom Effects v2 starter."""

    bl_idname = "fluidsim.snippet_add"
    bl_label = "Add Custom Effect"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..blender.effects import add_builtin_effect

        add_builtin_effect(context.scene, "fluidsim.vortex_force")
        self.report({"INFO"}, "Added Vortex Force in Custom Effects")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_remove(bpy.types.Operator):
    """deprecated compatibility adapter that removes the active Custom Effect."""

    bl_idname = "fluidsim.snippet_remove"
    bl_label = "Remove Custom Effect"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        result = bpy.ops.fluidsim.effect_remove()
        return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}


class FLUIDSIM_OT_snippet_argument_add(bpy.types.Operator):
    """Add an argument to the active snippet"""

    bl_idname = "fluidsim.snippet_argument_add"
    bl_label = "Add Argument"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        fs = getattr(getattr(context, "scene", None), "fluidsim", None)
        snippets = getattr(fs, "snippets", ())
        index = getattr(fs, "active_snippet", -1)
        if not isinstance(index, int) or not 0 <= index < len(snippets):
            cls.poll_message_set("Select a legacy snippet first")
            return False
        return True

    def execute(self, context):
        fs = context.scene.fluidsim
        if not self.poll(context):
            self.report({"WARNING"}, "Select a legacy snippet first")
            return {"CANCELLED"}
        fs.snippets[fs.active_snippet].arguments.add()
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_argument_remove(bpy.types.Operator):
    """Remove the last argument from the active snippet"""

    bl_idname = "fluidsim.snippet_argument_remove"
    bl_label = "Remove Argument"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        fs = getattr(getattr(context, "scene", None), "fluidsim", None)
        snippets = getattr(fs, "snippets", ())
        index = getattr(fs, "active_snippet", -1)
        if not isinstance(index, int) or not 0 <= index < len(snippets):
            cls.poll_message_set("Select a legacy snippet first")
            return False
        if not len(snippets[index].arguments):
            cls.poll_message_set("The active legacy snippet has no arguments")
            return False
        return True

    def execute(self, context):
        fs = context.scene.fluidsim
        if not self.poll(context):
            self.report({"WARNING"}, "The active legacy snippet has no arguments")
            return {"CANCELLED"}
        snip = fs.snippets[fs.active_snippet]
        snip.arguments.remove(len(snip.arguments) - 1)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_apply(bpy.types.Operator):
    """deprecated compatibility adapter that applies the Custom Effects v2 stack."""

    bl_idname = "fluidsim.snippet_apply"
    bl_label = "Compile & Apply Custom Effects"

    def execute(self, context):
        result = bpy.ops.fluidsim.effect_apply_stack()
        return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}


def _active_snippet(context):
    fs = context.scene.fluidsim
    if 0 <= fs.active_snippet < len(fs.snippets):
        return fs.snippets[fs.active_snippet]
    return None


def _snippet_code(pg) -> str:
    return pg.text_block.as_string() if pg.text_block else ""


def _set_snippet_code(pg, code: str):
    if pg.text_block is None:
        pg.text_block = bpy.data.texts.new(f"fluidsim_{pg.kernel_name}.py")
    pg.text_block.clear()
    pg.text_block.write(code if code.endswith("\n") or not code else code + "\n")


def _append_snippet_code(pg, addition: str):
    code = _snippet_code(pg)
    if code and not code.endswith("\n"):
        code += "\n"
    _set_snippet_code(pg, code + addition.rstrip("\n") + "\n")


class FLUIDSIM_OT_snippet_select(bpy.types.Operator):
    """Make this snippet the active one (shown in the editor)"""

    bl_idname = "fluidsim.snippet_select"
    bl_label = "Select Snippet"

    index: IntProperty(default=0, min=0)

    def execute(self, context):
        fs = context.scene.fluidsim
        if self.index < len(fs.snippets):
            fs.active_snippet = self.index
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_load_preset(bpy.types.Operator):
    """Load the selected library snippet into a new stack slot"""

    bl_idname = "fluidsim.snippet_load_preset"
    bl_label = "Load Snippet"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..properties import user_snippet_dir
        from ..sim import custom

        fs = context.scene.fluidsim
        selection = fs.snippet_library_selection
        if not selection or selection == "NONE":
            self.report({"ERROR"}, "No library snippet selected")
            return {"CANCELLED"}

        if selection.startswith("EX:"):
            preset = custom.preset_by_id(selection[3:])
            if preset is None:
                self.report({"ERROR"}, f"Unknown preset '{selection}'")
                return {"CANCELLED"}
            data = custom.preset_to_dict(
                preset.label,
                preset.description,
                preset.order,
                preset.run_over,
                preset.code,
                preset.arguments,
                preset.frame_start,
                preset.frame_end,
                preset.frame_step,
                snippet_type=getattr(preset, "snippet_type", "KERNEL"),
            )
            base_name = preset.preset_id
        else:  # USR:<filename>
            import os

            path = os.path.join(user_snippet_dir(), selection[4:])
            found = [
                d for d in custom.load_user_presets(user_snippet_dir()) if d.get("_path") == path
            ]
            if not found:
                self.report({"ERROR"}, "Saved snippet not found on disk")
                return {"CANCELLED"}
            data = found[0]
            base_name = os.path.splitext(selection[4:])[0]

        pg = fs.snippets.add()
        pg.kernel_name = f"{base_name}_{len(fs.snippets)}"
        pg.snippet_type = data.get("snippet_type", "KERNEL")
        pg.description = data.get("description", "")
        pg.order = data.get("order", "BEFORE_FORCES")
        pg.run_over = data.get("run_over", "PARTICLES")
        pg.frame_start = int(data.get("frame_start", 0))
        pg.frame_end = int(data.get("frame_end", 0))
        pg.frame_step = int(data.get("frame_step", 1))
        for arg in data.get("arguments", ()):  # serialized snippet arguments
            apg = pg.arguments.add()
            if arg.get("mode") == "CONSTANT":
                apg.mode = "CONSTANT"
                apg.constant_name = arg.get("name", "value")
                apg.constant_type = arg.get("type", "FLOAT")
                apg.constant_value = float(arg.get("value", 0.0))
            else:
                apg.mode = "FIELD"
                apg.field = arg.get("field", "temperature")
                apg.field_permission = arg.get("field_permission", "READ_WRITE")
        text = bpy.data.texts.new(f"fluidsim_{pg.kernel_name}.py")
        text.write(data.get("code", ""))
        pg.text_block = text
        fs.active_snippet = len(fs.snippets) - 1
        _snippets().validate_snippet_pg(pg)
        self.report(
            {"INFO"}, f"Loaded '{data.get('label', base_name)}' - press Apply Snippets to run it"
        )
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_validate(bpy.types.Operator):
    """Statically validate the active snippet (must pass before it can run)"""

    bl_idname = "fluidsim.snippet_validate"
    bl_label = "Validate"

    def execute(self, context):
        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        issues = _snippets().validate_snippet_pg(pg)
        errors = [i for i in issues if i.level == "ERROR"]
        warnings = [i for i in issues if i.level == "WARNING"]
        if errors:
            self.report({"WARNING"}, f"{len(errors)} error(s): {errors[0]}")
            return {"FINISHED"}
        if warnings:
            self.report({"INFO"}, f"Valid with {len(warnings)} warning(s)")
        else:
            self.report({"INFO"}, "Snippet is valid")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_autofix(bpy.types.Operator):
    """Auto-repair common snippet mistakes, then re-validate"""

    bl_idname = "fluidsim.snippet_autofix"
    bl_label = "Auto Fix"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..sim import custom

        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        code = _snippet_code(pg)
        fixed, fixes = custom.autofix_snippet(code)
        if fixed != code:
            _set_snippet_code(pg, fixed)
        _snippets().validate_snippet_pg(pg)
        if fixes:
            self.report({"INFO"}, f"Applied {len(fixes)} fix(es): {'; '.join(fixes[:3])}")
        else:
            self.report({"INFO"}, "Nothing to fix")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_test(bpy.types.Operator):
    """deprecated compatibility adapter that tests the active Custom Effects v2 draft."""

    bl_idname = "fluidsim.snippet_test"
    bl_label = "Test Custom Effect Draft"

    def execute(self, context):
        result = bpy.ops.fluidsim.effect_test_draft()
        return {"FINISHED"} if "FINISHED" in result else {"CANCELLED"}


class FLUIDSIM_OT_snippet_save_preset(bpy.types.Operator):
    """Save the active snippet to your snippet library for later reuse"""

    bl_idname = "fluidsim.snippet_save_preset"
    bl_label = "Save to Library"

    preset_name: StringProperty(
        name="Name",
        default="",
        description="Set the name used for the saved snippet preset; changing it changes the library label and output filename",
    )

    def invoke(self, context, event):
        pg = _active_snippet(context)
        if pg is not None and not self.preset_name:
            self.preset_name = pg.kernel_name
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        from ..properties import user_snippet_dir
        from ..sim import custom

        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        args = []
        for apg in pg.arguments:
            if apg.mode == "CONSTANT":
                args.append(
                    {
                        "mode": "CONSTANT",
                        "name": apg.constant_name,
                        "type": apg.constant_type,
                        "value": apg.constant_value,
                    }
                )
            else:
                args.append(
                    {
                        "mode": "FIELD",
                        "field": apg.field,
                        "field_permission": getattr(apg, "field_permission", "READ_WRITE"),
                    }
                )
        data = custom.preset_to_dict(
            self.preset_name or pg.kernel_name,
            pg.description or "User snippet",
            pg.order,
            pg.run_over,
            _snippet_code(pg),
            args,
            pg.frame_start,
            pg.frame_end,
            pg.frame_step,
            snippet_type=getattr(pg, "snippet_type", "KERNEL"),
        )
        path = custom.save_user_preset(user_snippet_dir(create=True), data)
        self.report({"INFO"}, f"Saved: {path}")
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_insert_shortcut(bpy.types.Operator):
    """Insert a code shortcut at the end of the active snippet"""

    bl_idname = "fluidsim.snippet_insert_shortcut"
    bl_label = "Insert Shortcut"
    bl_options = {"REGISTER", "UNDO"}

    shortcut_id: StringProperty()

    def execute(self, context):
        from ..sim import custom

        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        shortcut = custom.shortcut_by_id(self.shortcut_id)
        if shortcut is None:
            self.report({"ERROR"}, f"Unknown shortcut '{self.shortcut_id}'")
            return {"CANCELLED"}
        _append_snippet_code(pg, shortcut.code)
        _snippets().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_shortcut_field(bpy.types.Operator):
    """Bind the chosen solver field as an argument and reference it in code"""

    bl_idname = "fluidsim.snippet_shortcut_field"
    bl_label = "Add Field"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        fs = context.scene.fluidsim
        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        name = fs.shortcut_field
        if not any(a.mode == "FIELD" and a.field == name for a in pg.arguments):
            apg = pg.arguments.add()
            apg.mode = "FIELD"
            apg.field = name
        index = "particle_index" if pg.run_over == "PARTICLES" else "flat_voxel_index"
        if name.startswith("particle_") and pg.run_over != "PARTICLES":
            self.report(
                {"WARNING"},
                f"'{name}' is a particle field; set Run Over to Particles to index it correctly",
            )
        _append_snippet_code(pg, f"# field '{name}': read/write via {name}[{index}]")
        _snippets().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_shortcut_constant(bpy.types.Operator):
    """Add a named constant argument and reference it in code"""

    bl_idname = "fluidsim.snippet_shortcut_constant"
    bl_label = "Add Constant"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        fs = context.scene.fluidsim
        pg = _active_snippet(context)
        if pg is None:
            self.report({"ERROR"}, "No active snippet")
            return {"CANCELLED"}
        name = fs.shortcut_constant_name.strip()
        if not name.isidentifier():
            self.report({"ERROR"}, f"'{name}' is not a valid constant name")
            return {"CANCELLED"}
        existing = next(
            (a for a in pg.arguments if a.mode == "CONSTANT" and a.constant_name == name), None
        )
        if existing is not None:
            existing.constant_value = fs.shortcut_constant_value
        else:
            apg = pg.arguments.add()
            apg.mode = "CONSTANT"
            apg.constant_name = name
            apg.constant_type = "FLOAT"
            apg.constant_value = fs.shortcut_constant_value
        _append_snippet_code(
            pg,
            f"# constant '{name}' = {fs.shortcut_constant_value:g} "
            f"(edit value in the Arguments list)",
        )
        _snippets().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_edit_line(bpy.types.Operator):
    """Edit one line of the active snippet"""

    bl_idname = "fluidsim.snippet_edit_line"
    bl_label = "Edit Line"
    bl_options = {"REGISTER", "UNDO"}

    line_index: IntProperty(default=0, min=0)
    line_text: StringProperty(
        name="Line",
        default="",
        description="Edit the selected snippet source line; changing it replaces that line and triggers snippet validation",
    )

    def invoke(self, context, event):
        pg = _active_snippet(context)
        if pg is None:
            return {"CANCELLED"}
        lines = _snippet_code(pg).split("\n")
        if self.line_index < len(lines):
            self.line_text = lines[self.line_index]
        return context.window_manager.invoke_props_dialog(self, width=520)

    def execute(self, context):
        pg = _active_snippet(context)
        if pg is None:
            return {"CANCELLED"}
        lines = _snippet_code(pg).split("\n")
        while len(lines) <= self.line_index:
            lines.append("")
        lines[self.line_index] = self.line_text
        _set_snippet_code(pg, "\n".join(lines))
        _snippets().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_line_add(bpy.types.Operator):
    """Append a new line to the active snippet"""

    bl_idname = "fluidsim.snippet_line_add"
    bl_label = "Add Line"
    bl_options = {"REGISTER", "UNDO"}

    line_text: StringProperty(
        name="Line",
        default="",
        description="Set the source code appended as a new final line; changing it changes the code that is validated and compiled",
    )

    def invoke(self, context, event):
        self.line_text = ""
        return context.window_manager.invoke_props_dialog(self, width=520)

    def execute(self, context):
        pg = _active_snippet(context)
        if pg is None:
            return {"CANCELLED"}
        _append_snippet_code(pg, self.line_text)
        _snippets().validate_snippet_pg(pg)
        return {"FINISHED"}


class FLUIDSIM_OT_snippet_line_delete(bpy.types.Operator):
    """Delete one line of the active snippet"""

    bl_idname = "fluidsim.snippet_line_delete"
    bl_label = "Delete Line"
    bl_options = {"REGISTER", "UNDO"}

    line_index: IntProperty(default=0, min=0)

    def execute(self, context):
        pg = _active_snippet(context)
        if pg is None:
            return {"CANCELLED"}
        lines = _snippet_code(pg).split("\n")
        if self.line_index < len(lines):
            lines.pop(self.line_index)
            _set_snippet_code(pg, "\n".join(lines))
            _snippets().validate_snippet_pg(pg)
        return {"FINISHED"}
