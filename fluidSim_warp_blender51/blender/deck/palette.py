"""Command palette: fuzzy search over FluidSim operators, presets, demos and settings.

``fluidsim.command_palette`` (Ctrl+K in the 3D View) opens a centred search card.
Its index is built when it opens, from what is registered and authored now:

- every registered ``fluidsim.*`` operator (its label and description), except
  the Deck's own launchers and event router (:data:`EXCLUDED_OPERATORS`);
- the presets ``presets.library.discover_presets`` finds and the demo catalogue;
- every scene and domain setting with its current value, except the
  presentation/status and adapter tiers of the Pending Settings registry.

Enter runs an operator, toggles a boolean, or opens an inline numeric field for
an int/float that commits through the property itself (so its ``update=`` and
the Pending Settings classification run exactly as for a panel edit). Enum,
string and vector settings are read-only here and jump to the panel
(``fluidsim.open_scene_properties``). Rows that cannot run now (a failing operator
poll, a preset apply that would be refused, a demo whose file is missing) draw
dimmed with the reason; Enter on one, or an operator that refuses after Enter,
says why and keeps the palette open. A setting write pushes one undo step and
flashes the LIVE chip only for fields that apply straight away; Reset and
Reinitialize fields report that they are staged, and during a Production Bake
the edit is held. The palette keeps no settings state. Keys it
does not use pass through to Blender only as Ctrl/Cmd shortcuts (Ctrl+Z, Ctrl+Shift+Z);
other keys, Delete included, stay with the palette.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import bpy

from . import _draw, bar, controls, tokens

EXCLUDED_OPERATORS = frozenset({
    "fluidsim.command_palette", "fluidsim.radial_wheel", "fluidsim.deck_modal"})
JUMP_OPERATOR = "fluidsim.open_scene_properties"
_EDITABLE_TYPES = frozenset({"BOOLEAN", "INT", "FLOAT"})
VISIBLE_ROWS = 10


@dataclass(frozen=True)
class Entry:
    kind: str  # "operator" | "setting" | "preset" | "demo"
    target: str
    label: str
    description: str = ""
    value: str = ""
    value_type: str = ""
    editable: bool = False
    jump_operator: str = ""


# Index ------------------------------------------------------------------------

def operator_entries(ops_namespace=None) -> list[Entry]:
    """One entry per registered ``fluidsim`` operator, from its RNA label and description."""
    if ops_namespace is None:
        ops_namespace = bpy.ops.fluidsim
    entries = []
    for name in sorted(dir(ops_namespace)):
        if name.startswith("_"):
            continue
        idname = f"fluidsim.{name}"
        if idname in EXCLUDED_OPERATORS:
            continue
        try:
            rna = getattr(ops_namespace, name).get_rna_type()
        except (AttributeError, KeyError, RuntimeError):
            continue
        entries.append(Entry("operator", idname, str(getattr(rna, "name", "") or name),
                             str(getattr(rna, "description", "") or "")))
    return entries


def _rna_property(group, name):
    try:
        return group.bl_rna.properties[name]
    except (AttributeError, KeyError, TypeError):
        return None


def _value_type(group, name, value):
    prop = _rna_property(group, name)
    if prop is not None:
        if getattr(prop, "is_array", False):
            return "VECTOR"
        return str(prop.type)
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, str):
        return "ENUM"
    try:
        tuple(value)
    except TypeError:
        return "POINTER"
    return "VECTOR"


def exact_value_text(value, value_type: str) -> str:
    """The shortest text that reads back as ``value`` (a Blender float is 32-bit).

    The inline editor starts from this, so pressing Enter without typing keeps
    the value instead of writing the rounded one the row shows.
    """
    if value_type == "INT":
        return str(int(value))
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    target = struct.pack("<f", number)
    for digits in range(6, 18):
        text = f"{number:.{digits}g}"
        if struct.pack("<f", float(text)) == target:
            return text
    return repr(number)


def format_value(value, value_type: str) -> str:
    if value_type == "BOOLEAN":
        return "On" if value else "Off"
    if value_type == "INT":
        return str(int(value))
    if value_type == "FLOAT":
        return f"{float(value):.4g}"
    if value_type == "VECTOR":
        return "(" + ", ".join(f"{float(v):.3g}" for v in value) + ")"
    if value_type == "POINTER":
        return str(getattr(value, "name", "") or "—")
    return str(value)


def setting_entries(scene) -> list[Entry]:
    """Scene and domain settings with their current value, from the group that owns each."""
    from ..pending_settings import SETTING_TRANSITION_REGISTRY, SettingTier, declaring_group

    hidden = {SettingTier.PRESENTATION, SettingTier.ADAPTER}
    entries = []
    for name in sorted(SETTING_TRANSITION_REGISTRY):
        if SETTING_TRANSITION_REGISTRY[name] in hidden:
            continue
        group = declaring_group(scene, name)
        if group is None or not hasattr(group, name):
            continue
        value = getattr(group, name)
        value_type = _value_type(group, name, value)
        prop = _rna_property(group, name)
        label = str(getattr(prop, "name", "") or name.replace("_", " ").title())
        editable = value_type in _EDITABLE_TYPES
        entries.append(Entry(
            "setting", name, label, str(getattr(prop, "description", "") or ""),
            format_value(value, value_type), value_type, editable,
            "" if editable else JUMP_OPERATOR))
    return entries


def preset_entries(records=None) -> list[Entry]:
    """Presets as ``source:id`` targets, the identifiers ``product_preset`` takes."""
    if records is None:
        from ...presets.library import discover_presets

        records = discover_presets()
    return [Entry("preset", f"{record.source}:{record.preset.id}", str(record.preset.name),
                  str(getattr(record.preset, "description", "") or ""))
            for record in records]


def demo_entries() -> list[Entry]:
    from ..demo_gallery import DEMO_CATALOG

    return [Entry("demo", demo_id, str(demo.get("title", demo_id)),
                  str(demo.get("description", "")))
            for demo_id, demo in DEMO_CATALOG.items()]


def build_index(scene) -> list[Entry]:
    entries = operator_entries()
    for source in (lambda: setting_entries(scene), preset_entries, demo_entries):
        try:
            entries.extend(source())
        except Exception as exc:  # noqa: BLE001 - one bad source must not hide the rest
            print(f"FluidSim Liquid Deck: palette index skipped a source: {exc!r}")
    return entries


# Matching -----------------------------------------------------------------------

def fuzzy_score(query: str, text: str):
    """Score an ordered, case-insensitive subsequence match; ``None`` when absent.

    Consecutive characters and matches at word starts score higher, and gaps
    cost a little, so ``bake`` ranks "Bake Start" above "Big Area Kinetic Effect".
    """
    query = query.strip().lower()
    if not query:
        return 0
    text_lower = text.lower()
    score = 0.0
    position = -1
    previous = -2
    for char in query:
        if char == " ":
            continue
        position = text_lower.find(char, position + 1)
        if position < 0:
            return None
        score += 1.0
        if position == previous + 1:
            score += 5.0
        if position == 0 or not text_lower[position - 1].isalnum():
            score += 3.0
        if previous >= 0:
            score -= 0.1 * (position - previous - 1)
        previous = position
    return score


def search(query: str, entries) -> list[Entry]:
    scored = []
    for order, entry in enumerate(entries):
        best = None
        for text in (entry.label, entry.target):
            value = fuzzy_score(query, text)
            if value is not None and (best is None or value > best):
                best = value
        if best is not None:
            scored.append((-best, order, entry))
    return [entry for _score, _order, entry in sorted(scored, key=lambda t: t[:2])]


# Availability -------------------------------------------------------------------

APPLY_PRESET_OPERATOR = "fluidsim.apply_product_preset"
LOAD_DEMO_OPERATOR = "fluidsim.load_demo"
MISSING_DEMO_REASON = "(file missing)"
_NO_DOMAIN_REASON = "Designate a FluidSim domain object before applying a preset"


def _operator_status(context, idname: str, label: str = "") -> tuple[bool, str]:
    return controls.control_status(
        context, controls.Control("operator", idname, "", label or idname, "", ""))


def missing_demo_ids() -> frozenset:
    """Catalogue demos whose .blend is not on disk (the sidebar gallery's check)."""
    from ..demo_gallery import DEMO_CATALOG, available_demo_ids

    return frozenset(DEMO_CATALOG) - frozenset(available_demo_ids())


def _has_domain(scene) -> bool:
    try:
        from ..scene_access import active_domain_settings

        return active_domain_settings(scene) is not None
    except Exception:  # noqa: BLE001 - unknown is not a refusal; the operator decides
        return True


def row_status(context, entry: Entry, missing_demos=frozenset()) -> tuple[bool, str]:
    """``(can run now, why not)`` for one palette row, as the wheel greys its slots.

    Operator rows use their operator's ``poll`` and its message; a preset row
    uses the apply operator's; a demo row is also unavailable when its .blend is
    missing. Setting rows are always available: their writes go through the
    property, so the Pending Settings bake hold applies to them.
    """
    if entry.kind == "operator":
        return _operator_status(context, entry.target, entry.label)
    if entry.kind == "demo":
        if entry.target in missing_demos:
            return False, MISSING_DEMO_REASON
        return _operator_status(context, LOAD_DEMO_OPERATOR)
    if entry.kind == "preset":
        ok, reason = _operator_status(context, APPLY_PRESET_OPERATOR)
        if ok and not _has_domain(getattr(context, "scene", None)):
            return False, _NO_DOMAIN_REASON
        return ok, reason
    return True, ""


# Inline edit ----------------------------------------------------------------------

def _bake_hold_message() -> str:
    """The Pending Settings bake-hold message while any Production Bake runs, else ``""``."""
    try:
        from .. import pending_settings, production_bake

        return pending_settings._BAKE_HOLD_MESSAGE if production_bake.any_bake_active() else ""
    except Exception:  # noqa: BLE001 - no bake module means no bake
        return ""


# Tiers whose edits reach the running simulation or its outputs straight away.
_IMMEDIATE_TIERS = frozenset({"LIVE", "SYNC", "OUTPUT", "SURFACE", "WHITEWATER"})


def _write_feedback(scene, entry: Entry) -> str:
    """Say what a palette write did, by the field's Pending Settings tier.

    Only fields that apply straight away flash the LIVE chip. A Reset or
    Reinitialize field is staged: the pending chip shows it when a solver is
    running, and the message says when it applies. During a Production Bake every
    edit is held, as it is from the panels.
    """
    from ..pending_settings import SETTING_TRANSITION_REGISTRY, pending_presentation

    hold = _bake_hold_message()
    if hold:
        return hold
    tier = getattr(SETTING_TRANSITION_REGISTRY.get(entry.target), "value", "")
    if tier in _IMMEDIATE_TIERS:
        bar.note_deck_write(scene, entry.target)
        return ""
    if tier in {"RESET", "REINITIALIZE"}:
        try:
            summary = str(pending_presentation(scene).get("summary", ""))
        except Exception:  # noqa: BLE001 - the message falls back to the tier's wording
            summary = ""
        when = ("after reset" if tier == "RESET"
                else "when the solver is reinitialized")
        return summary or f"{entry.label} is staged and will apply {when}"
    return f"{entry.label} is set; it takes effect on the next run"


def _push_undo(message: str) -> None:
    """One undo step for a palette write; a modal operator gets none of its own."""
    ed = getattr(getattr(bpy, "ops", None), "ed", None)
    push = getattr(ed, "undo_push", None)
    if push is None:
        return
    try:
        push(message=message)
    except Exception as exc:  # noqa: BLE001 - a missed undo step must not lose the edit
        print(f"FluidSim Liquid Deck: palette could not push an undo step: {exc!r}")


def commit_inline(scene, entry: Entry, text: str) -> str:
    """Write an inline edit through the setting's own property; return what to tell the user.

    A boolean toggles; an int or float is parsed from ``text``. Anything else, or
    text that is not a finite number, raises ``ValueError`` and writes nothing.
    Blender clamps to the property's range and runs its ``update=`` callback, so
    Pending Settings classifies and holds the edit exactly as for a panel edit.
    """
    if entry.kind != "setting" or not entry.editable:
        raise ValueError(f"{entry.target} is read-only in the palette; edit it in the panel")
    from ..pending_settings import declaring_group

    group = declaring_group(scene, entry.target)
    if group is None:
        raise ValueError(f"no settings group holds {entry.target}")
    if entry.value_type == "BOOLEAN":
        value = not bool(getattr(group, entry.target))
    else:
        try:
            number = float(str(text).strip())
        except ValueError:
            raise ValueError(f"{text!r} is not a number") from None
        if not math.isfinite(number):
            raise ValueError(f"{text!r} is not a finite number")
        value = int(round(number)) if entry.value_type == "INT" else number
    setattr(group, entry.target, value)
    _push_undo(f"Set {entry.label}")
    return _write_feedback(scene, entry)


# Operator ------------------------------------------------------------------------

# What a palette filtered to one kind (the wheel's Preset and Demo slots) prompts.
KIND_PROMPTS = {"preset": "Choose a preset…", "demo": "Choose a demo…"}


def _is_shortcut(event) -> bool:
    """A Ctrl or Cmd chord is a shortcut, not text; Ctrl+Alt is AltGr text on Windows."""
    return (bool(getattr(event, "ctrl", False) or getattr(event, "oskey", False))
            and not getattr(event, "alt", False))


def _card_rect(region, scale):
    width = min(region.width - 40.0, 560.0 * scale)
    row = 22.0 * scale
    height = row * (VISIBLE_ROWS + 1) + 24.0 * scale
    x0 = (region.width - width) / 2.0
    y1 = region.height * 0.72
    return x0, y1 - height, x0 + width, y1, row


def _fit(text: str, size: float, width: float) -> str:
    """``text`` cut with an ellipsis to fit ``width`` pixels."""
    if width <= 0 or _draw.text_width(text, size) <= width:
        return text
    while text and _draw.text_width(text + "…", size) > width:
        text = text[:-1]
    return text + "…"


def _draw_palette(operator):
    context = bpy.context
    try:
        if getattr(context, "region", None) != operator.region:
            return
        palette = tokens.current_tokens(context)
        scale = bar._ui_scale(context)
        x0, y0, x1, y1, row = _card_rect(context.region, scale)
        size = 12.0 * scale
        radius = tokens.RADIUS_SCALE[2] * scale
        _draw.rounded_rect(x0, y0, x1, y1, radius, tokens.rgba(palette.surface, tokens.CARD_ALPHA))
        prompt = operator.edit_text if operator.editing else operator.query
        if operator.editing and getattr(operator, "edit_selected", False):
            # The whole value is selected: the first typed character replaces it.
            head = f"{operator.editing.label} = "
            left = x0 + 12 * scale + _draw.text_width(head, size)
            _draw.rounded_rect(left - 2 * scale, y1 - row - 4 * scale,
                               left + _draw.text_width(prompt, size) + 2 * scale,
                               y1 - row + size, tokens.RADIUS_SCALE[0] * scale,
                               tokens.rgba(palette.accent, 0.35))
        caption = (f"{operator.editing.label} = {prompt}_" if operator.editing
                   else (prompt + "_" if prompt else KIND_PROMPTS.get(operator.kind_filter,
                                                                      "Search FluidSim…")))
        _draw.text(x0 + 12 * scale, y1 - row, caption, size,
                   tokens.rgba(palette.text if prompt or operator.editing else palette.text_dim))
        missing = getattr(operator, "missing_demos", frozenset())
        value_x = x0 + (x1 - x0) * 0.58
        for index, entry in enumerate(operator.results[:VISIBLE_ROWS]):
            top = y1 - row * (index + 2) - 6 * scale
            ready, reason = row_status(context, entry, missing)
            if index == operator.selected:
                _draw.rounded_rect(x0 + 6 * scale, top - 5 * scale, x1 - 6 * scale,
                                   top + row - 7 * scale, tokens.RADIUS_SCALE[0] * scale,
                                   tokens.rgba(palette.accent, 0.25 if ready else 0.12))
            _draw.text(x0 + 12 * scale, top, entry.label, size,
                       tokens.rgba(palette.text if ready else palette.text_dim))
            kind = entry.kind
            kind_width = _draw.text_width(kind, size * 0.9)
            if not ready:
                _draw.text(value_x, top, _fit(reason, size, x1 - kind_width - 24 * scale - value_x),
                           size, tokens.rgba(palette.text_dim))
            elif entry.value:
                _draw.text(value_x, top, entry.value, size, tokens.rgba(palette.accent))
            _draw.text(x1 - kind_width - 12 * scale, top, kind,
                       size * 0.9, tokens.rgba(palette.text_dim))
    except Exception as exc:  # noqa: BLE001 - a draw callback must never raise
        print(f"FluidSim Liquid Deck: palette draw failed: {exc!r}")


class FLUIDSIM_OT_command_palette(bpy.types.Operator):
    """Search FluidSim operators, presets, demos and settings, and run or edit one."""

    bl_idname = "fluidsim.command_palette"
    bl_label = "FluidSim Command Palette"
    bl_description = (
        "Search FluidSim operators, presets, demos and settings; Enter runs an "
        "operator or edits a number inline. Click outside to close")

    kinds: bpy.props.StringProperty(
        name="Kinds", description="Show only entries of this kind (preset, demo); empty shows all",
        default="", options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        from . import deck_visible

        return (getattr(getattr(context, "area", None), "type", "") == "VIEW_3D"
                and deck_visible(context))

    def invoke(self, context, event):
        from .registry import REGISTRY

        self.entries = build_index(context.scene)
        self.kind_filter = str(getattr(self, "kinds", "") or "")
        if self.kind_filter:
            self.entries = [entry for entry in self.entries if entry.kind == self.kind_filter]
        self.query = ""
        self.results = search("", self.entries)
        self.selected = 0
        self.editing = None
        self.edit_text = ""
        self.edit_start = ""
        self.edit_selected = False
        try:
            self.missing_demos = missing_demo_ids()
        except Exception as exc:  # noqa: BLE001 - loading still reports a missing file
            print(f"FluidSim Liquid Deck: palette could not check the demo files: {exc!r}")
            self.missing_demos = frozenset()
        self.region = context.region
        # Replacing the handler retires a palette still open in another window.
        self.handle = REGISTRY.add_draw_handler("palette", bpy.types.SpaceView3D,
                                                _draw_palette, (self,), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _close(self, context):
        from .registry import REGISTRY

        if REGISTRY.draw_handle("palette") is getattr(self, "handle", None):
            try:
                REGISTRY.remove_draw_handler("palette")
            except Exception as exc:  # noqa: BLE001
                print(f"FluidSim Liquid Deck: could not remove the palette draw handler: {exc!r}")
        if getattr(context, "area", None) is not None:
            context.area.tag_redraw()

    def _refresh(self):
        self.results = search(self.query, self.entries)
        self.selected = 0

    def _call(self, idname, **properties):
        """Run ``idname``; ``None`` (keep the palette open) when it refused with a reason."""
        result, message = controls.call_operator(idname, **properties)
        if message:
            self.report({"WARNING"}, message)
            if "CANCELLED" in result:
                return None
        return result

    def _commit(self, context, entry, text):
        message = commit_inline(context.scene, entry, text)
        if message:
            self.report({"INFO"}, message)

    def _start_edit(self, context, entry):
        from ..pending_settings import declaring_group

        group = declaring_group(context.scene, entry.target)
        value = getattr(group, entry.target, None) if group is not None else None
        self.editing = entry
        self.edit_text = (exact_value_text(value, entry.value_type) if value is not None
                          else entry.value)
        self.edit_start = self.edit_text
        self.edit_selected = True

    def _run(self, context, entry):
        """Act on a row; ``None`` keeps the palette open (refused, or editing)."""
        ready, reason = row_status(context, entry, self.missing_demos)
        if not ready:
            self.report({"WARNING"}, f"{entry.label}: {reason}")
            return None
        if entry.kind == "operator":
            return self._call(entry.target)
        if entry.kind == "demo":
            return self._call(LOAD_DEMO_OPERATOR, demo_id=entry.target)
        if entry.kind == "preset":
            try:
                context.scene.fluidsim.product_preset = entry.target
            except (TypeError, ValueError) as exc:
                self.report({"WARNING"}, f"{entry.label}: this preset is no longer available ({exc})")
                return None
            return self._call(APPLY_PRESET_OPERATOR)
        if not entry.editable:
            return self._call(entry.jump_operator)
        if entry.value_type == "BOOLEAN":
            self._commit(context, entry, "")
            return {"FINISHED"}
        self._start_edit(context, entry)
        return None

    def _outside(self, context, event):
        x0, y0, x1, y1, _row = _card_rect(context.region, bar._ui_scale(context))
        return not (x0 <= event.mouse_region_x <= x1 and y0 <= event.mouse_region_y <= y1)

    def modal(self, context, event):
        from .registry import REGISTRY

        area = getattr(context, "area", None)
        if area is None or REGISTRY.draw_handle("palette") is not self.handle:
            # The area closed, or a newer palette took the handler: finish, keep the event.
            self._close(context)
            return {"CANCELLED", "PASS_THROUGH"}
        area.tag_redraw()
        if event.type in {"LEFTMOUSE", "RIGHTMOUSE"} and event.value == "PRESS":
            if self._outside(context, event):
                self._close(context)
                return {"CANCELLED"}
            return {"RUNNING_MODAL"}
        if event.value != "PRESS":
            return {"PASS_THROUGH"}
        if event.type == "ESC":
            if self.editing is not None:
                self.editing = None
                return {"RUNNING_MODAL"}
            self._close(context)
            return {"CANCELLED"}
        if event.type in {"RET", "NUMPAD_ENTER"}:
            if self.editing is not None:
                try:
                    if self.edit_text != self.edit_start:  # unchanged: nothing to write
                        self._commit(context, self.editing, self.edit_text)
                except ValueError as exc:
                    self.report({"WARNING"}, str(exc))
                    return {"RUNNING_MODAL"}
                self._close(context)
                return {"FINISHED"}
            if not self.results:
                return {"RUNNING_MODAL"}
            result = self._run(context, self.results[self.selected])
            if result is None:
                return {"RUNNING_MODAL"}
            self._close(context)
            return {"FINISHED"}
        if event.type in {"UP_ARROW", "DOWN_ARROW"}:
            step = -1 if event.type == "UP_ARROW" else 1
            limit = min(len(self.results), VISIBLE_ROWS)
            if limit:
                self.selected = (self.selected + step) % limit
            return {"RUNNING_MODAL"}
        if event.type == "BACK_SPACE":
            if self.editing is not None:
                # Backspace edits the value rather than clearing the selection.
                self.edit_selected = False
                self.edit_text = self.edit_text[:-1]
            else:
                self.query = self.query[:-1]
                self._refresh()
            return {"RUNNING_MODAL"}
        if event.unicode and event.unicode.isprintable() and not _is_shortcut(event):
            if self.editing is not None:
                if self.edit_selected:
                    self.edit_text = ""
                    self.edit_selected = False
                self.edit_text += event.unicode
            else:
                self.query += event.unicode
                self._refresh()
            return {"RUNNING_MODAL"}
        if _is_shortcut(event):
            return {"PASS_THROUGH"}  # Ctrl+Z, Ctrl+Shift+Z and other shortcuts
        # Any other key stays with the palette: Delete or X reaching the viewport
        # would delete the selection while the user is typing.
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._close(context)


CLASSES = (FLUIDSIM_OT_command_palette,)
