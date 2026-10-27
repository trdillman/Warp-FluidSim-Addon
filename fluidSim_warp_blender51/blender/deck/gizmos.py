"""Liquid Deck gizmos: the emitter, collider and domain handled in the viewport.

Each gizmo writes the property the Object or Scene panel draws, through plain
attribute assignment, so Blender runs the property's own ``update=`` callback and
Pending Settings classifies the edit exactly as for a panel edit (spec trap 5).

- Emitter (role ``SOURCE``): a velocity arrow that writes ``initial_velocity``
  (the spec's ``emitter_velocity``) and a fill dial that writes the property the
  Object panel shows for the source's Fill Target: ``source_sampling_multiplier``
  for "Sampling Multiplier", ``emit_rate`` for "Legacy Absolute", and no dial for
  "Inherit". Both rate properties are Reset-class, so the dial shows the RESET
  chip, as the panel does (the mockup's "LIVE" does not match the code).
- Collider (role ``COLLIDER``): a friction dial (``collider_friction``) and a
  Dynamic toggle bound to ``collider_dynamic`` only when that property exists (it
  arrives with the 4.0 rigid-coupling lane; until then no toggle is created).
- Domain: six face handles. Dragging writes the domain object's location and
  scale on every step, as typing its dimensions would, so the viewport follows
  the handle; the opposite face stays in place. The overlay previews the resized
  grid (cell counts, padded grid, cell size, from ``resolve_quality`` as the
  Scene panel's readout computes it) until release. ``resolution_divisions`` is unchanged: resizing changes
  the cell size, not the count. The domain bounds are Reinitialize-class, so the
  chip shows REINIT. The handles are offered only for an unparented, unrotated,
  positively scaled, unconstrained domain. The box is read from ``matrix_world``
  (so delta transforms count), and a face drag changes location and scale by the
  world-space offset and size ratio, which ``matrix_world`` follows exactly there.

The gizmos keep only drag bookkeeping (start value, start pointer) between
``invoke`` and ``exit``; a cancelled drag restores the start value. Gizmos write
only to the object the drag started on, and do nothing once that object has left
the scene (the active object may have changed or gone in the meantime).

Shift slows the drag from the moment it is pressed (no jump), Ctrl snaps to
:data:`SNAP_STEPS`. A committed drag or click pushes one undo step; a cancelled
one pushes none. While a Production Bake owns the scene the handles draw dimmed
and do not start a drag.
"""

from __future__ import annotations

from dataclasses import dataclass

import bpy

from . import _draw, bar, controls, tokens

ARROW_SCALE = 0.25  # metres of arrow per m/s
DIAL_PIXELS = 300.0  # horizontal drag that sweeps a dial's whole range
PRECISE_FACTOR = 0.1
MIN_DOMAIN_SIZE = 0.05  # metres; a face cannot be dragged past this
CLICK_PIXELS = 3.0  # pointer travel that turns a toggle click into a drag (Blender's default)
# Fallback ``(hard_min, hard_max, soft_min, soft_max)`` per dial property, used only
# when the settings expose no RNA (headless doubles); in Blender the dial reads the
# property's own range (:func:`dial_range`), so these mirror ``properties.py``.
DIAL_RANGES = {
    "source_sampling_multiplier": (0.0, 8.0, 0.0, 8.0),
    "emit_rate": (0.0, 64.0, 0.0, 32.0),
    "collider_friction": (0.0, 1.0, 0.0, 1.0),
}
# Ctrl snaps the value to these steps. Velocity in 0.5 m/s per axis; the Fill
# Multiplier in quarters; the legacy particles-per-voxel rate in whole particles;
# friction in 0.05. Domain faces snap the dragged size to 0.1 m: whole voxels
# would not stay whole, because a resize changes the cell size, not the count.
SNAP_STEPS = {
    "initial_velocity": 0.5,
    "source_sampling_multiplier": 0.25,
    "emit_rate": 1.0,
    "collider_friction": 0.05,
    "domain_size": 0.1,
}
DIM_ALPHA = 0.3  # handle opacity while a Production Bake holds edits
UNDO_MESSAGES = {
    "initial_velocity": "Liquid Deck: Emitter Velocity",
    "source_sampling_multiplier": "Liquid Deck: Fill",
    "emit_rate": "Liquid Deck: Fill",
    "collider_friction": "Liquid Deck: Friction",
    "collider_dynamic": "Liquid Deck: Dynamic",
    "domain": "Liquid Deck: Resize Domain",
}

_ARROW_LINES = (
    (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0), (0.08, 0.0, 0.85), (0.0, 0.0, 1.0), (-0.08, 0.0, 0.85),
    (0.0, 0.0, 1.0), (0.0, 0.08, 0.85), (0.0, 0.0, 1.0), (0.0, -0.08, 0.85),
)


def _circle_lines(radius=1.0, segments=32):
    import math

    points = [(radius * math.cos(2 * math.pi * i / segments),
               radius * math.sin(2 * math.pi * i / segments), 0.0) for i in range(segments)]
    lines = []
    for i in range(segments):
        lines.extend((points[i], points[(i + 1) % segments]))
    return tuple(lines)


_SQUARE_LINES = ((-1, -1, 0), (1, -1, 0), (1, -1, 0), (1, 1, 0),
                 (1, 1, 0), (-1, 1, 0), (-1, 1, 0), (-1, -1, 0))


def mouse_to_world(context, mouse, depth):
    """World point under region pixel ``mouse`` at the view depth of ``depth``."""
    from bpy_extras.view3d_utils import region_2d_to_location_3d

    point = region_2d_to_location_3d(context.region, context.region_data, mouse, depth)
    return tuple(float(c) for c in point)


def _mouse(event):
    return (float(event.mouse_region_x), float(event.mouse_region_y))


def _object_origin(obj):
    matrix = getattr(obj, "matrix_world", None)
    if matrix is not None:
        return tuple(float(c) for c in matrix.translation)
    return tuple(float(c) for c in obj.location)


def _started_object(context, target):
    """``target`` while it is still in the scene, else ``None``.

    Matched with ``==`` (same datablock), so a deleted object is never read.
    """
    try:
        for obj in getattr(getattr(context, "scene", None), "objects", ()):
            if obj == target:
                return obj
    except ReferenceError:  # ``target`` was removed through ``bpy.data``
        pass
    return None


def _write(context, settings, name, value):
    """Assign through the property, so its ``update=`` callback runs."""
    setattr(settings, name, value)
    scene = getattr(context, "scene", None)
    if scene is not None:
        bar.note_deck_write(scene, name)


def push_undo(name):
    """One undo step for a committed gizmo edit.

    Blender's gizmo tweak operator registers no undo, so the gizmo pushes its own.
    A missing or refusing ``ed.undo_push`` (``--background``, headless doubles)
    leaves the edit in place without an undo step.
    """
    push = getattr(getattr(getattr(bpy, "ops", None), "ed", None), "undo_push", None)
    if push is None:
        return
    try:
        push(message=UNDO_MESSAGES.get(name, "Liquid Deck"))
    except RuntimeError:
        pass


def bake_holds_edits(context) -> bool:
    """Whether a Production Bake owns this scene (or the session, from another scene).

    The same rule as the operators' bake poll: edits made now would only be held.
    """
    from .. import production_bake

    scene = getattr(context, "scene", None)
    if scene is not None and production_bake.bake_is_active(scene):
        return True
    return bool(production_bake.any_bake_active())


def snap(value, step):
    return round(float(value) / step) * step


_TWEAK_ACTIONS = {
    "PRECISE": ("PRECISION_ON", "PRECISION_OFF"),
    "SNAP": ("SNAP_ON", "SNAP_OFF"),
}
_MODIFIER_KEYS = {
    "shift": ("LEFT_SHIFT", "RIGHT_SHIFT"),
    "ctrl": ("LEFT_CTRL", "RIGHT_CTRL"),
    "alt": ("LEFT_ALT", "RIGHT_ALT"),
    "oskey": ("OSKEY",),
}


def _paired_modifier_action(items, keys):
    """An unambiguous ON/OFF action shared by both sides of a modifier.

    The mouse-down event reports a modifier family, not which side is held.
    Infer initial state only for equivalent, unconditional press/release pairs.
    """
    for flag, (on, off) in _TWEAK_ACTIONS.items():
        if all({(item.value, item.propvalue) for item in items if item.type == key}
               == {("PRESS", on), ("RELEASE", off)} for key in keys):
            return flag
    return None


class DragModifiers:
    """Bridge held-at-invoke modifiers until Blender's native tweak takes over.

    Blender starts gizmo tweak flags empty, even when a modifier was held before
    mouse-down. Seed that initial state from the effective user keymap, and retire
    it permanently on release or a native ON/OFF transition. Never re-arm from
    raw modifier bits: subsequent changes belong to the native modal keymap.
    """

    def __init__(self, context, event):
        self.initial = {}
        self.items = ()
        wm = getattr(context, "window_manager", None)
        config = getattr(getattr(wm, "keyconfigs", None), "user", None)
        keymaps = getattr(config, "keymaps", None)
        keymap = keymaps.get("Generic Gizmo Tweak Modal Map") if keymaps is not None else None
        if keymap is None:
            return
        active = tuple(item for item in keymap.keymap_items if item.active)
        actions = {action for pair in _TWEAK_ACTIONS.values() for action in pair}
        types = {item.type for item in active if item.propvalue in actions}
        # Include competing entries for these key types, in native order. Public
        # Event fields cannot reproduce arbitrary chord/conditional/mouse matching;
        # leave such maps entirely to Blender rather than guess a transition.
        items = tuple(item for item in active if item.type in types or item.type == "ANY")
        if any(not item.any or item.key_modifier != "NONE"
               or item.map_type != "KEYBOARD" or item.type in {"NONE", "ANY"}
               or item.value not in {"PRESS", "RELEASE"} for item in items):
            return
        self.items = items
        for modifier, keys in _MODIFIER_KEYS.items():
            if getattr(event, modifier, False):
                flag = _paired_modifier_action(items, keys)
                if flag is not None:
                    self.initial.setdefault(flag, set()).add(modifier)

    def resolve(self, event, tweak):
        flags = set(tweak)
        # keymap_items.match_event polls operator IDs, so it cannot match a modal
        # keymap's propvalue entries. These validated unconditional keyboard items
        # can be matched explicitly in their original order instead.
        action = next((item.propvalue for item in self.items
                       if item.active and item.type == event.type and item.value == event.value
                       and not (getattr(event, "is_repeat", False) and not item.repeat)), None)
        for flag, modifiers in tuple(self.initial.items()):
            if flag in flags or action in _TWEAK_ACTIONS[flag]:
                del self.initial[flag]
                continue
            modifiers.intersection_update(
                modifier for modifier in tuple(modifiers) if getattr(event, modifier, False))
            if modifiers:
                flags.add(flag)
            else:
                del self.initial[flag]
        return flags


class PrecisionDrag:
    """Pointer -> value, where Shift slows the motion from the point it was pressed.

    ``gain`` is value per unit of pointer travel. Whenever the precision state
    changes, the drag re-anchors at the last pointer position and the value it
    produced, so pressing or releasing Shift mid-drag never makes the value jump.
    Values and pointer positions are tuples of the same length.
    """

    def __init__(self, start_input, start_value, gain):
        self.anchor_input = tuple(float(c) for c in start_input)
        self.anchor_value = tuple(float(c) for c in start_value)
        self.last_input = self.anchor_input
        self.gain = float(gain)
        self.precise = False

    def _at(self, pointer):
        factor = self.gain * (PRECISE_FACTOR if self.precise else 1.0)
        return tuple(v + (p - a) * factor
                     for v, p, a in zip(self.anchor_value, pointer, self.anchor_input,
                                        strict=True))

    def value(self, pointer, precise):
        pointer = tuple(float(c) for c in pointer)
        precise = bool(precise)
        if precise != self.precise:
            self.anchor_value = self._at(self.last_input)
            self.anchor_input = self.last_input
            self.precise = precise
        self.last_input = pointer
        return self._at(pointer)


def dial_range(settings, prop):
    """``(low, high, span)``: clamp to the property's hard range, sweep its soft range.

    Read from the property's RNA so the dial allows what the panel allows; the
    ``DIAL_RANGES`` fallback serves only settings without RNA.
    """
    hard_min, hard_max, soft_min, soft_max = DIAL_RANGES[prop]
    try:
        rna = settings.bl_rna.properties[prop]
        hard_min, hard_max = float(rna.hard_min), float(rna.hard_max)
        soft_min, soft_max = float(rna.soft_min), float(rna.soft_max)
    except (AttributeError, KeyError, TypeError):
        pass
    span = soft_max - soft_min
    if span <= 0.0:
        span = hard_max - hard_min
    return hard_min, hard_max, span


def rate_property(settings):
    """The fill property the Object panel draws for this source, or ``None``."""
    mode = str(getattr(settings, "source_sampling_mode", ""))
    if mode == "MULTIPLIER":
        return "source_sampling_multiplier"
    if mode == "LEGACY_ABSOLUTE":
        return "emit_rate"
    return None


def dynamic_property(settings):
    """``collider_dynamic`` when the rigid-coupling property exists, else ``None``."""
    return "collider_dynamic" if hasattr(settings, "collider_dynamic") else None


def dial_property(settings, binding):
    if binding == "rate":
        return rate_property(settings)
    if binding == "friction":
        return "collider_friction"
    return None


# Gizmo types ------------------------------------------------------------------

class FLUIDSIM_GT_deck_velocity(bpy.types.Gizmo):
    """Drag the arrow head to set the emitter's initial velocity."""

    bl_idname = "FLUIDSIM_GT_deck_velocity"
    __slots__ = ("shape", "start_value", "target", "drag", "changed", "modifiers")

    def setup(self):
        self.shape = self.new_custom_shape("LINES", _ARROW_LINES)

    def draw(self, context):
        self.draw_custom_shape(self.shape)

    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.shape, select_id=select_id)

    def invoke(self, context, event):
        self.target, self.changed = None, False
        if bake_holds_edits(context):
            return {"CANCELLED"}
        self.modifiers = DragModifiers(context, event)
        obj = self.target = context.active_object
        self.start_value = tuple(float(c) for c in obj.fluidsim.initial_velocity)
        start_world = mouse_to_world(context, _mouse(event), _object_origin(obj))
        self.drag = PrecisionDrag(start_world, self.start_value, 1.0 / ARROW_SCALE)
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        obj = _started_object(context, getattr(self, "target", None))
        if obj is None:
            return {"RUNNING_MODAL"}
        tweak = self.modifiers.resolve(event, tweak)
        now = mouse_to_world(context, _mouse(event), _object_origin(obj))
        value = self.drag.value(now, "PRECISE" in tweak)
        if "SNAP" in tweak:
            value = tuple(snap(c, SNAP_STEPS["initial_velocity"]) for c in value)
        _write(context, obj.fluidsim, "initial_velocity", value)
        self.changed = True
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        obj = _started_object(context, getattr(self, "target", None))
        if obj is None or not getattr(self, "changed", False):
            return
        if cancel:
            _write(context, obj.fluidsim, "initial_velocity", self.start_value)
        else:
            push_undo("initial_velocity")


class FLUIDSIM_GT_deck_dial(bpy.types.Gizmo):
    """Drag sideways to turn a dial bound to an object setting (``binding``)."""

    bl_idname = "FLUIDSIM_GT_deck_dial"
    __slots__ = ("shape", "binding", "start_value", "prop", "target", "drag", "range",
                 "changed", "modifiers")

    def setup(self):
        self.shape = self.new_custom_shape("LINES", _circle_lines())

    def draw(self, context):
        self.draw_custom_shape(self.shape)

    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.shape, select_id=select_id)

    def invoke(self, context, event):
        self.target, self.prop, self.changed = None, None, False
        if bake_holds_edits(context):
            return {"CANCELLED"}
        self.modifiers = DragModifiers(context, event)
        self.target = context.active_object
        settings = self.target.fluidsim
        self.prop = dial_property(settings, getattr(self, "binding", ""))
        if self.prop is None:
            return {"CANCELLED"}
        self.start_value = float(getattr(settings, self.prop))
        self.range = dial_range(settings, self.prop)
        low, high, span = self.range
        self.drag = PrecisionDrag((float(event.mouse_region_x),), (self.start_value,),
                                  span / DIAL_PIXELS)
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        obj = _started_object(context, getattr(self, "target", None))
        if obj is None or getattr(self, "prop", None) is None:
            return {"RUNNING_MODAL"}
        tweak = self.modifiers.resolve(event, tweak)
        low, high, _span = self.range
        (value,) = self.drag.value((float(event.mouse_region_x),), "PRECISE" in tweak)
        if "SNAP" in tweak:
            value = snap(value, SNAP_STEPS[self.prop])
        _write(context, obj.fluidsim, self.prop, min(high, max(low, value)))
        self.changed = True
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        if getattr(self, "prop", None) is None or not getattr(self, "changed", False):
            return
        obj = _started_object(context, getattr(self, "target", None))
        if obj is None:
            return
        if cancel:
            _write(context, obj.fluidsim, self.prop, self.start_value)
        else:
            push_undo(self.prop)


class FLUIDSIM_GT_deck_toggle(bpy.types.Gizmo):
    """Click to flip the collider's Dynamic (rigid coupling) setting.

    Only a click flips it: the press starts on the toggle (Blender invokes the
    gizmo under the pointer) and the release must come before the pointer has
    travelled :data:`CLICK_PIXELS`, i.e. still on the toggle. A press that turns
    into a drag never flips it, wherever it is released.
    """

    bl_idname = "FLUIDSIM_GT_deck_toggle"
    __slots__ = ("shape", "target", "press", "dragged")

    def setup(self):
        self.shape = self.new_custom_shape("LINES", _SQUARE_LINES)

    def draw(self, context):
        self.draw_custom_shape(self.shape)

    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.shape, select_id=select_id)

    def invoke(self, context, event):
        self.target, self.dragged = None, False
        if bake_holds_edits(context):
            return {"CANCELLED"}
        self.target = context.active_object
        self.press = _mouse(event)
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        x, y = _mouse(event)
        px, py = getattr(self, "press", (x, y))
        if max(abs(x - px), abs(y - py)) > CLICK_PIXELS:
            self.dragged = True
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        if cancel or getattr(self, "dragged", False):
            return
        obj = _started_object(context, getattr(self, "target", None))
        if obj is None:
            return
        settings = obj.fluidsim
        prop = dynamic_property(settings)
        if prop is not None:
            _write(context, settings, prop, not bool(getattr(settings, prop)))
            push_undo(prop)


# Domain resize ------------------------------------------------------------------

def face_resize(low, high, scale, side, delta, snap_step=None):
    """``(new_scale, location_shift)`` for growing one face outward by ``delta`` metres.

    ``low``/``high`` are the object's local bounds on the axis, ``scale`` its
    world scale on the axis, ``side`` is +1 for the positive face and -1 for the
    negative one. The opposite face stays put, and the size never drops below
    :data:`MIN_DOMAIN_SIZE`. ``snap_step`` (Ctrl) rounds the new size to that many
    metres.
    """
    extent = float(high) - float(low)
    scale = float(scale)
    size = abs(extent * scale)
    if extent <= 0.0 or size <= 0.0:
        return scale, 0.0
    new_size = size + float(delta)
    if snap_step:
        new_size = snap(new_size, snap_step)
    new_size = max(MIN_DOMAIN_SIZE, new_size)
    new_scale = scale * new_size / size
    anchor = low if side > 0 else high
    return new_scale, anchor * (scale - new_scale)


def local_bounds(obj):
    corners = [tuple(float(c) for c in corner) for corner in obj.bound_box]
    return (tuple(min(c[i] for c in corners) for i in range(3)),
            tuple(max(c[i] for c in corners) for i in range(3)))


def world_axes(obj):
    """``(translation, per-axis world scale)`` of an axis-aligned object.

    Read from ``matrix_world``, so delta transforms count as Blender draws them;
    an object without one (headless doubles) falls back to location and scale.
    """
    matrix = getattr(obj, "matrix_world", None)
    if matrix is None:
        return (tuple(float(c) for c in obj.location), tuple(float(c) for c in obj.scale))
    return (tuple(float(matrix[i][3]) for i in range(3)),
            tuple(float(matrix[i][i]) for i in range(3)))


def world_bounds(location, scale, low, high):
    lo, hi = [], []
    for i in range(3):
        a = float(location[i]) + float(low[i]) * float(scale[i])
        b = float(location[i]) + float(high[i]) * float(scale[i])
        lo.append(min(a, b))
        hi.append(max(a, b))
    return tuple(lo), tuple(hi)


@dataclass(frozen=True)
class DomainPreview:
    bounds: tuple
    quality: object  # sim.quality_contract.ResolvedQuality
    divisions: int
    label: str


_ACTIVE_PREVIEW: DomainPreview | None = None


def active_domain_preview():
    """The grid preview of the domain drag in progress, or ``None``."""
    return _ACTIVE_PREVIEW


def domain_preview(scene, bounds) -> DomainPreview:
    """The grid the solver would build for ``bounds``, as the Scene panel's readout does."""
    from ...sim.quality_contract import (
        AuthoredQualitySettings,
        resolve_quality,
        scale_quality_value,
        voxel_size_for_divisions,
    )
    from ..scene_access import active_domain_settings
    from ..solver_settings import scene_quality_fidelity

    divisions = int(active_domain_settings(scene).resolution_divisions)
    longest = max(hi - lo for lo, hi in zip(*bounds))
    voxel = scale_quality_value("voxel_size", voxel_size_for_divisions(divisions, longest),
                                scene_quality_fidelity(scene))
    quality = resolve_quality(AuthoredQualitySettings(voxel_size=voxel), bounds)
    nx, ny, nz = quality.unpadded_voxel_dimensions
    px, py, pz = quality.padded_voxel_dimensions
    label = (f"{nx} × {ny} × {nz} cells · grid {px} × {py} × {pz} · "
             f"{quality.voxel_size:.4g} m · {divisions} divisions")
    return DomainPreview(bounds=bounds, quality=quality, divisions=divisions, label=label)


class FLUIDSIM_GT_deck_face(bpy.types.Gizmo):
    """Drag a domain face; preview the grid while dragging, commit on release."""

    bl_idname = "FLUIDSIM_GT_deck_face"
    __slots__ = ("shape", "axis", "side", "start_location", "start_scale", "start_bounds",
                 "world_origin", "world_scale", "local", "result", "target", "drag", "modifiers")

    def setup(self):
        self.shape = self.new_custom_shape("LINES", _SQUARE_LINES)

    def draw(self, context):
        self.draw_custom_shape(self.shape)

    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.shape, select_id=select_id)

    def _face_centre(self, bounds):
        lo, hi = bounds
        centre = [(a + b) / 2.0 for a, b in zip(lo, hi)]
        centre[self.axis] = hi[self.axis] if self.side > 0 else lo[self.axis]
        return tuple(centre)

    def invoke(self, context, event):
        global _ACTIVE_PREVIEW
        self.target, self.result = None, None
        if bake_holds_edits(context):
            return {"CANCELLED"}
        self.modifiers = DragModifiers(context, event)
        obj = self.target = context.active_object
        self.local = local_bounds(obj)
        self.start_location = tuple(float(c) for c in obj.location)
        self.start_scale = tuple(float(c) for c in obj.scale)
        # The world box comes from matrix_world, captured once: Blender does not
        # re-evaluate it between the drag's events.
        self.world_origin, self.world_scale = world_axes(obj)
        self.start_bounds = world_bounds(self.world_origin, self.world_scale, *self.local)
        start = mouse_to_world(context, _mouse(event), self._face_centre(self.start_bounds))
        self.drag = PrecisionDrag((self.side * start[self.axis],), (0.0,), 1.0)
        _ACTIVE_PREVIEW = domain_preview(context.scene, self.start_bounds)
        return {"RUNNING_MODAL"}

    def _apply(self, context, scale, location):
        """Write the face's axis of the domain it started on (gone: no-op)."""
        obj = _started_object(context, getattr(self, "target", None))
        if obj is None:
            return False  # the domain was deleted mid-drag
        obj.scale[self.axis] = scale
        obj.location[self.axis] = location
        return True

    def modal(self, context, event, tweak):
        global _ACTIVE_PREVIEW
        if getattr(self, "target", None) is None:
            return {"RUNNING_MODAL"}
        tweak = self.modifiers.resolve(event, tweak)
        axis = self.axis
        low, high = self.local
        now = mouse_to_world(context, _mouse(event), self._face_centre(self.start_bounds))
        (growth,) = self.drag.value((self.side * now[axis],), "PRECISE" in tweak)
        world_scale = self.world_scale[axis]
        new_world_scale, shift = face_resize(
            low[axis], high[axis], world_scale, self.side, growth,
            SNAP_STEPS["domain_size"] if "SNAP" in tweak else None)
        # Unparented and unrotated, location moves the world box one-to-one and
        # scale multiplies its size, also under delta transforms.
        ratio = new_world_scale / world_scale
        origin, scales = list(self.world_origin), list(self.world_scale)
        origin[axis] += shift
        scales[axis] = new_world_scale
        _ACTIVE_PREVIEW = domain_preview(context.scene, world_bounds(origin, scales, low, high))
        # Resize the domain while dragging, so the viewport follows the handle.
        if self._apply(context, self.start_scale[axis] * ratio,
                       self.start_location[axis] + shift):
            self.result = (ratio, shift)
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        global _ACTIVE_PREVIEW
        _ACTIVE_PREVIEW = None
        if getattr(self, "result", None) is None:
            return
        if cancel:
            self._apply(context, self.start_scale[self.axis], self.start_location[self.axis])
        elif _started_object(context, getattr(self, "target", None)) is not None:
            push_undo("domain")


# Gizmo groups ---------------------------------------------------------------------

def _visible(context) -> bool:
    from . import deck_visible

    return deck_visible(context)


def _active_role(context) -> str:
    obj = getattr(context, "active_object", None)
    return str(getattr(getattr(obj, "fluidsim", None), "role", "NONE"))


def _axis_aligned(obj) -> bool:
    """Unparented, unconstrained, unrotated and positively scaled.

    Only then does writing location and scale move ``matrix_world``'s box by the
    same offset and ratio (constraints could override or remap the write).
    """
    if getattr(obj, "parent", None) is not None:
        return False
    if any(getattr(c, "enabled", True) for c in getattr(obj, "constraints", ())):
        return False
    matrix = getattr(obj, "matrix_world", None)
    if matrix is None:
        return True
    return all(abs(matrix[r][c]) < 1e-6 for r in range(3) for c in range(3) if r != c) and all(
        matrix[i][i] > 0.0 for i in range(3))


def _place(gizmo, location, size=0.15):
    from mathutils import Matrix

    gizmo.matrix_basis = Matrix.Translation(location)
    gizmo.scale_basis = size


def show_held(group, context, gizmos):
    """Dim ``gizmos`` while a Production Bake holds edits; restore them after.

    Their ``invoke`` refuses a drag in that state, so the dimming is the cue.
    """
    held = bake_holds_edits(context)
    base = getattr(group, "base_alpha", None)
    if base is None:
        base = group.base_alpha = [(float(getattr(g, "alpha", 1.0)),
                                    float(getattr(g, "alpha_highlight", 1.0))) for g in gizmos]
    for gizmo, (alpha, highlight) in zip(gizmos, base):
        gizmo.alpha = alpha * DIM_ALPHA if held else alpha
        # No highlight while held: hovering must not suggest the handle responds.
        gizmo.alpha_highlight = gizmo.alpha if held else highlight
    return held


class _DeckGizmoGroup:
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT"}


class FLUIDSIM_GGT_deck_emitter(_DeckGizmoGroup, bpy.types.GizmoGroup):
    bl_idname = "FLUIDSIM_GGT_deck_emitter"
    bl_label = "Liquid Deck Emitter"

    @classmethod
    def poll(cls, context):
        return _visible(context) and _active_role(context) == "SOURCE"

    def setup(self, context):
        self.velocity = self.gizmos.new(FLUIDSIM_GT_deck_velocity.bl_idname)
        self.velocity.use_draw_scale = False
        self.dial = self.gizmos.new(FLUIDSIM_GT_deck_dial.bl_idname)
        self.dial.binding = "rate"

    def draw_prepare(self, context):
        from mathutils import Matrix, Vector

        obj = context.active_object
        settings = obj.fluidsim
        origin = Vector(_object_origin(obj))
        velocity = Vector(tuple(settings.initial_velocity))
        length = max(velocity.length * ARROW_SCALE, 0.2)
        direction = velocity.normalized() if velocity.length > 1e-6 else Vector((0.0, 0.0, 1.0))
        rotation = direction.to_track_quat("Z", "Y").to_matrix().to_4x4()
        self.velocity.matrix_basis = (Matrix.Translation(origin) @ rotation
                                      @ Matrix.Diagonal((length, length, length, 1.0)))
        self.dial.hide = rate_property(settings) is None
        _place(self.dial, origin + Vector((max(obj.dimensions) * 0.5 + 0.3, 0.0, 0.0)))
        for gizmo in (self.velocity, self.dial):
            gizmo.color = tokens.current_tokens(context).accent
            gizmo.color_highlight = tokens.current_tokens(context).text
        show_held(self, context, (self.velocity, self.dial))


class FLUIDSIM_GGT_deck_collider(_DeckGizmoGroup, bpy.types.GizmoGroup):
    bl_idname = "FLUIDSIM_GGT_deck_collider"
    bl_label = "Liquid Deck Collider"

    @classmethod
    def poll(cls, context):
        return _visible(context) and _active_role(context) == "COLLIDER"

    def setup(self, context):
        self.dial = self.gizmos.new(FLUIDSIM_GT_deck_dial.bl_idname)
        self.dial.binding = "friction"
        # Bound only when the rigid-coupling lane's property exists.
        self.toggle = None
        if dynamic_property(context.active_object.fluidsim) is not None:
            self.toggle = self.gizmos.new(FLUIDSIM_GT_deck_toggle.bl_idname)

    def draw_prepare(self, context):
        from mathutils import Vector

        obj = context.active_object
        origin = Vector(_object_origin(obj))
        offset = max(obj.dimensions) * 0.5 + 0.3
        _place(self.dial, origin + Vector((offset, 0.0, 0.0)))
        if self.toggle is not None:
            _place(self.toggle, origin + Vector((offset, 0.0, 0.4)), 0.1)
        show_held(self, context, [g for g in (self.dial, self.toggle) if g is not None])


class FLUIDSIM_GGT_deck_domain(_DeckGizmoGroup, bpy.types.GizmoGroup):
    bl_idname = "FLUIDSIM_GGT_deck_domain"
    bl_label = "Liquid Deck Domain"

    @classmethod
    def poll(cls, context):
        if not _visible(context):
            return False
        from ..scene_access import active_domain_object

        obj = getattr(context, "active_object", None)
        domain = active_domain_object(context.scene)
        return obj is not None and domain is not None and obj == domain and _axis_aligned(obj)

    def setup(self, context):
        self.faces = []
        for axis in range(3):
            for side in (-1, 1):
                gizmo = self.gizmos.new(FLUIDSIM_GT_deck_face.bl_idname)
                gizmo.axis, gizmo.side = axis, side
                self.faces.append(gizmo)

    def draw_prepare(self, context):
        obj = context.active_object
        bounds = world_bounds(*world_axes(obj), *local_bounds(obj))
        for gizmo in self.faces:
            _place(gizmo, gizmo._face_centre(bounds), 0.12)
        show_held(self, context, self.faces)


GIZMO_CLASSES = (FLUIDSIM_GT_deck_velocity, FLUIDSIM_GT_deck_dial, FLUIDSIM_GT_deck_toggle,
                 FLUIDSIM_GT_deck_face)
GROUP_CLASSES = (FLUIDSIM_GGT_deck_emitter, FLUIDSIM_GGT_deck_collider, FLUIDSIM_GGT_deck_domain)
CLASSES = GIZMO_CLASSES + GROUP_CLASSES


# Value labels beside the handles (drawn by the Deck's POST_PIXEL handler) -------------

def _project(context, point):
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    return location_3d_to_region_2d(context.region, context.region_data, point)


def _box_edges(bounds):
    lo, hi = bounds
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return [(corners[a], corners[b]) for a, b in (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6),
        (5, 7), (6, 7))]


def overlay_labels(context):
    """``(world point, text)`` pairs for the active object's handles."""
    obj = getattr(context, "active_object", None)
    settings = getattr(obj, "fluidsim", None)
    if settings is None:
        return []
    origin = _object_origin(obj)
    role = str(getattr(settings, "role", "NONE"))
    labels = []
    if role == "SOURCE":
        velocity = tuple(float(c) for c in settings.initial_velocity)
        labels.append((origin, "v " + ", ".join(f"{c:.2f}" for c in velocity) + " m/s"))
        prop = rate_property(settings)
        if prop is not None:
            labels.append(((origin[0], origin[1], origin[2] - 0.3),
                           f"Fill {float(getattr(settings, prop)):.2f}"))
    elif role == "COLLIDER":
        labels.append((origin, f"Friction {float(settings.collider_friction):.2f}"))
        prop = dynamic_property(settings)
        if prop is not None:
            labels.append(((origin[0], origin[1], origin[2] + 0.4),
                           "Dynamic " + ("on" if getattr(settings, prop) else "off")))
    return labels


def draw_overlay(context, palette):
    scale = bar._ui_scale(context)
    size = 11.0 * scale
    for point, label in overlay_labels(context):
        screen = _project(context, point)
        if screen is not None:
            _draw.text(screen[0] + 14 * scale, screen[1], label, size, tokens.rgba(palette.text))
    preview = _ACTIVE_PREVIEW
    if preview is None:
        return
    lo, hi = preview.bounds
    for a, b in _box_edges(preview.bounds):
        pa, pb = _project(context, a), _project(context, b)
        if pa is not None and pb is not None:
            _draw.polyline([tuple(pa), tuple(pb)], tokens.rgba(palette.accent), 1.5 * scale)
    top = _project(context, ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, hi[2]))
    if top is not None:
        _draw.text(top[0], top[1] + 10 * scale, preview.label, size, tokens.rgba(palette.text))


# The controls these gizmos write, for the parity test.
CONTROLS = controls.GIZMO_CONTROLS
