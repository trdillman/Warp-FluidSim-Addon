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
  chip shows REINIT. The handles are offered only for an unparented, unrotated
  domain, where location and scale map one-to-one onto the world box.

The gizmos keep only drag bookkeeping (start value, start pointer) between
``invoke`` and ``exit``; a cancelled drag restores the start value. Gizmos write
only to the object the drag started on, and do nothing once that object has left
the scene (the active object may have changed or gone in the meantime).
"""

from __future__ import annotations

from dataclasses import dataclass

import bpy

from . import _draw, bar, controls, tokens

ARROW_SCALE = 0.25  # metres of arrow per m/s
DIAL_PIXELS = 300.0  # horizontal drag that sweeps a dial's whole range
PRECISE_FACTOR = 0.1
MIN_DOMAIN_SIZE = 0.05  # metres; a face cannot be dragged past this
DIAL_RANGES = {
    "source_sampling_multiplier": (0.0, 8.0),
    "emit_rate": (0.0, 32.0),  # the property's soft maximum
    "collider_friction": (0.0, 1.0),
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
    __slots__ = ("shape", "start_world", "start_value", "target")

    def setup(self):
        self.shape = self.new_custom_shape("LINES", _ARROW_LINES)

    def draw(self, context):
        self.draw_custom_shape(self.shape)

    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.shape, select_id=select_id)

    def invoke(self, context, event):
        obj = self.target = context.active_object
        self.start_value = tuple(float(c) for c in obj.fluidsim.initial_velocity)
        self.start_world = mouse_to_world(context, _mouse(event), _object_origin(obj))
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        obj = context.active_object
        now = mouse_to_world(context, _mouse(event), _object_origin(obj))
        factor = PRECISE_FACTOR if "PRECISE" in tweak else 1.0
        value = tuple(start + (n - w) * factor / ARROW_SCALE
                      for start, n, w in zip(self.start_value, now, self.start_world))
        _write(context, obj.fluidsim, "initial_velocity", value)
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        obj = _started_object(context, getattr(self, "target", None)) if cancel else None
        if obj is not None:
            _write(context, obj.fluidsim, "initial_velocity", self.start_value)


class FLUIDSIM_GT_deck_dial(bpy.types.Gizmo):
    """Drag sideways to turn a dial bound to an object setting (``binding``)."""

    bl_idname = "FLUIDSIM_GT_deck_dial"
    __slots__ = ("shape", "binding", "start_x", "start_value", "prop", "target")

    def setup(self):
        self.shape = self.new_custom_shape("LINES", _circle_lines())

    def draw(self, context):
        self.draw_custom_shape(self.shape)

    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.shape, select_id=select_id)

    def invoke(self, context, event):
        self.target = context.active_object
        settings = self.target.fluidsim
        self.prop = dial_property(settings, getattr(self, "binding", ""))
        if self.prop is None:
            return {"CANCELLED"}
        self.start_value = float(getattr(settings, self.prop))
        self.start_x = float(event.mouse_region_x)
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        low, high = DIAL_RANGES[self.prop]
        factor = PRECISE_FACTOR if "PRECISE" in tweak else 1.0
        delta = (float(event.mouse_region_x) - self.start_x) * factor
        value = min(high, max(low, self.start_value + delta * (high - low) / DIAL_PIXELS))
        _write(context, context.active_object.fluidsim, self.prop, value)
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        if cancel and getattr(self, "prop", None) is not None:
            obj = _started_object(context, getattr(self, "target", None))
            if obj is not None:
                _write(context, obj.fluidsim, self.prop, self.start_value)


class FLUIDSIM_GT_deck_toggle(bpy.types.Gizmo):
    """Click to flip the collider's Dynamic (rigid coupling) setting."""

    bl_idname = "FLUIDSIM_GT_deck_toggle"
    __slots__ = ("shape", "target")

    def setup(self):
        self.shape = self.new_custom_shape("LINES", _SQUARE_LINES)

    def draw(self, context):
        self.draw_custom_shape(self.shape)

    def draw_select(self, context, select_id):
        self.draw_custom_shape(self.shape, select_id=select_id)

    def invoke(self, context, event):
        self.target = context.active_object
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        obj = None if cancel else _started_object(context, getattr(self, "target", None))
        if obj is None:
            return
        settings = obj.fluidsim
        prop = dynamic_property(settings)
        if prop is not None:
            _write(context, settings, prop, not bool(getattr(settings, prop)))


# Domain resize ------------------------------------------------------------------

def face_resize(low, high, scale, side, delta):
    """``(new_scale, location_shift)`` for growing one face outward by ``delta`` metres.

    ``low``/``high`` are the object's local bounds on the axis, ``side`` is +1 for
    the positive face and -1 for the negative one. The opposite face stays put,
    and the size never drops below :data:`MIN_DOMAIN_SIZE`.
    """
    extent = float(high) - float(low)
    scale = float(scale)
    size = abs(extent * scale)
    if extent <= 0.0 or size <= 0.0:
        return scale, 0.0
    new_size = max(MIN_DOMAIN_SIZE, size + float(delta))
    new_scale = scale * new_size / size
    anchor = low if side > 0 else high
    return new_scale, anchor * (scale - new_scale)


def local_bounds(obj):
    corners = [tuple(float(c) for c in corner) for corner in obj.bound_box]
    return (tuple(min(c[i] for c in corners) for i in range(3)),
            tuple(max(c[i] for c in corners) for i in range(3)))


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
    __slots__ = ("shape", "axis", "side", "start_world", "start_location", "start_scale",
                 "local", "result", "target")

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
        obj = self.target = context.active_object
        self.local = local_bounds(obj)
        self.start_location = tuple(float(c) for c in obj.location)
        self.start_scale = tuple(float(c) for c in obj.scale)
        bounds = world_bounds(self.start_location, self.start_scale, *self.local)
        self.start_world = mouse_to_world(context, _mouse(event), self._face_centre(bounds))
        self.result = None
        _ACTIVE_PREVIEW = domain_preview(context.scene, bounds)
        return {"RUNNING_MODAL"}

    def _apply(self, context, scale, location):
        """Write the face's axis of the domain it started on (gone: no-op)."""
        obj = _started_object(context, getattr(self, "target", None))
        if obj is None:
            return  # the domain was deleted mid-drag
        obj.scale[self.axis] = scale
        obj.location[self.axis] = location

    def modal(self, context, event, tweak):
        global _ACTIVE_PREVIEW
        axis = self.axis
        low, high = self.local
        bounds = world_bounds(self.start_location, self.start_scale, low, high)
        now = mouse_to_world(context, _mouse(event), self._face_centre(bounds))
        factor = PRECISE_FACTOR if "PRECISE" in tweak else 1.0
        growth = self.side * (now[axis] - self.start_world[axis]) * factor
        new_scale, shift = face_resize(low[axis], high[axis], self.start_scale[axis],
                                       self.side, growth)
        location = list(self.start_location)
        scale = list(self.start_scale)
        location[axis] += shift
        scale[axis] = new_scale
        self.result = (new_scale, shift)
        _ACTIVE_PREVIEW = domain_preview(context.scene, world_bounds(location, scale, low, high))
        # Resize the domain while dragging, so the viewport follows the handle.
        self._apply(context, new_scale, location[axis])
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        global _ACTIVE_PREVIEW
        _ACTIVE_PREVIEW = None
        if cancel and getattr(self, "result", None) is not None:
            self._apply(context, self.start_scale[self.axis], self.start_location[self.axis])


# Gizmo groups ---------------------------------------------------------------------

def _visible(context) -> bool:
    from . import deck_visible

    return deck_visible(context)


def _active_role(context) -> str:
    obj = getattr(context, "active_object", None)
    return str(getattr(getattr(obj, "fluidsim", None), "role", "NONE"))


def _axis_aligned(obj) -> bool:
    if getattr(obj, "parent", None) is not None:
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
        bounds = world_bounds(tuple(obj.location), tuple(obj.scale), *local_bounds(obj))
        for gizmo in self.faces:
            _place(gizmo, gizmo._face_centre(bounds), 0.12)


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
