"""Capacity Approval: apply one planner-verified capacity option, or change nothing.

The capacity planner proposes options as ``{setting: value}`` maps, optionally
with ``domain_min`` / ``domain_max`` that resize the domain object. Approving
one writes those settings and the domain transform. The dialog costed exact
numbers against the safe live budget, so a write that does not store exactly
those numbers must not be reported as approved. ``apply_suggestion`` is the whole
transaction: write, read back, resize, and on any failure put every value back
and raise ``CapacityApplyError``.

This is a leaf module: it imports neither ``bpy`` nor any other FluidSim
module, so the transaction runs against plain stand-in objects. The operator
stays responsible for the stale-dialog and live-budget checks that decide
whether to attempt it, and for the solver retry that follows.
"""

from __future__ import annotations

import math


class CapacityApplyError(RuntimeError):
    """The option was not applied; every write had been rolled back first."""


def find_suggestion(plan, key):
    """Return the plan's suggestion named ``key``, or ``None``."""
    return next(
        (item for item in getattr(plan, "suggestions", ()) if getattr(item, "key", "") == key),
        None,
    )


def format_value(value):
    """Render a setting value for the approval dialog."""
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(f"{float(item):.4g}" for item in value) + ")"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def apply_suggestion(settings, domain, changes, *, build_settings, on_transform_changed=None):
    """Apply ``changes`` to the domain ``settings`` and ``domain`` object.

    ``changes`` is a suggestion's change map. Its ``domain_min`` and
    ``domain_max`` entries resize ``domain`` (which must be axis-aligned and
    unparented); every other entry is a setting on ``settings``. A resize
    consults ``build_settings()``, the authored solver settings, for whichever
    bound the option omits. ``on_transform_changed`` runs after the domain
    transform is written, so the caller can refresh Blender's evaluated state.

    Returns the names that changed, with ``"domain bounds"`` last when the domain
    was resized. On any failure -- an unknown setting, a value Blender stored
    differently (RNA silently clamps to the declared range), an unusable domain
    -- the settings and transform are restored and ``CapacityApplyError`` is
    raised with the cause's message.
    """
    changes = dict(changes)
    domain_min = changes.pop("domain_min", None)
    domain_max = changes.pop("domain_max", None)
    old_values = {name: getattr(settings, name) for name in changes if hasattr(settings, name)}
    # ``voxel_size`` is derived from whole ``resolution_divisions`` over the
    # domain's longest axis, so it must be written against the NEW domain and the
    # divisions are snapshotted to restore exactly (restoring the derived value
    # would be computed against whichever domain size is current).
    old_divisions = getattr(settings, "resolution_divisions", None)
    old_transform = None if domain is None else (tuple(domain.location), tuple(domain.dimensions))
    changed = []
    try:
        resized = domain_min is not None or domain_max is not None
        if resized:
            authored = build_settings()
            _resize_domain(
                domain,
                domain_min if domain_min is not None else authored.domain_min,
                domain_max if domain_max is not None else authored.domain_max,
                on_transform_changed,
            )
        for name, value in changes.items():
            _write_and_verify(settings, name, value)
            changed.append(name)
        if resized:
            changed.append("domain bounds")
    except Exception as exc:  # noqa: BLE001
        if old_transform is not None:
            domain.location, domain.dimensions = old_transform
            if on_transform_changed is not None:
                try:
                    on_transform_changed()
                except Exception:  # noqa: BLE001
                    pass
        for name, value in old_values.items():
            setattr(settings, name, value)
        if old_divisions is not None:
            settings.resolution_divisions = old_divisions
        raise CapacityApplyError(str(exc)) from exc
    return changed


def _write_and_verify(settings, name, value):
    if not hasattr(settings, name):
        raise RuntimeError(f"Capacity option cannot apply setting {name!r}")
    rna = getattr(settings, "bl_rna", None)
    rna_property = getattr(rna, "properties", {}).get(name)
    if rna_property is not None and rna_property.type == "INT":
        value = int(value)
    setattr(settings, name, value)
    # RNA clamps out-of-range assignments to the declared [min, max] silently,
    # so a write can succeed while storing a different number than the one the
    # dialog costed. Reporting that as approved would break the budget promise
    # this whole flow exists to keep -- read back and fail into the rollback.
    stored = getattr(settings, name)
    if isinstance(value, float):
        matches = math.isclose(stored, value, rel_tol=1e-6)
    else:
        matches = stored == value
    if not matches:
        raise RuntimeError(
            f"setting {name!r} was stored as {stored!r} instead of "
            f"{value!r}; it is outside the range this build allows"
        )


def _resize_domain(domain, minimum, maximum, on_transform_changed):
    if domain is None:
        raise RuntimeError("The scene has no domain object to resize")
    if getattr(domain, "parent", None) is not None:
        raise RuntimeError("Automatic domain-bound adjustment requires an unparented domain")
    rotations = tuple(float(value) for value in domain.rotation_euler)
    delta_rotations = tuple(
        float(value) for value in getattr(domain, "delta_rotation_euler", (0.0, 0.0, 0.0))
    )

    def angle_distance(value):
        return abs((value + math.pi) % (2.0 * math.pi) - math.pi)

    if any(angle_distance(value) > 1.0e-6 for value in rotations + delta_rotations):
        raise RuntimeError("Automatic domain-bound adjustment requires an axis-aligned domain")
    minimum = tuple(float(value) for value in minimum)
    maximum = tuple(float(value) for value in maximum)
    dimensions = tuple(maximum[i] - minimum[i] for i in range(3))
    if any(value <= 0.0 for value in dimensions):
        raise RuntimeError("Suggested domain bounds are not valid")
    local_min, local_max = _local_bounds(domain)
    if any(local_max[i] - local_min[i] <= 0.0 for i in range(3)):
        raise RuntimeError("The domain mesh has no volume to resize")
    domain.dimensions = dimensions
    # The object origin need not sit at the mesh centre (a corner origin, or a
    # mesh with transforms applied), so place the origin from the local bound
    # box: on each axis the world minimum is
    # location + delta_location + min(scale * local_min, scale * local_max).
    scale = _effective_scale(domain)
    delta_location = tuple(
        float(value) for value in getattr(domain, "delta_location", (0.0, 0.0, 0.0)))
    location = []
    for i in range(3):
        factor = math.copysign(dimensions[i] / (local_max[i] - local_min[i]), scale[i])
        lowest = min(factor * local_min[i], factor * local_max[i])
        location.append(minimum[i] - delta_location[i] - lowest)
    domain.location = tuple(location)
    if on_transform_changed is not None:
        on_transform_changed()
        _verify_world_bounds(domain, minimum, maximum)


def _local_bounds(domain):
    """The mesh's local bound box; a centred unit box when it has none."""
    corners = getattr(domain, "bound_box", None)
    if corners is None:
        return (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
    points = [tuple(float(value) for value in corner) for corner in corners]
    return (tuple(min(point[i] for point in points) for i in range(3)),
            tuple(max(point[i] for point in points) for i in range(3)))


def _effective_scale(domain):
    scale = tuple(float(value) for value in getattr(domain, "scale", (1.0, 1.0, 1.0)))
    delta = tuple(float(value) for value in getattr(domain, "delta_scale", (1.0, 1.0, 1.0)))
    return tuple(scale[i] * delta[i] for i in range(3))


def _verify_world_bounds(domain, minimum, maximum):
    """Refuse a resize whose evaluated world bounds are not the approved ones.

    Runs after ``on_transform_changed`` refreshed Blender's evaluated
    transform; an object without ``matrix_world`` and ``bound_box`` (a plain
    stand-in) has nothing further to check.
    """
    matrix = getattr(domain, "matrix_world", None)
    corners = getattr(domain, "bound_box", None)
    if matrix is None or corners is None:
        return
    points = [
        tuple(
            sum(float(matrix[row][column]) * float(corner[column]) for column in range(3))
            + float(matrix[row][3])
            for row in range(3)
        )
        for corner in corners
    ]
    lowest = tuple(min(point[i] for point in points) for i in range(3))
    highest = tuple(max(point[i] for point in points) for i in range(3))
    for i in range(3):
        tolerance = 1.0e-5 * max(1.0, maximum[i] - minimum[i])
        if (abs(lowest[i] - minimum[i]) > tolerance
                or abs(highest[i] - maximum[i]) > tolerance):
            raise RuntimeError(
                "The resized domain landed at "
                f"{format_value(lowest)}..{format_value(highest)} instead of "
                f"{format_value(minimum)}..{format_value(maximum)}")
