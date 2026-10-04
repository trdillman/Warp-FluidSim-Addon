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
    old_transform = None if domain is None else (tuple(domain.location), tuple(domain.dimensions))
    changed = []
    try:
        for name, value in changes.items():
            _write_and_verify(settings, name, value)
            changed.append(name)
        if domain_min is not None or domain_max is not None:
            authored = build_settings()
            _resize_domain(
                domain,
                domain_min if domain_min is not None else authored.domain_min,
                domain_max if domain_max is not None else authored.domain_max,
                on_transform_changed,
            )
            changed.append("domain bounds")
    except Exception as exc:  # noqa: BLE001
        for name, value in old_values.items():
            setattr(settings, name, value)
        if old_transform is not None:
            domain.location, domain.dimensions = old_transform
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
    domain.location = tuple((minimum[i] + maximum[i]) * 0.5 for i in range(3))
    domain.dimensions = dimensions
    if on_transform_changed is not None:
        on_transform_changed()
