"""Narrow-band FLIP: the pure, bpy-free admission rules.

Narrow-band mode keeps particles only near the liquid surface, so any feature
that stores per-particle state in the interior cannot coexist with it. The
rules live here, with no ``bpy`` or Warp import, so the Setup stage, the
Initialize/Reset preflight and unit tests all read the same single answer.
"""

from __future__ import annotations

from typing import Iterable

#: Reasons are fixed strings so the UI, the preflight error and tests agree.
REASON_COLOR_MIXING = "Color mixing stores per-particle color"
REASON_VISCOSITY_FROM_PARTICLES = (
    "Viscosity from particles reads per-particle viscosity")
REASON_SCALAR_FIELD_SOURCE = (
    "A Scalar / Temperature field source stores per-particle values")
REASON_EXPORT_ATTRIBUTES = "Attribute export reads per-particle attributes"


def has_scalar_field_source(objects: Iterable[object]) -> bool:
    """True when any object is a FIELD source targeting the SCALAR field.

    ``objects`` are scene objects; each one's ``fluidsim`` group is read by
    duck typing so this stays free of ``bpy``.
    """
    for obj in objects:
        group = getattr(obj, "fluidsim", None)
        if group is None:
            continue
        if (getattr(group, "role", None) == "FIELD"
                and getattr(group, "field_operation", None) == "SCALAR"):
            return True
    return False


def blocking_reasons(
    settings: object, *, scalar_field_source: bool = False
) -> tuple[str, ...]:
    """Reasons narrow-band particles cannot run with ``settings``.

    ``settings`` is a domain group or ``SolverSettings``. The result is empty
    when the mode is allowed; it does not depend on ``narrow_band_enabled``
    itself, so the UI can grey the toggle before it is switched on.
    """
    reasons: list[str] = []
    if bool(getattr(settings, "species_enabled", False)):
        reasons.append(REASON_COLOR_MIXING)
    if (bool(getattr(settings, "viscosity_enabled", False))
            and bool(getattr(settings, "viscosity_from_particles", False))):
        reasons.append(REASON_VISCOSITY_FROM_PARTICLES)
    if scalar_field_source:
        reasons.append(REASON_SCALAR_FIELD_SOURCE)
    if bool(getattr(settings, "export_attributes", False)):
        reasons.append(REASON_EXPORT_ATTRIBUTES)
    return tuple(reasons)


def scene_blocking_reasons(
    settings: object, scene_objects: Iterable[object]
) -> tuple[str, ...]:
    """``blocking_reasons`` with the field-source check read from the scene."""
    return blocking_reasons(
        settings, scalar_field_source=has_scalar_field_source(scene_objects))


def refusal_message(reasons: Iterable[str]) -> str:
    """The Initialize refusal text for a non-empty ``reasons``."""
    return (
        "Narrow-Band Particles cannot run with this setup: "
        + "; ".join(reasons)
        + ". Turn Narrow-Band Particles off or remove the conflicting feature.")
