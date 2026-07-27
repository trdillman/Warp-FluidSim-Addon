"""Sole compatibility adapter from legacy source descriptors to SourcePlan.

Legacy descriptors remain accepted for saved scenes. Every plan owns primitive
values/NumPy arrays and contains no Blender, mathutils, evaluated-mesh or Warp
mesh lifetime.
"""

from __future__ import annotations

import numpy as np

from .enums import (
    MeshSourceRegion,
    SourceEmissionMode,
    SourceEvaluationCadence,
    SourceLifecycle,
    SourceOperation,
    SourcePayloadKind,
    SourceShapeSourceMode,
    SourceShapeType,
)
from .source_regions import (
    ActivationPlan,
    AnalyticRegionPlan,
    BlendPlan,
    HeightRegionPlan,
    MeshRegionPlan,
    PayloadPlan,
    SourceKey,
    SourcePlan,
)


def _source_key(desc, session_generation=0):
    explicit = getattr(desc, "runtime_source_key", "")
    if explicit:
        return SourceKey(str(explicit))
    legacy = getattr(desc, "source_id", "") or f"legacy:{id(desc)}"
    return SourceKey(f"session:{int(session_generation)}:{legacy}")


def _lifecycle(desc):
    return (SourceLifecycle.ONCE
            if int(getattr(desc, "emission_mode",
                           SourceEmissionMode.CONTINUOUS_INFLOW))
            == SourceEmissionMode.INITIAL_FILL
            else SourceLifecycle.TOP_UP)


def _operation(desc):
    stype = int(getattr(desc, "stype", SourceShapeType.SOURCE))
    if stype == SourceShapeType.SINK:
        return SourceOperation.SINK
    if stype == SourceShapeType.COLLISION:
        return SourceOperation.COLLISION
    if stype == SourceShapeType.ACTIVATION:
        return SourceOperation.ACTIVATION
    if (int(getattr(desc, "mode", SourceShapeSourceMode.PARTICLES))
            == SourceShapeSourceMode.FIELD):
        # Original FIELD semantics are pressure influence. Force-region shapes
        # remain independently ordered post-P2G by the legacy force path.
        if int(getattr(desc, "force_type", 0)):
            return SourceOperation.FORCE
        return SourceOperation.PRESSURE
    return SourceOperation.LIQUID


def _payload(desc, operation):
    if operation == SourceOperation.PRESSURE:
        return PayloadPlan(
            kind=SourcePayloadKind.PRESSURE,
            pressure=float(getattr(desc, "blend_amount", 1.0)),
        )
    kind = (SourcePayloadKind.LIQUID_ATTRIBUTES
            if operation == SourceOperation.LIQUID else SourcePayloadKind.NONE)
    return PayloadPlan(
        kind=kind,
        velocity=tuple(float(v) for v in getattr(desc, "velocity", (0, 0, 0))),
        temperature=float(getattr(desc, "temperature", 0.0)),
        viscosity=float(getattr(desc, "viscosity", 0.0)),
        density=float(getattr(desc, "density", 1.0)),
        lifespan=float(getattr(desc, "life_max", 0.0)),
        use_lifespan=bool(getattr(desc, "use_lifespan", False)),
        target_particles_per_voxel=float(getattr(desc, "emit_rate", 8.0)),
    )


def descriptor_to_plan(desc, *, session_generation=0):
    """Translate one legacy descriptor into a fully-owned immutable plan."""
    operation = _operation(desc)
    payload = _payload(desc, operation)
    if hasattr(desc, "vertices") and getattr(desc, "vertices", None) is not None:
        solid = (int(getattr(desc, "source_region",
                            MeshSourceRegion.SURFACE_SHELL))
                 == MeshSourceRegion.SOLID_VOLUME)
        region = MeshRegionPlan(
            np.asarray(desc.vertices, dtype=np.float32),
            np.asarray(desc.triangles, dtype=np.int32),
            solid=solid,
            shell_half_width=float(getattr(desc, "surface_thickness", 0.0)),
        )
        cadence = SourceEvaluationCadence(int(getattr(
            desc, "evaluation_cadence", SourceEvaluationCadence.FIRST_SIM_FRAME)))
    else:
        region = AnalyticRegionPlan(
            shape=desc.shape,
            position=tuple(float(v) for v in desc.position),
            rotation=tuple(float(v) for v in desc.rotation),
            half_extents=tuple(float(v) for v in desc.half_extents),
            radius=float(desc.radius),
            height=float(desc.height),
            feather=float(desc.feather),
        )
        cadence = SourceEvaluationCadence.EVERY_SIM_FRAME
    return SourcePlan(
        stable_id=_source_key(desc, session_generation),
        operation=operation,
        lifecycle=_lifecycle(desc),
        cadence=cadence,
        region=region,
        payload=payload,
        activation=ActivationPlan(enabled=bool(getattr(desc, "enabled", True))),
        blend=BlendPlan(
            mode=getattr(desc, "blend_mode", 1),
            amount=float(getattr(desc, "blend_amount", 1.0)),
        ),
    )


def height_descriptor_to_plan(desc, *, session_generation=0):
    return SourcePlan(
        stable_id=SourceKey(f"session:{session_generation}:builtin:height"),
        operation=SourceOperation.LIQUID,
        lifecycle=_lifecycle(desc),
        cadence=SourceEvaluationCadence.EVERY_SIM_FRAME,
        region=HeightRegionPlan(
            base_height=float(desc.base_height),
            back_mode=desc.back_mode,
            profile=getattr(desc, "profile", None),
        ),
        payload=PayloadPlan(
            kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
            velocity=tuple(float(v) for v in desc.velocity),
            target_particles_per_voxel=float(desc.emit_rate),
        ),
        activation=ActivationPlan(enabled=bool(desc.enabled)),
    )


def descriptors_to_plans(descriptors, *, session_generation=0):
    return tuple(descriptor_to_plan(
        desc, session_generation=session_generation) for desc in descriptors)
