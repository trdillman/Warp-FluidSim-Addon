"""Legacy custom-snippet records, kept only to migrate saved scenes.

1.x scenes stored custom Warp snippets on ``scene.fluidsim.snippets``. The
migration reader (``blender/snippets.py``) turns each one into these plain
records, and ``legacy_snippet_to_effect`` rewraps it as a Custom Effects v2
instance. Nothing here validates, compiles or launches code.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field as dc_field

from .enums import (
    CustomKernelFieldPermission,
    CustomKernelOrder,
    CustomKernelRunOver,
    CustomKernelArgumentMode,
    CustomKernelConstantType,
    CustomKernelType,
)


FUNCTION_UNSUPPORTED_MESSAGE = (
    "FUNCTION snippets are unsupported; helper linkage is undefined"
)


@dataclass
class CustomArgument:
    mode: CustomKernelArgumentMode = CustomKernelArgumentMode.FIELD
    field: str = "temperature"
    field_permission: CustomKernelFieldPermission = CustomKernelFieldPermission.READ_WRITE
    constant_type: CustomKernelConstantType = CustomKernelConstantType.FLOAT
    constant_name: str = "value"
    constant_value: float = 0.0


@dataclass
class CustomSnippet:
    enabled: bool = True
    kernel_name: str = "user_kernel"
    snippet_type: CustomKernelType = CustomKernelType.KERNEL
    order: CustomKernelOrder = CustomKernelOrder.FINAL_STEP
    run_over: CustomKernelRunOver = CustomKernelRunOver.VOXEL_LINEAR
    code: str = ""
    arguments: list = dc_field(default_factory=list)  # list[CustomArgument]
    # Layering window: restrict when this snippet fires so multiple snippets
    # can be combined at different times (0 = open ended, step = every Nth).
    frame_start: int = 0
    frame_end: int = 0
    frame_step: int = 1


# Custom Effects v2 compatibility conversion


def legacy_snippet_to_effect(snippet: CustomSnippet, *, instance_id: str | None = None):
    """Wrap a legacy one-kernel snippet as a strict Custom Effects v2 instance.

    Source, constants, field bindings, frame window, enable state, and legacy
    execution semantics are preserved.  ``VOXEL_TILE`` never had distinct tile
    scheduling in the Warp port, so it migrates to ``ACTIVE_VOXELS`` with an
    explicit warning rather than retaining a misleading public mode.
    """
    if snippet.snippet_type != CustomKernelType.KERNEL:
        raise ValueError(FUNCTION_UNSUPPORTED_MESSAGE)

    from .effects.enums import (
        EffectCadence,
        EffectDomain,
        EffectStage,
        FieldAccess,
        ParameterKind,
    )
    from .effects.schema import (
        EffectDefinitionV2,
        EffectInstanceV2,
        FieldBindingV2,
        KernelDefinitionV2,
        ParameterDefinitionV2,
    )

    warnings = []
    domain = (
        EffectDomain.PARTICLES
        if snippet.run_over == CustomKernelRunOver.PARTICLES
        else EffectDomain.ACTIVE_VOXELS
    )
    if snippet.run_over == CustomKernelRunOver.VOXEL_TILE:
        warnings.append(
            "Legacy VOXEL_TILE used the same flat active-voxel launch as "
            "VOXEL_LINEAR and was migrated to ACTIVE_VOXELS."
        )
    stage_map = {
        CustomKernelOrder.FIRST_STEP: EffectStage.BEFORE_EMISSION,
        CustomKernelOrder.BEFORE_SOURCING: EffectStage.BEFORE_EMISSION,
        CustomKernelOrder.BEFORE_FORCES: EffectStage.AFTER_PARTICLE_TRANSFER,
        CustomKernelOrder.BEFORE_PROJECT_NON_DIVERGENT_AND_ADVECTION: EffectStage.BEFORE_PRESSURE,
        CustomKernelOrder.FINAL_STEP: EffectStage.AFTER_PARTICLE_UPDATE,
    }
    fields = []
    parameters = []
    values = {}
    for argument in snippet.arguments:
        if argument.mode == CustomKernelArgumentMode.FIELD:
            access = {
                CustomKernelFieldPermission.READ: FieldAccess.READ,
                CustomKernelFieldPermission.WRITE: FieldAccess.WRITE_CURRENT,
                CustomKernelFieldPermission.READ_WRITE: FieldAccess.READ_WRITE_CURRENT,
            }[argument.field_permission]
            fields.append(FieldBindingV2(
                argument.field,
                access,
            ))
        else:
            kind = (
                ParameterKind.FLOAT
                if argument.constant_type == CustomKernelConstantType.FLOAT
                else ParameterKind.INT
            )
            value = (
                float(argument.constant_value)
                if kind is ParameterKind.FLOAT
                else int(round(argument.constant_value))
            )
            parameters.append(ParameterDefinitionV2(
                argument.constant_name,
                argument.constant_name.replace("_", " ").title(),
                kind,
                value,
                description="Migrated legacy constant.",
            ))
            values[argument.constant_name] = value

    slug = re.sub(r"[^a-z0-9]+", "-", snippet.kernel_name.lower()).strip("-") or "effect"
    digest = hashlib.sha256((snippet.code + repr(snippet.arguments)).encode("utf-8")).hexdigest()[:10]
    effect_id = f"legacy.{slug}-{digest}"
    kernel_id = re.sub(r"[^A-Za-z0-9_]+", "_", snippet.kernel_name).strip("_") or "legacy_kernel"
    if kernel_id[0].isdigit():
        kernel_id = "legacy_" + kernel_id
    definition = EffectDefinitionV2(
        effect_id=effect_id,
        version="1.0.0",
        name=snippet.kernel_name or "Legacy Custom Effect",
        summary="Migrated legacy FluidSim Warp snippet.",
        description=(
            "This effect was migrated without rewriting its source. Review and "
            "trust the exact content before enabling it."
        ),
        category="Legacy",
        author="Scene author",
        license="User supplied",
        parameters=tuple(parameters),
        kernels=(KernelDefinitionV2(
            kernel_id=kernel_id,
            label=snippet.kernel_name or "Legacy Kernel",
            domain=domain,
            stage=stage_map[snippet.order],
            cadence=EffectCadence.EVERY_SUBSTEP,
            source=snippet.code or "pass",
            fields=tuple(fields),
            parameter_keys=tuple(parameter.key for parameter in parameters),
            safe_mode=False,
        ),),
        built_in=False,
        package_origin="legacy-scene-snippet",
        cost_hint="UNKNOWN",
    )
    instance = EffectInstanceV2(
        instance_id=instance_id or f"legacy-{digest}",
        definition=definition,
        enabled=bool(snippet.enabled),
        parameter_values=values,
        frame_start=int(snippet.frame_start),
        frame_end=int(snippet.frame_end),
        frame_step=max(1, int(snippet.frame_step)),
        safe_mode=False,
        expert_acknowledged=True,
        display_name=snippet.kernel_name,
    )
    return instance, tuple(warnings)
