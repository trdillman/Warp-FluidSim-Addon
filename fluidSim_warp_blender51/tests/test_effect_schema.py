from __future__ import annotations

import pytest

from fluidsim_testpkg.sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    FieldAccess,
    ParameterKind,
)
from fluidsim_testpkg.sim.effects.schema import (
    EffectDefinitionV2,
    EffectInstanceV2,
    FieldBindingV2,
    KernelDefinitionV2,
    ParameterDefinitionV2,
)


def _definition() -> EffectDefinitionV2:
    return EffectDefinitionV2(
        effect_id="test.vortex",
        version="1.2.0",
        name="Test Vortex",
        summary="Turns particles around a controller.",
        description="A deterministic particle vortex.",
        category="Forces",
        author="Tests",
        license="GPL-3.0-or-later",
        parameters=(
            ParameterDefinitionV2(
                key="strength",
                label="Strength",
                kind=ParameterKind.FLOAT,
                default=2.0,
                minimum=0.0,
                maximum=10.0,
                unit="VELOCITY",
            ),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="vortex_particles",
                label="Particle Vortex",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="particle_velocity[particle_index] = velocity * strength",
                fields=(
                    FieldBindingV2(
                        field="particle_velocity",
                        access=FieldAccess.READ_WRITE_CURRENT,
                    ),
                ),
                parameter_keys=("strength",),
            ),
        ),
    )


def test_effect_schema_round_trip_is_strict_and_stable():
    definition = _definition()
    assert EffectDefinitionV2.from_dict(definition.to_dict()) == definition

    payload = definition.to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="unknown EffectDefinitionV2 field"):
        EffectDefinitionV2.from_dict(payload)


def test_instance_rejects_unknown_parameter_and_out_of_range_value():
    definition = _definition()
    with pytest.raises(ValueError, match="unknown parameter"):
        EffectInstanceV2(
            instance_id="instance-1",
            definition=definition,
            parameter_values={"not_declared": 1.0},
        )

    with pytest.raises(ValueError, match="above maximum"):
        EffectInstanceV2(
            instance_id="instance-1",
            definition=definition,
            parameter_values={"strength": 12.0},
        )


def test_duplicate_parameter_and_kernel_ids_are_rejected():
    base = _definition()
    with pytest.raises(ValueError, match="duplicate parameter"):
        EffectDefinitionV2(
            effect_id=base.effect_id,
            version=base.version,
            name=base.name,
            summary=base.summary,
            description=base.description,
            category=base.category,
            author=base.author,
            license=base.license,
            parameters=base.parameters + base.parameters,
            kernels=base.kernels,
        )
    with pytest.raises(ValueError, match="duplicate kernel"):
        EffectDefinitionV2(
            effect_id=base.effect_id,
            version=base.version,
            name=base.name,
            summary=base.summary,
            description=base.description,
            category=base.category,
            author=base.author,
            license=base.license,
            parameters=base.parameters,
            kernels=base.kernels + base.kernels,
        )


def test_vector_color_enum_bool_and_object_parameter_values_validate():
    definition = EffectDefinitionV2(
        effect_id="test.parameter-kinds",
        version="1.0.0",
        name="Kinds",
        summary="Typed values",
        description="Typed values",
        category="Tests",
        author="Tests",
        license="GPL-3.0-or-later",
        parameters=(
            ParameterDefinitionV2("axis", "Axis", ParameterKind.VECTOR, (0.0, 0.0, 1.0)),
            ParameterDefinitionV2("tint", "Tint", ParameterKind.COLOR, (1.0, 0.5, 0.25, 1.0)),
            ParameterDefinitionV2(
                "falloff", "Falloff", ParameterKind.ENUM, "SMOOTH",
                enum_items=(("LINEAR", "Linear"), ("SMOOTH", "Smooth")),
            ),
            ParameterDefinitionV2("enabled", "Enabled", ParameterKind.BOOL, True),
            ParameterDefinitionV2("controller", "Controller", ParameterKind.OBJECT, ""),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="k",
                label="K",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                cadence=EffectCadence.ONCE_PER_FRAME,
                source="pass",
            ),
        ),
    )
    instance = EffectInstanceV2(
        instance_id="kinds-1",
        definition=definition,
        parameter_values={
            "axis": (1.0, 0.0, 0.0),
            "tint": (0.2, 0.3, 0.4, 1.0),
            "falloff": "LINEAR",
            "enabled": False,
            "controller": "ControllerObject",
        },
    )
    assert instance.effective_parameter_values()["axis"] == (1.0, 0.0, 0.0)


def test_effect_status_has_explicit_disabled_state():
    from fluidsim_testpkg.sim.effects.enums import EffectStatus

    assert EffectStatus.DISABLED.value == "DISABLED"
