from __future__ import annotations

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
from fluidsim_testpkg.sim.effects.validation import validate_effect


def _instance(source: str, *, domain=EffectDomain.PARTICLES, fields=(), safe_mode=True):
    definition = EffectDefinitionV2(
        effect_id="test.validation",
        version="1.0.0",
        name="Validation",
        summary="Validation",
        description="Validation",
        category="Tests",
        author="Tests",
        license="GPL-3.0-or-later",
        parameters=(
            ParameterDefinitionV2("strength", "Strength", ParameterKind.FLOAT, 1.0),
        ),
        kernels=(
            KernelDefinitionV2(
                kernel_id="kernel",
                label="Kernel",
                domain=domain,
                stage=(EffectStage.AFTER_PARTICLE_TRANSFER
                       if domain is EffectDomain.PARTICLES
                       else EffectStage.BEFORE_PRESSURE),
                cadence=EffectCadence.EVERY_SUBSTEP,
                source=source,
                fields=tuple(fields),
                parameter_keys=("strength",),
                safe_mode=safe_mode,
            ),
        ),
    )
    return EffectInstanceV2("validation-1", definition, safe_mode=safe_mode)


def _codes(instance):
    return {issue.code for issue in validate_effect(instance).issues}


def test_wrong_domain_field_fails_static_validation():
    binding = FieldBindingV2("temperature", FieldAccess.READ)
    instance = _instance(
        "value = temperature[particle_index]",
        fields=(binding,),
    )
    assert "FIELD_DOMAIN_MISMATCH" in _codes(instance)


def test_unbounded_while_loop_is_rejected():
    instance = _instance("while strength > 0.0:\n    strength = strength - 1.0")
    assert "UNBOUNDED_LOOP" in _codes(instance)


def test_safe_mode_rejects_print_and_arbitrary_scatter():
    binding = FieldBindingV2("particle_velocity", FieldAccess.UNSAFE_ARBITRARY)
    instance = _instance(
        "print(strength)\nparticle_velocity[0] = velocity",
        fields=(binding,),
    )
    codes = _codes(instance)
    assert "PRINT_FORBIDDEN" in codes
    assert "UNSAFE_ACCESS_REQUIRES_EXPERT" in codes


def test_safe_current_element_write_is_accepted():
    binding = FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT)
    instance = _instance(
        "particle_velocity[particle_index] = velocity * strength",
        fields=(binding,),
    )
    result = validate_effect(instance)
    assert result.valid is True
    assert result.issues == ()


def test_safe_mode_rejects_non_current_index_even_when_binding_is_current_write():
    binding = FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT)
    instance = _instance(
        "particle_velocity[particle_index + 1] = velocity",
        fields=(binding,),
    )
    assert "NON_CURRENT_ELEMENT_ACCESS" in _codes(instance)


def test_constant_bounded_range_loop_is_accepted():
    instance = _instance("total = 0.0\nfor i in range(4):\n    total = total + float(i)")
    assert validate_effect(instance).valid is True


def test_validation_issues_identify_the_exact_effect_instance():
    instance = _instance(source="value = unknown_effect_name")
    result = validate_effect(instance)
    assert result.valid is False
    assert result.issues
    assert all(issue.instance_id == instance.instance_id for issue in result.issues)


def test_validation_keeps_mixed_ast_issue_order_and_locations():
    instance = _instance(
        "while strength > 0.0:\n"
        "    strength = strength - 1.0\n"
        "print(strength)"
    )

    result = validate_effect(instance)

    assert [(issue.code, issue.line, issue.column) for issue in result.issues] == [
        ("UNBOUNDED_LOOP", 1, 1),
        ("PRINT_FORBIDDEN", 3, 1),
        ("UNKNOWN_NAME", 3, 1),
    ]
