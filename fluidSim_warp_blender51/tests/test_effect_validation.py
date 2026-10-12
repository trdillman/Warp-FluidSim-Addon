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


def _generated_identifiers(domain):
    """Every name the generated wrapper binds around the user body (B65)."""
    import ast

    from fluidsim_testpkg.sim.effects.source import generate_kernel_source

    instance = _instance("pass", domain=domain)
    generated = generate_kernel_source(instance, 0)
    tree = ast.parse(generated.source)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.FunctionDef):
            names.update(arg.arg for arg in node.args.args)
            prelude = [s for s in node.body if s.lineno < generated.user_line_start]
            for statement in prelude:
                for child in ast.walk(statement):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        names.add(child.id)
    names.discard("strength")  # the test instance's own parameter
    return names


def _instance_with_parameter(key, *, domain=EffectDomain.PARTICLES):
    definition = EffectDefinitionV2(
        effect_id="test.reserved",
        version="1.0.0",
        name="Reserved",
        summary="Reserved",
        description="Reserved",
        category="Tests",
        author="Tests",
        license="GPL-3.0-or-later",
        parameters=(ParameterDefinitionV2(key, key, ParameterKind.FLOAT, 1.0),),
        kernels=(
            KernelDefinitionV2(
                kernel_id="kernel",
                label="Kernel",
                domain=domain,
                stage=EffectStage.BEFORE_PRESSURE,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="pass",
                parameter_keys=(key,),
            ),
        ),
    )
    return EffectInstanceV2("reserved-1", definition)


def test_every_generated_identifier_is_a_reserved_parameter_name():
    assert {"_count", "_positions", "_tid"} <= _generated_identifiers(EffectDomain.PARTICLES)
    for domain in (EffectDomain.PARTICLES, EffectDomain.ACTIVE_VOXELS):
        generated = _generated_identifiers(domain)
        for name in sorted(generated):
            codes = _codes(_instance_with_parameter(name, domain=domain))
            assert "RESERVED_PARAMETER_NAME" in codes, (domain, name)


def test_body_using_generated_underscore_names_gets_a_reserved_name_diagnostic():
    binding = FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT)
    instance = _instance(
        "_tid = 0\nparticle_velocity[particle_index] = velocity * float(_count[0])",
        fields=(binding,),
    )
    result = validate_effect(instance)
    reserved = [issue for issue in result.issues if issue.code == "RESERVED_NAME"]
    assert result.valid is False
    assert [(issue.line, issue.column) for issue in reserved] == [(1, 1), (2, 54)]
    assert "reserved for generated kernel code" in reserved[0].message
    assert "UNKNOWN_NAME" not in {issue.code for issue in result.issues}


def test_bare_underscore_loop_variable_stays_allowed():
    instance = _instance("total = 0.0\nfor _ in range(4):\n    total = total + strength")
    assert validate_effect(instance).valid is True


def _velocity(access):
    return (FieldBindingV2("particle_velocity", access),)


def test_array_alias_cannot_bypass_field_permissions():
    # B63: subscripts on an alias were never checked, so a READ binding could
    # be written (and a safe binding indexed arbitrarily) through it.
    for access, safe_mode in ((FieldAccess.READ, True),
                              (FieldAccess.READ_WRITE_CURRENT, True),
                              (FieldAccess.READ, False)):
        instance = _instance(
            "arr = particle_velocity\narr[0] = arr[0] + wp.vec3(1.0, 0.0, 0.0)",
            fields=_velocity(access), safe_mode=safe_mode,
        )
        result = validate_effect(instance)
        assert result.valid is False, (access, safe_mode)
        assert "FIELD_ALIAS_FORBIDDEN" in {issue.code for issue in result.issues}


def test_field_passed_to_a_non_atomic_call_is_an_alias():
    instance = _instance(
        "value = wp.length(particle_velocity)",
        fields=_velocity(FieldAccess.READ),
    )
    assert "FIELD_ALIAS_FORBIDDEN" in _codes(instance)


def test_atomic_on_read_binding_is_a_write_without_permission():
    for safe_mode in (True, False):
        instance = _instance(
            "wp.atomic_add(particle_velocity, particle_index, wp.vec3(1.0, 0.0, 0.0))",
            fields=_velocity(FieldAccess.READ), safe_mode=safe_mode,
        )
        assert "WRITE_REQUIRES_PERMISSION" in _codes(instance), safe_mode


def test_atomic_on_structural_voxel_field_is_rejected():
    instance = _instance(
        "wp.atomic_add(voxel_type, flat_voxel_index, 4)",
        domain=EffectDomain.ACTIVE_VOXELS,
        fields=(FieldBindingV2("voxel_type", FieldAccess.READ),),
    )
    assert "WRITE_REQUIRES_PERMISSION" in _codes(instance)


def test_safe_atomic_must_target_the_current_element():
    current = _instance(
        "wp.atomic_add(particle_velocity, particle_index, velocity)",
        fields=_velocity(FieldAccess.READ_WRITE_CURRENT),
    )
    assert validate_effect(current).valid is True
    neighbour = _instance(
        "wp.atomic_add(particle_velocity, 0, velocity)",
        fields=_velocity(FieldAccess.READ_WRITE_CURRENT),
    )
    assert "NON_CURRENT_ELEMENT_ACCESS" in _codes(neighbour)


def test_component_write_through_nested_subscript_needs_write_permission():
    instance = _instance(
        "particle_velocity[particle_index][0] = 1.0",
        fields=_velocity(FieldAccess.READ),
    )
    assert "WRITE_REQUIRES_PERMISSION" in _codes(instance)


def test_current_index_and_field_names_cannot_be_rebound():
    for source in (
        "particle_index = 0\nparticle_velocity[particle_index] = velocity",
        "for particle_index in range(4):\n    particle_velocity[particle_index] = velocity",
        "particle_velocity = particle_velocity\nparticle_velocity[particle_index] = velocity",
        "wp = 0",
    ):
        instance = _instance(source, fields=_velocity(FieldAccess.READ_WRITE_CURRENT))
        result = validate_effect(instance)
        assert result.valid is False, source
        assert "RESERVED_NAME_ASSIGNMENT" in {issue.code for issue in result.issues}, source


def test_voxel_current_index_cannot_be_rebound():
    instance = _instance(
        "flat_voxel_index = 0\ntemperature[flat_voxel_index] = 1.0",
        domain=EffectDomain.ACTIVE_VOXELS,
        fields=(FieldBindingV2("temperature", FieldAccess.READ_WRITE_CURRENT),),
    )
    assert "RESERVED_NAME_ASSIGNMENT" in _codes(instance)


_EXPERT_SOURCE = "particle_velocity[0] = velocity"


def _expert_instance(*, instance_safe, kernel_safe, acknowledged):
    from dataclasses import replace

    instance = _instance(
        _EXPERT_SOURCE, fields=_velocity(FieldAccess.UNSAFE_ARBITRARY))
    kernel = replace(instance.definition.kernels[0], safe_mode=kernel_safe)
    definition = replace(instance.definition, kernels=(kernel,))
    return replace(instance, definition=definition, safe_mode=instance_safe,
                   expert_acknowledged=acknowledged)


def test_turning_off_safety_without_acknowledgement_keeps_safe_restrictions():
    # B64: kernel (or instance) safety off used to grant Expert access even
    # though the instance never acknowledged it.
    for instance_safe, kernel_safe in ((True, False), (False, True), (False, False)):
        result = validate_effect(_expert_instance(
            instance_safe=instance_safe, kernel_safe=kernel_safe, acknowledged=False))
        codes = {issue.code for issue in result.issues}
        assert result.valid is False, (instance_safe, kernel_safe)
        assert {"UNSAFE_ACCESS_REQUIRES_EXPERT",
                "EXPERT_ACKNOWLEDGEMENT_REQUIRED"} <= codes
        # Callers report issues[0]; it must be the blocking error.
        assert result.issues[0].severity == "ERROR"


def test_acknowledged_expert_access_lifts_safe_restrictions():
    for instance_safe, kernel_safe in ((True, False), (False, True), (False, False)):
        result = validate_effect(_expert_instance(
            instance_safe=instance_safe, kernel_safe=kernel_safe, acknowledged=True))
        assert result.valid is True, (instance_safe, kernel_safe)
        assert result.issues == ()


def test_safe_effect_needs_no_acknowledgement():
    instance = _instance(
        "particle_velocity[particle_index] = velocity",
        fields=_velocity(FieldAccess.READ_WRITE_CURRENT))
    assert validate_effect(instance).issues == ()


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
