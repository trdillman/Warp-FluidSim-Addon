from __future__ import annotations

from dataclasses import replace

from fluidsim_testpkg.sim.effects.enums import EffectCadence, EffectDomain, EffectStage, ParameterKind
from fluidsim_testpkg.sim.effects.fingerprint import effect_fingerprint, stack_fingerprint
from fluidsim_testpkg.sim.effects.schema import (
    EffectDefinitionV2,
    EffectInstanceV2,
    KernelDefinitionV2,
    ParameterDefinitionV2,
)


def _instance(instance_id: str = "one") -> EffectInstanceV2:
    definition = EffectDefinitionV2(
        effect_id="test.fingerprint",
        version="1.0.0",
        name="Fingerprint",
        summary="Fingerprint",
        description="Fingerprint",
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
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source="value = strength",
                parameter_keys=("strength",),
            ),
        ),
    )
    return EffectInstanceV2(
        instance_id=instance_id,
        definition=definition,
        parameter_values={"strength": 1.0},
        controller_bindings={
            "controller": {
                "object_id": "controller-1",
                "matrix_world": [1.0, 0.0, 0.0, 0.0] * 4,
            }
        },
    )


def test_fingerprint_is_stable_for_equivalent_mapping_order():
    first = _instance()
    second = replace(
        first,
        controller_bindings={
            "controller": {
                "matrix_world": [1.0, 0.0, 0.0, 0.0] * 4,
                "object_id": "controller-1",
            }
        },
    )
    assert effect_fingerprint(first) == effect_fingerprint(second)
    assert len(effect_fingerprint(first)) == 64


def test_behavior_changes_mutate_fingerprint():
    base = _instance()
    mutations = [
        replace(base, enabled=False),
        replace(base, muted=True),
        replace(base, solo=True),
        replace(base, order=3),
        replace(base, frame_start=10),
        replace(base, frame_end=20),
        replace(base, frame_step=2),
        replace(base, parameter_values={"strength": 2.0}),
        replace(base, controller_bindings={"controller": {"object_id": "other"}}),
        replace(
            base,
            definition=replace(
                base.definition,
                kernels=(replace(base.definition.kernels[0], source="value = strength * 2.0"),),
            ),
        ),
        replace(
            base,
            definition=replace(
                base.definition,
                kernels=(replace(base.definition.kernels[0], cadence=EffectCadence.ONCE_PER_FRAME),),
            ),
        ),
    ]
    original = effect_fingerprint(base)
    assert all(effect_fingerprint(item) != original for item in mutations)


def test_stack_order_changes_aggregate_signature_and_inventory():
    one = _instance("one")
    two = replace(_instance("two"), order=1)
    sig_a, inventory_a = stack_fingerprint((one, two))
    sig_b, inventory_b = stack_fingerprint((two, one))
    assert sig_a != sig_b
    assert [item["instance_id"] for item in inventory_a] == ["one", "two"]
    assert [item["instance_id"] for item in inventory_b] == ["two", "one"]


def test_execution_stack_excludes_disabled_and_muted_and_honors_solo():
    from dataclasses import replace
    from fluidsim_testpkg.sim.effects.fingerprint import (
        effective_stack_instances,
        execution_stack_fingerprint,
    )

    base = _instance("base")
    muted = replace(_instance("muted"), muted=True, order=1)
    disabled = replace(_instance("disabled"), enabled=False, order=2)
    solo = replace(_instance("solo"), solo=True, order=3)
    other = replace(_instance("other"), order=4)
    effective = effective_stack_instances((base, muted, disabled, solo, other))
    assert [item.instance_id for item in effective] == ["solo"]
    signature, inventory = execution_stack_fingerprint((base, muted, disabled, solo, other))
    assert len(signature) == 64
    assert [item["instance_id"] for item in inventory] == ["solo"]


def test_execution_stack_empty_identity_is_blank_and_duplicate_instance_ids_fail():
    from dataclasses import replace
    import pytest
    from fluidsim_testpkg.sim.effects.fingerprint import (
        effective_stack_instances,
        execution_stack_fingerprint,
    )

    disabled = replace(_instance("disabled"), enabled=False)
    assert execution_stack_fingerprint((disabled,)) == ("", ())
    with pytest.raises(ValueError, match="duplicate effect instance_id"):
        effective_stack_instances((_instance("same"), _instance("same")))
