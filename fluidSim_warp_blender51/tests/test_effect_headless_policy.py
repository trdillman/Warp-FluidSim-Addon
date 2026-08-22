from __future__ import annotations

from dataclasses import replace

from fluidsim_testpkg.sim.effects.builtins import builtin_effect_definitions
from fluidsim_testpkg.sim.effects.policy import evaluate_custom_code_policy
from fluidsim_testpkg.sim.effects.schema import EffectInstanceV2
from fluidsim_testpkg.sim.effects.trust import TrustStore, content_trust_digest


def _instance(*, built_in=True, enabled=True, muted=False):
    definition = builtin_effect_definitions()[0]
    if not built_in:
        definition = replace(
            definition,
            effect_id="local.reviewed-vortex",
            built_in=False,
            package_origin="file:///tmp/reviewed-vortex.fseffect",
        )
    return EffectInstanceV2(
        "effect-1", definition, enabled=enabled, muted=muted
    )


def test_deny_rejects_any_active_effect(tmp_path):
    decision = evaluate_custom_code_policy(
        (_instance(),), "deny", TrustStore(tmp_path / "trust.json")
    )
    assert decision.allowed is False
    assert decision.code == "CUSTOM_CODE_DENIED"


def test_builtins_only_allows_builtins_and_rejects_local_code(tmp_path):
    store = TrustStore(tmp_path / "trust.json")
    assert evaluate_custom_code_policy((_instance(),), "builtins-only", store).allowed
    decision = evaluate_custom_code_policy(
        (_instance(built_in=False),), "builtins-only", store
    )
    assert decision.allowed is False
    assert decision.code == "CUSTOM_CODE_NOT_BUILTIN"


def test_trusted_policy_requires_exact_local_digest(tmp_path):
    instance = _instance(built_in=False)
    store = TrustStore(tmp_path / "trust.json")
    assert not evaluate_custom_code_policy((instance,), "trusted", store).allowed
    store.trust(content_trust_digest(instance.definition), "reviewed")
    assert evaluate_custom_code_policy((instance,), "trusted", store).allowed
    edited = replace(instance, definition=replace(
        instance.definition,
        kernels=(replace(
            instance.definition.kernels[0],
            source=instance.definition.kernels[0].source + "\n# edit",
        ),),
    ))
    assert not evaluate_custom_code_policy((edited,), "trusted", store).allowed


def test_disabled_or_muted_effects_do_not_execute_under_policy(tmp_path):
    store = TrustStore(tmp_path / "trust.json")
    disabled = _instance(built_in=False, enabled=False)
    muted = replace(_instance(built_in=False, muted=True), instance_id="effect-2")
    decision = evaluate_custom_code_policy(
        (disabled, muted),
        "deny",
        store,
    )
    assert decision.allowed
    assert decision.active_effects == ()


def test_cli_contract_exposes_explicit_custom_code_policy(tmp_path):
    from fluidsim_testpkg.blender.command_line import (
        EXIT_CUSTOM_CODE_POLICY,
        CliOptions,
        parse_args,
    )

    options = parse_args([
        "--cache", str(tmp_path / "cache"),
        "--custom-code-policy", "trusted"
    ])
    assert options.custom_code_policy == "trusted"
    assert EXIT_CUSTOM_CODE_POLICY == 17
    assert CliOptions.__dataclass_fields__["custom_code_policy"].default == "builtins-only"


def test_builtins_only_requires_exact_shipped_content_not_a_spoofed_flag(tmp_path):
    instance = _instance()
    modified = replace(
        instance,
        definition=replace(
            instance.definition,
            kernels=(replace(
                instance.definition.kernels[0],
                source=instance.definition.kernels[0].source + "\npass",
            ),),
        ),
    )
    decision = evaluate_custom_code_policy(
        (modified,), "builtins-only", TrustStore(tmp_path / "trust.json")
    )
    assert decision.allowed is False
    assert decision.code == "CUSTOM_CODE_NOT_BUILTIN"


def test_duplicate_instance_ids_return_structured_policy_rejection(tmp_path):
    first = _instance()
    duplicate = replace(first, definition=builtin_effect_definitions()[1])
    decision = evaluate_custom_code_policy(
        (first, duplicate), "builtins-only", TrustStore(tmp_path / "trust.json")
    )
    assert decision.allowed is False
    assert decision.code == "INVALID_EFFECT_STACK"
    assert "duplicate" in decision.message.lower()
