"""Non-executing trust policy for background Custom Effects use."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import TrustState
from .trust import TrustStore, content_trust_digest
from .fingerprint import effective_stack_instances


@dataclass(frozen=True)
class CustomCodePolicyDecision:
    allowed: bool
    code: str
    message: str
    policy: str
    active_effects: tuple[dict[str, object], ...] = ()


def _active(instances):
    return effective_stack_instances(tuple(instances))


def evaluate_custom_code_policy(
    instances,
    policy: str,
    trust_store: TrustStore,
) -> CustomCodePolicyDecision:
    """Evaluate policy without compiling, importing, or launching effect code."""

    normalized = str(policy).strip().lower().replace("_", "-")
    if normalized not in {"deny", "builtins-only", "trusted"}:
        raise ValueError(f"unknown custom-code policy {policy!r}")
    try:
        active = _active(tuple(instances))
    except ValueError as exc:
        return CustomCodePolicyDecision(
            False,
            "INVALID_EFFECT_STACK",
            f"Custom Effect stack is invalid: {exc}",
            normalized,
            (),
        )
    inventory_rows = []
    for item in active:
        digest = content_trust_digest(item.definition)
        trust_state = trust_store.status(digest, built_in=item.definition.built_in)
        inventory_rows.append({
            "instance_id": item.instance_id,
            "effect_id": item.definition.effect_id,
            "effect_version": item.definition.version,
            "built_in_claimed": bool(item.definition.built_in),
            "shipped_builtin": trust_state is TrustState.BUILT_IN_TRUSTED,
            "trust_state": trust_state.value,
            "content_digest": digest,
        })
    inventory = tuple(inventory_rows)
    if not active:
        return CustomCodePolicyDecision(
            True, "NO_CUSTOM_CODE", "No active Custom Effects will execute.",
            normalized, (),
        )
    if normalized == "deny":
        return CustomCodePolicyDecision(
            False,
            "CUSTOM_CODE_DENIED",
            "Active Custom Effects are forbidden by --custom-code-policy=deny.",
            normalized,
            inventory,
        )
    if normalized == "builtins-only":
        forbidden = tuple(
            item for item in active
            if trust_store.status(
                content_trust_digest(item.definition),
                built_in=item.definition.built_in,
            ) is not TrustState.BUILT_IN_TRUSTED
        )
        if forbidden:
            names = ", ".join(item.definition.name for item in forbidden)
            return CustomCodePolicyDecision(
                False,
                "CUSTOM_CODE_NOT_BUILTIN",
                f"Non-built-in Custom Effects are forbidden by policy: {names}",
                normalized,
                inventory,
            )
        return CustomCodePolicyDecision(
            True,
            "CUSTOM_CODE_BUILTINS_ALLOWED",
            "All active Custom Effects are bundled FluidSim effects.",
            normalized,
            inventory,
        )
    rejected = []
    for item in active:
        digest = content_trust_digest(item.definition)
        state = trust_store.status(digest, built_in=item.definition.built_in)
        if state not in {TrustState.BUILT_IN_TRUSTED, TrustState.LOCAL_TRUSTED}:
            rejected.append((item.definition.name, state.value))
    if rejected:
        rendered = ", ".join(f"{name} ({state})" for name, state in rejected)
        return CustomCodePolicyDecision(
            False,
            "CUSTOM_CODE_UNTRUSTED",
            "Active Custom Effects are not trusted at their exact content digest: " + rendered,
            normalized,
            inventory,
        )
    return CustomCodePolicyDecision(
        True,
        "CUSTOM_CODE_TRUSTED",
        "All active Custom Effects are built-in or locally trusted by exact SHA-256.",
        normalized,
        inventory,
    )
