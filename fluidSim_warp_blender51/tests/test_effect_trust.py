from __future__ import annotations

import json

from fluidsim_testpkg.sim.effects.enums import EffectCadence, EffectDomain, EffectStage, TrustState
from fluidsim_testpkg.sim.effects.schema import EffectDefinitionV2, KernelDefinitionV2
from fluidsim_testpkg.sim.effects.trust import TrustStore, content_trust_digest


def _definition(source: str = "pass", *, built_in: bool = False) -> EffectDefinitionV2:
    return EffectDefinitionV2(
        effect_id="test.trust",
        version="1.0.0",
        name="Trust",
        summary="Trust",
        description="Trust",
        category="Tests",
        author="Tests",
        license="GPL-3.0-or-later",
        built_in=built_in,
        kernels=(
            KernelDefinitionV2(
                kernel_id="kernel",
                label="Kernel",
                domain=EffectDomain.PARTICLES,
                stage=EffectStage.AFTER_PARTICLE_TRANSFER,
                cadence=EffectCadence.EVERY_SUBSTEP,
                source=source,
            ),
        ),
    )


def test_builtin_digest_is_trusted_without_local_record(tmp_path):
    from fluidsim_testpkg.sim.effects.builtins import builtin_effect_definitions

    store = TrustStore(tmp_path / "trust.json")
    definition = builtin_effect_definitions()[0]
    assert store.status(content_trust_digest(definition), built_in=True) is TrustState.BUILT_IN_TRUSTED


def test_local_trust_attaches_to_exact_digest_and_edit_invalidates_it(tmp_path):
    store = TrustStore(tmp_path / "trust.json")
    original = _definition("value = 1.0")
    edited = _definition("value = 2.0")
    digest = content_trust_digest(original)
    store.trust(digest, "Test Trust")
    assert store.status(digest) is TrustState.LOCAL_TRUSTED
    assert store.status(content_trust_digest(edited)) is TrustState.UNTRUSTED_IMPORTED


def test_blocked_digest_wins_over_local_trust(tmp_path):
    store = TrustStore(tmp_path / "trust.json")
    digest = content_trust_digest(_definition())
    store.trust(digest, "Test")
    store.block(digest, "Known unsafe")
    assert store.status(digest) is TrustState.BLOCKED


def test_trust_store_writes_strict_atomic_json(tmp_path):
    path = tmp_path / "trust.json"
    store = TrustStore(path)
    digest = content_trust_digest(_definition())
    store.trust(digest, "Test")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["records"][digest]["state"] == "LOCAL_TRUSTED"
    assert not path.with_name(path.name + ".tmp").exists()


def test_builtin_flag_cannot_trust_modified_or_spoofed_content(tmp_path):
    from dataclasses import replace

    from fluidsim_testpkg.sim.effects.builtins import builtin_effect_definitions

    store = TrustStore(tmp_path / "trust.json")
    shipped = builtin_effect_definitions()[0]
    modified_kernel = replace(shipped.kernels[0], source=shipped.kernels[0].source + "\npass")
    modified = replace(shipped, kernels=(modified_kernel, *shipped.kernels[1:]))
    spoofed = _definition("pass", built_in=True)

    assert store.status(
        content_trust_digest(shipped), built_in=True
    ) is TrustState.BUILT_IN_TRUSTED
    assert store.status(
        content_trust_digest(modified), built_in=True
    ) is TrustState.UNTRUSTED_IMPORTED
    assert store.status(
        content_trust_digest(spoofed), built_in=True
    ) is TrustState.UNTRUSTED_IMPORTED
