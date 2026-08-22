from __future__ import annotations

import numpy as np
import pytest
from fluidsim_testpkg.io.cache_schema import CacheManifestV2
from fluidsim_testpkg.io.checkpoint import (
    CheckpointPayload,
    read_checkpoint,
    write_checkpoint_atomic,
)

SIG = "a" * 64
INVENTORY = (
    {
        "instance_id": "vortex-1",
        "effect_id": "fluidsim.vortex_force",
        "effect_version": "1.0.0",
        "fingerprint": "b" * 64,
        "order": 0,
    },
)


def test_manifest_effect_identity_round_trip_and_legacy_defaults():
    manifest = CacheManifestV2.new(
        solver_version="1.0.251",
        scene_key="scene",
        settings_signature="settings",
        source_signature="sources",
        effect_signature=SIG,
        effect_inventory=INVENTORY,
        frame_start=1,
        frame_end=2,
    )
    restored = CacheManifestV2.from_dict(manifest.to_dict())
    assert restored.effect_signature == SIG
    assert restored.effect_inventory == INVENTORY

    legacy = manifest.to_dict()
    legacy.pop("effect_signature")
    legacy.pop("effect_inventory")
    restored_legacy = CacheManifestV2.from_dict(legacy)
    assert restored_legacy.effect_signature == ""
    assert restored_legacy.effect_inventory == ()


def test_manifest_rejects_inventory_without_signature():
    with pytest.raises(ValueError, match="effect inventory requires"):
        CacheManifestV2.new(
            solver_version="1.0.251",
            scene_key="scene",
            settings_signature="settings",
            source_signature="sources",
            effect_signature="",
            effect_inventory=INVENTORY,
            frame_start=1,
            frame_end=2,
        )


def test_checkpoint_effect_identity_round_trip_and_mismatch(tmp_path):
    payload = CheckpointPayload(
        frame=1,
        sim_time=0.1,
        settings_signature="settings",
        source_signature="sources",
        effect_signature=SIG,
        effect_inventory=INVENTORY,
        arrays={"particle_position": np.zeros((1, 3), dtype=np.float32)},
    )
    path = tmp_path / "checkpoint.npz"
    digest = write_checkpoint_atomic(path, payload)
    restored = read_checkpoint(
        path,
        expected_sha256=digest,
        expected_effect_signature=SIG,
    )
    assert restored.effect_signature == SIG
    assert restored.effect_inventory == INVENTORY
    with pytest.raises(ValueError, match="effect signature mismatch"):
        read_checkpoint(path, expected_effect_signature="c" * 64)


def test_legacy_checkpoint_is_playable_but_not_resumable_with_active_effects(tmp_path):
    payload = CheckpointPayload(
        frame=1,
        sim_time=0.1,
        settings_signature="settings",
        source_signature="sources",
        arrays={},
    )
    path = tmp_path / "legacy.npz"
    write_checkpoint_atomic(path, payload)
    restored = read_checkpoint(path)
    assert restored.effect_signature == ""
    with pytest.raises(ValueError, match="effect signature mismatch"):
        read_checkpoint(path, expected_effect_signature=SIG)


def test_production_bake_binds_effect_identity_to_every_transaction_surface():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "blender" / "production_bake.py").read_text(encoding="utf-8")
    for token in (
        "effect_identity",
        "EFFECT_SIGNATURE_MISMATCH",
        "effect_signature=self.effect_signature",
        "effect_inventory=self.effect_inventory",
        "expected_effect_signature=self.manifest.effect_signature",
    ):
        assert token in source


def test_cache_v2_frame_payload_carries_effect_identity():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "io" / "cache.py").read_text(encoding="utf-8")
    assert '"effect_signature"' in source
    assert '"effect_inventory_json"' in source
    assert "effect-signature mismatch" in source
