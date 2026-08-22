from __future__ import annotations

import json
import zipfile
from dataclasses import replace

import pytest
from fluidsim_testpkg.sim.effects import package as effect_package
from fluidsim_testpkg.sim.effects.builtins import builtin_effect_definitions
from fluidsim_testpkg.sim.effects.enums import TrustState
from fluidsim_testpkg.sim.effects.package import (
    migrate_legacy_preset,
    read_effect_package,
    write_effect_package,
)
from fluidsim_testpkg.sim.effects.trust import TrustStore, content_trust_digest


def _definition():
    return builtin_effect_definitions()[0]


def test_package_round_trip_is_strict_and_deterministic(tmp_path):
    path = tmp_path / "vortex.fseffect"
    report = write_effect_package(path, _definition())
    assert report.path == path
    assert len(report.sha256) == 64
    restored = read_effect_package(path)
    assert restored == _definition()
    second = tmp_path / "vortex-2.fseffect"
    write_effect_package(second, _definition())
    assert second.read_bytes() == path.read_bytes()


def test_package_rejects_path_traversal(tmp_path):
    path = tmp_path / "bad.fseffect"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.py", "pass")
        archive.writestr("effect.json", "{}")
    with pytest.raises(ValueError, match="unsafe archive member"):
        read_effect_package(path)


def test_package_rejects_duplicate_members(tmp_path):
    path = tmp_path / "duplicate.fseffect"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("effect.json", "{}")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("effect.json", "{}")
    with pytest.raises(ValueError, match="duplicate archive member"):
        read_effect_package(path)


def test_package_rejects_digest_mismatch(tmp_path):
    valid = tmp_path / "valid.fseffect"
    write_effect_package(valid, _definition())
    bad = tmp_path / "bad.fseffect"
    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(bad, "w") as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename.startswith("sources/"):
                payload += b"\n# corruption\n"
            target.writestr(info.filename, payload)
    with pytest.raises(ValueError, match="digest mismatch"):
        read_effect_package(bad)


def test_package_rejects_unknown_schema(tmp_path):
    valid = tmp_path / "valid.fseffect"
    write_effect_package(valid, _definition())
    bad = tmp_path / "bad.fseffect"
    with zipfile.ZipFile(valid) as source:
        members = {item.filename: source.read(item.filename) for item in source.infolist()}
    payload = json.loads(members["effect.json"])
    payload["package_schema_version"] = 99
    members["effect.json"] = json.dumps(payload).encode()
    with zipfile.ZipFile(bad, "w") as target:
        for name, data in members.items():
            target.writestr(name, data)
    with pytest.raises(ValueError, match="package schema"):
        read_effect_package(bad)


def test_package_rejects_excessive_member_count_before_reading_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(effect_package, "MAX_PACKAGE_MEMBERS", 2)
    path = tmp_path / "too-many.fseffect"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("effect.json", "{}")
        archive.writestr("sources/a.py", "pass")
        archive.writestr("sources/b.py", "pass")
    with pytest.raises(ValueError, match="too many archive members"):
        read_effect_package(path)


def test_package_rejects_oversized_or_high_ratio_members_before_decompression(
    tmp_path, monkeypatch
):
    oversized = tmp_path / "oversized.fseffect"
    monkeypatch.setattr(effect_package, "MAX_PACKAGE_MEMBER_BYTES", 8)
    with zipfile.ZipFile(oversized, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("effect.json", b"123456789")
    with pytest.raises(ValueError, match="uncompressed size"):
        read_effect_package(oversized)

    compressed = tmp_path / "ratio.fseffect"
    monkeypatch.setattr(effect_package, "MAX_PACKAGE_MEMBER_BYTES", 1024)
    monkeypatch.setattr(effect_package, "MAX_PACKAGE_COMPRESSION_RATIO", 2.0)
    with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("effect.json", b"0" * 512)
    with pytest.raises(ValueError, match="compression ratio"):
        read_effect_package(compressed)


def test_imported_package_does_not_import_trust_claim(tmp_path):
    custom = replace(_definition(), built_in=False, package_origin="external")
    path = tmp_path / "custom.fseffect"
    write_effect_package(path, custom)
    restored = read_effect_package(path)
    store = TrustStore(tmp_path / "trust.json")
    assert store.status(content_trust_digest(restored), built_in=False) is TrustState.UNTRUSTED_IMPORTED


def test_legacy_migration_preserves_source_and_maps_voxel_tile():
    legacy = {
        "schema_version": 1,
        "name": "Old Cool",
        "kernel_name": "old_cool",
        "run_over": "VOXEL_TILE",
        "order": "BEFORE_FORCES",
        "code": "temperature[flat_voxel_index] *= 0.9",
        "arguments": [{"mode": "FIELD", "field": "temperature"}],
    }
    migrated, warnings = migrate_legacy_preset(legacy)
    assert migrated.name == "Old Cool"
    assert migrated.kernels[0].source == legacy["code"]
    assert migrated.kernels[0].domain.value == "ACTIVE_VOXELS"
    assert any("VOXEL_TILE" in warning for warning in warnings)
