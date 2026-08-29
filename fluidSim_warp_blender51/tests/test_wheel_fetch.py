from __future__ import annotations

import hashlib
import json
import pathlib
import tomllib

import pytest
from tools.release import fetch_warp_wheels

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"
WHEEL = "warp_lang-1.16.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
SHA256 = "78a726fbea71c8ff70642ae7a0a44c31d61518500a023e0cb7918df7fbe78e6b"
SIZE = 163074461


def test_lock_pins_exact_external_warp_validation_wheel():
    manifest = tomllib.loads((ROOT / "blender_manifest.toml").read_text(encoding="utf-8"))
    lock = json.loads((TOOLS_RELEASE / "wheels.lock.json").read_text(encoding="utf-8"))
    assert "wheels" not in manifest or manifest["wheels"] == []
    assert lock["warp_version"] == "1.16.0"
    assert lock["distribution_version"] == "1.16.0+cu12"
    assert set(lock["files"]) == {WHEEL}
    assert lock["files"][WHEEL] == {
        "platform": "linux-x64",
        "sha256": SHA256,
        "size": SIZE,
        "source": "local-certified-binary",
    }


def _write_lock(path: pathlib.Path, filename: str, payload: bytes) -> pathlib.Path:
    lock = {
        "warp_version": "test",
        "distribution_version": "test+local",
        "files": {
            filename: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "source": "local-certified-binary",
                "platform": "linux-x64",
            }
        },
    }
    lock_path = path / "wheels.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return lock_path


def test_fetch_wheels_copies_and_verifies_local_certified_payload(tmp_path):
    filename = "warp_test.whl"
    payload = b"valid wheel payload"
    lock_path = _write_lock(tmp_path, filename, payload)
    source = tmp_path / "incoming"
    source.mkdir()
    (source / filename).write_bytes(payload)

    written = fetch_warp_wheels.fetch_wheels(
        lock_path=lock_path,
        wheel_dir=tmp_path / "wheels",
        source=source,
    )

    assert written == [tmp_path / "wheels" / filename]
    assert written[0].read_bytes() == payload
    assert not list((tmp_path / "wheels").glob("*.part"))


def test_fetch_wheels_accepts_an_exact_source_file(tmp_path):
    filename = "warp_test.whl"
    payload = b"valid wheel payload"
    lock_path = _write_lock(tmp_path, filename, payload)
    source_file = tmp_path / filename
    source_file.write_bytes(payload)

    written = fetch_warp_wheels.fetch_wheels(
        lock_path=lock_path,
        wheel_dir=tmp_path / "wheels",
        source=source_file,
    )
    assert written[0].read_bytes() == payload


def test_fetch_wheels_rejects_local_payload_drift(tmp_path):
    filename = "warp_test.whl"
    payload = b"valid wheel payload"
    lock_path = _write_lock(tmp_path, filename, payload)
    source = tmp_path / "incoming"
    source.mkdir()
    (source / filename).write_bytes(b"invalid wheel payload")

    with pytest.raises(RuntimeError, match="size mismatch|checksum mismatch"):
        fetch_warp_wheels.fetch_wheels(
            lock_path=lock_path,
            wheel_dir=tmp_path / "wheels",
            source=source,
        )


def test_fetch_wheels_fails_closed_without_existing_or_declared_source(tmp_path):
    filename = "warp_test.whl"
    payload = b"valid wheel payload"
    lock_path = _write_lock(tmp_path, filename, payload)

    with pytest.raises(RuntimeError, match="no verified source"):
        fetch_warp_wheels.fetch_wheels(
            lock_path=lock_path,
            wheel_dir=tmp_path / "wheels",
        )
