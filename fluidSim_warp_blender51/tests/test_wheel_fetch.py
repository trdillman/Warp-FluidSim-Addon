from __future__ import annotations

import hashlib
import json
import pathlib
import tomllib

import pytest
from tools.release import fetch_warp_wheels

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"
WHEEL = "warp_lang-1.17.0-py3-none-manylinux_2_28_x86_64.whl"
SHA256 = "47cd93636828dd16e55eeb4e77a0e3547b5fb0f383000a3d228629df68f28fd8"
SIZE = 164347636
WINDOWS_WHEEL = "warp_lang-1.17.0-py3-none-win_amd64.whl"


def test_lock_pins_exact_bundled_warp_wheels():
    manifest = tomllib.loads((ROOT / "blender_manifest.toml").read_text(encoding="utf-8"))
    lock = json.loads((TOOLS_RELEASE / "wheels.lock.json").read_text(encoding="utf-8"))
    assert sorted(manifest["wheels"]) == [f"./wheels/{WHEEL}", f"./wheels/{WINDOWS_WHEEL}"]
    assert lock["warp_version"] == "1.17.0"
    assert lock["distribution_version"] == "1.17.0+cu12"
    assert set(lock["files"]) == {WHEEL, WINDOWS_WHEEL}
    assert lock["files"][WHEEL]["platform"] == "linux-x64"
    assert lock["files"][WHEEL]["sha256"] == SHA256
    assert lock["files"][WHEEL]["size"] == SIZE
    assert lock["files"][WHEEL]["source"] == "pypi"


def test_default_fetch_target_is_the_bundled_wheels_directory():
    assert fetch_warp_wheels.WHEEL_DIR == ROOT / "wheels"


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


def test_fetch_wheels_platform_filter_skips_other_platform_entries(tmp_path):
    payload = b"linux payload"
    lock_path = _write_lock(tmp_path, "linux.whl", payload)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["files"]["windows.whl"] = {
        "sha256": hashlib.sha256(b"windows payload").hexdigest(),
        "size": len(b"windows payload"),
        "source": "local-certified-binary",
        "platform": "windows-x64",
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    source = tmp_path / "incoming"
    source.mkdir()
    (source / "linux.whl").write_bytes(payload)

    written = fetch_warp_wheels.fetch_wheels(
        lock_path=lock_path, wheel_dir=tmp_path / "wheels", source=source,
        platform="linux-x64")
    assert written == [tmp_path / "wheels" / "linux.whl"]

    with pytest.raises(RuntimeError, match="no verified source"):
        fetch_warp_wheels.fetch_wheels(
            lock_path=lock_path, wheel_dir=tmp_path / "wheels", source=source,
            offline=True)
    with pytest.raises(RuntimeError, match="no entry for platform"):
        fetch_warp_wheels.fetch_wheels(
            lock_path=lock_path, wheel_dir=tmp_path / "wheels", platform="macos-arm64")