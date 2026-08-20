from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from fluidsim_testpkg.release import delivery_policy


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_delivery_policy_accepts_addon_blender_source_and_nested_clean_zip(tmp_path: Path):
    inner = _zip_bytes({"blender/scene_bridge.py": b"# source", "runtime/warp_runtime.py": b"# source"})
    outer = tmp_path / "delivery.zip"
    outer.write_bytes(_zip_bytes({"installable/fluidsim.zip": inner, "README.md": b"ok"}))
    assert delivery_policy.verify_delivery_zip(outer) == []


@pytest.mark.parametrize(
    "name",
    [
        "warp_lang-1.16.0+cu12-py3-none-manylinux_2_28_x86_64.whl",
        "runtime/blender-5.2.0-linux-x64.tar.xz",
        "runtime/blender-5.2.0-linux-x64.zip",
        "vendor/warp/__init__.py",
        "vendor/warp_lang-1.16.0.dist-info/METADATA",
    ],
)
def test_delivery_policy_rejects_runtime_payloads_at_top_level(tmp_path: Path, name: str):
    outer = tmp_path / "delivery.zip"
    outer.write_bytes(_zip_bytes({name: b"forbidden"}))
    violations = delivery_policy.verify_delivery_zip(outer)
    assert violations
    assert name in violations[0]


def test_delivery_policy_rejects_wheel_hidden_in_nested_installable(tmp_path: Path):
    inner = _zip_bytes({"wheels/warp_lang-1.16.0.whl": b"forbidden"})
    outer = tmp_path / "delivery.zip"
    outer.write_bytes(_zip_bytes({"installable/fluidsim.zip": inner}))
    violations = delivery_policy.verify_delivery_zip(outer)
    assert violations
    assert "installable/fluidsim.zip!wheels/warp_lang-1.16.0.whl" in violations[0]
