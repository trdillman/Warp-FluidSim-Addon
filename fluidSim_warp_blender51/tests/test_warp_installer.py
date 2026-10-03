"""Tests for the Warp installer module."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from unittest import mock

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS_RELEASE = ROOT.parent / "tools" / "release"

# Ensure the addon package is importable
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))


def test_wheel_lock_exists_and_has_valid_structure():
    """Verify the wheel lock file exists and has the expected structure."""
    lock_path = TOOLS_RELEASE / "wheels.lock.json"
    assert lock_path.is_file(), f"Wheel lock file not found: {lock_path}"

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert "warp_version" in lock
    assert "distribution_version" in lock
    assert "files" in lock
    assert isinstance(lock["files"], dict)
    assert len(lock["files"]) >= 1

    for filename, meta in lock["files"].items():
        assert "sha256" in meta
        assert "size" in meta
        assert "url" in meta
        assert "platform" in meta
        assert len(meta["sha256"]) == 64
        assert isinstance(meta["size"], int)
        assert meta["size"] > 0
        assert meta["url"].startswith("https://")


def test_get_wheel_info_returns_info_for_supported_platforms():
    """Verify that get_wheel_info_for_platform returns info for known platforms."""
    from fluidSim_warp_blender51.warp_installer import get_wheel_info_for_platform

    for platform in ("windows-x64", "linux-x64"):
        info = get_wheel_info_for_platform(platform)
        assert info is not None, f"No wheel info for platform: {platform}"
        assert info.platform == platform
        assert info.filename.endswith(".whl")
        assert len(info.sha256) == 64
        assert info.size > 0
        assert info.url.startswith("https://")


def test_get_wheel_info_returns_none_for_unsupported_platform():
    """Verify that get_wheel_info_for_platform returns None for unsupported platforms."""
    from fluidSim_warp_blender51.warp_installer import get_wheel_info_for_platform

    assert get_wheel_info_for_platform("macos-arm64") is None
    assert get_wheel_info_for_platform("linux-arm64") is None


def test_install_result_dataclass():
    """Verify InstallResult dataclass works correctly."""
    from fluidSim_warp_blender51.warp_installer import InstallResult

    result = InstallResult(success=True, message="OK", version="1.17.0")
    assert result.success is True
    assert result.message == "OK"
    assert result.version == "1.17.0"
    assert result.already_installed is False

    result2 = InstallResult(
        success=True, message="Already installed", version="1.17.0", already_installed=True
    )
    assert result2.already_installed is True


def test_install_warp_returns_already_installed_when_certified():
    """Verify install_warp returns success without action when certified version is installed."""
    from fluidSim_warp_blender51.warp_installer import install_warp

    with mock.patch("fluidSim_warp_blender51.warp_installer.is_certified_warp_installed", return_value=True):
        with mock.patch("fluidSim_warp_blender51.warp_installer._get_installed_warp_version", return_value="1.17.0"):
            result = install_warp()
            assert result.success is True
            assert result.already_installed is True
            assert "already installed" in result.message.lower()


def test_install_warp_fails_for_unsupported_platform():
    """Verify install_warp returns failure for unsupported platforms."""
    from fluidSim_warp_blender51.warp_installer import install_warp

    with mock.patch("fluidSim_warp_blender51.warp_installer._host_platform", return_value=None):
        result = install_warp()
        assert result.success is False
        assert "platform" in result.message.lower()


def test_download_wheel_verifies_checksum():
    """Verify that _download_wheel rejects payloads with wrong checksums."""
    from fluidSim_warp_blender51.warp_installer import _download_wheel
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = pathlib.Path(tmpdir) / "test.whl"
        wrong_hash = "0" * 64

        class FakeResponse:
            def __init__(self, data):
                self.data = data
                self.pos = 0

            def read(self, size):
                chunk = self.data[self.pos:self.pos + size]
                self.pos += size
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with mock.patch("fluidSim_warp_blender51.warp_installer.urlopen") as mock_urlopen:
            mock_urlopen.return_value = FakeResponse(b"fake wheel data")
            with pytest.raises(RuntimeError, match="size mismatch|checksum mismatch"):
                _download_wheel(
                    "https://example.com/test.whl",
                    dest,
                    expected_sha256=wrong_hash,
                    expected_size=100,
                )


def test_download_wheel_rejects_non_https():
    """Verify that _download_wheel rejects non-HTTPS URLs."""
    from fluidSim_warp_blender51.warp_installer import _download_wheel
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = pathlib.Path(tmpdir) / "test.whl"
        with pytest.raises(RuntimeError, match="HTTPS"):
            _download_wheel(
                "http://example.com/test.whl",
                dest,
                expected_sha256="0" * 64,
                expected_size=100,
            )


def test_version_constants_match_lock():
    """Verify that version.py constants match the wheel lock."""
    from fluidSim_warp_blender51.version import CERTIFIED_WARP_VERSION, CERTIFIED_WARP_DISTRIBUTION

    lock = json.loads((TOOLS_RELEASE / "wheels.lock.json").read_text(encoding="utf-8"))
    assert lock["warp_version"] == CERTIFIED_WARP_VERSION
    assert lock["distribution_version"] == CERTIFIED_WARP_DISTRIBUTION
