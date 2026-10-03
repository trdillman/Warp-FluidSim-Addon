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

    for _filename, meta in lock["files"].items():
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
    """A certified install is success and does not download another copy."""
    from fluidSim_warp_blender51.warp_installer import install_warp

    with mock.patch(
        "fluidSim_warp_blender51.warp_installer._get_installed_warp_version",
        return_value="1.17.0",
    ):
        with mock.patch("fluidSim_warp_blender51.warp_installer._download_wheel") as download:
            result = install_warp()
    assert result.success is True
    assert result.already_installed is True
    assert "already installed" in result.message.lower()
    download.assert_not_called()


def test_install_warp_leaves_a_newer_untested_warp_in_place():
    """A newer 1.x Warp is already accepted; the button must not add a second copy."""
    from fluidSim_warp_blender51.warp_installer import install_warp

    with mock.patch(
        "fluidSim_warp_blender51.warp_installer._get_installed_warp_version",
        return_value="1.18.0",
    ):
        with mock.patch("fluidSim_warp_blender51.warp_installer._download_wheel") as download:
            result = install_warp()
    assert result.success is True
    assert result.already_installed is True
    download.assert_not_called()


def test_install_warp_replaces_an_incompatible_warp_from_the_locked_url(tmp_path):
    """An older Warp is replaced in its own directory from the locked URL only."""
    from fluidSim_warp_blender51.warp_installer import install_warp

    payload = b"certified-wheel"
    info = mock.Mock(
        filename="warp_lang-1.17.0-py3-none-manylinux_2_28_x86_64.whl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        url="https://files.pythonhosted.org/packages/locked/warp.whl",
        platform="linux-x64",
    )
    site = tmp_path / "site-packages"
    site.mkdir()
    commands = []

    def fake_download(url, destination, **_kwargs):
        assert url == info.url
        destination.write_bytes(payload)

    def fake_pip(wheel, destination, *, force):
        commands.append((wheel.name, destination, force))
        return None

    inserted_warp = "warp" not in sys.modules
    if inserted_warp:
        sys.modules["warp"] = mock.Mock()
    try:
        with mock.patch(
            "fluidSim_warp_blender51.warp_installer._get_installed_warp_version",
            return_value="1.16.0",
        ), mock.patch(
            "fluidSim_warp_blender51.warp_installer.get_wheel_info_for_platform",
            return_value=info,
        ), mock.patch(
            "fluidSim_warp_blender51.warp_installer._loaded_warp_site_packages",
            return_value=site,
        ), mock.patch(
            "fluidSim_warp_blender51.warp_installer._download_wheel",
            side_effect=fake_download,
        ), mock.patch(
            "fluidSim_warp_blender51.warp_installer._pip_install",
            side_effect=fake_pip,
        ), mock.patch(
            "fluidSim_warp_blender51.warp_installer._sha256_path",
            return_value=info.sha256,
        ):
            result = install_warp()
    finally:
        if inserted_warp:
            sys.modules.pop("warp", None)

    assert result.success is True
    assert result.already_installed is False
    assert "Restart Blender" in result.message
    assert commands == [(info.filename, site, True)]


def test_install_warp_reports_pip_failure_without_claiming_success(tmp_path):
    from fluidSim_warp_blender51.warp_installer import install_warp

    payload = b"certified-wheel"
    info = mock.Mock(
        filename="warp.whl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        url="https://files.pythonhosted.org/packages/locked/warp.whl",
        platform="linux-x64",
    )

    def fake_download(_url, destination, **_kwargs):
        destination.write_bytes(payload)

    with mock.patch(
        "fluidSim_warp_blender51.warp_installer._get_installed_warp_version",
        return_value=None,
    ), mock.patch(
        "fluidSim_warp_blender51.warp_installer.get_wheel_info_for_platform",
        return_value=info,
    ), mock.patch(
        "fluidSim_warp_blender51.warp_installer._default_install_target",
        return_value=tmp_path,
    ), mock.patch(
        "fluidSim_warp_blender51.warp_installer._download_wheel",
        side_effect=fake_download,
    ), mock.patch(
        "fluidSim_warp_blender51.warp_installer._sha256_path",
        return_value=info.sha256,
    ), mock.patch(
        "fluidSim_warp_blender51.warp_installer._pip_install",
        return_value="pip install failed: permission denied",
    ):
        result = install_warp()
    assert result.success is False
    assert "permission denied" in result.message


def test_shipped_wheel_lock_matches_the_release_lock():
    shipped = json.loads((ROOT / "warp_wheels.lock.json").read_text(encoding="utf-8"))
    release = json.loads((TOOLS_RELEASE / "wheels.lock.json").read_text(encoding="utf-8"))
    assert shipped == release
    assert shipped["warp_version"] == "1.17.0"


def test_preferences_register_the_install_warp_button():
    source = (ROOT / "preferences.py").read_text(encoding="utf-8")
    assert 'bl_idname = "fluidsim.install_warp"' in source
    assert 'text="Install Warp"' in source
    assert "network" not in source or "PyPI" in (ROOT / "blender_manifest.toml").read_text()


def test_install_warp_fails_for_unsupported_platform():
    """Verify install_warp returns failure for unsupported platforms."""
    from fluidSim_warp_blender51.warp_installer import install_warp

    with mock.patch(
        "fluidSim_warp_blender51.warp_installer._get_installed_warp_version",
        return_value=None,
    ), mock.patch(
        "fluidSim_warp_blender51.warp_installer._host_platform",
        return_value=None,
    ):
        result = install_warp()
    assert result.success is False
    assert "platform" in result.message.lower()


def test_download_wheel_verifies_checksum():
    """Verify that _download_wheel rejects payloads with wrong checksums."""
    import tempfile

    from fluidSim_warp_blender51.warp_installer import _download_wheel

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
    import tempfile

    from fluidSim_warp_blender51.warp_installer import _download_wheel

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
    from fluidSim_warp_blender51.version import CERTIFIED_WARP_DISTRIBUTION, CERTIFIED_WARP_VERSION

    lock = json.loads((TOOLS_RELEASE / "wheels.lock.json").read_text(encoding="utf-8"))
    assert lock["warp_version"] == CERTIFIED_WARP_VERSION
    assert lock["distribution_version"] == CERTIFIED_WARP_DISTRIBUTION
