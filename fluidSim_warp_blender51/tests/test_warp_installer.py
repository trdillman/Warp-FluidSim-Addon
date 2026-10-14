"""Tests for the Warp installer module."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import pathlib
import sys
import types
import zipfile
from types import SimpleNamespace
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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows filesystem path limit")
def test_extract_and_clear_owned_warp_with_long_paths(tmp_path):
    from fluidSim_warp_blender51 import warp_installer

    destination = tmp_path / ("extension-profile-" * 7) / ".local"
    member = "warp_lang-1.17.0.dist-info/licenses/licenses/reproducible_floating_sums-LICENSE.txt"
    assert len(str(destination / member)) > 260
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("warp/__init__.py", "__version__ = '1.17.0'\n")
        archive.writestr(member, "license evidence")
    warp_installer._extract_wheel(wheel, destination)
    extended = pathlib.Path("\\\\?\\" + str(destination.resolve()))
    assert (extended / member).read_text() == "license evidence"
    unrelated = destination / "keep.txt"
    unrelated.write_text("preserve")
    warp_installer._clear_owned_warp(destination)
    assert not (extended / "warp").exists()
    assert not (extended / "warp_lang-1.17.0.dist-info").exists()
    assert unrelated.read_text() == "preserve"


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


def _seed_owned_warp(site: pathlib.Path, version: str) -> None:
    package = site / "warp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (site / f"warp_lang-{version}.dist-info").mkdir()


def _isolate_install(monkeypatch, owned: pathlib.Path, *, importable: pathlib.Path | None = None):
    from fluidSim_warp_blender51 import warp_installer

    monkeypatch.setattr(warp_installer, "_user_site_warp_path", lambda: None)
    monkeypatch.setattr(warp_installer, "_importable_warp_file", lambda: importable)
    monkeypatch.setattr(
        warp_installer, "_default_install_target", lambda create=False: owned
    )


def _snapshot(root: pathlib.Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_install_warp_returns_already_installed_when_certified(tmp_path, monkeypatch):
    """An owned 1.17.0 copy is success and does not download or write."""
    from fluidSim_warp_blender51 import warp_installer

    _seed_owned_warp(tmp_path, "1.17.0")
    before = _snapshot(tmp_path)
    _isolate_install(
        monkeypatch,
        tmp_path,
        importable=(tmp_path / "warp" / "__init__.py").resolve(),
    )
    download = mock.Mock()
    extract = mock.Mock()
    monkeypatch.setattr(warp_installer, "_download_wheel", download)
    monkeypatch.setattr(warp_installer, "_extract_wheel", extract)

    result = warp_installer.install_warp()

    assert result.success is True
    assert result.already_installed is True
    assert "already installed" in result.message.lower()
    download.assert_not_called()
    extract.assert_not_called()
    assert _snapshot(tmp_path) == before


def test_install_warp_repairs_a_corrupted_owned_certified_copy(tmp_path, monkeypatch):
    """A repair leaves loaded files untouched, then applies before the next import."""
    from fluidSim_warp_blender51 import warp_installer

    _seed_owned_warp(tmp_path, "1.17.0")
    (tmp_path / "warp" / "__init__.py").write_text(
        'raise RuntimeError("corrupted runtime")\n',
        encoding="utf-8",
    )
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("warp/__init__.py", '__version__ = "1.17.0"\n')
        archive.writestr("warp_lang-1.17.0.dist-info/METADATA", "Name: warp-lang\n")
    wheel_bytes = payload.getvalue()
    info = mock.Mock(
        filename="warp_lang-1.17.0-py3-none-manylinux_2_28_x86_64.whl",
        sha256=hashlib.sha256(wheel_bytes).hexdigest(),
        size=len(wheel_bytes),
        url="https://files.pythonhosted.org/packages/locked/warp.whl",
        platform="linux-x64",
    )

    def fake_download(url, destination, **_kwargs):
        assert url == info.url
        destination.write_bytes(wheel_bytes)

    _isolate_install(
        monkeypatch,
        tmp_path,
        importable=(tmp_path / "warp" / "__init__.py").resolve(),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(warp_installer, "get_wheel_info_for_platform", lambda platform=None: info)
    monkeypatch.setattr(warp_installer, "_download_wheel", fake_download)
    monkeypatch.setattr(warp_installer, "_online_access_block_reason", lambda: None)
    clear_owned = mock.Mock(wraps=warp_installer._clear_owned_warp)
    monkeypatch.setattr(warp_installer, "_clear_owned_warp", clear_owned)
    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(__file__=str(tmp_path / "warp" / "__init__.py")),
    )

    result = warp_installer.install_warp(force_reinstall=True)

    assert result.success is True
    assert result.already_installed is False
    assert result.version == "1.17.0"
    assert "Restart Blender" in result.message
    clear_owned.assert_not_called()
    assert "corrupted runtime" in (tmp_path / "warp" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert warp_installer._pending_repair_path(tmp_path).is_file()

    monkeypatch.delitem(sys.modules, "warp")
    with pytest.raises(RuntimeError, match="Restart Blender"):
        warp_installer.ensure_installed_warp_on_path()
    clear_owned.assert_not_called()

    warp_installer._pending_repair_pid_path(tmp_path).write_text("0", encoding="ascii")
    warp_installer.ensure_installed_warp_on_path()

    clear_owned.assert_called_once_with(tmp_path)
    assert not warp_installer._pending_repair_path(tmp_path).exists()
    assert (tmp_path / "warp" / "__init__.py").read_text(encoding="utf-8") == (
        '__version__ = "1.17.0"\n'
    )


@pytest.mark.parametrize("missing", ["dist-info", "init"])
def test_loaded_owned_warp_with_missing_metadata_is_staged(tmp_path, monkeypatch, missing):
    """Damaged version metadata must never allow writes over loaded native code."""
    from fluidSim_warp_blender51 import warp_installer

    _seed_owned_warp(tmp_path, "1.17.0")
    init = tmp_path / "warp" / "__init__.py"
    (tmp_path / "warp" / "native.dll").write_bytes(b"loaded-native-code")
    if missing == "dist-info":
        (tmp_path / "warp_lang-1.17.0.dist-info").rmdir()
    else:
        init.unlink()
    before = _snapshot(tmp_path)
    _isolate_install(monkeypatch, tmp_path, importable=init.resolve())
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(__file__=str(init)))
    assert warp_installer._owned_distribution_version(tmp_path) is None
    payload = b"certified-locked-test-wheel"
    info = mock.Mock(
        filename="certified.whl", size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        url="https://files.pythonhosted.org/packages/locked/warp.whl",
    )
    monkeypatch.setattr(warp_installer, "get_wheel_info_for_platform", lambda platform=None: info)
    monkeypatch.setattr(
        warp_installer, "_download_wheel",
        lambda _url, destination, **_kwargs: destination.write_bytes(payload),
    )
    monkeypatch.setattr(warp_installer, "_online_access_block_reason", lambda: None)
    extract = mock.Mock()
    clear = mock.Mock()
    monkeypatch.setattr(warp_installer, "_extract_wheel", extract)
    monkeypatch.setattr(warp_installer, "_clear_owned_warp", clear)

    result = warp_installer.install_warp(force_reinstall=True)

    assert result.success is True
    assert "Restart Blender" in result.message
    extract.assert_not_called()
    clear.assert_not_called()
    for relative, content in before.items():
        assert (tmp_path / relative).read_bytes() == content
    assert warp_installer._pending_repair_path(tmp_path).read_bytes() == payload


def test_pending_repair_rejects_tampering_before_deleting_owned_files(
    tmp_path,
    monkeypatch,
):
    """Fresh startup re-verifies a staged wheel before touching the old runtime."""
    from fluidSim_warp_blender51 import warp_installer

    _seed_owned_warp(tmp_path, "1.17.0")
    original = _snapshot(tmp_path)
    wheel = tmp_path / "source.whl"
    wheel.write_bytes(b"certified-wheel")
    info = mock.Mock(
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        size=wheel.stat().st_size,
    )
    _isolate_install(monkeypatch, tmp_path)
    monkeypatch.delitem(sys.modules, "warp", raising=False)
    monkeypatch.setattr(warp_installer, "get_wheel_info_for_platform", lambda platform=None: info)
    warp_installer._stage_pending_repair(wheel, tmp_path, info)
    warp_installer._pending_repair_path(tmp_path).write_bytes(b"tampered")
    warp_installer._pending_repair_pid_path(tmp_path).write_text("0", encoding="ascii")
    clear_owned = mock.Mock(wraps=warp_installer._clear_owned_warp)
    monkeypatch.setattr(warp_installer, "_clear_owned_warp", clear_owned)

    with pytest.raises(RuntimeError, match="checksum or size"):
        warp_installer.ensure_installed_warp_on_path()

    clear_owned.assert_not_called()
    for relative, content in original.items():
        assert (tmp_path / relative).read_bytes() == content


@pytest.mark.parametrize("cross_volume", [False, True])
def test_pending_repair_extracts_outside_a_deep_owned_path(tmp_path, monkeypatch, cross_volume):
    """Fresh repair extraction does not amplify a deep Windows extension path."""
    from fluidSim_warp_blender51 import warp_installer

    owned = tmp_path / ("deep-extension-path-" * 5) / ("repository-name-" * 4) / ".local"
    _seed_owned_warp(owned, "1.17.0")
    wheel = tmp_path / "certified.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("warp/__init__.py", '__version__ = "1.17.0"\n')
        archive.writestr(
            "warp/examples/benchmarks/benchmark_tile_matmul_pipelined.py",
            "# installed\n",
        )
        archive.writestr("warp_lang-1.17.0.dist-info/METADATA", "Name: warp-lang\n")
        archive.writestr(
            "warp_lang-1.17.0.dist-info/licenses/licenses/reproducible_floating_sums-LICENSE.txt",
            "license evidence",
        )
    info = mock.Mock(
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        size=wheel.stat().st_size,
    )
    _isolate_install(monkeypatch, owned)
    monkeypatch.delitem(sys.modules, "warp", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(warp_installer, "get_wheel_info_for_platform", lambda platform=None: info)
    warp_installer._stage_pending_repair(wheel, owned, info)
    warp_installer._pending_repair_pid_path(owned).write_text("0", encoding="ascii")
    extract_wheel = warp_installer._extract_wheel
    extraction_targets = []

    def guarded_extract(source, destination):
        extraction_targets.append(destination)
        assert not destination.resolve().is_relative_to(owned.resolve())
        extract_wheel(source, destination)

    monkeypatch.setattr(warp_installer, "_extract_wheel", guarded_extract)
    if cross_volume:
        monkeypatch.setattr(
            warp_installer.shutil.os, "rename",
            mock.Mock(side_effect=OSError(errno.EXDEV, "cross-volume promotion")),
        )

    warp_installer.ensure_installed_warp_on_path()

    assert len(extraction_targets) == 1
    assert (owned / "warp" / "examples" / "benchmarks").is_dir()
    assert not warp_installer._pending_repair_path(owned).exists()
    license_path = (
        warp_installer._filesystem_path(owned)
        / "warp_lang-1.17.0.dist-info/licenses/licenses/reproducible_floating_sums-LICENSE.txt"
    )
    assert license_path.read_text() == "license evidence"
    warp_installer._clear_owned_warp(owned)
    assert not license_path.exists()


@pytest.mark.parametrize("member", ["../escape.txt", "/escape.txt"])
def test_wheel_extraction_rejects_escaping_members(tmp_path, member):
    from fluidSim_warp_blender51 import warp_installer

    unrelated = tmp_path / "escape.txt"
    unrelated.write_text("preserve")
    wheel = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, "overwrite")
    with pytest.raises(RuntimeError, match="escapes the install directory"):
        warp_installer._extract_wheel(wheel, tmp_path / "owned")
    assert unrelated.read_text() == "preserve"


def test_pending_repair_keeps_old_wheel_if_pid_guard_cannot_be_committed(
    tmp_path,
    monkeypatch,
):
    """A marker-write failure cannot expose a new wheel under an old PID guard."""
    from fluidSim_warp_blender51 import warp_installer

    pending = warp_installer._pending_repair_path(tmp_path)
    pid_marker = warp_installer._pending_repair_pid_path(tmp_path)
    pending.write_bytes(b"previous-wheel")
    pid_marker.write_text("0", encoding="ascii")
    wheel = tmp_path / "source.whl"
    wheel.write_bytes(b"replacement-wheel")
    info = mock.Mock(
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        size=wheel.stat().st_size,
    )
    replace = pathlib.Path.replace

    def fail_pid_guard(source, target):
        if source.name == f"{pid_marker.name}.part":
            raise OSError("simulated marker write failure")
        return replace(source, target)

    monkeypatch.setattr(pathlib.Path, "replace", fail_pid_guard)

    with pytest.raises(OSError, match="marker write failure"):
        warp_installer._stage_pending_repair(wheel, tmp_path, info)

    assert pending.read_bytes() == b"previous-wheel"
    assert pid_marker.read_text(encoding="ascii") == "0"


def test_install_warp_leaves_a_newer_untested_owned_warp_in_place(tmp_path, monkeypatch):
    """A newer 1.x copy in this extension's directory is not replaced."""
    from fluidSim_warp_blender51 import warp_installer

    _seed_owned_warp(tmp_path, "1.18.0")
    before = _snapshot(tmp_path)
    _isolate_install(monkeypatch, tmp_path)
    download = mock.Mock()
    monkeypatch.setattr(warp_installer, "_download_wheel", download)

    result = warp_installer.install_warp()

    assert result.success is True
    assert result.already_installed is True
    download.assert_not_called()
    assert _snapshot(tmp_path) == before


def test_install_warp_refuses_a_foreign_warp_without_writing(tmp_path, monkeypatch):
    """Warp loaded from another directory is not modified."""
    from fluidSim_warp_blender51 import warp_installer

    foreign = tmp_path / "other_addon" / "warp" / "__init__.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text('__version__ = "1.16.0"\n', encoding="utf-8")
    owned = tmp_path / "owned"
    _isolate_install(monkeypatch, owned, importable=foreign.resolve())
    download = mock.Mock()
    monkeypatch.setattr(warp_installer, "_download_wheel", download)

    result = warp_installer.install_warp()

    assert result.success is False
    assert "does not own" in result.message
    assert str(foreign.resolve()) in result.message
    download.assert_not_called()
    assert not owned.exists()
    assert foreign.read_text(encoding="utf-8") == '__version__ = "1.16.0"\n'


def test_install_warp_refuses_user_site_packages_without_writing(tmp_path, monkeypatch):
    """An older Warp in Blender's user site-packages is not replaced."""
    from fluidSim_warp_blender51 import warp_installer

    user_init = tmp_path / "site-packages" / "warp" / "__init__.py"
    user_init.parent.mkdir(parents=True)
    user_init.write_text('__version__ = "1.16.0"\n', encoding="utf-8")
    owned = tmp_path / "owned"
    monkeypatch.setattr(warp_installer, "_user_site_warp_path", lambda: user_init.resolve())
    monkeypatch.setattr(warp_installer, "_default_install_target", lambda create=False: owned)
    download = mock.Mock()
    monkeypatch.setattr(warp_installer, "_download_wheel", download)

    result = warp_installer.install_warp()

    assert result.success is False
    assert "user site-packages" in result.message
    download.assert_not_called()
    assert not owned.exists()
    assert user_init.read_text(encoding="utf-8") == '__version__ = "1.16.0"\n'


def test_install_warp_replaces_only_an_older_copy_in_the_owned_directory(tmp_path, monkeypatch):
    """An older Warp in this extension's directory is replaced from the locked URL."""
    from fluidSim_warp_blender51 import warp_installer

    _seed_owned_warp(tmp_path, "1.16.0")
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("warp/__init__.py", '__version__ = "1.17.0"\n')
        archive.writestr("warp_lang-1.17.0.dist-info/METADATA", "Name: warp-lang\n")
    wheel_bytes = payload.getvalue()
    info = mock.Mock(
        filename="warp_lang-1.17.0-py3-none-manylinux_2_28_x86_64.whl",
        sha256=hashlib.sha256(wheel_bytes).hexdigest(),
        size=len(wheel_bytes),
        url="https://files.pythonhosted.org/packages/locked/warp.whl",
        platform="linux-x64",
    )

    def fake_download(url, destination, **_kwargs):
        assert url == info.url
        destination.write_bytes(wheel_bytes)

    _isolate_install(monkeypatch, tmp_path)
    monkeypatch.setattr(warp_installer, "get_wheel_info_for_platform", lambda platform=None: info)
    monkeypatch.setattr(warp_installer, "_download_wheel", fake_download)
    monkeypatch.setattr(warp_installer, "_sha256_path", lambda _path: info.sha256)
    monkeypatch.setattr(warp_installer, "_online_access_block_reason", lambda: None)

    result = warp_installer.install_warp()

    assert result.success is True
    assert result.already_installed is False
    assert result.version == "1.17.0"
    assert (tmp_path / "warp_lang-1.17.0.dist-info").is_dir()
    assert not (tmp_path / "warp_lang-1.16.0.dist-info").exists()


def test_install_warp_stops_when_online_access_is_off(tmp_path, monkeypatch):
    """bpy.app.online_access false blocks the download."""
    from fluidSim_warp_blender51 import warp_installer

    bpy = types.ModuleType("bpy")
    bpy.app = SimpleNamespace(online_access=False, online_access_override=False)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    _isolate_install(monkeypatch, tmp_path)
    download = mock.Mock()
    monkeypatch.setattr(warp_installer, "_download_wheel", download)

    result = warp_installer.install_warp()

    assert result.success is False
    assert "Enable online access in System preferences" in result.message
    download.assert_not_called()
    assert _snapshot(tmp_path) == {}


def test_install_warp_stops_when_blender_was_started_offline(tmp_path, monkeypatch):
    """--offline-mode is bpy.app.online_access_override, not a private setting."""
    from fluidSim_warp_blender51 import warp_installer

    bpy = types.ModuleType("bpy")
    bpy.app = SimpleNamespace(online_access=False, online_access_override=True)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    _isolate_install(monkeypatch, tmp_path)
    download = mock.Mock()
    monkeypatch.setattr(warp_installer, "_download_wheel", download)

    result = warp_installer.install_warp()

    assert result.success is False
    assert "Launch Blender without --offline-mode" in result.message
    download.assert_not_called()


def test_install_warp_reports_extract_failure_without_claiming_success(tmp_path, monkeypatch):
    from fluidSim_warp_blender51 import warp_installer

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

    _isolate_install(monkeypatch, tmp_path)
    monkeypatch.setattr(warp_installer, "get_wheel_info_for_platform", lambda platform=None: info)
    monkeypatch.setattr(warp_installer, "_download_wheel", fake_download)
    monkeypatch.setattr(warp_installer, "_sha256_path", lambda _path: info.sha256)
    monkeypatch.setattr(warp_installer, "_online_access_block_reason", lambda: None)
    monkeypatch.setattr(
        warp_installer,
        "_extract_wheel",
        mock.Mock(side_effect=RuntimeError("permission denied")),
    )

    result = warp_installer.install_warp()
    assert result.success is False
    assert "permission denied" in result.message


def test_default_install_target_is_this_addons_extensions_local(tmp_path, monkeypatch):
    """The only install directory is this add-on's own extensions/.local."""
    from fluidSim_warp_blender51 import warp_installer

    captured = {}

    def extension_path_user(package, *, path="", create=False):
        captured["package"] = package
        captured["path"] = path
        captured["create"] = create
        return str(tmp_path / ".local")

    bpy = types.ModuleType("bpy")
    bpy.utils = SimpleNamespace(extension_path_user=extension_path_user)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(warp_installer, "__package__", "bl_ext.user_default.fluidsim_warp")

    assert warp_installer._default_install_target(create=False) == tmp_path / ".local"
    assert not (tmp_path / ".local").exists()
    assert captured == {
        "package": "bl_ext.user_default.fluidsim_warp",
        "path": ".local",
        "create": False,
    }
    assert not hasattr(warp_installer, "_loaded_warp_site_packages")


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


def test_install_warp_fails_for_unsupported_platform(tmp_path, monkeypatch):
    """Verify install_warp returns failure for unsupported platforms."""
    from fluidSim_warp_blender51 import warp_installer

    _isolate_install(monkeypatch, tmp_path)
    monkeypatch.setattr(warp_installer, "_online_access_block_reason", lambda: None)
    monkeypatch.setattr(warp_installer, "_host_platform", lambda: None)

    result = warp_installer.install_warp()
    assert result.success is False
    assert "platform" in result.message.lower()


def _operator(monkeypatch):
    """Import the preferences operator against a stand-in bpy package."""
    bpy = types.ModuleType("bpy")
    bpy.__path__ = []
    props = types.ModuleType("bpy.props")
    props.BoolProperty = lambda **_kwargs: None
    props.EnumProperty = lambda **_kwargs: None
    props.StringProperty = lambda **_kwargs: None
    types_mod = types.ModuleType("bpy.types")
    types_mod.Operator = type("Operator", (), {})
    types_mod.AddonPreferences = type("AddonPreferences", (), {})
    utils = types.ModuleType("bpy.utils")
    bpy.app = SimpleNamespace(online_access=True, online_access_override=False)
    bpy.context = SimpleNamespace()
    bpy.props = props
    bpy.types = types_mod
    bpy.utils = utils
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", props)
    monkeypatch.setitem(sys.modules, "bpy.types", types_mod)
    monkeypatch.setitem(sys.modules, "bpy.utils", utils)
    monkeypatch.delitem(sys.modules, "fluidSim_warp_blender51.preferences", raising=False)
    from fluidSim_warp_blender51 import preferences

    operator = preferences.FLUIDSIM_OT_install_warp.__new__(
        preferences.FLUIDSIM_OT_install_warp
    )
    operator.force_reinstall = False
    operator.reports = []
    operator.report = lambda level, message: operator.reports.append((set(level), message))
    return preferences, operator


def test_install_warp_button_operator_stops_when_online_access_is_off(tmp_path, monkeypatch):
    """The Preferences button calls fluidsim.install_warp, not install_warp() directly."""
    from fluidSim_warp_blender51 import warp_installer

    preferences, operator = _operator(monkeypatch)
    assert preferences.FLUIDSIM_OT_install_warp.bl_idname == "fluidsim.install_warp"
    bpy = sys.modules["bpy"]
    bpy.app.online_access = False
    bpy.app.online_access_override = True
    _isolate_install(monkeypatch, tmp_path)
    download = mock.Mock()
    monkeypatch.setattr(warp_installer, "_download_wheel", download)

    result = operator.invoke(None, None)

    assert result == {"CANCELLED"}
    assert any("Launch Blender without --offline-mode" in message for _level, message in operator.reports)
    download.assert_not_called()
    assert _snapshot(tmp_path) == {}


def test_install_warp_button_operator_second_press_writes_nothing(tmp_path, monkeypatch):
    """Invoking the button operator again reports success and does not write."""
    from fluidSim_warp_blender51 import warp_installer

    _seed_owned_warp(tmp_path, "1.17.0")
    before = _snapshot(tmp_path)
    _preferences, operator = _operator(monkeypatch)
    _isolate_install(monkeypatch, tmp_path)
    download = mock.Mock()
    extract = mock.Mock()
    monkeypatch.setattr(warp_installer, "_download_wheel", download)
    monkeypatch.setattr(warp_installer, "_extract_wheel", extract)

    result = operator.invoke(None, None)

    assert result == {"FINISHED"}
    assert any("already installed" in message.lower() for _level, message in operator.reports)
    download.assert_not_called()
    extract.assert_not_called()
    assert _snapshot(tmp_path) == before


def test_preferences_offer_repair_for_a_failed_owned_certified_runtime(monkeypatch):
    """A failed certified runtime probe exposes an explicit owned-copy repair."""
    preferences, _operator_instance = _operator(monkeypatch)
    monkeypatch.setattr(preferences, "foreign_warp_block_reason", lambda: None)
    monkeypatch.setattr(
        preferences,
        "get_installed_warp_info",
        lambda: (True, "Warp 1.17.0 (certified)"),
    )
    monkeypatch.setattr(preferences, "owned_install_status", lambda: "certified")
    monkeypatch.setattr(
        preferences,
        "probe_runtime",
        lambda: SimpleNamespace(
            last_error="RuntimeError: corrupted runtime",
            module_file="",
            warp_version="1.17.0",
        ),
    )
    monkeypatch.setattr(preferences, "configured_device", lambda: "cuda:0")
    monkeypatch.setattr(
        preferences,
        "runtime_status_for_device",
        lambda _selected, *, probe: (False, probe.last_error),
    )

    operators = []

    class RecordingLayout:
        def box(self):
            return self

        def row(self, **_kwargs):
            return self

        def label(self, **_kwargs):
            return None

        def prop(self, *_args, **_kwargs):
            return None

        def operator(self, identifier, **kwargs):
            properties = SimpleNamespace(force_reinstall=False)
            operators.append((identifier, kwargs, properties))
            return properties

    prefs = preferences.FLUIDSIM_AP_preferences.__new__(preferences.FLUIDSIM_AP_preferences)
    prefs.layout = RecordingLayout()
    prefs.draw(None)

    repair = next(call for call in operators if call[1].get("text") == "Repair Warp")
    assert repair[0] == "fluidsim.install_warp"
    assert repair[2].force_reinstall is True


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


def test_shared_extensions_local_is_not_treated_as_owned(tmp_path, monkeypatch):
    """A Warp installation in the shared extensions/.local directory is NOT owned by FluidSim."""
    from fluidSim_warp_blender51 import warp_installer

    shared_extensions = tmp_path / "extensions"
    shared_local = shared_extensions / ".local"
    _seed_owned_warp(shared_local, "1.17.0")

    # Set up bpy stand-in with user_resource returning shared_extensions
    # but extension_path_user failing or returning None
    bpy = types.ModuleType("bpy")
    bpy.utils = SimpleNamespace(
        user_resource=lambda kind: str(shared_extensions) if kind == "EXTENSIONS" else None,
        extension_path_user=lambda package, path="", create=False: None,
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(warp_installer, "__package__", "bl_ext.user_default.fluidsim_warp")

    # _default_install_target must raise RuntimeError rather than falling back to shared .local
    with pytest.raises(RuntimeError, match="extension directory"):
        warp_installer._default_install_target()

    # _owned_candidates must NOT include shared_local
    assert shared_local not in warp_installer._owned_candidates()
    assert warp_installer.owned_install_status() is None
    assert warp_installer.is_certified_warp_installed() is False


def test_shared_extensions_local_warp_does_not_count_as_fluidsim_installed(tmp_path, monkeypatch):
    """Another add-on's Warp 1.17.0 in shared extensions/.local does not count as installed for FluidSim."""
    from fluidSim_warp_blender51 import warp_installer

    shared_extensions = tmp_path / "extensions"
    shared_local = shared_extensions / ".local"
    _seed_owned_warp(shared_local, "1.17.0")

    fluidsim_own_dir = shared_extensions / ".user" / "user_default" / "fluidsim_warp" / ".local"

    bpy = types.ModuleType("bpy")
    bpy.utils = SimpleNamespace(
        user_resource=lambda kind: str(shared_extensions) if kind == "EXTENSIONS" else None,
        extension_path_user=lambda package, path="", create=False: str(fluidsim_own_dir),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(warp_installer, "__package__", "bl_ext.user_default.fluidsim_warp")

    # FluidSim's own directory is empty, but shared .local has Warp 1.17.0.
    # On the old buggy behavior, _owned_candidates() appended shared_local,
    # making is_certified_warp_installed() return True!
    # Under fixed behavior, it MUST return False and owned_install_status() must be None.
    assert shared_local not in warp_installer._owned_candidates()
    assert warp_installer.owned_install_status() is None
    assert warp_installer.is_certified_warp_installed() is False


def test_shared_extensions_local_target_is_refused_with_clear_error(tmp_path, monkeypatch):
    """If extension_path_user resolves to shared extensions/.local, refuse with clear error."""
    from fluidSim_warp_blender51 import warp_installer

    shared_extensions = tmp_path / "extensions"
    shared_local = shared_extensions / ".local"

    bpy = types.ModuleType("bpy")
    bpy.utils = SimpleNamespace(
        user_resource=lambda kind: str(shared_extensions) if kind == "EXTENSIONS" else None,
        extension_path_user=lambda package, path="", create=False: str(shared_local),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(warp_installer, "__package__", "bl_ext.user_default.fluidsim_warp")

    with pytest.raises(RuntimeError, match="Refusing to write Warp into Blender's shared extensions directory"):
        warp_installer._default_install_target()


def test_warp_loaded_from_shared_extensions_local_is_flagged_as_foreign(tmp_path, monkeypatch):
    """If Warp was imported from shared extensions/.local, foreign_warp_block_reason flags it."""
    from fluidSim_warp_blender51 import warp_installer

    shared_extensions = tmp_path / "extensions"
    shared_local = shared_extensions / ".local"
    _seed_owned_warp(shared_local, "1.17.0")

    fluidsim_own_dir = shared_extensions / ".user" / "user_default" / "fluidsim_warp" / ".local"

    bpy = types.ModuleType("bpy")
    bpy.utils = SimpleNamespace(
        user_resource=lambda kind: str(shared_extensions) if kind == "EXTENSIONS" else None,
        extension_path_user=lambda package, path="", create=False: str(fluidsim_own_dir),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(warp_installer, "__package__", "bl_ext.user_default.fluidsim_warp")
    monkeypatch.setattr(warp_installer, "_user_site_warp_path", lambda: None)

    # Simulate sys.modules["warp"] loaded from shared_local
    fake_warp = types.ModuleType("warp")
    fake_warp.__file__ = str(shared_local / "warp" / "__init__.py")
    monkeypatch.setitem(sys.modules, "warp", fake_warp)

    reason = warp_installer.foreign_warp_block_reason()
    assert reason is not None
    assert "which FluidSim does not own" in reason
    assert str(shared_local) in reason


@pytest.mark.parametrize("authorized,matching,expected_controls", [
    (True, True, True),
    (False, True, False),
    (True, False, False),
])
def test_preferences_external_override_requires_authorized_loaded_path(
    tmp_path, monkeypatch, authorized, matching, expected_controls,
):
    from fluidSim_warp_blender51 import warp_discovery

    preferences, _ = _operator(monkeypatch)
    external = tmp_path / "external"
    (external / "warp").mkdir(parents=True)
    init = external / "warp" / "__init__.py"
    init.write_text("__version__ = '1.17.0'\n", encoding="utf-8")
    monkeypatch.setenv("FLUIDSIM_ALLOW_EXTERNAL_WARP", "1" if authorized else "0")
    # Apply through the real developer entry point, including the authorization check.
    monkeypatch.setattr(sys, "path", list(sys.path))
    warp_discovery.enable_developer_override(str(external))
    loaded = types.ModuleType("warp")
    loaded.__file__ = str(init if matching else tmp_path / "other" / "warp" / "__init__.py")
    monkeypatch.setitem(sys.modules, "warp", loaded)
    monkeypatch.setattr(preferences, "foreign_warp_block_reason",
                        lambda: "Warp loaded from a directory which FluidSim does not own")
    monkeypatch.setattr(preferences, "get_installed_warp_info", lambda: (False, ""))
    monkeypatch.setattr(preferences, "owned_install_status", lambda: None)
    probe = SimpleNamespace(last_error="", module_file=loaded.__file__, warp_version="1.17.0")
    monkeypatch.setattr(preferences, "probe_runtime", lambda: probe)
    monkeypatch.setattr(preferences, "configured_device", lambda: "cuda:0")
    monkeypatch.setattr(preferences, "runtime_status_for_device",
                        lambda _device, *, probe: (True, "CUDA ready"))
    calls = []

    class Layout:
        def box(self):
            return self
        def row(self, **kwargs):
            return self
        def label(self, **kwargs):
            calls.append(("label", kwargs.get("text")))
        def prop(self, owner, name, **kwargs):
            calls.append(("prop", name))
        def operator(self, name, **kwargs):
            calls.append(("operator", name))
            return SimpleNamespace(force_reinstall=False)

    prefs = preferences.FLUIDSIM_AP_preferences.__new__(preferences.FLUIDSIM_AP_preferences)
    prefs.developer_warp_path = str(external)
    prefs.layout = Layout()
    prefs.draw(None)

    assert (("prop", "device") in calls) is expected_controls
    assert (("operator", "fluidsim.refresh_runtime_probe") in calls) is expected_controls
    assert (("label", "Remove the other Warp copy first") in calls) is not expected_controls


def test_refresh_available_runtime_reschedules_deferred_prewarm(monkeypatch):
    from fluidSim_warp_blender51.runtime import prewarm_timer

    preferences, _ = _operator(monkeypatch)
    registered = []
    monkeypatch.setattr(sys.modules["bpy"].app, "timers", SimpleNamespace(
        is_registered=lambda callback: callback in registered,
        register=lambda callback, **kwargs: registered.append(callback),
    ), raising=False)
    probe = SimpleNamespace(available=True, cuda_devices=())
    monkeypatch.setattr(preferences, "probe_runtime", lambda **kwargs: probe)
    monkeypatch.setattr(preferences, "configured_device", lambda: "cuda:0")
    monkeypatch.setattr(preferences, "runtime_status_for_device",
                        lambda _device, *, probe: (True, "ready"))

    preferences._refresh_runtime(None)
    preferences._refresh_runtime(None)

    assert registered == [prewarm_timer._background_prewarm]
