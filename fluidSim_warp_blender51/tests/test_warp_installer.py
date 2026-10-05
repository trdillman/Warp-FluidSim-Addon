"""Tests for the Warp installer module."""

from __future__ import annotations

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
    _isolate_install(monkeypatch, tmp_path)
    download = mock.Mock()
    extract = mock.Mock()
    monkeypatch.setattr(warp_installer, "_download_wheel", download)
    monkeypatch.setattr(warp_installer, "_extract_wheel", extract)

    result = warp_installer.install_warp(force_reinstall=True)

    assert result.success is True
    assert result.already_installed is True
    assert "already installed" in result.message.lower()
    download.assert_not_called()
    extract.assert_not_called()
    assert _snapshot(tmp_path) == before


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

