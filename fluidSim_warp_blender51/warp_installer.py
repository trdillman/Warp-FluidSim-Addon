"""Install the certified NVIDIA Warp wheel into this extension's own directory.

The pin is ``warp_wheels.lock.json`` next to this module. It is a copy of
``tools/release/wheels.lock.json`` and a test keeps the two files equal.
The installer downloads only that URL and accepts the file only when its size
and SHA-256 match the lock.

Blender installs bundled extension wheels by extracting them into a
site-packages directory (see the 5.2 manual, "Python Wheels"). That shared
``extensions/.local`` directory belongs to every extension, and Blender's
wheel sync removes packages that no enabled extension declares. This button
does not write there, and it does not write into the directory a Warp module
was imported from.

The only directory it creates is the one Blender assigns to this extension
via ``bpy.utils.extension_path_user`` (see the 5.2 manual, "Add-ons"):
``<extensions>/.user/<repository>/<extension id>/.local``. The wheel
is extracted into that directory the same way Blender extracts a wheel,
without pip. A second press when that owned copy is already Warp 1.17.0
reports success and does not download or write.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform as _platform
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from .runtime.warp_runtime import warp_version_support
from .version import CERTIFIED_WARP_VERSION

_CHUNK_SIZE = 1024 * 1024
_LOCK_PATH = Path(__file__).with_name("warp_wheels.lock.json")
_OWNED_LOCAL_DIR = ".local"


@dataclass(frozen=True)
class WheelInfo:
    """Metadata for one locked Warp wheel."""

    filename: str
    sha256: str
    size: int
    url: str
    platform: str


@dataclass
class InstallResult:
    """Result of a Warp installation attempt."""

    success: bool
    message: str
    version: str = ""
    already_installed: bool = False


def _host_platform() -> str | None:
    machine = _platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        return None
    return {"win32": "windows-x64", "linux": "linux-x64"}.get(sys.platform)


def _load_wheel_lock() -> dict:
    if not _LOCK_PATH.is_file():
        raise RuntimeError(f"Warp wheel lock is missing: {_LOCK_PATH.name}")
    return json.loads(_LOCK_PATH.read_text(encoding="utf-8"))


def get_wheel_info_for_platform(platform: str | None = None) -> WheelInfo | None:
    """Return the locked wheel for ``platform``, or the host platform."""
    if platform is None:
        platform = _host_platform()
    if platform is None:
        return None
    lock = _load_wheel_lock()
    for filename, meta in lock.get("files", {}).items():
        if meta.get("platform") == platform:
            return WheelInfo(
                filename=filename,
                sha256=str(meta["sha256"]),
                size=int(meta["size"]),
                url=str(meta["url"]),
                platform=platform,
            )
    return None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_wheel(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Download one locked wheel and reject it unless the checksum matches."""
    if not url.startswith("https://"):
        raise RuntimeError(f"Wheel URL must use HTTPS: {url}")

    temp = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(url, timeout=300) as response:
            with temp.open("wb") as handle:
                while chunk := response.read(_CHUNK_SIZE):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if progress_callback:
                        progress_callback(size, expected_size)
                handle.flush()
                os.fsync(handle.fileno())

        if size != expected_size:
            raise RuntimeError(
                f"Downloaded wheel size mismatch: expected {expected_size}, got {size}"
            )
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(
                "Downloaded wheel checksum mismatch: "
                f"expected {expected_sha256}, got {actual}"
            )
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _user_site_root() -> Path | None:
    import site

    candidates: list[Path] = []
    getter = getattr(site, "getusersitepackages", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # noqa: BLE001 - a missing user site is not an install
            value = None
        if value:
            candidates.append(Path(value))
    user_site = getattr(site, "USER_SITE", None)
    if user_site:
        candidates.append(Path(user_site))
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _user_site_warp_path() -> Path | None:
    """Return ``warp/__init__.py`` in Blender's user site-packages, if present."""
    root = _user_site_root()
    if root is None:
        return None
    init = root / "warp" / "__init__.py"
    if init.is_file():
        return init.resolve()
    return None


def _blender_extensions_dir() -> Path | None:
    try:
        import bpy
    except ImportError:
        return None
    resource = getattr(getattr(bpy, "utils", None), "user_resource", None)
    if not callable(resource):
        return None
    try:
        value = resource("EXTENSIONS")
    except Exception:  # noqa: BLE001 - preferences must still render
        return None
    if not value:
        return None
    return Path(value)


def _default_install_target(*, create: bool = False) -> Path:
    """Return the directory this add-on owns for Warp installation.

    Blender's extension user site for this add-on
    (``bpy.utils.extension_path_user``), or the extensions/.local path when
    running in an extension environment. It is never the directory an
    imported ``warp`` module came from, and it is never
    ``site.getusersitepackages()``. ``create`` is false on the
    already-installed check so a second press does not write.
    """
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "Install Warp must run inside Blender so it can use this "
            "add-on's own directory."
        ) from exc

    extension_path_user = getattr(getattr(bpy, "utils", None), "extension_path_user", None)
    package = __package__ or ""
    if callable(extension_path_user) and package:
        try:
            value = extension_path_user(package, path=_OWNED_LOCAL_DIR, create=create)
            if value:
                target = Path(value)
                user_site = _user_site_root()
                if user_site is not None and (
                    target.resolve() == user_site.resolve() or _is_relative_to(target, user_site)
                ):
                    raise RuntimeError(
                        "Refusing to write Warp into Blender's user site-packages. "
                        "Remove that copy, then press Install Warp."
                    )
                return target
        except Exception as exc:
            if "does not name an extension" not in str(exc):
                raise

    extensions = _blender_extensions_dir()
    if extensions is not None:
        target = extensions / ".local"
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target

    raise RuntimeError(
        "Could not determine this add-on's extension directory. "
        "Install Warp must run inside Blender with extensions enabled."
    )


def ensure_installed_warp_on_path() -> None:
    """Append this extension's site-packages when that directory already exists.

    This does not create the directory, download anything, or move an already
    imported module. A Warp loaded from anywhere else stays the loaded module.
    """
    if "warp" in sys.modules:
        return
    for target in _owned_candidates():
        if target.is_dir():
            value = str(target)
            if value not in sys.path:
                sys.path.append(value)


def _importable_warp_file() -> Path | None:
    """Return the Warp file Python would import, without executing it."""
    module = sys.modules.get("warp")
    if module is not None:
        file_name = getattr(module, "__file__", None)
        if file_name:
            return Path(file_name).resolve()
    spec = importlib.util.find_spec("warp")
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin == "namespace":
        return None
    return Path(origin).resolve()


def _owned_distribution_version(site_packages: Path) -> str | None:
    """Return the Warp version extracted into ``site_packages``, if complete."""
    if not site_packages.is_dir():
        return None
    if not (site_packages / "warp" / "__init__.py").is_file():
        return None
    versions = []
    for entry in site_packages.glob("warp_lang-*.dist-info"):
        stem = entry.name[: -len(".dist-info")]
        version = stem.split("-", 1)[1] if "-" in stem else ""
        if version:
            versions.append(version.split("+", 1)[0])
    if not versions:
        return None
    return versions[-1]


def _owned_candidates() -> list[Path]:
    candidates: list[Path] = []
    try:
        primary = _default_install_target(create=False)
        candidates.append(primary)
    except RuntimeError:
        pass
    extensions = _blender_extensions_dir()
    if extensions is not None:
        local_dir = extensions / ".local"
        if not any(c.resolve() == local_dir.resolve() for c in candidates):
            candidates.append(local_dir)
    return candidates


def foreign_warp_block_reason() -> str | None:
    """Return why FluidSim will not write, or ``None`` when no foreign Warp is present."""
    user_site = _user_site_warp_path()
    if user_site is not None:
        return (
            "Warp is already installed in Blender's user site-packages "
            f"({user_site.parent}). Remove that copy, then press Install Warp. "
            "FluidSim will not change it."
        )
    owned_list = _owned_candidates()
    if not owned_list:
        try:
            _default_install_target(create=False)
        except RuntimeError as exc:
            return str(exc)
    loaded = _importable_warp_file()
    if loaded is not None and not any(_is_relative_to(loaded, owned) for owned in owned_list):
        return (
            f"Warp is already loaded from {loaded}, which FluidSim does not own. "
            "Remove that copy, then press Install Warp. FluidSim will not change it."
        )
    return None


def owned_install_status() -> str | None:
    """Return ``certified``, ``untested``, ``unsupported``, or ``None``."""
    for owned in _owned_candidates():
        version = _owned_distribution_version(owned)
        if version is not None:
            status, _message = warp_version_support(version)
            return status
    return None


def _online_access_block_reason() -> str | None:
    """Return Blender's online-access refusal, or ``None`` when downloads are allowed.

    This reads ``bpy.app.online_access``. ``--offline-mode`` sets
    ``bpy.app.online_access_override`` and forces that flag off. Outside
    Blender there is no flag to read.
    """
    try:
        import bpy
    except ImportError:
        return None
    app = getattr(bpy, "app", None)
    if app is None or not hasattr(app, "online_access"):
        return None
    if app.online_access:
        return None
    if getattr(app, "online_access_override", False):
        return (
            "Online access required to install Warp. "
            "Launch Blender without --offline-mode"
        )
    return (
        "Online access required to install Warp. "
        "Enable online access in System preferences"
    )


def _clear_owned_warp(site_packages: Path) -> None:
    """Remove a Warp tree this extension previously extracted. Nothing else."""
    if not site_packages.is_dir():
        return
    for child in site_packages.iterdir():
        name = child.name
        if name not in {"warp", "warp.libs"} and not name.startswith("warp_lang-"):
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _extract_wheel(wheel: Path, destination: Path) -> None:
    """Extract one wheel into ``destination``, rejecting members that escape it."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(
                    f"Wheel member escapes the install directory: {info.filename}"
                )
        archive.extractall(destination)


def is_warp_installed() -> bool:
    """Return whether some Warp build can be imported."""
    return _importable_warp_file() is not None or owned_install_status() is not None


def is_certified_warp_installed() -> bool:
    """Return whether this extension's own directory holds Warp 1.17.0."""
    return owned_install_status() == "certified"


def get_installed_warp_info() -> tuple[bool, str]:
    """Return ``(is_installed, version_or_message)`` for the preferences panel."""
    blocked = foreign_warp_block_reason()
    if blocked:
        return True, blocked
    for owned in _owned_candidates():
        version = _owned_distribution_version(owned)
        if version is not None:
            status, _message = warp_version_support(version)
            if status == "certified":
                return True, f"Warp {version} (certified)"
            if status == "untested":
                return True, f"Warp {version} (newer than certified {CERTIFIED_WARP_VERSION})"
            return True, f"Warp {version} (incompatible; certified is {CERTIFIED_WARP_VERSION})"
    return False, "Warp is not installed"


def install_warp(
    *,
    force_reinstall: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> InstallResult:
    """Install the locked Warp wheel into this extension's own directory.

    A certified copy there, and a newer 1.x copy there, is success and does
    not download or write. ``force_reinstall`` does not change that. An older
    copy in that same directory is replaced. Warp loaded from any other path,
    including another add-on or Blender's user site-packages, is left untouched.
    """

    def report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    del force_reinstall  # A certified owned copy is never rewritten.

    blocked = foreign_warp_block_reason()
    if blocked:
        return InstallResult(success=False, message=blocked)

    for owned in _owned_candidates():
        owned_version = _owned_distribution_version(owned)
        if owned_version:
            status, _message = warp_version_support(owned_version)
            if status == "certified":
                return InstallResult(
                    success=True,
                    message=f"Warp {owned_version} is already installed",
                    version=owned_version,
                    already_installed=True,
                )
            if status == "untested":
                return InstallResult(
                    success=True,
                    message=(
                        f"Warp {owned_version} is already installed. It is newer than the "
                        f"certified {CERTIFIED_WARP_VERSION} and has not been tested "
                        "with FluidSim."
                    ),
                    version=owned_version,
                    already_installed=True,
                )

    online = _online_access_block_reason()
    if online:
        return InstallResult(success=False, message=online)

    wheel_info = get_wheel_info_for_platform()
    if wheel_info is None:
        host = _host_platform() or "unknown"
        return InstallResult(
            success=False,
            message=f"No certified Warp wheel is published for this platform ({host})",
        )

    report(
        f"Downloading Warp {CERTIFIED_WARP_VERSION} "
        f"({wheel_info.size / 1024 / 1024:.1f} MB)..."
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_path = Path(tmpdir) / wheel_info.filename
        try:
            _download_wheel(
                wheel_info.url,
                wheel_path,
                expected_sha256=wheel_info.sha256,
                expected_size=wheel_info.size,
            )
        except Exception as exc:  # noqa: BLE001 - the button reports the failure
            return InstallResult(success=False, message=f"Download failed: {exc}")
        if _sha256_path(wheel_path) != wheel_info.sha256:
            return InstallResult(
                success=False,
                message="Refusing to install a wheel that failed checksum verification",
            )

        try:
            destination = _default_install_target(create=True)
        except RuntimeError as exc:
            return InstallResult(success=False, message=str(exc))
        if _user_site_root() is not None and (
            _is_relative_to(destination, _user_site_root())
            or destination.resolve() == _user_site_root().resolve()
        ):
            return InstallResult(
                success=False,
                message=(
                    "Refusing to write Warp into Blender's user site-packages. "
                    "Remove any Warp there, then press Install Warp."
                ),
            )
        loaded = _importable_warp_file()
        if loaded is not None and not any(
            _is_relative_to(loaded, cand) for cand in _owned_candidates()
        ):
            return InstallResult(
                success=False,
                message=(
                    f"Warp is already loaded from {loaded}, which FluidSim does not own. "
                    "Remove that copy, then press Install Warp. FluidSim will not change it."
                ),
            )

        report("Installing Warp...")
        try:
            if owned_version:
                _clear_owned_warp(destination)
            _extract_wheel(wheel_path, destination)
        except Exception as exc:  # noqa: BLE001 - the button reports the failure
            return InstallResult(success=False, message=f"Installation failed: {exc}")

    installed = _owned_distribution_version(destination)
    if installed is None:
        return InstallResult(
            success=False,
            message=(
                "Installation finished, but the certified Warp files are missing "
                "from this extension's directory."
            ),
        )
    ensure_installed_warp_on_path()
    if "warp" in sys.modules:
        return InstallResult(
            success=True,
            message=(
                f"Installed Warp {installed} into this extension's directory. "
                "Restart Blender before running the solver."
            ),
            version=installed,
        )
    return InstallResult(
        success=True,
        message=f"Successfully installed Warp {installed}",
        version=installed,
    )
