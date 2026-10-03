"""Install the certified NVIDIA Warp wheel from PyPI into Blender's Python.

The pin is ``warp_wheels.lock.json`` next to this module. It is a copy of
``tools/release/wheels.lock.json`` and a test keeps the two files equal.
The installer downloads only that URL and accepts the file only when its size
and SHA-256 match the lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from .version import CERTIFIED_WARP_VERSION

_CHUNK_SIZE = 1024 * 1024
_LOCK_PATH = Path(__file__).with_name("warp_wheels.lock.json")


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


def _version_of(module: object) -> str | None:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    config = getattr(module, "config", None)
    configured = getattr(config, "version", None) if config is not None else None
    return str(configured) if configured else None


def _get_installed_warp_version() -> str | None:
    """Return the importable Warp version, or ``None`` when it is absent."""
    try:
        import warp as wp
    except ImportError:
        return None
    return _version_of(wp)


def _loaded_warp_site_packages() -> Path | None:
    """Return the site-packages directory that supplied the loaded Warp."""
    module = sys.modules.get("warp")
    if module is None:
        return None
    file_name = getattr(module, "__file__", None)
    if not file_name:
        return None
    return Path(file_name).resolve().parent.parent


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


def _default_install_target() -> Path:
    """Blender's extension wheel directory, or this interpreter's user site."""
    extensions = _blender_extensions_dir()
    if extensions is not None:
        return extensions / ".local"
    import site

    return Path(site.getusersitepackages())


def ensure_installed_warp_on_path() -> None:
    """Make a previously installed Warp importable in this process.

    This adds Blender's extension site-packages when that directory exists.
    It does not download anything and does not move an already imported module.
    """
    target = _default_install_target()
    if not target.is_dir():
        return
    value = str(target)
    if value in sys.path:
        return
    import site

    site.addsitedir(value)


def is_warp_installed() -> bool:
    """Return whether some Warp build can be imported."""
    return _get_installed_warp_version() is not None


def is_certified_warp_installed() -> bool:
    """Return whether the certified Warp release is the one that imports."""
    version = _get_installed_warp_version()
    if version is None:
        return False
    from .runtime.warp_runtime import parse_warp_version

    return parse_warp_version(version) == parse_warp_version(CERTIFIED_WARP_VERSION)


def get_installed_warp_info() -> tuple[bool, str]:
    """Return ``(is_installed, version_or_message)``."""
    version = _get_installed_warp_version()
    if version is None:
        return False, "Warp is not installed"
    from .runtime.warp_runtime import warp_version_support

    status, _message = warp_version_support(version)
    if status == "certified":
        return True, f"Warp {version} (certified)"
    if status == "untested":
        return True, f"Warp {version} (newer than certified {CERTIFIED_WARP_VERSION})"
    return True, f"Warp {version} (incompatible; certified is {CERTIFIED_WARP_VERSION})"


def _python_executable() -> str:
    """Return a Python that can run pip.

    Inside Blender, ``sys.executable`` is the Blender binary. The bundled
    interpreter lives under ``sys.prefix``.
    """
    executable = Path(sys.executable)
    if executable.name.lower().startswith("python"):
        return str(executable)
    bin_dir = Path(sys.prefix) / "bin"
    if sys.platform == "win32":
        candidates = [bin_dir / "python.exe"]
    else:
        candidates = [
            bin_dir / "python3",
            bin_dir / "python",
            *sorted(bin_dir.glob("python3.*")),
        ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("Could not find the Python interpreter bundled with Blender")


def _pip_install(wheel: Path, destination: Path, *, force: bool) -> str | None:
    """Install one verified wheel. Return an error string, or ``None``."""
    python = _python_executable()
    command = [
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--upgrade",
        "--target",
        str(destination),
        str(wheel),
    ]
    if force:
        command.insert(command.index("--upgrade") + 1, "--force-reinstall")

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, capture_output=True, text=True, timeout=300)

    try:
        result = run(command)
    except subprocess.TimeoutExpired:
        return "Installation timed out"
    except OSError as exc:
        return f"Installation failed: {exc}"
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout or "").strip()
    if "No module named pip" not in detail:
        return f"pip install failed: {detail}"
    try:
        boot = subprocess.run(
            [python, "-m", "ensurepip", "--upgrade"],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"Blender's Python has no pip: {exc}"
    if boot.returncode != 0:
        boot_detail = (boot.stderr or boot.stdout or "").strip()
        return f"Blender's Python has no pip: {boot_detail}"
    try:
        result = run(command)
    except subprocess.TimeoutExpired:
        return "Installation timed out"
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout or "").strip()
    return f"pip install failed: {detail}"


def install_warp(
    *,
    force_reinstall: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> InstallResult:
    """Download the locked Warp wheel and install that exact file.

    A certified install, and a newer 1.x install the solver already accepts,
    is success with no second copy. An older or wrong-major Warp is replaced
    in the directory that provided it. A Warp module already imported in this
    process stays loaded until Blender restarts.
    """

    def report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    from .runtime.warp_runtime import warp_version_support

    ensure_installed_warp_on_path()
    loaded = _get_installed_warp_version()
    if loaded and not force_reinstall:
        status, _message = warp_version_support(loaded)
        if status == "certified":
            return InstallResult(
                success=True,
                message=f"Warp {loaded} is already installed",
                version=loaded,
                already_installed=True,
            )
        if status == "untested":
            return InstallResult(
                success=True,
                message=(
                    f"Warp {loaded} is already installed. It is newer than the "
                    f"certified {CERTIFIED_WARP_VERSION} and has not been tested "
                    "with FluidSim."
                ),
                version=loaded,
                already_installed=True,
            )

    wheel_info = get_wheel_info_for_platform()
    if wheel_info is None:
        host = _host_platform() or "unknown"
        return InstallResult(
            success=False,
            message=f"No certified Warp wheel is published for this platform ({host})",
        )

    replacing = loaded is not None
    destination = _loaded_warp_site_packages() if replacing else None
    if destination is None:
        destination = _default_install_target()
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return InstallResult(
            success=False,
            message=f"Cannot create the Warp install directory {destination}: {exc}",
        )
    if not os.access(destination, os.W_OK):
        return InstallResult(
            success=False,
            message=(
                f"Cannot write Warp into {destination}. Remove that install and "
                "press Install Warp again."
            ),
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

        report("Installing Warp...")
        error = _pip_install(wheel_path, destination, force=replacing or force_reinstall)
        if error:
            return InstallResult(success=False, message=error)

    if loaded and "warp" in sys.modules:
        return InstallResult(
            success=True,
            message=(
                f"Installed Warp {CERTIFIED_WARP_VERSION} over Warp {loaded}. "
                "Restart Blender before running the solver."
            ),
            version=CERTIFIED_WARP_VERSION,
        )

    ensure_installed_warp_on_path()
    installed = _get_installed_warp_version()
    if installed is None:
        return InstallResult(
            success=False,
            message=(
                "Installation finished, but Warp cannot be imported. "
                "Restart Blender and press Install Warp again."
            ),
        )
    return InstallResult(
        success=True,
        message=f"Successfully installed Warp {installed}",
        version=installed,
    )
