"""Install the certified NVIDIA Warp wheel from PyPI into Blender's Python."""

from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

from .version import CERTIFIED_WARP_DISTRIBUTION, CERTIFIED_WARP_VERSION

_CHUNK_SIZE = 1024 * 1024
_LOCK_PATH = Path(__file__).resolve().parents[1] / "tools" / "release" / "wheels.lock.json"


@dataclass(frozen=True)
class WheelInfo:
    """Metadata for a locked Warp wheel."""

    filename: str
    sha256: str
    size: int
    url: str
    platform: str


def _host_platform() -> str | None:
    machine = _platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        return None
    return {"win32": "windows-x64", "linux": "linux-x64"}.get(sys.platform)


def _load_wheel_lock() -> dict:
    if not _LOCK_PATH.is_file():
        raise RuntimeError(f"Wheel lock file not found: {_LOCK_PATH}")
    return json.loads(_LOCK_PATH.read_text(encoding="utf-8"))


def get_wheel_info_for_platform(platform: str | None = None) -> WheelInfo | None:
    """Return the WheelInfo for the given platform from the lock file."""
    if platform is None:
        platform = _host_platform()
    if platform is None:
        return None
    lock = _load_wheel_lock()
    for filename, meta in lock.get("files", {}).items():
        if meta.get("platform") == platform:
            return WheelInfo(
                filename=filename,
                sha256=meta["sha256"],
                size=meta["size"],
                url=meta["url"],
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
    """Download a wheel from URL and verify its checksum."""
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
                f"Downloaded wheel checksum mismatch: expected {expected_sha256}, got {actual}"
            )
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _get_installed_warp_version() -> str | None:
    """Return the installed warp version, or None if not installed."""
    try:
        import warp as wp
        version = getattr(wp, "__version__", None)
        if version:
            return str(version)
        config = getattr(wp, "config", None)
        return str(getattr(config, "version", None)) if config else None
    except ImportError:
        return None


def is_warp_installed() -> bool:
    """Return True if Warp is importable."""
    return _get_installed_warp_version() is not None


def is_certified_warp_installed() -> bool:
    """Return True if the certified Warp version is installed."""
    version = _get_installed_warp_version()
    if version is None:
        return False
    release = version.split("+", 1)[0].strip()
    return release == CERTIFIED_WARP_VERSION


def get_installed_warp_info() -> tuple[bool, str]:
    """Return (is_installed, version_or_message)."""
    version = _get_installed_warp_version()
    if version is None:
        return False, "Warp is not installed"
    if version.split("+", 1)[0].strip() == CERTIFIED_WARP_VERSION:
        return True, f"Warp {version} (certified)"
    return True, f"Warp {version} (expected {CERTIFIED_WARP_VERSION})"


@dataclass
class InstallResult:
    """Result of a Warp installation attempt."""

    success: bool
    message: str
    version: str = ""
    already_installed: bool = False


def install_warp(
    *,
    force_reinstall: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> InstallResult:
    """Install the certified Warp wheel into Blender's Python environment.

    Args:
        force_reinstall: If True, reinstall even if already installed.
        progress_callback: Optional callback for status messages.

    Returns:
        InstallResult with success status and message.
    """
    def report(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    wheel_info = get_wheel_info_for_platform()
    if wheel_info is None:
        return InstallResult(
            success=False,
            message=f"No certified Warp wheel available for this platform ({_host_platform() or 'unknown'})",
        )

    if not force_reinstall and is_certified_warp_installed():
        version = _get_installed_warp_version() or CERTIFIED_WARP_VERSION
        return InstallResult(
            success=True,
            message=f"Warp {version} is already installed",
            version=version,
            already_installed=True,
        )

    report(f"Downloading Warp {CERTIFIED_WARP_VERSION} ({wheel_info.size / 1024 / 1024:.1f} MB)...")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_path = Path(tmpdir) / wheel_info.filename

        try:
            _download_wheel(
                wheel_info.url,
                wheel_path,
                expected_sha256=wheel_info.sha256,
                expected_size=wheel_info.size,
            )
        except Exception as e:
            return InstallResult(
                success=False,
                message=f"Download failed: {e}",
            )

        report("Installing Warp into Blender's Python...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--force-reinstall" if force_reinstall else "--upgrade",
                    str(wheel_path),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                return InstallResult(
                    success=False,
                    message=f"pip install failed: {result.stderr.strip() or result.stdout.strip()}",
                )
        except subprocess.TimeoutExpired:
            return InstallResult(
                success=False,
                message="Installation timed out",
            )
        except Exception as e:
            return InstallResult(
                success=False,
                message=f"Installation failed: {e}",
            )

    # Clear any cached import state
    for key in list(sys.modules.keys()):
        if key == "warp" or key.startswith("warp."):
            del sys.modules[key]

    # Verify installation
    version = _get_installed_warp_version()
    if version is None:
        return InstallResult(
            success=False,
            message="Installation completed but Warp cannot be imported",
        )

    return InstallResult(
        success=True,
        message=f"Successfully installed Warp {version}",
        version=version,
    )
