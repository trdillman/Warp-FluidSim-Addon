"""Verify a checksum-locked external NVIDIA Warp validation wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as _platform
import shutil
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
LOCK_PATH = Path(__file__).resolve().with_name("wheels.lock.json")
WHEEL_DIR = ROOT / "build" / "external-warp-wheel"
_CHUNK_SIZE = 1024 * 1024


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_existing(path: Path, expected_sha256: str, expected_size: int) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and _sha256_path(path) == expected_sha256
    )


def _write_verified_stream(
    response,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    temp = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with temp.open("wb") as handle:
            while chunk := response.read(_CHUNK_SIZE):
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if size != expected_size:
            raise RuntimeError(
                f"payload size mismatch for {destination.name}: "
                f"expected {expected_size}, got {size}"
            )
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(
                f"payload checksum mismatch for {destination.name}: "
                f"expected {expected_sha256}, got {actual}"
            )
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _copy_verified(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if not source.is_file():
        raise RuntimeError(f"wheel source does not exist: {source}")
    if source.stat().st_size != expected_size:
        raise RuntimeError(
            f"source wheel size mismatch for {source.name}: "
            f"expected {expected_size}, got {source.stat().st_size}"
        )
    actual = _sha256_path(source)
    if actual != expected_sha256:
        raise RuntimeError(
            f"source wheel checksum mismatch for {source.name}: "
            f"expected {expected_sha256}, got {actual}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".part")
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _locked_files(lock: dict) -> dict[str, dict]:
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("wheel lock must contain a non-empty files object")
    result: dict[str, dict] = {}
    for filename, metadata in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise RuntimeError(f"invalid wheel filename in lock: {filename!r}")
        if not isinstance(metadata, dict):
            raise TypeError(f"invalid wheel metadata for {filename}")
        result[filename] = metadata
    return result


def _candidate_from_source(source: Path, filename: str) -> Path:
    return source if source.is_file() else source / filename


def fetch_wheels(
    *,
    lock_path: Path = LOCK_PATH,
    wheel_dir: Path = WHEEL_DIR,
    source: Path | None = None,
    source_dirs: Iterable[Path] = (),
    opener: Callable = urlopen,
    offline: bool = False,
    platform: str | None = None,
) -> list[Path]:
    """Verify existing payloads, copy certified local payloads, or use locked HTTPS URLs."""
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    files = _locked_files(lock)
    if platform is not None:
        files = {name: meta for name, meta in files.items() if meta.get("platform") == platform}
        if not files:
            raise RuntimeError(f"wheel lock has no entry for platform {platform!r}")
    wheel_dir = Path(wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    supplied = ([source] if source is not None else []) + list(source_dirs)
    sources = tuple(Path(path).expanduser().resolve() for path in supplied)
    verified: list[Path] = []

    for filename, expected in sorted(files.items()):
        expected_sha256 = str(expected.get("sha256", ""))
        expected_size = expected.get("size")
        if len(expected_sha256) != 64:
            raise RuntimeError(f"invalid SHA-256 lock for {filename}")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise RuntimeError(f"invalid size lock for {filename}")

        destination = wheel_dir / filename
        if _verify_existing(destination, expected_sha256, expected_size):
            verified.append(destination)
            continue

        for supplied_source in sources:
            candidate = _candidate_from_source(supplied_source, filename)
            if not candidate.is_file():
                continue
            _copy_verified(
                candidate,
                destination,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
            break
        else:
            url = str(expected.get("url", ""))
            if offline or not url:
                raise RuntimeError(
                    f"no verified source is available for locked wheel {filename}; "
                    "provide --source/--source-dir or place the verified file in the external validation directory"
                )
            if not url.startswith("https://"):
                raise RuntimeError(f"wheel URL must use HTTPS: {filename}")
            with opener(url, timeout=300) as response:
                _write_verified_stream(
                    response,
                    destination,
                    expected_sha256=expected_sha256,
                    expected_size=expected_size,
                )

        verified.append(destination)

    return verified


def _environment_sources() -> list[Path]:
    value = os.environ.get("FLUIDSIM_WARP_WHEEL_SOURCE", "")
    return [Path(item) for item in value.split(os.pathsep) if item]


def host_platform() -> str | None:
    machine = _platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        return None
    return {"win32": "windows-x64", "linux": "linux-x64"}.get(sys.platform)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--wheel-dir", type=Path, default=WHEEL_DIR)
    parser.add_argument(
        "--source",
        "--source-dir",
        dest="sources",
        type=Path,
        action="append",
        default=[],
        help="Certified wheel file or directory; may be repeated",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--platform",
        default=host_platform(),
        help="Locked platform to verify (default: this host); 'all' verifies every entry",
    )
    args = parser.parse_args(argv)
    sources = [*args.sources, *_environment_sources()]
    for path in fetch_wheels(
        lock_path=args.lock,
        wheel_dir=args.wheel_dir,
        source_dirs=sources,
        offline=args.offline,
        platform=None if args.platform in (None, "all") else args.platform,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
