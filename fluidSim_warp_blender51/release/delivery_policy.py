"""Reject Blender and NVIDIA Warp runtime packages from FluidSim ZIP deliverables."""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path, PurePosixPath

_MAX_NESTED_ZIP_DEPTH = 4


def _is_forbidden_member(name: str) -> bool:
    path = PurePosixPath(name)
    parts = tuple(part.lower() for part in path.parts)
    filename = path.name.lower()

    if path.suffix.lower() == ".whl":
        return True
    if parts and parts[0] == "wheels":
        return True
    if any(part == "warp" or part.startswith("warp_lang-") for part in parts):
        return True
    if filename.startswith("blender-") and (
        filename.endswith(".zip")
        or filename.endswith(".tar.xz")
        or filename.endswith(".tar.gz")
        or filename.endswith(".tgz")
    ):
        return True
    if filename == "blender" and (not parts or parts[0] != "blender"):
        return True
    return False


def _scan_zip(
    archive: zipfile.ZipFile,
    *,
    prefix: str,
    depth: int,
    violations: list[str],
) -> None:
    for info in archive.infolist():
        if info.is_dir():
            continue
        member = info.filename
        qualified = f"{prefix}!{member}" if prefix else member
        if _is_forbidden_member(member):
            violations.append(f"forbidden Blender/Warp runtime payload: {qualified}")
            continue
        if depth >= _MAX_NESTED_ZIP_DEPTH or not member.lower().endswith(".zip"):
            continue
        payload = archive.read(info)
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as nested:
                _scan_zip(
                    nested,
                    prefix=qualified,
                    depth=depth + 1,
                    violations=violations,
                )
        except zipfile.BadZipFile:
            # A file named .zip can be opaque test/support data; only recurse when valid.
            continue


def verify_delivery_zip(path: Path) -> list[str]:
    """Return policy violations found in a delivery ZIP, including nested ZIPs."""
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"delivery archive does not exist: {path}")
    violations: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            _scan_zip(archive, prefix="", depth=0, violations=violations)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"invalid delivery ZIP: {path}") from exc
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    violations = verify_delivery_zip(args.archive)
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    print(f"delivery policy passed: {args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
