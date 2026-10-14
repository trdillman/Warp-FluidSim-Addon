"""Run the former hosted release checks on an owned local machine.

Use one invocation per target OS. The report binds the result to a Git commit,
Python environment, Blender executable, and package inspection report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import runpy
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
ADDON = REPOSITORY / "fluidSim_warp_blender51"
REQUIREMENTS = ADDON / "requirements-dev.txt"
VALIDATORS = ("phase4", "phase5", "custom_effects")


def pinned_requirements() -> dict[str, str]:
    pins = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)", line)
            if match is None:
                raise RuntimeError(f"Unpinned release requirement: {line}")
            pins[match.group(1)] = match.group(2)
    return pins


def check_environment() -> dict[str, str]:
    pins = pinned_requirements()
    for name, expected in pins.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"{name} is {actual}; release gate requires {expected}")
    return pins


def installed_warp_root() -> Path:
    import warp

    return Path(warp.__file__).resolve().parent


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_platform() -> str:
    system = platform.system()
    if system == "Windows":
        return "windows-x64"
    if system == "Linux":
        return "linux-x64"
    raise RuntimeError(f"No certified release smoke platform for {system}")


def check_blender_version(blender: Path) -> str:
    certified = runpy.run_path(str(ADDON / "version.py"))["CERTIFIED_BLENDER_VERSION"]
    expected = ".".join(str(part) for part in certified)
    first_line = subprocess.check_output([str(blender), "--version"], text=True).splitlines()[0]
    if first_line != f"Blender {expected}":
        raise RuntimeError(f"Expected Blender {expected}, got {first_line}")
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--blender-sha256")
    args = parser.parse_args(argv)
    if not args.checks_only and (args.blender is None or args.blender_sha256 is None):
        parser.error("package gate requires --blender and an approved --blender-sha256")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "local-gate-report.json"
    report_path.unlink(missing_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPOSITORY, text=True
        ).strip()
        if dirty:
            raise RuntimeError("Release gate requires a clean checkout")
        pins = check_environment()
        commands = [
            ([sys.executable, "-m", "ruff", "check", "."], ADDON),
            ([sys.executable, "tools/release/build_extension.py", "--verify-only"], REPOSITORY),
            ([sys.executable, "-m", "compileall", "-q", "fluidSim_warp_blender51", "tools", "tests"], REPOSITORY),
            ([sys.executable, "-m", "pytest", "-q"], REPOSITORY),
        ]
        warp_root = installed_warp_root()
        for name in VALIDATORS:
            commands.append(([
                sys.executable, f"tools/release/validate_{name}_warp.py",
                "--device", "cpu", "--expected-warp-root", str(warp_root),
                "--report-json", str(output / f"{name}-warp-cpu-validation.json"),
            ], REPOSITORY))
        for command, cwd in commands:
            run(command, cwd=cwd, env=env)

        report: dict[str, object] = {
            "commit": commit,
            "python": sys.executable,
            "requirements": pins,
            "cpu_checks_passed": True,
            "package_smoke_passed": False,
        }
        if not args.checks_only:
            blender = args.blender.resolve()
            expected = args.blender_sha256.lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise RuntimeError("Blender SHA-256 must be 64 hexadecimal characters")
            actual = sha256(blender)
            if actual != expected:
                raise RuntimeError(f"Blender SHA-256 mismatch: {actual}")
            blender_version = check_blender_version(blender)
            smoke = host_platform()
            build_report = output / "package-report.json"
            run([
                sys.executable, "tools/release/build_extension.py",
                "--blender", str(blender), "--output", str(output / "packages"),
                "--skip-python-checks", "--skip-lint", "--smoke-platform", smoke,
                "--json-report", str(build_report),
            ], cwd=REPOSITORY, env=env)
            package_result = json.loads(build_report.read_text(encoding="utf-8"))
            if package_result.get("release_ready") is not True or package_result.get("smoke_platform") != smoke:
                raise RuntimeError("Package build or installed smoke did not pass")
            report.update({
                "blender": str(blender), "blender_sha256": actual,
                "blender_version": blender_version,
                "smoke_platform": smoke, "package_report": str(build_report),
                "package_smoke_passed": True,
            })
        final_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip()
        final_dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPOSITORY, text=True
        ).strip()
        if final_commit != commit or final_dirty:
            raise RuntimeError("Checkout changed during release checks; no receipt issued")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"LOCAL-GATE-PASS: {report_path}")
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"LOCAL-GATE-FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
