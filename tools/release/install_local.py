"""Install a built FluidSim extension into a local Blender and prove what loads.

Blender never re-syncs an installed add-on from the repository, so a stale copy
can keep loading for weeks. This installs the release zip, warns about legacy
copies that would shadow or confuse it, and runs a headless probe that prints
the FluidSim version, the Warp build actually imported, and whether the CUDA
runtime is usable. It exits non-zero when the installed build is not ready.

    python tools/release/install_local.py --blender "C:/path/to/blender.exe"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__:
    from .build_extension import _blender_process_environment
else:  # Direct execution: python tools/release/install_local.py.
    from build_extension import _blender_process_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
ADDON_ROOT = REPO_ROOT / "fluidSim_warp_blender51"
PROBE_MARKER = "FLUIDSIM_INSTALL_PROBE="
LEGACY_NAME = "fluidsim_warp_blender51"  # compared case-insensitively

# Runs inside Blender. Prints one JSON line so the host side can parse it.
PROBE_SCRIPT = r"""
import importlib, json, sys
import bpy
report = {"blender": ".".join(map(str, bpy.app.version))}
modules = [a.module for a in bpy.context.preferences.addons
           if a.module.endswith(".fluidsim_warp") or a.module == "fluidsim_warp"]
report["modules"] = modules
if len(modules) == 1:
    root = importlib.import_module(modules[0])
    version = importlib.import_module(modules[0] + ".version")
    runtime = importlib.import_module(modules[0] + ".runtime.warp_runtime")
    report["fluidsim_version"] = version.VERSION_STRING
    report["fluidsim_path"] = root.__file__
    probe = runtime.probe_runtime(force=True)
    report["runtime_available"] = bool(probe.available)
    report["runtime_message"] = probe.message
    warp = sys.modules.get("warp")
    report["warp_version"] = getattr(warp, "__version__", None)
    report["warp_path"] = getattr(warp, "__file__", None)
print("__MARKER__" + json.dumps(report))
""".replace("__MARKER__", PROBE_MARKER)


def release_identity() -> dict:
    """Read version.py without importing the add-on (which needs bpy)."""
    path = ADDON_ROOT / "version.py"
    namespace: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


def default_zip() -> Path:
    identity = release_identity()
    name = f"{identity['EXTENSION_ID']}-{identity['VERSION_STRING']}-windows_x64.zip"
    return ADDON_ROOT / "dist" / name


def legacy_copies(config_root: Path) -> list[Path]:
    """Old add-on folders that predate the extension package."""
    found = []
    for version_dir in sorted(config_root.glob("*")):
        for parent in (
            version_dir / "scripts" / "addons",
            version_dir / "extensions" / "user_default",
        ):
            if parent.is_dir():
                found.extend(p for p in parent.iterdir() if p.name.lower() == LEGACY_NAME)
    return found


def parse_probe(output: str) -> dict | None:
    for line in output.splitlines():
        if line.startswith(PROBE_MARKER):
            return json.loads(line[len(PROBE_MARKER):])
    return None


def check_probe(report: dict | None, expected_version: str) -> list[str]:
    """Return the reasons the installed build is not ready (empty when ready)."""
    if report is None:
        return ["the probe produced no report"]
    if len(report.get("modules", [])) != 1:
        return [f"expected one enabled FluidSim extension, found {report.get('modules')}"]
    problems = []
    if report.get("fluidsim_version") != expected_version:
        problems.append(
            f"loaded FluidSim {report.get('fluidsim_version')}, expected {expected_version}"
        )
    if not report.get("runtime_available"):
        problems.append(
            f"CUDA runtime unavailable: {report.get('runtime_message')} "
            f"(Warp imported from {report.get('warp_path')})"
        )
    return problems


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_blender_process_environment(os.environ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install FluidSim into a local Blender and probe it.")
    parser.add_argument("--blender", required=True, type=Path, help="blender executable")
    parser.add_argument("--zip", type=Path, default=None, help="extension zip (default: dist build)")
    parser.add_argument("--probe-only", action="store_true", help="skip install, only probe")
    args = parser.parse_args(argv)

    expected = release_identity()["VERSION_STRING"]
    if not args.probe_only:
        package = args.zip or default_zip()
        if not package.is_file():
            print(f"error: extension zip not found: {package}\n"
                  "build it first with tools/release/build_extension.py", file=sys.stderr)
            return 2
        result = _run([str(args.blender), "--background", "--command", "extension",
                       "install-file", "-r", "user_default", "-e", str(package)])
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            return result.returncode

    appdata = os.environ.get("APPDATA")
    if appdata:
        for path in legacy_copies(Path(appdata) / "Blender Foundation" / "Blender"):
            print(f"warning: legacy FluidSim copy (remove or archive it): {path}")

    result = _run([str(args.blender), "--background", "--python-expr", PROBE_SCRIPT])
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-2000:], sep="\n", file=sys.stderr)
        print(f"NOT READY: Blender probe exited {result.returncode}", file=sys.stderr)
        return result.returncode
    report = parse_probe(result.stdout)
    if report:
        for key in ("blender", "fluidsim_version", "fluidsim_path", "warp_version",
                    "warp_path", "runtime_available", "runtime_message"):
            print(f"{key}: {report.get(key)}")
    problems = check_probe(report, expected)
    if report is None:
        print(result.stdout[-2000:], result.stderr[-2000:], sep="\n", file=sys.stderr)
    for problem in problems:
        print(f"NOT READY: {problem}", file=sys.stderr)
    if not problems:
        print(f"READY: FluidSim {expected} installed and CUDA runtime available")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
