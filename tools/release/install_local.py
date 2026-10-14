"""Install a built FluidSim extension into a local Blender and prove what loads.

Blender never re-syncs an installed add-on from the repository, so a stale copy
can keep loading for weeks. This installs the release zip, warns about legacy
copies that would shadow or confuse it, and runs a headless probe that prints
the FluidSim version, the Warp build actually imported, and whether the CUDA
runtime is usable. It exits non-zero when the installed build is not ready.
Warp is downloaded only when ``--install-warp`` is explicitly requested.

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
INSTALL_WARP_MARKER = "FLUIDSIM_WARP_INSTALL="
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

# Runs inside Blender. Online access is enabled only for this explicit install
# request and restored before the process exits. The extension's installer
# owns the download, checksum verification, destination checks, and writes.
INSTALL_WARP_SCRIPT = r"""
import importlib, json
import bpy
report = {"success": False}
modules = [a.module for a in bpy.context.preferences.addons
           if a.module.endswith(".fluidsim_warp") or a.module == "fluidsim_warp"]
if len(modules) != 1:
    report["message"] = "expected one enabled FluidSim extension, found " + repr(modules)
else:
    system = bpy.context.preferences.system
    previous_online_access = system.use_online_access
    try:
        system.use_online_access = True
        installer = importlib.import_module(modules[0] + ".warp_installer")
        result = installer.install_warp()
        report.update(
            success=bool(result.success),
            message=result.message,
            version=result.version,
            already_installed=bool(result.already_installed),
        )
    except Exception as exc:
        report["message"] = f"Warp installation raised {type(exc).__name__}: {exc}"
    finally:
        system.use_online_access = previous_online_access
print("__MARKER__" + json.dumps(report))
""".replace("__MARKER__", INSTALL_WARP_MARKER)


def release_identity() -> dict:
    """Read version.py without importing the add-on (which needs bpy)."""
    path = ADDON_ROOT / "version.py"
    namespace: dict = {}
    exec(  # noqa: S102 - read release constants without importing bpy
        compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace
    )
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


def parse_warp_install(output: str) -> dict | None:
    for line in output.splitlines():
        if line.startswith(INSTALL_WARP_MARKER):
            return json.loads(line[len(INSTALL_WARP_MARKER):])
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
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install FluidSim into a local Blender and probe it.")
    parser.add_argument("--blender", required=True, type=Path, help="blender executable")
    parser.add_argument("--zip", type=Path, default=None, help="extension zip (default: dist build)")
    parser.add_argument("--probe-only", action="store_true", help="skip install, only probe")
    parser.add_argument(
        "--install-warp",
        action="store_true",
        help="explicitly allow downloading certified Warp into the installed extension",
    )
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

    if args.install_warp:
        result = _run([str(args.blender), "--background", "--python-expr", INSTALL_WARP_SCRIPT])
        if result.returncode != 0:
            print(result.stdout[-2000:], result.stderr[-2000:], sep="\n", file=sys.stderr)
            print(f"NOT READY: Warp installer exited {result.returncode}", file=sys.stderr)
            return result.returncode
        install_report = parse_warp_install(result.stdout)
        if not install_report:
            print(result.stdout[-2000:], result.stderr[-2000:], sep="\n", file=sys.stderr)
            print("NOT READY: Warp installer produced no report", file=sys.stderr)
            return 1
        print(f"warp_install: {install_report.get('message')}")
        if not install_report.get("success"):
            print(f"NOT READY: {install_report.get('message')}", file=sys.stderr)
            return 1

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
    if problems and report and not report.get("warp_path") and not args.install_warp:
        print(
            "NEXT STEP: rerun with --install-warp to download the certified Warp "
            "runtime into this extension, then probe again.",
            file=sys.stderr,
        )
    if not problems:
        print(f"READY: FluidSim {expected} installed and CUDA runtime available")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
