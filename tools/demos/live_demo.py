#!/usr/bin/env python3
"""Show any demo live: latest add-on in a temporary Blender profile, playing in the viewport.

One command builds the add-on from this checkout, installs it into a brand new
Blender profile (your normal profile is never touched), opens the demo, turns on
Live Simulation with cache playback off and the Liquid Deck on, and presses Play
in the 3D Viewport. The solver steps as the timeline plays; nothing is baked or
replayed from a cache.

    python tools/demos/live_demo.py --blender "C:/path/to/blender.exe" --demo dam_break
    python tools/demos/live_demo.py --blender ... --demo nets_showcase --build
    python tools/demos/live_demo.py --list

The temporary profile is deleted when Blender closes (``--keep-profile`` keeps it).
The extension needs the certified Warp; it is fetched once (checksum locked) into
the temporary profile, the same way the release smoke test does it.
Needs a CUDA GPU, like every FluidSim run.

Opt-in demos (``tsunami_parthenon``, ``upres_cove``, ``nets_showcase``) are not
tracked ``.blend`` files. Their scene is built first with ``--build`` (or already
exists in ``artifacts/review_demos``). ``nets_showcase`` also needs the Surface
Nets backend, which this script switches on; ``--preset draft`` builds the 2 mm
version, which plays far faster than the 1 mm hero.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BOOT_SCRIPT = Path(__file__).resolve().with_name("live_demo_boot.py")
EXTENSION_ID = "fluidsim_warp"
REPOSITORY_ID = "fluidsim_live"

SHIPPED = (
    "dam_break", "crown_splash", "paddle_stir", "chromatic_vortex",
    "buoyancy_ladder", "water_wheel", "viscosity_race", "deep_plunge",
    "capillary_droplet",
)
OPT_IN = ("tsunami_parthenon", "upres_cove", "nets_showcase")
ALL_DEMOS = SHIPPED + OPT_IN

# Opt-in demos whose builder saves into demos/ rather than artifacts/review_demos.
SAVED_IN_DEMOS = ("tsunami_parthenon",)

# The Surface Nets backend is read from the environment when the solver is
# created and a .blend cannot store it. Keep these in step with
# tools/demos/nets_showcase.py (BACKEND_ENV / BACKEND).
DEMO_ENV = {"nets_showcase": {"FLUIDSIM_SURFACE_BACKEND": "nets"}}


def demo_blend(demo: str, repo: Path = REPO) -> Path:
    """Where a demo's .blend lives: tracked demos/, or the opt-in review folder."""
    if demo in SHIPPED or demo in SAVED_IN_DEMOS:
        return repo / "demos" / f"{demo}.blend"
    if demo in OPT_IN:
        return repo / "artifacts" / "review_demos" / f"{demo}.blend"
    raise ValueError(f"unknown demo {demo!r}; choose one of: {', '.join(ALL_DEMOS)}")


def host_platform_suffix() -> str:
    return "windows_x64" if os.name == "nt" else "linux_x64"


def blender_command(blender: Path, blend: Path, boot: Path) -> list[str]:
    """Normal interactive start, offline. No --factory-startup: it would ignore the
    profile's preferences, which is where the installed extension is switched on."""
    return [str(blender), "--offline-mode", str(blend), "--python", str(boot)]


def build_builder_command(blender: Path, demo: str) -> list[str]:
    return [str(blender), "--background", "--factory-startup", "--offline-mode",
            "--python", str(REPO / "tools" / "demos" / "build_demo_scenes.py"), "--", demo]


def launch_environment(demo: str, profile: Path, config: Path, base=None) -> dict[str, str]:
    """A profile-isolated environment for one demo. Installed add-on only: developer
    import overrides (PYTHONPATH and friends) are dropped so the temp install is what runs."""
    sys.path.insert(0, str(REPO))
    try:
        from tools.release.build_extension import _blender_process_environment
    finally:
        sys.path.pop(0)
    env = _blender_process_environment(
        dict(os.environ if base is None else base), profile_dir=profile)
    env.update(DEMO_ENV.get(demo, {}))
    env["FLUIDSIM_LIVE_DEMO_CONFIG"] = str(config)
    return env


def builder_environment(build_profile: Path, base: dict[str, str]) -> dict[str, str]:
    """Environment for the scene builder. It loads the add-on from source, which only
    finds Warp on PYTHONPATH, so the locked Warp is unpacked there (not into an extension)."""
    sys.path.insert(0, str(REPO))
    try:
        from tools.release import build_extension as be
    finally:
        sys.path.pop(0)
    build_profile.mkdir(parents=True, exist_ok=True)
    env = be._blender_process_environment(
        base, profile_dir=build_profile, allow_source_imports=True)
    warp_site = build_profile / "warp-site"
    be._provision_smoke_warp(warp_site)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(warp_site) + (os.pathsep + existing if existing else "")
    return env


def write_config(path: Path, *, demo: str, report: Path, profile: Path,
                 shading: str, camera_view: bool) -> None:
    path.write_text(json.dumps({
        "demo": demo,
        "report": str(report),
        "profile": str(profile),
        "extension_id": EXTENSION_ID,
        "repository_id": REPOSITORY_ID,
        "shading": shading,
        "camera_view": camera_view,
        "expected_env": DEMO_ENV.get(demo, {}),
    }, indent=2), encoding="utf-8")


def _run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True, cwd=REPO)


def build_package(blender: Path, work: Path) -> Path:
    sys.path.insert(0, str(REPO))
    try:
        from tools.release import build_extension as be
    finally:
        sys.path.pop(0)
    packages = be.build(be.ROOT, blender, work / "dist", run_python_checks=False)
    wanted = f"-{host_platform_suffix()}.zip"
    for package in packages:
        if str(package).endswith(wanted):
            return Path(package)
    raise RuntimeError(f"no {wanted} package among {[str(p) for p in packages]}")


def install_package(blender: Path, package: Path, profile: Path, env: dict[str, str]) -> None:
    """Install the zip into the temp profile's own extension repository and switch it on."""
    repo_dir = profile / "extension-repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO))
    try:
        from tools.release import build_extension as be
    finally:
        sys.path.pop(0)
    be._provision_smoke_warp(
        Path(env["BLENDER_USER_EXTENSIONS"]) / ".user" / REPOSITORY_ID / EXTENSION_ID / ".local")
    prefix = [str(blender), "--background", "--offline-mode"]
    _run([*prefix, "--command", "extension", "repo-add", REPOSITORY_ID,
          "--name", "FluidSim Live Demo", "--directory", str(repo_dir), "--clear-all"], env)
    _run([*prefix, "--command", "extension", "install-file",
          "-r", REPOSITORY_ID, "-e", str(package)], env)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--blender", type=Path, help="blender executable")
    parser.add_argument("--demo", choices=ALL_DEMOS, help="demo to show")
    parser.add_argument("--list", action="store_true", help="list demos and exit")
    parser.add_argument("--build", action="store_true",
                        help="build the opt-in demo's .blend first (or rebuild it)")
    parser.add_argument("--preset", choices=("hero", "draft"),
                        help="nets_showcase only: build the 1 mm hero or the 2 mm draft")
    parser.add_argument("--shading", choices=("MATERIAL", "SOLID", "RENDERED"),
                        default="MATERIAL", help="viewport shading (default MATERIAL)")
    parser.add_argument("--no-camera-view", action="store_true",
                        help="stay in the free viewport instead of looking through the camera")
    parser.add_argument("--keep-profile", action="store_true",
                        help="do not delete the temporary profile when Blender closes")
    parser.add_argument("--report", type=Path,
                        help="where Blender writes its live status JSON (default: in the profile)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="build and install, then stop without opening Blender")
    args = parser.parse_args(argv)
    if args.list:
        return args
    if args.blender is None or args.demo is None:
        parser.error("--blender and --demo are required")
    if args.preset and args.demo != "nets_showcase":
        parser.error("--preset only applies to nets_showcase")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        print("shipped:", " ".join(SHIPPED))
        print("opt-in (build with --build):", " ".join(OPT_IN))
        return 0
    blender = args.blender.resolve()
    blend = demo_blend(args.demo)
    # Short path: Blender's Python on Windows struggles beyond MAX_PATH.
    profile = Path(tempfile.mkdtemp(prefix="fsl-")).resolve()
    try:
        env = launch_environment(args.demo, profile, profile / "config.json")
        if args.build or (args.demo in OPT_IN and not blend.is_file()):
            build_env = dict(os.environ)
            build_env.update(DEMO_ENV.get(args.demo, {}))
            if args.preset:
                build_env["NETS_SHOWCASE_PRESET"] = args.preset
            _run(build_builder_command(blender, args.demo),
                 builder_environment(profile / "build", build_env))
        if not blend.is_file():
            print(f"error: {blend} does not exist. "
                  f"Run again with --build to create it.", file=sys.stderr)
            return 2
        package = build_package(blender, profile)
        install_package(blender, package, profile, env)
        report = args.report or (profile / "live_status.json")
        write_config(profile / "config.json", demo=args.demo, report=report, profile=profile,
                     shading=args.shading, camera_view=not args.no_camera_view)
        print(f"profile: {profile}\nstatus file: {report}", flush=True)
        if args.prepare_only:
            args.keep_profile = True
            return 0
        started = time.time()
        subprocess.run(blender_command(blender, blend, BOOT_SCRIPT), env=env, cwd=REPO,
                       check=False)
        print(f"Blender closed after {time.time() - started:.0f} s", flush=True)
        return 0
    finally:
        if args.keep_profile:
            print(f"kept temporary profile: {profile}")
        else:
            shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
