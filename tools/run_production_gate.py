#!/usr/bin/env python3
"""Run the production verification gate for master-touching changes.

This script is intentionally strict: it checks repo hygiene, runs the CUDA/Warp
solver tests through Blender's bundled Python, and proves live viewport playback
advances at least two frames in Blender background mode. It also checks the
canonical demo analysis and saved-demo blend surfaces.

Control surface
---------------
- Default: headless/background validation only, ending in HEADLESS-GATE-PASS.
- --include-gui or FLUIDSIM_FULL_GUI_ACCEPTANCE=1: also runs the interactive
  GUI acceptance lane, ending in FULL-PRODUCTION-GATE-PASS.
- BLENDER_EXE, BLENDER_PY, and PYTEST_PATH override runtime resolution.

Debug surface
-------------
The gate always prints timestamp, git head, dirty state, tracked ignored-file
state, resolved Blender paths, and whether GUI acceptance is included.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt":
    DEFAULT_BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
    DEFAULT_BLENDER_PY = r"C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe"
else:
    DEFAULT_BLENDER_EXE = "/mnt/c/Program Files/Blender Foundation/Blender 5.1/blender.exe"
    DEFAULT_BLENDER_PY = "/mnt/c/Program Files/Blender Foundation/Blender 5.1/5.1/python/bin/python.exe"
DEFAULT_PYTEST_PATH = ""
BLENDER_FAILURE_MARKERS = (
    "Traceback (most recent call last):",
    "CACHE-EXPORT-PIPELINE-FAIL",
    "VIEWPORT-GATE-FAIL",
    "BACKGROUND-ACCEPTANCE-FAIL",
    "ATTRIBUTE-VIZ-ACCEPTANCE-FAIL",
    "GUI-ACCEPTANCE-FAIL",
    "DEMO-BLEND-CHECK-FAIL",
)
CANONICAL_DEMO_SCENES = (
    "dam_break",
    "crown_splash",
    "slosh_tank",
    "tidal_lock",
    "material_response",
    "maelstrom_basin",
)


def _env():
    env = os.environ.copy()
    extra = env.get("PYTEST_PATH", DEFAULT_PYTEST_PATH)
    if extra:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = extra if not existing else extra + os.pathsep + existing
    return env


def _run(label, command, env=None, reject_output=False):
    print(f"\n== {label} ==")
    print(" ".join(str(x) for x in command))
    if not reject_output:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    for marker in BLENDER_FAILURE_MARKERS:
        if marker in result.stdout:
            raise subprocess.CalledProcessError(1, command)


def _check_no_tracked_npz():
    out = subprocess.check_output(["git", "ls-files", "*.npz"], cwd=ROOT, text=True)
    files = [line for line in out.splitlines() if line.strip()]
    if files:
        print("Tracked .npz artifacts are not allowed:")
        for path in files[:50]:
            print(f"  {path}")
        if len(files) > 50:
            print(f"  ... {len(files) - 50} more")
        raise SystemExit(1)


def _check_no_tracked_ignored_files():
    out = subprocess.check_output(
        ["git", "ls-files", "-ci", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    files = [line for line in out.splitlines() if line.strip()]
    if files:
        print("Tracked files matching ignore rules are not allowed:")
        for path in files[:50]:
            print(f"  {path}")
        if len(files) > 50:
            print(f"  ... {len(files) - 50} more")
        raise SystemExit(1)


def _git_text(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as exc:  # noqa: BLE001 - metadata should not hide gate checks
        return f"unavailable: {exc}"


def _print_gate_metadata(blender_exe, blender_py, include_gui):
    status = _git_text("status", "--porcelain")
    ignored = _git_text("ls-files", "-ci", "--exclude-standard")
    print("\n== gate metadata ==")
    print(f"timestamp_utc: {_dt.datetime.now(_dt.UTC).isoformat()}")
    print(f"git_head: {_git_text('rev-parse', '--short=12', 'HEAD')}")
    print(f"git_dirty: {bool(status)}")
    print(f"tracked_ignored_files: {bool(ignored)}")
    print(f"blender_exe: {blender_exe}")
    print(f"blender_py: {blender_py}")
    print(f"gui_acceptance: {'included' if include_gui else 'not-run-headless-gate'}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run FluidSim verification gates with explicit GUI semantics.",
        epilog=(
            "Default status is HEADLESS-GATE-PASS. Use --include-gui for "
            "FULL-PRODUCTION-GATE-PASS. Metadata printed at start includes "
            "timestamp, git head, dirty state, tracked ignored-file state, "
            "resolved Blender paths, and GUI inclusion."
        ),
    )
    parser.add_argument(
        "--include-gui",
        action="store_true",
        help="also run tests/blender_gui_acceptance.py and report full production proof",
    )
    args = parser.parse_args(argv)
    blender_exe = os.environ.get("BLENDER_EXE", DEFAULT_BLENDER_EXE)
    blender_py = os.environ.get("BLENDER_PY", DEFAULT_BLENDER_PY)
    include_gui = args.include_gui or os.environ.get("FLUIDSIM_FULL_GUI_ACCEPTANCE") == "1"
    env = _env()

    _print_gate_metadata(blender_exe, blender_py, include_gui)
    _run("git whitespace", ["git", "diff", "--check"])
    _check_no_tracked_npz()
    _check_no_tracked_ignored_files()
    _run("python syntax", [
        blender_py, "-m", "py_compile",
        "tools/profile_gpu_solver.py",
        "tools/profile_user_path.py",
        "tools/analyze_demos.py",
        "tools/build_demo_scenes.py",
        "tools/build_maelstrom_demo.py",
        "tools/check_demo_blends.py",
        "tools/open_maelstrom_basin_playable.py",
            "tools/run_production_gate.py",
            "tools/validate_gui_playback_report.py",
        "tests/blender_cache_export_pipeline.py",
        "tests/blender_memory_safety_gate.py",
        "tests/blender_gui_acceptance.py",
        "tests/blender_viewport_2frame_gate.py",
        "tests/test_demo_analysis_maelstrom.py",
    ])
    _run("pytest CUDA regressions", [
        blender_py, "-m", "pytest",
        "tests/test_pressure_projection_regressions.py",
        "tests/test_cache_io.py",
        "tests/test_io_path_policy.py",
        "tests/test_demo_analysis_maelstrom.py",
        "-q",
    ], env=env)
    for script in (
        "tests/smoke_core.py",
        "tests/smoke_solver.py",
        "tests/test_solver_validation.py",
        "tests/test_validation.py",
        "tests/test_kernel_coverage.py",
    ):
        _run(script, [blender_py, script])
    _run("GPU profiler smoke", [
        blender_py, "tools/profile_gpu_solver.py",
        "--device", "cuda:0",
        "--frames", "3",
        "--warmup", "1",
        "--emit-frames", "2",
        "--voxel-size", "0.1",
        "--max-particles", "100000",
        "--max-whitewater", "50000",
        "--no-surface",
    ])
    for script in (
        "tests/blender_cache_export_pipeline.py",
        "tests/blender_memory_safety_gate.py",
        "tests/blender_handler_persistence_gate.py",
        "tests/blender_viewport_2frame_gate.py",
        "tests/blender_background_acceptance.py",
        "tests/blender_attribute_viz_acceptance.py",
    ):
        _run(script, [blender_exe, "--background", "--factory-startup", "--python", script],
             reject_output=True)
    for scene_name in CANONICAL_DEMO_SCENES:
        _run(f"demo cache build {scene_name}", [
            blender_exe,
            "--background",
            "--factory-startup",
            "--python",
            "tools/build_demo_scenes.py",
            "--",
            scene_name,
        ], reject_output=True)
    _run("demo analysis", [blender_py, "tools/analyze_demos.py"], env=env)
    _run("demo blend checks", [
        blender_exe,
        "--background",
        "--factory-startup",
        "--python",
        "tools/check_demo_blends.py",
    ], reject_output=True)
    if include_gui:
        _run("interactive GUI acceptance", [
            blender_exe,
            "--factory-startup",
            "--python",
            "tests/blender_gui_acceptance.py",
        ], reject_output=True)
        playback_report = os.path.abspath("artifacts/gui_playback_probe.json")
        playback_env = env.copy()
        playback_env.update({
            "FLUIDSIM_PLAYBACK_BLEND": os.path.abspath("demos/material_response.blend"),
            "FLUIDSIM_PLAYBACK_REPORT": playback_report,
        })
        _run("saved-demo GUI timeline playback", [
            blender_exe,
            "--factory-startup",
            "--python",
            "tests/blender_gui_playback_probe.py",
        ], env=playback_env, reject_output=True)
        _run("saved-demo GUI playback report", [
            blender_py,
            "tools/validate_gui_playback_report.py",
            playback_report,
        ])
    if os.environ.get("FLUIDSIM_USER_PATH_PROFILE") == "1":
        _run("interactive user-path profile", [
            blender_exe,
            "--background",
            "--factory-startup",
            "--python",
            "tools/profile_user_path.py",
            "--",
            "--frames",
            "6",
            "--voxel-size",
            "0.08",
            "--max-particles",
            "200000",
            "--collider-subdiv",
            "2",
            "--max-mean-view-ms",
            "120",
            "--max-mean-total-ms",
            "350",
            "--json-out",
            "artifacts/user_path_profile.json",
        ])
    if include_gui:
        print("\nFULL-PRODUCTION-GATE-PASS")
    else:
        print("\nHEADLESS-GATE-PASS")
        print("GUI acceptance not run; use --include-gui for full production proof.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(f"\nPRODUCTION-GATE-FAIL: {exc}")
        sys.exit(exc.returncode or 1)
