#!/usr/bin/env python3
"""Run the production verification gate for master-touching changes.

This script is intentionally strict: it checks repo hygiene, runs the CUDA/Warp
solver tests through Blender's bundled Python, and proves live viewport playback
advances at least two frames in Blender background mode. It also checks the
canonical demo analysis and saved-demo blend surfaces.
"""

from __future__ import annotations

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


def _env():
    env = os.environ.copy()
    extra = env.get("PYTEST_PATH", DEFAULT_PYTEST_PATH)
    if extra:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = extra if not existing else extra + os.pathsep + existing
    return env


def _run(label, command, env=None):
    print(f"\n== {label} ==")
    print(" ".join(str(x) for x in command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


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


def main():
    blender_exe = os.environ.get("BLENDER_EXE", DEFAULT_BLENDER_EXE)
    blender_py = os.environ.get("BLENDER_PY", DEFAULT_BLENDER_PY)
    env = _env()

    _run("git whitespace", ["git", "diff", "--check"])
    _check_no_tracked_npz()
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
        "tests/blender_cache_export_pipeline.py",
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
        "tests/blender_viewport_2frame_gate.py",
        "tests/blender_background_acceptance.py",
        "tests/blender_attribute_viz_acceptance.py",
    ):
        _run(script, [blender_exe, "--background", "--factory-startup", "--python", script])
    _run("demo analysis", [blender_py, "tools/analyze_demos.py"], env=env)
    _run("demo blend checks", [
        blender_exe,
        "--background",
        "--factory-startup",
        "--python",
        "tools/check_demo_blends.py",
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
    print("\nPRODUCTION-GATE-PASS")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nPRODUCTION-GATE-FAIL: {exc}")
        sys.exit(exc.returncode or 1)
