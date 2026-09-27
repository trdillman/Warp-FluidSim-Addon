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
- Warp comes from PYTHONPATH. Every lane runs with PYTHONNOUSERSITE=1 and
  Blender lanes pass --python-use-system-env (Blender 5.2 ignores PYTHONPATH
  and the user site otherwise), so a stray user-site Warp can never shadow the
  certified one. The gate refuses to start unless Blender's Python imports
  exactly version.py's CERTIFIED_WARP_VERSION.

Debug surface
-------------
The gate always prints timestamp, git head, dirty state, tracked ignored-file
state, resolved Blender paths, and whether GUI acceptance is included.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt":
    DEFAULT_BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
    DEFAULT_BLENDER_PY = r"C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe"
else:
    DEFAULT_BLENDER_EXE = "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
    DEFAULT_BLENDER_PY = "/mnt/c/Program Files/Blender Foundation/Blender 5.2/5.2/python/bin/python.exe"
DEFAULT_PYTEST_PATH = ""
BLENDER_FAILURE_MARKERS = (
    "Traceback (most recent call last):",
    "CACHE-EXPORT-PIPELINE-FAIL",
    "MEMORY-SAFETY-GATE-FAIL",
    "HANDLER-PERSISTENCE-GATE-FAIL",
    "VIEWPORT-2FRAME-GATE-FAIL",
    "BACKGROUND-ACCEPTANCE-FAIL",
    "ATTRIBUTE-VIZ-ACCEPTANCE-FAIL",
    "MESH-SOURCE-ACCEPTANCE-FAIL",
    "FORCE-REGION-ACCEPTANCE-FAIL",
    "PHASE4-LIVE-INTERACTION-FAIL",
    "SIMULATION-ADMISSION-ACCEPTANCE-FAIL",
    "GUI-ACCEPTANCE-FAIL",
    "GUI-PLAYBACK-PROBE-FAIL",
    "DEMO-BLEND-CHECK-FAIL",
)
CANONICAL_DEMO_SCENES = (
    "dam_break",
    "crown_splash",
)
GATE_MANIFEST = ROOT / "artifacts" / "production_gate_manifest.json"
RELEASE_WARP_VALIDATORS = ("phase4", "phase5", "custom_effects")


def _warp_root(python, env=None):
    """Return the directory the given interpreter imports Warp from."""
    return subprocess.run(
        [python, "-c",
         "import pathlib, warp; print(pathlib.Path(warp.__file__).parents[1])"],
        cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()


def _warp_version(python, env=None):
    """Return the Warp version the given interpreter imports."""
    return subprocess.run(
        [python, "-c", "import warp; print(warp.__version__)"],
        cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip()


def _check_certified_warp(python, env):
    """Refuse to gate against anything but the certified Warp build."""
    expected = _version_constant("CERTIFIED_WARP_VERSION")
    found = _warp_version(python, env)
    root = _warp_root(python, env)
    print(f"warp: {found} from {root}")
    if found != expected:
        print(f"Gate requires certified Warp {expected}; {python} imports {found} "
              f"from {root}. Put the certified Warp on PYTHONPATH.")
        raise SystemExit(1)
    return root


def _env():
    env = os.environ.copy()
    extra = env.get("PYTEST_PATH", DEFAULT_PYTEST_PATH)
    if extra:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = extra if not existing else extra + os.pathsep + existing
    # A user-site Warp must never shadow the certified one on PYTHONPATH.
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _blender(blender_exe, *args):
    """Blender command line that honours the gate's PYTHONPATH and PYTHON* env."""
    return [blender_exe, "--python-use-system-env", *args]


def _run(label, command, env=None, reject_output=False, expect_marker=None):
    """Run a gate lane, optionally requiring positive proof in stdout.

    Blender can return process exit code 0 when a Python script did not finish
    or raised inside a UI callback. A zero exit and absence of known failure
    text is therefore insufficient: every Blender lane must emit its explicit
    success marker.
    """
    print(f"\n== {label} ==")
    print(" ".join(str(x) for x in command))
    capture = reject_output or expect_marker is not None
    if not capture:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
    )
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    for marker in BLENDER_FAILURE_MARKERS:
        if marker in output:
            raise subprocess.CalledProcessError(1, command)
    if expect_marker is not None and expect_marker not in output:
        print(
            f"Missing required success marker {expect_marker!r} from {label}")
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


def _clear_gate_manifest():
    try:
        GATE_MANIFEST.unlink()
    except FileNotFoundError:
        pass


def _version_constant(name):
    namespace = {}
    source = ROOT / "fluidSim_warp_blender51" / "version.py"
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    return namespace[name]


def _fluidsim_version():
    return _version_constant("VERSION_STRING")


def _write_gate_manifest(blender_exe, warp_version=None):
    artifact = ROOT / "artifacts" / "demo_analysis" / "analysis_report.json"
    if not artifact.is_file():
        raise RuntimeError(f"missing generated demo artifact: {artifact}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "gate_run_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "gate_verdict": "FULL-PRODUCTION-GATE-PASS",
        "git_head": _git_text("rev-parse", "HEAD"),
        "fluidsim_version": _fluidsim_version(),
        "warp_version": warp_version,
        "blender_executable": str(blender_exe),
        "include_gui": True,
        "demo_artifact": "artifacts/demo_analysis/analysis_report.json",
        "demo_artifact_sha256": digest,
    }
    GATE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    GATE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")


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

    _clear_gate_manifest()
    _print_gate_metadata(blender_exe, blender_py, include_gui)
    _run("git whitespace", ["git", "diff", "--check"])
    _check_no_tracked_npz()
    _check_no_tracked_ignored_files()
    warp_root = _check_certified_warp(blender_py, env)
    _run("python syntax", [
        blender_py, "-m", "py_compile",
        "tools/profile_gpu_solver.py",
        "tools/profile_user_path.py",
        "tools/analyze_demos.py",
        "tools/build_demo_scenes.py",
        "tools/check_demo_blends.py",
        "tools/run_production_gate.py",
        "tools/validate_gui_playback_report.py",
        "tests/blender_cache_export_pipeline.py",
        "tests/blender_memory_safety_gate.py",
        "tests/blender_mesh_source_acceptance.py",
        "tests/blender_force_region_acceptance.py",
        "tests/blender_phase3_source_ui_acceptance.py",
        "tests/blender_capacity_planner_acceptance.py",
        "tests/blender_simulation_admission_acceptance.py",
        "tests/blender_closed_domain_stability_acceptance.py",
        "tests/blender_custom_effects_acceptance.py",
        "tests/blender_phase3_profile_safe_review.py",
        "tests/blender_phase4_live_interaction_acceptance.py",
        "tests/blender_gui_acceptance.py",
        "tests/blender_viewport_2frame_gate.py",
        "tests/blender_pre_bridge_file_acceptance.py",
    ], env=env)
    _run("pytest CUDA regressions", [
        blender_py, "-m", "pytest",
        "tests/test_pressure_projection_regressions.py",
        "tests/test_cache_io.py",
        "tests/test_io_path_policy.py",
        "-q",
    ], env=env)
    for script in (
        "tests/smoke_core.py",
        "tests/smoke_solver.py",
        "tests/test_solver_validation.py",
        "tests/test_validation.py",
        "tests/test_kernel_coverage.py",
    ):
        _run(script, [blender_py, script], env=env)
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
    ], env=env)
    # Hosted CI runs these on Warp CPU; the release gate is where they run on CUDA.
    for check in RELEASE_WARP_VALIDATORS:
        _run(f"{check} kernels on CUDA", [
            blender_py, f"tools/release/validate_{check}_warp.py",
            "--expected-warp-root", warp_root,
            "--device", "cuda:0",
            "--report-json",
            os.path.abspath(f"artifacts/release_validation/{check}-warp-cuda-validation.json"),
        ], env=env)
    for script, success_marker in (
        ("tests/blender_cache_export_pipeline.py", "CACHE-EXPORT-PIPELINE-PASS"),
        ("tests/blender_memory_safety_gate.py", "MEMORY-SAFETY-GATE-PASS"),
        ("tests/blender_handler_persistence_gate.py", "HANDLER-PERSISTENCE-GATE-PASS"),
        ("tests/blender_viewport_2frame_gate.py", "VIEWPORT-2FRAME-GATE-PASS"),
        ("tests/blender_background_acceptance.py", "BACKGROUND-ACCEPTANCE-PASS"),
        ("tests/blender_pre_bridge_file_acceptance.py", "PRE-BRIDGE-FILE-ACCEPTANCE-PASS"),
        ("tests/blender_attribute_viz_acceptance.py", "ATTRIBUTE-VIZ-ACCEPTANCE-PASS"),
        ("tests/blender_mesh_source_acceptance.py", "MESH-SOURCE-ACCEPTANCE-PASS"),
        ("tests/blender_force_region_acceptance.py", "FORCE-REGION-ACCEPTANCE-PASS"),
        ("tests/blender_phase4_live_interaction_acceptance.py", "PHASE4-LIVE-INTERACTION-PASS"),
        ("tests/blender_simulation_admission_acceptance.py", "SIMULATION-ADMISSION-ACCEPTANCE-PASS"),
        ("tests/blender_closed_domain_stability_acceptance.py", "CLOSED-DOMAIN-STABILITY-ACCEPTANCE-PASS"),
        ("tests/blender_custom_effects_acceptance.py", "CUSTOM-EFFECTS-ACCEPTANCE-PASS"),
    ):
        _run(
            script,
            _blender(blender_exe, "--background", "--factory-startup", "--python", script),
            env=env,
            reject_output=True,
            expect_marker=success_marker,
        )
    _run(
        "Phase 3 source UI acceptance",
        _blender(blender_exe, "--background", "--factory-startup", "--python",
                 "tests/blender_phase3_source_ui_acceptance.py", "--", "--json-out",
                 os.path.abspath("artifacts/phase3_source_ui_acceptance.json")),
        env=env, reject_output=True, expect_marker="PHASE3-SOURCE-UI-ACCEPTANCE-PASS")
    _run(
        "source capacity planner acceptance",
        _blender(blender_exe, "--background", "--factory-startup", "--python",
                 "tests/blender_capacity_planner_acceptance.py", "--", "--json-out",
                 os.path.abspath("artifacts/capacity_planner_acceptance.json")),
        env=env, reject_output=True, expect_marker="CAPACITY-PLANNER-ACCEPTANCE-PASS")
    for scene_name in CANONICAL_DEMO_SCENES:
        _run(f"demo cache build {scene_name}", _blender(
            blender_exe,
            "--background",
            "--factory-startup",
            "--python",
            "tools/build_demo_scenes.py",
            "--",
            scene_name,
        ), env=env, reject_output=True, expect_marker="DEMO-BUILD-PASS")
    _run(
        "demo analysis", [blender_py, "tools/analyze_demos.py"], env=env,
        expect_marker="DEMO-ANALYSIS-PASS (18/18)")
    _run("demo blend checks", _blender(
        blender_exe,
        "--background",
        "--factory-startup",
        "--python",
        "tools/check_demo_blends.py",
    ), env=env, reject_output=True, expect_marker="DEMO-BLEND-CHECK-PASS")
    if include_gui:
        _run("interactive GUI acceptance", _blender(
            blender_exe,
            "--factory-startup",
            "--python",
            "tests/blender_gui_acceptance.py",
        ), env=env, reject_output=True, expect_marker="GUI-ACCEPTANCE-PASS")
        playback_report = os.path.abspath("artifacts/gui_playback_probe.json")
        # A stale passing artifact must never validate a probe that exited 0
        # without writing a new report.
        try:
            os.remove(playback_report)
        except FileNotFoundError:
            pass
        playback_env = env.copy()
        playback_env.update({
            "FLUIDSIM_PLAYBACK_BLEND": os.path.abspath("demos/dam_break.blend"),
            "FLUIDSIM_PLAYBACK_REPORT": playback_report,
        })
        _run("saved-demo GUI timeline playback", _blender(
            blender_exe,
            "--factory-startup",
            "--python",
            "tests/blender_gui_playback_probe.py",
        ), env=playback_env, reject_output=True,
           expect_marker="GUI-PLAYBACK-PROBE-PASS")
        _run("saved-demo GUI playback report", [
            blender_py,
            "tools/validate_gui_playback_report.py",
            playback_report,
            "--expected-blend",
            playback_env["FLUIDSIM_PLAYBACK_BLEND"],
        ], env=env, expect_marker="SAVED-DEMO-GUI-PLAYBACK-PASS")
    if os.environ.get("FLUIDSIM_USER_PATH_PROFILE") == "1":
        _run("interactive user-path profile", _blender(
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
        ), env=env)
    if include_gui:
        _write_gate_manifest(
            blender_exe, _version_constant("CERTIFIED_WARP_VERSION"))
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
