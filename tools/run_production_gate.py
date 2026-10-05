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
- Blender lanes run with BLENDER_USER_EXTENSIONS set to
  artifacts/gate_blender_user_extensions, so --factory-startup never syncs
  (deletes) the wheels of the user's installed extensions. A caller-set
  BLENDER_USER_EXTENSIONS or BLENDER_USER_RESOURCES is kept instead.

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
sys.path.insert(0, str(ROOT))
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
# --factory-startup Blender treats every installed extension as disabled, and its
# wheel sync deletes their wheels (Warp included) from the user profile's
# extensions/.local. Blender lanes therefore get this private extensions dir.
GATE_BLENDER_USER_EXTENSIONS = ROOT / "artifacts" / "gate_blender_user_extensions"
# Background Blender acceptance scripts that take no arguments: (script, marker).
BACKGROUND_ACCEPTANCE = (
    ("tests/acceptance/blender_cache_export_pipeline.py", "CACHE-EXPORT-PIPELINE-PASS"),
    ("tests/acceptance/blender_memory_safety_gate.py", "MEMORY-SAFETY-GATE-PASS"),
    ("tests/acceptance/blender_handler_persistence_gate.py", "HANDLER-PERSISTENCE-GATE-PASS"),
    ("tests/acceptance/blender_viewport_2frame_gate.py", "VIEWPORT-2FRAME-GATE-PASS"),
    ("tests/acceptance/blender_background_acceptance.py", "BACKGROUND-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_pre_bridge_file_acceptance.py", "PRE-BRIDGE-FILE-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_attribute_viz_acceptance.py", "ATTRIBUTE-VIZ-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_mesh_source_acceptance.py", "MESH-SOURCE-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_force_region_acceptance.py", "FORCE-REGION-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_phase4_live_interaction_acceptance.py", "PHASE4-LIVE-INTERACTION-PASS"),
    ("tests/acceptance/blender_simulation_admission_acceptance.py", "SIMULATION-ADMISSION-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_closed_domain_stability_acceptance.py", "CLOSED-DOMAIN-STABILITY-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_custom_effects_acceptance.py", "CUSTOM-EFFECTS-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_chromatic_vortex_containment_acceptance.py",
     "CHROMATIC-VORTEX-CONTAINMENT-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_color_mixing_runtime_acceptance.py", "COLOR-MIXING-RUNTIME-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_color_mixing_ui_acceptance.py", "COLOR-MIXING-UI-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_mixed_source_batch_acceptance.py", "MIXED-SOURCE-BATCH-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_quick_setup_idempotence.py", "QUICK-SETUP-IDEMPOTENCE-PASS"),
    ("tests/acceptance/blender_source_sampling_acceptance.py", "SOURCE-SAMPLING-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_tooltip_acceptance.py", "TOOLTIP-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_topology_transaction_acceptance.py", "TOPOLOGY-TRANSACTION-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_whitewater_truth_acceptance.py", "WHITEWATER-TRUTH-ACCEPTANCE-PASS"),
    ("tests/acceptance/blender_animated_cache_playback_acceptance.py",
     "ANIMATED-CACHE-PLAYBACK-ACCEPTANCE-PASS"),
)
# Source acceptance scripts that write a JSON report: (capability, marker).
SOURCE_REPORT_ACCEPTANCE = (
    ("source_region", "SOURCE-REGION-ACCEPTANCE-PASS"),
    ("point_source", "POINT-SOURCE-ACCEPTANCE-PASS"),
    ("source_lifecycle", "SOURCE-LIFECYCLE-ACCEPTANCE-PASS"),
    ("animated_source", "ANIMATED-SOURCE-ACCEPTANCE-PASS"),
    ("volume_source", "VOLUME-SOURCE-ACCEPTANCE-PASS"),
)
# tests/acceptance/blender_*.py scripts this gate deliberately does not run, with why.
# tests/gate/test_production_gate_coverage.py requires every script to be run here
# or listed here, so a new acceptance script cannot silently go unrun.
MANUAL_BLENDER_PROBES = {
    "tests/acceptance/blender_artist_bake_deleted_scene_acceptance.py":
        "needs a real GUI modal bake; exits silently in --background",
    "tests/acceptance/blender_artist_bake_gui_acceptance.py":
        "timer-driven GUI probe: call start(report_path) in a GUI session",
    "tests/acceptance/blender_artist_context_consent_acceptance.py":
        "needs real editor contexts and simulated ESC/RET events in a GUI session",
    "tests/acceptance/blender_artist_workflow_acceptance.py":
        "GUI timeline probe: call start(report_path) in a GUI session",
    "tests/acceptance/blender_custom_effect_example_scenes_acceptance.py":
        "plays back all 12 custom-effect example scenes; run manually after tools/release/build_custom_effect_example_scenes.py",
    "tests/acceptance/blender_effect_add_invoke_acceptance.py":
        "needs the interactive Add Effect search popup",
    "tests/acceptance/blender_particle_autogrowth_live_review.py":
        "profile-safe live review via tools/gate/launch_profile_safe_addon_review.py; leaves the GUI open",
    "tests/acceptance/blender_phase1_source_ui_acceptance.py":
        "profile-safe interactive review via tools/gate/launch_profile_safe_addon_review.py",
    "tests/acceptance/blender_phase3_profile_safe_review.py":
        "profile-safe visible review via tools/gate/launch_profile_safe_addon_review.py",
    "tests/acceptance/blender_performance_playback_benchmark.py":
        "separate foreground handler/visibility evidence; the mandatory saved-demo budget runs via tools/benchmark_versions.py",
}
RELEASE_WARP_VALIDATORS = ("phase4", "phase5", "custom_effects")
# tools/release/validate_{check}_blender.py: (check, inputs, success marker).
# Inputs name the optional arguments a validator takes. Before 2.0 nothing ran
# these, and five of six had silently broken against the 1.1 settings split.
RELEASE_PASSED = '"status": "passed"'
RELEASE_BLENDER_VALIDATORS = (
    ("phase3", ("work_dir",), RELEASE_PASSED),
    ("phase4", ("work_dir",), RELEASE_PASSED),
    ("phase5", ("work_dir", "warp_root"), RELEASE_PASSED),
    ("properties_ui", ("work_dir", "warp_root"), "FLUIDSIM_PROPERTIES_UI_VALIDATION_OK"),
    ("custom_effects", ("work_dir", "warp_root", "examples", "cuda"),
     "FLUIDSIM_CUSTOM_EFFECTS_BLENDER_VALIDATED"),
    ("examples", ("examples",), "FLUIDSIM_EXAMPLES_VALIDATED"),
)


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
    # Keep a caller's own profile: BLENDER_USER_EXTENSIONS would override the
    # extensions dir of a BLENDER_USER_RESOURCES clean profile.
    if not (env.get("BLENDER_USER_EXTENSIONS") or env.get("BLENDER_USER_RESOURCES")):
        env["BLENDER_USER_EXTENSIONS"] = str(GATE_BLENDER_USER_EXTENSIONS)
    return env


def _blender(blender_exe, *args):
    """Blender command line that honours the gate's PYTHONPATH and PYTHON* env."""
    return [blender_exe, "--python-use-system-env", *args]


def _release_blender_validator_command(blender_exe, check, inputs, warp_root):
    """Blender command line for one tools/release/validate_{check}_blender.py."""
    out = ROOT / "artifacts" / "release_validation"
    optional = {
        "work_dir": ("--work-dir", str(out / f"{check}-blender-work")),
        "warp_root": ("--expected-warp-root", str(warp_root)),
        "examples": ("--examples", str(ROOT / "fluidSim_warp_blender51" / "resources" / "examples")),
        "cuda": ("--device", "cuda:0"),
    }
    return _blender(
        blender_exe, "--background", "--factory-startup", "--python-exit-code", "1",
        "--python", f"tools/release/validate_{check}_blender.py", "--",
        "--module", "fluidSim_warp_blender51",
        "--report-json", str(out / f"{check}-blender-validation.json"),
        *(arg for name in inputs for arg in optional[name]),
    )


def _release_env(env):
    """The validators import the add-on by name, so the repo root must be importable."""
    release_env = env.copy()
    existing = release_env.get("PYTHONPATH", "")
    release_env["PYTHONPATH"] = str(ROOT) if not existing else existing + os.pathsep + str(ROOT)
    return release_env


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


def _print_gate_metadata(blender_exe, blender_py, include_gui, env):
    status = _git_text("status", "--porcelain")
    ignored = _git_text("ls-files", "-ci", "--exclude-standard")
    print("\n== gate metadata ==")
    print(f"timestamp_utc: {_dt.datetime.now(_dt.UTC).isoformat()}")
    print(f"git_head: {_git_text('rev-parse', '--short=12', 'HEAD')}")
    print(f"git_dirty: {bool(status)}")
    print(f"tracked_ignored_files: {bool(ignored)}")
    print(f"blender_exe: {blender_exe}")
    print(f"blender_py: {blender_py}")
    print("blender_user_extensions: "
          + (env.get("BLENDER_USER_EXTENSIONS") or "BLENDER_USER_RESOURCES/extensions"))
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


def _validate_performance_evidence():
    from tools.benchmark_versions import REFERENCE_COMMITS, _inventory
    from tools.performance_budget import evaluate_budget
    performance = ROOT / "artifacts" / "performance_budget" / "summary.json"
    report = json.loads(performance.read_text(encoding="utf-8"))
    if report.get("passed") is not True or report.get("gitHead") != _git_text("rev-parse", "HEAD"):
        raise RuntimeError("missing passing performance budget evidence for this HEAD")
    if report.get("candidateInventorySha256") != _inventory(ROOT):
        raise RuntimeError("source or demo bytes changed after the performance comparison")
    recomputed = evaluate_budget(report.get("runs"))
    if not recomputed["passed"]:
        raise RuntimeError("raw performance evidence does not satisfy the budget")
    if any(report.get(key) != value for key, value in recomputed.items()):
        raise RuntimeError("stored performance summary differs from raw evidence")
    if report.get("referenceCommits") != REFERENCE_COMMITS:
        raise RuntimeError("performance reference commits differ from pinned releases")
    observed_sources = {}
    for run in report['runs']:
        baseline = run['label'] == 'baseline'
        expected_head = REFERENCE_COMMITS[run['referenceVersion']] if baseline else report['gitHead']
        root_inventory = run.get('rootInventorySha256')
        root = Path(run['root'])
        if (run.get('gitHead') != expected_head
                or run.get('candidateInventorySha256') != report['candidateInventorySha256']
                or type(run.get('processId')) is not int or run['processId'] <= 0
                or not isinstance(root_inventory, str) or len(root_inventory) != 64
                or any(value not in '0123456789abcdef' for value in root_inventory)
                or Path(run.get('addon', '')).resolve() != (root / 'fluidSim_warp_blender51/__init__.py').resolve()
                or (not baseline and root_inventory != report['candidateInventorySha256'])):
            raise RuntimeError('raw performance trial provenance is not bound to selected sources')
        cohort = run['referenceVersion'] if baseline else 'candidate'
        observed = (str(root.resolve()), root_inventory)
        if cohort in observed_sources and observed_sources[cohort] != observed:
            raise RuntimeError('raw performance source inventory changes between trials')
        observed_sources[cohort] = observed
    return report


def _write_gate_manifest(blender_exe, warp_version=None):
    report = _validate_performance_evidence()
    artifact = ROOT / "artifacts" / "demo_analysis" / "analysis_report.json"
    if not artifact.is_file():
        raise RuntimeError(f"missing generated demo artifact: {artifact}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    performance = ROOT / "artifacts" / "performance_budget" / "summary.json"
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
        "performance_artifact": "artifacts/performance_budget/summary.json",
        "performance_artifact_sha256": hashlib.sha256(performance.read_bytes()).hexdigest(),
        "performance_reference_commits": report["referenceCommits"],
    }
    GATE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    GATE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")


def main(argv=None):
    """Run storage regressions/reports around the unchanged production gate."""
    from tools.storage_hygiene.integration import production_storage

    with production_storage(ROOT):
        return _production_main(argv)


def _production_main(argv=None):
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
        help="also run tests/acceptance/blender_gui_acceptance.py and report full production proof",
    )
    args = parser.parse_args(argv)
    blender_exe = os.environ.get("BLENDER_EXE", DEFAULT_BLENDER_EXE)
    blender_py = os.environ.get("BLENDER_PY", DEFAULT_BLENDER_PY)
    include_gui = args.include_gui or os.environ.get("FLUIDSIM_FULL_GUI_ACCEPTANCE") == "1"
    env = _env()

    _clear_gate_manifest()
    _print_gate_metadata(blender_exe, blender_py, include_gui, env)
    _run("git whitespace", ["git", "diff", "--check"])
    _check_no_tracked_npz()
    _check_no_tracked_ignored_files()
    warp_root = _check_certified_warp(blender_py, env)
    _run("python syntax", [
        blender_py, "-m", "py_compile",
        "tools/profiling/profile_gpu_solver.py",
        "tools/profiling/profile_user_path.py",
        "tools/demos/analyze_demos.py",
        "tools/demos/build_demo_scenes.py",
        "tools/demos/check_demo_blends.py",
        "tools/run_production_gate.py",
        "tools/benchmark_versions.py",
        "tools/benchmark_demo_trial.py",
        "tools/performance_budget.py",
        "tools/gate/validate_gui_playback_report.py",
        "tests/acceptance/blender_cache_export_pipeline.py",
        "tests/acceptance/blender_memory_safety_gate.py",
        "tests/acceptance/blender_mesh_source_acceptance.py",
        "tests/acceptance/blender_force_region_acceptance.py",
        "tests/acceptance/blender_phase3_source_ui_acceptance.py",
        "tests/acceptance/blender_capacity_planner_acceptance.py",
        "tests/acceptance/blender_simulation_admission_acceptance.py",
        "tests/acceptance/blender_closed_domain_stability_acceptance.py",
        "tests/acceptance/blender_custom_effects_acceptance.py",
        "tests/acceptance/blender_phase3_profile_safe_review.py",
        "tests/acceptance/blender_phase4_live_interaction_acceptance.py",
        "tests/acceptance/blender_gui_acceptance.py",
        "tests/acceptance/blender_gui_playback_probe.py",
        "tests/acceptance/blender_viewport_2frame_gate.py",
        "tests/acceptance/blender_pre_bridge_file_acceptance.py",
        "tests/acceptance/_phase2_source_acceptance_common.py",
        *(script for script, _ in BACKGROUND_ACCEPTANCE),
        *(f"tests/acceptance/blender_{capability}_acceptance.py"
          for capability, _ in SOURCE_REPORT_ACCEPTANCE),
        *(f"tools/release/validate_{check}_blender.py"
          for check, _, _ in RELEASE_BLENDER_VALIDATORS),
    ], env=env)
    _run("pytest CUDA regressions", [
        blender_py, "-m", "pytest",
        "tests/solver/test_pressure_projection_regressions.py",
        "tests/cache_io/test_cache_io.py",
        "tests/cache_io/test_io_path_policy.py",
        "-q",
    ], env=env)
    for script in (
        "tests/scripts/smoke_core.py",
        "tests/scripts/smoke_solver.py",
        "tests/validation/test_solver_validation.py",
        "tests/validation/test_validation.py",
    ):
        _run(script, [blender_py, script], env=env)
    _run("GPU profiler smoke", [
        blender_py, "tools/profiling/profile_gpu_solver.py",
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
    for check, inputs, success_marker in RELEASE_BLENDER_VALIDATORS:
        _run(
            f"{check} release validation in Blender",
            _release_blender_validator_command(blender_exe, check, inputs, warp_root),
            env=_release_env(env),
            reject_output=True,
            expect_marker=success_marker,
        )
    for script, success_marker in BACKGROUND_ACCEPTANCE:
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
                 "tests/acceptance/blender_phase3_source_ui_acceptance.py", "--", "--json-out",
                 os.path.abspath("artifacts/phase3_source_ui_acceptance.json")),
        env=env, reject_output=True, expect_marker="PHASE3-SOURCE-UI-ACCEPTANCE-PASS")
    _run(
        "source capacity planner acceptance",
        _blender(blender_exe, "--background", "--factory-startup", "--python",
                 "tests/acceptance/blender_capacity_planner_acceptance.py", "--", "--json-out",
                 os.path.abspath("artifacts/capacity_planner_acceptance.json")),
        env=env, reject_output=True, expect_marker="CAPACITY-PLANNER-ACCEPTANCE-PASS")
    for capability, success_marker in SOURCE_REPORT_ACCEPTANCE:
        _run(
            f"{capability} acceptance",
            _blender(blender_exe, "--background", "--factory-startup", "--python",
                     f"tests/acceptance/blender_{capability}_acceptance.py", "--", "--json-out",
                     os.path.abspath(f"artifacts/phase2_source_acceptance/{capability}.json")),
            env=env, reject_output=True, expect_marker=success_marker)
    for scene_name in CANONICAL_DEMO_SCENES:
        _run(f"demo cache build {scene_name}", _blender(
            blender_exe,
            "--background",
            "--factory-startup",
            "--python",
            "tools/demos/build_demo_scenes.py",
            "--",
            scene_name,
        ), env=env, reject_output=True, expect_marker="DEMO-BUILD-PASS")
    _run(
        "demo analysis", [blender_py, "tools/demos/analyze_demos.py"], env=env,
        expect_marker="DEMO-ANALYSIS-PASS (21/21)")
    _run("demo blend checks", _blender(
        blender_exe,
        "--background",
        "--factory-startup",
        "--python",
        "tools/demos/check_demo_blends.py",
    ), env=env, reject_output=True, expect_marker="DEMO-BLEND-CHECK-PASS")
    if include_gui:
        _run("interactive GUI acceptance", _blender(
            blender_exe,
            "--factory-startup",
            "--python",
            "tests/acceptance/blender_gui_acceptance.py",
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
            "tests/acceptance/blender_gui_playback_probe.py",
        ), env=playback_env, reject_output=True,
           expect_marker="GUI-PLAYBACK-PROBE-PASS")
        _run("saved-demo GUI playback report", [
            blender_py,
            "tools/gate/validate_gui_playback_report.py",
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
            "tools/profiling/profile_user_path.py",
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
    # Run after demo rebuilds and GUI probes, so the budget covers the final
    # delivered saved files, not the pre-build inputs.
    _run("paired demo performance budget", [
        blender_py, "tools/benchmark_versions.py", "--demo-budget",
        "--blender", blender_exe, "--candidate-root", str(ROOT),
        "--trials", "8", "--frames", "48",
        "--out-dir", "artifacts/performance_budget",
    ], env=env, expect_marker="PERFORMANCE-BUDGET-PASS")
    _validate_performance_evidence()
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
