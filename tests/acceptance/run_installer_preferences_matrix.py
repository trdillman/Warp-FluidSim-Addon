"""Run native installed Preferences scenarios in disposable offline profiles.

python tests/acceptance/run_installer_preferences_matrix.py --blender /path/blender \
    --package /path/fluidsim.zip --wheel /path/locked-warp.whl --output /owned/evidence

Requires certified Blender 5.2.0 and a foreground display. No downloads. Successful
profiles are removed; failed profiles and every subprocess transcript are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SCENARIOS = ("no-warp", "good-owned", "damaged-owned", "foreign-preloaded",
             "offline-local", "cancel", "repeated-clicks")
PROFILE_PATHS = {
    "BLENDER_USER_CONFIG": "config", "BLENDER_USER_SCRIPTS": "scripts",
    "BLENDER_USER_EXTENSIONS": "extensions", "BLENDER_USER_DATAFILES": "datafiles",
    "PYTHONUSERBASE": "python-user-base", "WARP_CACHE_PATH": "warp-cache",
    "XDG_CACHE_HOME": "cache", "TEMP": "temp", "TMP": "temp", "TMPDIR": "temp",
}
REQUIRED_CHECKS = {
    "no-warp": {"online_refused"},
    "good-owned": {"owned_loaded", "repeated_noop"},
    "damaged-owned": {"owned_failure", "repair_offline_refused"},
    "foreign-preloaded": {"foreign_preserved", "install_refused"},
    "offline-local": {"local_install", "owned_loaded", "post_install_draw"},
    "cancel": {"picker_cancelled", "files_preserved"},
    "repeated-clicks": {"local_install", "owned_loaded", "repeated_noop", "post_install_draw"},
}
BASE_CHECKS = {"private_profile", "installed_extension", "network_off", "scenario_state"}


def profile_environment(profile):
    profile = Path(profile).resolve()
    profile.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key, relative in PROFILE_PATHS.items():
        path = profile / relative
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "BLENDER_USER_RESOURCES",
                "FLUIDSIM_ALLOW_EXTERNAL_WARP"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    (profile / ".installer-matrix-owned").write_text("installer-matrix-v1\n", encoding="utf-8")
    return env


def require_owned_profile(profile, environment):
    profile = Path(profile).resolve()
    if (profile / ".installer-matrix-owned").read_text(encoding="utf-8") != "installer-matrix-v1\n":
        raise ValueError("Missing owned installer matrix profile marker")
    for key, relative in PROFILE_PATHS.items():
        value = environment.get(key, "")
        expected = profile / relative
        if not value or not Path(value).is_absolute() or Path(value).resolve() != expected:
            raise ValueError(f"{key} must name its private owned directory")


def validate_scenario_report(report, scenario):
    if (report.get("diagnostic") or not report.get("certified_runtime")
            or report.get("scenario") != scenario or report.get("passed") is not True):
        raise ValueError("Missing passing certified native scenario report")
    required = ({"fluidsim.refresh_runtime_probe"} if scenario == "good-owned" else
                {"fluidsim.refresh_runtime_probe", "fluidsim.install_warp"}
                if scenario == "damaged-owned" else
                {"fluidsim.install_warp", "fluidsim.install_warp_from_file"})
    if not any(required <= set(draw.get("operators", ())) for draw in report.get("draws", ())):
        raise ValueError("Native controls were not drawn together")
    checks = report.get("checks", {})
    if (not isinstance(checks, dict)
            or not (BASE_CHECKS | REQUIRED_CHECKS[scenario]) <= checks.keys()
            or not all(value is True for value in checks.values())):
        raise ValueError("Missing or failed native scenario checks")


def verify_local_wheel(wheel, lock):
    wheel = Path(wheel)
    metadata = lock.get("files", {}).get(wheel.name)
    if metadata is None or wheel.stat().st_size != metadata["size"]:
        raise ValueError("Supply the exact locked local Warp wheel")
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("Local Warp wheel checksum mismatch")
    return metadata


def run_matrix(blender, package, wheel, output, *, scenarios=SCENARIOS, runner=subprocess.run):
    root = Path(__file__).resolve().parents[2]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((root / "tools/release/wheels.lock.json").read_text(encoding="utf-8"))
    verify_local_wheel(wheel, lock)
    package = Path(package).resolve(strict=True)
    script = Path(__file__).with_name("blender_installer_preferences_acceptance.py")
    reports = []
    for scenario in scenarios:
        case = Path(tempfile.mkdtemp(prefix=f"{scenario}-", dir=output))
        profile = case / "profile"
        env = profile_environment(profile)
        repository = profile / "repository"
        repository.mkdir()
        result_path = case / "result.json"
        commands = [
            [str(blender), "--background", "--offline-mode", "--command", "extension", "repo-add",
             "installer_matrix", "--name", "Installer Matrix", "--directory", str(repository), "--clear-all"],
            [str(blender), "--background", "--offline-mode", "--command", "extension", "install-file",
             "-r", "installer_matrix", str(package)],
            [str(blender), "--offline-mode", "--python-exit-code", "1",
             "--python", str(script), "--", "--addon-module", "bl_ext.installer_matrix.fluidsim_warp",
             "--profile-root", str(profile), "--scenario", scenario, "--wheel", str(Path(wheel).resolve()),
             "--json-out", str(result_path)],
        ]
        try:
            for index, command in enumerate(commands):
                completed = runner(command, env=env, capture_output=True, text=True,
                                   timeout=180, check=False)
                (case / f"process-{index}.log").write_text(
                    completed.stdout + "\n" + completed.stderr, encoding="utf-8")
                if completed.returncode:
                    raise RuntimeError(f"{scenario} process {index} exited {completed.returncode}")
            report = json.loads(result_path.read_text(encoding="utf-8"))
            validate_scenario_report(report, scenario)
            reports.append({"scenario": scenario, "report": str(result_path), "passed": True})
            shutil.rmtree(profile)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            reports.append({"scenario": scenario, "passed": False, "error": repr(exc), "evidence": str(case)})
    summary = {"passed": bool(reports) and all(item["passed"] for item in reports), "scenarios": reports}
    (output / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return 0 if run_matrix(args.blender.resolve(), args.package, args.wheel, args.output)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
