"""Launch actual Production Bake phases in separate profile-isolated Blender processes.

Exit 1: unavailable runtime or a failed invariant. Exit 2: all structural checks
ran, but compiled-emitter continuation has no approved numerical tolerance yet.
No invocation qualifies a release or emits a release PASS marker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.release.build_extension import _blender_process_environment

PHASES = ("preflight", "baseline", "partial", "resume", "rebake", "preview")


def run_phases(blender, state_root, device, *, execute=subprocess.run):
    root = Path(state_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(
            "Use a new state-root; existing evidence must not be overwritten"
        )
    source_env = os.environ.copy()
    # Source diagnostics may use an explicit dependency PYTHONPATH, but must
    # not replace Blender's Python home or execute a caller's startup script.
    for name in ("PYTHONHOME", "PYTHONSTARTUP"):
        source_env.pop(name, None)
    env = _blender_process_environment(
        source_env, profile_dir=root / "profile", allow_source_imports=True
    )
    reports = []
    status = "failed"
    try:
        for phase in PHASES:
            command = [
                str(blender),
                "--background",
                "--factory-startup",
                "--python-use-system-env",
                "--python-exit-code",
                "1",
                "--python",
                str(ROOT / "tests/acceptance/blender_frozen_production_acceptance.py"),
                "--",
                "--phase",
                phase,
                "--state-root",
                str(root),
                "--device",
                device,
            ]
            with (root / f"{phase}.log").open("w") as log:
                result = execute(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            if result.returncode:
                return 1
            path = root / f"{phase}-report.json"
            if not path.is_file():
                return 1
            report = json.loads(path.read_text())
            if (
                report.get("phase") != phase
                or report.get("qualified") is not False
                or report.get("status") != "structural_checks_passed"
            ):
                return 1
            reports.append(report)
        status = "pending_compiled_emitter_numerical_policy"
        return 2
    finally:
        (root / "run-report.json").write_text(
            json.dumps(
                {"status": status, "qualified": False, "phases": reports}, indent=2
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    raise SystemExit(run_phases(args.blender, args.state_root, args.device))
