"""Mandatory local storage tests and observational pre/post task reports."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from . import storage_hygiene as checker

HERE = Path(__file__).resolve().parent
REQUIRED_CONTAINERS = {r'C:\tmp', r'C:\fg-target', r'C:\awt', r'C:\Users\Tyler\Documents\Codex'}


def policy(root, *, default=False):
    override = None if default else os.environ.get("FLUIDSIM_STORAGE_POLICY")
    path = Path(override) if override else HERE / "policy.json"
    try:
        required = json.loads((HERE / "policy.json").read_text(encoding="utf-8-sig"))
        checker.validate(required)
        if {checker.normalized(entry['path']) for entry in required['roots']} != {checker.normalized(path) for path in REQUIRED_CONTAINERS}:
            raise ValueError('Mandatory policy must establish all four container monitors')
        config = copy.deepcopy(required) if path == HERE / "policy.json" else json.loads(path.read_text(encoding="utf-8-sig"))
        checker.validate(config)
        # Overrides configure output ownership/budgets, never remove holds or
        # retry denied paths. Missing bundled protections disables scanning
        # with an explicit diagnostic rather than probing the override alone.
        required_paths = required["never_probe"] + [
            path for entry in required["roots"] for path in entry.get("protected", [])
        ] + [str(Path(root).resolve())]
        config["never_probe"] = list(dict.fromkeys(
            config.get("never_probe", []) + required_paths))
        return config
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"STORAGE-HYGIENE-DIAGNOSTIC policy unavailable: {exc}",
              file=sys.stderr, flush=True)
        if default:
            raise
        return {}  # emit diagnoses invalid configuration in both phases


def emit_configs(configs, phase):
    for label, config in configs:
        result = checker.emit(config, phase, label)
        if label == "liquid" and result is None:
            print(f"STORAGE-HYGIENE-DIAGNOSTIC mandatory telemetry measurement unavailable phase={phase}", file=sys.stderr, flush=True)
            return False
        for _, target in configs:
            checker.exclude_inaccessible(target, result)
    return True


@contextmanager
def production_storage(root):
    try:
        required = policy(root, default=True)
        checker.validate(required)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"STORAGE-HYGIENE-DEPENDENCY-FAIL mandatory policy unavailable: {exc}; production not started", file=sys.stderr, flush=True)
        raise SystemExit(2)
    configs = [("liquid", required)]
    if os.environ.get("FLUIDSIM_STORAGE_POLICY"):
        configs.append(("liquid:owned", policy(root)))
    # Owner overrides may add holds/denials, never relax the mandatory pass.
    denied = list(dict.fromkeys(
        [path for _, config in configs for path in config.get("never_probe", [])] +
        [path for _, config in configs for entry in config.get("roots", [])
         for path in entry.get("protected", [])]))
    for _, config in configs:
        config["never_probe"] = denied.copy()
    if not emit_configs(configs, "pre-production"):
        print("STORAGE-HYGIENE-DEPENDENCY-FAIL mandatory telemetry unavailable; production not started", file=sys.stderr, flush=True)
        raise SystemExit(2)
    try:
        subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s",
             str(HERE / "tests"), "-v"],
            cwd=HERE, check=True,
        )
        yield
    finally:
        emit_configs(configs, "end-task")
