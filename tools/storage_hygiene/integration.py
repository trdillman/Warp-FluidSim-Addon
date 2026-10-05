"""Mandatory local storage tests and observational pre/post task reports."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

from . import storage_hygiene as checker

HERE = Path(__file__).resolve().parent


def policy(root, *, default=False):
    override = None if default else os.environ.get("FLUIDSIM_STORAGE_POLICY")
    path = Path(override) if override else HERE / "policy.json"
    try:
        required = json.loads((HERE / "policy.json").read_text(encoding="utf-8-sig"))
        checker.validate(required)
        config = json.loads(path.read_text(encoding="utf-8-sig"))
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
        return {}  # emit diagnoses invalid configuration in both phases


def emit_configs(configs, phase):
    for label, config in configs:
        result = checker.emit(config, phase, label)
        for _, target in configs:
            checker.exclude_inaccessible(target, result)


@contextmanager
def production_storage(root):
    configs = [("liquid", policy(root, default=True))]
    if os.environ.get("FLUIDSIM_STORAGE_POLICY"):
        configs.append(("liquid:owned", policy(root)))
    # Owner overrides may add holds/denials, never relax the mandatory pass.
    denied = list(dict.fromkeys(
        [path for _, config in configs for path in config.get("never_probe", [])] +
        [path for _, config in configs for entry in config.get("roots", [])
         for path in entry.get("protected", [])]))
    for _, config in configs:
        config["never_probe"] = denied.copy()
    emit_configs(configs, "pre-production")
    try:
        subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s",
             str(HERE / "tests"), "-v"],
            cwd=HERE, check=True,
        )
        yield
    finally:
        emit_configs(configs, "end-task")
