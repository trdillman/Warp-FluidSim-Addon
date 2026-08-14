#!/usr/bin/env python3
"""Launch an owned, profile-safe Blender add-on review session.

The launcher deliberately inherits Blender's normal user profile.  It snapshots
profile state, starts a new Blender process, installs a verified temporary copy
of this checkout's add-on only in that process, and proves the profile is
unchanged after the owned process exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ADDON_NAME = "fluidSim_warp_blender51"
CONFLICTING_ADDONS = ("bl_ext.user_default.nexus_clone_warp",)
REPORT_SCHEMA = "fluidsim-profile-safe-addon-review-v1"
VISIBLE_REVIEW_RESULT_ENV = "FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT"
PROFILE_OVERRIDE_NAMES = (
    "BLENDER_USER_CONFIG",
    "BLENDER_USER_SCRIPTS",
    "BLENDER_USER_DATAFILES",
)
_IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_file(path: Path) -> dict[str, Any]:
    """Return non-mutating existence/hash evidence for one profile file."""

    path = path.resolve()
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if result["exists"]:
        stat = path.stat()
        result.update(
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            sha256=_sha256(path),
        )
    else:
        result.update(size=None, mtime_ns=None, sha256=None)
    return result


def snapshot_profile_files(config_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        name: snapshot_file(config_dir / name)
        for name in ("userpref.blend", "startup.blend")
    }


def _file_signature(snapshot: Mapping[str, Any]) -> tuple[Any, ...]:
    return snapshot.get("exists"), snapshot.get("size"), snapshot.get("sha256")


def compare_profile_state(
    before_probe: Mapping[str, Any],
    after_probe: Mapping[str, Any],
    before_files: Mapping[str, Mapping[str, Any]],
    after_files: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    """Compare only persistent state: enabled modules and file existence/content."""

    drift: list[str] = []
    if sorted(before_probe.get("enabled_addons", [])) != sorted(
        after_probe.get("enabled_addons", [])
    ):
        drift.append("enabled_addons")
    for name in ("userpref.blend", "startup.blend"):
        if _file_signature(before_files[name]) != _file_signature(after_files[name]):
            drift.append(name)
    return not drift, drift


def _candidate_blenders() -> list[Path]:
    candidates: list[Path] = []
    if os.name == "nt":
        candidates.extend(
            [
                Path(
                    r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
                ),
                Path(
                    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
                ),
            ]
        )
    found = shutil.which("blender")
    if found:
        candidates.append(Path(found))
    return candidates


def discover_blender(
    explicit: str | os.PathLike[str] | None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve Blender deterministically: CLI, environment, known path, PATH."""

    environ = os.environ if environ is None else environ
    raw_candidates: list[Path] = []
    if explicit:
        raw_candidates.append(Path(explicit))
    else:
        for key in ("FLUIDSIM_BLENDER_EXECUTABLE", "BLENDER_EXECUTABLE"):
            if environ.get(key):
                raw_candidates.append(Path(environ[key]))
                break
        raw_candidates.extend(_candidate_blenders())

    attempted: list[str] = []
    for candidate in raw_candidates:
        resolved = candidate.expanduser().resolve()
        attempted.append(str(resolved))
        if resolved.is_file():
            return resolved
    detail = ", ".join(attempted) if attempted else "no candidates"
    raise FileNotFoundError(
        "Blender executable not found; pass --blender or set "
        f"FLUIDSIM_BLENDER_EXECUTABLE. Attempted: {detail}"
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _probe_source() -> str:
    return r'''from __future__ import annotations
import json
import os
from pathlib import Path
import bpy

out = Path(os.environ["FLUIDSIM_PROFILE_PROBE_RESULT"])
payload = {
    "ok": True,
    "blender_version": bpy.app.version_string,
    "blender_version_tuple": list(bpy.app.version),
    "blender_binary": str(Path(bpy.app.binary_path).resolve()),
    "config_dir": str(Path(bpy.utils.user_resource("CONFIG")).resolve()),
    "enabled_addons": sorted(a.module for a in bpy.context.preferences.addons),
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("FLUIDSIM-PROFILE-PROBE " + json.dumps(payload))
'''


def _bootstrap_source() -> str:
    # This file runs inside Blender.  Keep it self-contained: the checkout is
    # intentionally not importable until after the verified temporary copy.
    return r'''from __future__ import annotations
import hashlib
import importlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import addon_utils
import bpy

CONFIG_PATH = Path(os.environ["FLUIDSIM_PROFILE_SAFE_REVIEW_CONFIG"])
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
MARKER = Path(CONFIG["marker"])
ADDON_NAME = CONFIG["addon_name"]
SOURCE_ROOT = Path(CONFIG["source_root"]).resolve()
ADDONS_ROOT = Path(CONFIG["addons_root"]).resolve()
PACKAGE_ROOT = ADDONS_ROOT / ADDON_NAME
EXPECTED_BLEND = Path(CONFIG["blend"]).resolve()
IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
result = {"stage": "start", "ok": False}


def write_marker():
    MARKER.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root):
    entries = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        entries[relative.as_posix()] = {"size": path.stat().st_size, "sha256": digest(path)}
    return entries


def ignore_copy(_directory, names):
    return [name for name in names if name in IGNORED_DIRS or Path(name).suffix.lower() in IGNORED_SUFFIXES]


def under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


try:
    result["enabled_addons_before"] = sorted(
        addon.module for addon in bpy.context.preferences.addons
    )
    result["conflicting_addons"] = {}
    identities = [ADDON_NAME] + list(CONFIG["conflicting_addons"])
    for identity in identities:
        default_enabled, loaded = addon_utils.check(identity)
        info = {
            "default_enabled": bool(default_enabled),
            "loaded_before": bool(loaded),
        }
        if loaded:
            addon_utils.disable(identity, default_set=False)
        info["loaded_after_disable"] = bool(addon_utils.check(identity)[1])
        result["conflicting_addons"][identity] = info

    stale = [
        name for name in tuple(sys.modules)
        if name == ADDON_NAME or name.startswith(ADDON_NAME + ".")
    ]
    for name in stale:
        sys.modules.pop(name, None)
    result["purged_stale_modules"] = stale
    importlib.invalidate_caches()

    ADDONS_ROOT.mkdir(parents=True, exist_ok=True)
    if PACKAGE_ROOT.exists():
        raise RuntimeError("temporary package root unexpectedly already exists")
    shutil.copytree(SOURCE_ROOT, PACKAGE_ROOT, ignore=ignore_copy)

    source_entries = inventory(SOURCE_ROOT)
    copied_entries = inventory(PACKAGE_ROOT)
    missing = sorted(set(source_entries) - set(copied_entries))
    extra = sorted(set(copied_entries) - set(source_entries))
    mismatched = sorted(
        name for name in set(source_entries) & set(copied_entries)
        if source_entries[name] != copied_entries[name]
    )
    result["inventory"] = {
        "source": source_entries,
        "copy": copied_entries,
        "source_count": len(source_entries),
        "copy_count": len(copied_entries),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "complete": not (missing or extra or mismatched),
    }
    if not result["inventory"]["complete"]:
        raise RuntimeError("temporary add-on copy failed inventory verification")

    sys.path.insert(0, str(ADDONS_ROOT))
    importlib.invalidate_caches()
    importlib.import_module(ADDON_NAME)
    addon_utils.enable(ADDON_NAME, default_set=False, persistent=False)
    module = sys.modules[ADDON_NAME]
    module_path = Path(module.__file__).resolve()
    loaded_modules = {}
    outside_modules = []
    for name, loaded_module in tuple(sys.modules.items()):
        if name != ADDON_NAME and not name.startswith(ADDON_NAME + "."):
            continue
        loaded_path = getattr(loaded_module, "__file__", None)
        if loaded_path:
            loaded_modules[name] = str(Path(loaded_path).resolve())
            if not under(loaded_path, PACKAGE_ROOT):
                outside_modules.append(name)

    current_blend = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
    default_enabled, runtime_loaded = addon_utils.check(ADDON_NAME)
    result["module"] = {
        "name": ADDON_NAME,
        "version": list(getattr(module, "bl_info", {}).get("version", ())),
        "path": str(module_path),
        "under_temp_root": under(module_path, PACKAGE_ROOT),
        "runtime_loaded": bool(runtime_loaded),
        "default_enabled": bool(default_enabled),
        "preferences_entry_present": ADDON_NAME in {
            addon.module for addon in bpy.context.preferences.addons
        },
        "nonpersistent_enable_requested": True,
        "loaded_module_paths": loaded_modules,
        "outside_temp_modules": outside_modules,
    }
    result["blend"] = {
        "requested": str(EXPECTED_BLEND),
        "active": str(current_blend) if current_blend else None,
        "matches": current_blend == EXPECTED_BLEND,
    }
    result["blender"] = {
        "version": bpy.app.version_string,
        "version_tuple": list(bpy.app.version),
        "binary": str(Path(bpy.app.binary_path).resolve()),
    }
    conflicts_unloaded_at_ready = True
    for identity in CONFIG["conflicting_addons"]:
        loaded_at_ready = bool(addon_utils.check(identity)[1])
        result["conflicting_addons"][identity]["loaded_at_ready"] = loaded_at_ready
        conflicts_unloaded_at_ready = conflicts_unloaded_at_ready and not loaded_at_ready
    result["one_package_identity"] = conflicts_unloaded_at_ready and not outside_modules
    result["ok"] = bool(
        result["inventory"]["complete"]
        and result["module"]["under_temp_root"]
        and result["module"]["runtime_loaded"]
        and result["blend"]["matches"]
        and result["one_package_identity"]
    )
    if not result["ok"]:
        raise RuntimeError("profile-safe bootstrap identity validation failed")
    # Do not call addon_utils.disable() from Python atexit. Blender 5.1 has
    # already started RNA teardown at that point, and iterating preferences
    # during extensions_refresh can dereference destroyed RNA. The enable is
    # nonpersistent and the parent owns/removes the temporary package.
    result["stage"] = "ready"
except Exception as exc:
    result["stage"] = "error"
    result["error"] = repr(exc)
    result["traceback"] = traceback.format_exc()
finally:
    write_marker()

print("FLUIDSIM-PROFILE-SAFE-BOOTSTRAP " + json.dumps(result, default=str))
'''


def _run_profile_probe(
    blender: Path,
    script: Path,
    result_path: Path,
    env: Mapping[str, str],
    timeout: float,
) -> dict[str, Any]:
    probe_env = dict(env)
    probe_env["FLUIDSIM_PROFILE_PROBE_RESULT"] = str(result_path)
    completed = subprocess.run(
        [str(blender), "--background", "--python", str(script)],
        env=probe_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            "profile probe failed "
            f"(exit {completed.returncode}): {completed.stderr[-2000:]}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(f"profile probe returned failure: {payload}")
    return payload


def _wait_for_marker(marker: Path, process: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.is_file():
            try:
                return json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"owned Blender process exited with {returncode} before readiness"
            )
        time.sleep(0.1)
    raise TimeoutError(f"Blender did not become review-ready within {timeout:g}s")


def _wait_for_review_result(
    result_path: Path,
    process: subprocess.Popen[Any],
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    malformed: Exception | None = None
    while time.monotonic() < deadline:
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise TypeError("visible-review result must be a JSON object")
                return payload
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                malformed = exc
        returncode = process.poll()
        if returncode is not None:
            break
        time.sleep(0.1)
    if malformed is not None:
        raise RuntimeError(f"visible-review result is malformed: {malformed}")
    raise RuntimeError(
        "visible-review script did not write its required JSON result")


def _stop_owned_process(process: subprocess.Popen[Any]) -> None:
    """Stop only the Popen handle created by this launcher."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _find_recent_crash(started_epoch: float, blend: Path) -> str | None:
    roots = {Path(tempfile.gettempdir())}
    for name in ("TEMP", "TMP"):
        if os.environ.get(name):
            roots.add(Path(os.environ[name]))
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        # Blender names crash artifacts from the active .blend.  Do not attach
        # an unrelated concurrently-running Blender's generic crash report to
        # this owned session.
        try:
            candidates.extend(
                path
                for path in root.glob(f"{blend.stem}*.crash.txt")
                if path.is_file() and path.stat().st_mtime >= started_epoch - 2
            )
        except OSError:
            continue
    if not candidates:
        return None
    newest = max(set(candidates), key=lambda path: path.stat().st_mtime)
    return str(newest.resolve())


def _backup_profile_files(
    snapshots: Mapping[str, Mapping[str, Any]], destination: Path, prefix: str
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, snapshot in snapshots.items():
        if snapshot.get("exists"):
            shutil.copy2(Path(str(snapshot["path"])), destination / f"{prefix}_{name}")


def _persist_drift_evidence(
    report_path: Path,
    before_backup: Path,
    after_files: Mapping[str, Mapping[str, Any]],
) -> Path:
    destination = Path(
        tempfile.mkdtemp(
            prefix=report_path.stem + "_profile_drift_", dir=report_path.parent
        )
    )
    for path in before_backup.glob("*"):
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    _backup_profile_files(after_files, destination, "after")
    return destination


def _empty_report(blend: Path, report_path: Path) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "ok": False,
        "stage": "initializing",
        "created_at": _utc_now(),
        "finished_at": None,
        "process": {
            "owned": True,
            "pid": None,
            "returncode": None,
            "command": [],
            "stdout_log": str(report_path.with_suffix(report_path.suffix + ".stdout.log")),
            "stderr_log": str(report_path.with_suffix(report_path.suffix + ".stderr.log")),
        },
        "blender": {},
        "module": {"name": ADDON_NAME},
        "blend": {"requested": str(blend), "active": None, "matches": False},
        "inventory": {
            "source": {},
            "copy": {},
            "source_count": 0,
            "copy_count": 0,
            "missing": [],
            "extra": [],
            "mismatched": [],
            "complete": False,
        },
        "profile": {
            "environment_overrides": {},
            "before": {},
            "after": {},
            "unchanged": False,
            "drift": [],
            "evidence_path": None,
        },
        "cleanup": {
            "owned_process_exited": False,
            "temp_copy_removed": False,
            "temp_root": None,
        },
        "crash_report_path": None,
        "visible_review": {
            "requested": False,
            "script": None,
            "result": None,
            "ok": None,
        },
        "errors": [],
    }


def run_review(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    blend = Path(args.blend).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = _empty_report(blend, report_path)
    review_script_raw = getattr(args, "review_script", None)
    review_script = (
        Path(review_script_raw).expanduser().resolve()
        if review_script_raw else None
    )
    report["visible_review"].update(
        requested=review_script is not None,
        script=str(review_script) if review_script is not None else None,
    )
    process: subprocess.Popen[Any] | None = None
    temp_root: Path | None = None
    before_probe: dict[str, Any] | None = None
    after_probe: dict[str, Any] | None = None
    before_files: dict[str, dict[str, Any]] | None = None
    after_files: dict[str, dict[str, Any]] | None = None
    before_backup: Path | None = None
    marker_payload: dict[str, Any] | None = None
    review_started_epoch = time.time()
    env = os.environ.copy()
    report["profile"]["environment_overrides"] = {
        key: env[key] for key in PROFILE_OVERRIDE_NAMES if key in env
    }

    try:
        if not blend.is_file():
            raise FileNotFoundError(f"review blend does not exist: {blend}")
        if review_script is not None and not review_script.is_file():
            raise FileNotFoundError(
                f"visible-review script does not exist: {review_script}")
        source_root = repo_root / ADDON_NAME
        if not (source_root / "__init__.py").is_file():
            raise FileNotFoundError(f"add-on source is incomplete: {source_root}")
        blender = discover_blender(args.blender, env)
        report["blender"]["executable"] = str(blender)
        report["blender"]["executable_sha256"] = _sha256(blender)

        temp_root = Path(tempfile.mkdtemp(prefix="fluidsim_profile_safe_review_"))
        report["cleanup"]["temp_root"] = str(temp_root)
        probe_script = temp_root / "profile_probe.py"
        probe_script.write_text(_probe_source(), encoding="utf-8")
        before_probe = _run_profile_probe(
            blender,
            probe_script,
            temp_root / "profile_before.json",
            env,
            args.probe_timeout,
        )
        config_dir = Path(before_probe["config_dir"])
        before_files = snapshot_profile_files(config_dir)
        before_backup = temp_root / "profile_before_backup"
        _backup_profile_files(before_files, before_backup, "before")
        report["profile"]["before"] = {
            "enabled_addons": before_probe["enabled_addons"],
            "files": before_files,
        }
        report["blender"].update(
            version=before_probe["blender_version"],
            version_tuple=before_probe["blender_version_tuple"],
            binary_reported=before_probe["blender_binary"],
        )

        marker = temp_root / "review_ready.json"
        bootstrap = temp_root / "review_bootstrap.py"
        bootstrap.write_text(_bootstrap_source(), encoding="utf-8")
        addons_root = temp_root / "addons"
        config = {
            "addon_name": ADDON_NAME,
            "source_root": str(source_root),
            "addons_root": str(addons_root),
            "marker": str(marker),
            "blend": str(blend),
            "conflicting_addons": list(dict.fromkeys(args.conflicting_addon)),
        }
        config_path = temp_root / "review_config.json"
        _atomic_write_json(config_path, config)
        child_env = dict(env)
        child_env["FLUIDSIM_PROFILE_SAFE_REVIEW_CONFIG"] = str(config_path)
        command = [str(blender), str(blend), "--python", str(bootstrap)]
        visible_result_path = temp_root / "visible_review_result.json"
        if review_script is not None:
            child_env[VISIBLE_REVIEW_RESULT_ENV] = str(visible_result_path)
            command.extend(["--python", str(review_script)])
        report["process"]["command"] = command
        report["stage"] = "launching"
        _atomic_write_json(report_path, report)

        stdout_path = Path(report["process"]["stdout_log"])
        stderr_path = Path(report["process"]["stderr_log"])
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                env=child_env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            report["process"]["pid"] = process.pid
            report["stage"] = "waiting_for_ready"
            _atomic_write_json(report_path, report)
            marker_payload = _wait_for_marker(marker, process, args.startup_timeout)
            if marker_payload.get("stage") != "ready" or not marker_payload.get("ok"):
                raise RuntimeError(
                    "Blender bootstrap failed: "
                    + str(marker_payload.get("error", marker_payload))
                )
            report["module"] = marker_payload["module"]
            report["blend"] = marker_payload["blend"]
            report["inventory"] = marker_payload["inventory"]
            report["blender"].update(marker_payload.get("blender", {}))
            report["one_package_identity"] = marker_payload.get(
                "one_package_identity", False
            )
            report["conflicting_addons"] = marker_payload.get(
                "conflicting_addons", {}
            )
            if review_script is not None:
                try:
                    visible_payload = _wait_for_review_result(
                        visible_result_path, process, args.startup_timeout
                    )
                    report["visible_review"]["result"] = visible_payload
                    report["visible_review"]["ok"] = visible_payload.get("ok") is True
                    if not report["visible_review"]["ok"]:
                        report["errors"].append(
                            "RuntimeError: visible-review script reported failure: "
                            + str(visible_payload.get("errors", visible_payload))
                        )
                except Exception as exc:  # noqa: BLE001 - retain profile proof
                    report["errors"].append(f"{type(exc).__name__}: {exc}")
                if report["visible_review"]["ok"]:
                    report["stage"] = "review_ready"
                    ready_marker = "FLUIDSIM-PROFILE-SAFE-REVIEW-READY"
                else:
                    report["stage"] = "runtime_review_failed"
                    ready_marker = None
                    _stop_owned_process(process)
            else:
                report["stage"] = "identity_ready"
                ready_marker = "FLUIDSIM-PROFILE-SAFE-IDENTITY-READY"
            _atomic_write_json(report_path, report)
            if ready_marker is not None:
                print(
                    ready_marker + " "
                    f"pid={process.pid} blend={blend}",
                    flush=True,
                )
            report["process"]["returncode"] = process.wait()
            report["cleanup"]["owned_process_exited"] = True

        after_probe = _run_profile_probe(
            blender,
            probe_script,
            temp_root / "profile_after.json",
            env,
            args.probe_timeout,
        )
        after_files = snapshot_profile_files(Path(after_probe["config_dir"]))
        report["profile"]["after"] = {
            "enabled_addons": after_probe["enabled_addons"],
            "files": after_files,
        }
        unchanged, drift = compare_profile_state(
            before_probe, after_probe, before_files, after_files
        )
        report["profile"]["unchanged"] = unchanged
        report["profile"]["drift"] = drift
        if not unchanged:
            evidence = _persist_drift_evidence(
                report_path, before_backup, after_files
            )
            report["profile"]["evidence_path"] = str(evidence)
            report["errors"].append("persistent Blender profile drift: " + ", ".join(drift))

    except KeyboardInterrupt:
        report["errors"].append("launcher interrupted by user")
    except Exception as exc:  # noqa: BLE001 - report every safety failure
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None and process.poll() is None:
            try:
                _stop_owned_process(process)
            except Exception as exc:  # noqa: BLE001
                report["errors"].append(f"owned process cleanup failed: {exc}")
        if process is not None:
            returncode = process.poll()
            if returncode is not None:
                report["process"]["returncode"] = returncode
                report["cleanup"]["owned_process_exited"] = True
        report["crash_report_path"] = _find_recent_crash(review_started_epoch, blend)
        if report["crash_report_path"]:
            report["errors"].append(
                "Blender crash report detected: " + report["crash_report_path"]
            )
        if temp_root is not None:
            try:
                shutil.rmtree(temp_root)
                report["cleanup"]["temp_copy_removed"] = not temp_root.exists()
            except Exception as exc:  # noqa: BLE001
                report["errors"].append(f"temporary copy cleanup failed: {exc}")
        process_ok = (
            report["cleanup"]["owned_process_exited"]
            and report["process"]["returncode"] == 0
        )
        marker_ok = bool(marker_payload and marker_payload.get("ok"))
        visible_review_ok = (
            not report["visible_review"]["requested"]
            or report["visible_review"]["ok"] is True
        )
        report["ok"] = bool(
            process_ok
            and marker_ok
            and visible_review_ok
            and report["inventory"].get("complete")
            and report.get("one_package_identity")
            and report["blend"].get("matches")
            and report["profile"].get("unchanged")
            and report["cleanup"].get("temp_copy_removed")
            and not report["errors"]
        )
        report["stage"] = "complete" if report["ok"] else "failed"
        report["finished_at"] = _utc_now()
        _atomic_write_json(report_path, report)

    result_kind = (
        "REVIEW" if report["visible_review"]["requested"] else "IDENTITY")
    print(
        "FLUIDSIM-PROFILE-SAFE-"
        + result_kind
        + "-"
        + ("PASS" if report["ok"] else "FAIL")
        + f" report={report_path}",
        flush=True,
    )
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", required=True, help="Blend file to open for review")
    parser.add_argument("--report", required=True, help="JSON report output path")
    parser.add_argument(
        "--blender",
        help=(
            "Blender executable override (otherwise FLUIDSIM_BLENDER_EXECUTABLE, "
            "BLENDER_EXECUTABLE, the Blender 5.1 install path, then PATH)"
        ),
    )
    parser.add_argument(
        "--conflicting-addon",
        action="append",
        default=list(CONFLICTING_ADDONS),
        help="Additional installed module identity to disable only in the owned process",
    )
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--probe-timeout", type=float, default=60.0)
    parser.add_argument(
        "--review-script",
        help=(
            "Optional Blender Python script run after the temporary add-on "
            "identity is ready. It must write JSON with ok:true to the path "
            f"provided in {VISIBLE_REVIEW_RESULT_ENV}. Without this option, "
            "the owned review remains open for manual review."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_review(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
