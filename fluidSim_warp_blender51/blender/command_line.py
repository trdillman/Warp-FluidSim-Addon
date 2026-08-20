"""Farm-friendly Blender command line entry point for production baking.

The module deliberately imports neither :mod:`bpy` nor Warp at import time so
argument parsing and exit-code behavior remain unit-testable in normal Python.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping


EXIT_SUCCESS = 0
EXIT_ARGUMENTS = 2
EXIT_RUNTIME_IMPORT = 10
EXIT_DEVICE_UNAVAILABLE = 11
EXIT_NO_CUDA = 12
EXIT_SCENE_INVALID = 13
EXIT_RESUME_INCOMPATIBLE = 14
EXIT_BAKE_FAILURE = 15
EXIT_CANCELLED = 16
EXIT_CUSTOM_CODE_POLICY = 17


@dataclass(frozen=True)
class CliOptions:
    scene: str
    frame_start: int | None
    frame_end: int | None
    cache: Path
    device: str
    resume: str
    checkpoint_interval: int
    save_after_bake: bool
    report_json: Path | None
    custom_code_policy: str = "builtins-only"


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="FluidSim production bake",
        description="Bake a resumable FluidSim cache from Blender background mode.",
    )
    parser.add_argument("--scene", default="", help="Blender scene name; current scene when omitted")
    parser.add_argument("--start", dest="frame_start", type=int)
    parser.add_argument("--end", dest="frame_end", type=int)
    parser.add_argument("--cache", required=True, type=_absolute_path)
    parser.add_argument("--device", default="", help="Warp CUDA alias, for example cuda:0")
    parser.add_argument(
        "--resume", choices=("never", "auto", "required"), default="auto"
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--save-after-bake", action="store_true")
    parser.add_argument("--report-json", type=_absolute_path)
    parser.add_argument(
        "--custom-code-policy",
        choices=("deny", "builtins-only", "trusted"),
        default="builtins-only",
        help="Policy for executable Custom Effects in background mode",
    )
    return parser


def parse_args(argv: list[str]) -> CliOptions:
    args = _parser().parse_args(argv)
    if args.frame_start is not None and args.frame_end is not None:
        if args.frame_end < args.frame_start:
            _parser().error("--end must be greater than or equal to --start")
    if args.checkpoint_interval < 1:
        _parser().error("--checkpoint-interval must be at least 1")
    return CliOptions(
        scene=str(args.scene),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        cache=args.cache,
        device=str(args.device),
        resume=str(args.resume),
        checkpoint_interval=int(args.checkpoint_interval),
        save_after_bake=bool(args.save_after_bake),
        report_json=args.report_json,
        custom_code_policy=str(args.custom_code_policy),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one durable machine-readable report without exposing partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")), flush=True)


def _scene_validation(scene) -> tuple[bool, str]:
    fs = getattr(scene, "fluidsim", None)
    if fs is None:
        return False, "FluidSim scene settings are not registered"
    domain = getattr(fs, "domain_object", None)
    if domain is None:
        return False, "No FluidSim domain is assigned"
    sources = [
        obj
        for obj in getattr(scene, "objects", ())
        if getattr(getattr(obj, "fluidsim", None), "role", "NONE") == "SOURCE"
        and bool(getattr(getattr(obj, "fluidsim", None), "enabled", True))
    ]
    if not sources and not bool(getattr(fs, "height_enabled", False)):
        return False, "No enabled liquid source or height fill is configured"
    return True, "Scene setup is valid"


def _select_scene(bpy, name: str):
    if name:
        scene = bpy.data.scenes.get(name)
        if scene is None:
            raise ValueError(f"scene {name!r} does not exist")
        bpy.context.window.scene = scene
        return scene
    return bpy.context.scene


def _report_path(options: CliOptions) -> Path:
    return options.report_json or options.cache / "fluidsim_bake_report.json"


def run(options: CliOptions) -> int:
    """Execute production baking and return a stable process exit code."""
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": _utc_now(),
        "options": {
            **asdict(options),
            "cache": str(options.cache),
            "report_json": str(options.report_json) if options.report_json else "",
        },
        "status": "failed",
        "exit_code": EXIT_BAKE_FAILURE,
        "message": "Bake did not start",
        "frames_committed": 0,
    }
    code = EXIT_BAKE_FAILURE
    try:
        import bpy

        from .effects import scene_effect_instances, trust_store_for_scene
        from ..sim.effects.policy import evaluate_custom_code_policy
        scene = _select_scene(bpy, options.scene)
        policy_decision = evaluate_custom_code_policy(
            scene_effect_instances(scene),
            options.custom_code_policy,
            trust_store_for_scene(scene),
        )
        report["custom_code"] = asdict(policy_decision)
        if not policy_decision.allowed:
            code = EXIT_CUSTOM_CODE_POLICY
            report["message"] = f"{policy_decision.code}: {policy_decision.message}"
            return code

        from ..runtime.warp_runtime import EXPECTED_WARP_VERSION, probe_runtime
        from .production_bake import (
            BakeController,
            BakeStatus,
            ProductionBakeBackend,
            SignalCancellation,
        )
        from ..io.cache_schema import load_manifest, manifest_path

        report["blender"] = {
            "version": str(bpy.app.version_string),
            "python": sys.version.split()[0],
        }
        probe = probe_runtime(force=True)
        report["runtime"] = {
            "warp_version": probe.warp_version,
            "module_file": probe.module_file,
            "cuda_driver_version": probe.cuda_driver_version,
            "cuda_toolkit_version": probe.cuda_toolkit_version,
            "cuda_devices": [asdict(item) for item in probe.cuda_devices],
            "message": probe.message,
        }
        if not probe.warp_version or probe.warp_version != EXPECTED_WARP_VERSION:
            code = EXIT_RUNTIME_IMPORT
            report["message"] = probe.message
            return code
        if not probe.cuda_devices:
            code = EXIT_NO_CUDA
            report["message"] = probe.message
            return code
        aliases = tuple(device.alias for device in probe.cuda_devices)
        selected_device = options.device or aliases[0]
        if selected_device not in aliases:
            code = EXIT_DEVICE_UNAVAILABLE
            report["message"] = (
                f"Requested CUDA device {selected_device!r} is unavailable; "
                f"detected {', '.join(aliases)}"
            )
            return code
        report["runtime"]["selected_device"] = selected_device

        valid, message = _scene_validation(scene)
        if not valid:
            code = EXIT_SCENE_INVALID
            report["message"] = message
            return code
        fs = scene.fluidsim
        fs.cache_path = str(options.cache)
        if options.frame_start is not None:
            fs.bake_start = options.frame_start
        if options.frame_end is not None:
            fs.bake_end = options.frame_end
        fs.checkpoint_interval = options.checkpoint_interval
        start = int(fs.bake_start)
        end = int(fs.bake_end)
        if end < start:
            code = EXIT_ARGUMENTS
            report["message"] = f"Bake end {end} is before start {start}"
            return code

        # Bind the selected alias to the installed extension preferences where
        # possible; the solver settings bridge reads this same preference.
        try:
            prefs = bpy.context.preferences.addons[__package__.split(".blender")[0]].preferences
            prefs.device = selected_device
        except (KeyError, AttributeError, TypeError):
            pass

        backend = ProductionBakeBackend(
            checkpoint_interval=options.checkpoint_interval,
            compressed=bool(getattr(fs, "cache_compress", False)),
        )
        controller = BakeController(backend=backend, progress_callback=_emit)
        existing = load_manifest(manifest_path(options.cache))
        if options.resume == "required":
            decision = backend.resume_decision(scene)
            if not decision.allowed:
                code = EXIT_RESUME_INCOMPATIBLE
                report["message"] = f"{decision.code}: {decision.message}"
                return code
            controller.resume(scene, end=end)
        elif options.resume == "auto" and existing is not None:
            decision = backend.resume_decision(scene)
            if not decision.allowed:
                code = EXIT_RESUME_INCOMPATIBLE
                report["message"] = f"{decision.code}: {decision.message}"
                return code
            controller.resume(scene, end=end)
        else:
            if options.resume == "never" and existing is not None:
                code = EXIT_RESUME_INCOMPATIBLE
                report["message"] = (
                    "A cache-v2 manifest already exists and --resume=never will not overwrite it"
                )
                return code
            controller.start(scene, start, end, restart=False)

        with SignalCancellation(controller):
            while controller.status in {
                BakeStatus.INITIALIZING,
                BakeStatus.RUNNING,
                BakeStatus.CANCELLING,
            }:
                controller.tick(scene)

        result = controller.result()
        report["frames_committed"] = (
            0
            if result.last_committed_frame is None
            else result.last_committed_frame - result.frame_start + 1
        )
        report["last_committed_frame"] = result.last_committed_frame
        report["cache"] = str(options.cache)
        if result.status is BakeStatus.CANCELLED:
            code = EXIT_CANCELLED
            report["message"] = result.message
            return code
        if result.status is not BakeStatus.COMPLETE:
            code = EXIT_BAKE_FAILURE
            report["message"] = controller.error or result.message
            return code
        if options.save_after_bake:
            bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
        code = EXIT_SUCCESS
        report["status"] = "success"
        report["message"] = result.message
        return code
    except ValueError as exc:
        code = EXIT_SCENE_INVALID
        report["message"] = f"{type(exc).__name__}: {exc}"
        return code
    except Exception as exc:  # Blender log retains the traceback via stderr
        import traceback

        traceback.print_exc()
        code = EXIT_BAKE_FAILURE
        report["message"] = f"{type(exc).__name__}: {exc}"
        return code
    finally:
        report["exit_code"] = code
        report["finished_at"] = _utc_now()
        report["elapsed_seconds"] = max(0.0, time.monotonic() - started)
        try:
            write_json_atomic(_report_path(options), report)
        except Exception as report_exc:
            print(
                f"FluidSim could not write final JSON report: {report_exc}",
                file=sys.stderr,
                flush=True,
            )
        _emit(
            {
                "event": "bake_finished",
                "exit_code": code,
                "status": report["status"],
                "message": report["message"],
            }
        )


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[sys.argv.index("--") + 1 :] if argv is None and "--" in sys.argv else (argv or [])))
