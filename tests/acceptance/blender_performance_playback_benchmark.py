"""Measure real FluidSim GUI playback through its registered frame handler.

Run in foreground Blender, never with ``--background``::

    blender --factory-startup --python tests/acceptance/blender_performance_playback_benchmark.py -- \
        --addon-root C:/path/to/source-checkout --expected-head <sha> \
        --expected-module-root C:/isolated/fluidSim_warp_blender51 \
        --owned-process-pid <pid> --blend demos/dam_break.blend \
        --pair-id pair-01 --order baseline-first --abba-slot A1 \
        --frames 60 --surface-attributes default --json-out artifacts/handler.json

One report is one raw trial. Approval still requires paired fresh-process trials.
The harness never calls the solver directly and never changes the scene's saved
``live_surface_publish_cadence``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

try:  # Keep argument/report helpers importable under system Python.
    import bpy
except ModuleNotFoundError:  # pragma: no cover - exercised by system tests
    bpy = None


MEASUREMENT_KIND = "handler_driven_gui_playback"
REPORT_SCHEMA = "fluidsim-handler-playback-evidence-v2"
HANDLER_IDENTITY = "fluidSim_warp_blender51.blender.playback._on_frame_change"
ADDON_NAME = "fluidSim_warp_blender51"
SURFACE_ATTRIBUTE_MODES = ("off", "default", "all")
PAIR_IDS = tuple(f"pair-{index:02d}" for index in range(1, 8))
ABBA_SLOTS = ("A1", "B1", "B2", "A2")


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--addon-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-module-root", required=True)
    parser.add_argument("--owned-process-pid", type=int, required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--pair-id", choices=PAIR_IDS, required=True)
    parser.add_argument(
        "--order", choices=("baseline-first", "candidate-first"),
        required=True)
    parser.add_argument("--abba-slot", choices=ABBA_SLOTS, required=True)
    parser.add_argument(
        "--surface-attributes", choices=SURFACE_ATTRIBUTE_MODES,
        default="default",
        help="required preconfigured surface attribute inventory",
    )
    args = parser.parse_args(argv)
    if args.frames < 60:
        parser.error("--frames must be at least 60 for approval-grade GUI evidence")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive and finite")
    return args


def cadence_frame_budget(required_handler_samples, required_surface_samples, cadence):
    """Return advances guaranteeing both sample counts from any cadence phase."""
    values = (required_handler_samples, required_surface_samples, cadence)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("sample counts and cadence must be positive integers")
    # The extra cadence-1 advances cover an unknown initial solver-frame phase.
    surface_budget = required_surface_samples * cadence + cadence - 1
    return max(required_handler_samples, surface_budget)


def resolve_target_root(value):
    """Resolve a checkout root and require the benchmark's target contracts."""
    root = Path(value).expanduser().resolve(strict=True)
    required = (
        root / ADDON_NAME / "__init__.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            f"--addon-root is not a FluidSim checkout; missing: {missing}")
    return root


def resolve_expected_module_root(value):
    root = Path(value).expanduser().resolve(strict=True)
    if root.name != ADDON_NAME or not (root / "__init__.py").is_file():
        raise ValueError(
            "--expected-module-root must be the isolated copied addon package")
    return root


def module_is_under_target(module_file, expected_module_root):
    if not module_file:
        return False
    try:
        Path(module_file).resolve(strict=True).relative_to(
            Path(expected_module_root).resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _git_provenance(target_root):
    result = {
        "git_available": False,
        "git_head": None,
        "git_tree_dirty": None,
        "git_status_porcelain": None,
    }
    try:
        head = subprocess.run(
            ["git", "-C", str(target_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10)
        status = subprocess.run(
            ["git", "-C", str(target_root), "status", "--porcelain=v1"],
            check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return result
    porcelain = status.stdout.rstrip("\r\n")
    result.update({
        "git_available": True,
        "git_head": head.stdout.strip(),
        "git_tree_dirty": bool(porcelain),
        "git_status_porcelain": porcelain.splitlines(),
    })
    return result


def _trial_variant(order, abba_slot):
    first = "baseline" if order == "baseline-first" else "candidate"
    second = "candidate" if first == "baseline" else "baseline"
    return first if abba_slot.startswith("A") else second


def _protected_finalize(*, restore, build_report, write_report):
    """Attempt restoration, then build and write one raw-trial report."""
    errors = []
    try:
        restore()
    except Exception as exc:  # noqa: BLE001 - finalization must continue
        errors.append(f"restoration failed: {exc!r}")
    try:
        report = build_report(tuple(errors))
    except Exception as exc:  # noqa: BLE001 - retain a writable failure report
        errors.append(f"report construction failed: {exc!r}")
        report = {
            "schema": REPORT_SCHEMA,
            "measurement_kind": MEASUREMENT_KIND,
            "approval_eligible": False,
            "external_paired_comparator_required": True,
            "error": "; ".join(errors),
            "passed": False,
        }
    try:
        write_report(report)
    except Exception as exc:  # noqa: BLE001 - quit still belongs to this trial
        errors.append(f"report write failed: {exc!r}")
    return report, tuple(errors)


def _runtime_inventory(source_root, module_root):
    """Bind Git-tracked addon source bytes to the isolated runtime copy."""
    result = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "-z", "--", ADDON_NAME],
        check=True, capture_output=True, timeout=20)
    raw = result.stdout if isinstance(result.stdout, bytes) else result.stdout.encode()
    tracked = sorted(
        (Path(value.decode("utf-8")) for value in raw.split(b"\0") if value),
        key=lambda value: value.as_posix())
    if not tracked:
        raise RuntimeError("Git tracked addon inventory is empty")
    digest = hashlib.sha256()
    missing = []
    mismatched = []
    for repo_relative in tracked:
        package_relative = repo_relative.relative_to(ADDON_NAME)
        source_path = source_root / repo_relative
        runtime_path = module_root / package_relative
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        digest.update(repo_relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_hash.encode("ascii"))
        digest.update(b"\n")
        if not runtime_path.is_file():
            missing.append(repo_relative.as_posix())
        elif hashlib.sha256(runtime_path.read_bytes()).hexdigest() != source_hash:
            mismatched.append(repo_relative.as_posix())
    return {
        "runtime_inventory_digest": digest.hexdigest(),
        "runtime_inventory_count": len(tracked),
        "runtime_inventory_missing": missing,
        "runtime_inventory_mismatched": mismatched,
        "runtime_inventory_matches": not missing and not mismatched,
    }


def evaluate_playback_report(report, *, required_frames=60):
    """Recompute live truth checks; never trust a stored verdict."""
    all_samples = report.get("all_handler_frame_samples")
    surface_samples = report.get("surface_publication_frame_samples")

    def valid_samples(samples):
        return (
            isinstance(samples, list)
            and len(samples) >= required_frames
            and all(
                type(value) in (int, float)
                and math.isfinite(value) and value > 0.0
                for value in samples
            )
        )

    requested = report.get("requested_surface_attributes")
    published = report.get("published_surface_attributes")
    attribute_contract = (
        isinstance(requested, list)
        and isinstance(published, list)
        and set(requested).issubset(published)
    )
    if report.get("surface_attribute_profile") == "off":
        attribute_contract = requested == [] and published == []

    checks = {
        "report_schema": report.get("schema") == REPORT_SCHEMA,
        "measurement_kind": report.get("measurement_kind") == MEASUREMENT_KIND,
        "single_trial_not_approval": report.get("approval_eligible") is False,
        "external_comparator_required": (
            report.get("external_paired_comparator_required") is True),
        "required_frames_declared": report.get("required_frames") == required_frames,
        "target_root_validated": report.get("target_root_validated") is True,
        "git_available": report.get("git_available") is True,
        "expected_git_head": (
            bool(report.get("expected_git_head"))
            and report.get("git_head") == report.get("expected_git_head")),
        "source_tree_clean": report.get("git_tree_dirty") is False,
        "addon_enabled": report.get("addon_enabled") is True,
        "scene_registration_marker": (
            report.get("scene_registration_marker") is True),
        "imported_from_expected_module_root": (
            report.get("imported_from_expected_module_root") is True),
        "runtime_inventory_matches": (
            report.get("runtime_inventory_matches") is True),
        "active_blend_matches": report.get("active_blend_matches") is True,
        "process_pid_matches": report.get("process_pid_matches") is True,
        "launcher_owns_cleanup": (
            report.get("cleanup_owner") == "blender_mcp_launcher"),
        "registered_handler_unique": report.get("registered_handler_count") == 1,
        "registered_handler_identity": (
            report.get("registered_handler_identity") == HANDLER_IDENTITY),
        "all_handler_samples": valid_samples(all_samples),
        "surface_publication_samples": valid_samples(surface_samples),
        "handler_invoked": report.get("handler_invocations", 0) >= required_frames,
        "depsgraph_delivered": report.get("depsgraph_deliveries", 0) >= required_frames,
        "handler_steps": report.get("step_calls_inside_handler", 0) >= required_frames,
        "surface_published": report.get("surface_publications", 0) >= required_frames,
        "attribute_contract": attribute_contract,
        "no_direct_step": report.get("direct_step_calls") == 0,
        "viewport_non_empty": report.get("viewport_non_empty") is True,
        "viewport_changed": report.get("viewport_changed") is True,
        "session_active": report.get("session_active") is True,
        "no_error": not report.get("error"),
    }
    return {
        **report,
        "approval_eligible": False,
        "external_paired_comparator_required": True,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _viewport_sample(output_publication):
    surface = bpy.data.objects.get("FluidSim Surface")
    if surface is not None and surface.type == "MESH":
        count = len(surface.data.vertices)
        if count > 0:
            sample = tuple(
                round(float(value), 6)
                for item in surface.data.vertices[:min(32, count)]
                for value in item.co
            )
            return sample, count, "surface"
    particles = bpy.data.objects.get("FluidSim Particles")
    count = output_publication.live_point_count(particles)
    if particles is None or particles.type != "POINTCLOUD" or count <= 0:
        return (), 0, "none"
    position = particles.data.attributes.get("position")
    if position is None:
        return (), 0, "none"
    sample = tuple(
        round(float(value), 6)
        for item in position.data[:min(32, count)]
        for value in item.vector
    )
    return sample, count, "particles"


def _script_args():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _write_json_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def _failure_report(args, *, error, provenance=None, extra=None):
    report = {
        "schema": REPORT_SCHEMA,
        "measurement_kind": MEASUREMENT_KIND,
        "approval_eligible": False,
        "external_paired_comparator_required": True,
        "required_frames": args.frames,
        "target_addon_root": str(Path(args.addon_root).expanduser().resolve()),
        "target_root_validated": False,
        "expected_git_head": args.expected_head,
        "expected_module_root": str(
            Path(args.expected_module_root).expanduser().resolve()),
        "pair_id": args.pair_id,
        "order": args.order,
        "abba_slot": args.abba_slot,
        "variant": _trial_variant(args.order, args.abba_slot),
        "expected_process_pid": args.owned_process_pid,
        "owned_process_pid": os.getpid(),
        "process_pid_matches": args.owned_process_pid == os.getpid(),
        "cleanup_owner": "blender_mcp_launcher",
        "surface_attribute_profile": args.surface_attributes,
        "all_handler_frame_samples": [],
        "surface_publication_frame_samples": [],
        "direct_step_calls": 0,
        "error": error,
    }
    if provenance:
        report.update(provenance)
    if extra:
        report.update(extra)
    return evaluate_playback_report(report, required_frames=args.frames)


def main(argv=None):
    if bpy is None:
        raise RuntimeError("this benchmark must run inside foreground Blender")
    args = _parse_args(_script_args() if argv is None else argv)
    report_path = Path(args.json_out).expanduser().resolve()
    owns_process = args.owned_process_pid == os.getpid()
    provenance = None
    inventory = None
    instrumentation_restore = lambda: None

    def setup_failure(error, extra=None):
        def build(finalization_errors):
            combined = [error, *finalization_errors]
            return _failure_report(
                args, error="; ".join(filter(None, combined)),
                provenance=provenance, extra=extra)

        report, finalization_errors = _protected_finalize(
            restore=instrumentation_restore,
            build_report=build,
            write_report=lambda value: _write_json_report(report_path, value),
        )
        print("PERFORMANCE-PLAYBACK-RAW-TRIAL-FAIL", flush=True)
        return report, finalization_errors

    try:
        if not owns_process:
            raise RuntimeError(
                f"owned process PID mismatch: expected {args.owned_process_pid}, "
                f"actual {os.getpid()}")
        target_root = resolve_target_root(args.addon_root)
        expected_module_root = resolve_expected_module_root(
            args.expected_module_root)
        blend_path = Path(args.blend).expanduser().resolve(strict=True)
        provenance = _git_provenance(target_root)
        if not provenance["git_available"]:
            raise RuntimeError("source checkout Git provenance is unavailable")
        if provenance["git_head"] != args.expected_head:
            raise RuntimeError(
                f"source HEAD {provenance['git_head']} does not match "
                f"expected {args.expected_head}")
        if provenance["git_tree_dirty"]:
            raise RuntimeError("source checkout is dirty")
        inventory = _runtime_inventory(target_root, expected_module_root)
        if not inventory["runtime_inventory_matches"]:
            raise RuntimeError(
                "isolated runtime package differs from tracked source: "
                f"missing={inventory['runtime_inventory_missing']}, "
                f"mismatched={inventory['runtime_inventory_mismatched']}")

        import addon_utils

        addon = sys.modules.get(ADDON_NAME)
        if addon is None:
            raise RuntimeError(
                "isolated addon is not preloaded; the harness will not load it")
        addon_enabled = bool(addon_utils.check(ADDON_NAME)[1])
        imported_from_expected_root = module_is_under_target(
            getattr(addon, "__file__", None), expected_module_root)
        if not addon_enabled:
            raise RuntimeError("isolated addon is not enabled")
        if not imported_from_expected_root:
            raise RuntimeError(
                f"{ADDON_NAME} imported from {getattr(addon, '__file__', None)}, "
                f"not {expected_module_root}")

        from fluidSim_warp_blender51.blender import (
            output_publication,
            playback,
            scene_bridge,
        )
        from fluidSim_warp_blender51.blender.scene_access import (
            active_domain_settings,
        )
        from fluidSim_warp_blender51.sim.attributes import (
            OUTPUT_ATTRIBUTE_KEYS,
            enabled_attribute_keys,
        )

        scene = bpy.context.scene
        active_blend_matches = (
            bool(bpy.data.filepath)
            and Path(bpy.data.filepath).resolve(strict=True) == blend_path)
        if not active_blend_matches:
            raise RuntimeError(
                f"active Blender file {bpy.data.filepath!r} does not match "
                f"expected {blend_path}")
        scene_registration_marker = hasattr(bpy.types.Scene, "fluidsim")
        registered = [
            callback for callback in bpy.app.handlers.frame_change_post
            if callback is playback._on_frame_change
        ]
        if not scene_registration_marker:
            raise RuntimeError("FluidSim Scene registration marker is absent")
        if len(registered) != 1:
            raise RuntimeError(
                "registered FluidSim frame_change_post handler is not unique")

        if args.surface_attributes == "off":
            requested_attributes = []
        elif args.surface_attributes == "default":
            requested_attributes = ["velocity", "motion"]
        else:
            requested_attributes = list(OUTPUT_ATTRIBUTE_KEYS["surface"])
        # The attribute toggles and the publish cadence are domain settings.
        domain_settings = active_domain_settings(scene)
        if domain_settings is None:
            raise RuntimeError("the saved scene has no designated FluidSim domain")
        configured_attributes = list(enabled_attribute_keys(
            domain_settings, "surface"))
        if set(configured_attributes) != set(requested_attributes):
            raise RuntimeError(
                "preconfigured surface attributes do not match requested profile: "
                f"configured={configured_attributes}, "
                f"requested={requested_attributes}")

        surface_cadence = int(domain_settings.live_surface_publish_cadence)
        timeline_budget = cadence_frame_budget(
            args.frames, args.frames, surface_cadence)
        start_frame = int(scene.frame_current)
        scene.frame_end = max(
            int(scene.frame_end), start_frame + timeline_budget + 5)
    except Exception as exc:  # noqa: BLE001 - preserve setup failure evidence
        setup_failure(repr(exc))
        return

    state = {
        "in_handler": False,
        "handler_invocations": 0,
        "depsgraph_deliveries": 0,
        "output_update_calls": 0,
        "payload_publications": 0,
        "surface_publications": 0,
        "step_calls_inside_handler": 0,
        "direct_step_calls": 0,
        "all_handler_frame_samples": [],
        "surface_publication_frame_samples": [],
        "events": [],
        "first_viewport_sample": None,
        "last_viewport_sample": None,
        "viewport_output_count": 0,
        "viewport_output_kind": "none",
        "published_surface_attributes": set(),
    }
    original_handle = playback.handle_frame_change
    original_step_frame = scene_bridge.step_frame
    original_update_outputs = output_publication.update_outputs
    original_prepare_payload = output_publication.prepare_live_payload
    original_publish_payload = output_publication.publish_payload
    started_at = time.monotonic()
    finished = False
    finishing = False

    def measured_update_outputs(*call_args, **call_kwargs):
        state["output_update_calls"] += 1
        return original_update_outputs(*call_args, **call_kwargs)

    def measured_prepare_payload(*call_args, **call_kwargs):
        payload = original_prepare_payload(*call_args, **call_kwargs)
        if payload.publish_surface and payload.surface_visible:
            state["surface_publications"] += 1
            state["published_surface_attributes"].update(
                payload.surface_attributes)
        return payload

    def measured_publish_payload(*call_args, **call_kwargs):
        state["payload_publications"] += 1
        return original_publish_payload(*call_args, **call_kwargs)

    def measured_step_frame(*call_args, **call_kwargs):
        key = "step_calls_inside_handler" if state["in_handler"] else "direct_step_calls"
        state[key] += 1
        return original_step_frame(*call_args, **call_kwargs)

    def measured_handle_frame_change(sess, current_scene, depsgraph):
        state["handler_invocations"] += 1
        if depsgraph is not None:
            state["depsgraph_deliveries"] += 1
        before_surface = state["surface_publications"]
        before_steps = state["step_calls_inside_handler"]
        before_solver_frame = (
            int(sess.solver.frame) if getattr(sess, "active", False) else None)
        state["in_handler"] = True
        started = time.perf_counter()
        try:
            return original_handle(sess, current_scene, depsgraph)
        finally:
            duration = time.perf_counter() - started
            state["in_handler"] = False
            surface_published = state["surface_publications"] > before_surface
            stepped = state["step_calls_inside_handler"] > before_steps
            solver_frame = (
                int(sess.solver.frame) if getattr(sess, "active", False) else None)
            valid_handler_frame = stepped and depsgraph is not None
            event = {
                "scene_frame": int(current_scene.frame_current),
                "solver_frame_before": before_solver_frame,
                "solver_frame_after": solver_frame,
                "duration_seconds": duration,
                "depsgraph_delivered": depsgraph is not None,
                "surface_published": surface_published,
                "stepped_inside_handler": stepped,
            }
            state["events"].append(event)
            if (valid_handler_frame
                    and len(state["all_handler_frame_samples"]) < args.frames):
                state["all_handler_frame_samples"].append(duration)
            if (valid_handler_frame and surface_published
                    and len(state["surface_publication_frame_samples"]) < args.frames):
                state["surface_publication_frame_samples"].append(duration)
                sample, count, kind = _viewport_sample(output_publication)
                state["viewport_output_count"] = count
                state["viewport_output_kind"] = kind
                if state["first_viewport_sample"] is None:
                    state["first_viewport_sample"] = sample
                state["last_viewport_sample"] = sample

    def restore_instrumentation():
        playback.handle_frame_change = original_handle
        scene_bridge.step_frame = original_step_frame
        output_publication.update_outputs = original_update_outputs
        output_publication.prepare_live_payload = original_prepare_payload
        output_publication.publish_payload = original_publish_payload

    instrumentation_restore = restore_instrumentation
    try:
        playback.handle_frame_change = measured_handle_frame_change
        scene_bridge.step_frame = measured_step_frame
        output_publication.update_outputs = measured_update_outputs
        output_publication.prepare_live_payload = measured_prepare_payload
        output_publication.publish_payload = measured_publish_payload
    except Exception as exc:  # noqa: BLE001 - restore partial installation
        setup_failure(repr(exc), {
            "target_root_validated": True,
            "addon_enabled": addon_enabled,
            "scene_registration_marker": scene_registration_marker,
            "imported_from_expected_module_root": imported_from_expected_root,
            **(inventory or {}),
        })
        return

    def finish(error=""):
        nonlocal finished, finishing
        if finished or finishing:
            return None
        finishing = True

        def restore_all():
            if bpy.context.screen and bpy.context.screen.is_animation_playing:
                bpy.ops.screen.animation_cancel(restore_frame=False)
            restore_instrumentation()

        def build(finalization_errors):
            sess = scene_bridge.session()
            first_sample = state["first_viewport_sample"]
            last_sample = state["last_viewport_sample"]
            errors = [error, str(sess.error or ""), *finalization_errors]
            report = {
                "schema": REPORT_SCHEMA,
                "measurement_kind": MEASUREMENT_KIND,
                "approval_eligible": False,
                "external_paired_comparator_required": True,
                "required_frames": args.frames,
                "timeline_advance_budget": timeline_budget,
                "pair_id": args.pair_id,
                "order": args.order,
                "abba_slot": args.abba_slot,
                "variant": _trial_variant(args.order, args.abba_slot),
                "blend": str(blend_path),
                "blend_sha256": hashlib.sha256(
                    blend_path.read_bytes()).hexdigest(),
                "target_addon_root": str(target_root),
                "target_root_validated": True,
                "expected_git_head": args.expected_head,
                "expected_module_root": str(expected_module_root),
                "imported_addon_file": str(Path(addon.__file__).resolve()),
                "imported_from_expected_module_root": imported_from_expected_root,
                "addon_enabled": addon_enabled,
                "scene_registration_marker": scene_registration_marker,
                **inventory,
                "active_blend_matches": active_blend_matches,
                **provenance,
                "blender_binary": bpy.app.binary_path,
                "blender_version": bpy.app.version_string,
                "python_version": sys.version.split()[0],
                "owned_process_pid": os.getpid(),
                "expected_process_pid": args.owned_process_pid,
                "process_pid_matches": owns_process,
                "cleanup_owner": "blender_mcp_launcher",
                "surface_attribute_profile": args.surface_attributes,
                "requested_surface_attributes": sorted(requested_attributes),
                "configured_surface_attributes": sorted(configured_attributes),
                "published_surface_attributes": sorted(
                    state["published_surface_attributes"]),
                "live_surface_publish_cadence": surface_cadence,
                "start_frame": start_frame,
                "end_frame": int(scene.frame_current),
                "elapsed_seconds": time.monotonic() - started_at,
                "registered_handler_count": len(registered),
                "registered_handler_identity": HANDLER_IDENTITY,
                "handler_invocations": state["handler_invocations"],
                "depsgraph_deliveries": state["depsgraph_deliveries"],
                "output_update_calls": state["output_update_calls"],
                "payload_publications": state["payload_publications"],
                "surface_publications": state["surface_publications"],
                "step_calls_inside_handler": state["step_calls_inside_handler"],
                "direct_step_calls": state["direct_step_calls"],
                "all_handler_frame_samples": state["all_handler_frame_samples"],
                "surface_publication_frame_samples": (
                    state["surface_publication_frame_samples"]),
                "events": state["events"],
                "viewport_output_count": state["viewport_output_count"],
                "viewport_output_kind": state["viewport_output_kind"],
                "viewport_non_empty": bool(first_sample),
                "viewport_changed": bool(
                    first_sample and last_sample and first_sample != last_sample),
                "session_active": bool(sess.active),
                "solver_frame": int(sess.solver.frame) if sess.active else None,
                "device": str(sess.solver.device) if sess.active else None,
                "error": "; ".join(value for value in errors if value),
            }
            return evaluate_playback_report(report, required_frames=args.frames)

        report, finalization_errors = _protected_finalize(
            restore=restore_all,
            build_report=build,
            write_report=lambda value: _write_json_report(report_path, value),
        )
        finished = True
        finishing = False
        marker = (
            "PERFORMANCE-PLAYBACK-RAW-TRIAL-PASS"
            if report.get("passed") and not finalization_errors
            else "PERFORMANCE-PLAYBACK-RAW-TRIAL-FAIL")
        print(marker, flush=True)
        return None

    def poll():
        enough_handler = len(state["all_handler_frame_samples"]) >= args.frames
        enough_surface = (
            len(state["surface_publication_frame_samples"]) >= args.frames)
        if enough_handler and enough_surface:
            return finish()
        if time.monotonic() - started_at > args.timeout_seconds:
            return finish("timed out before collecting required handler/surface frames")
        if (state["step_calls_inside_handler"] >= timeline_budget
                and not enough_surface):
            return finish("timeline budget exhausted before required surface publications")
        return 0.1

    def start():
        try:
            window = bpy.context.window_manager.windows[0]
            area = next(
                (candidate for candidate in window.screen.areas
                 if candidate.type == "VIEW_3D"), None)
            if area is None:
                return finish("no VIEW_3D area available")
            region = next(
                candidate for candidate in area.regions if candidate.type == "WINDOW")
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.screen.animation_play()
            return None
        except Exception as exc:  # noqa: BLE001 - report live setup failure
            return finish(repr(exc))

    bpy.app.timers.register(poll, first_interval=0.2)
    bpy.app.timers.register(start, first_interval=0.1)


if __name__ == "__main__":
    main()
