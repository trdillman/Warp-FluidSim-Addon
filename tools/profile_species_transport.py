#!/usr/bin/env python3
"""Profile disabled, D=0 transport, and two-substep species diffusion."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUDGETS = {
    "transport_only_overhead_fraction": 0.12,
    "diffusion_on_overhead_fraction": 0.35,
    "allocation_relative_error": 0.05,
    "substep_host_synchronizations": 0,
}

CASE_NAMES = ("disabled", "transport_only", "diffusion_on")
PRODUCTION_MINIMUMS = {"frames": 60, "warmup": 15, "particles": 200_000}
# Decision receipts are intentionally short-lived: validation after one hour
# requires a fresh run against the current checkout and profiler bytes.
MAX_EVIDENCE_AGE_SECONDS = 60 * 60
MAX_FUTURE_CLOCK_SKEW_SECONDS = 60
CHECK_KEYS = frozenset({
    "disabled_zero_species_resources",
    "transport_only_overhead",
    "diffusion_on_overhead",
    "diffusion_on_k2",
    "allocation_matches_admission",
    "no_substep_host_synchronization",
    "timed_samples_uninstrumented",
    "adjacent_disabled_brackets",
    "raw_sample_counts",
    "cuda_runtime",
})
METRIC_KEYS = frozenset({
    "disabled_median_ms",
    "transport_only_overhead_fraction",
    "diffusion_on_overhead_fraction",
    "transport_only_p95_delta_ms",
    "diffusion_on_p95_delta_ms",
    "admitted_species_device_bytes",
    "admitted_species_cache_staging_bytes",
    "measured_species_peak_increment_bytes",
    "allocation_relative_error",
})

SOURCE_AUDIT_PATHS = {
    "fluidSim_warp_blender51/sim/cuda_reduction.py":
        ROOT / "fluidSim_warp_blender51" / "sim" / "cuda_reduction.py",
    "fluidSim_warp_blender51/sim/fields.py":
        ROOT / "fluidSim_warp_blender51" / "sim" / "fields.py",
    "fluidSim_warp_blender51/sim/particles.py":
        ROOT / "fluidSim_warp_blender51" / "sim" / "particles.py",
    "fluidSim_warp_blender51/sim/species.py":
        ROOT / "fluidSim_warp_blender51" / "sim" / "species.py",
}
RUNTIME_SOURCE_SUFFIXES = frozenset({
    ".py", ".cu", ".c", ".cpp", ".h", ".hpp", ".cl", ".json", ".toml"})
RUNTIME_TREE_EXCLUDED_PARTS = frozenset({
    ".codex", ".git", ".pytest_cache", "__pycache__", "artifacts",
    "demos", "receipts"})
RUNTIME_EXPLICIT_DEPENDENCIES = (
    Path("tools/profile_species_transport.py"),
    Path("tools/run_species_transport_profile_repetitions.py"),
)


def _utc_now():
    return datetime.now(timezone.utc)


def _normalized_dependency_path(value):
    path = Path(value)
    normalized = path.as_posix().lstrip("./")
    if not normalized or normalized == "." or normalized.startswith("../"):
        raise ValueError(f"invalid runtime dependency path: {value}")
    return normalized


def _runtime_dependency_tree_sha256(
        *, root=ROOT, tracked_relative_paths=None,
        explicit_relative_paths=RUNTIME_EXPLICIT_DEPENDENCIES):
    """Hash normalized runtime-source paths, bytes, and tracked deletions."""
    root = Path(root).resolve()
    if tracked_relative_paths is None:
        tracked_output = subprocess.run(
            ["git", "ls-files", "--cached", "-z", "--",
             "fluidSim_warp_blender51"],
            cwd=root, check=True, capture_output=True).stdout
        tracked_relative_paths = tuple(
            item.decode("utf8") for item in tracked_output.split(b"\0") if item)

    tracked = {
        _normalized_dependency_path(path) for path in tracked_relative_paths}
    explicit = {
        _normalized_dependency_path(path) for path in explicit_relative_paths}
    candidates = set(tracked) | explicit
    package_root = root / "fluidSim_warp_blender51"
    if package_root.is_dir():
        for path in package_root.rglob("*"):
            if path.is_file():
                candidates.add(_normalized_dependency_path(path.relative_to(root)))

    def included(relative):
        path = Path(relative)
        return (path.suffix.lower() in RUNTIME_SOURCE_SUFFIXES
                and not any(part.lower() in RUNTIME_TREE_EXCLUDED_PARTS
                            for part in path.parts))

    digest = hashlib.sha256(b"fluidsim-runtime-dependency-tree-v1\0")
    for relative in sorted(path for path in candidates if included(path)):
        encoded_path = relative.encode("utf8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        absolute = root / Path(relative)
        if absolute.is_file():
            content = absolute.read_bytes()
            digest.update(b"F")
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        elif relative in tracked or relative in explicit:
            digest.update(b"D")
    return digest.hexdigest()


def _current_evidence_identity():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    return {
        "git_head": head,
        "runtime_dependency_tree_sha256": _runtime_dependency_tree_sha256(),
        "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _launch_inventory(tree):
    launches = Counter()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and _call_name(node.func) == "launch"
                and node.args):
            launches[_call_name(node.args[0]) or "<dynamic>"] += 1
    return dict(sorted(launches.items()))


def _function_node(tree, name):
    return next(
        (node for node in tree.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == name),
        None)


def _is_candidate_lt_one_guard(test):
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare):
            continue
        left = ast.unparse(node.left).lower()
        for operation, comparator in zip(node.ops, node.comparators):
            if (isinstance(operation, ast.Lt)
                    and isinstance(comparator, ast.Constant)
                    and comparator.value == 1
                    and ("candidate" in left or "alpha" in left)):
                return True
    return False


def _is_tid_zero_guard(test):
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare):
            continue
        values = [node.left, *node.comparators]
        if (any(isinstance(value, ast.Name) and value.id == "tid"
                for value in values)
                and any(isinstance(value, ast.Constant) and value.value == 0
                        for value in values)
                and any(isinstance(operation, ast.Eq)
                        for operation in node.ops)):
            return True
    return False


def _atomic_calls_guarded(function, call_matches, guard_matches):
    if function is None:
        return False
    parents = {
        child: parent for parent in ast.walk(function)
        for child in ast.iter_child_nodes(parent)}
    calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call) and call_matches(node)]
    if not calls:
        return False
    for call in calls:
        current = parents.get(call)
        guarded = False
        while current is not None and current is not function:
            if isinstance(current, ast.If) and guard_matches(current.test):
                guarded = True
                break
            current = parents.get(current)
        if not guarded:
            return False
    return True


def _source_ir_hotpath_audit():
    """Return a deterministic source/AST snapshot of known species hot paths."""
    source_bytes = {
        name: path.read_bytes() for name, path in SOURCE_AUDIT_PATHS.items()}
    sources = {
        name: content.decode("utf8") for name, content in source_bytes.items()
    }
    trees = {name: ast.parse(source) for name, source in sources.items()}
    species_name = "fluidSim_warp_blender51/sim/species.py"
    particles_name = "fluidSim_warp_blender51/sim/particles.py"
    fields_name = "fluidSim_warp_blender51/sim/fields.py"
    cuda_name = "fluidSim_warp_blender51/sim/cuda_reduction.py"
    species_tree = trees[species_name]
    particles_tree = trees[particles_name]
    fields_tree = trees[fields_name]
    species_launches = _launch_inventory(species_tree)
    particle_launches = _launch_inventory(particles_tree)

    generic_optional_loops = 0
    for node in ast.walk(particles_tree):
        if not isinstance(node, ast.For):
            continue
        iteration = ast.unparse(node.iter)
        launched = {
            _call_name(call.args[0])
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and _call_name(call.func) == "launch"
            and call.args
        }
        if (iteration == "self.float_attrs"
                and "k_compact_float_scatter" in launched):
            generic_optional_loops += 1
        if (iteration == "self.vec4_attrs"
                and "k_compact_vec4_scatter" in launched):
            generic_optional_loops += 1

    direct_writes = 0
    atomic_mins = 0
    for node in ast.walk(species_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "global_alpha"):
                    direct_writes += 1
        if (isinstance(node, ast.Call) and _call_name(node.func) == "atomic_min"
                and node.args and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "global_alpha"):
            atomic_mins += 1

    def global_alpha_atomic(call):
        return (_call_name(call.func) == "atomic_min" and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "global_alpha")

    def min_alpha_receipt_atomic(call):
        return (_call_name(call.func) == "atomic_min" and len(call.args) >= 2
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "receipt"
                and isinstance(call.args[1], ast.Name)
                and call.args[1].id == "MIN_ALPHA")

    g2p_guarded = _atomic_calls_guarded(
        _function_node(species_tree, "k_species_g2p_delta"),
        global_alpha_atomic, _is_candidate_lt_one_guard)
    particle_alpha_guarded = _atomic_calls_guarded(
        _function_node(species_tree, "k_species_particle_alpha"),
        global_alpha_atomic, _is_candidate_lt_one_guard)
    delta_apply_guarded = _atomic_calls_guarded(
        _function_node(species_tree, "k_species_delta_apply"),
        min_alpha_receipt_atomic, _is_tid_zero_guard)

    assigned_attributes = Counter(
        target.attr
        for node in ast.walk(species_tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if (isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self")
    )
    neighbor_function = _function_node(fields_tree, "neighbor_flat_index")
    neighbor_calls = Counter(
        _call_name(node.func) for node in ast.walk(neighbor_function)
        if isinstance(node, ast.Call)) if neighbor_function else Counter()
    interior_fast_path = False
    if neighbor_function is not None:
        for condition in ast.walk(neighbor_function):
            if not isinstance(condition, ast.If):
                continue
            for statement in condition.body:
                if not isinstance(statement, ast.Return) or statement.value is None:
                    continue
                rendered = ast.unparse(statement.value)
                if "-1" not in rendered and "voxel_flat_index" not in rendered:
                    interior_fast_path = True

    specialized = particle_launches.get("k_species_compact_alive_scatter", 0)
    compact_function = next(
        (node for node in ast.walk(particles_tree)
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == "compact"),
        None)
    fused_species_channels = set()
    if compact_function is not None:
        for condition in ast.walk(compact_function):
            if not isinstance(condition, ast.If):
                continue
            rendered = ast.unparse(condition.test)
            if ("species_sink_ledger is not None" not in rendered
                    or not any(isinstance(node, ast.Continue)
                               for statement in condition.body
                               for node in ast.walk(statement))):
                continue
            for channel in ("represented_volume", "species_concentration"):
                if repr(channel) in rendered or f'"{channel}"' in rendered:
                    fused_species_channels.add(channel)
    duplicate_species_paths = (
        sum(channel not in fused_species_channels for channel in (
            "represented_volume", "species_concentration"))
        if specialized else 0)
    species_scatter = _function_node(
        particles_tree, "k_species_compact_alive_scatter")

    def scatter_receipt_atomics(operation):
        if species_scatter is None:
            return 0
        return sum(
            1 for node in ast.walk(species_scatter)
            if isinstance(node, ast.Call)
            and _call_name(node.func) == operation
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "receipt")

    cuda_source = sources[cuda_name]
    return {
        "method": "python-ast-source-v1",
        "source_sha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in source_bytes.items()
        },
        "compaction": {
            "specialized_species_scatter_launch_sites": specialized,
            "generic_optional_scatter_loop_sites": generic_optional_loops,
            "fused_species_channels": sorted(fused_species_channels),
            "duplicate_species_scatter_paths": duplicate_species_paths,
            "duplicate_species_scatter_active": bool(duplicate_species_paths),
            "scatter_receipt_atomic_add_sites": scatter_receipt_atomics(
                "atomic_add"),
            "scatter_receipt_atomic_min_sites": scatter_receipt_atomics(
                "atomic_min"),
            "scatter_receipt_atomic_max_sites": scatter_receipt_atomics(
                "atomic_max"),
            "native_postcompact_reducer_declared": (
                "fluid_species_postcompact_reduce" in cuda_source),
            "native_device_count_pointer": "const int *count" in cuda_source,
            "native_float_cas_sites": cuda_source.count("atomicCAS"),
        },
        "limiter": {
            "global_alpha_direct_write_sites": direct_writes,
            "global_alpha_atomic_min_sites": atomic_mins,
            "g2p_global_alpha_atomic_candidate_lt_one_guarded": g2p_guarded,
            "particle_alpha_atomic_candidate_lt_one_guarded": (
                particle_alpha_guarded),
            "delta_apply_min_alpha_receipt_tid_zero_guarded": (
                delta_apply_guarded),
            "init_alpha_kernel_declared": (
                _function_node(species_tree, "_k_init_alpha") is not None),
            "init_alpha_launch_sites": species_launches.get("_k_init_alpha", 0),
            "particle_alpha_launch_sites": species_launches.get(
                "k_species_particle_alpha", 0),
            "direct_write_tid_zero_guard_present": (
                "if tid == 0:\n        global_alpha[0] = 1.0"
                in sources[species_name]),
        },
        "scratch": {
            "face_delta_declared": assigned_attributes["face_delta"] > 0,
            "face_delta_declaration_sites": assigned_attributes["face_delta"],
            "face_conductance_declared": (
                assigned_attributes["face_conductance"] > 0),
            "face_conductance_declaration_sites": assigned_attributes[
                "face_conductance"],
        },
        "neighbor": {
            "interior_fast_path_present": interior_fast_path,
            "global_voxel_coord_reconstruction_calls": neighbor_calls[
                "voxel_coord_of_flat"],
            "global_leaf_table_lookup_calls": neighbor_calls[
                "voxel_flat_index"],
            "positive_face_neighbor_call_sites_in_species": sum(
                1 for node in ast.walk(species_tree)
                if isinstance(node, ast.Call)
                and _call_name(node.func) == "positive_face_neighbor"),
        },
    }


def _argv_after_separator(argv):
    return argv[argv.index("--") + 1:] if "--" in argv else argv[1:]


def _parse_size(text):
    match = re.search(r"([0-9.]+)\s+(B|KB|MB|GB|KiB|MiB|GiB)", text)
    if not match:
        return 0
    scale = {
        "B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3,
        "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3,
    }
    return int(float(match.group(1)) * scale[match.group(2)])


def _peak_cuda_bytes(report, device_alias="cuda:0"):
    match = re.search(
        rf"Peak usage:\s*\n\s*{re.escape(str(device_alias))}\s+([^\n]+)",
        report)
    return _parse_size(match.group(1)) if match else 0


def _paired_overhead(value, before, after):
    """Compare a case to the arithmetic mean of adjacent baselines."""
    baseline = (float(before) + float(after)) * 0.5
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("paired baseline must be finite and positive")
    return float(value) / baseline - 1.0


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    position = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _finite_positive_samples(value):
    if not isinstance(value, list) or not value:
        return None
    try:
        samples = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(item) or item <= 0.0 for item in samples):
        return None
    return samples


def _derived_timing(raw_samples):
    if not isinstance(raw_samples, dict) or set(raw_samples) != set(CASE_NAMES):
        raise ValueError("raw timing cases do not match the schema")
    disabled = _finite_positive_samples(raw_samples["disabled"])
    transport = _finite_positive_samples(raw_samples["transport_only"])
    diffusion = _finite_positive_samples(raw_samples["diffusion_on"])
    if disabled is None or transport is None or diffusion is None:
        raise ValueError("raw timings must be finite positive lists")
    if len(transport) != len(diffusion) or len(disabled) != 2 * len(transport) + 1:
        raise ValueError("raw timings do not form adjacent brackets")

    paired = {"transport_only": [], "diffusion_on": []}
    deltas = {"transport_only": [], "diffusion_on": []}
    for index, value in enumerate(transport):
        before, after = disabled[2 * index:2 * index + 2]
        baseline = (before + after) * 0.5
        paired["transport_only"].append(_paired_overhead(value, before, after))
        deltas["transport_only"].append(abs(value - baseline))
    for index, value in enumerate(diffusion):
        before, after = disabled[2 * index + 1:2 * index + 3]
        baseline = (before + after) * 0.5
        paired["diffusion_on"].append(_paired_overhead(value, before, after))
        deltas["diffusion_on"].append(abs(value - baseline))
    return {
        "paired": paired,
        "disabled_median_ms": statistics.median(disabled),
        "transport_only_overhead_fraction": statistics.median(
            paired["transport_only"]),
        "diffusion_on_overhead_fraction": statistics.median(
            paired["diffusion_on"]),
        "transport_only_p95_delta_ms": _percentile(
            deltas["transport_only"], 0.95),
        "diffusion_on_p95_delta_ms": _percentile(
            deltas["diffusion_on"], 0.95),
    }


def _numbers_match(left, right):
    try:
        left_value, right_value = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(left_value) and math.isfinite(right_value)
            and math.isclose(left_value, right_value, rel_tol=1.0e-9,
                             abs_tol=1.0e-9))


def _production_configuration(configuration):
    try:
        return (
            int(configuration["frames"]) >= PRODUCTION_MINIMUMS["frames"]
            and int(configuration["warmup"]) >= PRODUCTION_MINIMUMS["warmup"]
            and int(configuration["particles"]) >= PRODUCTION_MINIMUMS["particles"]
            and int(configuration["diffusion_substeps"]) == 2
        )
    except (KeyError, TypeError, ValueError):
        return False


def _validate_provenance(provenance):
    errors = []
    required = {
        "timestamp_utc", "executable", "process_id", "command", "git_head",
        "runtime_dependency_tree_sha256", "profiler_sha256", "cuda_device",
        "python_version", "warp_version",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        result = ["provenance"]
        if not isinstance(provenance, dict) or "cuda_device" not in provenance:
            result.append("cuda_identity")
        return result
    try:
        stamp = datetime.fromisoformat(str(provenance["timestamp_utc"]))
        if stamp.utcoffset() != timezone.utc.utcoffset(stamp):
            errors.append("provenance")
    except (TypeError, ValueError):
        errors.append("provenance")
    if (not isinstance(provenance["executable"], str)
            or not Path(provenance["executable"]).is_absolute()
            or not isinstance(provenance["process_id"], int)
            or provenance["process_id"] <= 0
            or not isinstance(provenance["command"], list)
            or not provenance["command"]
            or any(not isinstance(item, str) for item in provenance["command"])
            or not re.fullmatch(r"[0-9a-f]{40}", str(provenance["git_head"]))
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(provenance["runtime_dependency_tree_sha256"]))
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(provenance["profiler_sha256"]))
            or not provenance["python_version"] or not provenance["warp_version"]):
        errors.append("provenance")
    device = provenance.get("cuda_device")
    if (not isinstance(device, dict)
            or set(device) != {"alias", "name", "ordinal"}
            or not str(device.get("alias", "")).startswith("cuda:")
            or not str(device.get("name", "")).strip()
            or not isinstance(device.get("ordinal"), int)):
        errors.append("cuda_identity")
    return errors


def _validate_current_provenance(provenance):
    if not isinstance(provenance, dict):
        return []
    errors = []
    try:
        current = _current_evidence_identity()
    except (OSError, subprocess.SubprocessError):
        return ["current_identity_unavailable"]
    comparisons = {
        "git_head": "stale_git_head",
        "runtime_dependency_tree_sha256": "stale_runtime_dependency_tree",
        "profiler_sha256": "stale_profiler_hash",
    }
    for key, error in comparisons.items():
        if provenance.get(key) != current[key]:
            errors.append(error)
    try:
        stamp = datetime.fromisoformat(str(provenance["timestamp_utc"]))
        age = (_utc_now() - stamp).total_seconds()
        if (age > MAX_EVIDENCE_AGE_SECONDS
                or age < -MAX_FUTURE_CLOCK_SKEW_SECONDS):
            errors.append("stale_timestamp")
    except (KeyError, TypeError, ValueError):
        pass
    return errors


def _validate_audit(audit):
    errors = []
    if (not isinstance(audit, dict)
            or set(audit) != {
                "instrumentation_active", "cases", "allocation",
                "source_ir_hotpath"}
            or audit.get("instrumentation_active") is not True):
        return ["audit"]
    cases = audit.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(CASE_NAMES):
        return ["audit_cases"]
    required_case = {
        "active_particles", "active_leaf_count", "active_voxel_count",
        "solver_substeps_max", "species_substeps_max",
        "species_kernel_launches", "substep_device_to_host_transfers",
        "tracked_peak_cuda_bytes", "species_object_present",
        "all_kernel_launches", "kernel_launches_by_phase",
        "species_bounds_reduction_backend",
    }
    for case in cases.values():
        if not isinstance(case, dict) or set(case) != required_case:
            errors.append("audit_work_counts")
            continue
        for key in required_case - {
                "species_object_present", "kernel_launches_by_phase",
                "species_bounds_reduction_backend"}:
            if not isinstance(case[key], int) or case[key] < 0:
                errors.append("audit_work_counts")
                break
        if not isinstance(case["species_object_present"], bool):
            errors.append("audit_work_counts")
        if case.get("species_bounds_reduction_backend") not in {
                "disabled", "native_cuda", "warp_atomic"}:
            errors.append("audit_work_counts")
        inventory = case.get("kernel_launches_by_phase")
        if (not isinstance(inventory, dict)
                or set(inventory) != {"diffusion", "lifecycle_other"}
                or any(not isinstance(phase, dict) for phase in inventory.values())
                or any(not isinstance(name, str) or not name
                       or not isinstance(count, int) or count < 1
                       for phase in inventory.values()
                       for name, count in phase.items())):
            errors.append("audit_launch_inventory")
        else:
            all_launches = sum(
                sum(phase.values()) for phase in inventory.values())
            species_launches = sum(
                count for phase in inventory.values()
                for name, count in phase.items()
                if "species" in name.lower())
            if (all_launches != case.get("all_kernel_launches")
                    or species_launches != case.get("species_kernel_launches")):
                errors.append("audit_launch_inventory")
    allocation = audit.get("allocation")
    required_allocation = {
        "admitted_species_device_bytes",
        "admitted_species_cache_staging_bytes",
        "measured_species_peak_increment_bytes",
    }
    if not isinstance(allocation, dict) or set(allocation) != required_allocation:
        errors.append("audit_allocation")
    elif any(not isinstance(value, int) or value < 0
             for value in allocation.values()):
        errors.append("audit_allocation")
    try:
        if audit.get("source_ir_hotpath") != _source_ir_hotpath_audit():
            errors.append("source_ir_hotpath")
    except (OSError, SyntaxError, UnicodeError):
        errors.append("source_ir_hotpath")
    return errors


def _validate_admission_bytes(report):
    """Recompute fixed-profile admission bytes, not just duplicated summaries."""
    try:
        from fluidSim_warp_blender51.sim.memory_safety import (
            estimate_species_cache_staging_bytes,
            estimate_species_transport_bytes,
        )

        configuration = report["configuration"]
        device = report["provenance"]["cuda_device"]["alias"]
        args = argparse.Namespace(
            particles=int(configuration["particles"]), device=device)
        settings = _settings(
            args, enabled=True,
            diffusivity=float(configuration["diffusivity_k2"]))
        allocation = report["audit"]["allocation"]
        expected = {
            "admitted_species_device_bytes": int(
                estimate_species_transport_bytes(settings)),
            "admitted_species_cache_staging_bytes": int(
                estimate_species_cache_staging_bytes(settings)),
        }
    except (KeyError, TypeError, ValueError):
        return ["admission_consistency"]
    if any(allocation.get(key) != value for key, value in expected.items()):
        return ["admission_consistency"]
    return []


def _recomputed_metrics(report):
    timing = _derived_timing(report["timing"]["raw_samples_ms"])
    allocation = report["audit"]["allocation"]
    admitted = int(allocation["admitted_species_device_bytes"])
    measured = int(allocation["measured_species_peak_increment_bytes"])
    result = {
        key: timing[key] for key in (
            "disabled_median_ms", "transport_only_overhead_fraction",
            "diffusion_on_overhead_fraction", "transport_only_p95_delta_ms",
            "diffusion_on_p95_delta_ms")
    }
    result.update(allocation)
    result["allocation_relative_error"] = abs(measured - admitted) / max(admitted, 1)
    return result, timing["paired"]


def _recomputed_checks(report, metrics):
    timing = report.get("timing", {})
    cases = report.get("audit", {}).get("cases", {})
    disabled = cases.get("disabled", {})
    diffusion = cases.get("diffusion_on", {})
    try:
        frames = int(report["configuration"]["frames"])
        raw = timing["raw_samples_ms"]
        sample_counts = (
            len(raw["disabled"]) == 2 * frames + 1
            and len(raw["transport_only"]) == frames
            and len(raw["diffusion_on"]) == frames)
    except (KeyError, TypeError, ValueError):
        sample_counts = False
    return {
        "disabled_zero_species_resources": (
            disabled.get("species_kernel_launches") == 0
            and disabled.get("substep_device_to_host_transfers") == 0
            and disabled.get("species_object_present") is False),
        "transport_only_overhead": (
            metrics["transport_only_overhead_fraction"]
            <= BUDGETS["transport_only_overhead_fraction"]),
        "diffusion_on_overhead": (
            metrics["diffusion_on_overhead_fraction"]
            <= BUDGETS["diffusion_on_overhead_fraction"]),
        "diffusion_on_k2": diffusion.get("species_substeps_max") == 2,
        "allocation_matches_admission": (
            metrics["allocation_relative_error"]
            <= BUDGETS["allocation_relative_error"]),
        "no_substep_host_synchronization": all(
            case.get("substep_device_to_host_transfers")
            == BUDGETS["substep_host_synchronizations"]
            for case in cases.values()),
        "timed_samples_uninstrumented": (
            timing.get("instrumentation_active") is False),
        "adjacent_disabled_brackets": (
            timing.get("method") == "adjacent-disabled-brackets"),
        "raw_sample_counts": sample_counts,
        "cuda_runtime": (
            str(report.get("provenance", {}).get("cuda_device", {}).get(
                "alias", "")).startswith("cuda:")
            and timing.get("context_backends") == {
                "disabled": "disabled",
                "transport_only": "native_cuda",
                "diffusion_on": "native_cuda",
            }
            and all(cases.get(name, {}).get(
                "species_bounds_reduction_backend") == "native_cuda"
                    for name in ("transport_only", "diffusion_on"))),
    }


def validate_report(report, *, require_production=True):
    """Recompute and fail closed on stale or incomplete child evidence."""
    errors = []
    if not isinstance(report, dict):
        return ["report", "status"]
    if report.get("schema_version") != 2:
        errors.append("schema_version")
    if report.get("budgets") != BUDGETS:
        errors.append("budgets")
    configuration = report.get("configuration")
    required_configuration = {
        "frames", "warmup", "particles", "diffusivity_k2",
        "diffusion_substeps",
    }
    if (not isinstance(configuration, dict)
            or set(configuration) != required_configuration):
        errors.append("configuration")
        configuration = {}
    elif (not isinstance(configuration["frames"], int)
          or not isinstance(configuration["warmup"], int)
          or not isinstance(configuration["particles"], int)
          or not isinstance(configuration["diffusion_substeps"], int)
          or configuration["frames"] < 1
          or configuration["warmup"] < 0
          or configuration["particles"] < 2
          or configuration["diffusion_substeps"] != 2
          or not _numbers_match(configuration["diffusivity_k2"], 0.06)):
        errors.append("configuration")
    production = _production_configuration(configuration)
    if report.get("evidence_kind") not in {"production", "diagnostic"}:
        errors.append("evidence_kind")
    elif (report.get("evidence_kind") == "production") != production:
        errors.append("production_minimums")
    if require_production and not production:
        errors.append("production_evidence")
    errors.extend(_validate_provenance(report.get("provenance")))
    errors.extend(_validate_current_provenance(report.get("provenance")))

    timing = report.get("timing")
    if (not isinstance(timing, dict)
            or set(timing) != {
                "method", "instrumentation_active", "raw_samples_ms",
                "paired_overheads", "context_backends"}):
        errors.append("timing")
    errors.extend(_validate_audit(report.get("audit")))
    errors.extend(_validate_admission_bytes(report))

    recomputed_checks = None
    try:
        recomputed_metrics, recomputed_pairs = _recomputed_metrics(report)
    except (KeyError, TypeError, ValueError, statistics.StatisticsError):
        errors.append("raw_sample_counts")
    else:
        supplied_pairs = timing.get("paired_overheads")
        if (not isinstance(supplied_pairs, dict)
                or set(supplied_pairs) != set(recomputed_pairs)
                or any(len(supplied_pairs[name]) != len(recomputed_pairs[name])
                       or any(not _numbers_match(left, right)
                              for left, right in zip(
                                  supplied_pairs[name], recomputed_pairs[name]))
                       for name in recomputed_pairs)):
            errors.append("paired_overheads_consistency")
        metrics = report.get("metrics")
        if (not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS
                or any(not _numbers_match(metrics.get(key), value)
                       for key, value in recomputed_metrics.items())):
            errors.append("metrics_consistency")
        recomputed_checks = _recomputed_checks(report, recomputed_metrics)
        if not recomputed_checks["raw_sample_counts"]:
            errors.append("raw_sample_counts")
        if not recomputed_checks["diffusion_on_k2"]:
            errors.append("diffusion_on_k2")
        if not recomputed_checks["cuda_runtime"]:
            errors.append("cuda_identity")
        checks = report.get("checks")
        if (not isinstance(checks, dict) or set(checks) != CHECK_KEYS
                or checks != recomputed_checks):
            errors.append("checks")

    if production:
        expected_status = (
            "pass" if recomputed_checks is not None
            and all(recomputed_checks.values()) and not errors else "fail")
    else:
        expected_status = "diagnostic"
    if report.get("status") != expected_status or (require_production and errors):
        errors.append("status")
    return sorted(set(errors))


def _settings(args, *, enabled, diffusivity):
    from fluidSim_warp_blender51.sim.solver import SolverSettings

    return SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(2.4, 2.4, 2.4),
        voxel_size=0.1, leaf_capacity=27, max_particles=args.particles,
        max_whitewater=0, particles_per_voxel=8.0,
        frame_rate=24.0, time_scale=1.0, min_substeps=1, max_substeps=1,
        gravity=(0.0, 0.0, 0.0), open_boundaries=True,
        domain_sdf_walls=False, relax_enabled=False,
        surface_enabled=False, whitewater_enabled=False,
        auto_grow_particles=False, species_enabled=enabled,
        species_diffusivity=diffusivity,
        species_max_diffusion_substeps=2, device=args.device)


def _points(count):
    side = max(1, int(math.ceil(count ** (1.0 / 3.0))))
    index = np.arange(count, dtype=np.int64)
    x = index % side
    y = (index // side) % side
    z = index // (side * side)
    scale = 2.0 / max(side, 1)
    return np.column_stack((x, y, z)).astype(np.float32) * scale + 0.2


def _new_solver(args, *, enabled, diffusivity):
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    solver = FluidSolver(_settings(args, enabled=enabled, diffusivity=diffusivity))
    points = _points(args.particles)
    midpoint = len(points) // 2
    if enabled:
        solver.inject_particles_exact(
            points[:midpoint], species_concentration=(1.0, 0.0, 0.0, 0.0))
        solver.inject_particles_exact(
            points[midpoint:], species_concentration=(0.0, 1.0, 0.0, 0.0))
    else:
        solver.inject_particles_exact(points[:midpoint])
        solver.inject_particles_exact(points[midpoint:])
    return solver


def _new_contexts(args, diffusion):
    specifications = {
        "disabled": (False, 0.0),
        "transport_only": (True, 0.0),
        "diffusion_on": (True, diffusion),
    }
    return {
        name: {"solver": _new_solver(
            args, enabled=enabled, diffusivity=diffusivity)}
        for name, (enabled, diffusivity) in specifications.items()
    }


def _release_contexts(contexts):
    for context in contexts.values():
        context["solver"].release()


def _run_warmup(contexts, frames):
    for _index in range(int(frames)):
        for name in CASE_NAMES:
            contexts[name]["solver"].step_frame()


def _run_timed_schedule(
        contexts, frames, device, *, wp_module=None, clock=time.perf_counter):
    """Measure adjacent brackets without installing any instrumentation."""
    if wp_module is None:
        import warp as wp_module
    samples = {name: [] for name in CASE_NAMES}
    work = {
        name: {"solver_substeps_max": 0, "species_substeps_max": 0}
        for name in CASE_NAMES
    }

    def measure(name):
        wp_module.synchronize_device(device)
        begin = clock()
        stats = contexts[name]["solver"].step_frame()
        wp_module.synchronize_device(device)
        samples[name].append((clock() - begin) * 1000.0)
        work[name]["solver_substeps_max"] = max(
            work[name]["solver_substeps_max"],
            int(getattr(stats, "substeps", 0)))
        receipt = getattr(stats, "species_receipt", None)
        work[name]["species_substeps_max"] = max(
            work[name]["species_substeps_max"],
            int(getattr(receipt, "diffusion_substeps", 0)))

    measure("disabled")
    for _index in range(int(frames)):
        measure("transport_only")
        measure("disabled")
        measure("diffusion_on")
        measure("disabled")
    return samples, work


def _audit_case(args, name, *, enabled, diffusivity):
    import warp as wp

    tracker_output = []
    tracker = wp.ScopedMemoryTracker(
        name=f"species-profile/audit/{name}", print=False,
        report_func=tracker_output.append)
    solver = None
    original_launch = wp.launch
    original_sync = wp.synchronize
    original_sync_device = wp.synchronize_device
    array_type = None
    original_numpy = None
    species_kernels = 0
    species_bounds_reduction_backend = "disabled"
    transfers = 0
    stage = {"phase": "lifecycle_other"}
    kernel_inventory = {
        "diffusion": Counter(), "lifecycle_other": Counter()}
    try:
        gc.collect()
        tracker.clear()
        tracker.__enter__()
        solver = _new_solver(args, enabled=enabled, diffusivity=diffusivity)
        tracker.report()

        def counted_launch(kernel, *args_, **kwargs_):
            nonlocal species_kernels
            module_name = str(getattr(getattr(kernel, "module", None), "name", ""))
            key = str(getattr(kernel, "key", "")
                      or getattr(kernel, "__name__", "<unknown>"))
            identity = f"{module_name or '<unknown>'}::{key}"
            kernel_inventory[stage["phase"]][identity] += 1
            if "species" in module_name or "species" in key:
                species_kernels += 1
            return original_launch(kernel, *args_, **kwargs_)

        def counted_sync(*args_, **kwargs_):
            nonlocal transfers
            if stage["phase"] == "diffusion":
                transfers += 1
            return original_sync(*args_, **kwargs_)

        def counted_sync_device(*args_, **kwargs_):
            nonlocal transfers
            if stage["phase"] == "diffusion":
                transfers += 1
            return original_sync_device(*args_, **kwargs_)

        if enabled:
            species = solver.species
            species_bounds_reduction_backend = species.bounds_reduction_backend
            original_diffuse = species.diffuse

            def counted_diffuse(dt):
                previous = stage["phase"]
                stage["phase"] = "diffusion"
                try:
                    return original_diffuse(dt)
                finally:
                    stage["phase"] = previous

            species.diffuse = counted_diffuse
            native_reducer = getattr(species, "_bounds_reducer", None)
            if native_reducer is not None:
                original_native_launch = native_reducer.launch

                def counted_native_launch(*args_, **kwargs_):
                    nonlocal species_kernels
                    identity = (
                        "native_cuda::fluid_species_postcompact_reduce")
                    kernel_inventory[stage["phase"]][identity] += 1
                    species_kernels += 1
                    return original_native_launch(*args_, **kwargs_)

                native_reducer.launch = counted_native_launch
            array_type = type(species.global_alpha)
            original_numpy = array_type.numpy

            def counted_numpy(self, *args_, **kwargs_):
                nonlocal transfers
                if stage["phase"] == "diffusion":
                    transfers += 1
                return original_numpy(self, *args_, **kwargs_)

            array_type.numpy = counted_numpy
        wp.launch = counted_launch
        wp.synchronize = counted_sync
        wp.synchronize_device = counted_sync_device
        stats = solver.step_frame()
        active_particles = int(solver.particles.host_count)
        active_leaves = int(getattr(solver.grid, "active_leaf_count", 0))
        active_voxels = int(getattr(solver.grid, "active_voxel_count", 0))
        receipt = getattr(stats, "species_receipt", None)
        tracker.report()
        return {
            "active_particles": active_particles,
            "active_leaf_count": active_leaves,
            "active_voxel_count": active_voxels,
            "solver_substeps_max": int(getattr(stats, "substeps", 0)),
            "species_substeps_max": int(
                getattr(receipt, "diffusion_substeps", 0)),
            "species_kernel_launches": species_kernels,
            "all_kernel_launches": sum(
                sum(phase.values()) for phase in kernel_inventory.values()),
            "kernel_launches_by_phase": {
                phase: dict(sorted(inventory.items()))
                for phase, inventory in kernel_inventory.items()},
            "substep_device_to_host_transfers": transfers,
            "tracked_peak_cuda_bytes": _peak_cuda_bytes(
                "\n".join(tracker_output), args.device),
            "species_object_present": hasattr(solver, "species"),
            "species_bounds_reduction_backend": (
                species_bounds_reduction_backend),
        }
    finally:
        stage["phase"] = "lifecycle_other"
        wp.launch = original_launch
        wp.synchronize = original_sync
        wp.synchronize_device = original_sync_device
        if array_type is not None and original_numpy is not None:
            array_type.numpy = original_numpy
        try:
            tracker.__exit__(None, None, None)
        except Exception:
            pass
        if solver is not None:
            solver.release()


def _run_audit_pass(args, diffusion):
    specifications = {
        "disabled": (False, 0.0),
        "transport_only": (True, 0.0),
        "diffusion_on": (True, diffusion),
    }
    return {
        name: _audit_case(
            args, name, enabled=enabled, diffusivity=diffusivity)
        for name, (enabled, diffusivity) in specifications.items()
    }


def _git_output(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def _provenance(args, wp, invocation):
    identity = _current_evidence_identity()
    device = wp.get_device(args.device)
    ordinal = getattr(device, "ordinal", None)
    if ordinal is None:
        try:
            ordinal = int(str(args.device).partition(":")[2])
        except ValueError:
            ordinal = -1
    return {
        "timestamp_utc": _utc_now().isoformat(),
        "executable": str(Path(sys.executable).resolve()),
        "process_id": os.getpid(),
        "command": [str(Path(sys.executable).resolve()),
                    str(Path(__file__).resolve()), *invocation],
        **identity,
        "cuda_device": {
            "alias": str(args.device), "name": str(device.name),
            "ordinal": int(ordinal),
        },
        "python_version": sys.version,
        "warp_version": str(getattr(wp, "__version__", "unknown")),
    }


def _make_report(
        args, diffusion, raw_samples, timed_backends, audit_cases, provenance):
    from fluidSim_warp_blender51.sim.memory_safety import (
        estimate_species_cache_staging_bytes,
        estimate_species_transport_bytes,
    )

    enabled_settings = _settings(args, enabled=True, diffusivity=diffusion)
    admitted_device = int(estimate_species_transport_bytes(enabled_settings))
    allocation = {
        "admitted_species_device_bytes": admitted_device,
        "admitted_species_cache_staging_bytes": int(
            estimate_species_cache_staging_bytes(enabled_settings)),
        "measured_species_peak_increment_bytes": max(
            0, int(audit_cases["diffusion_on"]["tracked_peak_cuda_bytes"])
            - int(audit_cases["disabled"]["tracked_peak_cuda_bytes"])),
    }
    configuration = {
        "frames": int(args.frames), "warmup": int(args.warmup),
        "particles": int(args.particles), "diffusivity_k2": diffusion,
        "diffusion_substeps": 2,
    }
    report = {
        "schema_version": 2, "status": "diagnostic",
        "evidence_kind": (
            "production" if _production_configuration(configuration)
            else "diagnostic"),
        "provenance": provenance,
        "configuration": configuration,
        "budgets": dict(BUDGETS),
        "timing": {
            "method": "adjacent-disabled-brackets",
            "instrumentation_active": False,
            "raw_samples_ms": raw_samples,
            "paired_overheads": {},
            "context_backends": timed_backends,
        },
        "audit": {
            "instrumentation_active": True,
            "cases": audit_cases,
            "source_ir_hotpath": _source_ir_hotpath_audit(),
            "allocation": allocation,
        },
        "metrics": {}, "checks": {},
    }
    metrics, pairs = _recomputed_metrics(report)
    report["timing"]["paired_overheads"] = pairs
    report["metrics"] = metrics
    report["checks"] = _recomputed_checks(report, metrics)
    if report["evidence_kind"] == "production":
        report["status"] = "pass" if all(report["checks"].values()) else "fail"
    return report


def _species_backend(solver):
    """Report the optional transport backend without requiring its attribute."""
    species = getattr(solver, "species", None)
    return "disabled" if species is None else species.bounds_reduction_backend


def main(argv=None):
    invocation = (_argv_after_separator(sys.argv) if argv is None else list(argv))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--particles", type=int, default=200_000)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(invocation)
    if args.frames < 1 or args.warmup < 0 or args.particles < 2:
        raise SystemExit("frames/particles must be positive and warmup nonnegative")

    import warp as wp

    wp.init()
    if not str(args.device).startswith("cuda:") or not wp.is_cuda_available():
        raise RuntimeError("a CUDA device is required")
    diffusion = 0.06
    contexts = _new_contexts(args, diffusion)
    try:
        timed_backends = {
            name: _species_backend(context["solver"])
            for name, context in contexts.items()
        }
        _run_warmup(contexts, args.warmup)
        raw_samples, _timed_work = _run_timed_schedule(
            contexts, args.frames, args.device, wp_module=wp)
    finally:
        _release_contexts(contexts)
    audit_cases = _run_audit_pass(args, diffusion)
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=True)
    report = _make_report(
        args, diffusion, raw_samples, timed_backends, audit_cases,
        _provenance(args, wp, invocation))
    errors = validate_report(
        report, require_production=report["evidence_kind"] == "production")
    if errors and report["evidence_kind"] == "production":
        report["status"] = "fail"

    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")
    marker = {
        "pass": "SPECIES-TRANSPORT-PROFILE-PASS",
        "diagnostic": "SPECIES-TRANSPORT-PROFILE-DIAGNOSTIC",
        "fail": "SPECIES-TRANSPORT-PROFILE-FAIL",
    }[report["status"]]
    print(marker)
    print(json.dumps(report["metrics"], sort_keys=True))
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
