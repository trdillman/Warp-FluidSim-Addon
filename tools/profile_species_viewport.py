#!/usr/bin/env python3
"""Profile synchronized Blender surface publication with clear-dye output."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from types import MappingProxyType

import numpy as np


VIEWPORT_GATES = MappingProxyType({
    "clear_dye": MappingProxyType({
        "median_fraction": 0.05, "p95_delta_ms": 1.00,
    }),
})
INVARIANT_FIELDS = frozenset({
    "particles", "active_leaves", "active_voxels", "solver_substeps",
    "surface_cadence", "surface_topology", "active_species",
})
TIMING_FIELDS = ("solve_ms", "view_ms", "total_ms")
TIMING_METHOD = "device-synchronized-adjacent-disabled-brackets"
MAD_SCALE = 1.4826
PRODUCTION_MINIMUMS = {"frames": 60, "warmup": 15, "particles": 200_000}
PROVENANCE_FIELDS = frozenset({
    "timestamp_utc", "executable", "blender_version", "python_version",
    "warp_version", "cuda_device", "git_head", "source_tree_sha256",
    "profiler_sha256",
})
PRESENTATION_FIELDS = frozenset({
    "attribute_name", "attribute_domain", "cpu_nearest_sample_calls",
    "corner_color_present", "display_color_present", "launch_inventory",
    "readback_inventory",
})
ROOT = Path(__file__).resolve().parents[1]


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_metrics(rows, field):
    pairs = []
    for offset in range(0, len(rows), 3):
        before, feature, after = rows[offset:offset + 3]
        baseline = (float(before[field]) + float(after[field])) * 0.5
        value = float(feature[field])
        pairs.append({
            "baseline_ms": baseline,
            "feature_ms": value,
            "fraction": value / baseline - 1.0,
            "delta_ms": abs(value - baseline),
        })
    fractions = [pair["fraction"] for pair in pairs]
    median_fraction = statistics.median(fractions)
    mad = statistics.median(
        abs(value - median_fraction) for value in fractions)
    return {
        "baseline_ms": statistics.median(
            pair["baseline_ms"] for pair in pairs),
        "feature_median_ms": statistics.median(
            pair["feature_ms"] for pair in pairs),
        "median_fraction": median_fraction,
        "scaled_mad": MAD_SCALE * mad,
        "p95_delta_ms": _percentile(
            [pair["delta_ms"] for pair in pairs], 0.95),
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def _row_structure_ok(rows):
    if not isinstance(rows, list) or not rows or len(rows) % 3:
        return False
    for offset in range(0, len(rows), 3):
        before, feature, after = rows[offset:offset + 3]
        if (before.get("case") != "disabled"
                or feature.get("case") != "clear_dye"
                or after.get("case") != "disabled"):
            return False
        if any(row.get("status") not in {"supported", "unsupported"}
               for row in (before, feature, after)):
            return False
    return True


def _timings_ok(rows):
    for row in rows:
        try:
            sync = float(row["sync_ms"])
            solve, view, total = (float(row[field]) for field in TIMING_FIELDS)
        except (KeyError, TypeError, ValueError):
            return False
        if (any(not math.isfinite(value) or value <= 0.0
                for value in (sync, solve, view, total))
                or not math.isclose(
                    total, sync + solve + view,
                    rel_tol=1.0e-9, abs_tol=1.0e-9)):
            return False
    return True


def _invariants_ok(rows):
    for offset in range(0, len(rows), 3):
        invariants = [row.get("invariants") for row in rows[offset:offset + 3]]
        if any(not isinstance(item, dict)
               or set(item) != INVARIANT_FIELDS for item in invariants):
            return False
        shared = INVARIANT_FIELDS - {"active_species"}
        if any(invariants[index][key] != invariants[0][key]
               for index in (1, 2) for key in shared):
            return False
        if (invariants[0]["active_species"] != 0
                or invariants[1]["active_species"] != 2
                or invariants[2]["active_species"] != 0):
            return False
    return True


def _presentation_ok(rows):
    for offset in range(0, len(rows), 3):
        presentation = rows[offset + 1].get("presentation")
        if (not isinstance(presentation, dict)
                or set(presentation) != PRESENTATION_FIELDS
                or presentation["attribute_name"] != "fluidsim_color"
                or presentation["attribute_domain"] != "POINT"
                or presentation["cpu_nearest_sample_calls"] != 0
                or presentation["corner_color_present"] is not False
                or presentation["display_color_present"] is not False
                or not isinstance(presentation["launch_inventory"], dict)
                or not isinstance(presentation["readback_inventory"], dict)):
            return False
    return True


def _provenance_ok(provenance):
    return (
        isinstance(provenance, dict) and set(provenance) == PROVENANCE_FIELDS
        and isinstance(provenance["executable"], str)
        and bool(provenance["blender_version"])
        and str(provenance["cuda_device"]).startswith("cuda:")
        and len(str(provenance["git_head"])) == 40
        and len(str(provenance["source_tree_sha256"])) == 64
        and len(str(provenance["profiler_sha256"])) == 64)


def build_profile_report(
        rows, *, production, configuration=None, provenance=None):
    """Reduce synchronized viewport triplets without hiding unsupported cases."""
    rows = list(rows)
    adjacent = _row_structure_ok(rows)
    timings = adjacent and _timings_ok(rows)
    invariants = adjacent and _invariants_ok(rows)
    presentation = adjacent and _presentation_ok(rows)
    unsupported = adjacent and any(
        row["status"] == "unsupported" for row in rows)
    supported_rows = adjacent and not unsupported
    metrics = {}
    if supported_rows and timings:
        metrics = {
            field: _paired_metrics(rows, field) for field in TIMING_FIELDS
        }
    gate = VIEWPORT_GATES["clear_dye"]
    configuration = dict(configuration or {})
    provenance = dict(provenance or {})
    try:
        minimums = (
            len(rows) == 3 * int(configuration["frames"])
            and all(int(configuration[key]) >= value
                    for key, value in PRODUCTION_MINIMUMS.items()))
    except (KeyError, TypeError, ValueError):
        minimums = False
    checks = {
        "adjacent_disabled_brackets": adjacent,
        "timing_invariants": timings,
        "case_invariants": invariants,
        "presentation_contract": presentation,
        "production_minimums": minimums,
        "provenance": _provenance_ok(provenance),
        "supported_cases": not unsupported,
        "view_gate": bool(metrics) and (
            metrics["view_ms"]["median_fraction"] <= gate["median_fraction"]
            and metrics["view_ms"]["p95_delta_ms"] <= gate["p95_delta_ms"]),
        "total_gate": bool(metrics) and (
            metrics["total_ms"]["median_fraction"] <= gate["median_fraction"]
            and metrics["total_ms"]["p95_delta_ms"] <= gate["p95_delta_ms"]),
    }
    status = "diagnostic"
    if production:
        status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": 1,
        "status": status,
        "production": bool(production),
        "timing_method": TIMING_METHOD,
        "gates": {name: dict(value) for name, value in VIEWPORT_GATES.items()},
        "configuration": configuration,
        "provenance": provenance,
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
    }


def validate_report(report, *, require_production=True):
    """Recompute the receipt and reject shifted work or mismatched cases."""
    if not isinstance(report, dict):
        return ["report", "status"]
    errors = []
    production = report.get("production") is True
    try:
        expected = build_profile_report(
            report["rows"], production=production,
            configuration=report.get("configuration"),
            provenance=report.get("provenance"))
    except (KeyError, TypeError, ValueError, statistics.StatisticsError):
        return ["rows", "status"]
    for key in (
            "schema_version", "timing_method", "gates", "configuration",
            "provenance", "metrics", "checks", "status"):
        if report.get(key) != expected[key]:
            errors.append(key)
    for name, ok in expected["checks"].items():
        if not ok:
            if name == "supported_cases":
                if require_production:
                    errors.append("unsupported_cases")
            elif (name in {"view_gate", "total_gate"}
                  and not expected["checks"]["supported_cases"]):
                continue
            elif (name in {
                    "presentation_contract", "production_minimums", "provenance"}
                  and not require_production):
                continue
            else:
                errors.append(name)
    if require_production and not production:
        errors.append("production_evidence")
    return sorted(set(errors))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_tree_sha256():
    digest = hashlib.sha256()
    for relative in (
            "tools/profile_species_viewport.py", "tools/profile_user_path.py",
            "fluidSim_warp_blender51/blender/output_publication.py",
            "fluidSim_warp_blender51/sim/frame_output.py",
            "fluidSim_warp_blender51/sim/solver.py",
            "fluidSim_warp_blender51/sim/species.py"):
        path = ROOT / relative
        digest.update(relative.encode("utf8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _runtime_provenance(bpy, wp, args):
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "executable": str(Path(sys.executable).resolve()),
        "blender_version": ".".join(str(value) for value in bpy.app.version),
        "python_version": sys.version,
        "warp_version": str(getattr(wp, "__version__", "unknown")),
        "cuda_device": str(args.device), "git_head": head,
        "source_tree_sha256": _source_tree_sha256(),
        "profiler_sha256": _sha256(__file__),
    }


def _two_inflow_turbine_particles(count):
    """Deterministic equal-size inflows with a shared turbine velocity field."""
    count = int(count)
    index = np.arange(count, dtype=np.int64)
    half = count // 2
    local = np.where(index < half, index, index - half)
    side = max(1, int(math.ceil(max(half, count - half) ** (1.0 / 3.0))))
    x = local % side
    y = (local // side) % side
    z = local // (side * side)
    spacing = 0.7 / side
    positions = np.column_stack((x, y, z)).astype(np.float32) * spacing
    positions[:, 0] += np.where(index < half, 0.35, 1.35)
    positions[:, 1:] += 0.7
    center = np.array((1.2, 1.05, 1.05), dtype=np.float32)
    radial = positions - center
    velocities = np.column_stack(
        (-radial[:, 1], radial[:, 0], np.zeros(count, dtype=np.float32)))
    return positions, np.asarray(velocities, dtype=np.float32)


def _new_solver(args, *, species_enabled):
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(2.4, 2.4, 2.4),
        voxel_size=0.1, leaf_capacity=64, max_particles=int(args.particles),
        max_whitewater=0, frame_rate=24.0, min_substeps=1, max_substeps=1,
        gravity=(0.0, 0.0, 0.0), open_boundaries=True,
        domain_sdf_walls=False, relax_enabled=False, whitewater_enabled=False,
        surface_enabled=True, live_surface_publish_cadence=1,
        species_enabled=bool(species_enabled), species_count=2,
        species_diffusivity=0.06 if species_enabled else 0.0,
        device=args.device,
    )
    solver = FluidSolver(settings)
    positions, velocities = _two_inflow_turbine_particles(args.particles)
    half = len(positions) // 2
    if species_enabled:
        solver.inject_particles_exact(
            positions[:half], velocities=velocities[:half],
            species_concentration=(1.0, 0.0, 0.0, 0.0))
        solver.inject_particles_exact(
            positions[half:], velocities=velocities[half:],
            species_concentration=(0.0, 1.0, 0.0, 0.0))
    else:
        solver.inject_particles_exact(positions, velocities=velocities)
    return solver


def _build_scene_geometry(bpy):
    for name, location in (
            ("Species Inflow A", (0.7, 1.05, 1.05)),
            ("Species Inflow B", (1.7, 1.05, 1.05))):
        bpy.ops.mesh.primitive_cube_add(size=0.7, location=location)
        bpy.context.object.name = name
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32, radius=0.35, depth=0.12,
        location=(1.2, 1.05, 1.05), rotation=(math.pi / 2.0, 0.0, 0.0))
    bpy.context.object.name = "Species Turbine"


def _surface_presentation(bpy, output_publication, launches, readbacks, cpu_calls):
    obj = bpy.data.objects.get(output_publication.SURFACE_NAME)
    attrs = list(getattr(getattr(obj, "data", None), "color_attributes", ()))
    named = {attr.name: attr for attr in attrs}
    selected = named.get("fluidsim_color") or named.get("display_color")
    return {
        "attribute_name": getattr(selected, "name", ""),
        "attribute_domain": getattr(selected, "domain", ""),
        "cpu_nearest_sample_calls": int(cpu_calls),
        "corner_color_present": any(attr.domain == "CORNER" for attr in attrs),
        "display_color_present": "display_color" in named,
        "launch_inventory": dict(launches),
        "readback_inventory": dict(readbacks),
    }


def _topology_identity(bpy, output_publication):
    obj = bpy.data.objects.get(output_publication.SURFACE_NAME)
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return "missing"
    values = np.empty(len(mesh.loops), dtype=np.int32)
    if len(values):
        mesh.loops.foreach_get("vertex_index", values)
    digest = hashlib.sha256(values.tobytes())
    digest.update(f"{len(mesh.vertices)}:{len(mesh.polygons)}".encode("ascii"))
    return digest.hexdigest()


def main(argv=None):
    invocation = list(argv if argv is not None else (
        sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--particles", type=int, default=200_000)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(invocation)
    if args.frames < 1 or args.warmup < 0 or args.particles < 2:
        raise SystemExit("frames/particles must be positive and warmup nonnegative")

    import addon_utils
    import bpy
    import warp as wp

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    tools_root = str(Path(__file__).resolve().parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    addon_utils.enable(
        "fluidSim_warp_blender51", default_set=False, persistent=False)
    from fluidSim_warp_blender51.blender import output_publication
    from fluidSim_warp_blender51.sim.frame_output import (
        host_output_operation_counts,
    )
    from profile_user_path import measure_synchronized_frame

    wp.init()
    if not str(args.device).startswith("cuda:") or not wp.is_cuda_available():
        raise RuntimeError("a CUDA device is required")
    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.show_surface = True
    fs.show_particles = False
    fs.show_whitewater = False
    fs.surface_enabled = True
    fs.live_surface_publish_cadence = 1
    output_publication.ensure_output_shells(scene)
    _build_scene_geometry(bpy)

    from types import SimpleNamespace
    contexts = {
        "before": _new_solver(args, species_enabled=False),
        "feature": _new_solver(args, species_enabled=True),
        "after": _new_solver(args, species_enabled=False),
    }
    try:
        for _index in range(args.warmup):
            for solver in contexts.values():
                solver.step_frame()
        rows = []
        original_launch = wp.launch
        original_nearest_aligned = output_publication._nearest_aligned_values
        original_nearest = output_publication._nearest_values
        active_launches = Counter()
        cpu_calls = {"count": 0}

        def counted_launch(kernel, *args_, **kwargs_):
            name = str(getattr(kernel, "key", "")
                       or getattr(kernel, "__name__", "<unknown>"))
            active_launches[name] += 1
            return original_launch(kernel, *args_, **kwargs_)

        def counted_nearest_aligned(*args_, **kwargs_):
            cpu_calls["count"] += 1
            return original_nearest_aligned(*args_, **kwargs_)

        def counted_nearest(*args_, **kwargs_):
            cpu_calls["count"] += 1
            return original_nearest(*args_, **kwargs_)

        wp.launch = counted_launch
        output_publication._nearest_aligned_values = counted_nearest_aligned
        output_publication._nearest_values = counted_nearest
        try:
            for _index in range(args.frames):
                for key, case in (
                        ("before", "disabled"),
                        ("feature", "clear_dye"),
                        ("after", "disabled")):
                    solver = contexts[key]
                    fs.surface_attribute_mode = (
                        "CLEAR_DYE" if case == "clear_dye" else "NONE")
                    launches_before = Counter(active_launches)
                    readbacks_before = host_output_operation_counts()
                    cpu_before = cpu_calls["count"]
                    timing, stats = measure_synchronized_frame(
                        scene, SimpleNamespace(active=True, solver=solver),
                        output_publication, wp,
                        sync_callback=bpy.context.view_layer.update,
                        force_surface=True)
                    launch_delta = active_launches - launches_before
                    readbacks_after = host_output_operation_counts()
                    readback_delta = {
                        name: readbacks_after[name] - readbacks_before[name]
                        for name in readbacks_after}
                    rows.append({
                        "case": case, "status": "supported", **timing,
                        "invariants": {
                            "particles": int(stats.active_particles),
                            "active_leaves": int(solver.grid.active_leaf_count),
                            "active_voxels": int(stats.active_voxels),
                            "solver_substeps": int(stats.substeps),
                            "surface_cadence": 1,
                            "surface_topology": _topology_identity(
                                bpy, output_publication),
                            "active_species": 2 if case == "clear_dye" else 0,
                        },
                        "presentation": _surface_presentation(
                            bpy, output_publication, launch_delta,
                            readback_delta, cpu_calls["count"] - cpu_before),
                    })
        finally:
            wp.launch = original_launch
            output_publication._nearest_aligned_values = original_nearest_aligned
            output_publication._nearest_values = original_nearest
    finally:
        for solver in contexts.values():
            solver.release()

    configuration = {
        "frames": int(args.frames), "warmup": int(args.warmup),
        "particles": int(args.particles),
    }
    production = all(
        configuration[key] >= value
        for key, value in PRODUCTION_MINIMUMS.items())
    report = build_profile_report(
        rows, production=production, configuration=configuration,
        provenance=_runtime_provenance(bpy, wp, args))
    validation_errors = validate_report(
        report, require_production=production)
    if validation_errors and production:
        report["status"] = "fail"
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")
    print("SPECIES-VIEWPORT-PROFILE-" + report["status"].upper())
    return 0 if report["status"] in {"pass", "diagnostic"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
