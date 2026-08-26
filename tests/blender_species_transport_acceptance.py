"""Deterministic CUDA acceptance for two-species clear-dye transport.

Run with Blender 5.1 so the final case can save/reopen a real ``.blend`` and
exercise cache-payload publication in the reopened process state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THRESHOLDS = {
    "closed_box_relative_conservation": 1.0e-5,
    "min_concentration": -1.0e-6,
    "max_loading": 1.0 + 1.0e-6,
    "max_ledger_residual": 2.0e-6,
    "max_receipt_continuity_error": 2.0e-6,
    "min_particle_mixing_index_delta": 1.0e-6,
    "min_particle_color_distribution_delta": 1.0e-6,
    "min_surface_color_delta": 1.0e-6,
    "gaussian_relative_l2": 0.02,
    "seam_relative_l2": 1.0e-5,
    "checkpoint_relative_l2": 1.0e-4,
    "substep_device_to_host_transfers": 0,
}


def validate_report(report):
    """Return fail-closed schema/threshold errors for a saved receipt."""
    errors = []
    required_runtime = {
        "blender_version", "blender_executable", "python_version",
        "addon_version", "warp_version", "device_alias", "device_name",
    }
    if report.get("schema_version") != 2:
        errors.append("schema_version")
    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or not required_runtime <= set(runtime):
        errors.append("runtime_identity")
    if report.get("thresholds") != THRESHOLDS:
        errors.append("thresholds")
    provenance = report.get("provenance")
    expected_provenance = {
        "solver": "FluidSolver.step_frame",
        "cache_write": "io.cache.write_frame",
        "cache_read": "io.cache.read_frame",
        "playback": "blender.playback.play_cache_frame",
        "reopened_blend": True,
        "measured_segment": {
            "advance": "FluidSolver.step_frame",
            "seeded_before_measurement": True,
            "post_baseline_injection_count": 0,
            "active_source_count": 0,
            "direct_species_transport_calls": 0,
        },
        "lifecycle_boundary_segment": {
            "advance": "FluidSolver.step_frame",
            "source_descriptor": "SourceShapeDesc",
            "sink_descriptor": "SourceShapeDesc",
            "source_frame_count": 1,
            "sink_frame_count": 1,
        },
    }
    if provenance != expected_provenance:
        errors.append("integrated_provenance")
    metrics = report.get("metrics")
    checks = report.get("checks")
    if not isinstance(metrics, dict) or not isinstance(checks, dict):
        errors.append("metrics_or_checks")
        return errors
    finite_keys = (
        "closed_box_relative_conservation", "min_concentration",
        "max_loading", "gaussian_relative_l2", "face_seam_relative_l2",
        "edge_seam_relative_l2", "corner_seam_relative_l2",
        "checkpoint_relative_l2", "long_run_frames",
        "long_run_receipt_frames", "long_run_solver_substeps_min",
        "long_run_diffusion_substeps_min", "measured_source_amount",
        "max_ledger_residual", "max_receipt_continuity_error",
        "particle_mixing_index_delta",
        "reopen_particle_mixing_index_first",
        "reopen_particle_mixing_index_second",
        "reopen_particle_mixing_index_delta",
        "reopen_diffusion_substeps_min", "reopen_limiter_alpha_min",
        "particle_color_distribution_delta", "surface_color_delta",
        "particle_color_samples_min", "surface_color_samples_min",
        "lifecycle_source_amount_a", "lifecycle_source_amount_b",
        "lifecycle_sink_amount_a", "lifecycle_sink_amount_b",
        "lifecycle_max_ledger_residual",
        "lifecycle_removed_particle_count",
        "solver_substeps", "cache_frames_written", "cache_frames_read",
        "substep_device_to_host_transfers",
    )
    if any(key not in metrics or not math.isfinite(float(metrics[key]))
           for key in finite_keys):
        errors.append("finite_metrics")
    comparisons = (
        ("closed_box_relative_conservation", "<=",
         THRESHOLDS["closed_box_relative_conservation"]),
        ("min_concentration", ">=", THRESHOLDS["min_concentration"]),
        ("max_loading", "<=", THRESHOLDS["max_loading"]),
        ("gaussian_relative_l2", "<=", THRESHOLDS["gaussian_relative_l2"]),
        ("face_seam_relative_l2", "<=", THRESHOLDS["seam_relative_l2"]),
        ("edge_seam_relative_l2", "<=", THRESHOLDS["seam_relative_l2"]),
        ("corner_seam_relative_l2", "<=", THRESHOLDS["seam_relative_l2"]),
        ("checkpoint_relative_l2", "<=",
         THRESHOLDS["checkpoint_relative_l2"]),
        ("substep_device_to_host_transfers", "<=", 0),
        ("max_ledger_residual", "<=",
         THRESHOLDS["max_ledger_residual"]),
        ("max_receipt_continuity_error", "<=",
         THRESHOLDS["max_receipt_continuity_error"]),
        ("lifecycle_max_ledger_residual", "<=",
         THRESHOLDS["max_ledger_residual"]),
    )
    for key, op, limit in comparisons:
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError):
            errors.append(key)
            continue
        if (op == "<=" and value > limit) or (op == ">=" and value < limit):
            errors.append(key)
    try:
        if (int(metrics["long_run_frames"]) != 240
                or int(metrics["long_run_receipt_frames"]) != 240):
            errors.append("real_solver_long_run")
        if float(metrics["measured_source_amount"]) != 0.0:
            errors.append("zero_measured_sources")
        if float(metrics["max_ledger_residual"]) > THRESHOLDS[
                "max_ledger_residual"] or float(
                    metrics["max_receipt_continuity_error"]) > THRESHOLDS[
                        "max_receipt_continuity_error"]:
            errors.append("ledger_closure")
        if (int(metrics["long_run_diffusion_substeps_min"]) < 1
                or float(metrics["particle_mixing_index_delta"])
                <= THRESHOLDS["min_particle_mixing_index_delta"]):
            errors.append("diffusion_caused_particle_mixing")
        if float(metrics["particle_color_distribution_delta"]) <= THRESHOLDS[
                "min_particle_color_distribution_delta"]:
            errors.append("derived_particle_color_distribution_change")
        if float(metrics["surface_color_delta"]) <= THRESHOLDS[
                "min_surface_color_delta"]:
            errors.append("saved_reopen_surface_color")
        limiter_alpha = float(metrics["reopen_limiter_alpha_min"])
        if not 0.0 <= limiter_alpha <= 1.0:
            errors.append("reopen_limiter_alpha_range")
        reopen_first = float(metrics["reopen_particle_mixing_index_first"])
        reopen_second = float(metrics["reopen_particle_mixing_index_second"])
        reopen_delta = float(metrics["reopen_particle_mixing_index_delta"])
        if (reopen_first < 0.0
                or reopen_second <= reopen_first
                or reopen_delta <= THRESHOLDS[
                    "min_particle_mixing_index_delta"]
                or not math.isclose(
                    reopen_delta, reopen_second - reopen_first,
                    rel_tol=1.0e-9, abs_tol=1.0e-12)
                or int(metrics["reopen_diffusion_substeps_min"]) < 1):
            errors.append("reopen_diffusion_only_mixing")
        if (int(metrics["particle_color_samples_min"]) <= 0
                or int(metrics["surface_color_samples_min"]) <= 0):
            errors.append("nonempty_publication_samples")
        lifecycle_amounts = tuple(float(metrics[key]) for key in (
            "lifecycle_source_amount_a", "lifecycle_source_amount_b",
            "lifecycle_sink_amount_a", "lifecycle_sink_amount_b"))
        if (any(value <= 0.0 for value in lifecycle_amounts)
                or float(metrics["lifecycle_max_ledger_residual"])
                > THRESHOLDS["max_ledger_residual"]
                or int(metrics["lifecycle_removed_particle_count"]) <= 0):
            errors.append("real_solver_lifecycle_boundaries")
        if int(metrics["long_run_solver_substeps_min"]) < 2:
            errors.append("multi_cfl_substeps")
        if int(metrics["solver_substeps"]) < 2:
            errors.append("multi_cfl_substeps")
        if (int(metrics["cache_frames_written"]) < 2
                or int(metrics["cache_frames_read"]) < 2):
            errors.append("cache_write_read")
    except (KeyError, TypeError, ValueError):
        errors.append("integrated_metrics")
    for key in ("mixing_off_identity", "finite", "real_solver_long_run",
                "zero_measured_sources", "ledger_closure",
                "diffusion_caused_particle_mixing",
                "derived_particle_color_distribution_change",
                "reopen_diffusion_only_mixing",
                "nonempty_publication_samples",
                "real_solver_lifecycle_boundaries",
                "saved_reopen_surface_color", "real_solver_step_frame",
                "multi_cfl_substeps", "cache_write_read",
                "playback_after_reopen"):
        if checks.get(key) is not True:
            errors.append(key)
    if (report.get("status") != "pass" or errors
            or any(value is not True for value in checks.values())):
        errors.append("status")
    return sorted(set(errors))


def _argv_after_separator(argv):
    return argv[argv.index("--") + 1:] if "--" in argv else argv[1:]


def _fixture(coords, concentrations, *, device, diffusivity, volumes=None):
    import warp as wp
    from fluidSim_warp_blender51.sim.fields import SparseGrid
    from fluidSim_warp_blender51.sim.particles import ParticleSystem
    from fluidSim_warp_blender51.sim.species import SpeciesPalette, SpeciesTransport

    coords = np.asarray(coords, dtype=np.float32)
    concentration = np.zeros((len(coords), 4), dtype=np.float32)
    concentration[:, :2] = np.asarray(concentrations, dtype=np.float32)
    if volumes is None:
        volumes = np.ones(len(coords), dtype=np.float32)
    positions = coords + 0.5
    particles = ParticleSystem(
        max(1, len(coords)), device,
        ("represented_volume", "species_concentration"))
    wp.copy(particles.position, wp.array(positions, dtype=wp.vec3, device=device))
    wp.copy(particles.attr("represented_volume"), wp.array(
        np.asarray(volumes, dtype=np.float32), dtype=wp.float32, device=device))
    wp.copy(particles.attr("species_concentration"), wp.array(
        concentration, dtype=wp.vec4, device=device))
    wp.copy(particles.count, wp.array(
        np.asarray([len(coords)], np.int32), dtype=wp.int32, device=device))
    particles.host_count = len(coords)
    max_coord = np.max(coords, axis=0) if len(coords) else np.zeros(3)
    dims = tuple(max(3, int(math.ceil((float(value) + 9.0) / 8.0)))
                 for value in max_coord)
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=tuple(8.0 * value for value in dims),
        voxel_size=1.0, leaf_capacity=math.prod(dims), device=device)
    grid.begin_topology()
    grid.mark_points(particles.position, particles.count, len(coords))
    grid.expand(iterations=1)
    grid.finalize_topology()
    palette = SpeciesPalette.create(
        ids=("dye_a", "dye_b"),
        absorption_rgb=((0.05, 1.75, 2.75), (2.75, 1.25, 0.05)),
        reference_path_length=1.0)
    transport = SpeciesTransport(
        particles, grid, device, enabled=True, diffusivity=diffusivity,
        max_diffusion_substeps=64, palette=palette)
    return particles, grid, transport


def _concentrations(particles):
    return np.asarray(particles.attr("species_concentration").numpy()[
        :particles.host_count], dtype=np.float64)


def _amounts(particles):
    c = _concentrations(particles)[:, :2]
    v = np.asarray(particles.attr("represented_volume").numpy()[
        :particles.host_count], dtype=np.float64)
    return np.sum(v[:, None] * c, axis=0)


def _relative_l2(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.linalg.norm(actual - expected) /
                 max(np.linalg.norm(expected), 1.0e-12))


def _mixing_index(concentrations):
    """Permutation-invariant overlap of the two carried species."""
    values = np.asarray(concentrations, dtype=np.float64)
    if not len(values):
        return 0.0
    return float(np.mean(4.0 * values[:, 0] * values[:, 1]))


def _color_distribution_signature(values):
    """Summarize color distribution independently of particle ordering."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.zeros(15, dtype=np.float64)
    return np.quantile(
        values[:, :3], (0.1, 0.25, 0.5, 0.75, 0.9), axis=0).reshape(-1)


def _transport_metrics(device):
    import warp as wp

    metrics = {}
    checks = {}

    p, _g, t = _fixture(
        ((6, 0, 0), (7, 0, 0)), ((1, 0), (0, 1)),
        device=device, diffusivity=0.0)
    before = _concentrations(p).copy()
    t.diffuse(1.0)
    checks["mixing_off_identity"] = bool(np.array_equal(
        _concentrations(p), before) and t.frame_diffusion_substeps == 0)

    xs = np.arange(6, 43, dtype=np.float64)
    center, sigma0 = 24.0, 4.0
    gaussian0 = np.exp(-0.5 * ((xs - center) / sigma0) ** 2)
    p, _g, t = _fixture(
        [(x, 4, 4) for x in xs], np.column_stack((gaussian0, np.zeros_like(xs))),
        device=device, diffusivity=0.5)
    steps, dt = 40, 1.0 / 40.0
    for _ in range(steps):
        t.diffuse(dt)
        t.finalize_frame_receipt()
    sigma = math.sqrt(sigma0 * sigma0 + 2.0 * 0.5 * steps * dt)
    expected = (sigma0 / sigma) * np.exp(-0.5 * ((xs - center) / sigma) ** 2)
    metrics["gaussian_relative_l2"] = _relative_l2(
        _concentrations(p)[:, 0], expected)

    cube = [(x, y, z) for z in (5, 6) for y in (5, 6) for x in (5, 6)]
    pattern = [(0.15 + 0.1 * i, 0.7 - 0.08 * i) for i in range(8)]
    pattern = [(a, max(0.0, min(b, 1.0 - a))) for a, b in pattern]

    def evolved(shift):
        shifted = [tuple(c[i] + shift[i] for i in range(3)) for c in cube]
        particles, _grid, transport = _fixture(
            shifted, pattern, device=device, diffusivity=0.1)
        transport.diffuse(0.5)
        return _concentrations(particles)

    reference = evolved((0, 0, 0))
    for name, shift in (("face", (2, 0, 0)), ("edge", (2, 2, 0)),
                        ("corner", (2, 2, 2))):
        metrics[f"{name}_seam_relative_l2"] = _relative_l2(
            evolved(shift), reference)

    p, _g, t = _fixture(
        ((6, 4, 4), (7, 4, 4), (8, 4, 4)),
        ((1, 0), (0.25, 0.5), (0, 1)), device=device, diffusivity=0.1)
    for _ in range(4):
        t.diffuse(0.25)
        t.finalize_frame_receipt()
    arrays = t.checkpoint_arrays(p.host_count)
    scalars = t.checkpoint_scalars()
    resumed_p, _resumed_g, resumed_t = _fixture(
        ((6, 4, 4), (7, 4, 4), (8, 4, 4)),
        ((0, 0), (0, 0), (0, 0)), device=device, diffusivity=0.1)
    resumed_t.restore_checkpoint(arrays, scalars, resumed_p.host_count)
    for _ in range(4):
        t.diffuse(0.25)
        t.finalize_frame_receipt()
        resumed_t.diffuse(0.25)
        resumed_t.finalize_frame_receipt()
    metrics["checkpoint_relative_l2"] = _relative_l2(
        _concentrations(resumed_p), _concentrations(p))

    p, _g, t = _fixture(
        ((6, 0, 0), (7, 0, 0)), ((1, 0), (0, 1)),
        device=device, diffusivity=0.1)
    array_type = type(t.global_alpha)
    original_numpy = array_type.numpy
    original_sync = wp.synchronize
    original_sync_device = wp.synchronize_device
    transfers = {"numpy": 0, "sync": 0}
    active = {"value": False}

    def counted_numpy(self, *args, **kwargs):
        if active["value"]:
            transfers["numpy"] += 1
        return original_numpy(self, *args, **kwargs)

    def counted_sync(*args, **kwargs):
        if active["value"]:
            transfers["sync"] += 1
        return original_sync(*args, **kwargs)

    def counted_sync_device(*args, **kwargs):
        if active["value"]:
            transfers["sync"] += 1
        return original_sync_device(*args, **kwargs)

    try:
        array_type.numpy = counted_numpy
        wp.synchronize = counted_sync
        wp.synchronize_device = counted_sync_device
        active["value"] = True
        t.diffuse(0.25)
        active["value"] = False
    finally:
        active["value"] = False
        array_type.numpy = original_numpy
        wp.synchronize = original_sync
        wp.synchronize_device = original_sync_device
    metrics["substep_device_to_host_transfers"] = int(
        transfers["numpy"] + transfers["sync"])
    return metrics, checks


def _real_solver_long_run_metrics(device):
    """Advance a bounded, source-free fixture through the public solver seam."""
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.125, leaf_capacity=8, max_particles=16,
        max_whitewater=0, frame_rate=240.0, time_scale=1.0,
        cfl=0.001, min_substeps=1, max_substeps=4,
        gravity=(0.0, 0.0, 0.0), pressure_iterations=4,
        relax_enabled=False, surface_enabled=False,
        whitewater_enabled=False, species_enabled=True,
        species_diffusivity=0.05, species_max_diffusion_substeps=16,
        device=device,
    )
    yz = (0.4375, 0.5625)
    first_points = np.asarray([
        (0.4375, y, z) for z in yz for y in yz], dtype=np.float32)
    second_points = np.asarray([
        (0.5625, y, z) for z in yz for y in yz], dtype=np.float32)
    solver = FluidSolver(settings)
    try:
        solver.inject_particles_exact(
            first_points, velocity=(0.1, 0.0, 0.0),
            species_concentration=(1.0, 0.0, 0.0, 0.0))
        solver.inject_particles_exact(
            second_points, velocity=(0.1, 0.0, 0.0),
            species_concentration=(0.0, 1.0, 0.0, 0.0))
        seed_receipt = solver.species.finalize_frame_receipt()
        if solver.sources or getattr(solver.sourcing, "plans", ()):
            raise RuntimeError("long-run fixture unexpectedly has active sources")

        amount_before = _amounts(solver.particles)
        concentrations_before = _concentrations(solver.particles)
        prior_after = np.asarray(seed_receipt.after, dtype=np.float64)
        source_amount = 0.0
        max_residual = 0.0
        max_continuity_error = 0.0
        minimum = math.inf
        maximum_loading = -math.inf
        solver_substeps_min = math.inf
        diffusion_substeps_min = math.inf
        receipt_frames = 0
        receipt_values = []
        for _frame in range(240):
            stats = solver.step_frame()
            receipt = stats.species_receipt
            receipt_frames += int(receipt is not None)
            if receipt is None:
                continue
            source_amount += float(np.sum(np.abs(receipt.sources)))
            max_residual = max(
                max_residual, float(np.max(np.abs(receipt.residual))))
            max_continuity_error = max(
                max_continuity_error,
                float(np.max(np.abs(
                    np.asarray(receipt.before, dtype=np.float64) - prior_after))))
            prior_after = np.asarray(receipt.after, dtype=np.float64)
            minimum = min(minimum, float(receipt.min_concentration))
            maximum_loading = max(
                maximum_loading, float(receipt.max_loading))
            solver_substeps_min = min(solver_substeps_min, int(stats.substeps))
            diffusion_substeps_min = min(
                diffusion_substeps_min, int(receipt.diffusion_substeps))
            receipt_values.extend((
                *receipt.before, *receipt.sources, *receipt.sinks,
                *receipt.after, *receipt.residual, receipt.min_concentration,
                receipt.max_loading, receipt.limiter_alpha,
            ))

        concentrations_after = _concentrations(solver.particles)
        amount_after = _amounts(solver.particles)
    finally:
        solver.release()

    conservation = float(np.max(
        np.abs(amount_after - amount_before) /
        np.maximum(np.abs(amount_before), 1.0e-12)))
    return ({
        "closed_box_relative_conservation": conservation,
        "min_concentration": float(min(
            minimum, np.min(concentrations_after[:, :2]))),
        "max_loading": float(max(
            maximum_loading,
            np.max(np.sum(concentrations_after[:, :2], axis=1)))),
        "long_run_frames": 240,
        "long_run_receipt_frames": receipt_frames,
        "long_run_solver_substeps_min": int(solver_substeps_min),
        "long_run_diffusion_substeps_min": int(diffusion_substeps_min),
        "measured_source_amount": source_amount,
        "max_ledger_residual": max_residual,
        "max_receipt_continuity_error": max_continuity_error,
        "particle_mixing_index_delta": float(
            _mixing_index(concentrations_after)
            - _mixing_index(concentrations_before)),
    }, {
        "finite": bool(
            np.all(np.isfinite(concentrations_after))
            and np.all(np.isfinite(receipt_values))),
    })


def _read_colors(obj, count):
    attrs = (obj.data.color_attributes if obj.type == "MESH"
             else obj.data.attributes)
    attr = attrs.get("display_color")
    if attr is None:
        return np.zeros((0, 4), dtype=np.float32)
    values = np.zeros(len(attr.data) * 4, dtype=np.float32)
    attr.data.foreach_get("color", values)
    return values.reshape(-1, 4)[:count]


def _real_solver_lifecycle_boundary_metrics(device):
    """Exercise actual analytic source and sink receipts in distinct frames."""
    from fluidSim_warp_blender51.sim.enums import (
        SourceEmissionMode,
        SourceShapeShape,
        SourceShapeSourceMode,
        SourceShapeType,
    )
    from fluidSim_warp_blender51.sim.sourcing import SourceShapeDesc
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.25, leaf_capacity=8, particles_per_voxel=8.0,
        max_particles=256, max_whitewater=0,
        gravity=(0.0, 0.0, 0.0), pressure_iterations=4,
        relax_enabled=False, surface_enabled=False, whitewater_enabled=False,
        species_enabled=True, species_diffusivity=0.0,
        species_max_diffusion_substeps=16, device=device,
    )
    source = SourceShapeDesc(
        shape=SourceShapeShape.BOX,
        stype=SourceShapeType.SOURCE,
        mode=SourceShapeSourceMode.PARTICLES,
        position=(0.5, 0.5, 0.5), half_extents=(0.2, 0.2, 0.2),
        emit_rate=1.0, emission_mode=SourceEmissionMode.INITIAL_FILL,
        source_id="species-acceptance-source",
        species_concentration=(0.25, 0.5, 0.0, 0.0),
    )
    sink = SourceShapeDesc(
        shape=SourceShapeShape.BOX,
        stype=SourceShapeType.SINK,
        mode=SourceShapeSourceMode.PARTICLES,
        position=(0.5, 0.5, 0.5), half_extents=(0.5, 0.5, 0.5),
        source_id="species-acceptance-sink",
    )
    solver = FluidSolver(settings)
    try:
        solver.set_sources((source,))
        source_stats = solver.step_frame()
        source_count = int(source_stats.active_particles)
        source_receipt = source_stats.species_receipt

        solver.set_sources((sink,))
        sink_stats = solver.step_frame()
        sink_count = int(sink_stats.active_particles)
        sink_receipt = sink_stats.species_receipt
    finally:
        solver.release()

    return {
        "lifecycle_source_amount_a": float(source_receipt.sources[0]),
        "lifecycle_source_amount_b": float(source_receipt.sources[1]),
        "lifecycle_sink_amount_a": float(sink_receipt.sinks[0]),
        "lifecycle_sink_amount_b": float(sink_receipt.sinks[1]),
        "lifecycle_max_ledger_residual": float(max(
            np.max(np.abs(source_receipt.residual)),
            np.max(np.abs(sink_receipt.residual)))),
        "lifecycle_removed_particle_count": int(source_count - sink_count),
    }


def _integrated_reopen_metrics(device):
    import bpy
    from types import SimpleNamespace
    from tools.repo_addon_loader import enable_repo_addon
    from fluidSim_warp_blender51.blender import playback, scene_bridge
    from fluidSim_warp_blender51.io import cache
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    enable_repo_addon(str(ROOT), default_set=True, persistent=True)
    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.species_enabled = True
    fs.show_particles = True
    fs.show_surface = True
    fs.surface_enabled = True
    fs.particle_attribute_mode = "CLEAR_DYE"
    fs.surface_attribute_mode = "CLEAR_DYE"
    fs.species_id_a = "dye_a"
    fs.species_id_b = "dye_b"
    fs.species_absorption_a = (0.05, 1.75, 2.75)
    fs.species_absorption_b = (2.75, 1.25, 0.05)
    fs.species_reference_path_length = 1.0

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.125, leaf_capacity=8, max_particles=256,
        max_whitewater=0, frame_rate=240.0, time_scale=1.0,
        cfl=0.001, min_substeps=1, max_substeps=4,
        gravity=(0.0, 0.0, 0.0), pressure_iterations=12,
        relax_enabled=False, surface_enabled=True,
        surface_splat_radius=2, surface_filter_iterations=0,
        surface_smooth_iterations=0, whitewater_enabled=False,
        species_enabled=True, species_diffusivity=0.05,
        species_max_diffusion_substeps=16,
        species_ids=(fs.species_id_a, fs.species_id_b),
        species_absorption_rgb=(tuple(fs.species_absorption_a),
                                tuple(fs.species_absorption_b)),
        species_reference_path_length=fs.species_reference_path_length,
        device=device,
    )
    yz = (0.4375, 0.5625)
    points = np.asarray([
        (x, y, z)
        for z in yz
        for y in yz
        for x in (0.4375, 0.5625)
    ], dtype=np.float32)
    first_points = points[points[:, 0] < 0.5]
    second_points = points[points[:, 0] >= 0.5]

    with tempfile.TemporaryDirectory(prefix="fluidsim-species-") as tmp:
        fs.cache_path = tmp
        blend = str(Path(tmp) / "species_reopen.blend")
        solver = FluidSolver(settings)
        try:
            solver.inject_particles_exact(
                first_points, velocity=(0.1, 0.0, 0.0),
                species_concentration=(1.0, 0.0, 0.0, 0.0))
            solver.inject_particles_exact(
                second_points, velocity=(0.1, 0.0, 0.0),
                species_concentration=(0.0, 1.0, 0.0, 0.0))
            solver.species.finalize_frame_receipt()
            if solver.sources or getattr(solver.sourcing, "plans", ()):
                raise RuntimeError("reopen fixture unexpectedly has active sources")
            session = SimpleNamespace(solver=solver, scene_key="")
            first_stats = solver.step_frame()
            first_frame = int(solver.frame)
            first_path = cache.write_frame(scene, session, frame=first_frame)
            second_stats = solver.step_frame()
            second_frame = int(solver.frame)
            second_path = cache.write_frame(scene, session, frame=second_frame)
            palette_hash = solver.species.palette.palette_hash
        finally:
            solver.release()

        first = cache.read_frame(
            tmp, first_frame, species_enabled=True,
            expected_species_palette_hash=palette_hash)
        second = cache.read_frame(
            tmp, second_frame, species_enabled=True,
            expected_species_palette_hash=palette_hash)
        cache_frames_written = int(
            Path(first_path).is_file() + Path(second_path).is_file())
        bpy.ops.wm.save_as_mainfile(filepath=blend, check_existing=False)
        bpy.ops.wm.open_mainfile(filepath=blend, load_ui=False)
        scene = bpy.context.scene
        playback_session = scene_bridge.session()
        playback.play_cache_frame(playback_session, scene, first_frame)
        particle_before = _read_colors(
            bpy.data.objects["FluidSim Particles"], len(first["positions"]))
        surface_before = _read_colors(
            bpy.data.objects["FluidSim Surface"], len(first["surface_verts"]))
        playback.play_cache_frame(playback_session, scene, second_frame)
        particle_after = _read_colors(
            bpy.data.objects["FluidSim Particles"], len(second["positions"]))
        surface_after = _read_colors(
            bpy.data.objects["FluidSim Surface"], len(second["surface_verts"]))

    return {
        "reopen_particle_mixing_index_first": float(
            _mixing_index(first["particle_species_concentration"])),
        "reopen_particle_mixing_index_second": float(
            _mixing_index(second["particle_species_concentration"])),
        "reopen_particle_mixing_index_delta": float(
            _mixing_index(second["particle_species_concentration"])
            - _mixing_index(first["particle_species_concentration"])),
        "particle_color_samples_min": int(min(
            len(particle_before), len(particle_after))),
        "surface_color_samples_min": int(min(
            len(surface_before), len(surface_after))),
        "reopen_diffusion_substeps_min": int(min(
            first_stats.species_receipt.diffusion_substeps,
            second_stats.species_receipt.diffusion_substeps)),
        "reopen_limiter_alpha_min": float(min(
            first_stats.species_receipt.limiter_alpha,
            second_stats.species_receipt.limiter_alpha)),
        "particle_color_distribution_delta": float(np.linalg.norm(
            _color_distribution_signature(particle_after)
            - _color_distribution_signature(particle_before))),
        "surface_color_delta": float(np.linalg.norm(
            _color_distribution_signature(surface_after)
            - _color_distribution_signature(surface_before))),
        "reopen_measured_source_amount": float(sum(
            abs(value)
            for stats in (first_stats, second_stats)
            for value in stats.species_receipt.sources)),
        "solver_substeps": int(min(first_stats.substeps, second_stats.substeps)),
        "cache_frames_written": cache_frames_written,
        "cache_frames_read": int(first is not None) + int(second is not None),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args(_argv_after_separator(sys.argv) if argv is None else argv)
    import bpy
    import warp as wp
    from fluidSim_warp_blender51.version import VERSION

    wp.init()
    if str(args.device).startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("CUDA is not available")
    metrics, checks = _transport_metrics(args.device)
    long_run_metrics, long_run_checks = _real_solver_long_run_metrics(args.device)
    metrics.update(long_run_metrics)
    checks.update(long_run_checks)
    metrics.update(_real_solver_lifecycle_boundary_metrics(args.device))
    metrics.update(_integrated_reopen_metrics(args.device))
    metrics["measured_source_amount"] += metrics.pop(
        "reopen_measured_source_amount")
    checks["real_solver_long_run"] = (
        metrics["long_run_frames"] == 240
        and metrics["long_run_receipt_frames"] == 240)
    checks["zero_measured_sources"] = metrics["measured_source_amount"] == 0.0
    checks["ledger_closure"] = (
        metrics["max_ledger_residual"] <= THRESHOLDS["max_ledger_residual"]
        and metrics["max_receipt_continuity_error"]
        <= THRESHOLDS["max_receipt_continuity_error"])
    checks["diffusion_caused_particle_mixing"] = (
        metrics["long_run_diffusion_substeps_min"] >= 1
        and metrics["particle_mixing_index_delta"]
        > THRESHOLDS["min_particle_mixing_index_delta"])
    checks["derived_particle_color_distribution_change"] = (
        metrics["particle_color_distribution_delta"]
        > THRESHOLDS["min_particle_color_distribution_delta"])
    checks["reopen_diffusion_only_mixing"] = (
        metrics["reopen_particle_mixing_index_second"]
        > metrics["reopen_particle_mixing_index_first"]
        and metrics["reopen_particle_mixing_index_delta"]
        > THRESHOLDS["min_particle_mixing_index_delta"]
        and math.isclose(
            metrics["reopen_particle_mixing_index_delta"],
            metrics["reopen_particle_mixing_index_second"]
            - metrics["reopen_particle_mixing_index_first"],
            rel_tol=1.0e-9, abs_tol=1.0e-12)
        and metrics["reopen_diffusion_substeps_min"] >= 1
        and 0.0 <= metrics["reopen_limiter_alpha_min"] <= 1.0)
    checks["nonempty_publication_samples"] = (
        metrics["particle_color_samples_min"] > 0
        and metrics["surface_color_samples_min"] > 0)
    checks["real_solver_lifecycle_boundaries"] = (
        min(
            metrics["lifecycle_source_amount_a"],
            metrics["lifecycle_source_amount_b"],
            metrics["lifecycle_sink_amount_a"],
            metrics["lifecycle_sink_amount_b"],
        ) > 0.0
        and metrics["lifecycle_max_ledger_residual"]
        <= THRESHOLDS["max_ledger_residual"]
        and metrics["lifecycle_removed_particle_count"] > 0)
    checks["saved_reopen_surface_color"] = (
        metrics["surface_color_delta"] > THRESHOLDS["min_surface_color_delta"])
    checks["real_solver_step_frame"] = metrics["solver_substeps"] >= 1
    checks["multi_cfl_substeps"] = (
        metrics["solver_substeps"] >= 2
        and metrics["long_run_solver_substeps_min"] >= 2)
    checks["cache_write_read"] = (
        metrics["cache_frames_written"] == 2
        and metrics["cache_frames_read"] == 2)
    checks["playback_after_reopen"] = (
        checks["derived_particle_color_distribution_change"]
        and checks["saved_reopen_surface_color"])
    report = {
        "schema_version": 2,
        "status": "pass",
        "runtime": {
            "blender_version": bpy.app.version_string,
            "blender_executable": bpy.app.binary_path,
            "python_version": sys.version,
            "addon_version": list(VERSION),
            "warp_version": getattr(wp, "__version__", "unknown"),
            "device_alias": args.device,
            "device_name": str(wp.get_device(args.device).name),
        },
        "thresholds": dict(THRESHOLDS),
        "provenance": {
            "solver": "FluidSolver.step_frame",
            "cache_write": "io.cache.write_frame",
            "cache_read": "io.cache.read_frame",
            "playback": "blender.playback.play_cache_frame",
            "reopened_blend": True,
            "measured_segment": {
                "advance": "FluidSolver.step_frame",
                "seeded_before_measurement": True,
                "post_baseline_injection_count": 0,
                "active_source_count": 0,
                "direct_species_transport_calls": 0,
            },
            "lifecycle_boundary_segment": {
                "advance": "FluidSolver.step_frame",
                "source_descriptor": "SourceShapeDesc",
                "sink_descriptor": "SourceShapeDesc",
                "source_frame_count": 1,
                "sink_frame_count": 1,
            },
        },
        "metrics": metrics,
        "checks": checks,
    }
    report["status"] = "pass" if not validate_report(report) else "fail"
    errors = validate_report(report)
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")
    if errors:
        print(f"SPECIES-TRANSPORT-ACCEPTANCE-FAIL {errors}")
        return 1
    print("SPECIES-TRANSPORT-ACCEPTANCE-PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
