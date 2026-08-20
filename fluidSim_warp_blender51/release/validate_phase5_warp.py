"""Execute Phase-5 surface-tension kernels with the supplied Warp package.

The script is intentionally independent of Blender and the CUDA-only full
FluidSolver.  It builds one sparse 8^3 leaf on the requested Warp device,
loads an analytic spherical SDF, executes every Phase-5 kernel, and verifies
force direction, finite values, collision exclusion, coefficient scaling, MAC
integration, and the capillary stability helpers.  Running on ``cuda:0`` is a
release certification gate; running on ``cpu`` proves the supplied package and
kernel code paths in environments without an NVIDIA device.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

VALIDATION_PACKAGE = "fluidsim_validation_source"


def _ensure_source_package(module_name: str) -> None:
    """Install a synthetic package namespace without executing Blender code."""

    from types import ModuleType

    if module_name in sys.modules:
        return
    if not module_name.isidentifier():
        raise ValueError(f"validation module must be a Python identifier: {module_name!r}")
    package = ModuleType(module_name)
    package.__file__ = str(ROOT / "__init__.py")
    package.__package__ = module_name
    package.__path__ = [str(ROOT)]
    sys.modules[module_name] = package


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _launch_pipeline(surface_tension, grid, sdf, collision_sdf, coefficient, *, device):
    count = int(grid.active_voxel_count)
    normals_a = wp.zeros(count, dtype=wp.vec3, device=device)
    normals_b = wp.zeros(count, dtype=wp.vec3, device=device)
    acceleration = wp.zeros(count, dtype=wp.vec3, device=device)
    active = wp.zeros(1, dtype=wp.int32, device=device)
    max_curvature = wp.zeros(1, dtype=float, device=device)
    max_acceleration = wp.zeros(1, dtype=float, device=device)
    frame_max_active = wp.zeros(1, dtype=wp.int32, device=device)
    h = float(grid.voxel_size)
    half_width = 1.5 * h
    # One smoothing pass plus the curvature stencil requires two cells of
    # normal support beyond the force band.
    normal_band = half_width + 2.0 * h
    wp.launch(
        surface_tension.k_surface_tension_normals,
        dim=count,
        inputs=[
            sdf, collision_sdf, grid.leaf_table, grid.leaf_coords,
            grid.voxels_dim, grid.leaves_dim, grid.num_leaves,
            h, normal_band,
        ],
        outputs=[normals_a],
        device=device,
    )
    wp.launch(
        surface_tension.k_surface_tension_smooth_normals,
        dim=count,
        inputs=[
            normals_a, sdf, collision_sdf, grid.leaf_table, grid.leaf_coords,
            grid.voxels_dim, grid.leaves_dim, grid.num_leaves, normal_band,
        ],
        outputs=[normals_b],
        device=device,
    )
    wp.launch(
        surface_tension.k_surface_tension_acceleration,
        dim=count,
        inputs=[
            normals_b, sdf, collision_sdf, grid.leaf_table, grid.leaf_coords,
            grid.voxels_dim, grid.leaves_dim, grid.num_leaves,
            h, half_width, float(coefficient), 1000.0, 2.0 / h,
        ],
        outputs=[acceleration, active, max_curvature, max_acceleration],
        device=device,
    )
    wp.launch(
        surface_tension.k_surface_tension_accumulate_frame_active_count,
        dim=1,
        inputs=[active],
        outputs=[frame_max_active],
        device=device,
    )
    wp.synchronize_device(device)
    return {
        "normals": normals_b.numpy(),
        "acceleration": acceleration.numpy(),
        "active": int(active.numpy()[0]),
        "frame_max_active": int(frame_max_active.numpy()[0]),
        "max_curvature": float(max_curvature.numpy()[0]),
        "max_acceleration": float(max_acceleration.numpy()[0]),
        "device_acceleration": acceleration,
    }


def validate(expected_warp_root: Path, module_name: str, *, device: str = "cpu") -> dict:
    wp.init()
    warp_module = Path(wp.__file__).resolve()
    expected_root = expected_warp_root.resolve()
    if expected_root not in warp_module.parents:
        raise AssertionError(
            f"Warp loaded from {warp_module}, expected under {expected_root}")
    if str(getattr(wp, "__version__", "")) != "1.16.0":
        raise AssertionError(f"unexpected Warp version {getattr(wp, '__version__', '')}")

    aliases = {str(item.alias) for item in wp.get_devices()}
    if device not in aliases:
        raise AssertionError(
            f"requested Warp device {device!r} is unavailable; detected {sorted(aliases)}")
    selected = wp.get_device(device)
    if device.startswith("cuda:") and not bool(getattr(selected, "is_cuda", False)):
        raise AssertionError(f"requested device {device!r} is not CUDA")

    _ensure_source_package(module_name)
    fields = importlib.import_module(f"{module_name}.sim.fields")
    surface_tension = importlib.import_module(f"{module_name}.sim.surface_tension")
    solver_module = importlib.import_module(f"{module_name}.sim.solver")

    class _Profiler:
        def __init__(self):
            self.bytes = 0

        def tally_buffer(self, delta):
            self.bytes += int(delta)

    profiler = _Profiler()
    resource_settings = SimpleNamespace(
        surface_tension_enabled=True,
        surface_tension_coefficient=0.072,
    )
    resource_owner = SimpleNamespace(
        device=device,
        profiler=profiler,
        grid=SimpleNamespace(voxel_capacity=512),
        settings=resource_settings,
    )
    resource_solver = surface_tension.SurfaceTensionSolver(resource_owner)
    resource_solver.ensure_resources()
    expected_scratch = 512 * 24
    if resource_solver.temporary_bytes != expected_scratch or profiler.bytes != expected_scratch:
        raise AssertionError("surface-tension scratch allocation accounting is incorrect")
    resource_settings.surface_tension_coefficient = 0.0
    resource_owner.particles = SimpleNamespace(host_count=0)
    resource_solver.apply(0.01)
    if resource_solver.temporary_bytes != 0 or profiler.bytes != 0:
        raise AssertionError("zero coefficient retained optional surface-tension scratch")

    h = 0.1
    grid = fields.SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8),
        voxel_size=h, leaf_capacity=1, device=device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.799, 0.799, 0.799))
    grid.finalize_topology(sync=True)
    if grid.active_leaf_count != 1 or grid.active_voxel_count != 512:
        raise AssertionError("analytic proof grid did not allocate exactly one leaf")

    leaf_coords = grid.leaf_coords.numpy()[:1]
    coordinates = np.empty((512, 3), dtype=np.float32)
    for flat in range(512):
        local = flat
        coordinates[flat] = (
            ((local & 7) + 0.5) * h,
            (((local >> 3) & 7) + 0.5) * h,
            (((local >> 6) & 7) + 0.5) * h,
        )
    center = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    radius = 0.25
    sdf_host = np.linalg.norm(coordinates - center, axis=1).astype(np.float32) - radius
    sdf = wp.array(sdf_host, dtype=float, device=device)
    collision_host = np.ones(512, dtype=np.float32)
    collision_sdf = wp.array(collision_host, dtype=float, device=device)

    baseline = _launch_pipeline(
        surface_tension, grid, sdf, collision_sdf, 0.072, device=device)
    strong = _launch_pipeline(
        surface_tension, grid, sdf, collision_sdf, 0.144, device=device)

    for label, result in (("baseline", baseline), ("strong", strong)):
        if result["active"] <= 0:
            raise AssertionError(f"{label} surface-tension band had no active voxels")
        if result["frame_max_active"] != result["active"]:
            raise AssertionError(f"{label} frame-active reduction did not preserve count")
        if not np.isfinite(result["normals"]).all():
            raise AssertionError(f"{label} normals contain non-finite values")
        if not np.isfinite(result["acceleration"]).all():
            raise AssertionError(f"{label} acceleration contains non-finite values")
        if not 0.0 < result["max_curvature"] <= 2.0 / h + 1.0e-5:
            raise AssertionError(f"{label} curvature diagnostic is invalid")
        if result["max_acceleration"] <= 0.0:
            raise AssertionError(f"{label} acceleration diagnostic is zero")

    ratio = strong["max_acceleration"] / baseline["max_acceleration"]
    if not 1.98 <= ratio <= 2.02:
        raise AssertionError(f"surface-tension coefficient did not scale linearly: {ratio}")

    band = np.abs(sdf_host) < 1.5 * h
    positive_x = np.flatnonzero(
        band & (coordinates[:, 0] > center[0])
        & (np.abs(coordinates[:, 1] - center[1]) < 0.08)
        & (np.abs(coordinates[:, 2] - center[2]) < 0.08))
    negative_x = np.flatnonzero(
        band & (coordinates[:, 0] < center[0])
        & (np.abs(coordinates[:, 1] - center[1]) < 0.08)
        & (np.abs(coordinates[:, 2] - center[2]) < 0.08))
    if not len(positive_x) or not len(negative_x):
        raise AssertionError("analytic SDF did not expose radial interface samples")
    if float(np.mean(baseline["acceleration"][positive_x, 0])) >= 0.0:
        raise AssertionError("positive-x droplet surface was not accelerated inward")
    if float(np.mean(baseline["acceleration"][negative_x, 0])) <= 0.0:
        raise AssertionError("negative-x droplet surface was not accelerated inward")

    # Prove integration into the weighted MAC field.
    count = grid.active_voxel_count
    u = wp.zeros(count, dtype=float, device=device)
    v = wp.zeros(count, dtype=float, device=device)
    w = wp.zeros(count, dtype=float, device=device)
    wu = wp.ones(count, dtype=float, device=device)
    wv = wp.ones(count, dtype=float, device=device)
    ww = wp.ones(count, dtype=float, device=device)
    wp.launch(
        surface_tension.k_apply_surface_tension_to_mac,
        dim=count,
        inputs=[
            baseline["device_acceleration"], collision_sdf,
            grid.leaf_table, grid.leaf_coords, grid.voxels_dim,
            grid.leaves_dim, grid.num_leaves, 0.01,
            u, v, w, wu, wv, ww,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    velocities = np.column_stack((u.numpy(), v.numpy(), w.numpy()))
    if not np.isfinite(velocities).all() or not np.any(np.abs(velocities) > 0.0):
        raise AssertionError("surface-tension acceleration did not reach MAC velocity")

    # Collision exclusion: mark one interface voxel as solid and prove its
    # acceleration is cleared and the active count decreases.
    collision_index = int(positive_x[0])
    collision_host[collision_index] = -1.0
    collision_sdf_masked = wp.array(collision_host, dtype=float, device=device)
    masked = _launch_pipeline(
        surface_tension, grid, sdf, collision_sdf_masked, 0.072, device=device)
    if np.any(masked["acceleration"][collision_index] != 0.0):
        raise AssertionError("surface tension applied inside collision SDF")
    if masked["active"] >= baseline["active"]:
        raise AssertionError("collision exclusion did not reduce active force voxels")

    capillary = surface_tension.capillary_dt(1000.0, h, 0.072, 0.5)
    fine_capillary = surface_tension.capillary_dt(1000.0, h * 0.5, 0.072, 0.5)
    strong_capillary = surface_tension.capillary_dt(1000.0, h, 0.144, 0.5)
    if not fine_capillary < capillary or not strong_capillary < capillary:
        raise AssertionError("capillary timestep trend validation failed")
    required = surface_tension.required_capillary_substeps(1.0 / 24.0, capillary)
    sdf_splat_radius = surface_tension.required_sdf_splat_radius_voxels(
        0.6, 1.5, 1, 2)
    if sdf_splat_radius != 8:
        raise AssertionError(
            f"surface-tension SDF support radius drifted: {sdf_splat_radius}")

    # Prove that the actual FluidSolver frame-level limiter applies capillary
    # stability before any first-frame source emission.  A zero-particle fake
    # deliberately bypasses CFL/device reads while exercising the real method.
    limiter_settings = SimpleNamespace(
        min_substeps=1,
        max_substeps=64,
        surface_tension_enabled=True,
        surface_tension_coefficient=1.0,
        surface_tension_density=1000.0,
        surface_tension_safety=0.5,
        voxel_size=0.005,
        cfl=0.8,
        gravity=(0.0, 0.0, 0.0),
    )
    limiter_owner = SimpleNamespace(
        settings=limiter_settings,
        particles=SimpleNamespace(host_count=0),
    )
    limiter_frame_dt = 1.0 / 24.0
    limiter_stable_dt = surface_tension.capillary_dt(
        limiter_settings.surface_tension_density,
        limiter_settings.voxel_size,
        limiter_settings.surface_tension_coefficient,
        limiter_settings.surface_tension_safety,
    )
    limiter_expected = surface_tension.required_capillary_substeps(
        limiter_frame_dt, limiter_stable_dt)
    limiter_selected, limiter_cfl_clamped = (
        solver_module.FluidSolver._compute_substeps(
            limiter_owner, limiter_frame_dt))
    if limiter_expected <= 1 or limiter_selected != limiter_expected:
        raise AssertionError(
            "FluidSolver did not select the capillary-required substep count")
    if limiter_cfl_clamped:
        raise AssertionError("capillary-only limiter was misreported as CFL-clamped")
    if limiter_owner._last_substep_diagnostics["substep_limiter"] != "CAPILLARY":
        raise AssertionError("FluidSolver did not identify the capillary limiter")

    limiter_settings.max_substeps = 8
    limiter_selected_clamped, _ = solver_module.FluidSolver._compute_substeps(
        limiter_owner, limiter_frame_dt)
    if limiter_selected_clamped != 8:
        raise AssertionError("FluidSolver did not enforce the authored max_substeps")
    if not limiter_owner._last_substep_diagnostics["capillary_substeps_clamped"]:
        raise AssertionError("FluidSolver did not report capillary substep clamping")

    cuda_devices = [str(item.alias) for item in wp.get_devices() if item.is_cuda]
    return {
        "status": "passed",
        "warp_version": str(wp.__version__),
        "warp_module": str(warp_module),
        "device_executed": device,
        "cuda_devices": cuda_devices,
        "cuda_certified": bool(device.startswith("cuda:")),
        "grid": {
            "leaf_coords": leaf_coords.tolist(),
            "active_voxels": int(count),
            "voxel_size": h,
        },
        "diagnostics": {
            "force_voxels": baseline["active"],
            "max_abs_curvature": baseline["max_curvature"],
            "max_acceleration": baseline["max_acceleration"],
            "coefficient_scaling_ratio": ratio,
            "capillary_dt": capillary,
            "required_substeps_at_24fps": required,
            "sdf_splat_radius_voxels": sdf_splat_radius,
            "limiter_probe_capillary_dt": limiter_stable_dt,
            "limiter_probe_required_substeps": limiter_expected,
            "limiter_probe_clamped_substeps": limiter_selected_clamped,
        },
        "checks": {
            "supplied_warp_module": "passed",
            "optional_scratch_lifecycle": "passed",
            "analytic_sdf_normals": "passed",
            "inward_force_direction": "passed",
            "finite_curvature_and_acceleration": "passed",
            "frame_active_voxel_reduction": "passed",
            "curvature_clamp": "passed",
            "coefficient_scaling": "passed",
            "collision_exclusion": "passed",
            "mac_integration": "passed",
            "capillary_timestep_trends": "passed",
            "sdf_support_radius": "passed",
            "solver_capillary_substep_selection": "passed",
            "solver_capillary_substep_clamping": "passed",
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-warp-root", type=Path, required=True)
    parser.add_argument("--module", default=VALIDATION_PACKAGE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-json", type=Path, required=True)
    options = parser.parse_args(argv)
    report = validate(
        options.expected_warp_root, options.module, device=options.device)
    _write_json(options.report_json, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
