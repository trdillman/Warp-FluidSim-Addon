"""Headless surfacing-resolution acceptance for Blender + Warp.

This test exercises the render-surface grid introduced in 1.0.233.  Defaults
are intentionally production-relevant for this project: at least 500,000 liquid
particles over at least 24 frames.  It compares the legacy simulation-grid
surface against a 2x render grid, verifies cache invalidation and validates that
surface filter settings are actually consumed by the solver path.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np


def _ensure_package_on_path() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    root = here.parents[1]
    package_dir = root / "fluidSim_warp_blender51"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return package_dir


def _initial_particles(particle_count: int = 500_000, speed: float = 3.0, seed: int = 233):
    target = max(500_000, int(particle_count))
    rng = np.random.default_rng(seed)
    nx = max(8, int(round((target * 2.4) ** (1.0 / 3.0))))
    ny = max(8, int(round(nx * 0.62)))
    nz = max(4, int(math.ceil(target / max(nx * ny, 1))))
    xs = np.linspace(-0.62, 0.62, nx, dtype=np.float32)
    ys = np.linspace(-0.36, 0.36, ny, dtype=np.float32)
    zs = np.linspace(0.32, 0.98, nz, dtype=np.float32)
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)[:target].astype(np.float32, copy=True)
    pts += rng.uniform(-0.0035, 0.0035, pts.shape).astype(np.float32)
    phase = pts[:, 1] * 8.0 + pts[:, 2] * 5.0
    vels = np.empty_like(pts, dtype=np.float32)
    vels[:, 0] = 0.45 * speed * np.sin(phase)
    vels[:, 1] = 0.35 * speed * np.cos(phase * 1.6)
    vels[:, 2] = 0.18 * speed * np.sin(phase * 1.2)
    return pts, vels


def _settings(device: str, multiplier: float, particle_count: int):
    from fluidSim_warp_blender51.sim.enums import ParticleSampleType, SurfaceResolutionMode, SurfaceUpdateCadence
    from fluidSim_warp_blender51.sim.solver import SolverSettings

    s = SolverSettings()
    s.device = device
    s.domain_min = (-1.25, -0.85, 0.0)
    s.domain_max = (1.25, 0.85, 1.35)
    s.voxel_size = 0.10
    s.max_particles = max(620_000, int(particle_count) + 32_768)
    s.max_whitewater = 0
    s.frame_rate = 24.0
    s.time_scale = 1.0
    s.cfl = 12.0
    s.min_substeps = 1
    s.max_substeps = 8
    s.gravity = (0.0, 0.0, 0.0)
    s.sample_type = ParticleSampleType.FLIP
    s.flip_ratio = 0.97
    s.pressure_iterations = 48
    s.pressure_tolerance = 1.0e-5
    s.relax_enabled = False
    s.surface_enabled = True
    s.particle_radius_scale = 1.35
    s.surface_splat_radius = 2
    if multiplier <= 1.0001:
        s.surface_resolution_mode = SurfaceResolutionMode.SIMULATION
        s.surface_resolution_multiplier = 1.0
    else:
        s.surface_resolution_mode = SurfaceResolutionMode.MULTIPLIER
        s.surface_resolution_multiplier = float(multiplier)
    s.surface_leaf_capacity_scale = max(1.0, float(multiplier) ** 3)
    s.surface_max_dense_dim = 768
    s.surface_update_cadence = SurfaceUpdateCadence.FRAME
    s.surface_build_lowres_sdf = True
    s.surface_filter_iterations = 0
    s.surface_smooth_iterations = 1
    s.whitewater_enabled = False
    return s


def _mesh_summary(solver):
    verts, tris = solver.surface_mesh()
    if len(verts):
        bmin = verts.min(axis=0).astype(float).tolist()
        bmax = verts.max(axis=0).astype(float).tolist()
        checksum = float(np.mean(verts[:, 0]) + 3.0 * np.mean(verts[:, 1]) + 7.0 * np.mean(verts[:, 2]))
    else:
        bmin = bmax = [0.0, 0.0, 0.0]
        checksum = 0.0
    return {
        "vertices": int(len(verts)),
        "triangles": int(len(tris)),
        "bbox_min": bmin,
        "bbox_max": bmax,
        "checksum": checksum,
        "surface_voxel_size": float(solver.surface.metadata()["voxel_size"]),
        "surface_active_leaves": int(getattr(solver.surface.grid, "active_leaf_count", 0)),
        "surface_active_voxels": int(getattr(solver.surface.grid, "active_leaf_count", 0) * 512),
        "surface_grid_is_sim_grid": bool(solver.surface.grid is solver.grid),
    }


def _run_case(device: str, multiplier: float, frames: int, particles: int):
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    settings = _settings(device, multiplier, particles)
    solver = FluidSolver(settings)
    pts, vels = _initial_particles(particles)
    solver.inject_particles(pts, velocities=vels)
    t0 = time.perf_counter()
    substeps = []
    for _ in range(frames):
        st = solver.step_frame()
        substeps.append(int(st.substeps))
    elapsed = time.perf_counter() - t0
    mesh = _mesh_summary(solver)
    n = solver.particles.refresh_host_count()
    return solver, {
        "multiplier": float(multiplier),
        "requested_particles": int(particles),
        "frames": int(frames),
        "particles": int(n),
        "wall_time_seconds": float(elapsed),
        "mean_substeps": float(np.mean(substeps)),
        "surface": mesh,
    }


def main(argv: list[str] | None = None) -> int:
    package_dir = _ensure_package_on_path()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from _test_device import script_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=script_device())
    parser.add_argument("--particles", type=int, default=500_000)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--output", default=str(package_dir / "demo_outputs" / "surface_resolution_pipeline.json"))
    args, _ = parser.parse_known_args(argv)

    particles = max(500_000, int(args.particles))
    frames = max(24, int(args.frames))
    _solver_1x, one = _run_case(args.device, 1.0, frames, particles)
    solver_2x, two = _run_case(args.device, 2.0, frames, particles)

    # Public rebuild / filter-path acceptance on the final 2x state.
    mesh_before = _mesh_summary(solver_2x)
    from fluidSim_warp_blender51.sim.enums import SdfFilterType
    solver_2x.settings.surface_filter_type = SdfFilterType.DIALATE
    solver_2x.settings.surface_filter_iterations = 1
    solver_2x.settings.surface_filter_strength = 1.0
    solver_2x.rebuild_surface()
    mesh_after = _mesh_summary(solver_2x)
    bbox_delta = float(max(
        np.max(np.abs(np.asarray(one["surface"]["bbox_min"]) - np.asarray(two["surface"]["bbox_min"]))),
        np.max(np.abs(np.asarray(one["surface"]["bbox_max"]) - np.asarray(two["surface"]["bbox_max"]))),
    ))

    payload = {
        "device": args.device,
        "particles_required": 500_000,
        "frames_required": 24,
        "one_x": one,
        "two_x": two,
        "filter_rebuild_probe": {
            "mesh_before": mesh_before,
            "mesh_after": mesh_after,
            "checksum_delta": float(abs(mesh_after["checksum"] - mesh_before["checksum"])),
        },
        "bbox_delta_1x_2x": bbox_delta,
    }
    checks = [
        ("1x particle requirement met", one["particles"] >= 500_000),
        ("2x particle requirement met", two["particles"] >= 500_000),
        ("frame requirement met", one["frames"] >= 24 and two["frames"] >= 24),
        ("1x mesh valid", one["surface"]["vertices"] > 0 and one["surface"]["triangles"] > 0),
        ("2x mesh valid", two["surface"]["vertices"] > 0 and two["surface"]["triangles"] > 0),
        ("2x uses smaller voxel", two["surface"]["surface_voxel_size"] < one["surface"]["surface_voxel_size"]),
        ("2x uses separate surface grid", not two["surface"]["surface_grid_is_sim_grid"]),
        ("2x has at least 1x vertex density", two["surface"]["vertices"] >= one["surface"]["vertices"]),
        ("bounds remain comparable", bbox_delta < 0.18),
        ("public filter rebuild changes mesh", payload["filter_rebuild_probe"]["checksum_delta"] > 1.0e-7 or mesh_after["vertices"] != mesh_before["vertices"]),
    ]
    payload["validation"] = {"passed": all(ok for _, ok in checks), "checks": [{"name": n, "pass": bool(ok)} for n, ok in checks]}
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"surface resolution pipeline: {'PASS' if payload['validation']['passed'] else 'FAIL'}")
    print(f"  1x verts={one['surface']['vertices']} h={one['surface']['surface_voxel_size']:.5f} wall={one['wall_time_seconds']:.3f}s")
    print(f"  2x verts={two['surface']['vertices']} h={two['surface']['surface_voxel_size']:.5f} wall={two['wall_time_seconds']:.3f}s")
    print(f"  output={out}")
    return 0 if payload["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
