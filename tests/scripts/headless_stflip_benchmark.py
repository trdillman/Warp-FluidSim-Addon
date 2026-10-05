"""Headless ST-FLIP benchmark/acceptance run for Blender + Warp.

This is intentionally small enough for continuous validation, but it exercises
real solver work on the target Warp device. It compares:
  - low-CFL FLIP reference
  - high-CFL FLIP stress
  - high-CFL ST-FLIP
and writes a JSON metrics file next to the add-on.
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
    root = here.parents[2]
    package_dir = root / "fluidSim_warp_blender51"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return package_dir


def _cube_mesh(center, size):
    cx, cy, cz = center
    sx, sy, sz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    verts = np.array([
        [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.int32)
    return verts, tris


def _make_collider(center, size, device):
    import warp as wp
    from fluidSim_warp_blender51.sim.solver import ColliderDesc

    verts, tris = _cube_mesh(center, size)
    desc = ColliderDesc()
    desc.mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(tris.reshape(-1), dtype=wp.int32, device=device),
    )
    desc.aabb_min = tuple(float(x) for x in verts.min(axis=0))
    desc.aabb_max = tuple(float(x) for x in verts.max(axis=0))
    desc.center = tuple(float(x) for x in center)
    return desc


def _initial_particles(particle_count: int = 500_000, speed: float = 22.0, seed: int = 13):
    """Create at least ``particle_count`` liquid particles.

    The layout is deterministic and vectorized so the benchmark can reliably
    exercise large-particle ST-FLIP transfer/advection without Python loop
    overhead dominating the test.  The default 125x80x50 lattice is exactly
    500,000 particles.
    """
    target = max(500_000, int(particle_count))
    rng = np.random.default_rng(seed)
    # Preserve a long/high-speed inlet block but scale dimensions for any
    # requested count.  These ratios fill the flume while keeping the initial
    # condition coherent enough for CFL comparisons.
    nx = max(8, int(round((target * 2.5) ** (1.0 / 3.0))))
    ny = max(8, int(round(nx * 0.64)))
    nz = max(4, int(math.ceil(target / max(nx * ny, 1))))
    total = nx * ny * nz
    xs = np.linspace(-1.55, -0.65, nx, dtype=np.float32)
    ys = np.linspace(-0.42, 0.42, ny, dtype=np.float32)
    zs = np.linspace(0.20, 1.20, nz, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    pts = grid[:target].astype(np.float32, copy=True)
    pts += rng.uniform(-0.0045, 0.0045, pts.shape).astype(np.float32)
    phase = pts[:, 1] * 9.0 + pts[:, 2] * 5.0
    vels = np.empty_like(pts, dtype=np.float32)
    vels[:, 0] = speed + 0.8 * np.sin(phase)
    vels[:, 1] = 0.55 * np.sin(phase * 1.7)
    vels[:, 2] = 0.35 * np.cos(phase * 1.3)
    return pts, vels

def _settings(device: str, cfl: float, spacetime: bool, particle_count: int = 500_000):
    from fluidSim_warp_blender51.sim.enums import ParticleSampleType
    from fluidSim_warp_blender51.sim.solver import SolverSettings

    s = SolverSettings()
    s.device = device
    s.domain_min = (-1.8, -0.55, 0.0)
    s.domain_max = (1.8, 0.55, 1.45)
    s.voxel_size = 0.12
    s.max_particles = max(600_000, int(particle_count) + 16_384)
    s.max_whitewater = 0
    s.frame_rate = 24.0
    s.time_scale = 1.25
    s.cfl = cfl
    s.min_substeps = 1
    s.max_substeps = 16
    s.gravity = (0.0, 0.0, -9.81)
    s.sample_type = ParticleSampleType.FLIP
    s.flip_ratio = 0.97
    s.pressure_iterations = 64
    s.pressure_tolerance = 1.0e-5
    s.relax_enabled = True
    s.relax_strength = 0.25
    s.surface_enabled = True
    s.particle_radius_scale = 1.5
    s.surface_resolution_mode = 1
    s.surface_resolution_multiplier = 2.0
    s.surface_leaf_capacity_scale = 8.0
    s.surface_max_dense_dim = 512
    s.surface_update_cadence = 1
    s.surface_build_lowres_sdf = True
    s.surface_filter_iterations = 0
    s.surface_smooth_iterations = 1
    s.whitewater_enabled = False
    s.spacetime_enabled = spacetime
    s.spacetime_jitter_strength = 1.0
    s.spacetime_adaptive_jitter = True
    s.spacetime_variable_projection = True
    s.spacetime_phase_eta = 0.5
    s.spacetime_phase_threshold = 0.25
    s.spacetime_local_cfl = 1.0
    s.spacetime_max_local_substeps = 64
    s.spacetime_seed = 1729
    return s


def _analyze(solver, settings):
    n = solver.particles.refresh_host_count()
    pos = solver.particle_positions()
    vel = solver.particle_velocities()
    speed = np.linalg.norm(vel, axis=1) if len(vel) else np.zeros(0, dtype=np.float32)
    offsets = solver.particles.time_offset[:n].numpy() if n else np.zeros(0, dtype=np.float32)
    phase = solver.density.data[:solver._nvox].numpy()
    weights = solver.vel.wu[:solver._nvox].numpy()
    verts, tris = solver.surface_mesh()
    frame_dt = settings.time_scale / settings.frame_rate
    return {
        "particles": int(n),
        "surface_vertices": int(len(verts)),
        "surface_triangles": int(len(tris)),
        "surface_voxel_size": float(solver.surface.metadata()["voxel_size"]),
        "surface_active_leaves": int(getattr(solver.render_surface_grid, "active_leaf_count", 0)),
        "surface_grid_is_sim_grid": bool(solver.render_surface_grid is solver.grid),
        "centroid": pos.mean(axis=0).astype(float).tolist() if len(pos) else [0.0, 0.0, 0.0],
        "bbox_min": pos.min(axis=0).astype(float).tolist() if len(pos) else [0.0, 0.0, 0.0],
        "bbox_max": pos.max(axis=0).astype(float).tolist() if len(pos) else [0.0, 0.0, 0.0],
        "speed_max": float(speed.max()) if speed.size else 0.0,
        "speed_mean": float(speed.mean()) if speed.size else 0.0,
        "estimated_cfl": float(speed.max() * frame_dt / settings.voxel_size) if speed.size else 0.0,
        "time_offset_abs_max": float(np.max(np.abs(offsets))) if offsets.size else 0.0,
        "time_offset_std": float(offsets.std()) if offsets.size else 0.0,
        "phase_max": float(np.max(phase)) if phase.size else 0.0,
        "phase_mean_nonzero": float(np.mean(phase[phase > 0])) if np.any(phase > 0) else 0.0,
        "face_weight_max": float(np.max(weights)) if weights.size else 0.0,
        "pressure_status": solver.pressure_solver.last_pressure_status,
        "pressure_residual": float(solver.pressure_solver.last_residual),
        "substeps_last": int(solver.stats.substeps),
        "m0_reference": float(solver._st_reference_mass()),
    }


def run_case(device: str, name: str, cfl: float, spacetime: bool, frames: int, particle_count: int):
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    settings = _settings(device, cfl, spacetime, particle_count)
    solver = FluidSolver(settings)
    solver.set_colliders([
        _make_collider((-0.08, -0.25, 0.56), (0.11, 0.52, 1.08), device),
        _make_collider((0.62, 0.22, 0.50), (0.13, 0.42, 1.00), device),
    ])
    pts, vels = _initial_particles(particle_count=particle_count)
    solver.inject_particles(pts, velocities=vels)
    frame_dt = settings.time_scale / settings.frame_rate
    initial_cfl = float(np.linalg.norm(vels, axis=1).max() * frame_dt / settings.voxel_size)
    records = []
    t0 = time.perf_counter()
    for frame in range(1, frames + 1):
        st = solver.step_frame()
        rec = _analyze(solver, settings)
        rec["frame"] = frame
        rec["step_time_seconds"] = float(st.step_time)
        records.append(rec)
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "cfl_target": cfl,
        "spacetime_enabled": spacetime,
        "initial_particles": int(len(pts)),
        "initial_cfl_estimate": initial_cfl,
        "frames": frames,
        "wall_time_seconds": elapsed,
        "total_substeps": int(sum(r["substeps_last"] for r in records)),
        "mean_substeps": float(np.mean([r["substeps_last"] for r in records])),
        "mean_step_time_seconds": float(np.mean([r["step_time_seconds"] for r in records])),
        "final": records[-1],
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    package_dir = _ensure_package_on_path()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "support"))
    from _test_device import script_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=script_device())
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--particles", type=int, default=500_000)
    parser.add_argument("--output", default=str(package_dir / "demo_outputs" / "stflip_headless_benchmark.json"))
    args, _ = parser.parse_known_args(argv)

    cases = [
        ("FLIP_CFL2_REFERENCE", 2.0, False),
        ("FLIP_CFL16_STRESS", 16.0, False),
        ("STFLIP_CFL16", 16.0, True),
    ]
    frames = max(24, int(args.frames))
    particles = max(500_000, int(args.particles))
    results = [run_case(args.device, name, cfl, st, frames, particles) for name, cfl, st in cases]
    by_name = {r["name"]: r for r in results}
    st = by_name["STFLIP_CFL16"]
    checks = [
        ("low-cfl baseline uses more substeps", by_name["FLIP_CFL2_REFERENCE"]["mean_substeps"] > by_name["FLIP_CFL16_STRESS"]["mean_substeps"]),
        ("stress cases run as high-cfl single-substep cases", by_name["FLIP_CFL16_STRESS"]["mean_substeps"] <= 1.25 and st["mean_substeps"] <= 1.25),
        ("initial CFL is high enough", min(r["initial_cfl_estimate"] for r in results) > 5.0),
        ("ST residual active", st["final"]["time_offset_std"] > 1.0e-5),
        ("ST residual bounded", st["final"]["time_offset_abs_max"] <= 0.5 * (1.25 / 24.0) + 2.0e-4),
        ("ST phase active", st["final"]["phase_max"] > 0.0 and st["final"]["face_weight_max"] > 0.0),
        ("ST calibrated m0 plausible", 4.0 <= st["final"]["m0_reference"] <= 12.0),
        ("particles retained", all(r["final"]["particles"] > 0.60 * r["initial_particles"] for r in results)),
        ("pressure finite", all(np.isfinite(r["final"]["pressure_residual"]) for r in results)),
        ("particle count requirement met", all(r["initial_particles"] >= 500_000 and r["final"]["particles"] >= 500_000 for r in results)),
        ("frame count requirement met", all(r["frames"] >= 24 for r in results)),
    ]
    validation = {"passed": all(ok for _, ok in checks), "checks": [{"name": n, "pass": bool(ok)} for n, ok in checks]}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"benchmark": "ST-FLIP high-CFL headless benchmark", "device": args.device, "minimum_particles": particles, "minimum_frames": frames, "results": results, "validation": validation}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ST-FLIP headless benchmark: {'PASS' if validation['passed'] else 'FAIL'}")
    for r in results:
        print(f"  {r['name']}: mean_substeps={r['mean_substeps']:.2f}, wall={r['wall_time_seconds']:.3f}s, particles={r['final']['particles']}, verts={r['final']['surface_vertices']}, m0={r['final']['m0_reference']:.4f}")
    print(f"  metrics={output}")
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
