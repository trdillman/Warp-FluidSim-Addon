"""Before/after for Sharp Surfacing on the same particles: time and mesh stats.

Steps a bpy-free scene, then for each sampled frame builds the render surface
three ways from the same particles: plain (round balls), Sharp Surfacing, and
Sharp Surfacing plus Preserve Splashes. Prints build+extract time, triangle
count and quality, and can write each mesh as OBJ for a side-by-side look.

    python tools/profiling/compare_sharp_surfacing.py --solver-preset dam_break \\
        --device cuda:0 --voxel-size 0.06 --frames 20 40 \\
        --obj-dir artifacts/sharp_surfacing --json-out artifacts/sharp_surfacing.json

Set ``FLUIDSIM_SURFACE_BACKEND=nets`` (Warp 1.19 Surface Nets branch) to mesh
with Surface Nets instead of sparse marching cubes; the comparison is the same.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

VARIANTS = {
    "plain": {"surface_detail_enabled": False,
              "surface_smooth_preserve_splashes": 0.0},
    "sharp": {"surface_detail_enabled": True,
              "surface_smooth_preserve_splashes": 0.0},
    "sharp_preserve": {"surface_detail_enabled": True,
                       "surface_smooth_preserve_splashes": 0.75},
}


def triangle_quality(verts, tris):
    import numpy as np

    if len(tris) == 0:
        return np.zeros(0)
    v = np.asarray(verts, np.float64)[np.asarray(tris)]
    e0, e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(e0, -e2), axis=1)
    denom = (e0 ** 2).sum(1) + (e1 ** 2).sum(1) + (e2 ** 2).sum(1)
    return 4.0 * np.sqrt(3.0) * area / np.maximum(denom, 1e-30)


def write_obj(path, verts, tris):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts)
        fh.writelines(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n" for t in tris)


def measure_variant(solver, overrides, repeats):
    """Rebuild the render surface with ``overrides`` and time build+extract."""
    import warp as wp

    s = solver.settings
    saved = {k: getattr(s, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(s, k, v)
        pipeline = solver.surface
        times = []
        verts = tris = None
        for _ in range(repeats + 1):  # first run warms up
            # Plain can reuse the physical field, so invalidate it along
            # with the render products to time a fresh build for each variant.
            pipeline.invalidate_particle_state()
            wp.synchronize_device(solver.device)
            t0 = time.perf_counter()
            verts, tris = pipeline.mesh()
            wp.synchronize_device(solver.device)
            times.append((time.perf_counter() - t0) * 1000.0)
        q = triangle_quality(verts, tris)
        return verts, tris, {
            "ms": statistics.median(times[1:]),
            "verts": len(verts),
            "tris": len(tris),
            "quality_mean": float(q.mean()) if len(q) else 0.0,
            "backend": pipeline.render_extractor.backend,
        }
    finally:
        for k, v in saved.items():
            setattr(s, k, v)
        solver.surface.built_key = None
        solver.surface.invalidate()


def run(args):
    import warp as wp

    from fluidSim_warp_blender51.sim.solver import FluidSolver
    from tools.profiling import profile_gpu_solver as P

    wp.init()
    ns = argparse.Namespace(
        device=args.device, voxel_size=args.voxel_size,
        max_particles=args.max_particles, min_substeps=None, max_substeps=None,
        max_whitewater=0, no_surface=False, whitewater=False,
        scene_preset=args.solver_preset, no_whitewater=True, viscosity=False,
        surface_tension=False)
    settings = P._build_settings(ns)
    solver = FluidSolver(settings)
    solver.set_source_batch(P.build_sources(ns))
    wanted = sorted(set(args.frames))
    frames = []
    try:
        for frame in range(1, max(wanted) + 1):
            if frame == args.emit_frames + 1:
                solver.set_source_batch()
            solver.step_frame()
            if frame not in wanted:
                continue
            record = {"frame": frame,
                      "particles": int(solver.particles.count.numpy()[0])}
            for name, overrides in VARIANTS.items():
                verts, tris, stats = measure_variant(solver, overrides, args.repeats)
                record[name] = stats
                if args.obj_dir:
                    write_obj(os.path.join(args.obj_dir, f"f{frame:03d}_{name}.obj"),
                              verts, tris)
            frames.append(record)
            print(json.dumps(record))
    finally:
        solver.release()
    report = {"preset": args.solver_preset, "device": str(args.device),
              "voxel_size": settings.voxel_size, "frames": frames}
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print("COMPARE-SHARP-SURFACING-DONE")
    return report


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solver-preset", choices=("dam_break", "benchmark"),
                   default="dam_break")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--voxel-size", type=float, default=None)
    p.add_argument("--max-particles", type=int, default=None)
    p.add_argument("--frames", type=int, nargs="+", default=[20, 40])
    p.add_argument("--emit-frames", type=int, default=6)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--obj-dir")
    p.add_argument("--json-out")
    args = p.parse_args(argv)
    if min(args.frames) < 1:
        p.error("--frames must be at least 1")
    if args.emit_frames < 0:
        p.error("--emit-frames must be nonnegative")
    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    return args


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    run(parse(argv))
