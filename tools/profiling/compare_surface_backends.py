"""Compare the liquid surface backends on the same SDF: sparse, dense, nets.

For every sampled frame the liquid SDF is built once, then each backend
extracts a mesh from that same SDF, so the comparison is apples to apples.
Reported per backend and frame:

- ``ms``: median extraction time (includes the copy back to the CPU)
- ``verts`` / ``tris``
- triangle quality (1 = equilateral, 0 = flat): mean, 5th percentile, and the
  share of slivers (quality < 0.1)
- ``open_edges`` / ``bad_edges``: edges used by one / by more than two
  triangles (a clean closed surface has none; liquid cut by the domain wall
  legitimately has open edges, so compare backends against each other)
- ``repeatable``: two fresh extractors give identical mesh bytes

Run on a saved demo (needs Blender; use this for dam_break and crown_splash):

    blender --background --factory-startup --python \\
        tools/profiling/compare_surface_backends.py -- \\
        --demo dam_break --frames 20 60 120 --json-out artifacts/surface_dam_break.json

Or without Blender, on the bpy-free dam-break benchmark scene:

    python tools/profiling/compare_surface_backends.py --solver-preset dam_break \\
        --device cuda:0 --frames 8 16 --json-out artifacts/surface_solver.json

This measures triangle output only.
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

BACKENDS = ("sparse", "dense", "nets")
SLIVER_QUALITY = 0.1


def triangle_quality(verts, tris):
    """Per-triangle quality 4*sqrt(3)*area / sum(edge^2)."""
    import numpy as np

    if len(tris) == 0:
        return np.zeros(0)
    v = np.asarray(verts, np.float64)[np.asarray(tris)]
    e0, e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(e0, -e2), axis=1)
    denom = (e0 ** 2).sum(1) + (e1 ** 2).sum(1) + (e2 ** 2).sum(1)
    return 4.0 * np.sqrt(3.0) * area / np.maximum(denom, 1e-30)


def edge_report(tris):
    """Count edges used once (open) and more than twice (non-manifold)."""
    import numpy as np

    if len(tris) == 0:
        return 0, 0
    t = np.asarray(tris)
    edges = np.sort(np.concatenate(
        [t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts > 2).sum())


def mesh_stats(verts, tris):
    import numpy as np

    q = triangle_quality(verts, tris)
    open_edges, bad_edges = edge_report(tris)
    return {
        "verts": int(len(verts)),
        "tris": int(len(tris)),
        "quality_mean": float(q.mean()) if len(q) else 0.0,
        "quality_p5": float(np.percentile(q, 5)) if len(q) else 0.0,
        "sliver_percent": float(100.0 * (q < SLIVER_QUALITY).mean()) if len(q) else 0.0,
        "open_edges": open_edges,
        "bad_edges": bad_edges,
    }


def _backend_available(name):
    from fluidSim_warp_blender51.sim import surface as SURF

    return {"sparse": SURF.SPARSE_MARCHING_CUBES_AVAILABLE,
            "dense": True,
            "nets": SURF.SURFACE_NETS_AVAILABLE}[name]


def compare_backends_on_sdf(grid, sdf_field, device, max_dense_dim, *,
                            repeats=5, backends=BACKENDS, iso=0.0):
    """Extract ``sdf_field`` with every backend and return one record each."""
    import warp as wp

    from fluidSim_warp_blender51.sim import surface as SURF

    out = {}
    for name in backends:
        if not _backend_available(name):
            out[name] = {"skipped": "needs a newer Warp"}
            continue
        try:
            ext = SURF.SurfaceExtractor(grid, device, max_dense_dim=max_dense_dim,
                                        backend=name)
            verts, tris = ext.extract(sdf_field, iso)  # warm-up / first build
            times = []
            for _ in range(max(1, repeats)):
                wp.synchronize_device(device)
                t0 = time.perf_counter()
                verts, tris = ext.extract(sdf_field, iso)
                wp.synchronize_device(device)
                times.append((time.perf_counter() - t0) * 1000.0)
            fresh = SURF.SurfaceExtractor(grid, device, max_dense_dim=max_dense_dim,
                                          backend=name)
            v2, t2 = fresh.extract(sdf_field, iso)
            record = mesh_stats(verts, tris)
            record["ms"] = statistics.median(times)
            record["ms_min"] = min(times)
            record["repeatable"] = bool(
                verts.tobytes() == v2.tobytes() and tris.tobytes() == t2.tobytes())
            out[name] = record
        except Exception as exc:  # noqa: BLE001 - a backend failing is a result
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _print_frame(label, result):
    print(f"\n== {label}")
    print(f"{'backend':8} {'ms':>8} {'verts':>9} {'tris':>9} {'q_mean':>7} "
          f"{'q_p5':>6} {'sliver%':>8} {'open':>6} {'bad':>5} {'repeat':>7}")
    for name, r in result.items():
        if "skipped" in r or "error" in r:
            print(f"{name:8} {r.get('skipped') or r.get('error')}")
            continue
        print(f"{name:8} {r['ms']:8.2f} {r['verts']:9d} {r['tris']:9d} "
              f"{r['quality_mean']:7.3f} {r['quality_p5']:6.3f} "
              f"{r['sliver_percent']:8.2f} {r['open_edges']:6d} "
              f"{r['bad_edges']:5d} {str(r['repeatable']):>7}")


def _summary(frames):
    """Mean of each number per backend across frames."""
    summary = {}
    for name in BACKENDS:
        rows = [f["backends"][name] for f in frames
                if "ms" in f["backends"].get(name, {})]
        if not rows:
            continue
        summary[name] = {
            "frames": len(rows),
            "ms_mean": statistics.fmean(r["ms"] for r in rows),
            "quality_mean": statistics.fmean(r["quality_mean"] for r in rows),
            "sliver_percent_mean": statistics.fmean(r["sliver_percent"] for r in rows),
            "tris_mean": statistics.fmean(r["tris"] for r in rows),
            "all_repeatable": all(r["repeatable"] for r in rows),
        }
    return summary


def _finish(args, source, frames, extra):
    summary = _summary(frames)
    print("\n== summary (mean over sampled frames)")
    for name, s in summary.items():
        print(f"{name:8} {s['ms_mean']:8.2f} ms  quality {s['quality_mean']:.3f}  "
              f"slivers {s['sliver_percent_mean']:.2f}%  tris {s['tris_mean']:.0f}  "
              f"repeatable={s['all_repeatable']}")
    report = {"source": source, "frames": frames, "summary": summary, **extra}
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.json_out}")
    print("COMPARE-SURFACE-BACKENDS-DONE")
    return report


def _sample(solver, frame, args):
    """Build the liquid SDF for the current frame and compare all backends."""
    import warp as wp

    pipeline = solver.surface
    pipeline.mesh()  # makes sure render_sdf is current for this frame
    wp.synchronize_device(solver.device)
    max_dense = int(pipeline.render_extractor.max_dense_dim)
    return {
        "frame": int(frame),
        "backends": compare_backends_on_sdf(
            pipeline.grid, pipeline.render_sdf, solver.device, max_dense,
            repeats=args.repeats),
    }


def run_blender_demo(args):
    import bpy

    from tools.repo_addon_loader import enable_repo_addon

    path = os.path.join(REPO, "demos", args.demo + ".blend")
    bpy.ops.wm.open_mainfile(filepath=path)
    enable_repo_addon(REPO, default_set=False, persistent=False)
    from fluidSim_warp_blender51.blender import scene_bridge

    scene = bpy.context.scene
    wanted = sorted(set(args.frames))
    frames = []
    for frame in range(scene.frame_start + 1, max(wanted) + 1):
        scene.frame_set(frame)  # the real playback path, colliders included
        if frame not in wanted:
            continue
        sess = scene_bridge.session()
        if not (sess.active and sess.solver is not None):
            raise RuntimeError(f"solver inactive at frame {frame}")
        record = _sample(sess.solver, frame, args)
        _print_frame(f"{args.demo} frame {frame}", record["backends"])
        frames.append(record)
    return _finish(args, f"demo:{args.demo}", frames, {})


def run_solver_preset(args):
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
            if frame == args.emit_frames:
                solver.set_source_batch()
            solver.step_frame()
            if frame not in wanted:
                continue
            record = _sample(solver, frame, args)
            _print_frame(f"{args.solver_preset} frame {frame}", record["backends"])
            frames.append(record)
    finally:
        solver.release()
    return _finish(args, f"solver-preset:{args.solver_preset}", frames,
                   {"device": str(args.device)})


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", help="saved demo in demos/ (needs Blender)")
    src.add_argument("--solver-preset", choices=("dam_break", "benchmark"),
                     help="bpy-free solver scene from profile_gpu_solver")
    p.add_argument("--frames", type=int, nargs="+", default=[20, 60],
                   help="frames to sample (default: 20 60)")
    p.add_argument("--repeats", type=int, default=5,
                   help="timed extractions per backend per frame")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--voxel-size", type=float, default=None)
    p.add_argument("--max-particles", type=int, default=None)
    p.add_argument("--emit-frames", type=int, default=4)
    p.add_argument("--json-out")
    return p.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parse(argv)
    if args.demo:
        return run_blender_demo(args)
    return run_solver_preset(args)


if __name__ == "__main__":
    main()
