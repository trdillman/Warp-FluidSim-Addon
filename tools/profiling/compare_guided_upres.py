"""Guided shell upres trial: coarse vs guided-fine (vs optional full fine).

Runs the bpy-free dam break coarse, steps a GuidedUpres fine solver behind it
every frame, and reports per-frame ms, particle counts and surface sizes.
With ``--full-fine`` it also runs an unguided full-depth sim at the fine
voxel size, for the speed comparison Nielsen and Bridson report.

    python tools/profiling/compare_guided_upres.py --device cuda:0 \\
        --voxel-size 0.06 --factor 2 --shell 4 --frames 20 40 \\
        --obj-dir artifacts/guided_upres --json-out artifacts/guided_upres/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _settings(args, voxel_size):
    from tools.profiling import profile_gpu_solver as P

    ns = argparse.Namespace(
        device=args.device, voxel_size=voxel_size,
        max_particles=args.max_particles, min_substeps=None, max_substeps=None,
        max_whitewater=0, no_surface=False, whitewater=False,
        scene_preset="dam_break", no_whitewater=True, viscosity=False,
        surface_tension=False)
    return P._build_settings(ns), P.build_sources(ns)


def _timed(device, fn):
    import warp as wp

    wp.synchronize_device(device)
    t0 = time.perf_counter()
    out = fn()
    wp.synchronize_device(device)
    return out, (time.perf_counter() - t0) * 1000.0


def _write_obj(path, verts, tris):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts)
        fh.writelines(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n" for t in tris)


def run(args):
    import warp as wp

    from fluidSim_warp_blender51.sim.guided_upres import GuidedUpres
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    wp.init()
    settings, sources = _settings(args, args.voxel_size)
    coarse = FluidSolver(settings)
    coarse.set_source_batch(sources)
    upres = GuidedUpres(coarse, factor=args.factor, shell_voxels=args.shell)
    upres.fine.settings.surface_detail_enabled = bool(args.sharp)
    full = None
    if args.full_fine:
        fs, fsrc = _settings(args, args.voxel_size / args.factor)
        fs.max_particles = int(settings.max_particles * args.factor ** 3)
        full = FluidSolver(fs)
        full.set_source_batch(fsrc)
    wanted = set(args.frames)
    frames = []
    try:
        for frame in range(1, max(wanted) + 1):
            if frame == args.emit_frames + 1:
                coarse.set_source_batch()
                if full is not None:
                    full.set_source_batch()
            _, coarse_ms = _timed(coarse.device, coarse.step_frame)
            _, fine_ms = _timed(coarse.device, upres.step_frame)
            full_ms = None
            if full is not None:
                _, full_ms = _timed(coarse.device, full.step_frame)
            rec = {
                "frame": frame,
                "coarse_ms": coarse_ms,
                "guided_fine_ms": fine_ms,
                "full_fine_ms": full_ms,
                "coarse_particles": int(coarse.particles.count.numpy()[0]),
                "fine_particles": int(upres.fine.particles.count.numpy()[0]),
            }
            if full is not None:
                rec["full_fine_particles"] = int(full.particles.count.numpy()[0])
            if frame in wanted:
                cv, ct = coarse.surface_mesh()
                fv, ft = upres.fine.surface_mesh()
                rec["coarse_tris"], rec["fine_tris"] = len(ct), len(ft)
                if args.obj_dir:
                    _write_obj(os.path.join(args.obj_dir, f"f{frame:03d}_coarse.obj"), cv, ct)
                    _write_obj(os.path.join(args.obj_dir, f"f{frame:03d}_guided.obj"), fv, ft)
                    if full is not None:
                        uv, ut = full.surface_mesh()
                        _write_obj(os.path.join(args.obj_dir, f"f{frame:03d}_full.obj"), uv, ut)
            print(json.dumps(rec))
            frames.append(rec)
    finally:
        upres.release()
        coarse.release()
        if full is not None:
            full.release()
    report = {"voxel_size": args.voxel_size, "factor": args.factor,
              "shell": args.shell, "frames": frames}
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print("COMPARE-GUIDED-UPRES-DONE")
    return report


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--voxel-size", type=float, default=0.06)
    p.add_argument("--max-particles", type=int, default=None)
    p.add_argument("--factor", type=int, default=2)
    p.add_argument("--shell", type=float, default=4.0)
    p.add_argument("--frames", type=int, nargs="+", default=[20, 40])
    p.add_argument("--emit-frames", type=int, default=6)
    p.add_argument("--sharp", action="store_true",
                   help="also turn on Sharp Surfacing for the fine surface")
    p.add_argument("--full-fine", action="store_true",
                   help="also run an unguided full-depth sim at the fine size")
    p.add_argument("--obj-dir")
    p.add_argument("--json-out")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse(sys.argv[1:]))
