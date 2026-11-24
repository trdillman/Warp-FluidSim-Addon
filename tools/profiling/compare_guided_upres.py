"""Guided shell upres trial: coarse vs guided-fine (vs optional full fine).

Runs the bpy-free dam break coarse, steps a GuidedUpres fine solver behind it
every frame, and reports per-frame ms, particle counts and surface sizes.
With ``--full-fine`` it also runs an unguided full-depth sim at the fine
voxel size, for the speed comparison Nielsen and Bridson report.

    python tools/profiling/compare_guided_upres.py --device cuda:0 \\
        --voxel-size 0.06 --factor 2 --shell 4 --frames 20 40 \\
        --obj-dir artifacts/guided_upres --json-out artifacts/guided_upres/report.json

With ``--demo NAME`` it runs inside Blender on a demo scene instead: the
demo's own add-on solver is the coarse sim (colliders and all), and the run
saves a review file, ``artifacts/guided_upres/NAME_review.blend``, holding
the coarse, guided and full fine surfaces of the sampled frames in one
collection each (only the guided one is shown at first; each mesh is visible
on its own timeline frame). ``upres_cove`` is the scene made for this; when
``demos/NAME.blend`` has not been built yet, the scene is authored on the fly
from ``tools/demos/build_demo_scenes.py``. The full fine sim runs after the
guided one, so the two fine solves never hold VRAM at the same time. Demo
mode requires static collider, source and force inputs because it steps the
solvers without advancing the timeline. ``--full-fine`` requires a scene
without automatic particle-object seeds.

From PowerShell, with a private extension folder: ``--factory-startup``
otherwise lets Blender's wheel sync delete the installed extension wheels
(Warp included) from your regular profile (see AGENTS.md):

    $env:BLENDER_USER_RESOURCES = Join-Path $env:TEMP ("fluidsim-upres-" + [guid]::NewGuid())
    $env:BLENDER_USER_EXTENSIONS = Join-Path $env:BLENDER_USER_RESOURCES 'extensions'
    blender --background --factory-startup --python-use-system-env --python tools/profiling/compare_guided_upres.py -- --demo upres_cove --full-fine --phases --json-out artifacts/guided_upres/upres_cove.json
"""

from __future__ import annotations

import argparse
import gc
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


def _solver_stats(stats, profiler):
    """Where a fine frame went: substeps, pressure work, grid size, phases."""
    if stats is None:
        return None
    out = {name: getattr(stats, name, None) for name in (
        "substeps", "pressure_iterations", "active_voxels", "liquid_voxels")}
    if profiler is not None:
        out["phase_ms"] = {key: round(float(value), 3)
                           for key, value in sorted(profiler.frame.timers.items())}
    return out


def _write_obj(path, verts, tris):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts)
        fh.writelines(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n" for t in tris)


def run(args):
    if args.demo:
        return run_demo(args)
    import warp as wp

    from fluidSim_warp_blender51.sim.guided_upres import GuidedUpres
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    wp.init()
    settings, sources = _settings(args, args.voxel_size)
    coarse = FluidSolver(settings)
    coarse.set_source_batch(sources)
    fine_prof = full_prof = None
    if args.phases:
        from fluidSim_warp_blender51.diagnostics.profiler import Profiler

        fine_prof = Profiler(device=args.device)
        full_prof = Profiler(device=args.device)
    upres = GuidedUpres(coarse, factor=args.factor, shell_voxels=args.shell,
                        core_shells=args.core_shells, profiler=fine_prof)
    upres.fine.settings.surface_detail_enabled = bool(args.sharp)
    full = None
    if args.full_fine:
        fs, fsrc = _settings(args, args.voxel_size / args.factor)
        fs.max_particles = int(settings.max_particles * args.factor ** 3)
        full = FluidSolver(fs, profiler=full_prof)
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
            fine_stats, fine_ms = _timed(coarse.device, upres.step_frame)
            full_ms = full_stats = None
            if full is not None:
                full_stats, full_ms = _timed(coarse.device, full.step_frame)
            rec = {
                "frame": frame,
                "coarse_ms": coarse_ms,
                "guided_fine_ms": fine_ms,
                # What the upres really costs: the coarse solve it follows too.
                "guided_total_ms": coarse_ms + fine_ms,
                "full_fine_ms": full_ms,
                "guided_fine": _solver_stats(fine_stats, fine_prof),
                "full_fine": _solver_stats(full_stats, full_prof),
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
              "shell": args.shell, "core_shells": args.core_shells,
              "summary": summarize(frames), "frames": frames}
    return _finish(args, report)


def _finish(args, report):
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print("COMPARE-GUIDED-UPRES-DONE")
    return report


# --- Demo mode (inside Blender) -------------------------------------------

REVIEW_SETS = (("coarse", "Coarse"), ("guided", "Guided upres"), ("full", "Full fine"))


def demo_defaults(name):
    """Upres defaults a demo publishes in ``tools/demos/<name>.py``, if any."""
    import importlib

    try:
        mod = importlib.import_module(f"tools.demos.{name}")
    except ImportError:
        return None
    return mod if hasattr(mod, "UPRES_FACTOR") else None


def resolve_demo_args(args, mod):
    """Fill the options left unset from the demo's defaults (or the generic ones)."""
    def pick(value, attr, fallback):
        if value is not None:
            return value
        return getattr(mod, attr, fallback) if mod is not None else fallback

    args.factor = int(pick(args.factor, "UPRES_FACTOR", 2))
    args.shell = float(pick(args.shell, "SHELL_VOXELS", 4.0))
    args.core_shells = float(pick(args.core_shells, "CORE_SHELLS", 2.0))
    args.frames = list(pick(args.frames, "REVIEW_FRAMES", (20, 40)))
    return args


def open_demo(name):
    """Open ``demos/<name>.blend``, or author it when it has not been built."""
    import bpy

    from tools.repo_addon_loader import enable_repo_addon

    path = os.path.join(REPO, "demos", name + ".blend")
    if os.path.exists(path):
        bpy.ops.wm.open_mainfile(filepath=path)
        enable_repo_addon(REPO, default_set=False, persistent=False)
        return bpy.context.scene, path
    enable_repo_addon(REPO, default_set=False, persistent=False)
    from tools.demos import build_demo_scenes as builder

    author = getattr(builder, f"author_{name}", None)
    if author is None:
        raise FileNotFoundError(
            f"{path} does not exist and build_demo_scenes has no author_{name}")
    return author(), None


def _mesh_object(name, verts, tris, collection, material, frame, frame_start):
    """A surface mesh shown only on timeline ``frame``."""
    import bpy
    import numpy as np

    mesh = bpy.data.meshes.new(name)
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    mesh.from_pydata(v.tolist(), [], t.tolist())
    mesh.validate()
    if material is not None:
        mesh.materials.append(material)
    mesh.shade_smooth()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    keys = [(frame, False), (frame + 1, True)]
    if frame > frame_start:
        keys.insert(0, (frame_start, True))
    for f, hidden in keys:
        obj.hide_viewport = obj.hide_render = hidden
        obj.keyframe_insert("hide_viewport", frame=f)
        obj.keyframe_insert("hide_render", frame=f)
    return obj


class ReviewFile:
    """Collections of sampled surfaces, saved as a separate review .blend."""

    def __init__(self, scene, demo):
        import bpy

        self.scene = scene
        self.demo = demo
        self.material = getattr(scene.fluidsim, "surface_material", None)
        self.collections = {}
        root = bpy.data.collections.new(f"Upres Review ({demo})")
        scene.collection.children.link(root)
        for key, label in REVIEW_SETS:
            col = bpy.data.collections.new(f"{label} ({demo})")
            root.children.link(col)
            # Guided upres shows first; tick the others on to compare.
            col.hide_viewport = col.hide_render = key != "guided"
            self.collections[key] = col
        self.count = {key: 0 for key, _ in REVIEW_SETS}

    def add(self, key, frame, verts, tris):
        label = dict(REVIEW_SETS)[key]
        _mesh_object(f"{label} f{frame:03d}", verts, tris, self.collections[key],
                     self.material, frame, self.scene.frame_start)
        self.count[key] += 1

    def save(self, path, report):
        import bpy

        fs = self.scene.fluidsim
        # The review file is for looking: no live solve, no live liquid.
        for prop, value in (("live", False), ("show_surface", False),
                            ("show_particles", False)):
            if hasattr(fs, prop):
                setattr(fs, prop, value)
        text = bpy.data.texts.get("README_Upres_Review") or bpy.data.texts.new(
            "README_Upres_Review")
        text.clear()
        text.write(
            f"Guided upres review for {self.demo}\n\n"
            "Each collection under 'Upres Review' holds one method's surface;\n"
            "each mesh shows only on its own frame. Guided upres is on; tick\n"
            "Coarse or Full fine on (and Guided off) to compare at the same\n"
            "camera. Live simulation is off in this file.\n\n"
            + json.dumps({k: report[k] for k in report if k != "frames"}, indent=2)
            + "\n")
        self.scene.frame_set(min(report["review_frames"]))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path), copy=True,
                                    check_existing=False)
        print(f"review file: {path}")


def animated_inputs(objects):
    """Names of simulation inputs driven by animation or constraints.

    Demo mode steps the solvers without advancing the timeline, so such an
    input would sit frozen at its first pose.
    """
    def animated(data):
        anim = getattr(data, "animation_data", None)
        return anim is not None and (anim.action is not None or anim.drivers
                                     or getattr(anim, "nla_tracks", ()))

    out = []
    for obj in objects:
        settings = getattr(obj, "fluidsim", None)
        if (getattr(settings, "role", "") not in ("COLLIDER", "SOURCE", "FORCE")
                or not getattr(settings, "enabled", True)):
            continue
        node = obj
        while node is not None:
            data = getattr(node, "data", None)
            if (animated(node) or animated(data) or animated(getattr(data, "shape_keys", None))
                    or getattr(node, "constraints", ())):
                out.append(obj.name)
                break
            node = getattr(node, "parent", None)
    return out


def automatic_seed_objects(objects):
    """Identify automatic particle seed inputs before Blender admission."""
    out = []
    for obj in objects:
        settings = getattr(obj, "fluidsim", None)
        if getattr(settings, "enabled", False) and getattr(settings, "auto_inject_seed", False):
            out.append(obj.name)
    return out


def _full_fine_settings(coarse_settings, factor, mod):
    if mod is not None and hasattr(mod, "fine_settings"):
        return mod.fine_settings(coarse_settings, factor)
    import dataclasses

    return dataclasses.replace(
        coarse_settings,
        voxel_size=float(coarse_settings.voxel_size) / factor,
        max_particles=int(coarse_settings.max_particles * factor ** 3),
        leaf_capacity=int(coarse_settings.leaf_capacity * factor ** 3),
        whitewater_enabled=False)


def run_demo(args):
    import bpy

    from tools.demos.metadata import assign_input_identities

    mod = demo_defaults(args.demo)
    resolve_demo_args(args, mod)
    scene, path = open_demo(args.demo)
    # Opening the demo reloads the repository add-on. Use the same module
    # instances as its registered Blender operators.
    from fluidSim_warp_blender51.blender.scene_bridge import session
    from fluidSim_warp_blender51.sim.guided_upres import GuidedUpres, fine_colliders
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    wanted = sorted(set(args.frames))
    last = max(wanted)
    fs = scene.fluidsim
    if hasattr(fs, "live"):
        fs.live = False  # this run steps the solvers itself
    scene.frame_set(scene.frame_start)
    ds = fs.domain_object.fluidsim_domain
    if args.voxel_size is not None:
        ds.voxel_size = float(args.voxel_size)
    if args.max_particles is not None:
        ds.max_particles = int(args.max_particles)
    moving = animated_inputs(scene.objects)
    if moving:
        raise NotImplementedError(
            f"{args.demo}: --demo does not replay animated inputs yet "
            f"({', '.join(sorted(moving))})")
    if args.full_fine and (seeded := automatic_seed_objects(scene.objects)):
        raise NotImplementedError(
            f"{args.demo}: --full-fine cannot reproduce particle-object seeds "
            f"({', '.join(sorted(seeded))}). Run without it")
    assign_input_identities(scene)
    if bpy.ops.fluidsim.initialize() != {"FINISHED"}:
        raise RuntimeError(f"{args.demo}: the add-on refused to initialize the scene")
    sess = session()
    coarse = sess.solver
    if coarse is None:
        raise RuntimeError(f"{args.demo}: no solver after initialize")
    cs = coarse.settings
    print(f"{args.demo}: coarse {cs.voxel_size:.4f} m, fine "
          f"{cs.voxel_size / args.factor:.4f} m, frames {wanted}")

    fine_prof = full_prof = None
    if args.phases:
        from fluidSim_warp_blender51.diagnostics.profiler import Profiler

        fine_prof = Profiler(device=coarse.device)
        full_prof = Profiler(device=coarse.device)
    leaf_scale = mod.leaf_budget_scale() if mod is not None else None
    upres = GuidedUpres(coarse, factor=args.factor, shell_voxels=args.shell,
                        core_shells=args.core_shells, leaf_budget_scale=leaf_scale,
                        profiler=fine_prof)
    upres.fine.settings.surface_detail_enabled = bool(args.sharp)
    review = ReviewFile(scene, args.demo) if args.review_blend != "none" else None
    records = []

    def grab(key, frame, solver):
        verts, tris = solver.surface_mesh()
        if review is not None:
            review.add(key, frame, verts, tris)
        if args.obj_dir:
            _write_obj(os.path.join(args.obj_dir, f"{args.demo}_f{frame:03d}_{key}.obj"),
                       verts, tris)
        return len(tris)

    # Pass 1: the demo's own solve as the coarse sim, guided fine behind it.
    try:
        for frame in range(1, last + 1):
            _, coarse_ms = _timed(coarse.device, coarse.step_frame)
            fine_stats, fine_ms = _timed(coarse.device, upres.step_frame)
            rec = {
                "frame": frame,
                "coarse_ms": coarse_ms,
                "guided_fine_ms": fine_ms,
                "guided_total_ms": coarse_ms + fine_ms,
                "full_fine_ms": None,
                "guided_fine": _solver_stats(fine_stats, fine_prof),
                "full_fine": None,
                "coarse_particles": int(coarse.particles.count.numpy()[0]),
                "fine_particles": int(upres.fine.particles.count.numpy()[0]),
            }
            if frame in wanted:
                rec["coarse_tris"] = grab("coarse", frame, coarse)
                rec["fine_tris"] = grab("guided", frame, upres.fine)
            print(json.dumps(rec))
            records.append(rec)
    finally:
        # Drop every reference too: release() keeps the fine solver object,
        # and pass 2 must not allocate while it still holds VRAM.
        upres.release()
        upres = None
        gc.collect()

    # Pass 2: the unguided full sim at the fine size, from the same inputs.
    if args.full_fine:
        full = FluidSolver(_full_fine_settings(cs, args.factor, mod), profiler=full_prof)
        try:
            full.set_colliders(fine_colliders(getattr(coarse, "colliders", ())))
            batch = coarse.sourcing.batch
            full.set_source_batch(batch.legacy_descriptors, batch.native_plans)
            fields = getattr(coarse, "force_fields", None)
            if fields is not None:
                full.set_force_fields(fields)
            for rec in records:
                full_stats, full_ms = _timed(coarse.device, full.step_frame)
                rec["full_fine_ms"] = full_ms
                rec["full_fine"] = _solver_stats(full_stats, full_prof)
                rec["full_fine_particles"] = int(full.particles.count.numpy()[0])
                if rec["frame"] in wanted:
                    rec["full_tris"] = grab("full", rec["frame"], full)
                print(json.dumps({k: rec[k] for k in (
                    "frame", "full_fine_ms", "full_fine_particles") if k in rec}))
        finally:
            full.release()
    bpy.ops.fluidsim.release()

    report = {
        "demo": args.demo,
        "demo_file": path or "authored on the fly",
        "voxel_size": float(cs.voxel_size),
        "fine_voxel_size": float(cs.voxel_size) / args.factor,
        "factor": args.factor, "shell": args.shell, "core_shells": args.core_shells,
        "review_frames": wanted,
        "summary": summarize(records),
        "frames": records,
    }
    if review is not None:
        out = args.review_blend or os.path.join(
            REPO, "artifacts", "guided_upres", f"{args.demo}_review.blend")
        review.save(out, report)
        report["review_blend"] = out
    return _finish(args, report)


def summarize(records, skip=1):
    """Mean ms per method over the frames after ``skip`` warm-up frames."""
    import statistics

    rows = records[skip:] or records
    out = {}
    for key in ("coarse_ms", "guided_fine_ms", "guided_total_ms", "full_fine_ms"):
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            out[f"mean_{key}"] = round(statistics.fmean(values), 3)
    total, full = out.get("mean_guided_total_ms"), out.get("mean_full_fine_ms")
    if total and full:
        # Above 1: the upres (coarse + guided fine) beats the full fine sim.
        out["speedup_vs_full_fine"] = round(full / total, 3)
    return out


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo",
                   help="run inside Blender on demos/<name>.blend (upres_cove is "
                        "the scene made for this) and save a review .blend")
    p.add_argument("--review-blend",
                   help="--demo: where to save the review file (default "
                        "artifacts/guided_upres/<demo>_review.blend; 'none' skips it)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--voxel-size", type=float, default=None,
                   help="coarse voxel size (default 0.06, or the demo's own)")
    p.add_argument("--max-particles", type=int, default=None)
    p.add_argument("--factor", type=int, default=None, help="default 2, or the demo's")
    p.add_argument("--shell", type=float, default=None, help="default 4, or the demo's")
    p.add_argument("--core-shells", type=float, default=None,
                   help="fine liquid deeper than this many shells is a coarse-driven "
                        "solid in the fine pressure solve (0 solves the whole depth; "
                        "default 2, or the demo's)")
    p.add_argument("--phases", action="store_true",
                   help="record per-phase ms for the guided and full fine solvers")
    p.add_argument("--frames", type=int, nargs="+", default=None,
                   help="frames to sample (default 20 40, or the demo's review frames)")
    p.add_argument("--emit-frames", type=int, default=6)
    p.add_argument("--sharp", action="store_true",
                   help="also turn on Sharp Surfacing for the fine surface")
    p.add_argument("--full-fine", action="store_true",
                   help="also run an unguided full-depth sim at the fine size")
    p.add_argument("--obj-dir")
    p.add_argument("--json-out")
    args = p.parse_args(argv)
    if not args.demo:
        # The bpy-free dam break keeps its historical defaults.
        args.voxel_size = 0.06 if args.voxel_size is None else args.voxel_size
        args.factor = 2 if args.factor is None else args.factor
        args.shell = 4.0 if args.shell is None else args.shell
        args.core_shells = 2.0 if args.core_shells is None else args.core_shells
        args.frames = [20, 40] if args.frames is None else args.frames
    return args


def main(argv=None):
    if argv is None:
        # Under Blender the script's own options follow a bare "--".
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return run(parse(argv))


if __name__ == "__main__":
    main()
