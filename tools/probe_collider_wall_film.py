"""Headless probe for the collider wall film (no Blender).

A water column starts flush against the vertical inner face of a box collider
and collapses under gravity. Particles that hang on the face instead of
falling with the column form the "film". The probe counts particles within one
voxel of the face that are still more than 0.1 m above the column centre's
99th-percentile height.

    python tools/probe_collider_wall_film.py            # collider + domain-wall control
    python tools/probe_collider_wall_film.py --frames 30 --aligned

The domain-wall control puts the same column flush against the domain side
wall instead, which never showed the film.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import warp as wp  # noqa: E402

from fluidSim_warp_blender51.sim.solver import (  # noqa: E402
    ColliderDesc, FluidSolver, SolverSettings,
)

VOXEL = 0.03
SPACING = VOXEL / 2          # 8 particles per voxel
COLUMN_X = (0.0, 0.3)
COLUMN_HALF_Y = 0.3
COLUMN_TOP = 0.9
WALL_THICKNESS = 0.3


def box_collider(lo, hi, device, friction=0.0):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    verts = np.asarray([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    tris = np.asarray([
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
        2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
    ], dtype=np.int32)
    return ColliderDesc(
        mesh=wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device=device),
                     indices=wp.array(tris, dtype=wp.int32, device=device)),
        center=tuple(((verts.min(0) + verts.max(0)) / 2).tolist()),
        aabb_min=tuple(verts.min(0).tolist()),
        aabb_max=tuple(verts.max(0).tolist()),
        friction=friction, restitution=0.0)


def column_points():
    def axis(lo, hi):
        return np.arange(lo + SPACING / 2, hi, SPACING)
    xs = axis(*COLUMN_X)
    ys = axis(-COLUMN_HALF_Y, COLUMN_HALF_Y)
    zs = axis(0.0, COLUMN_TOP)
    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    return g.astype(np.float32)


def run(case, frames, aligned, device, friction=0.0):
    """Return per-frame film counts for one configuration.

    case "collider": box colliders fill |y| > 0.3 out to the domain walls.
    case "domain":   the domain side walls themselves sit at |y| = 0.3.
    """
    # Offset the domain by a fraction of a voxel unless faces should land on
    # voxel boundaries (the demo's faces were not grid aligned).
    shift = 0.0 if aligned else 0.382 * VOXEL
    if case == "collider":
        half_y = COLUMN_HALF_Y + WALL_THICKNESS
        lo = (-shift, -half_y - shift, 0.0)
        hi = (1.2 - shift, half_y - shift, 1.2)
    else:
        # Domain walls sit on voxel boundaries by construction (the grid
        # origin is domain_min), so they stay exactly at |y| = 0.3.
        lo = (-shift, -COLUMN_HALF_Y, 0.0)
        hi = (1.2 - shift, COLUMN_HALF_Y, 1.2)
    settings = SolverSettings(
        domain_min=lo, domain_max=hi, voxel_size=VOXEL,
        max_particles=400_000, max_whitewater=8, device=device)
    settings.surface_enabled = False
    settings.whitewater_enabled = False
    solver = FluidSolver(settings)
    if case == "collider":
        solver.set_colliders([
            box_collider((lo[0] - 0.1, COLUMN_HALF_Y, -0.1),
                         (0.75, hi[1] + 0.1, 1.4), device, friction),
            box_collider((lo[0] - 0.1, lo[1] - 0.1, -0.1),
                         (0.75, -COLUMN_HALF_Y, 1.4), device, friction),
        ])
    pts = column_points()
    solver.inject_particles(pts)
    counts = []
    for frame in range(1, frames + 1):
        solver.step_frame()
        p = solver.particle_positions()
        v = solver.particle_velocities()
        # Only the stretch of wall the column started against counts; the
        # outflow splashing up the far end is not film.
        footprint = p[:, 0] < COLUMN_X[1]
        centre = footprint & (np.abs(p[:, 1]) < 0.1)
        near = footprint & ((COLUMN_HALF_Y - np.abs(p[:, 1])) < VOXEL)
        ref = np.percentile(p[centre, 2], 99) if centre.any() else 0.0
        film = near & (p[:, 2] > ref + 0.1)
        speed = np.linalg.norm(v[film], axis=1)
        # No-penetration: nothing may end up past the wall faces.
        inside = (np.abs(p[:, 1]) > COLUMN_HALF_Y + 0.25 * VOXEL) & (p[:, 0] < 0.75 - VOXEL)
        worst = float(np.abs(p[:, 1]).max() - COLUMN_HALF_Y) if len(p) else 0.0
        counts.append((frame, int(film.sum()), int(inside.sum()), worst,
                       int((film & (p[:, 1] < 0)).sum()),
                       int((film & (p[:, 1] > 0)).sum()), float(ref),
                       float(np.median(speed)) if film.any() else 0.0,
                       float(p[film, 2].max()) if film.any() else 0.0))
    solver.release()
    return len(pts), counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Collider wall film probe")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--aligned", action="store_true",
                    help="put the wall faces on voxel boundaries")
    ap.add_argument("--friction", type=float, default=0.0)
    ap.add_argument("--cases", default="collider,domain")
    ap.add_argument("--every", type=int, default=4)
    args = ap.parse_args(argv)
    wp.init()
    device = "cuda:0"
    for case in args.cases.split(","):
        n, counts = run(case, args.frames, args.aligned, device, args.friction)
        print(f"[{case}] particles={n} aligned={args.aligned} friction={args.friction}")
        for frame, film, inside, worst, lo_n, hi_n, ref, med, zmax in counts:
            if frame % args.every == 0 or frame == args.frames:
                print(f"  frame {frame:3d}: film={film:6d} (-y {lo_n:5d}, +y {hi_n:5d})"
                      f"  centre_p99={ref:.3f}"
                      f"  film_med|v|={med:.3f}  film_zmax={zmax:.3f}"
                      f"  past_wall={inside}")


if __name__ == "__main__":
    main()
