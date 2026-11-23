# Unreleased: surface upres pipeline

Status: stage 2 (Sharp Surfacing) implemented and unit-tested on CPU; **no
GPU or Blender run yet**. Stage 1 (guided shell upres) is planned, not built.
Research survey and owner decisions: Claude Docs "Surface upres: research
survey" (23 November 2026).

The goal is detail on low-resolution liquid simulations of any kind without
the cost of a full high-resolution simulation. Two stages, both opt-in:

1. **Guided shell upres** (planned). Re-simulate only a thin outer shell of
   the liquid at 2x to 4x finer resolution, driven from below by the coarse
   velocity (Nielsen and Bridson 2011, the idea behind Houdini FLIP upres and
   Bifrost guided liquids). Runs on a coarse bake and live. Reuses the
   narrow-band FLIP machinery.
2. **Sharp Surfacing** (this change). Mesh particles as ellipsoids shaped by
   their neighbours, plus splash-preserving smoothing.

## Sharp Surfacing

Settings (domain, Surface Reconstruction > Filters):

| Setting | Default | What it does |
|---|---|---|
| Sharp Surfacing | off | Draw each particle as an ellipsoid shaped by its neighbours (Yu and Turk 2013) |
| Neighbour Radius | 2.0 sim voxels | How far a particle looks for neighbours to choose its shape |
| Max Stretch | 4.0 | Cap on longest/shortest axis; volume is kept, so a disc's wide axes grow by Max Stretch^(1/3) |
| Flatten | 0.5 | Pull each particle toward its neighbours' weighted mean (removes ball bumps) |
| Preserve Splashes (Smooth pass) | 0.0 | Lower the Smooth pass where particles are sparse (Loeschner et al. 2023) |

How it works (`sim/surface_anisotropy.py`):

- A Warp hash grid finds neighbours. Each particle gets the weighted
  covariance of its neighbours (weight `1 - (d/r)^3`); its eigenvectors are
  the ellipsoid axes and the square roots of its eigenvalues set their
  relative lengths, capped by Max Stretch and normalised so the ellipsoid has
  the ball's volume. Fewer than 6 neighbours (isolated drops) keeps the ball.
- The splat writes the ray distance to the ellipsoid, `|v| (1 - 1/|G v|)`,
  with atomic min. It is exact for a ball, signed correctly, and zero exactly
  on the ellipsoid, so the mesh is the true union of ellipsoids.
- Preserve Splashes counts particles per voxel, sums the 3x3x3 neighbourhood,
  and blends each Smooth pass back toward its input by a weight that is 1 in
  calm liquid and `1 - Preserve` with no particles nearby.

Boundaries:

- Render surface only. In Simulation resolution mode the render surface gets
  its own field while either option is on, so the solver SDF read by surface
  tension, whitewater and effects is never built from ellipsoids.
- Works before meshing, so it feeds every extractor backend the same way:
  sparse and dense marching cubes, and Surface Nets (`FLUIDSIM_SURFACE_BACKEND=nets`,
  PR 81) once that lands.
- Both options are omitted from cache identity at their defaults
  (`io/cache.py`), so existing caches stay valid; changing them re-bakes.
- Not verified: GPU cost, the look on real demos, and mesh repeatability
  (the ellipsoid splat uses float atomic min like the ball splat, so the SDF
  is order-independent, but the hash grid build has not been checked).

## How to check it (Windows GPU host)

```
python -m pytest -q tests/surface/test_surface_sharp_surfacing.py
python tools/profiling/compare_sharp_surfacing.py --solver-preset dam_break --device cuda:0 --voxel-size 0.06 --frames 20 40 --obj-dir artifacts/sharp_surfacing --json-out artifacts/sharp_surfacing.json
```

The second command prints, per frame, the render-surface build+extract time,
triangle count and triangle quality for plain, Sharp Surfacing, and Sharp
Surfacing with Preserve Splashes 0.75, and writes each mesh as OBJ.
