# Unreleased: surface upres pipeline

Status: stage 2 (Sharp Surfacing) implemented and checked on CUDA. Stage 1
(guided shell upres) has its solver core and CPU kernel tests; **no GPU run,
Blender settings, live playback or bake post-pass yet**.
Research survey and owner decisions: Claude Docs "Surface upres: research
survey" (23 November 2026).

The goal is detail on low-resolution liquid simulations of any kind without
the cost of a full high-resolution simulation. Two stages, both opt-in:

1. **Guided shell upres** (core built, see below). Re-simulate only a thin outer shell of
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
- Preserve Splashes counts particles on the simulation grid. At each render
  voxel it samples the surrounding 3x3x3 simulation cells and blends each
  Smooth pass toward its input. The weight is 1 in calm liquid and
  `1 - Preserve` where there are no particles nearby.

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

## Guided shell upres (core only)

`sim/guided_upres.py` runs a second, finer `FluidSolver` with the narrow band
on, behind a coarse one. Each frame the coarse liquid SDF and MAC velocity are
copied into a `GuideFrame`; the fine narrow band's `guide` hook then:

- marks fine leaves wherever the coarse liquid is (`mark_leaves`);
- sets the fine phi to the coarse phi eroded by the shell, so deep liquid is
  liquid and the outer `shell_voxels` fine voxels are left to fine particles,
  and caps deep fine liquid at the coarse surface plus the shell
  (`apply_phi`, after advection and again after the band union);
- writes the coarse velocity, blended in time between the two coarse frames,
  into the fine faces deeper than the shell (`apply_velocity`).

- fills the coarse shell with fine particles on the first frame, then only
  where the coarse liquid advances, at the coarse velocity (`seed`).

Before use, each coarse level set is rebuilt as a true inside distance
(`GuideFrame.reconstruct_depth`), because a particle-splat SDF is never
deeper than a particle radius. The fine solve only carries particles in the
shell, so its particle budget follows the surface area (`factor**2`); its
leaves cover the whole coarse liquid, so the leaf budget follows the volume
(`factor**3`). Dynamic (two-way) colliders are refused for now; other
colliders are copied, so the fine solve never writes into the coarse ones.
`GuidedUpres(coarse, factor=2, shell_voxels=4)` builds the fine solver from
the coarse settings. Not wired to Blender yet.

```
python -m pytest -q tests/solver/test_guided_upres_kernels.py
python tools/profiling/compare_guided_upres.py --device cuda:0 --voxel-size 0.06 --factor 2 --shell 4 --frames 20 40 --full-fine --obj-dir artifacts/guided_upres --json-out artifacts/guided_upres/report.json
```

The second command steps the coarse dam break, the guided fine solve and
(with `--full-fine`) an unguided full solve at the fine size, and prints
per-frame ms, particle counts and triangle counts, plus OBJ meshes.

## How to check it (Windows GPU host)

```
python -m pytest -q tests/surface/test_surface_sharp_surfacing.py
python tools/profiling/compare_sharp_surfacing.py --solver-preset dam_break --device cuda:0 --voxel-size 0.06 --frames 20 40 --obj-dir artifacts/sharp_surfacing --json-out artifacts/sharp_surfacing.json
```

The second command prints, per frame, the render-surface build+extract time,
triangle count and triangle quality for plain, Sharp Surfacing, and Sharp
Surfacing with Preserve Splashes 0.75, and writes each mesh as OBJ.
For timing evidence, run the comparison under an exclusive GPU lease in
Blender's Python with the locked Warp version. Pass the comparison arguments
after Blender's `--`, and retain the lease report alongside the comparison JSON.
