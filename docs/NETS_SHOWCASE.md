# Nets Showcase

A high-resolution triple splash built to show the Surface Nets render surface.
Opt-in, like `tsunami_parthenon` and `upres_cove`: it is not in the Demo
Gallery or the demo gate, and its `.blend` is saved to `artifacts/review_demos/`.

Three drops of different sizes fall into a long, shallow marble trough one after
the other, left to right: a crown, then a drop into its ripples, then a drop
into the wreckage, with the crown sheets meeting in the middle. Slow motion
(Time Scale 0.3), 144 frames.

## How much finer than before

| | Upres Cove (finest before) | Nets Showcase hero | Nets Showcase draft |
|---|---|---|---|
| Voxel (sim and surface) | 20 mm | **1 mm** | 2 mm |
| Cells along the longest axis | 280 | **600** (2.1x) | 300 |
| Cells in the whole domain | 2.2 M | **47 M** (21x) | 5.9 M |
| Particles at start | ~864 k | **~4.1 M** (4.7x), Narrow Band off | ~0.51 M |

The shipped demos are coarser still (dam_break 187 cells, crown_splash 80).
"Twice the fidelity" is read as twice the cells along the longest axis, with
the domain cell count and particle count higher multiples. The scenes are
different sizes (Upres Cove is a 5.6 m cove, this is a 0.6 m domain around a 0.32 m trough), so the
voxel size in millimetres is the plainest comparison. The particle count is a
estimate until the build reports the real one (in `meta.json`).
The numbers live in `tools/demos/nets_showcase.py` and are checked by
`tests/demos/test_nets_showcase.py`.

## Layout choices that come from the Surface Nets review

- No liquid touches a domain wall. Surface Nets stops half a voxel short of its
  extraction box where liquid meets a domain wall, which would show as a gap. The
  pool sits on a collider slab 16 mm above the domain floor and the trough walls
  (180 mm) are well above the spray. The first GPU runs threw spray 0.28 m up from
  2.0-2.35 m/s impacts, over 50 mm walls, and the water ran out over the slab to the
  domain walls. Splash height grows with the square of the impact speed, so all
  three drops now land at about 1.2 m/s (the later two are thrown upwards so they come
  down late and slowly); that is what lets the walls, the domain height and the leaf
  capacity stay small. The wall on the camera side is rendered as clear glass. A
  stray droplet can still reach a wall, which costs a speck; the build fails if more
  than 0.1% of the particles get within 10 voxels of a wall.
- The admission planner books capacity before the first step, and two things it
  books are easy to blow up (a GPU run found both):
  - An analytic source is booked as a cube of its longest side. A 0.44 m pool was a
    0.44 m cube: 46 M particles and 11,000 leaves, refused, and then booked at the
    whole cube. So the pool is a grid of ten small box sources (`pool_tiles`).
  - Every collider books the leaves of its bounding box, so each wall is its own
    mesh and the slab only covers the trough, not the whole floor.
- The solver pads every marked leaf by a halo that grows with how far the liquid
  can travel in one frame (`ceil(substeps x CFL / 8)` leaves). At 1 mm with a
  12.5 ms frame the halo reached 3-4 leaves, so the thin walls' leaf count tripled
  and a hero run overran its capacity (51,347 leaves against 49,152). A 2 mm draft
  never showed it (its halo is 1-2). So the hero frame is 8.3 ms of liquid time
  (`TIME_SCALE` 0.20, slower slow-motion), the trough is smaller (300 x 160 mm inside, 180 mm
  walls), and the drops land at 1.2 m/s. The planner books about 16,800 leaves at a
  halo of 1 and 25,000 at 2, 34,000 at 3 and about 43,000 at 4 (the most the 32-substep
  limit allows); hero's leaf capacity is 52,000 (about 9 GiB of the 12.2 GiB
  estimate).
- Every drop is at least 32 voxels across at hero resolution. Surface Nets
  shrinks very small droplets more than marching cubes (about 2.5 voxels: 83%
  of the volume against 91%); fine spray will shrink a little, the drops will not.
- The pool is a thin film (8 mm) and Narrow Band is off. A source's startup demand
  is its whole volume, booked before any narrow-band culling, so a deep pool
  would be budgeted at tens of millions of particles whatever the band.
- The domain is a long box (600 x 260 x 300 cells), not a cube. The Surface Nets
  extractor works in a dense box, and the add-on's own estimate puts the whole
  hero scene at about 12.2 GiB, which is tight on a 16 GB card.
- The backend is chosen by `FLUIDSIM_SURFACE_BACKEND=nets` and is not saved in a
  `.blend` or a bake. A misspelt value silently falls back to marching cubes, so
  both scripts set it themselves and then read the backend the render surface
  actually uses (`solver.surface.render_extractor.backend`), failing the run if it
  is not `nets`. Each frame's backend is recorded in `sim_series.json` and the
  build writes it to `meta.json`.

## Run it (Windows PowerShell, from the repo root)

Use a private extensions folder, as AGENTS.md requires: `--factory-startup`
without one can delete the installed Warp wheels from your normal Blender
profile. Hero needs a 16 GB card; try `draft` first.

```powershell
$env:BLENDER_USER_RESOURCES = Join-Path $env:TEMP ("fluidsim-nets-" + [guid]::NewGuid())
$env:BLENDER_USER_EXTENSIONS = Join-Path $env:BLENDER_USER_RESOURCES 'extensions'
$env:NETS_SHOWCASE_PRESET = 'draft'    # or 'hero' (the default)
blender --background --factory-startup --python tools/demos/build_demo_scenes.py -- nets_showcase
blender --background --factory-startup --python tools/demos/render_nets_showcase.py -- sim
blender --background --factory-startup --python tools/demos/render_nets_showcase.py -- render --samples 128
```

`build` authors the scene, runs 24 frames as a check and saves
`artifacts/review_demos/nets_showcase.blend` plus `meta.json`. `sim` plays the
saved file through the real frame handler and saves every frame's mesh under
`artifacts/nets_showcase/meshes/`. `render` is a separate Blender run so the
simulation's VRAM is free for Cycles; it writes PNGs and
`artifacts/nets_showcase/nets_showcase.mp4`. Useful options: `--frames A B`,
`--still N`, `--size W H`, `--engine EEVEE`, `--resume` (keep PNGs already
rendered with the same settings and meshes; otherwise every frame is rendered
again). The extension wheels (Warp) must be present in the private extensions
folder, as for the production gate (`tools/run_production_gate.py` sets this up).

To open the scene by hand set `FLUIDSIM_SURFACE_BACKEND=nets` first. The scene
is far from real time.
