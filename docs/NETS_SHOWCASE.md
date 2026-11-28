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
| Cells along the longest axis | 280 | **560** (2.0x) | 280 |
| Cells in the whole domain | 2.2 M | **76 M** (34x) | 9.5 M |
| Particles at start | ~864 k | **~6.8 M** (7.9x), Narrow Band off | ~0.85 M |

The shipped demos are coarser still (dam_break 187 cells, crown_splash 80).
"Twice the fidelity" is read as twice the cells along the longest axis, with
the domain cell count and particle count higher multiples. The scenes are
different sizes (Upres Cove is a 5.6 m cove, this is a 0.56 m trough), so the
voxel size in millimetres is the plainest comparison. The particle count is a
estimate until the build reports the real one (in `meta.json`).
The numbers live in `tools/demos/nets_showcase.py` and are checked by
`tests/demos/test_nets_showcase.py`.

## Layout choices that come from the Surface Nets review

- No liquid touches a domain wall. Surface Nets stops half a voxel short of its
  extraction box where liquid meets a domain wall, which would show as a gap. The
  pool sits on a collider slab 30 mm above the domain floor and the trough wall
  keeps it at least 46 mm from the sides. The build fails if any particle gets within
  10 voxels of a domain wall.
- Every drop is at least 32 voxels across at hero resolution. Surface Nets
  shrinks very small droplets more than marching cubes (about 2.5 voxels: 83%
  of the volume against 91%); fine spray will shrink a little, the drops will not.
- The pool is a thin film (8 mm) and Narrow Band is off. A source's startup demand
  is its whole volume, booked before any narrow-band culling, so a deep pool
  would be budgeted at tens of millions of particles whatever the band.
- The domain is a long box (560 x 340 x 400 cells), not a cube. The Surface Nets
  extractor works in a dense box, and the add-on's own estimate puts the whole
  hero scene at about 10 GiB; a cube would not fit a 16 GB card.
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
