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
3. **Surface Waves** (see below). Wave detail finer than the surface voxels on
   the coarse surface itself, with no second simulation (Mercier et al. 2015).
   Owner decision, 24 November 2026: after guided upres came out slower than
   full fine and broke into specks on Upres Cove, upres moves to this
   surface-only method.

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

- fills the coarse shell with fine particles until the fine solve holds
  liquid, then only where the coarse liquid advances (including liquid the
  previous coarse frame did not cover at all), at the coarse velocity (`seed`).

Before use, each coarse level set is rebuilt as a true inside distance
(`GuideFrame.reconstruct_depth`), because a particle-splat SDF is never
deeper than a particle radius (a collider next to the liquid is a wall, not
air). The fine solve only carries particles in the
shell, but liquid thinner than the shell (a sheet spreading over a beach) is
all shell, so both its particle and leaf budgets follow the volume
(`factor**3`). A surface-area budget (`factor**2`) ran out on Upres Cove and
left holes in the fine surface. When the fine solve still has to clip its
particle seeding at the budget, it prints a warning once and `GuidedUpres.particle_budget_hit` is set
(the compare tool reports it as `fine_out_of_particles`). Guided upres only
saves work where the liquid is much deeper than the shell; on thin water it
costs about as much as a full fine solve. Dynamic (two-way) colliders are refused for now; other
colliders are copied, so the fine solve never writes into the coarse ones, and
fine solid and sink cells stay air under the guide. The fine solve gets every
coarse source except liquid emission (sinks, forces, collision and activation
regions) and the coarse proxy force fields; the guide seeds its liquid, never
into fine solid or sink cells. One-shot sources the coarse solve already
fired when the upres starts stay fired in the fine solve. With adaptive
substeps the fine schedule counts the fastest guide velocity, so an empty fine
solve still takes enough substeps for fast coarse liquid. Moving colliders get
a private fine mesh. The
fine solve exports no per-particle attributes, anything narrow-band reseeding
cannot carry (species, particle-derived viscosity, scalar field sources) is refused,
and factors run from 2 to 8.
Only the shell gets a fine pressure solve. Fine liquid deeper than
`core_shells` shells (default 2) below the coarse surface is a solid that
moves with the coarse velocity for the pressure solve only, which is the
boundary condition Nielsen and Bridson use. Without it the fine pressure solve
spans the whole liquid volume and costs as much as a full fine sim. The core
leaves stay allocated so the fine level set and mesh stay whole; only the
pressure unknowns shrink. The guide supplies velocity at the staggered core
faces. These faces stay valid for level-set advection after projection, and
real colliders keep their own boundary velocities. `core_shells=0` solves
the whole depth.
`GuidedUpres(coarse, factor=2, shell_voxels=4)` builds the fine solver from
the coarse settings. Not wired to Blender yet.

```
python -m pytest -q tests/solver/test_guided_upres_kernels.py
python tools/profiling/compare_guided_upres.py --device cuda:0 --voxel-size 0.06 --factor 2 --shell 4 --frames 20 40 --full-fine --phases --obj-dir artifacts/guided_upres --json-out artifacts/guided_upres/report.json
```

The second command steps the coarse dam break, the guided fine solve and
(with `--full-fine`) an unguided full solve at the fine size, and prints
per-frame ms, particle counts and triangle counts, plus OBJ meshes.

### Review scene: `upres_cove`

`upres_cove` is the dam_break cove (same reservoir and colliders) at 2 cm
voxels, compared with the shipped demo's 3 cm. The coarse sim emits about
864,000 particles and the guided fine solve runs at 1 cm. It is opt-in
and stays out of the Demo Gallery and the demo gate, like `tsunami_parthenon`.
Its numbers live in `tools/demos/upres_cove.py`.

From PowerShell (the private extension folder keeps `--factory-startup` from
deleting your installed extension wheels, Warp included; see AGENTS.md):

```
$env:BLENDER_USER_RESOURCES = Join-Path $env:TEMP ("fluidsim-upres-" + [guid]::NewGuid())
$env:BLENDER_USER_EXTENSIONS = Join-Path $env:BLENDER_USER_RESOURCES 'extensions'
blender --background --factory-startup --python-use-system-env --python tools/profiling/compare_guided_upres.py -- --demo upres_cove --full-fine --phases --json-out artifacts/guided_upres/upres_cove.json
```

This needs no built `.blend`: until `build_demo_scenes.py -- upres_cove`
has saved `artifacts/review_demos/upres_cove.blend`, the scene is authored on
the fly. The build saves there, not in `demos/`, because the demo gate
(`check_demo_blends.py`) fails on any `demos/*.blend` outside its inventory. The demo's own add-on solver is the coarse sim. The guided fine solve
follows it, then the full fine sim runs from the same inputs in a second
pass, so the two fine solves never share VRAM. At the preset resolutions,
both grids have room for every leaf in the domain, including the collider
bounds and the spreading liquid. The report's
`summary.speedup_vs_full_fine` is full fine ms over coarse plus guided fine ms,
after one warm-up frame; above
1 the upres is faster. `artifacts/guided_upres/upres_cove_review.blend` holds
the coarse, guided and full fine surfaces of frames 24, 48 and 72, one
collection each, every mesh visible only on its own frame. `--voxel-size`,
`--factor`, `--core-shells` and `--frames` override the scene's defaults.
Demo mode requires static simulation inputs. The unguided `--full-fine`
reference keeps the authored narrow-band setting and requires a scene without
automatic particle-object seeds.

## Surface Waves

`sim/surface_waves.py`, after Mercier, Beauchemin, Thuerey, Kim and
Nowrouzezahrai, "Surface Turbulence for Particle-Based Liquid Simulations"
(SIGGRAPH Asia 2015), in the line of Kim, Thuerey and Tessendorf, "Closest
Point Turbulence for Liquid Surfaces" (TOG 2013). It works on the extracted
render mesh, so it runs with every surfacing backend and any scene, and it
never changes the simulation. Each frame:

1. Wave points (spacing **Wave Detail** times the surface voxel) move with
   the coarse velocity at the surface and are projected back onto the new
   mesh. Points that lose the surface die.
2. Points closer than half the spacing to an older point die; triangles left
   sparse get new points, thinned against each other the same way.
3. Waves start where the coarse surface bends more sharply than its grid
   resolves (mean curvature above one over two surface voxels), with a phase
   from smooth noise of each point's birth position, so neighbours oscillate
   together and form wave fronts.
4. A damped wave equation (capillary-gravity speed of the finest wave, four
   spacings long; tangent-plane point Laplacian; explicit substeps under a 0.5
   CFL) carries them across the surface. Heights are capped at half a surface
   voxel.
5. The render mesh is displaced along its vertex normals by the wave height
   (same vertices and triangles).

Cost follows surface area: one neighbour search per point per substep. The
wave points are capped at 4 million, and `memory_safety` counts that budget
when the option is on. Settings: **Surface Waves**, **Wave Detail**,
**Wave Strength**, **Wave Damping** (Surface Filters panel, under Sharp
Surfacing). Waves step once per solver frame in `SurfacePipeline.after_frame`.

Review it on Upres Cove (same PowerShell setup as below):

```
blender --background --factory-startup --python-use-system-env --python tools/profiling/compare_guided_upres.py -- --demo upres_cove --waves --no-guided --full-fine --json-out artifacts/guided_upres/upres_cove_waves.json
```

The review file then holds Coarse, Surface waves and Full fine collections,
Surface waves shown first. `summary.waves_speedup_vs_full_fine` is full fine
ms over coarse plus waves ms (waves ms include extracting the coarse mesh and
sampling its velocity).

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
