# FluidSim Warp — Technical Report

Port of the synthesized `fluidSim` OpenCL liquid solver (v1.0.226,
`liquid.cl` = `fluidSim.cl`, 20,640 lines, 188 kernels) to NVIDIA Warp/CUDA
with a native Blender 5.1 add-on. The synthesized reference package
(`fluidSim_synth`: kernel manifest, enums, dispatch/custom-kernel contracts,
profiler contract, UI contract, validation plan) is the fidelity source; no
original OpenCL or Houdini code is included or referenced at runtime.

## Architecture

```
fluidSim_warp_blender51/
├── sim/            GPU solver core (bpy-free, importable headless)
│   ├── fields.py       sparse tiled grid (8³ leaves), MAC velocity fields
│   ├── particles.py    GPU particle system, stream compaction
│   ├── kernels_core.py core buffer/field ops, SDF, transfer (80 core kernels)
│   ├── kernels_fluid.py FLIP pipeline kernels (P2G/G2P, advection, forces,
│   │                    indexing/sorting, lifecycle, relaxation)
│   ├── pressure.py     divergence, Jacobi + PCG, leaf-coarse guess, project
│   ├── viscosity.py    explicit smoothing + implicit Jacobi/PCG solve
│   ├── surface.py      surface SDF build/filters, marching-cubes extractor
│   ├── whitewater.py   churn, emission, foam/spray/bubble advection
│   ├── sourcing.py     shape/height/particle/volume sourcing (24 kernels)
│   ├── custom.py       user Warp-snippet compiler (contract-conformant)
│   ├── solver.py       FluidSolver orchestration, GPU-resident state
│   ├── registry.py     kernel identity registry (188 canonical + aliases)
│   └── validation.py   13-test GPU validation suite
├── blender/        scene bridge, sources/colliders, viewport, materials, render
├── diagnostics/    profiler, benchmark, report (contract-conformant)
├── io/             optional npz cache + OBJ/PLY export
└── properties/operators/panels/preferences  Blender UI layer
```

### Sparse grid

The OpenCL octree field is realized as a sparse tiled grid: dense leaf
table over the domain (leaves of 8³ voxels), with a compact array of
allocated leaves. Voxel addressing matches the source (leaf-linear layout,
x-fastest within leaves; `voxel_flat_index` / `voxel_coord_of_flat`
roundtrip is validation test 1). Topology rebuilds per substep from
particle positions, source/collider AABBs and height-source regions, with
27-neighborhood expansion; persistent fields (churn) are remapped across
rebuilds via the previous leaf table.

### FLIP pipeline (per substep)

1. custom hook FIRST_STEP → topology rebuild → field clear
2. mesh colliders → SDF (Warp BVH mesh queries) + rigid collider velocities
3. custom hook BEFORE_SOURCING → particle indexing → sourcing (shapes,
   height field, particle injection) → re-index
4. P2G (staggered trilinear scatter + normalize) → active-voxel marking
5. save velocities (FLIP reference) → custom hook BEFORE_FORCES → gravity,
   speed limit → viscosity (explicit or implicit)
6. collision face classification (per-axis collision bits) + collider
   velocity override/extrapolation
7. custom hook BEFORE_PROJECT → pressure projection → velocity update
8. G2P (PIC/FLIP blend) → particle advection (Euler/midpoint/modified
   midpoint/RK3, SDF collision response with restitution/friction)
9. lifecycle (age, lifespan, sinks, compaction) → position relaxation
10. surface SDF build + filtering → churn → whitewater emit/advect
11. custom hook FINAL_STEP

Adaptive substepping uses a GPU max-velocity reduction and CFL condition
clamped to configured min/max substeps.

### Pressure projection

Solves `A q = −∇·u h²` on fluid voxels with unallocated/collision voxels
treated as solid walls and solid faces constrained to collider velocities.
Available solvers per the source: Jacobi (fixed iterations) and
preconditioned conjugate gradient (inverse-diagonal preconditioner, device-
side α/β updates, periodic residual readback). The source's electrostatic
leaf-aggregate solve is ported as the multigrid coarse initial guess
(per-leaf aggregate Jacobi → sampled up as PCG x₀). Pressure thresholding
and scaling match the source contracts. Divergence reduction and residual
convergence are validation tests 8–9.

### Viscosity

Explicit path (Jacobi): stable 6-neighbor blending with per-voxel viscosity.
Implicit path (PCG): backward-Euler `(I + c(D−adj))x = u` per MAC component
with variational weighting; unconditionally stable (validation test 11).
Viscosity is constant or splatted from the per-particle attribute.

### Whitewater

Churn accumulates curl + local velocity variance with exponential decay
(`fluidSimComputeChurn`); emission is probabilistic in churn-active surface
voxels with atomic slot allocation; advection classifies foam (surface
band, field-following), spray (above surface, ballistic w/ drag) and
bubbles (below surface, buoyant drag toward liquid velocity), with SDF
collision pushout and lifespan/domain kill (validation test 12).

### Kernel identity

All 188 canonical kernel names from the manifest are registered in
`sim/registry.py` and resolvable via `lookup()` (including `paradigm*` →
`fluidSim*` source aliases); `tests/test_kernel_coverage.py` enforces
coverage against `fluidSim_synth.kernel_manifest`. Several OpenCL
size-specialized variants (e.g. `...Jacobi{2,4,8,16,32}_{original,
optimizedUnroll}`, reduce widths 128–1024) map to single Warp kernels, as
permitted by the dispatch contract; NanoVDB/OpenVDB sourcing kernels map to
`wp.Volume` sampling.

### Custom kernels

Per the custom-kernel contract: snippets are Warp kernel bodies with
declared field/constant arguments, an execution-order slot and a run-over
domain (voxel tile/linear/particles). The generated module exposes the
contract's index/world/time context names. Warp 1.12 requires kernels in
real source files, so generated modules are content-hash-cached under the
system temp directory and imported via `importlib`. Multiple snippets per
order slot run in declaration order (resolving the source audit item where
only the first snippet executed).

### Audit items resolved

- Staggered magnitude Y/Z read suspect → symmetric axis-pair averaging in
  `coreStaggeredVectorFieldToMagnitude`.
- Custom snippet loop early-return → all enabled snippets execute.
- Houdini UI → native Blender panels/operators/properties per UI contract.

## Blender integration

- `Scene.fluidsim` / `Object.fluidsim` property groups cover every solver
  knob; objects are tagged as sources/colliders/sinks/forces.
- A `frame_change_post` handler drives interactive playback: play = step,
  rewind-to-start = reset. No cache required; optional npz cache and
  OBJ/PLY export are supplemental.
- Outputs are real Blender data: surface mesh (marching cubes, smooth
  shaded, materialized), particle and whitewater point clouds (Blender 5.1
  `PointCloud` datablocks), all render-visible in EEVEE/Cycles.
- Colliders are evaluated-depsgraph triangulated meshes converted to Warp
  BVH meshes each sync, with frame-differenced rigid velocities.
- The viewport overlay (blf draw handler) shows status, counts, and
  per-phase GPU timers.

## GPU-only policy

All simulation state lives in `wp.array` GPU buffers for the session
lifetime. CPU work is limited to: Blender data sync (mesh upload, output
mesh/point-cloud download), orchestration scalars (substep count, PCG
residual checks every N iterations, leaf/particle counts), UI, profiling,
validation and I/O — per the dispatch contract's REQUIRED_GPU_POLICY.

## Diagnostics contracts

- Profiler `fluidSim_profiler::1.0`: all PROFILE_COMPUTE_FIELDS (device,
  API, memory tallies) and PROFILE_FRAME_FIELDS (step time, voxel/particle
  counts and rates, allocation ratios, timers) with the contract's timer
  track order; solver phases map to `compute_0..15` slots.
- Benchmark: standard dam-break scenario; emits all BENCHMARK_FIELDS;
  comparison is exact on active-voxel statistics and tolerance-based on
  `timeStepAverage`/`voxelsPerSecondAverage`.
- Reports serialize to JSON for offline analysis.

## Verification (RTX 4080 SUPER, CUDA 12.9, Warp 1.12, Blender 5.1.0)

- `tests/test_kernel_coverage.py` — 188/188 manifest kernels registered.
- `tests/test_validation.py` — 13/13 GPU validation tests pass
  (headless Python + inside Blender).
- `tests/smoke_solver.py` — 12-frame dam-break: 105K particles, 44K active
  voxels, PCG residuals 1e-6..1e-9, ~180 ms/frame at 0.05 voxel size.
- `tests/blender_background_acceptance.py` — 34/34 checks in
  `blender --background`: operators, live handler stepping, surface mesh +
  materials, point clouds, GPU snippet execution, profiler, report, export.
- `tests/blender_gui_acceptance.py` — 24/24 checks in the real GUI:
  timeline playback stepped the solver 17 frames in 3.1 s at ~384K
  particles, rewind-reset, panel registration and redraw, property edits,
  viewport screenshot, EEVEE render with visible content, benchmark
  operator.

## Performance notes

Representative steady-state frame at voxel 0.08, ~190K particles, ~138K
active voxels: ~40–75 ms/frame (13–25 fps interactive playback) including
mesh/point-cloud output sync — orders of magnitude faster than Blender's
CPU FLIP workflows at comparable resolution, and fully interactive rather
than bake-first. Kernel compilation is cached by Warp after first run.
