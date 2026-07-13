# FluidSim Warp — Technical Report

Native Blender 5.1 port of the synthesized `fluidSim` liquid solver
(`fluidSim.cl`, version **1.0.226**, **188 canonical kernels**) onto
**NVIDIA Warp / CUDA**. The solver is **GPU-resident** and **interactive**:
surface mesh, liquid particles, and whitewater are written into Blender
datablocks every frame for live viewport playback and rendering. There is no
required cache-bake step; optional npz caching and OBJ/PLY export are
supplemental tooling.

- Package: `fluidSim_warp_blender51` · `bl_info` version `(1, 0, 227)`
- Blender target: `5.1` · Category: `Physics`
- Backend: NVIDIA Warp / CUDA · `enable_backward=False` everywhere

---

## 1. Architecture Tree

```
fluidSim_warp_blender51/
├── __init__.py              bl_info + register()/unregister()
├── properties.py            PropertyGroups + attribute-viz item tuples
├── operators.py             31 fluidsim.* operators
├── panels.py                16 sidebar panels
├── preferences.py           add-on preferences
├── blender/                 bpy integration layer
│   ├── scene_bridge.py      Session, gather sources/colliders, frame handler
│   ├── viewport.py          output datablocks + display_color + overlay
│   ├── materials.py         liquid / particle / whitewater / attribute mats
│   ├── sources.py           tag_source / tag_sink / inject_object_particles
│   ├── colliders.py         tag_collider → wp.Mesh
│   └── render.py            render handler
├── sim/                     GPU core — NO bpy imports
│   ├── constants.py         TOOL_*, voxel bits, LEAF_DIM=8, phase slots
│   ├── enums.py             IntEnum mirrors of synthesized tables
│   ├── fields.py            SparseGrid / ScalarField / MacVelocityField
│   ├── particles.py         ParticleSystem / ParticleBins
│   ├── pressure.py          PressureSolver (Jacobi / PCG)
│   ├── viscosity.py         ViscositySolver
│   ├── surface.py           SDF build / smoothing / marching cubes
│   ├── whitewater.py        churn / emit / advect whitewater kernels
│   ├── sourcing.py          source shapes, height source, particle inject
│   ├── kernels_core.py      core kernels (registered)
│   ├── kernels_fluid.py     fluid kernels (registered)
│   ├── solver.py            FluidSolver + SolverSettings orchestration
│   ├── validation.py        15-test GPU validation suite
│   ├── registry.py          KERNEL_REGISTRY dispatch contract
│   └── custom.py            custom Warp snippet manager
├── diagnostics/             bpy-free
│   ├── profiler.py          fluidSim_profiler::1.0
│   ├── benchmark.py         deterministic benchmark + comparison
│   └── report.py            JSON report builder
└── io/                      supplemental
    ├── cache.py             optional npz cache
    └── export.py            OBJ surface + PLY points
```

**Boundary rule:** `sim/` and `diagnostics/` never import `bpy`. Only
`__init__`, `properties`, `operators`, `panels`, `preferences`, `blender/`,
and `io/cache.py` may. This keeps the core and validation suite runnable
headlessly with only `warp`.

---

## 2. Sparse Tiled Grid

`SparseGrid` (`sim/fields.py`) is a sparse tiled leaf-grid structure:

- **Leaves are fixed 8³ voxel tiles** (`LEAF_DIM=8`, `LEAF_VOLUME=512`).
- A `leaf_table` (dense over `leaves_dim`) maps a leaf coordinate to an
  allocated slot; `leaf_coords[slot]` is the inverse.
- Per-leaf voxel storage is flat: `flat = slot*512 + (lz*8+ly)*8+lx`, decoded
  by `voxel_coord_of_flat`.
- There is no parent/child hierarchy, adaptive level traversal, or runtime
  tree-refinement contract; sparsity comes from demand-allocated leaf slots.
- **Allocation is demand-driven.** Each frame `_rebuild_topology()`:
  1. copies the current `leaf_table` to `old_leaf_table`,
  2. `begin_topology()`, marks active voxels from particle positions, source
     AABBs, collider AABBs, height-source regions, and whitewater positions,
  3. `expand(topology_expand)` dilates the active set,
  4. `finalize_topology()` compacts leaves and rebuilds `leaf_table`.
- **Persistent field carry:** only `churn` survives a topology rebuild
  (`k_remap_leaf_field` remaps it via `old_leaf_table`).
- **Domain walls:** the grid pads `voxels_dim` to leaf multiples, so allocated
  voxels can exceed the domain box. For closed domains (`domain_sdf_walls`)
  `k_mark_domain_walls` writes the domain exterior into the collision SDF so
  fluid does not compress against the particle clamp.

Voxel-type bits (`constants.py`, `Bit_Core_VoxelType`): `COLLISION`, `SINK`,
`FLUID`, `SOURCE_REGION_TRACKING`, axis flags, `SDF`, `GRADIENT`, `ACTIVE`,
etc. `k_source_resolve_type` finalizes the type mask each frame.

---

## 3. FLIP Pipeline (`FluidSolver.step_frame`)

Per display frame, substeps are chosen by CFL
(`dt_cfl = cfl * voxel_size / (vmax + g*dt)`), clamped to
`[min_substeps, max_substeps]`. Each substep runs:

| # | Phase (`compute_*` slot) | Work |
|---|---|---|
| 0 | `FIRST_STEP` custom hook | snippet run |
| — | misc (15) | rebuild topology, clear grid fields |
| — | collision (8) | mesh→collision SDF (`k_create_sdf_from_mesh_axis`) + collider velocity |
| — | indexing (1) | count particles per voxel, build bin starts, scatter-sort particles into bins |
| — | sourcing (0) | `k_source_shapes` (voxel type/velocity/temperature/viscosity), `k_source_height`, particle top-up emission `k_source_shapes_particles`, then re-index |
| — | p2g (2) | `k_particle_set_grid_data` (MAC velocity + weights), `k_normalize_mac`, `k_particle_to_grid_active` (fluid mask); optional viscosity attribute splat |
| — | save velocities | copy MAC field to `vel_saved_{u,v,w}` for FLIP delta |
| — | `BEFORE_FORCES` custom hook | snippet run |
| — | forces (3) | `k_forces_gravity`, optional `k_forces_speed_limit` |
| — | viscosity (4) | explicit+implicit viscosity (Jacobi/PCG), constant or per-particle |
| — | collision (8) | `k_sdf_to_axis_collision` + `k_expand_shift_axis_velocity_in_collider` |
| — | `BEFORE_PROJECT…` custom hook | snippet run |
| — | pressure (5) | build RHS + preconditioner, Jacobi or PCG, optional threshold, project |
| — | sanitize MAC | `k_sanitize_mac_velocity` (kill inf/nan, clamp to CFL bound) |
| — | g2p (6) | `k_staggered_field_to_particle` (PIC/FLIP blend via `flip_ratio`) + `k_advect_particles` (Euler/midpoint/modified-midpoint/RK3) with SDF collision response |
| — | sanitize particles | `k_sanitize_particles` (kill non-finite, clamp speed) |
| — | lifecycle (14) | age, lifespan expiry, sink-region kill, `compact()` |
| — | relax (10) | optional `k_particles_relax` (local-voxel or neighbor-voxels) using the bin structure |
| — | surface (11) | build particle SDF + smoothing filter + (extraction deferred to output) |
| — | whitewater (12) | churn, emit, advect, grid density |
| — | `FINAL_STEP` custom hook | snippet run |

After substeps: `wp.synchronize_device`, `frame += 1`, `FrameStats` populated
from host-read counts; profiler timers collected.

---

## 4. Pressure & Viscosity Solvers

### Pressure (`sim/pressure.py`, `PressureSolver`)
- **RHS build** from MAC divergence, with collision velocities and an optional
  `count_stiffness` anti-clumping term (rest count = `particles_per_voxel`).
- **Preconditioner** built from the voxel-type mask.
- **Jacobi** (`solve_jacobi`) or **PCG** (`solve_pcg`, tolerance + iteration
  cap).
- Optional **relative pressure threshold** (`apply_threshold`) zeros tiny
  pressures.
- **Project** subtracts the pressure gradient from the MAC field and applies
  collision velocities. `last_iterations` / `last_residual` recorded for stats.

### Viscosity (`sim/viscosity.py`, `ViscositySolver`)
- Explicit smoothing (`smooth`) or implicit PCG (`solve`, `use_pcg=True`).
- Viscosity field is either a global constant (`viscosity_constant`) or splatted
  from the per-particle viscosity attribute (`viscosity_from_particles`,
  averaged via `k_particle_to_grid_collect`/`k_particle_to_grid_average`).
- Implicit viscosity **damps, never amplifies** (validated by test 11).

Both solvers operate only on fluid/solid voxels via the voxel-type mask.

---

## 5. Surface & Whitewater

### Surface (`sim/surface.py`)
- `k_surface_sdf_reset` seeds a far-field SDF; `k_build_surface_sdf` splats
  each particle as a signed-distance contribution (radius =
  `0.5*voxel_size*particle_radius_scale`, splat radius 2 voxels).
- **Realtime surface stack**: after particle SDF build, surfacing applies the
  existing kernels in fixed order: dilate, erode, then smooth. Dilate and erode
  each expose independent iteration/strength controls. Smooth exposes Box or
  Mean Curvature Flow with independent iteration/strength controls. Defaults
  preserve the prior behavior: no dilate, no erode, one Box smoothing pass at
  the legacy strength.
- `SurfaceExtractor.extract` runs **marching cubes** on the SDF and returns
  `(verts Nx3 float32, tris Mx3 int32)` consumed by the viewport bridge.

### Whitewater (`sim/whitewater.py`, `WhitewaterSystem`)
- **Churn** (`k_compute_churn`): accumulates velocity magnitude * divergence in
  fluid voxels, decaying each step (`churn_decay`/`churn_gain`). Churn is the
  only grid field carried across topology rebuilds.
- **Emission** (`k_emit_whitewater`): spawns foam/spray/bubble particles where
  churn exceeds `ww_churn_threshold`, near the surface band, and above
  `ww_min_speed`; assigns life from `[ww_life_min, ww_life_max]`.
- **Advection** (`k_advect_whitewater`): advects through the grid velocity
  field with gravity, surface/airband classification, and differential drag
  (`ww_bubble_drag`, `ww_spray_drag`).
- `k_whitewater_set_grid_data` splats whitewater into `ww_density` for viz.

---

## 6. Kernel Registry & Dispatch Contract

`sim/registry.py` implements the `kernel_lookup` dispatch contract:

- `KERNEL_REGISTRY: dict[name, RegistryEntry]` maps every canonical name to
  `(fn, strategy, notes)`. Strategy is `flat` / `tiled` / `host_orchestrated`.
- `@register("fluidSim...")` / `register_fn(fn, "fluidSim...")` register a
  Warp kernel under one or more names.
- **Alias derivation:** any `fluidSim*` registration auto-creates the
  `paradigm*` alias (`fluidSimAdvectParticles` → `paradigmAdvectParticles`).
  `lookup()` resolves canonical names and aliases.
- **Size-suffixed variants** (OpenCL workgroup knobs like `Jacobi8/16/32`,
  `ReduceSum128..1024`) collapse to one parameterized Warp kernel registered
  under all aliases.
- Kernel identity registration coverage is asserted at **188/188** canonical names by
  `tests/test_kernel_coverage.py` against `fluidSim_synth.kernel_manifest`.

---

## 7. Custom Kernel Contract (`sim/custom.py`)

Port of the custom-OpenCL-kernel feature to controlled Warp extension points:

- **Five order slots** (`CustomKernelOrder`): `FIRST_STEP`, `BEFORE_SOURCING`,
  `BEFORE_FORCES`, `BEFORE_PROJECT_NON_DIVERGENT_AND_ADVECTION`, `FINAL_STEP`.
  All snippets in a slot run in declaration order (the original tool stopped
  after the first — this port executes all of them).
- **Run-over** (`CustomKernelRunOver`): `VOXEL_TILE`, `VOXEL_LINEAR` (both
  flat-launch over `voxel_capacity`, exposing leaf-local + global + world
  indices; collision voxels skipped), or `PARTICLES` (over particle capacity;
  dead particles skipped).
- **Arguments** (`CustomKernelArgumentMode`): `FIELD` binds a named solver
  field from `FIELD_ACCESSORS` (velocity components, weights, temperature,
  viscosity, density, pressure, divergence, surface/collision SDF, churn,
  voxel_type, voxel_counts, and particle arrays) as a typed `wp.array`;
  `CONSTANT` binds a float/int scalar.
- **Compilation:** the snippet body is embedded into a generated Warp kernel
  template and written to a **content-hashed temp file**
  (`<tmp>/fluidsim_warp_snippets/<name>_<sha1>.py`), then imported — Warp
  re-reads kernel source from disk for codegen, so in-memory `exec` is not an
  option. Compiled modules are cached by hash across sessions.
- A **time context** (`frame`, `time`, `real_time_step`, `solver_time_step`,
  `time_scale`, `substeps`) is always in scope. Compile errors are surfaced to
  each snippet's `last_error` UI field.

---

## 8. Blender Integration

- **`blender/scene_bridge.py`** owns the `Session` (`solver`, `profiler`,
  `frame_stats`, `validation_results`, `benchmark_result`, `error`). It
  translates tagged Blender objects into `SourceShapeDesc` / `ColliderDesc`
  (mesh → `wp.Mesh` via evaluated depsgraph, with linear/angular velocity from
  animation), calls `sync_scene_to_solver`, and registers the
  `frame_change_post` handler that steps the solver when `fs.live` is on.
- **`blender/viewport.py` (`update_outputs`)** writes solver output into real
  datablocks each frame: `FluidSim Surface` (mesh via fast bulk
  `foreach_set`), `FluidSim Particles` and `FluidSim Whitewater` (PointCloud
  objects with `position`/`radius`/`display_color` attributes, falling back to
  a vertex cloud if `bpy.data.pointclouds` is absent). The `_draw_overlay`
  handler renders the live status + per-phase timers via `blf`. Output object
  colors and material diffuse colors are also written for Solid viewport
  legibility without changing the user's global viewport shading mode.
- **`blender/scene_bridge.py` (`request_scene_update`)** is the property-update
  bridge for realtime UI edits. Surface radius/dilate/erode/smooth changes
  push into the active solver, rebuild the current-frame surface SDF, invalidate
  the cached mesh, and refresh output datablocks without advancing playback.
  Inactive output/cache objects receive visibility/material refreshes only;
  geometry recomputation still requires an active solver frame or cached
  particles.
- **`blender/materials.py`** provides default liquid / particle / whitewater
  Principled BSDF materials, plus **attribute materials** that add a
  `ShaderNodeAttribute` (`attribute_name = "display_color"`) linked into the
  BSDF `Base Color`, so attribute coloring is visible in viewport and render.
- **`properties.py`** defines the scene + object PropertyGroups. Attribute-viz
  item tuples and the `liquid_view_mode` update callback live here.

---

## 9. Diagnostics Contracts

### Profiler (`diagnostics/profiler.py`)
Implements `fluidSim_profiler::1.0`:
- `TIMER_TRACK_ORDER` mirrors the synthesized contract: `dcc*`, `input`,
  `solver`, `script`, `compute`, `blit`, `compute_0..15`, `blit_0..15`,
  `eventWait_0..15`, queue tracks, `debug_0..15`, `total`.
- `PHASE_SLOTS` maps solver phase names to `compute_*` slots (sourcing→0,
  pressure→5, whitewater→12, …).
- `track(phase)` brackets a region; on CUDA it calls `wp.synchronize_device`
  for accurate timings (`sync=False` for approximate).
- `compute_profile()` emits device metadata (`PROFILE_COMPUTE_FIELDS`).
- `summary(last_n)` averages timers for overlay/UI display.

### Benchmark (`diagnostics/benchmark.py`)
- Deterministic dam-break scenario (`default_benchmark_settings/sources`):
  domain (-1,-1,0)→(1,1,2), voxel 0.05, 1.5 M particles, emits for 6 of 24
  frames. Aggregates into `BENCHMARK_FIELDS`.
- `compare_benchmark`: **exact** match on `activeVoxels{Average,Min,Max}`
  (determinism); **25 % relative tolerance** on `timeStepAverage` /
  `voxelsPerSecondAverage` (performance). Result is `pass` / `fail` /
  `performance_regression`.

### Report (`diagnostics/report.py`)
Aggregates profiler + frame stats + benchmark + validation into one JSON
document with tool/backend/host/warp version metadata.

---

## 10. GPU-Only Policy

All simulation state is `wp.array(device="cuda:0")`. Host code sequences kernel
launches and reads back only small statistics (counts, max-speed reduction,
per-frame stats). There is **no CPU or OpenCL fallback**; `ComputeAPI` /
`DeviceType` enums are retained for fidelity only. Every Warp module sets
`wp.set_module_options({"enable_backward": False})` — autodiff is unused and
disabling it shrinks codegen.

---

## 11. Validation Methodology

`sim/validation.py` defines **15 tests**, each decorated with
`@_test(name)` and registered in `_TESTS`. `run_all(device, progress)`
executes them on the real CUDA device; each returns
`(passed, detail)` wrapped in a `ValidationResult`. Tests cover: sparse tiled
leaf-grid coordinate/index roundtrip; leaf addressing; 16/32-bit precision parity; P2G
count/average/active masks; G2P sampling; source shape coverage + attribute
transfer; collision/sink voxel bits; divergence reduction under projection;
Jacobi + PCG residual convergence; surface SDF build + box-filter parity;
viscosity stability (damps, never amplifies); whitewater churn→emit→advect
behaviour; and benchmark/profile schema compatibility. Failures are reported,
never crash the suite.

---

## 12. Verification Results

| Check | Result |
|---|---|
| Kernel identity registration coverage | **188 / 188** canonical names registered (`KERNEL-COVERAGE-PASS`) |
| GPU validation | **15 / 15** passing (`VALIDATION-PASS`) |
| Pressure projection regressions | **5 / 5** passing (`pytest tests/test_pressure_projection_regressions.py`) |
| Blender background acceptance | **34 / 34** checks (`BACKGROUND-ACCEPTANCE-PASS`) |
| Attribute visualization acceptance | **pass** — surface + particle + whitewater `display_color` + attribute materials (`ATTRIBUTE-VIZ-ACCEPTANCE-PASS`) |
| Demo `dam_break` | **9 / 9** (mass, containment, front speed ≈ 0.94·√(gH), Martin–Moses front, column collapse, whitewater bore) |
| Demo `crown_splash` | **8 / 8** (mass, crown formation, √t radius corr 0.971, Worthington jet, azimuthal symmetry CV 0.007) |

---

## 13. Performance Notes

- The sparse grid only allocates leaves where fluid/sources/colliders exist;
  `Leaf Capacity` bounds allocation; `Topology Expand` trades memory for
  fewer per-frame topology rebuilds under fast-moving fluid.
- **Pressure** (`compute_5`) and **Surface** (`compute_11`) dominate at high
  resolution. Surface filter iterations and splat radius control surface cost.
- Profiling with **Accurate GPU Timing** syncs each phase (precise but slower);
  disable it for lower-overhead approximate timings.
- The benchmark's exact-match on active voxels makes it a deterministic
  regression detector: a code change that alters the particle/grid pipeline
  will fail the exact fields even if it stays fast.
- Surface attribute coloring performs a host-side nearest-neighbour sample from
  particles to mesh vertices; sample counts are capped for interactive use.

## Decoupled render surfacing and ST-FLIP beta

Version 1.0.235 adds a separate sparse render-surface grid with multiplier or
custom voxel sizing, frame/manual rebuild cadence, filtering, bounded dense
extraction, and shared viewport/cache/export mesh ownership. The simulation
grid remains authoritative for physics and optional low-resolution surface
fields.

The beta ST-FLIP lane adds a device-resident particle time residual,
spatiotemporal P2G deposition, phase-field activation, optional
variable-coefficient pressure projection, local particle CFL substeps, and
temporary render-surface re-synchronization. Default FLIP remains unchanged
and retains its recorded-launch/CUDA-graph PCG path.
