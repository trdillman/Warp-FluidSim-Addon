# FluidSim Warp — Technical Report

FluidSim Warp 1.0.248 is a Blender 5.1+ extension around the existing sparse
8³-leaf NVIDIA Warp PIC/FLIP solver. Blender 5.2.0 LTS/Linux x64 remains the
certified release target. Live Preview remains interactive;
Production Bake adds a separate, manifest-authoritative cache-v2 transaction and
checkpoint layer without rewriting the solver into a different architecture.

- Extension ID: `fluidsim_warp` · manifest/runtime version `1.0.248`
- Validated host: Blender `5.2.0 LTS` on Linux x64
- Integration compatibility: Blender `5.1.0` on Windows x64 with Warp `1.12.1`
- Runtime dependency: external NVIDIA Warp `1.16.0+cu12`; never bundled
- Solver backend policy: NVIDIA CUDA only; CPU is used only for surrounding
  package/runtime smoke evidence
- Persistent cache policy: v2 writes/resume; v1 playback compatibility

---

## 1. Architecture Tree

```text
fluidsim_warp/
├── version.py                    canonical release identity
├── properties.py / operators.py / panels.py
├── runtime/warp_runtime.py       lazy external-Warp/CUDA preflight
├── presets/
│   ├── schema.py                 strict versioned allow-list schema
│   └── library.py                built-in/user discovery and atomic writes
├── resources/
│   ├── presets/*.json            six built-in product presets
│   ├── examples/*.blend          generated Phase 2 example scenes
│   └── geometry_nodes/*.blend    generated immutable render-node asset
├── blender/
│   ├── scene_bridge.py           Blender/session integration
│   ├── setup_wizard.py           selected-object setup transaction
│   ├── production_bake.py        bake state machine and frame transactions
│   ├── cache_manager.py          inspect/free/move/rename/orphan operations
│   ├── command_line.py           headless contract and JSON reporting
│   ├── geometry_nodes.py         revisioned asset installation/input sync
│   └── output_publication.py     typed attribute-aware publication
├── io/
│   ├── cache_schema.py           strict cache-v2 manifest schema
│   ├── checkpoint.py             strict pickle-free checkpoint I/O
│   └── cache.py                  cache-v1 reader plus v2 frame adapter
├── sim/
│   ├── attributes.py             authoritative output/cache schema
│   └── ...                       GPU solver core; no bpy imports
└── release/
    └── run_headless_bake.py      installed process-exit entry point
```

The solver remains authoritative for simulation semantics. `io/` owns durable
serialization, `blender/production_bake.py` owns transaction orchestration, and
`blender/setup_wizard.py` owns product onboarding. The package boundary keeps
`sim/` free of `bpy`; release-only build/smoke/example-generation tools are
excluded from the installed archive except the headless entry point.

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
| — | forces (3) | gravity, optional directional/radial acceleration, velocity diffusion, particle-attribute dissipation, collision-SDF attraction, then speed limit |
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

`SourcePipeline.apply(dt)` owns the sourcing phase's state, cached Warp
arrays, launch order, emission seed, and post-growth particle-count refresh.
`FluidSolver` retains its public source setters and injection entry point as
thin delegation to that Module.

After substeps: `wp.synchronize_device`, `frame += 1`, `FrameStats` populated
from host-read counts; profiler timers collected.

---

## 4. Pressure & Viscosity Solvers

### Pressure (`sim/pressure.py`, `PressureSolver`)
- **`project_frame` is the authoritative Interface** for RHS construction,
  Jacobi/PCG and space-time selection, convergence handling, thresholding,
  safe projection, and the status returned to `FluidSolver`.
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

### Phase-3 output attribute contract

`sim/attributes.py` is the sole inventory for public names, Blender data types,
`POINT` domains, component counts, cache fields, and checkpoint policy. Surface,
liquid-particle, and whitewater inventories are ordered and signature-bound.
The optional persistent liquid channels are `color` (`wp.vec4`), `source_id`
(`wp.int32`), and `uid` (`wp.int32`); they are allocated only when required by
an enabled surface or particle output. Fixed physical channels remain allocated
for solver use.

Emission kernels carry source color and deterministic source ID through analytic,
mesh, compiled-region, height-fill, and direct-injection paths. UID assignment
runs once after an admitted emission transaction and advances `next_particle_uid`
only by the committed count. Capacity growth and scan compaction copy every
enabled channel. Checkpoints serialize the optional inventory, each enabled
prefix, and the next UID; restore validates inventory equality and UID
monotonicity before accepting state.

`HostFrame` derives speed on the host and samples projected MAC-field curl with
a Warp kernel for particle vorticity. Surface vertices reuse one deterministic,
memory-bounded nearest-particle index map across all enabled attributes. Motion
output converts physical velocity to displacement per scene frame and is
published as Blender `velocity`; physical velocity remains
`fluidsim_velocity`.

Cache-v2 frame payloads include `attribute_schema_version`, an ordered JSON
inventory, and `surface_attribute_transfer_method`. Read and publication paths
fail closed on missing fields, wrong shape/type/domain, non-finite values,
misalignment, or invalid UID data. Frames without this metadata retain legacy
playback compatibility.

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

- **One lifecycle Module:** validation, conservative autofix, presets,
  shortcuts, persistence, compilation, runtime grouping, and launch all live
  in `sim/custom.py`; Blender callers use that Interface directly.
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
- **`blender/output_publication.py`** writes solver/cache facts into real
  datablocks using bulk `foreach_set`: `FluidSim Surface` as a mesh and
  `FluidSim Particles` / `FluidSim Whitewater` as padded PointCloud objects
  where available. It recreates conflicting managed attributes, validates
  Phase-3 values before mutation, preserves display-only `display_color`
  separately, and tracks live point count independently of padded capacity.
  `blender/viewport.py` remains the timeline/overlay adapter.
- **`blender/scene_bridge.py` (`request_scene_update`)** is the property-update
  bridge for realtime UI edits. Its `SceneUpdate` enum owns the supported
  actions and default status reasons, so property callbacks do not exchange
  raw dispatcher strings. Surface radius/dilate/erode/smooth changes
  push into the active solver, rebuild the current-frame surface SDF, invalidate
  the cached mesh, and refresh output datablocks without advancing playback.
  Inactive output/cache objects receive visibility/material refreshes only;
  geometry recomputation still requires an active solver frame or cached
  particles.
- **`blender/materials.py`** provides default production materials that read
  `fluidsim_color`; the liquid material also maps `fluidsim_viscosity` to
  roughness. Legacy debug materials read only `display_color`, so visualization
  never overwrites simulation color.
- **`blender/geometry_nodes.py`** appends three immutable revision-tagged groups
  from `resources/geometry_nodes/fluidsim_render_nodes.blend`, attaches one
  owned modifier per output, and synchronizes typed Blender 5.2 material/radius
  inputs. Point groups realize instances before material assignment so named
  point attributes reach F12 shaders. Explicit updates skip user-owned
  modifiers and never mutate user node groups.
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

`sim/validation.py` defines **16 tests**, each decorated with
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

## 12. Verification Results for 1.0.248

| Gate | Result in this environment |
|---|---|
| Pure-Python suite | **150 passed** |
| Python compilation | Passed |
| Source/release verification | Passed; generated render asset present and no bundled Blender/Warp payload |
| Supplied Blender | **5.2.0 LTS**; registration, attribute publication, cache round-trip, save/reopen, Geometry Nodes evaluation, typed modifier inputs, user-modifier protection, and Eevee F12 rendering passed |
| Supplied Warp | **1.16.0+cu12**, module `1.16.0`, loaded from the supplied external package; CPU kernels for source metadata, velocity preservation, optional compaction, checkpoint restore, and vorticity passed |
| Shader attribute survival | F12 point render contained independently detected red, green, and blue regions; surface render contained red/blue regions from `fluidsim_color` |
| Cache/checkpoint transactions | Phase-2 manifest/checkpoint guarantees remain covered; Phase-3 inventory, dtype/shape/finiteness/alignment, optional persistent channels, and UID validation added |
| Installed headless no-CUDA exit | Required process-level exit `12` remains the expected host result |
| NVIDIA CUDA solver/render gate | **Open** — no usable CUDA driver/device was exposed |

The CPU and Blender evidence proves surrounding data, cache, publication, asset,
and render integration. It does not certify CUDA solver numerics, CUDA resume
determinism, GPU memory/performance, or NVIDIA-specific render behavior.

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
- Enabled surface attributes reuse one bounded spatial nearest-particle map per
  frame. Very large particle pools use a stable sample stride to cap host work.
- Point render groups realize instances so arbitrary named attributes reach
  shaders. This intentionally trades render geometry/memory for correctness;
  display fraction, output toggles, and attribute inventory control the cost.
- Disabled optional color/source-ID/UID channels allocate no full-capacity pool.

## Decoupled render surfacing and ST-FLIP beta

Historical feature note: version 1.0.235 added a separate sparse render-surface grid with multiplier or
custom voxel sizing, frame/manual rebuild cadence, filtering, bounded dense
extraction, and shared viewport/cache/export mesh ownership. The simulation
grid remains authoritative for physics and optional low-resolution surface
fields.

The beta ST-FLIP lane adds a device-resident particle time residual,
spatiotemporal P2G deposition, phase-field activation, optional
variable-coefficient pressure projection, local particle CFL substeps, and
temporary render-surface re-synchronization. Default FLIP remains unchanged
and retains its recorded-launch/CUDA-graph PCG path.
