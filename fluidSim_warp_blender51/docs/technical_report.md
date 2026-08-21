# FluidSim Warp — Technical Report

FluidSim Warp 1.0.255 targets Blender 5.1.0 or newer around the existing
sparse 8³-leaf NVIDIA Warp PIC/FLIP solver. Live Preview remains interactive;
Production Bake adds a separate, manifest-authoritative cache-v2 transaction and
checkpoint layer without rewriting the solver into a different architecture.

- Extension ID: `fluidsim_warp` · manifest/runtime version `1.0.255`
- Product platforms: Windows x64 and Linux x64
- Current validation host: Blender `5.1.0` on Windows x64 with an NVIDIA
  GeForce RTX 4080 SUPER
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
│   ├── presets/*.json            eight built-in product presets
│   ├── examples/*.blend          generated Phase 2 example scenes
│   └── geometry_nodes/*.blend    generated immutable render-node asset
├── blender/
│   ├── scene_bridge.py           Blender/session integration
│   ├── setup_wizard.py           selected-object setup transaction
│   ├── production_bake.py        bake state machine and frame transactions
│   ├── cache_manager.py          inspect/free/move/rename/orphan operations
│   ├── command_line.py           headless contract and JSON reporting
│   ├── geometry_nodes.py         revisioned asset installation/input sync
│   ├── output_publication.py     typed attribute-aware publication
│   ├── ui_common.py              native panel bases and derived UI state
│   ├── ui_scene.py               Scene Properties workflow ownership
│   ├── ui_object.py              role-aware Object Properties workflow
│   ├── ui_cache_state.py         mutually exclusive bake presentation
│   ├── effect_ui.py              artist-facing Custom Effects workflow
│   └── effect_workspace.py       developer Warp Code workspace
├── io/
│   ├── cache_schema.py           strict cache-v2 manifest schema
│   ├── checkpoint.py             strict pickle-free checkpoint I/O
│   └── cache.py                  cache-v1 reader plus v2 frame adapter
├── sim/
│   ├── attributes.py             authoritative output/cache schema
│   ├── whitewater_types.py       stable type enum and host policy
│   ├── surface_tension_math.py   dependency-free physical/stability helpers
│   ├── surface_tension.py        CSF kernels and optional scratch owner
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

Per display frame, substeps are the maximum of the authored minimum, CFL
(`dt_cfl = cfl * voxel_size / (vmax + g*dt)`), and the explicit capillary
limit (`safety × sqrt(rho h³ / (4πσ))`), clamped to
`[min_substeps, max_substeps]`. The capillary limit is evaluated even before
the first source emission so a newly filled frame cannot bypass stability.
Each substep runs:

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
| — | forces (3) | gravity, optional directional/radial acceleration, velocity diffusion, particle-attribute dissipation, collision-SDF attraction, CSF surface tension, then speed limit |
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

### Surface tension (`sim/surface_tension.py`)

- Uses the simulation-grid liquid SDF with the negative-inside convention.
  Outward normals are normalized central SDF gradients and signed curvature is
  `-div(n)`, yielding an inward spherical force.
- Applies `a = sigma * kappa * delta(phi) * n / rho` in a compact cosine band,
  clamps curvature to `2 / voxel_size`, skips collision interiors, and rejects
  non-finite/overflowing acceleration before writing.
- A configurable normalized seven-point smoothing pass reduces normal noise.
  Two `vec3` sparse fields ping-pong normals and reuse the inactive buffer for
  acceleration, for 24 bytes per allocated voxel. Allocation is lazy and the
  profiler tally is reversed when the feature is disabled.
- Cell-centred acceleration is averaged to weighted staggered MAC faces before
  viscosity and pressure projection. The SDF key is invalidated after particle
  advection so whitewater and render surfacing cannot consume a pre-advection
  interface.
- Surface tension contributes no persistent stochastic or restart state.
  Scratch fields are intentionally reconstructed after reset/checkpoint restore.
- Cross-release playback and resume are separate contracts. Older frame caches
  remain readable, but production resume requires the current solver/settings
  signatures and, in 1.0.251, the exact effective Custom Effects identity.
  Phase-4 and pre-Custom-Effects checkpoints therefore require rebaking for a
  current resumable run even when the new force and effect stack are disabled.

### Whitewater (`sim/whitewater.py`, `WhitewaterSystem`)
- **One-pool state:** position, velocity, age, life, alive, and persistent int32
  `type` channels share one GPU capacity. Foam, bubble, and spray are not
  separate simulation systems.
- **Stable semantics:** `WhitewaterType.FOAM=0`, `BUBBLE=1`, and `SPRAY=2` in
  `sim/whitewater_types.py`; value `3` is reserved and rejected.
- **Churn** (`k_compute_churn`): accumulates velocity magnitude * divergence in
  fluid voxels, decaying each step (`churn_decay`/`churn_gain`). Churn is the
  only grid field carried across topology rebuilds.
- **Typed emission:** candidate type derives deterministically from liquid SDF,
  speed, and churn. Commit stores type and applies a type-specific lifespan
  multiplier in the same admitted transaction.
- **Hysteretic advection:** separated enter/exit SDF and speed thresholds avoid
  rapid type flicker. Foam follows the liquid with preservation/density terms;
  bubbles use fluid drag and buoyancy; spray uses air drag and scaled gravity.
- **Aligned lifecycle:** scan compaction, capacity clearing, cache capture,
  checkpoint restore, and frame output copy type in lockstep with every other
  whitewater channel. A device reduction reports foam/bubble/spray/invalid
  counts each frame.
- `k_whitewater_set_grid_data` splats whitewater into `ww_density` for viz.

### Phase-3/Phase-4 output attribute contract

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
inventory, and `surface_attribute_transfer_method`. Phase-4 schema version 2
requires `whitewater_type` and validates int32 values against the stable enum.
Read and publication paths fail closed on missing fields, wrong shape/type/domain,
non-finite values, misalignment, invalid UID data, or reserved type values.
Phase-3 schema-v1 frames retain playback compatibility as combined legacy-
unclassified output. Checkpoints additionally carry
`whitewater_type_schema_version=1`; older checkpoints are intentionally not
resumable because deterministic type-transition state cannot be inferred.

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

## 7. Custom Effects v2 Contract

`sim/effects/` is the authoritative custom-code product boundary. `sim/custom.py`
remains a one-release legacy conversion adapter; it is not the public runtime.

- `schema.py` defines strict effect, kernel, parameter, field-binding, instance,
  and diagnostic records with API 2 / ABI 1.
- `fields.py` owns domain, dtype, units, legal stages, permissions, persistence,
  runtime accessors, and prerequisite capabilities.
- `fingerprint.py` filters enabled/mute/solo state, rejects duplicate instance
  IDs, canonicalizes all execution inputs, and emits SHA-256 identity plus an
  ordered inventory.
- `trust.py` records local trust by exact content digest. A claimed built-in is
  trusted only when its digest equals a shipped definition.
- `validation.py` rejects unsafe or undeclared AST behavior before source
  generation.
- `source.py` creates deterministic content-addressed modules and maps generated
  diagnostics back to user Text lines.
- `compiler.py` imports the module and forces `wp.load_module` on the selected
  device. Import alone is never success.
- `sandbox.py` launches bounded particle/voxel fixtures, synchronizes, and checks
  mutation canaries plus finite data.
- `runtime.py` installs only complete candidates, derives topology/SDF/collision
  work from capabilities, schedules public stages/cadence, profiles calls and
  elements, and latches fail-closed after `EffectExecutionError`.
- `package.py` validates deterministic `.fseffect` ZIPs before decompression,
  including member/count/size/ratio limits and per-file SHA-256.

Public stages are Before Emission, After Particle Transfer, Before Pressure,
and After Particle Update. Cadence is Every Substep, Once Per Frame, or
Initialization. Particle-only effects do not force topology rebuilds; active-
voxel and SDF/collision consumers request only their declared prerequisites.

Cache-v2 manifests, frames, checkpoints, production-bake reports, headless
reports, and support reports include `effect_signature` and
`effect_inventory`. Production admission recomputes the current Text datablock
and controller snapshots instead of trusting stored update-callback state.

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
  `FluidSim Particles`, combined whitewater, foam, bubbles, and spray as padded
  PointCloud objects where available. It recreates conflicting managed
  attributes, validates schema/type values before mutation, preserves display-
  only `display_color` separately, and tracks live point count independently of
  padded capacity. Separate views filter aligned host arrays by type; they do
  not allocate separate GPU pools. Legacy-unclassified points remain visible
  only on the combined object.
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
- **`blender/geometry_nodes.py`** appends six immutable P4 revision-tagged
  groups from `resources/geometry_nodes/fluidsim_render_nodes.blend`, attaches
  one owned modifier per output, and synchronizes typed Blender 5.1+ material/
  radius inputs. Point groups realize instances before material assignment so
  named point attributes reach F12 shaders. Explicit updates skip user-owned
  modifiers and never mutate user node groups.
- **`blender/ui_common.py`**, **`blender/ui_scene.py`**, and
  **`blender/ui_object.py`** define the authoritative Scene/Object draw and RNA
  ownership surfaces. One derived `activity_snapshot` owns Idle, Live Preview,
  Cache Playback, and Production Bake presentation;
  `pending_settings.pending_presentation` remains the lifecycle-policy owner.
  The Scene hierarchy follows setup → domain → physics → Custom Effects →
  reconstruction/whitewater → outputs → cache → diagnostics. Object panels
  poll canonical SOURCE/SINK/COLLIDER/FORCE/FIELD roles and hide inapplicable
  analytic, mesh, point, constant, and volume payloads.
- **`panels.py`** retains the compact View3D status/lifecycle surface.
  **`blender/ui_view3d.py`** mirrors the authoritative Scene and Custom Effects
  panels beneath one collapsed Full Settings parent by delegating to their draw
  and poll functions; it introduces no second settings model.
- **`blender/ui_physics.py`** recognizes the domain by exact scene pointer,
  outputs by stable tags, and inputs by canonical object role. It reuses existing
  Scene/Object panels and routes single-output deletion through identity-checked
  Output Publication ownership. **`preferences.py`** remains the device owner.
- **`blender/effect_ui.py`** draws Effect Stack, Selected Effect, Workflow, and
  Status in Scene Properties. Package administration is a closed child.
  **`blender/effect_workspace.py`** is a closed Warp Code child that prefers an
  existing Text Editor before repurposing the active area. Legacy snippet
  panels are not registered.
- **`blender/ui_cache_state.py`** presents exactly one Start, Resume, or Restart
  production action from the manifest boundary. Full checks remain in the bake
  execute path. Cache mutation dialogs bind to the exact resolved path or
  orphan inventory; `cache_manager` independently refuses arbitrary
  directories and recognizes interrupted first-frame FluidSim transactions.
- **`properties.py`** defines the scene + object PropertyGroups. Attribute-viz
  item tuples and the `liquid_view_mode` update callback live here. Field
  payload callbacks keep Volume Grid coverage and Volume Grid payload selection
  coherent so no configured payload becomes unreachable.

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

`sim/validation.py` defines **21 tests**, each decorated with
`@_test(name)` and registered in `_TESTS`. `run_all(device, progress)` executes
them on the real CUDA device and wraps `(passed, detail)` in a
`ValidationResult`. The first sixteen retain sparse-grid, transfer, source,
collision, pressure, surface, viscosity, whitewater, profiler, and ST-FLIP
coverage. Tests 17–21 add static-droplet boundedness/volume proxy, a
Rayleigh–Lamb oscillation trend oracle, zero-coefficient full-solver
equivalence, capillary timestep trends, and finite-energy/diagnostic guards.
Failures are reported rather than crashing the suite.

`release/validate_phase5_warp.py` is a separate supplied-package gate. It
executes all P5 kernels on one analytic sparse leaf, validates inward force,
finite/clamped curvature and acceleration, linear coefficient scaling,
collision exclusion, MAC integration, optional-scratch lifecycle, and timestep
trends. It also invokes the real frame-level solver limiter with an empty initial
particle pool, proving that capillarity selects 27 requested substeps in the
probe case and reports the authored clamp to 8. CPU execution proves code paths;
`cuda:0` execution is required for release certification.

`release/validate_properties_ui_blender.py` runs in Blender 5.1 rather than
treating source strings as UI proof. It draws 49 authored surfaces, validates
one collapsed root plus 51 View3D mirrors with exact poll/draw parity, exercises
domain/input contexts and all six output kinds in Physics Properties, revalidates
guarded domain/output actions, covers ten role/geometry configurations, then
saves/reopens and unregisters/re-registers the extension. Installed-package
smoke separately requires the same modules and representative panels.

---

## 12. Verification Results for 1.0.255

The table records current executable evidence from the integrated Windows
source at proof HEAD `7643ddcc7084e07d0a941ecb96243e056e6b4275`.

| Gate | Result in this environment |
|---|---|
| Full Python suite | **1,033 passed**; only the linked-worktree pytest cache emitted a non-test warning |
| Python compilation | Passed |
| Source/release verification | Passed; seven example scenes, three durable Custom Effects renders, eight built-ins, and no bundled Blender/Warp payload |
| Blender host | **5.1.0 Windows x64**; GUI acceptance passed **40/40** and saved-demo timeline playback passed |
| Native UI gate | Passed: 49 authored draws, 51 delegated View3D mirrors under one collapsed root, domain/input/six-output Physics contexts, guarded actions, ten role cases, save/reopen, and registration cleanup |
| Warp/runtime | External Warp **1.16.0**, CUDA `cuda:0`, NVIDIA GeForce RTX 4080 SUPER |
| P5 analytic kernel proof | Inward droplet force, finite normals/curvature/acceleration, curvature clamp, 2× coefficient scaling, collision exclusion, MAC integration, scratch lifecycle, and real solver-limiter selection/clamp passed |
| Cache/checkpoint behavior | P5 adds no persistent stochastic state; cache-v1 and older cache-v2 playback remain; resume remains manifest/settings-signature strict |
| Installed package lifecycle | Pending the final 1.0.255 archive build after this source proof; the unchanged 1.0.253 packaging lane previously passed twice in clean profiles |
| Installed headless no-CUDA exit | Prior packaging proof exited `12`, wrote an atomic report, committed **0 frames**, and removed the package cleanly; final 1.0.255 artifact rerun remains required |
| Custom Effects Python suite | Strict schema, fields, fingerprints, trust, JIT, sandbox, runtime, packages, cache, policy, UI-source, and demo contracts passed |
| Custom Effects supplied-Warp proof | Eight built-ins plus positive/negative fixtures compiled and executed on Warp 1.16.0 CPU; false-positive compile cases were rejected |
| Custom Effects supplied-Blender proof | Three demos, F12 evidence, Test/Apply, controller, order/mute/solo/duplicate, line mapping, trust, and save/reopen passed |
| Release workflow coverage | Linux package job now runs and uploads the native Properties UI report in addition to P4/P5 Warp, Custom Effects Warp/Blender, package build, and installed smoke evidence |
| NVIDIA CUDA solver gate | **Passed 21/21**, including pressure, viscosity, collision/sink masks, ST-FLIP, surface tension, and bounded diagnostics |
| Interactive user path | **Passed**: 6.116 ms mean viewport publication, 16.336 ms mean total, with no added readbacks, synchronization, Blender ID churn, depsgraph updates, or structural replacement |

The GUI-inclusive production gate proves the current Windows source, Blender
registration/lifecycle, CUDA solver invariants, cache-v2 workflow, demo playback,
and bounded interactive publication. GPU results are physical/schema/lifecycle
bounds, not a claim of bitwise determinism across atomic or compaction order.

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
- Surface tension allocates two optional `vec3` fields (24 bytes per sparse-grid
  voxel) only while enabled. Fine voxels and high coefficients can make the
  capillary limit, rather than CFL, dominate substep cost.
- Typed whitewater always uses one persistent int32 type array. Separate Blender
  outputs are filtered publication views; `BOTH` duplicates host/render views,
  not the device simulation pool.

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

## Current promoted demo evidence

The GUI-inclusive 1.0.255 production gate at proof HEAD
`7643ddcc7084e07d0a941ecb96243e056e6b4275` passed all 18 bounded demo checks.
Its promoted values are `front_speed_over_c0=1.0109923602`,
`column_height_end=0.3391113281`, `crown_sqrt_t_corr=0.9713800298`, and
`max_crown_height=0.2380029410`. The authoritative source is
`artifacts/demo_analysis/analysis_report.json`, SHA-256
`6ce0fb25a8e31102f82171a62b97673f96780c3ec708bf34381f68a00ad20d76`.
