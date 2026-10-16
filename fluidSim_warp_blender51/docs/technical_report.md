# FluidSim Warp — Technical Report

FluidSim Warp 4.0.0 targets Blender 5.2.0 or newer around the existing
sparse 8³-leaf NVIDIA Warp PIC/FLIP solver. Live Preview remains interactive;
Production Bake adds a separate, manifest-authoritative cache-v3 transaction and
checkpoint layer without rewriting the solver into a different architecture.

- Extension ID: `fluidsim_warp` · manifest/runtime version `4.0.0`
- Product platforms: Windows x64 (certified for 4.0) and Linux x64 (uncertified
  in 4.0, including USD export; certification is planned for 4.1)
- Historical validation host: Blender `5.2.0` on Windows x64 with an NVIDIA
  GeForce RTX 4080 SUPER (earlier evidence, not current candidate qualification)
- Runtime dependency: NVIDIA Warp `1.18.0+cu13` (1.17.0 still loads), installed from Preferences (never bundled)
- Solver backend policy: NVIDIA CUDA only; CPU is used only for surrounding
  package/runtime smoke evidence
- Persistent cache policy: v3 writes/playback/resume; v1 and v2 are refused

## Current installer candidate qualification

The 4.0.0 candidate is the `release/4.0` branch (PR #30). Its per-item
acceptance, the Multigrid stop-rule decision and the gate result are recorded
in `../../docs/RELEASE_4_0_ACCEPTANCE.md`. Release approval requires the
candidate package checks and the GPU/GUI production gate on Windows x64.

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
│   ├── cache_schema.py           strict cache-v3 manifest schema
│   ├── checkpoint.py             strict pickle-free checkpoint I/O (format 2)
│   ├── vdb_writer.py             OpenVDB level-set writer (not wired in)
│   └── cache.py                  cache-v3 frame writer/reader; refuses v1 and v2
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
`sim/` free of `bpy`. Repository-level `tools/release/` owns build, smoke,
generation, and validation workflows; the runtime package retains only the
installed headless entry point.

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
  Jacobi/PCG selection, convergence handling, thresholding,
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

### Multigrid-preconditioned PCG (`sim/pressure.py`, opt-in)
`pcg_preconditioner = MULTIGRID` replaces the Jacobi preconditioner inside the
same PCG loop (default stays Jacobi). `_apply_mg_vcycle` applies one symmetric
V-cycle per iteration:

- 4 levels per 8³ leaf (8³, 4³, 2³, 1³), so no cross-leaf coupling is coarsened.
- 2 pre- and 2 post-smoothing sweeps of damped Jacobi (omega 0.67), 4 sweeps on
  the coarsest level (16 before 4.0).
- The residual is checked after every iteration (4.0; every 5th before), so
  the solve stops at the first V-cycle that meets the tolerance.
- Restriction is 0.25 × the gathered sum of the children; prolongation is
  piecewise-constant. Both are gathers: the cycle is atomic-free.
- The cycle is captured in the CUDA conditional graph with the rest of the PCG
  loop.

Measured on the 32³ test problem: 13 PCG iterations against 123 with Jacobi. A
per-frame speed-up has not been measured.

### Viscosity (`sim/viscosity.py`, `ViscositySolver`)
- Explicit smoothing (`smooth`) or implicit PCG (`solve`, `use_pcg=True`).
- Viscosity field is either a global constant (`viscosity_constant`) or splatted
  from the per-particle viscosity attribute (`viscosity_from_particles`,
  averaged via `k_particle_to_grid_collect`/`k_particle_to_grid_average`).
- Implicit viscosity **damps, never amplifies** (validated by test 11).

Both solvers operate only on fluid/solid voxels via the voxel-type mask.

---

### Particle-grid transfer: PIC, FLIP, APIC
`ParticleSampleType` has PIC, FLIP (default) and APIC (`= 2`). APIC runs its own
`*_apic` kernels (P2G and G2P); the PIC/FLIP kernels are unchanged, so FLIP
results are bitwise identical to the build before APIC. Each particle carries a
3x3 affine matrix (`mat33`, 36 B), double buffered for compaction, so 72 B per
particle, allocated only when APIC is selected (`ensure_affine`). Angular
momentum retained in the probe: PIC 0.596, FLIP 0.940, APIC 0.939. APIC does not
beat FLIP there because the loss is in the shared free-surface path; the
enforced gate is APIC > 0.90 and > PIC + 0.25. Only momentum was measured;
reduced visual noise is not claimed.

### Resolution Divisions (`sim/quality_contract.py`)
The grid is authored as `resolution_divisions` cells along the longest domain
axis (default 64, minimum 8). The cell size is `longest axis / divisions`;
`voxel_size` is a derived get/set property (set rounds to the nearest division).
Cell counts per axis use `ceil(extent / cell * (1 - CELL_COUNT_TOLERANCE))` with
`CELL_COUNT_TOLERANCE = 1e-6`: a float32 cell size sits a few parts in 1e8 low,
and an unguarded `ceil` added a whole cell and then an 8³ leaf of padding. The
grid is padded to whole leaves. Files saved by 2.x migrate on load
(`round(longest axis / voxel size)`, minimum 8). Changing divisions is a REINIT
tier setting.

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

Cache-v3 frame payloads include `attribute_schema_version`, an ordered JSON
inventory, and `surface_attribute_transfer_method`. Phase-4 schema version 2
requires `whitewater_type` and validates int32 values against the stable enum.
Read and publication paths fail closed on missing fields, wrong shape/type/domain,
non-finite values, misalignment, invalid UID data, or reserved type values.
Attribute-schema-v1 frames (no whitewater type) are still accepted by the frame
validator and publish as combined legacy-unclassified output, but no 3.0 bake
writes them and 2.x caches are refused outright. Checkpoints additionally carry
`whitewater_type_schema_version=1`; older checkpoints are intentionally not
resumable because deterministic type-transition state cannot be inferred.

---

### Cache format 3 and checkpoint format 2 (`io/cache_schema.py`, `io/checkpoint.py`)
- The manifest `FORMAT_VERSION` is 3. 1.0 and 2.x caches raise
  `LegacyCacheFormatError`; 2.x carries `LEGACY_2X_CACHE_MESSAGE`. The headless
  bake maps it to exit code 14. Nothing is migrated.
- Each frame records `source_frame_stamp`: a sha256 over the world matrix and
  evaluated geometry of every source and collider at that frame. Resume compares
  stamps and refuses at the first mismatch with `SOURCE_ANIMATION_CHANGED`; the
  UI offers Re-bake from Frame N.
- `liquid_sdf_*` arrays are written empty until the narrow band lands.
- Checkpoint format 2 always stores two reserved slots: `particle_affine`
  (float32, (N, 9); reserved for the APIC matrix) and
  `narrow_band_seed_state` (uint32). A checkpoint without them is rejected, and
  format 1 reports "unsupported checkpoint format 1; expected 2".

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
- Kernel identity registration covered all **188** canonical names when the
  reference kernel manifest was removed (it lives in git history).

---

## 7. Custom Effects v2 Contract

`sim/effects/` is the authoritative custom-code product boundary. 2.0 removed
the 1.x snippet data and its conversion adapter: 1.1 converted saved snippets
on load, and admission refuses a scene whose snippets were never converted
(`blender/file_compat.py`).

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

Cache-v3 manifests, frames, checkpoints, production-bake reports, headless
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
  `blender/viewport_overlay.py` remains the diagnostic overlay adapter.
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
- **`panels.py`** draws the pinned View3D status/transport strip: activity,
  runtime, Preview / Pause / Resume / Restart, and Apply Changes. It binds no
  settings. **`blender/ui_stages.py`** hosts the staged sidebar (Setup, Simulate,
  Look, Bake & Export) and reuses the authoritative Scene draw functions
  (`draw_cache_playback`, `draw_cache_bake`, `draw_cache_manage`) and
  `pending_settings.setting_owner` ownership; it introduces no second settings
  model. The former Live Tuning / Full Settings mirrors and the
  Scene-tab Transport and Cache panels were removed because a stage now owns
  each control.
- **`blender/ui_header.py`** appends a **FluidSim** popover
  (`FLUIDSIM_PT_header_popover`, `VIEW_3D`/`HEADER`) to `VIEW3D_HT_header` and
  registers `fluidsim.open_workspace`. The popover draws `panels.draw_transport`,
  `panels.draw_pending_apply`, `ui_stages.draw_quality` and
  `ui_stages.draw_simulate_controls`, the same module-level functions the strip
  and stages call, so the surfaces share one implementation and one set of
  `setting_owner` bindings. The operator switches to an existing workspace named
  `FluidSim`, or runs `bpy.ops.workspace.duplicate()`, finds the new workspace by
  name difference, renames it, and `prepare_workspace` guarantees a 3D Viewport
  with `show_region_ui`, a Timeline (`DOPESHEET_EDITOR`, `ui_type` TIMELINE, taken
  from the smallest area that is not a Viewport or Properties editor) and Object
  mode. Blender 5.2 gives an add-on no way to select the sidebar tab
  (`Region.active_panel_category` is read-only) or to delete a workspace
  through the data API except `bpy.data.batch_remove`; the add-on never deletes
  the workspace, and the validator removes its own copy that way.
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

`sim/validation.py` defines **20 tests**, each decorated with
`@_test(name)` and registered in `_TESTS`. `run_all(device, progress)` executes
them on the real CUDA device and wraps `(passed, detail)` in a
`ValidationResult`. The first fifteen retain sparse-grid, transfer, source,
collision, pressure, surface, viscosity, whitewater, and profiler
coverage. Tests 16–20 add static-droplet boundedness/volume proxy, a
Rayleigh–Lamb oscillation trend oracle, zero-coefficient full-solver
equivalence, capillary timestep trends, and finite-energy/diagnostic guards.
Failures are reported rather than crashing the suite.

`tools/release/validate_phase5_warp.py` is a separate supplied-package gate. It
executes all P5 kernels on one analytic sparse leaf, validates inward force,
finite/clamped curvature and acceleration, linear coefficient scaling,
collision exclusion, MAC integration, optional-scratch lifecycle, and timestep
trends. It also invokes the real frame-level solver limiter with an empty initial
particle pool, proving that capillarity selects 27 requested substeps in the
probe case and reports the authored clamp to 8. CPU execution proves code paths;
`cuda:0` execution is required for release certification.

`tools/release/validate_properties_ui_blender.py` runs in Blender 5.1 rather than
treating source strings as UI proof. It draws every authored surface, validates
the collapsed root plus each View3D and Physics mirror with exact poll/draw parity, exercises
domain/input contexts and all six output kinds in Physics Properties, revalidates
guarded domain/output actions, covers ten role/geometry configurations, then
saves/reopens and unregisters/re-registers the extension. Installed-package
smoke separately requires the same modules and representative panels.

---

## 12. Verification Results for 1.0.255

The table retains the 1.0.255 verification baseline originally captured from
the integrated Windows source at proof HEAD
`bb6d3fc5a5be9b944d4316c46aeb9ed47f011409`. The current GUI-inclusive gate
and promoted demo evidence are recorded below the baseline.

| Gate | Result in this environment |
|---|---|
| Full Python suite | **1,043 passed** (historical baseline row; the original Windows run did not record whether this counted the nested package suite alone or the parent repository suite, and no run at that proof HEAD is available to re-attribute it — see current counts below the table) |
| Python compilation | Passed |
| Source/release verification | Passed; seven example scenes, three durable Custom Effects renders, eight built-ins, and no bundled Blender/Warp payload |
| Blender host | **5.1.0 Windows x64**; GUI acceptance passed **40/40** and saved-demo timeline playback passed |
| Native UI gate | Passed (1.0.255 counts: 49 authored draws, 51 delegated View3D mirrors under one collapsed root; 3.0 changed the panel set, so those counts no longer apply), domain/input/six-output Physics contexts, guarded actions, ten role cases, save/reopen, and registration cleanup |
| Warp/runtime | External Warp **1.16.0**, CUDA `cuda:0`, NVIDIA GeForce RTX 4080 SUPER |
| P5 analytic kernel proof | Inward droplet force, finite normals/curvature/acceleration, curvature clamp, 2× coefficient scaling, collision exclusion, MAC integration, scratch lifecycle, and real solver-limiter selection/clamp passed |
| Cache/checkpoint behavior | P5 adds no persistent stochastic state; cache-v1 and older cache-v2 playback remained in 1.0.255 (3.0 refuses both); resume remains manifest/settings-signature strict |
| Installed package lifecycle | **Passed** twice from the final 1.0.255 Windows archive in clean profiles on Blender 5.1.0 with external Warp 1.16.0; registration cycled twice per run, Custom Effects applied, CUDA was detected, and the builder reported `release_ready=true` |
| Installed headless no-CUDA exit | **Passed (rerun 2026-09-24, master `b5e0050` plus the signal-restore fix).** The rebuilt 1.0.255 Windows archive was installed into a clean profile on Blender 5.1.0 with external Warp 1.16.0 and `CUDA_VISIBLE_DEVICES=-1` (an empty value does not hide devices on Windows). `release/run_headless_bake.py` on `dam_break` exited `12`, wrote an atomic report, committed **0 frames**, left no cache frames, and the package was removed cleanly. The same rerun found that every CLI bake crashed on exit restoring a C-level signal handler; fixed in `blender/production_bake.py` |
| Custom Effects Python suite | Strict schema, fields, fingerprints, trust, JIT, sandbox, runtime, packages, cache, policy, UI-source, and static validation-example contracts passed |
| Custom Effects supplied-Warp proof | Eight built-ins plus positive/negative fixtures compiled and executed on Warp 1.16.0 CPU; false-positive compile cases were rejected |
| Custom Effects supplied-Blender proof | Three static validation/review scenes, F12 evidence, Test/Apply, controller, order/mute/solo/duplicate, line mapping, trust, and save/reopen passed; this is not live demo-playback proof |
| Release workflow coverage | Linux package job now runs and uploads the native Properties UI report in addition to P4/P5 Warp, Custom Effects Warp/Blender, package build, and installed smoke evidence |
| NVIDIA CUDA solver gate | **Passed 21/21**, including pressure, viscosity, collision/sink masks, ST-FLIP, surface tension, and bounded diagnostics |
| Interactive user path | **Passed**: 7.785 ms mean viewport publication, 23.353 ms mean total, with no added readbacks, synchronization, Blender ID churn, depsgraph updates, or structural replacement |

The GUI-inclusive production gate proves the current Windows source, Blender
registration/lifecycle, CUDA solver invariants, cache workflow (v2 in 1.0.255), demo playback,
and bounded interactive publication. GPU results are physical/schema/lifecycle
bounds. Since `d2a6b6d` the dam_break and crown_splash bakes are bitwise
reproducible run-to-run on one device. Mesh and compiled-source emitters and
some float atomics (color mixing, splat, pressure RHS) are still unaudited, and
there is no cross-device determinism claim.

### Current attributed Python suite counts

Distinct from the historical, unattributed `1,043 passed` baseline row above,
these two counts are each from a specific, reproducible scope, run in a
GPU-less cloud session after merging master's `21d4d37` (the demo-bake
determinism fix and four-demo cleanup), 2026-09-24:

- Nested package suite (`python3 -m pytest fluidSim_warp_blender51/tests -q`):
  **426 passed**.
- Parent repository suite (`python3 -m pytest -q` from the repo root):
  **1,440 passed, 116 skipped**. The skips are CUDA/`bpy`-only tests that
  require GPU/Blender hardware this environment doesn't have; they are not
  failures.

These are the counts on the tip of this PR's branch as merged (verified
directly against that checkout, not carried forward from an earlier commit
message). An earlier 2026-09-22 run at HEAD `df1868b`, before merging
`21d4d37`, reported `427`/`1,443 passed, 115 skipped` — the difference is
`21d4d37` removing the adhesive-whitewater calibration tool and its tests
along with the pruned non-demo scenes, not drift or a stale rerun. Likewise,
this PR's own CUDA-test-fix commit message (`ba9ca3d`, 2026-09-21) cites
`1443 passed, 115 skipped` for the tree as it stood *at that commit*, before
`21d4d37` landed; it is a historical record of that commit, not a claim
about the current tree.

Neither count is a substitute for the GUI-inclusive gate's own scoped pytest
lane (`tests/solver/test_pressure_projection_regressions.py`,
`tests/cache_io/test_cache_io.py`, `tests/cache_io/test_io_path_policy.py`; 79 passed at
`gate_run_at: 2026-09-22T06:23:28Z`) — that lane is real-GPU evidence for
those regressions specifically, while the two counts above are broader but
CPU-only.

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

## Decoupled render surfacing

Historical feature note: version 1.0.235 added a separate sparse render-surface grid with multiplier or
custom voxel sizing, frame/manual rebuild cadence, filtering, bounded dense
extraction, and shared viewport/cache/export mesh ownership. The simulation
grid remains authoritative for physics and optional low-resolution surface
fields.

## Current promoted demo evidence

The GUI-inclusive 4.0.0 production gate (Warp 1.18.0) at proof HEAD
`f4c38cd56444f7b73efcb0528b76e5f74f8537a6` passed all 21 bounded demo checks
(analysis report sha256 `b20d6606ea3a10f34feb99aeca48bcd87592a977e36c7b634e91f0a63300bcf8`).
Its promoted values are `front_speed_over_c0=1.0742932332`,
`column_height_end=0.6299800873`, `crown_sqrt_t_corr=0.9711003511`, and
`max_crown_height=0.0875732452`. The authoritative source is
`artifacts/demo_analysis/analysis_report.json`.

These values are reproducible. Before `d2a6b6d`, `max_crown_height` varied
by about 16% between runs of the same commit. The cause was four
order-dependent sites: the PCG dot-product combine, the P2G scatter, the
shape-emission slot reservation, and the whitewater free list. That commit
made all four order-independent. The dam_break and crown_splash bakes are now
bitwise identical run-to-run on one device, and two consecutive gate runs (at
`d2a6b6d` and `cb9ad00`) produced the same artifact SHA-256.

Since `19864c3`, `max_crown_height` is the peak crown height over the crown
phase only. That window runs from the moment the droplet has fully entered the
pool to rim collapse, which is frames 2-5 on the shipped scene. The earlier
statistic took its maximum at frame 15, which is Worthington-jet spray. It was
deterministic but chaotic: a 1e-6 relative nudge to the drop speed moved it by
up to 20%. The crown-phase value is unchanged under the same nudges.
