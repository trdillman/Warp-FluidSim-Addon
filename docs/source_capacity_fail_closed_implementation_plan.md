# Source Capacity Planner and Fail-Closed Emission — Implementation Plan

## 1. Truth contract

The definitive behavior is:

1. A liquid/particle source must never be partially populated merely because the fixed GPU particle pool or sparse-leaf pool was exhausted.
2. Initial scene population covers automatic point seeds plus all armed frame-one liquid emitters as one admitted configuration. Particle-only shortfalls automatically choose a guarded runtime capacity before solver creation; configurations that still cannot be proven safe remain blocked before replacing an existing solver.
3. Every later source-emission batch is also admitted or rejected as a whole. Continuous inflow must not reintroduce the same block-hole defect later in playback.
4. A capacity-deferred batch emits zero particles and does not mark `ONCE`/`INITIAL_FILL` sources as fired. The initialized solver is never released by the guarded growth path. Depending on the advanced user preference, playback either continues while the whole source batch remains armed for retry, or Live pauses while preserving solver state.
5. Planning and drawing never mutate authored settings. **Initial Particle Reserve** is a starting allocation; guarded runtime growth allocates/copies/swaps transactionally and reports the active capacity without rewriting the scene value. Semantic resolution/domain alternatives still require explicit approval.
6. Particle and leaf requirements may be labelled **exact**, **conservative upper bound**, or **unknown**. Unknown fails closed. VRAM results are always labelled estimates, not OOM guarantees.

This is broader than hiding the visible holes: it removes capacity overflow as a normal control path.

## 2. Evidence and current verdict

### Confirmed source behavior

- `sim/sourcing.py::k_source_particles`, `k_source_owned_points`, `k_source_shapes_particles`, `k_mesh_source_particles`, and `k_emit_compiled_source_particles` reserve slots with `atomic_add` and return/break after `slot >= capacity`. Earlier threads have already committed particles.
- `SourcePipeline.apply_compiled()` only reports `capacity_limited` after mutation and can mark one-shot sources fired after a truncated launch.
- The legacy height/analytic/mesh path stamps grid state and advances its random seed before capacity success is known.
- `Session.initialize()` and `Session.reset()` inject automatic seeds separately from first-step source emission.
- Existing runtime reports repeatedly land exactly on configured capacities (for example 600,000), which corroborates saturation. A dedicated runtime regression reproducer is still required before claiming the complete causal chain proven in Blender.

### Claim matrix

| Claim | Current status | Required implementation/evidence |
|---|---|---|
| Capacity exhaustion cannot create arbitrary empty blocks | **Nonconformant** | Whole-batch count/admission/commit plus runtime spatial regression |
| Frame-one population is all-or-nothing | **Missing** | Aggregate seed/source preflight and no-mutation rejection tests |
| Later continuous emission cannot truncate spatially | **Nonconformant** | Per-substep batch gate |
| Required particle count is exact | **Partial** | Exact for direct points; identical count-only kernel pass for runtime rasterized demand; otherwise label bounds |
| Required sparse leaves are known | **Partial** | Shared 8³ leaf-union/expansion logic and initial/full-domain classifications |
| Suggested configuration will not OOM | **Must not be claimed** | Conservative estimate plus explicit disclaimer and real-device evidence |
| Suggestions do not alter authored setup | **Missing contract** | Immutable report, UI-only presentation, RNA snapshot tests |
| Existing source lifecycle remains correct | **At risk** | Fired-state commit only after admitted successful emission |

## 3. Architectural seams

### 3.1 New pure module: `fluidSim_warp_blender51/sim/capacity_planner.py`

Own all policy and report construction behind a small Blender-independent API:

```python
plan_initialization(settings, source_footprints, seed_payloads,
                    collider_footprints, resources) -> CapacityPlan

suggest_configurations(plan, settings, resources) -> tuple[CapacitySuggestion, ...]
```

Use frozen data objects:

- `DemandValue(value, quality=EXACT|UPPER_BOUND|UNKNOWN, reason)`
- `SourceDemand(stable_id, lifecycle, particles, leaves, staging_bytes)`
- `CapacityBlocker(code, required, available, source_ids)`
- `CapacitySuggestion(changes, estimated_bytes, tradeoffs, disclaimer)`
- `CapacityPlan(verdict, particle_demand, leaf_demand, memory_estimate, blockers, suggestions, scene_signature)`

The module must not import `bpy`, allocate Warp arrays, inspect Blender geometry, or mutate settings.

### 3.2 Source footprint seam: `sim/source_regions.py`

Use the existing `CompileContext.upload` flag with `upload=False` (already exercised by source-region tests) rather than creating a second rasterizer. Extend that no-device-upload path to expose a compact immutable footprint from `CompiledSourceBatch`:

- stable source identity, operation, lifecycle, cadence;
- direct-point count;
- compact leaf coordinates and coverage;
- covered voxel count;
- target occupancy after the real 1..64 clamp and rounding rule;
- compiler diagnostics and host/device staging estimates.

The footprint API must preserve the flat fixed 8×8×8 leaf contract. It must not invent hierarchical/adaptive storage.

### 3.3 Blender planning snapshot: `blender/scene_inputs.py` and `blender/sources.py`

Add one read-only gather operation that evaluates the scene once and returns:

- normalized source plans/descriptors in deterministic stable order;
- prepared automatic-seed point payloads without injection;
- height source;
- collider/source bounds needed for leaf and domain suggestions;
- a signature covering geometry, transforms, lifecycle settings, voxel size, and source ordering;
- explicit cleanup ownership for evaluated meshes/OpenVDB data.

Reuse this snapshot during normal `sync_scene()` after admission so expensive evaluated geometry is not gathered twice.

Split `inject_auto_seeds()` into `prepare_auto_seeds()` and `commit_auto_seeds()`. Preparing may allocate bounded host arrays but must not touch solver particle buffers.

### 3.4 Runtime authority: `sim/sourcing.py`

Introduce a single `EmissionTransaction` coordinator used by compiled, legacy analytic, mesh, height, direct-point, and automatic-seed emitters.

Each emitter supplies a count-only operation that uses the same predicates, occupancy clamp, coverage rounding, domain lookup, and lifecycle state as its commit kernel. The coordinator:

1. resets aggregate demand/scratch occupancy;
2. counts the entire liquid batch without particle/grid/source-lifecycle mutation;
3. atomically sets one device `admitted` flag when `current_count + demand <= capacity`;
4. launches commit kernels guarded by that flag;
5. reuses the existing aggregate source-result readback so admission adds no per-source or additional device synchronization; the performance budget is one aggregate source result read at the existing end-of-source-batch seam when an emitter ran;
6. commits particle-source `_initial_fired` entries and RNG advancement only after success.

Compiled one-shot pressure and post-P2G scalar/vector field lifecycles do not allocate particles and are outside `EmissionTransaction`. Keep their existing phase-specific fired-state commits, but give them separate tests proving a rejected liquid transaction neither fires nor suppresses those non-particle operations. Force sources remain post-P2G and do not use one-shot particle admission.

Kernel bounds checks remain as corruption defenses. Any overflow despite admission is a fatal invariant violation: stop/release the session rather than keep partial output.

For overlapping sources, the count pass must mirror actual ordering and per-voxel occupancy updates. Until a legacy emitter has a predicate-identical count pass, classify it as unknown and block; do not guess from `covered_voxel_count`.

### 3.5 Solver transaction seam: `sim/solver.py`

Factor the source-phase admission boundary so the order is explicit:

`allocate topology footprint -> clear/collisions -> index current particles -> count/admit all liquid emitters -> commit fluid/source-type stamps and particles -> reindex -> P2G`

`_rebuild_topology()` may allocate/mark the height source's **topology footprint** before admission; that is address-space preparation, not a `VOXEL_FLUID` source stamp. The height `k_source_height` fluid/type/velocity stamp must move behind or be guarded by the same admission flag as height particle emission. On rejection, provisional topology may exist only inside a solver session that is stopped/discarded; no fluid stamp, particles, cache frame, or viewport frame may be committed.

Initial source/seed capacity is planned before destructive reset/reinitialize. Automatic seeds may be committed during initialization only after the complete unchanged scene signature has proved that seeds plus the conservative frame-one source bound fit. Before first source execution, recheck the signature. A changed signature invalidates the plan and requires replan/reinitialize; it must not proceed on stale admission.

### 3.6 Session and UI seams

`blender/scene_bridge.py` owns ordering and persistence:

1. build settings and planning snapshot;
2. run source/leaf plan and existing VRAM assessment before releasing a current solver;
3. on rejection, retain the current solver/pending action where one exists;
4. only then release/reallocate, sync the admitted snapshot, and commit seeds;
5. store `Session.capacity_plan` plus draw-safe cached presentation;
6. on runtime rejection, stop playback before cache/viewport publication.

`panels.py` presents required/available particles and leaves, demand quality, source attribution, VRAM estimate/budget, and suggestions. Source-integrity failures must not expose the existing **Run Anyway** memory bypass. An override cannot make partial emission truthful.

`operators.py` may offer explicit “apply selected suggestion and reinitialize,” but only as a later user action with confirmation. Drawing or planning never writes RNA. The existing **Apply Safe Settings** memory-recovery operator remains an explicit user action and may continue changing a copied recommendation into RNA values; it must display the exact changes and must not be called automatically. `ignore_memory_safeguard=True` may bypass only the conservative VRAM estimate—it must never bypass particle/leaf integrity blockers or enable partial source emission.

## 4. Capacity calculations

### Particle demand

- Direct append and prepared automatic seeds: exact finalized point-array length.
- Compiled raster source at runtime: exact count-only kernel using `ceil(target * coverage)`, current voxel occupancy, valid leaf lookup, and the runtime target clamp/rounding.
- Initialization planner: exact when all inputs and overlap semantics are available; otherwise a conservative upper bound such as the sum of per-voxel targets before collider/sink subtraction.
- Direct append kernels currently perform no domain filter, so outside-domain points consume particle capacity even when topology marking ignores them; WP0 must lock this behavior with a test before the planner encodes it.
- Continuous inflow has no finite lifetime requirement. Report exact current-substep demand and optional explicitly labelled N-frame scenarios only.
- Validate all host totals with Python integers and reject values that cannot be represented safely by device counters.

### Sparse leaf demand

Compute a deduplicated set of leaf coordinates from source coverage, in-domain seeds/current particles, height fill, colliders, and required field regions. Apply the same domain clipping and `topology_expand` semantics as `SparseGrid.expand()`. Test coordinates on every 8-voxel boundary.

Report separately:

- exact/conservative initial active leaves;
- configured leaf capacity;
- full-domain leaf-cell upper bound;
- dynamic-motion uncertainty for later frames.

### VRAM and configuration suggestions

Reuse `estimate_solver_bytes()` and the existing live budget `min(85% total, 90% free)`, adding bounded source staging and transaction scratch allocations. Never call blocking `resource_snapshot()` from a frame handler.

Produce ranked, non-mutating candidates:

1. **Preserve authored domain/resolution:** raise particle/leaf capacity only when estimated under budget.
2. **Preserve bounds/density:** monotonically increase voxel size until particle, leaf, and estimated-memory constraints fit.
3. **Tighten empty X/Y/Z margins:** propose voxel-aligned domain-object bounds enclosing all required source/collider/seed bounds plus topology/motion margin. Never crop required content automatically.
4. **Balanced:** search bounded combinations of modest voxel-size increase, empty-margin tightening, and capacity changes.

Every candidate states changed parameters, lost resolution/domain margin, estimated peak bytes, live budget, and “estimate—not an OOM guarantee.” Use bounded binary/grid search with iteration limits, not an open-ended tuning loop.

## 5. Test-first work packages

### WP0 — Contract lock, then owned reproducer (critical path)

**Read-only owner:** lead/oracle maps and freezes the contract first. **Writer:** one test specialist then owns the new reproducer file in an isolated worktree.

- Freeze whole-batch atomicity, exact/bound/unknown terminology, rounding/clamping, stable source ordering, and diagnostic codes without edits.
- The assigned test writer creates a minimal CUDA/CPU reproduction that exceeds particle capacity by one and records spatial occupancy, particle count, fired IDs, and source result.
- Confirm source signatures across rename, duplicate, reset, and animated geometry.

Checkpoint: current failure is reproduced behaviorally, not inferred only from code/artifacts.

### WP1 — Pure planner and formula tests

**Files:** new `sim/capacity_planner.py`, new `tests/test_capacity_planner.py`.

- Immutable reports; exact/bound/unknown aggregation.
- Particle/leaf blockers and suggestion search.
- No mutation of settings/resources.
- Domain X/Y/Z candidates and disclaimers.
- Counter overflow and invalid/missing settings fail closed.

### WP2 — Source footprints and leaf math

**Files:** `sim/source_regions.py`, `tests/test_source_region_compilation.py`, new focused leaf-footprint tests if needed.

- No-upload footprints and deterministic order.
- Direct points, analytic, mesh, height, volume, fractional coverage.
- 64-particle clamp and predicate metadata.
- Unioned leaves, clipping, 8³ boundaries, expansion.
- Compiler error/uncertainty propagation.

WP1 and WP2 can run in parallel after WP0.

### WP3 — Transactional emission runtime

**Files:** `sim/sourcing.py`, `sim/solver.py`, `tests/test_source_emission_lifecycle.py`, focused sourcing tests.

- Count/admit/commit for compiled and legacy paths.
- Whole-batch exact fit, over-by-one, multi-source, overlap, height, mesh, direct append.
- Zero particle/fluid-stamp/particle-fired-state/RNG commit on rejection; provisional topology allocation is permitted only in the stopped/discarded session and is never published.
- Continuous inflow runtime rejection.
- One aggregate synchronization budget per substep; no per-source readback.

Depends on WP1/WP2 contracts. This is the safety critical path and gets one coding owner.

### WP4 — Scene snapshot, auto-seeds, and session ordering

**Files:** `blender/scene_inputs.py`, `blender/sources.py`, `blender/scene_bridge.py`, new `tests/test_scene_bridge_capacity_initialization.py`, relevant pending-interaction tests.

- Dry-run seed payloads and shared scene snapshot.
- Preflight before destructive reset/reinitialize.
- Stale signature handling before first step.
- Existing solver/pending state retained on preflight rejection.
- Runtime rejection stops playback before output/cache publication.
- Multi-scene isolation and cache-playback bypass.

Depends on WP3 admission API.

### WP5 — Presentation and explicit actions

**Files:** `panels.py`, `operators.py`, `properties.py` only if persisted presentation is required; panel/operator tests.

- Draw-safe report; no geometry inspection or mutation in `draw()`.
- Ranked recommended option plus alternatives; capacity-only changes preserve simulation semantics and rank first when they fit the conservative budget.
- No source-integrity “Run Anyway.”
- Explicit selected-suggestion approval dialog shows exact changes and the estimate disclaimer, rechecks a fresh scene signature, then applies and reinitializes.

Can run in parallel with WP6 after WP4 freezes report fields.

### WP6 — Real Blender/CUDA acceptance

**Files:** new `tests/blender_capacity_planner_acceptance.py`, source acceptance helper only under one owner, production gate registration after stability.

Behavioral cases:

- source demand exceeds capacity by one: zero initial emission;
- auto seed + source each fits alone but not together: zero source transaction and visible blocker;
- exact fit succeeds with spatial coverage in every expected leaf;
- increasing capacity explicitly then retrying matches a fresh run;
- no capacity, voxel-size, or domain suggestion is applied before explicit approval;
- leaf capacity failure is distinct from particle capacity;
- runtime continuous inflow reaches insufficient headroom and stops without partial new coverage.

### WP7 — Independent review and full validation

A fresh reviewer examines source and runtime behavior, not labels or acceptance markers. It checks no archived experiment was used and audits performance, pending-state, lifecycle, and memory claims.

## 6. Parallel coding orchestration

The active checkout contains unrelated modified demo files. Do not let coding agents write concurrently in this working directory and do not touch those files.

Use new disposable worktrees outside the active checkout, based only on the integration branch `master` at a kickoff-recorded commit (currently `ca89f2d`). Do not use the separate `phase4/live-interaction-merge-20260728` checkout as a base. Never use or mine the four archived experiment checkouts.

Parallel wave 1:

- **Implementer A:** WP1 planner module/tests.
- **Implementer B:** WP2 footprint/leaf API/tests.
- **Reviewer C:** adversarial API and test review, read-only.

Lead integrates A/B only after focused tests and reviewer feedback.

Parallel wave 2:

- **Implementer D:** WP3 runtime transaction (sole owner of `sourcing.py`/`solver.py`).
- **Test specialist E:** prepares additional black-box acceptance cases in an isolated test-only worktree; lead reconciles tests to avoid overlapping file writers.
- **Reviewer F:** CUDA synchronization and kernel-predicate parity review, read-only.

Parallel wave 3 after WP3:

- **Implementer G:** WP4 Blender/session integration.
- **Implementer H:** prepares WP5 UI against frozen report interfaces, in an isolated worktree.
- **Validation agent I:** runs focused non-writing checks on integrated commits.

Final wave:

- **Implementer J:** WP6 acceptance/gate registration.
- **Independent reviewer K:** truth/standards review.
- **Lead:** applies fixes, runs full gates, and writes final evidence.

Keep one writer per worktree and one integrator in the active checkout. Subagents return commits/patches plus commands run; the lead reviews and integrates sequentially.

## 7. Edge cases and regression traps

- Capacity exactly equal to demand versus over by one.
- Multiple sources whose aggregate exceeds capacity although each fits alone.
- Overlapping emitters and stable ordering.
- Fractional coverage and target rounding divergence between planner and kernel.
- Targets above the kernel limit of 64.
- Direct points outside the domain.
- Height source grid stamp without particle commit.
- One-shot source marked fired after rejection.
- RNG advanced after rejection, breaking retry reproducibility.
- Source rename/duplicate/stable-ID collision.
- Animated mesh/volume changes after initialization.
- Settings edits between preflight and first step.
- `particles_per_voxel` live edits that increase runtime demand.
- Voxel-size/max-particle reinitialize failure preserving pending state.
- Source compiler host-memory/staging failure.
- Sparse-leaf overflow independent of particle capacity.
- Domain padding and coordinates exactly on leaf/domain boundaries.
- Surface-resolution memory changes that must not alter particle demand.
- Cache playback and rewind must not reseed or plan live emission.
- CPU and CUDA behavior parity.
- Device count overflow and plain non-atomic diagnostic flags.
- No `nvidia-smi` or per-source `.numpy()` in the hot path; admission must reuse the existing one aggregate end-of-source-batch result readback and add no new synchronization point.
- Existing memory recovery must distinguish an integrity rejection from OOM. Its explicit Apply Safe Settings action remains available for memory estimates, but no memory bypass or recommendation may override a source-integrity blocker.

## 8. Validation sequence

Fast contract loop:

```bash
python -m pytest -q \
  tests/test_capacity_planner.py \
  tests/test_source_region_compilation.py \
  tests/test_source_emission_lifecycle.py \
  tests/test_scene_bridge_capacity_initialization.py \
  tests/test_memory_safety_recovery.py \
  tests/test_phase4_pending_interaction.py
```

Required cache/live-playback, sparse-grid, gate-truthfulness, and active-viscosity lane:

```bash
python -m pytest -q \
  tests/test_frame_output.py \
  tests/test_scene_bridge_cache_playback.py \
  tests/test_snippet_library.py \
  tests/test_production_gate_truthfulness.py \
  tests/test_sparse_leaf_grid_contract.py \
  tests/test_viscosity_active_work_regressions.py
```

Required source/role lane:

```bash
python -m pytest -q \
  tests/test_force_subsystem_contracts.py \
  tests/test_source_field_force_contract.py \
  tests/test_source_emission_lifecycle.py
blender --background --factory-startup --python tests/blender_force_region_acceptance.py
```

Capacity acceptance:

```bash
blender --background --factory-startup --python tests/blender_capacity_planner_acceptance.py \
  -- --json-out artifacts/capacity_planner_acceptance.json
```

Required source UI/schema lane:

```bash
blender --background --factory-startup --python tests/blender_phase3_source_ui_acceptance.py \
  -- --json-out artifacts/phase3_source_ui_acceptance.json
```

Expect `PHASE3-SOURCE-UI-ACCEPTANCE-PASS`. Run all five Phase 2 source capability scripts (lifecycle, region, animated, point, and volume) with their JSON outputs and validate each with `tools/validate_source_acceptance_report.py`, as specified in `AGENTS.md`.

Deep-module regression lane:

```bash
python -m pytest -q \
  tests/test_sourcing_pipeline.py \
  tests/test_mesh_source_pipeline.py \
  tests/test_pressure_projection_regressions.py \
  tests/test_memory_safety_recovery.py \
  tests/test_performance_hotpath_contracts.py \
  tests/test_snippet_library.py
```

Then:

```bash
python -m pytest -q
python tools/run_production_gate.py
```

Use `--include-gui` and policy-drift validation only if the UI/playback acceptance scope is exercised and reports are intentionally refreshed.

## 9. Definition of done

- The original Blender reproduction no longer produces missing spatial blocks.
- Over-capacity frame-one and later source batches commit zero new particles and publish no partial frame.
- Rejected one-shot sources remain armed; retry matches a fresh run.
- Planner/kernel demand parity is proven across representative compiled and legacy emitters.
- Particle, leaf, and estimated-memory constraints are independently reported.
- Semantic suggestions never mutate authored settings without an explicit operator action; runtime particle growth changes only the live allocation and preserves the authored initial reserve.
- VRAM wording never promises OOM immunity.
- Focused, full Python, Blender acceptance, and production gates pass.
- Debug instrumentation and temporary worktrees are removed.
- Unrelated modified demo files remain untouched.

## 10. Execution result (2026-07-30)

Implemented WP1-WP6 and completed WP7 validation.

- Added a pure immutable capacity planner with exact/upper-bound/unknown demand quality, particle/leaf/memory blockers, transaction scratch accounting, and ranked capacity/resolution/domain/balanced suggestions.
- Added host-only compiled source footprints and isolated upload/no-upload compiler cache generations.
- Replaced silent fixed-capacity truncation with GPU count/admit/guarded-commit transactions across compiled, legacy analytic/mesh/height, and direct particle paths.
- Added dry-run automatic seed batches, aggregate seed admission, initialization/reset preflight, old-session preservation on preflight rejection, and rejection before viewport/cache publication.
- Added draw-safe planner reporting and transactional particle-pool growth. The live VRAM guard charges the full replacement pool while old and new buffers coexist; allocation/copy failure leaves the old pool authoritative. Growth-failure behavior is an advanced Continue/Defer versus Pause preference, and pending actions use one neutral fixed summary rather than repeated red rows.
- Added `tests/blender_capacity_planner_acceptance.py` and registered it in the production gate.

Evidence:

- `python -m pytest -q`: **473 passed** after transactional auto-growth, growth-failure policy, and neutral pending-UI coverage.
- Five Phase 2 Blender source capabilities passed and their reports validated.
- `PHASE3-SOURCE-UI-ACCEPTANCE-PASS`.
- `CAPACITY-PLANNER-ACCEPTANCE-PASS` on CUDA: exact fit committed, aggregate over-by-one committed zero, rejected height fill preserved lifecycle/RNG state, and the add-on changed nothing before approval then applied the minimum capacity-only recommendation and initialized successfully.
- Default production gate: **`HEADLESS-GATE-PASS`** in a disposable worktree. Isolation was required because the gate intentionally rebuilds demos and the active checkout contained unrelated user demo edits.
- `git diff --check`: passed.

No OOM guarantee is claimed. The pre-existing modified demo files retained their original byte sizes and the temporary validation worktree was removed.
