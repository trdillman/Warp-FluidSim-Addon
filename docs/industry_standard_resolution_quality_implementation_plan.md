# Industry-Standard Resolution and Quality Model — Review and Implementation Plan

**Status:** Proposed; no implementation authorized by this document  
**Scope:** User-facing liquid resolution/quality controls, their runtime meanings, and the capacity/safety modules they cross  
**Explicit non-goal:** This plan does **not** implement the previously proposed lifetime peak-particle predictor or a source/sink-derived predictive autogrowth ceiling.

## 1. Truth contract

The user should normally make artistic/physical decisions, not allocator decisions.

The basic interface must expose:

1. **Simulation Resolution** — one spatial-fidelity control, with resolved grid dimensions shown read-only.
2. **Liquid Sampling** — one authoritative particle sampling target for ordinary liquid.
3. **Temporal Quality** — adaptive CFL and maximum substeps, with effective substeps and clamping visible.
4. **Surface Quality** — surface up-resolution, particle influence, smoothing, and update cadence.
5. **Whitewater Quality** — honest controls for the behavior actually implemented.

The implementation must derive or manage liquid-particle, whitewater, sparse-leaf, and surface-grid capacities separately. Advanced users may inspect or override those limits, but capacities must not masquerade as fidelity controls.

No capacity recovery may silently change voxel size, particle sampling, domain bounds, source behavior, or other authored semantics. No recoverable capacity refusal may partially emit, publish a partial frame, destroy the current solver, or discard its last valid state.

## 2. Evidence used

This plan synthesizes three independent evidence paths:

- Primary-source industry research: `.pi-subagents/artifacts/outputs/0d7f5483-d831-45f8-9e32-85bd097228e6/artifacts/industry_resolution_controls_research.md`
- Independent adversarial implementation review: `.pi-subagents/artifacts/outputs/0d7f5483-d831-45f8-9e32-85bd097228e6/artifacts/resolution_controls_independent_review.md`
- Parent review of the current UI artifacts and connected implementation, including `properties.py`, `panels.py`, `scene_bridge.py`, `scene_inputs.py`, `fields.py`, `solver.py`, `surface.py`, `whitewater.py`, `capacity_planner.py`, `particle_capacity.py`, `particles.py`, `sourcing.py`, `memory_safety.py`, and the current Blender screenshots.

The industry sources retained by the researcher were official SideFX FLIP/whitewater/surfacing documentation and official Blender fluid-domain/liquid-mesh/liquid-particle/cache documentation.

## 3. Industry model and current verdict

### 3.1 Industry-familiar model

The common model is:

- one primary spatial-resolution control (cell/voxel size, resolution divisions, or particle separation);
- particle sampling derived from or clearly subordinate to spatial resolution;
- CFL/substeps treated as temporal quality rather than spatial resolution;
- surface meshing treated as a separate output-resolution stage;
- whitewater treated as a separate secondary simulation;
- sparse storage and pool capacities treated as performance/safety limits, usually hidden or advanced.

### 3.2 Current implementation

The underlying solver concepts are largely standard, but the user-facing interface is a custom engineer-facing mixture:

- `Voxel Size` is a real spatial-fidelity control.
- Scene `Particles / Voxel` affects pressure rest count, relaxation spacing, ST-FLIP calibration, and height fill.
- Per-source `Target Particles / Voxel` independently controls source occupancy.
- `Leaf Capacity`, `Initial Particle Reserve`, `Max Whitewater`, `Surface Leaf Capacity Scale`, and `Max Dense Dim` are allocator limits.
- Surface output has a correct independent-resolution seam, but exposes many low-level implementation controls in the basic workflow.
- The sidebar contains 22 panel classes (19 top-level panels plus child panels/popover), while the user guide still says 13 panels.

Therefore the system has standard internals but does **not** yet present an industry-standard basic control model.

## 4. Confirmed correctness defects to fix before UI simplification

These are not optional polish. Hiding controls before correcting their runtime contracts would make the interface simpler but less truthful.

### P0.1 Mixed legacy and normalized source plans can be ignored

`ScenePlanningSnapshot` can contain legacy descriptors and `extra_plans`. `_apply_scene_snapshot()` first converts descriptors into `sourcing.plans`, then appends `extra_plans`. However:

- `SourcePipeline.apply()` compiles/applies plans only when `not self.sources` (`sim/sourcing.py`, `apply()`), so native point/volume plans are skipped whenever any legacy source exists.
- `FluidSolver._rebuild_topology()` likewise marks compiled-plan leaves only when `not self.sources` (`sim/solver.py`, `_rebuild_topology()`).

**Required correction:** replace the ambiguous `sources`/`plans` state with one deep source-batch interface that explicitly owns both legacy descriptors and normalized plans. Count, topology marking, and commit must union both representations without double-emitting descriptor-converted plans.

**Acceptance:** one scene containing an analytic legacy emitter plus point and volume-grid emitters must mark all required leaves and emit all admitted source types in the same substep.

### P0.2 Direct point injection still has a partial-append escape path

`SourcePipeline.inject_particles()` admits from the host mirror without first refreshing the device count. `k_source_particles` still truncates internally if the count is stale, while `inject_particles()` reports full success. Some callers have a post-check; the public solver interface does not.

**Required correction:** bring direct injection under the same count/admit/guarded-commit contract, or refresh and prove exact capacity immediately before a commit guarded by device state. The public interface must return an exact admitted/emitted result.

### P0.3 Zero source occupancy is not zero

The source property says zero emits no particles, but count and commit clamp the target to at least one. Planner fallback logic treats zero differently.

**Required correction:** choose and enforce one contract. Recommended: zero disables particle emission while allowing any explicitly authored non-particle field operation. Apply the same predicate in property bounds, planner, count kernel, and commit kernel.

### P0.4 Reset can plan against fictional capacity and then release the solver

When auto-growth is disabled, `Session.reset()` plans against authored `max_particles` rather than the active `solver.particles.capacity`. A raised authored reserve can therefore pass preflight despite not existing at runtime; prepared-seed admission can fail after `solver.reset()`, and the exception path releases the solver. A previously grown pool can also be spuriously rejected when authored reserve is lower.

**Required correction:** for every active session, admission availability is always the actual runtime pool capacity, independent of growth policy. Reconcile capacity before any reset mutation. A refused reset must leave the previous solver and frame authoritative.

**Important rejection of one review suggestion:** do not simply grow after `solver.reset()` to avoid a copy. That would destroy the valid state before allocation success is known. Add a reset-specific prepare path that allocates an empty replacement pool and swaps only after all allocations succeed, or retain the safe copy-before-reset behavior until that prepare path exists.

### P0.5 Runtime sparse-topology overflow currently destroys the solver

`SparseGrid.begin_topology()` clears live topology before `finalize_topology()` discovers overflow. The frame-handler exception path turns Live off and releases the solver.

**Required correction:** make topology rebuild transactional. Build/count candidate flags while the current leaf table remains valid; admit against capacity; commit a new table only after success. On refusal, preserve the last valid solver state and publish no partial frame. Do not route a normal leaf-capacity refusal through generic fatal-error teardown.

### P0.6 Leaf planning is labelled exact when it is not runtime-equivalent

The planner unconditionally labels leaf demand `EXACT`. Runtime topology marks legacy source AABBs and adds a `4 * voxel_size` band above height fill; planning uses a mixture of tighter coverage leaves and AABB upper bounds and omits that height band.

**Required correction:** share one topology-demand module between planning and runtime. Propagate `EXACT`, `UPPER_BOUND`, or `UNKNOWN` from every contributor. The displayed quality must be the weakest contributing quality.

### P0.7 Whitewater can consume an unbuilt surface SDF

`SurfacePipeline.build()` returns immediately when `surface_enabled` is false, including the low-resolution SDF path requested by whitewater. Whitewater emission still samples `surface_sdf`, whose zero-initialized values can incorrectly classify broad regions as surface.

**Required correction:** separate **solver liquid SDF** from **render mesh generation**. Whitewater must force a current simulation-resolution SDF even when the visible/render surface is disabled. If that cannot be built, whitewater must fail closed with a dependency message.

### P0.8 Whitewater capacity silently truncates emission

Whitewater kernels use atomic append and return when `Max Whitewater` is reached, with no truthful rejected/dropped count or guarded growth policy.

**Required correction:** at minimum publish required/admitted/dropped counts and an explicit exhaustion policy. Preferred: give whitewater its own transactional capacity module and whole-batch or explicitly probabilistic admission contract. Do not imply liquid auto-growth protects whitewater.

### P0.9 Stale capacity reports cross release and scene ownership

`Session._clear_runtime()` does not clear `capacity_plan`; panel/operator polling can continue presenting old recommendations. `refresh_capacity_plan()` can overwrite the active session's plan from another scene.

**Required correction:** bind every assessment to scene identity plus authored/demand signature, clear it on release/load, and reject display/use when ownership or signature is stale. Execution must still re-sample VRAM and revalidate.

## 5. Target module seams

### 5.1 `sim/quality_contract.py` — authoritative quality resolver

Create a pure, immutable module with a small interface:

```python
resolve_quality(authored_quality, domain_bounds) -> ResolvedQuality
resolve_source_sampling(resolved_quality, source_sampling) -> EffectiveSourceSampling
```

`ResolvedQuality` owns:

- physical voxel size;
- unpadded and 8-aligned voxel dimensions;
- leaf dimensions and total domain leaf cells;
- authoritative liquid particles-per-cell;
- derived nominal particle spacing/rest-count inputs;
- resolved surface voxel size/upres;
- validation diagnostics.

Callers must not reimplement cell/leaf rounding. The domain panel, `SparseGrid`, capacity planner, source compiler, pressure setup, relaxation, surface resource sizing, cache signature, and tests all consume this interface.

This is a deep module: one small interface hides the rounding, bounds, density, migration, and validation rules currently repeated across UI, planner, memory estimator, fields, surface, and solver code.

### 5.2 `sim/source_batch.py` or a deepened `SourcePipeline`

Expose one interface for a frame's source work:

```python
prepare(scene_inputs, quality, frame) -> PreparedSourceBatch
assess(prepared, runtime_capacity, topology_capacity) -> SourceAdmission
commit(prepared, admission, solver_state) -> SourceCommitResult
```

The prepared batch must include legacy analytic/mesh sources, normalized point/volume/field plans, height fill, and direct prepared seeds without ambiguous parallel lists. It owns topology footprints and particle demand so planning and runtime cannot diverge.

### 5.3 `sim/capacity_service.py` — one capacity/safety authority

Consolidate the overlapping roles of `capacity_planner.py`, `particle_capacity.py`, and `memory_safety.recommend_safe_settings()` behind:

```python
assess(resolved_quality, prepared_scene, resources, runtime_state=None) -> CapacityAssessment
recommend(assessment) -> tuple[CapacityOption, ...]
prepare_growth(runtime_state, required, resources) -> PreparedGrowth
```

Keep low-level byte-estimation helpers internal. There must be one assessment, one ranked option list, and one confirmation flow. Remove runtime GPU/CPU utilization heuristics from fidelity recommendations; transient utilization is not a valid reason to alter simulation semantics.

### 5.4 Separate authored, resolved, and runtime state

Do not continue using one `SolverSettings.max_particles` value to mean all of:

- the Blender-authored initial reserve;
- the actual runtime allocation;
- the planner's candidate capacity;
- the growth hard limit.

Introduce explicit records:

- `AuthoredQualitySettings`
- `ResolvedQuality`
- `AuthoredCapacityPolicy`
- `RuntimeCapacityState`

Migration can be incremental, but new interfaces and UI must use the correct record rather than infer meaning from `SolverSettings`.

## 6. User-facing model

### 6.1 Basic Quality panel

Expose only:

- **Simulation Resolution** — retain Voxel Size as the canonical stored value; optionally offer mutually exclusive presets or a Resolution Divisions input mode, never two simultaneously active spatial levers.
- read-only **Cell Size**, **Grid X × Y × Z**, and padded/sparse status;
- **Liquid Sampling** — one global target or Low/Standard/High/Custom preset;
- **Transfer** — PIC/FLIP and FLIP Ratio;
- **Temporal Quality** — Adaptive CFL and Max Substeps; Min Substeps under Advanced;
- a compact effective-substeps/CFL-clamped status.

`Particles / Voxel` must no longer be live. It changes pressure rest count, relaxation spacing, ST-FLIP calibration, height fill, and source population assumptions; changing it requires at least Reset so the particle population and solver reference state are rebuilt consistently.

### 6.2 Source sampling

New sources default to **Inherit Simulation Sampling**.

Expose an advanced **Source Fill Target Override** only when needed. Do not call it a second resolution control. Its UI must state that it changes source occupancy/volume correction behavior, not solver spatial resolution.

Recommended schema:

- `sampling_mode = INHERIT | MULTIPLIER | LEGACY_ABSOLUTE`
- `sampling_multiplier` for new explicit overrides;
- existing `emit_rate` retained for migration and old-file behavior.

For old scenes, preserve authored output: schema-migrate existing values to `LEGACY_ABSOLUTE`; do not guess that an old value equal to 8 was intended to inherit. New scenes use `INHERIT`.

All effective values are clamped/validated once in `quality_contract.py`; UI and kernels share the same 0..64 contract.

### 6.3 Capacity & Safety (Advanced)

Default presentation is read-only:

- live liquid particles / allocated capacity;
- active leaves / allocated leaves;
- whitewater live / allocated capacity;
- estimated current allocation / live safety budget;
- last growth or refusal reason.

Hide these authored overrides behind **Manual Capacity Overrides**:

- Initial Particle Reserve;
- Leaf Capacity;
- Whitewater Capacity;
- surface leaf scale/dense extraction cap;
- topology expansion;
- growth failure policy and hard limit.

Keep serialized properties for compatibility. Do not delete advanced control.

When an active solver exists and the initial reserve changes, either:

1. treat an increase as an explicit guarded live-growth request and preserve the solver on refusal; or
2. persistently mark **Applies on next solver start** with one Restart action.

Do not leave only a transient status message. A decrease never shrinks the active pool automatically.

### 6.4 Surface Quality

Basic controls:

- Generate Render Surface;
- Surface Upres: Simulation / 2× / Custom;
- Particle Radius;
- Smoothing / Preserve Volume;
- Viewport Update Cadence.

Move splat radius, leaf scale, dense max, raw filter/dilate/erode passes, and low-resolution SDF retention to Advanced. Rename visible-surface generation separately from the solver SDF so disabling the mesh cannot disable whitewater's required field.

### 6.5 Whitewater

Basic controls should describe only implemented behavior: Enable, emission/churn threshold, amount, speed gate, lifespan, and drag. Enforce `life_max >= life_min`.

Do not expose Foam/Spray/Bubble toggles until species identity is persistently represented and separately controllable. The current runtime classification is transient; label aggregate whitewater honestly.

### 6.6 Panel information architecture

Reduce the default top-level sidebar to approximately seven groups:

1. **Simulation** — initialize/live/reset/release/status/pending action.
2. **Quality** — spatial, liquid sampling, temporal.
3. **Inputs** — child panels for emitters, sinks, colliders, forces, height fill, point/volume inputs.
4. **Physics** — pressure, viscosity, transfer/advanced solver modes.
5. **Surface & Whitewater**.
6. **Output & Cache**.
7. **Advanced** — capacity, field sources, snippets, profiling, validation/benchmarks.

Keep one neutral pending summary in Simulation and the header popover. Add small local text such as **Requires Reset** beside restart-tier controls, but do not duplicate action rows across panels.

All warnings must wrap at normal N-panel width. Present the recommended action first and collapse alternatives/details.

## 7. Planner and performance corrections

### 7.1 One host-only planning snapshot

Current planning can evaluate/upload meshes and compile more than once during initialize/reset. Refactor to:

1. gather host geometry and source metadata once;
2. build one `PreparedScene` and demand model;
3. sample live VRAM after host preparation and before device allocation;
4. evaluate capacity candidates from the same prepared data;
5. upload only after admission;
6. release compiler/prepared resources in `try/finally`.

Changing only particle capacity must not re-evaluate modifier stacks or recompile source geometry.

### 7.2 Candidate recommendations must be recalculated honestly

A coarser-voxel suggestion currently retains particle demand measured at the finer grid. This is conservative but can incorrectly hide a viable option. Capacity-only options can be evaluated directly; semantic options must be re-evaluated through the host demand model at the candidate resolution before presentation and again before application.

### 7.3 Progress and cancellation

Expensive source compilation/planning needs visible progress before solver initialization. Use Blender progress reporting or a modal/timer preparation stage where safe. Cancellation before allocation leaves the active solver untouched.

## 8. Small but user-visible correctness fixes

- Domain resolution display currently uses truncating `int(size / voxel_size)` and ignores 8-cell padding. Render dimensions from `ResolvedQuality`/`SparseGrid` using the same ceiling and alignment rules.
- Show runtime leaf capacity separately from authored override; `SparseGrid` clamps allocation to total domain leaves.
- Update the guide's stale “13 panels” statement and startup instructions after the panel reorganization.
- Bound per-source occupancy in RNA to the backend-supported maximum, while preserving zero-as-disabled semantics.
- Clear capacity reports on Release/load/scene switch.
- Replace the old direct `Apply Safe Settings & Retry` path with the single review/confirm recommendation flow.
- Replace source-text hot-path tests with behavioral synchronization/launch-count tests.

## 9. Implementation sequence

### Phase 0 — Truth and state preservation

1. Mixed source-batch execution/topology parity.
2. Transactional direct injection.
3. Zero occupancy contract.
4. Reset planning against actual capacity with no post-reset release.
5. Transactional topology overflow handling.
6. Planner/runtime leaf-demand parity and honest quality.
7. Whitewater SDF invariant and visible capacity exhaustion.
8. Stale plan ownership/clear rules.

**Exit criterion:** all normal capacity refusals preserve the prior valid solver; all enabled source representations work together; no UI number claims stronger demand quality than the implementation proves.

### Phase 1 — Authoritative quality contract

1. Add pure resolver and immutable records.
2. Route `SparseGrid`, panel dimensions, pressure rest count, relaxation spacing, ST-FLIP calibration, source sampling, surface sizing, planner, and cache signatures through it.
3. Make liquid sampling Reset-tier.
4. Add source inherit/multiplier schema and migration.

### Phase 2 — Consolidated capacity service

1. Merge assessment/recommendation authority.
2. Separate authored reserve, runtime capacity, and hard limit.
3. Make planning host-only/reusable and add progress.
4. Add transactional empty-pool prepare for Reset.
5. Add whitewater capacity policy; design leaf growth only after transactional topology is proven.

### Phase 3 — Basic/Advanced UI reorganization

1. Add Quality and Capacity & Safety panels.
2. Reparent existing panels under the seven-group model.
3. Hide allocator/implementation knobs by default without deleting them.
4. Add compact read-only quality/capacity summary and wrapped warnings.
5. Remove the duplicate recommender and stale labels/docs.

### Phase 4 — Surface/whitewater UX

1. Separate solver SDF from render mesh enable.
2. Expose only supported whitewater semantics.
3. Add lifespan validation, occupancy/capacity reporting, and optional guarded growth.
4. Keep species-level UX deferred until storage and runtime behavior are real.

### Phase 5 — Live Blender review and stabilization

Exercise new scene, existing migrated scene, mixed source types, initial fill, continuous inflow, sinks, whitewater with render mesh off, capacity refusal, reset, reinitialize, cache playback, and scene switching in a fresh profile-safe Blender process. Review at normal and narrow sidebar widths.

## 10. Validation contract

### Pure/unit tests

- quality resolution uses exactly the same ceil/padding rules as `SparseGrid`;
- monotonic cell-size, particle-density, leaf, and memory relationships;
- source inherit/multiplier/legacy migration;
- zero occupancy emits zero in planner, count, and commit;
- mixed legacy/native source batch topology and emission;
- actual runtime capacity always wins for active reset admission;
- stale scene/release plans cannot draw or execute;
- leaf demand quality propagation and height/legacy topology parity;
- whitewater requires a valid solver SDF;
- whitewater capacity refusal/drop reporting;
- no-swap/no-release behavior for particle, leaf, and whitewater capacity refusal;
- behavioral synchronization-count tests for hot paths.

### Blender/CUDA acceptance

- basic Quality panel contains no raw leaf/pool/dense-grid controls;
- advanced controls remain available and old files retain behavior;
- changing Liquid Sampling produces one clear Reset pending state;
- changing Voxel Size produces one clear Reinitialize pending state;
- mixed analytic + point + volume sources all emit in one scene;
- whitewater remains correct with render mesh hidden;
- direct injection either commits the whole batch or commits zero;
- runtime topology refusal preserves the initialized solver and last valid output;
- Release and scene switching clear capacity reports;
- one confirmation shows exact old/new values and tradeoffs.

### Required gates

- focused pure regressions during each phase;
- full `python -m pytest -q`;
- relevant Blender source, Phase 3 UI, Phase 4 interaction, whitewater, surface, capacity, cache, and persistence acceptance scripts;
- profile-safe normal-profile GUI review;
- `python tools/run_production_gate.py` in an isolated clean worktree;
- `git diff --check`.

## 11. Decisions intentionally deferred

- predictive lifetime peak-particle modelling from sinks and future source motion;
- a universal numeric “quality” preset mapping without measured scene/hardware evidence;
- persistent Foam/Spray/Bubble species controls;
- automatic sparse-leaf growth implementation details beyond transactional topology preparation;
- replacing Voxel Size with Resolution Divisions. The recommended first step is to retain Voxel Size and improve naming/readouts, avoiding another conflicting lever;
- adding a Houdini/Blender-style narrow-band particle mode. The current solver retains full liquid particles; do not expose a narrow-band control until the storage and solver behavior are genuinely implemented.

## 12. Recommended first delivery slice

Do **not** begin with panel rearrangement. First deliver Phase 0.1–0.7 as a correctness branch, with mixed-source, reset-preservation, direct-injection, topology-overflow, and whitewater-SDF CUDA acceptance. Only after those contracts pass should the quality resolver and UI hide the raw allocator controls.
