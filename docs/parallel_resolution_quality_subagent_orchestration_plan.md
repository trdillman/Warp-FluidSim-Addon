# Parallel Subagent Orchestration Plan — Resolution, Quality, and Capacity

**Status:** Executed. The three agent lanes and the integration lane exist as
worktrees with distinct histories (`agent/res-quality-t01-source-truth`,
`agent/res-quality-t02-session-capacity`, `agent/res-quality-t03-whitewater-truth`,
`feature/resolution-quality-standardization`). None is merged to `master`;
`docs/superpowers/plans/2026-08-02-resolution-quality-merge.md` drives that merge.  
**Parent orchestrator:** Main Pi session  
**Canonical product/architecture plan:** `docs/industry_standard_resolution_quality_implementation_plan.md`  
**Rule:** Every major changeset is traced by a read-only subagent before implementation and reviewed afterward by a separate fresh-context reviewer.

## 1. Goal and completion contract

Use isolated implementation worktrees and parallel subagents where dependencies and file ownership permit, while preserving one writer per worktree and one serial integration path.

Completion requires:

- all Phase-0 truth/state-preservation defects corrected;
- one authoritative quality-resolution interface;
- one authoritative capacity/safety interface;
- industry-familiar Basic controls with technical capacity controls retained under Advanced;
- old-file behavior migrated rather than guessed or silently changed;
- no recoverable capacity refusal partially emits, publishes a partial frame, releases the current solver, or loses the last valid state;
- every task has trace evidence, focused tests, independent review, and reviewed integration;
- full Python, Blender/CUDA, profile-safe GUI, production-gate, and policy-drift validation passes.

The lifetime predictive-cap proposal, narrow-band particles, and persistent Foam/Spray/Bubble species remain out of scope.

## 2. Non-negotiable orchestration rules

1. **No parallel writers in one worktree.** Each implementation lane receives a dedicated branch and worktree.
2. **No worker starts without its trace.** A trace must identify actual call flows, invariants, likely files, test seams, and collision risks.
3. **Reviewer independence.** Reviewers use fresh context, receive the task contract and base SHA, and inspect the actual diff. They do not receive the worker's explanation as evidence.
4. **Reviewers never edit.** A separate fix worker applies accepted findings. The reviewer re-reviews material fixes.
5. **Serial integration.** One integration worker or the parent cherry-picks reviewed commits into the integration branch. Parallel lanes never merge themselves.
6. **No merge on unresolved correctness findings.** High/medium correctness findings block integration. Optional polish does not.
7. **No archived experiments.** The four archived experimental checkouts named in `AGENTS.md` are `EXPERIMENT — NOT A MERGE CANDIDATE` and may not be mined, merged, or used as implementation sources.
8. **Do not touch unrelated existing changes.** Preserve the excluded demo files and unrelated worktree-prune backlog task listed in `AGENTS.md` and the prior implementation context.
9. **Implementation truth outranks UI labels/tests.** Each reviewer traces the runtime behavior and verifies behavioral tests, not symbol/string presence.
10. **No source mutation during trace/review.** Configured output artifacts are allowed.
11. **No commit to `master`.** All subagent commits land on task branches; integration occurs on a dedicated integration branch only after the baseline handshake.
12. **GPU/profile isolation.** Blender GUI work uses the profile-safe temporary add-on workflow and strict owned-process cleanup.

## 3. Required baseline handshake

The current root worktree is dirty and contains both relevant work and unrelated changes. Native parallel worktrees must not branch from bare `HEAD`, because that would omit today's uncommitted particle-capacity implementation.

Before executing any worker:

1. Parent inventories the exact relevant diff and untracked files.
2. Parent creates a clean integration worktree from `HEAD` outside the repository, for example:
   `C:\Windows\Temp\fluidsim-resolution-quality-integration`.
3. Parent applies/copies only the approved relevant changes into that worktree.
4. Parent excludes demo binaries/previews, the unrelated prune backlog task, and archived experiments.
5. Parent runs the full Python suite, focused Blender/CUDA capacity acceptance, and `git diff --check` in the integration worktree.
6. Parent presents the exact baseline file manifest and validation results to the user.
7. **Only after explicit approval**, parent creates a checkpoint commit on
   `feature/resolution-quality-standardization` and records `BASE_SHA`.
8. Every task branch/worktree starts from that `BASE_SHA` or from the reviewed integration SHA specified by its dependency wave.

This checkpoint is an integration-branch implementation detail; it does not commit or clean the user's current `master` worktree.

## 4. Agent roles and review loop

### Trace agent

- Agent: `scout` or `context-builder`
- Context: fresh
- Permissions: read-only
- Output: one bounded trace artifact
- Must identify exact symbols, flows, invariants, test gaps, dependencies, and overlap with other lanes.

### Implementation agent

- Agent: `worker`
- Context: fresh, with the trace artifact and task contract
- Worktree: dedicated task worktree
- Permissions: sole writer for that worktree
- Output: source changes, focused tests, one scoped commit, summary artifact
- Must stop and ask the parent rather than inventing unapproved semantics.

### Independent reviewer

- Agent: `reviewer`
- Context: fresh
- Worktree: task worktree after worker completion
- Permissions: read-only
- Input: task contract, base SHA, expected validation; not the worker narrative
- Output: PASS/FAIL plus severity-ranked evidence and smallest safe fixes.

### Fix worker

- Agent: `worker`
- Context: fresh or resumed implementation worker
- Worktree: same task worktree, after reviewer completion
- Input: accepted reviewer findings only
- Output: fix commit and focused validation.

### Review loop

Maximum two fix/re-review rounds per task. Stop and ask the user when:

- a finding requires a new product decision;
- the fix expands beyond task scope;
- two rounds do not obtain a clean correctness review;
- acceptance requires changing a previously approved simulation semantic.

## 5. Dependency graph and parallel waves

```text
T00 Baseline checkpoint
 ├─ Wave 1 (parallel): T01 Source truth, T02 Session/capacity state, T03 Whitewater truth
 │    └─ I01 serial integration + IR01 independent cross-lane review
 ├─ Wave 2:
 │    ├─ T04 Transactional topology + leaf-demand parity
 │    └─ D05 Quality-interface design fanout (read-only, parallel with T04)
 │         └─ I02 integration/review
 ├─ Wave 3: T05 Authoritative quality contract
 │         └─ I03 integration/review
 ├─ Wave 4 (conditional parallel):
 │    ├─ T06 Consolidated capacity service and host-only planning
 │    └─ T07 Sampling schema/migration and restart semantics
 │         └─ I04 serial integration + IR04 independent cross-lane review
 ├─ Wave 5: T08 Basic/Advanced UI and information architecture
 │         └─ I05 integration/review
 └─ Wave 6 (parallel validation):
      ├─ V01 pure/contract audit
      ├─ V02 Blender/CUDA acceptance audit
      └─ V03 profile-safe GUI UX audit
           └─ T09 one final fix worker, IR09 final independent review, production gate
```

### Conditional Wave-4 parallelism

T06 and T07 may run simultaneously only if both traces prove that their edits to `scene_bridge.py` and settings translation do not touch the same functions. If overlap is material, T06 runs first and T07 rebases onto reviewed I04a. Parallelism is subordinate to correctness.

## 6. Task list

| ID | Changeset | Depends on | Parallel lane | Primary ownership | Focused acceptance |
|---|---|---|---|---|---|
| T00 | Clean integration baseline | none | serial | parent | existing 474+ tests/gates reproduce |
| T01 | Unified source-batch truth, direct injection, zero occupancy | T00 | Wave 1A | sourcing/source inputs | mixed legacy+point+volume all emit; direct batch all-or-zero |
| T02 | Runtime capacity truth, reset preservation, plan freshness | T00 | Wave 1B | scene/session bridge | actual pool governs reset; no release on refusal; no stale/cross-scene plan |
| T03 | Whitewater SDF and capacity truth | T00 | Wave 1C | surface/whitewater | render mesh off still yields valid SDF; capacity exhaustion visible and deterministic |
| I01 | Integrate/review Wave 1 | T01–T03 | serial | integration | combined regressions and independent cross-lane review |
| T04 | Transactional sparse topology and honest leaf demand | I01 | Wave 2A | fields/topology/planner | overflow preserves solver/output; planner matches runtime contributors/quality |
| D05 | Design authoritative quality interface three ways | I01 | Wave 2B read-only | architecture | parent selects smallest deep interface before T05 |
| I02 | Integrate/review T04 | T04 | serial | integration | topology contract passes on combined Wave-1 state |
| T05 | Implement quality contract and shared dimension/surface rules | I02 + D05 | serial | quality/grid/surface | one resolver owns ceil/padding/spacing/surface resolution |
| I03 | Integrate/review T05 | T05 | serial | integration | no duplicate dimension/quality authorities remain on active paths |
| T06 | Consolidate capacity service; host-only reusable planning | I03 | Wave 4A | capacity/planning | one assessment/recommender; one snapshot; no preflight GPU upload leaks |
| T07 | Unify liquid/source sampling semantics and migration | I03 | Wave 4B conditional | RNA/source adapter | new sources inherit; old scenes preserve; sampling is Reset-tier |
| I04 | Integrate/review T06–T07 | T06 + T07 | serial | integration | capacity and sampling contracts agree |
| T08 | Reorganize UI into Basic/Advanced model | I04 | serial | panels/operators/docs | technical capacities hidden by default; one recommender/confirmation |
| I05 | Integrate/review T08 | T08 | serial | integration | UI remains connected to reviewed backend behavior |
| V01 | Pure contract/test audit | T08 | Wave 6A | tests only/read-only first | monotonicity, migration, no stale state, no partial commit |
| V02 | Blender/CUDA acceptance audit | T08 | Wave 6B | acceptance | mixed sources/topology/whitewater/capacity workflows |
| V03 | Profile-safe GUI UX audit | T08 | Wave 6C | live UI | normal-width clarity, no profile mutation, owned process cleanup |
| T09 | Final accepted fixes and documentation | V01–V03 | serial | integration | all review blockers resolved |
| IR09 | Final independent review and production gate | T09 | serial/read-only review | whole diff | no blockers; full gate passes |

## 7. Task prompts

The parent inserts the actual `BASE_SHA`, worktree path, trace artifact path, and focused commands before launch.

### T01 — Trace prompt

```text
Goal: Trace the complete source-emission ownership and execution flow needed to fix mixed legacy/normalized source handling, direct point injection, and zero-occupancy semantics.

Read the canonical plan at docs/industry_standard_resolution_quality_implementation_plan.md, especially P0.1–P0.3. Inspect the current integration baseline at {BASE_SHA}. Trace SceneInputs gathering, ScenePlanningSnapshot application, FluidSolver source setters/topology, SourcePipeline plans/sources compilation, count/admit/commit, direct prepared seeds, public inject_particles callers, lifecycle/RNG state, and relevant planner footprints/tests.

Required output:
- exact call graph and state ownership;
- proof of how mixed legacy + point/volume plans are currently skipped or, if the premise is wrong, counter-evidence;
- a recommended deep interface that avoids double-emitting converted legacy plans;
- exact predicates that must match for zero target and count/commit;
- every direct injection caller and its result expectations;
- likely files/symbols, overlap risks, and focused test commands;
- unresolved product decisions.

Do not modify project/source files. Do not inspect or reuse archived experimental worktrees. Return file/line evidence and a compact implementation handoff.
```

### T01 — Implementation prompt

```text
You are the sole writer in {T01_WORKTREE}, based on {BASE_SHA}. Implement the approved T01 source-truth changes using {T01_TRACE}.

Required behavior:
- one explicit source-batch ownership model supports legacy analytic/mesh descriptors and native point/volume/field plans in the same scene;
- topology preparation and emission include every representation exactly once;
- direct point injection is admitted and committed all-or-nothing from authoritative device count;
- target occupancy zero emits zero particles consistently in UI adapter, planner/count, and commit;
- rejected work preserves lifecycle/RNG/particle state and publishes no partial success;
- do not weaken GPU-resident hot paths with per-source host synchronization.

Add behavioral unit tests and focused Blender/CUDA acceptance for mixed source types and direct injection. Preserve old source semantics except the documented zero bug. Do not touch UI reorganization or unrelated files. Run the focused tests from the trace plus relevant source lifecycle/mesh/source acceptance. Commit the scoped changes on branch agent/res-quality-t01-source-truth and report changed files, commands, failures, commit SHA, and remaining risks. Stop for any unapproved semantic decision.
```

### T01 — Independent review prompt

```text
Independently review T01 in {T01_WORKTREE} against base {BASE_SHA}. Do not read the worker summary as evidence and do not modify files.

Contract: mixed legacy analytic/mesh and native point/volume/field plans must all mark topology and emit exactly once; direct injection must commit the whole batch or zero; zero occupancy must emit zero; rejection must preserve count, lifecycle, RNG, and published state; no per-source synchronization may enter the hot path.

Inspect the actual diff and runtime call paths. Run focused tests where practical. Look specifically for double conversion/emission, ignored plans, count/commit predicate divergence, stale host count, lifecycle rearming errors, topology omissions, and source-text-only tests. Return PASS or FAIL, severity-ranked findings with file/line evidence, commands run, and the smallest safe fixes.
```

### T02 — Trace prompt

```text
Goal: Trace active-session particle capacity, reset/reinitialize mutation ordering, Initial Particle Reserve semantics, capacity-plan lifecycle, scene ownership, and every exception path that can release a solver.

Read the canonical plan P0.4 and P0.9. Inspect Session.initialize/reset/release/_clear_runtime, build_settings, runtime particle growth, prepared-seed commit, pending signatures, capacity refresh/operator polling, frame handlers, load handlers, and related tests.

Required output:
- authored reserve versus actual pool versus growth hard-limit data flow;
- exact reset timeline and every mutation before admission is proven;
- proof of false-admit/spurious-block cases with auto-growth off;
- all paths that call release after a recoverable capacity refusal;
- stale-plan and cross-scene paths;
- recommended transaction seam for an empty reset replacement without destroying the live solver;
- focused regression matrix and likely file overlaps.

Do not modify project/source files or inspect archived experiments. Cite exact symbols/lines.
```

### T02 — Implementation prompt

```text
You are the sole writer in {T02_WORKTREE}, based on {BASE_SHA}. Implement T02 using {T02_TRACE}.

Required behavior:
- active-session planning always uses actual runtime particle capacity as available capacity, regardless of auto-growth policy;
- authored reserve, runtime capacity, and hard limit remain distinguishable;
- a refused reset/reinitialize leaves the previous solver, frame, particles, output, and statistics authoritative;
- no recoverable particle/plan-capacity path releases the solver;
- capacity plans are bound to scene identity/signature, cleared on release/load, and cannot be displayed/executed from another scene;
- changing Initial Particle Reserve has persistent, truthful next-action feedback; decreases never shrink the active pool automatically.

Do not reorganize the full UI or consolidate all capacity modules in this task. Add behavioral tests for auto-grow-off reset with both smaller and larger authored reserves, failed empty replacement preparation, release/load/scene switch, and stale operator polling. Run focused session/capacity/pending/cache tests. Commit on agent/res-quality-t02-session-capacity and report evidence.
```

### T02 — Independent review prompt

```text
Independently review T02 in {T02_WORKTREE} against {BASE_SHA}; do not edit.

Attempt to falsify these claims: actual pool capacity governs active reset; no refusal mutates then releases the solver; reserve edits have a persistent truthful application path; release/load/scene switching cannot leave a usable stale plan; memory-only bypass cannot bypass integrity.

Inspect mutation ordering and exception paths rather than trusting tests. Include auto-growth on/off, previously grown pools, prepared seeds, another-scene ownership, and failed allocation. Return PASS/FAIL with severity, file/line evidence, commands, and smallest fixes.
```

### T03 — Trace prompt

```text
Goal: Trace the whitewater dependency and capacity contracts from UI settings through SurfacePipeline, solver phase order, whitewater candidate/emission kernels, storage, statistics, and output.

Read canonical plan P0.7–P0.8. Determine exactly what surface_enabled and surface_build_lowres_sdf control, when surface_sdf is current, what whitewater samples, how random candidate predicates and free slots work, and what happens at capacity. Inventory life-range handling and all visible counters/status.

Required output:
- phase/call graph and proof of stale/zero SDF behavior with render surface disabled;
- a recommended seam separating solver SDF from render mesh;
- an honest whitewater capacity contract compatible with probabilistic emission;
- count/admit/commit or explicit drop-reporting design with RNG parity;
- storage/copy/growth byte inventory and tests;
- file/symbol overlap with T01/T02.

Read-only; no archived experiments; cite exact evidence.
```

### T03 — Implementation prompt

```text
You are the sole writer in {T03_WORKTREE}, based on {BASE_SHA}. Implement the traced T03 correctness scope.

Required behavior:
- whitewater always consumes a current simulation-resolution liquid SDF, independently of whether a render mesh is generated or visible;
- render-surface disable cannot silently turn the solver SDF into zeros/stale data;
- whitewater capacity exhaustion is deterministic and visible: no scheduling-dependent silent truncation presented as success;
- any guarded whitewater growth/candidate admission preserves the old pool until allocation succeeds;
- life ranges are normalized or rejected so max is never below min;
- no unsupported Foam/Spray/Bubble controls or species claims are added.

Keep broad UI reorganization for T08; expose only the minimal status/result seam needed later. Add pure and Blender/CUDA tests, including render mesh off + whitewater on and capacity refusal. Commit on agent/res-quality-t03-whitewater-truth and report evidence.
```

### T03 — Independent review prompt

```text
Independently review T03 in {T03_WORKTREE} against {BASE_SHA}; do not edit.

Verify source-level and runtime truth for: solver SDF freshness with render surface disabled, phase ordering before whitewater emission, capacity/RNG/count predicate parity, no silent drops, no pool swap on failed growth, valid life ranges, and honest species semantics. Look for new GPU synchronization or full-capacity scans. Run focused tests where possible. Return PASS/FAIL with evidence and smallest fixes.
```

### I01 — Integration worker prompt

```text
You are the only writer in {INTEGRATION_WORKTREE}. Integrate the independently reviewed commits for T01, T02, and T03 onto {BASE_SHA} in that order. Do not copy changes from archived experiments.

Resolve conflicts by preserving each task's accepted behavioral contracts, not by choosing the easiest textual side. Run the union of all focused tests plus the source lifecycle, capacity, pending-interaction, surface, whitewater, and cache lanes. Produce one integration commit and a conflict/decision log. Stop if a conflict requires changing an approved semantic.
```

### IR01 — Cross-lane review prompt

```text
Fresh independent review of integrated Wave 1 at {I01_SHA} versus {BASE_SHA}. Do not edit.

Review cross-lane interactions that isolated task reviews could miss: source batches after particle growth, reset with mixed prepared sources, whitewater after source deferral, stale capacity results after release, exception paths, GPU sync count, and lifecycle/RNG state. Run the combined focused suite. PASS only if no high/medium correctness finding remains.
```

### Generic fix-worker prompt

Use this after any task reviewer reports accepted findings:

```text
You are the sole fix writer in {TASK_WORKTREE}. The independent review artifact is {REVIEW_ARTIFACT}. Apply only these parent-accepted findings: {ACCEPTED_FINDINGS}.

Preserve the original task contract and do not broaden scope. Add a behavioral regression for each correctness fix, run the task's focused validation, and commit the fixes separately. Report changed files, tests, commit SHA, and anything not fixed. Stop if a finding requires a new semantic/product decision.
```

### Generic integration and independent integration-review prompts

Use for I02, I03, I04, and I05 (I01 uses the more specific prompts above).

```text
[Integration worker]
You are the only writer in {INTEGRATION_WORKTREE}. Starting from reviewed {INTEGRATION_BASE_SHA}, integrate these independently reviewed task commits in order: {TASK_COMMITS}. Resolve conflicts according to their accepted behavioral contracts, never by dropping tests or weakening invariants. Run {COMBINED_FOCUSED_COMMANDS}, git diff --check, and the relevant Blender acceptance markers. Produce one integration commit plus a decision/conflict log. Stop if conflict resolution changes an approved semantic.
```

```text
[Independent integration reviewer]
Fresh review of integrated {WAVE_ID} at {INTEGRATION_SHA} versus {INTEGRATION_BASE_SHA}. Do not edit and do not treat worker summaries as evidence. Inspect the full integrated diff and the cross-module runtime paths named in {CROSS_LANE_RISKS}. Run {COMBINED_FOCUSED_COMMANDS}. Return PASS/FAIL, severity-ranked evidence, missing tests, and smallest fixes. PASS only with no unresolved high/medium correctness finding.
```

Parameters:

| Integration | Commits | Cross-lane review focus |
|---|---|---|
| I02 | reviewed T04 commits | source-batch topology contributors, overflow preservation, leaf quality and session error handling |
| I03 | reviewed T05 commits | grid/planner/surface/cache dimension parity and authored-vs-resolved values |
| I04 | reviewed T06 then T07 commits | sampling-driven demand, reset/admission, migration, confirmation, and duplicate recommenders |
| I05 | reviewed T08 commits | panel callbacks, property reachability, old-file migration, warning/action truth, and backend/UI drift |

### T04 — Trace prompt

```text
Goal: Trace sparse topology rebuild as a state transaction and align runtime topology contributors with planner leaf-demand quality.

Inspect SparseGrid begin/mark/expand/finalize, old leaf table and churn remap, solver phase ordering, particles/sources/colliders/height/whitewater contributors, SourceBatch output from Wave 1, capacity planner coordinate union, memory estimates, overflow exception handling, and viewport/cache publication.

Required output:
- every array/state mutated before overflow is known;
- candidate topology design that preserves the current table/fields until admission;
- exact contributor parity table between planner and runtime;
- DemandQuality propagation rules;
- refusal policy preserving solver and last valid frame;
- allocation/VRAM implications and focused GPU tests.

Read-only, exact file/line evidence, no archived experiments.
```

### T04 — Implementation prompt

```text
You are the sole writer in {T04_WORKTREE}, based on reviewed {I01_SHA}. Implement transactional sparse topology and planner/runtime leaf parity using {T04_TRACE}.

Required behavior:
- candidate flags/count are produced without destroying the active leaf table;
- overflow is known before topology/table/field commit;
- refusal preserves solver, particles, prior topology, prior viewport/cache frame, and lifecycle state;
- runtime and planner include the same particles, every source representation, colliders, height safety band, whitewater, and expansion rules;
- leaf demand quality is EXACT/UPPER_BOUND/UNKNOWN truthfully;
- normal overflow does not route through generic solver teardown.

Add behavioral pure and CUDA tests, including overflow after several valid frames and mixed topology contributors. Preserve sparse 8×8×8 leaf storage. Commit on agent/res-quality-t04-topology and report evidence.
```

### T04 — Independent review prompt

```text
Independently review T04 in {T04_WORKTREE} against {I01_SHA}; do not edit.

Try to prove candidate topology mutates live state before admission, misses a contributor, mislabels quality, corrupts churn remap, publishes after refusal, or releases the solver. Compare planner and runtime contributor logic directly and run overflow tests. Check allocation cost and synchronization. Return PASS/FAIL with evidence and smallest fixes.
```

### D05 — Three design prompts

Launch these three read-only agents in parallel while T04 is implemented.

#### D05-A minimal interface

```text
Design a pure quality-resolution module for this solver with at most three public entry points. Maximize leverage and locality. It must own voxel ceil/padding, leaf dimensions, global liquid sampling/rest density/particle spacing, source-effective sampling, and surface voxel resolution while remaining Blender/Warp independent. Define types, invariants, error modes, dependencies, usage examples, migration strategy, and tests. Inspect the reviewed integration code; do not edit.
```

#### D05-B migration-first interface

```text
Design an alternative quality-resolution interface optimized for preserving existing .blend files and staged incremental adoption. Separate authored, resolved, and runtime state without requiring a one-shot SolverSettings rewrite. Define adapters for Blender RNA, legacy per-source emit_rate, SparseGrid, pressure, relaxation, ST-FLIP, surface, planner, and cache signatures. Include invariants/tradeoffs. Read-only.
```

#### D05-C caller-first interface

```text
Design an alternative quality-resolution interface that makes the common caller trivial: one resolution setting, one liquid sampling setting, and read-only derived results. Show how solver, planner, UI, and source adapters consume it without recomputing dimensions. Include extension strategy for future narrow-band modes without exposing a fake control now. Read-only.
```

### D05 synthesis prompt

```text
Read the three D05 designs and the actual reviewed code. Compare depth, locality, migration risk, testability, and caller complexity. Recommend one interface or a concrete hybrid with exact types and 1–3 public entry points. Reject unnecessary abstractions. This is advisory and read-only; the parent approves the final seam before T05.
```

### T05 — Trace prompt

```text
Using the parent-approved D05 interface, trace every duplicate calculation and consumer that must migrate: domain bounds, panel dimensions, SparseGrid padding, capacity planner leaf dims, memory estimator, pressure rest count, relaxation spacing, ST-FLIP reference calibration, source/height sampling, surface voxel/leaf sizing, cache settings signature, and tests. Identify adapters that can remain temporarily and define a safe migration order. Read-only with exact evidence.
```

### T05 — Implementation prompt

```text
You are the sole writer in {T05_WORKTREE}, based on reviewed {I02_SHA}. Implement the approved deep quality interface from {D05_SYNTHESIS} and migration trace {T05_TRACE}.

Requirements:
- one pure resolver owns physical voxel size, unpadded/aligned voxel dims, leaf dims, authoritative liquid sampling, nominal particle spacing/rest reference, and surface voxel size;
- SparseGrid, planner, memory estimate, pressure, relaxation, ST-FLIP, surface, panel readout adapter, and cache signature consume resolved values or a deliberate temporary adapter;
- no new active resolution lever is added;
- existing behavior remains numerically stable except confirmed rounding/display bugs;
- panel dimensions exactly match runtime padded/unpadded definitions.

Add contract/monotonicity/parity tests. Do not yet add source schema migration or full panel reorganization. Commit on agent/res-quality-t05-quality-contract.
```

### T05 — Independent review prompt

```text
Independently review T05 against {I02_SHA}; do not edit. Search for surviving duplicate ceil/padding/sampling/surface calculations, unit mismatches, circular dependencies, hidden Blender/Warp imports in the pure module, changed cache compatibility, and callers still using authored values where resolved values are required. Run contract and focused solver/planner tests. PASS only if the interface is genuinely deep rather than another pass-through layer.
```

### T06 — Trace prompt

```text
Goal: Trace and consolidate every capacity/memory assessment and recommendation path, plus planning resource ownership and repeated scene/source compilation.

Inspect capacity_planner, particle_capacity, memory_safety assessment/recommendation, Session._plan_scene/initialize/reset, SourceCompiler ownership/cache, SceneInputs mesh allocation, operator confirmation/revalidation, panel rendering, runtime growth, and tests. Identify duplicated estimates, conflicting recommenders, preflight GPU allocations, discarded snapshots, missing try/finally, candidate re-evaluation requirements, and progress seams.

Propose one pure capacity-service interface using the approved ResolvedQuality and PreparedSourceBatch. Keep fidelity changes approval-gated. Read-only, evidence-backed.
```

### T06 — Implementation prompt

```text
You are the sole writer in {T06_WORKTREE}, based on reviewed {I03_SHA}. Implement the traced capacity-service consolidation.

Requirements:
- one assessment and one ranked recommendation authority;
- capacity-only options precede semantic tradeoffs;
- no machine utilization heuristic changes fidelity;
- host scene/source geometry is prepared once and reused across particle-capacity candidates;
- planning performs no unaccounted GPU uploads before admission;
- compiler/prepared resources release in try/finally;
- semantic candidate recommendations are re-evaluated at candidate resolution before display and application;
- VRAM is re-sampled immediately before device allocation;
- existing active solver remains authoritative on cancellation/refusal;
- operator confirmation still shows exact old/new values and revalidates.

Keep broad UI reparenting for T08. Add focused performance ownership, stale signature, recommender uniqueness, and no-mutation tests. Commit on agent/res-quality-t06-capacity-service.
```

### T06 — Independent review prompt

```text
Independently review T06 against {I03_SHA}; do not edit. Search for a second active recommender, duplicated byte formulas, untracked GPU allocations, repeated mesh/compile work, stale resource samples, missing cleanup, semantic mutation before confirmation, incorrect candidate demand reuse, and runtime/initial capacity conflation. Run focused capacity/memory/performance tests and report PASS/FAIL with evidence.
```

### T07 — Trace prompt

```text
Goal: Trace global particles_per_voxel and per-source emit_rate through RNA callbacks, schema versions, gather adapters, pressure rest count, relaxation, ST-FLIP, height fill, source count/commit, pending state, saved-file migration, UI, and tests.

Design the exact migration for sampling_mode = INHERIT | MULTIPLIER | LEGACY_ABSOLUTE. New sources must inherit; old scenes must preserve behavior without guessing intent. Determine Reset/Reinitialize tiers and effective-value bounds/zero semantics. Identify overlap with T06 before parallel launch. Read-only.
```

### T07 — Implementation prompt

```text
You are the sole writer in {T07_WORKTREE}, based on reviewed {I03_SHA} or the parent-specified rebased SHA. Implement authoritative sampling semantics using {T07_TRACE} and ResolvedQuality.

Requirements:
- one global Liquid Sampling value is authoritative for solver rest density/spacing and inherited source occupancy;
- new sources default to INHERIT;
- explicit new overrides use a clearly labelled multiplier;
- existing scenes migrate to LEGACY_ABSOLUTE and preserve emitted occupancy;
- effective target is validated once with consistent 0..64 behavior;
- global sampling is Reset-tier, not a live pressure/relaxation mutation;
- source override UI/status states that it changes fill occupancy, not spatial resolution;
- no automatic density mutation for capacity recovery.

Add migration, callback/pending, source count/commit, pressure/relaxation consistency, and Blender RNA tests. Avoid broad panel reorganization. Commit on agent/res-quality-t07-sampling.
```

### T07 — Independent review prompt

```text
Independently review T07 against its base; do not edit. Test old-file migration, new-source defaults, zero and 64 bounds, Reset-tier behavior, pressure/rest-count consistency, height fill, initial-fill lifecycle, continuous top-up, and cache signatures. Look for guessed migration, dual active density controls, silent semantic changes, and callbacks that mutate an active solver inconsistently. Return PASS/FAIL with evidence.
```

### I04 — Integration/review prompts

Use the I01 integration prompt pattern, integrating T06 then T07. The independent reviewer must focus on quality/capacity separation, confirmation/revalidation, sampling-driven demand estimates, reset behavior, and duplicate settings translation.

### T08 — Trace prompt

```text
Goal: Trace current panel hierarchy, every resolution/sampling/capacity/surface/whitewater control, pending-status placement, operator ownership, RNA visibility/migration, warning wrapping, and current live screenshots. Map each control to Basic Quality, Inputs, Physics, Surface & Whitewater, Output & Cache, or Advanced.

Identify controls that are implementation details, unsupported claims, duplicates, or required for old-file compatibility. Define exact Blender parent/child panel structure, default-open state, narrow-width copy, one recommender/confirmation path, and screenshot acceptance. Read-only; use current reviewed backend behavior, not old screenshots as truth.
```

### T08 — Implementation prompt

```text
You are the sole writer in {T08_WORKTREE}, based on reviewed {I04_SHA}. Implement the Basic/Advanced information architecture from the canonical plan and {T08_TRACE}.

Requirements:
- approximately seven coherent top-level groups: Simulation, Quality, Inputs, Physics, Surface & Whitewater, Output & Cache, Advanced;
- Basic Quality shows one Simulation Resolution, one Liquid Sampling, Transfer/FLIP, Adaptive CFL/Max Substeps, and exact derived readouts;
- raw particle/leaf/whitewater/surface capacities, topology expansion, dense limits, low-level filters, snippets, profiling, and validation remain available under Advanced;
- one neutral pending summary and one action location; local controls may show compact Requires Reset/Reinitialize text without duplicate action rows;
- one capacity report, one ranked recommendation list, one confirmation flow;
- warnings wrap and remain actionable at normal/narrow N-panel width;
- render-surface labels do not imply disabling the solver SDF;
- no unsupported whitewater species controls;
- Quick Setup is made idempotent or explicitly prompts before adding another setup;
- guide and panel count are updated.

Preserve property storage and migration. Add mechanical panel hierarchy tests and Blender UI acceptance. Run profile-safe live review using a temporary add-on copy. Commit on agent/res-quality-t08-ui.
```

### T08 — Independent review prompt

```text
Independently review T08 against {I04_SHA}; do not edit. Trace every moved control to ensure it remains reachable and correctly tiered. Review normal and narrow screenshots, pending interaction, Quick Setup idempotency, capacity confirmation, exact grid/capacity readouts, old-file migration, accessibility of Advanced controls, and absence of unsupported claims. Run UI/property tests and inspect the live profile-safe report. Return PASS/FAIL with concrete UX and runtime findings.
```

## 8. Wave-6 validation prompts

### V01 — Pure contract auditor

```text
Read-only final audit of the integrated branch. Build a claim matrix for source all-or-nothing behavior, reset/topology state preservation, quality rounding, sampling migration, capacity-service uniqueness, whitewater SDF/capacity, stale-plan ownership, and UI callback tiers. Run the full Python suite plus focused contract lanes. Report missing behavioral tests, source-text-only assertions, and any high/medium defect. Do not edit.
```

### V02 — Blender/CUDA acceptance auditor

```text
Read-only final runtime audit in Blender 5.1/CUDA. Run mixed analytic+point+volume source acceptance, direct injection rejection/growth, reset with auto-grow off, topology overflow preservation, whitewater with render mesh disabled, capacity confirmation, source lifecycle, cache, handler persistence, and Phase-4 interaction. Use factory-startup/background lanes and report exact markers/artifacts. Do not edit source files.
```

### V03 — Profile-safe GUI UX auditor

```text
Launch a fresh owned normal-profile Blender through the canonical profile-safe temporary add-on workflow. Verify profile hashes before/after and control Blender through the available MCP path. Review the Basic/Advanced hierarchy at normal and narrow widths; execute Quick Setup twice, initialize, change sampling and voxel size, review pending actions, trigger safe capacity warning/confirmation, run live playback, reset/reinitialize/release, switch scenes, and test whitewater with render mesh hidden. Capture screenshots and a severity-ranked UX/runtime report. Do not modify project source or user profile.
```

### T09 — Final fix worker prompt

```text
You are the sole final writer on the integration branch. Read V01, V02, V03, and the latest integration review. Apply only accepted high/medium correctness or required UX fixes. Do not expand scope into deferred prediction, narrow-band, or species features. Add/adjust regressions for every fix, run focused validation, and produce one final fix commit with an evidence table. Stop for product decisions.
```

### IR09 — Final independent review prompt

```text
Fresh final review of the complete branch against BASE_SHA. Do not edit. Review Standards and Spec axes: repository contracts/AGENTS plus docs/industry_standard_resolution_quality_implementation_plan.md. Verify every task claim against source and runtime evidence, ensure no archived experiment was integrated, inspect migrations and UI truth, and run/inspect the full required gate evidence. PASS only with no unresolved blocker or high/medium finding.
```

## 9. Parent execution protocol

For each wave the parent will:

1. call `subagent({action: "list"})` before launching agents;
2. launch trace agents async with fresh context and unique artifacts;
3. continue parent-side integration preparation while traces run;
4. wait because each wave is a run-to-completion delivery;
5. synthesize traces and resolve contradictions before worker launch;
6. create dedicated branches/worktrees from the recorded reviewed SHA;
7. launch workers async in parallel only for approved non-overlapping lanes;
8. wait, inspect each diff and validation, then launch fresh independent reviewers in parallel;
9. route accepted findings to one fix worker per task worktree;
10. re-review material fixes;
11. launch one serial integration worker;
12. launch one fresh integration reviewer;
13. update the task ledger with SHAs, tests, artifacts, findings, and remaining risk;
14. stop and ask the user at any unapproved product/architecture decision.

Suggested task ledger path:

`artifacts/resolution_quality_orchestration/task_ledger.md`

Each row records:

- task ID and state;
- base/implementation/fix/integration SHA;
- trace and review artifact paths;
- focused tests and verdicts;
- unresolved findings;
- integration conflicts/decisions;
- profile/process cleanup evidence where applicable.

## 10. Merge and acceptance policy

A task is merge-ready only when:

- trace artifact is complete;
- worker commit is scoped and focused tests pass;
- independent reviewer returns PASS or all blocking findings have a reviewed fix;
- no unrelated files changed;
- `git diff --check` passes;
- task-specific acceptance markers pass.

A wave is complete only when its integrated diff receives a fresh cross-lane PASS.

The project is complete only after:

- full Python suite passes;
- required Blender/CUDA lanes pass;
- profile-safe GUI review passes with unchanged profile hashes;
- isolated `python tools/run_production_gate.py` reports `HEADLESS-GATE-PASS` (and GUI-inclusive gate if authorized/required);
- policy-drift reports validate after GUI-inclusive evidence refresh;
- final independent review returns no blocker/high/medium finding;
- user reviews the final UI and explicitly approves merge/commit disposition.
