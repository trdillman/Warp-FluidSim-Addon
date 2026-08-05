# Resolution Quality Merge Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Integrate the clean feature/resolution-quality-standardization branch into the current master checkout without losing active master capacity work, duplicating the three resolution-quality slices, or importing any archived experiment.

**Architecture:** Treat feature/resolution-quality-standardization as the umbrella integration branch. Its history contains the source-truth, session-capacity, and whitewater-truth sequences that also exist as the three sibling agent/res-quality-* branches; merge the umbrella once, then reconcile current master working-tree changes at the shared planning, transaction, UI, and gate seams. Keep authored quality, resolved quality, capacity plans, runtime capacity, and published frame output as distinct state layers.

**Tech Stack:** Python 3, Blender 5.1 bundled Python, Warp/CUDA runtime, pytest, Git worktrees, PowerShell.

## Global Constraints

- Do not reset, clean, stash, delete, overwrite, or broadly stage the current master working tree.
- The live field-storage contract remains a sparse tiled leaf grid with fixed 8x8x8 leaves addressed through a flat leaf table.
- Production simulation remains GPU-resident; CPU behavior is preview-only and cannot replace CUDA acceptance.
- Capacity refusal is fail-closed: no partial source emission, topology publication, fired-state advancement, solver release, or partial frame publication.
- feature/resolution-quality-standardization, agent/res-quality-t01-source-truth, agent/res-quality-t02-session-capacity, and agent/res-quality-t03-whitewater-truth all fork from 53564ac; do not merge the three sibling slices independently after taking the umbrella branch.
- The four branches explicitly classified by AGENTS.md as experiments (worktree-stflip-implementation, claude/particle-surface-rendering-apis-f62356, codex/native-gpu-interop, and codex/native-gpu-interop-main-20260715) are not merge candidates.
- Use Blender's bundled Python for Blender/OpenVDB workflows. Use system Python only for pure Python tests that do not import Blender or OpenVDB.
- A green focused test is not the completion gate. The canonical production gate and its marker/manifest truth must pass after integration.

---

## 0. Live branch map and authority

The following state was verified from the registered worktrees on 2026-08-02.

| Worktree / branch | Relation to master | Work carried | Merge treatment |
|---|---:|---|---|
| master / 2840d514 | target; dirty | 29 tracked edits plus 12 untracked files; active capacity/source work | Preserve and reconcile after the clean branch merge |
| feature/resolution-quality-standardization / c82988f | 21 commits ahead, 2 behind; clean | 67 files; 23 source, 37 tests, 5 docs, 2 tools | Canonical umbrella merge source |
| agent/res-quality-t02-session-capacity / e36d84d | 6 ahead, 2 behind; clean | Focused capacity transaction fixes | Reference slice only; already represented in umbrella history |
| agent/res-quality-t01-source-truth / 2a55a62 | 3 ahead, 2 behind; clean | Focused source-emission truth fixes | Reference slice only; already represented in umbrella history |
| agent/res-quality-t03-whitewater-truth / 365db1f | 3 ahead, 2 behind; clean | Focused whitewater truth fixes | Reference slice only; already represented in umbrella history |
| phase4/live-interaction-merge-20260728 / 701490ed | 10 behind, no unique diff; clean | No branch diff | Do not merge; stale synchronization worktree |
| claude/particle-surface-rendering-apis-f62356 / 0b47a05 | 2 ahead; dirty | Rendering/API experiment, demo/evidence dirt | Preserve; explicitly excluded |
| codex/native-gpu-interop / 3135f8b4 | 1 ahead; dirty | Native interop plus 756 untracked distribution/evidence files | Preserve; explicitly excluded |
| worktree-stflip-implementation / 556bce2 | no unique commits ahead; dirty | ST-FLIP/sourcing edits plus 95 benchmark files | Preserve; explicitly excluded |
| codex/native-gpu-interop-main-20260715 / c182d3a3 | 68 behind, no branch diff; dirty | Staged/unstaged interop edits plus evidence files | Preserve; explicitly excluded |

The commit graph proves that the umbrella branch carries the sibling sequences as equivalent commits after the common capacity baseline. The sibling branches overlap each other by 32–42 changed files. Merging all four would duplicate and conflict the same seams.

## 1. Impact graph and merge seams

~~~mermaid
flowchart LR
  UI["RNA properties / panels / operators"] --> BR["SceneBridge planning + lifecycle"]
  BR --> SRC["sources.py + SourceCompiler"]
  SRC --> Q["quality_contract.py"]
  SRC --> CP["capacity_service.py -> capacity_planner.py"]
  CP --> SOL["FluidSolver + ParticleSystem"]
  SOL --> WW["SurfacePipeline + WhitewaterSystem"]
  SOL --> OUT["frame stats / cache / viewport publication"]
  OUT --> GATE["production gate + Blender acceptance"]
~~~

### 1.1 Authoritative planning seam

- Umbrella branch: fluidSim_warp_blender51/sim/quality_contract.py:23-227 owns authored-to-resolved quality and source sampling (resolve_quality at 123-185, resolve_source_sampling at 187-227).
- Umbrella branch: fluidSim_warp_blender51/sim/capacity_planner.py:23-574 owns demand quality, sparse-leaf demand, particle demand, VRAM estimates, signatures, and immutable suggestions (plan_initialization at 290-398, suggest_configurations at 420-574).
- Umbrella branch: fluidSim_warp_blender51/sim/capacity_service.py:18-75 is the single public assessment/recommendation authority (assess at 32-53, recommend at 55-75).
- Current master target: fluidSim_warp_blender51/blender/scene_bridge.py:159-420 currently gathers the plan and initializes the solver; its public wrappers are 1462-1517 and the policy-update seam is 1794-1809.

### 1.2 Source-footprint and transaction seam

- fluidSim_warp_blender51/sim/source_regions.py:335-615 defines SourcePlan, normalized liquid targets (463-490), CompiledSourceFootprint (497-551), and SourceCompiler.compile (615-658).
- fluidSim_warp_blender51/sim/sourcing.py:2102-2268 owns SourcePipeline compilation; 2430-2555 owns emission transaction/growth admission; 2612-2828 applies an admitted batch; exact packet injection is 3063-3146.
- Current master target equivalents are source_regions.py:335-580 and sourcing.py:2057-2456; these are direct conflict points with the umbrella additions.

### 1.3 Runtime capacity and publication seam

- fluidSim_warp_blender51/sim/particle_capacity.py:21-83 defines geometric growth and atomic replacement planning (growth_target at 47, plan_particle_growth at 64).
- fluidSim_warp_blender51/sim/solver.py:408-780 owns settings, solver construction, source-plan binding (717), and capacity replacement (750); _source is 1098, step_frame is 1685, and exact direct injection is 1813.
- fluidSim_warp_blender51/sim/particles.py:100-182 owns ParticleSystem, resized_copy (159), and live-count/clear behavior.
- fluidSim_warp_blender51/sim/whitewater.py:33-280 owns deterministic admission kernels; WhitewaterSystem is 460-618, including candidate workspace allocation (521) and emission (563).
- fluidSim_warp_blender51/sim/surface.py:35-314 owns surface resolution, solver-SDF freshness (203), render build (227), and cadence hooks (249-276).
- The publication invariant is guarded at scene_bridge.py:750-853 (step) and 854-895 (play_cache_frame) on the umbrella branch; current master equivalents are 485-617 and 582-617.

### 1.4 UI and operator seam

- fluidSim_warp_blender51/properties.py:665-766 owns settings registration and capacity callbacks (_request_particle_capacity at 166, max_particles at 733/feature 747, auto_grow_particles at 741/feature 755). Surface quality controls are 967-1004 on current master and 981-1004 on the umbrella branch.
- fluidSim_warp_blender51/panels.py:115-180 draws capacity plans, 193-350 draws simulation/particle capacity state, and 477-520 draws source status on the umbrella branch. Current master has corresponding UI at 115-180, 185-350, and 403-520.
- fluidSim_warp_blender51/operators.py:67-220 owns initialize/capacity suggestion actions, 341-430 owns safe-settings and pending reinitialize, and 518-540 owns live toggle on the umbrella branch. Current master has the same entry points at 36-220, 292-430, and 519-540.
- The scene bridge public entry points are initialize_session (1838), set_live_enabled (1848), sync_scene_to_solver (1858), step_frame (1863), and play_cache_frame (1883) on the umbrella branch; current master equivalents are 1467, 1477, 1487, 1492, and 1512.

### 1.5 Verification seam

- Pure contract tests: tests/test_capacity_planner.py:44-190, tests/test_capacity_service.py:31-60, tests/test_quality_contract.py:25-132, and tests/test_particle_capacity_growth.py:15-204.
- Session/transaction tests: tests/test_scene_bridge_capacity_initialization.py:67-212, tests/test_session_capacity_transactions.py:153-709, and tests/test_source_emission_lifecycle.py:50-701.
- Source/topology/surface/whitewater tests: tests/test_source_sampling_migration.py:24-56, tests/test_topology_transaction.py:28-127, tests/test_surface_resolution_feature_contracts.py:17-136, and tests/test_whitewater_truth_contract.py:21-164.
- Cache/playback tests: tests/test_scene_bridge_cache_playback.py:138-1156.
- Gate truth tests: tests/test_production_gate_truthfulness.py:9-170; gate orchestration is tools/run_production_gate.py:73-105, 182-370.

## 2. Merge order and non-destructive setup

### Task 1: Freeze evidence without disturbing the current master

**Files:**
- Create outside the repository: C:\tmp\fluidsim-master-tracked.patch
- Create outside the repository: C:\tmp\fluidsim-master-status.txt
- Create outside the repository: C:\tmp\fluidsim-master-untracked\ with the 12 current untracked files listed below.

**Current untracked files to preserve explicitly:**

~~~text
backlog/tasks/TASK-WORKTREE-PRUNE-001-RETIRE-REDUNDANT-FLUIDSIM-CHECKOUTS.md
docs/industry_standard_resolution_quality_implementation_plan.md
docs/parallel_resolution_quality_subagent_orchestration_plan.md
docs/source_capacity_fail_closed_implementation_plan.md
fluidSim_warp_blender51/sim/capacity_planner.py
fluidSim_warp_blender51/sim/particle_capacity.py
pi-session-2026-07-31T18-16-55-019Z_019fb964-bdea-745e-9b05-862bba431909.html
tests/blender_capacity_planner_acceptance.py
tests/blender_particle_autogrowth_live_review.py
tests/test_capacity_planner.py
tests/test_particle_capacity_growth.py
tests/test_scene_bridge_capacity_initialization.py
~~~

- [ ] Run: git -C C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port diff --binary > C:\tmp\fluidsim-master-tracked.patch
- [ ] Run: git -C C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port status --short --untracked-files=all > C:\tmp\fluidsim-master-status.txt
- [ ] Copy the 12 listed files to C:\tmp\fluidsim-master-untracked\ with their relative paths intact; record SHA-256 hashes for every copy.
- [ ] Verify the original master status is byte-for-byte unchanged. Do not use git stash, git clean, git reset, or git checkout.

**Impact trace:** This creates a recoverable input snapshot only. It must not alter Blender code, generated demos, source lifecycle, or reports.

### Task 2: Create a disposable integration worktree from current master

**Files:**
- Create worktree: C:\tmp\fluidsim-resolution-quality-merge
- Create branch: codex/merge-resolution-quality

- [ ] Add the worktree at current master HEAD, not at any archived branch.
- [ ] Verify the new worktree is clean and records master HEAD 2840d514 as its base.
- [ ] Do not copy the 14 MB session HTML, modified demo blends, or preview PNGs into the integration worktree.

**Impact trace:** All merges and conflict resolution occur outside the user's dirty checkout. The original master remains the source of truth for the active uncommitted implementation until each slice is deliberately ported.

### Task 3: Merge only the umbrella resolution-quality branch

**Files:**
- Merge source: feature/resolution-quality-standardization
- Main conflict surfaces: the seams in Section 1.

- [ ] In the disposable worktree, run: git merge --no-ff --no-commit feature/resolution-quality-standardization
- [ ] Confirm the merge brings in capacity_service.py, quality_contract.py, the feature versions of capacity_planner.py and particle_capacity.py, surface/whitewater changes, and their tests.
- [ ] Do not merge agent/res-quality-t01-source-truth, agent/res-quality-t02-session-capacity, or agent/res-quality-t03-whitewater-truth afterward. Use them only for ancestry and conflict comparison.
- [ ] Run git diff --check before resolving semantic conflicts.

**Impact trace:** This preserves the umbrella branch's linear correction sequence: source truth -> session capacity -> whitewater truth -> quality/capacity/UI/topology integration. Taking sibling branches independently would reapply the same files and invalidate the intended sequence.

## 3. Conflict-resolution tasks by seam

### Task 4: Establish one quality and capacity authority

**Files:**
- Add/retain: fluidSim_warp_blender51/sim/quality_contract.py:23-227
- Add/retain: fluidSim_warp_blender51/sim/capacity_service.py:18-75
- Reconcile: fluidSim_warp_blender51/sim/capacity_planner.py:22-527 (dirty master) with :23-574 (umbrella)
- Reconcile: fluidSim_warp_blender51/sim/particle_capacity.py:17-98 (dirty master) with umbrella :21-83
- Reconcile consumers: fluidSim_warp_blender51/blender/scene_bridge.py:159-420 (master) with umbrella :145-524
- Tests: tests/test_capacity_planner.py:44-190, tests/test_capacity_service.py:31-60, tests/test_quality_contract.py:25-132, tests/test_particle_capacity_growth.py:15-204

- [ ] Compare the dirty-master and umbrella versions before choosing content. The source-capacity plan document is byte-identical in both checkouts; the planner, particle-capacity module, and related tests are not.
- [ ] Keep capacity_service.assess/recommend as the only public assessment path. Callers must not independently call planner internals and then draw a different recommendation.
- [ ] Preserve ResolvedQuality and resolve_source_sampling as the authored-to-runtime seam. Do not write resolved values back into Blender RNA during planning or suggestion display.
- [ ] Preserve feature planner fields for leaf_quality, leaf_reason, transaction_scratch_bytes, and additional_transaction_bytes; these account for topology and whitewater workspace residency in the admission decision.
- [ ] Preserve feature particle-growth accounting that charges the whitewater candidate workspace in the same atomic replacement transaction.
- [ ] Ensure scene ownership/signature invalidation remains scene-specific: a plan from another scene or changed authored settings must be rejected rather than applied.
- [ ] Keep suggestions immutable estimates; applying a suggestion must be an explicit operator action.

**Impact trace:**

properties.py callback -> scene_bridge.build_settings -> resolve_quality -> SourceCompiler footprints -> capacity_service.assess -> CapacityPlan/signature -> panel/operator display. No solver or authored setting is mutated in this path. The only transition that mutates runtime capacity is the guarded operator/initialization path in the next task.

- [ ] Run:

~~~text
python -m pytest -q tests/test_capacity_planner.py tests/test_capacity_service.py tests/test_quality_contract.py tests/test_particle_capacity_growth.py
~~~

- [ ] Commit this seam only after the tests pass: refactor(capacity): centralize quality and admission authority.

### Task 5: Reconcile scene planning, solver replacement, and source transactions

**Files:**
- fluidSim_warp_blender51/blender/scene_bridge.py:264-414 (_plan_scene), :415-519 (plan binding and candidate replacement), :524-688 (initialize), :723-895 (sync/step/cache), :1069-1110 (frame-change handoff)
- Current master conflict source: scene_bridge.py:159-420, :455-617, :733-804
- fluidSim_warp_blender51/sim/solver.py:408-780, :1098-1140, :1685-1830
- fluidSim_warp_blender51/sim/particles.py:100-182
- fluidSim_warp_blender51/sim/sourcing.py:2102-2268, :2430-2655, :3063-3146
- fluidSim_warp_blender51/sim/source_regions.py:335-615
- Tests: tests/test_scene_bridge_capacity_initialization.py:67-212, tests/test_session_capacity_transactions.py:153-709, tests/test_source_emission_lifecycle.py:50-701, tests/test_sourcing_pipeline.py:339-404, tests/test_topology_transaction.py:28-127

- [ ] Preserve the umbrella _plan_scene order: gather descriptors/plans/height source, prepare automatic seeds, compile source footprints without upload, assess capacity, then return an immutable-enough ScenePlanningSnapshot.
- [ ] Preserve the candidate-solver seam (_prepare_solver_replacement / _commit_solver_replacement) so reset/reinitialize failures discard only the candidate pool and retain the authoritative live solver.
- [ ] Preserve atomic particle resize: ParticleSystem.resized_copy succeeds before the live pool is swapped; failure leaves the old pool, source batch, topology, and status intact.
- [ ] Preserve source transaction ordering: count/admit the complete batch, then launch/apply, then advance one-shot/initial-fill lifecycle state only after successful admission.
- [ ] Keep inject_particle_packets_exact as the exact packet seam. Do not reintroduce a position-only partial append path while resolving the dirty master implementation.
- [ ] Preserve topology transaction behavior: candidate leaf coordinates/arrays are committed only after successful overflow handling; deferred overflow leaves live sparse state unchanged.
- [ ] Preserve cache playback isolation: play_cache_frame reads cached output and updates presentation state without initializing or stepping a live solver.
- [ ] Preserve frame publication ordering: capacity/growth diagnostics are assembled before the frame is published, and a capacity-refused frame cannot publish partial output.

**Impact trace:**

initialize_session -> Session._plan_scene -> capacity assessment -> candidate solver/seed admission -> _commit_solver_replacement -> live step -> SourcePipeline._begin_emission_transaction -> solver/source application -> surface/whitewater update -> frame/cache/viewport publication. A failed node must return to the previous authoritative state at that node; it must not fall through to a partial downstream state.

- [ ] Run:

~~~text
python -m pytest -q tests/test_scene_bridge_capacity_initialization.py tests/test_session_capacity_transactions.py tests/test_source_emission_lifecycle.py tests/test_sourcing_pipeline.py tests/test_topology_transaction.py tests/test_scene_bridge_cache_playback.py
~~~

- [ ] Commit only after the focused transaction and playback tests pass: fix(runtime): preserve atomic capacity and source publication.

### Task 6: Reconcile source sampling, surface resolution, and whitewater truth

**Files:**
- fluidSim_warp_blender51/sim/quality_contract.py:123-227
- fluidSim_warp_blender51/sim/source_regions.py:463-615
- fluidSim_warp_blender51/sim/sourcing.py:720-964, :1850-2008, :2102-2256
- fluidSim_warp_blender51/sim/surface.py:35-314
- fluidSim_warp_blender51/sim/whitewater.py:33-280, :460-618
- fluidSim_warp_blender51/sim/solver.py:408-444, :1098-1130, :1685-1813
- Tests: tests/test_source_sampling_migration.py:24-56, tests/test_surface_resolution_feature_contracts.py:17-136, tests/test_whitewater_truth_contract.py:21-164, tests/test_height_source_pipeline.py, tests/test_surface_stack.py

- [ ] Route inherited source sampling through resolve_source_sampling; preserve explicit legacy absolute rates without guessing an inherited multiplier.
- [ ] Preserve SourcePlan normalization (normalized_liquid_target, liquid_coverage_emission_enabled, plan_requires_source_topology) so capacity demand and actual emission use the same semantic target.
- [ ] Preserve surface-resolution decoupling: render surface voxel size may differ from simulation voxel size, while solver-SDF freshness remains independent of render-mesh enablement.
- [ ] Preserve surface cadence hooks (after_substep, after_frame) and frame identity advancement; do not make a render-only surface rebuild mutate solver state.
- [ ] Preserve whitewater dependency ordering: solver SDF must be available before whitewater candidate marking; SDF failure must fail whitewater closed without churn or hidden emission.
- [ ] Preserve deterministic whitewater admission and candidate workspace capacity in the same growth budget used by particle replacement.
- [ ] Verify the public stats contract reports availability, drops, life range, and dependency failure explicitly rather than silently clipping.

**Impact trace:**

Authored sampling/surface/whitewater controls -> ResolvedQuality -> source plan target/footprint -> capacity plan -> solver runtime -> surface SDF -> whitewater admission -> frame stats. The resolved values must remain monotonic and must not alter authored settings during recovery.

- [ ] Run:

~~~text
python -m pytest -q tests/test_source_sampling_migration.py tests/test_surface_resolution_feature_contracts.py tests/test_whitewater_truth_contract.py tests/test_height_source_pipeline.py tests/test_surface_stack.py
~~~

- [ ] Commit: fix(quality): keep sampling, surface, and whitewater contracts aligned.

### Task 7: Reconcile Blender properties, panels, and operators

**Files:**
- fluidSim_warp_blender51/properties.py:665-766, :967-1004, and callbacks :148-203
- fluidSim_warp_blender51/panels.py:115-180, :193-350, :394-520
- fluidSim_warp_blender51/operators.py:67-220, :341-430, :518-540
- fluidSim_warp_blender51/blender/scene_bridge.py:154-188, :1813-1888, :1540-1779
- Tests: tests/test_operator_panel_behavior.py, tests/test_phase4_pending_interaction.py, tests/test_scene_bridge_cache_playback.py:837-1066, tests/blender_phase3_source_ui_acceptance.py

- [ ] Preserve the settings schema and update callbacks for max_particles, auto_grow_particles, particle_growth_failure_policy, inherited source sampling, surface resolution, and whitewater controls.
- [ ] Ensure _request_particle_capacity refreshes the scene-owned plan and status only; it must not create or replace a solver as a side effect of editing a property.
- [ ] Ensure _draw_capacity_plan only renders the scene-owned, signature-matching plan and presents stale-plan/application errors explicitly.
- [ ] Keep FLUIDSIM_OT_apply_capacity_suggestion and FLUIDSIM_OT_apply_safe_settings as explicit actions. They must revalidate the plan signature before changing settings.
- [ ] Keep pending reinitialize/reset behavior truthful: a pending setting must not be reported as applied until the solver replacement succeeds.
- [ ] Keep toggle_live and cache playback transitions routed through the scene bridge public wrappers rather than duplicating lifecycle logic in operators.
- [ ] Preserve source role separation: Force/Field objects affect the solver force/field phases and do not become liquid sources.

**Impact trace:**

RNA edit -> callback -> scene-owned plan refresh -> panel display. Explicit operator -> signature validation -> scene bridge transition -> candidate solver/source transaction -> status/pending state. No UI path may bypass the runtime transaction seam.

- [ ] Run:

~~~text
python -m pytest -q tests/test_operator_panel_behavior.py tests/test_phase4_pending_interaction.py tests/test_scene_bridge_cache_playback.py
blender --background --factory-startup --python tests/blender_phase3_source_ui_acceptance.py -- --json-out artifacts/phase3_source_ui_acceptance.json
~~~

- [ ] Commit: fix(ui): bind capacity and quality controls to owned runtime state.

### Task 8: Reconcile gate, report, and documentation wiring

**Files:**
- tools/run_production_gate.py:73-105, :182-370
- tests/blender_capacity_planner_acceptance.py:61-329
- tests/blender_particle_autogrowth_live_review.py:15-105
- tests/blender_mixed_source_batch_acceptance.py:56-186
- tests/blender_source_sampling_acceptance.py:21-67
- tests/blender_topology_transaction_acceptance.py:19-53
- tests/blender_whitewater_truth_acceptance.py:26-177
- tests/blender_quick_setup_idempotence.py:15-37
- tests/test_production_gate_truthfulness.py:9-170
- Docs: docs/industry_standard_resolution_quality_implementation_plan.md:124-190, :319-404; docs/source_capacity_fail_closed_implementation_plan.md:39-133, :169-252, :317-424; fluidSim_warp_blender51/docs/user_guide.md sections for capacity/quality/source workflows.

- [ ] Keep the capacity planner acceptance in the canonical gate with marker CAPACITY-PLANNER-ACCEPTANCE-PASS.
- [ ] Keep gate output fail-closed: traceback, missing marker, stale manifest, or zero-exit false green must fail the gate.
- [ ] Preserve the distinction between the default headless production gate and --include-gui; do not claim GUI acceptance from a headless pass.
- [ ] Keep acceptance reports and hashes generated under artifacts/; do not stage old evidence from archived worktrees.
- [ ] Reconcile the two differing resolution-quality plan documents by diffing them against the current implementation; retain only statements proven by the merged code and gates. The source-capacity plan is identical in both source checkouts and needs no duplicate copy.

**Impact trace:**

Each runtime seam above -> focused unit tests -> Blender acceptance marker -> production gate manifest -> optional GUI/playback lanes. A pass is valid only when the current HEAD, report metadata, artifact hashes, and marker verdict agree.

- [ ] Run the focused gate after Tasks 4–7:

~~~text
python tools/run_production_gate.py
~~~

- [ ] Run the full GUI-inclusive gate after the headless lane is green:

~~~text
python tools/run_production_gate.py --include-gui
python tools/validate_policy_drift_reports.py
~~~

- [ ] Validate any generated GUI playback report with python tools/validate_gui_playback_report.py artifacts/gui_playback_probe.json.

## 4. Integration validation matrix

Run these in order; stop at the first failed gate and repair the seam named by the failure.

1. **Diff hygiene:** git diff --check; inspect git status --short --untracked-files=all.
2. **Pure capacity/quality:** planner, service, quality, growth tests from Task 4.
3. **Runtime transaction:** scene initialization, session replacement, source lifecycle, sourcing, topology, and cache/playback tests from Task 5.
4. **Surface/whitewater:** sampling, surface, whitewater, height-source, and surface-stack tests from Task 6.
5. **UI contract:** operator/pending interaction tests and Phase 3 source UI acceptance from Task 7.
6. **Blender acceptance:** capacity planner, mixed-source, autogrowth, source sampling, topology transaction, whitewater truth, and quick-setup scripts.
7. **Canonical production gate:** python tools/run_production_gate.py.
8. **GUI production gate:** python tools/run_production_gate.py --include-gui plus policy-drift and playback report validators.
9. **Full Python regression:** python -m pytest -q.

### Acceptance invariants

- Unknown particle or leaf demand fails closed; no invented count is admitted.
- Initial automatic seeds and frame-one source emissions are planned as one admission configuration.
- Growth charges particle, topology, transaction scratch, and whitewater candidate workspace residency.
- A failed growth/replacement leaves the authoritative solver, source lifecycle, leaf table, and published output unchanged.
- Continuous/initial-fill/direct source lifecycle state advances only after a whole batch is admitted and applied.
- Cache playback does not initialize or step a live solver and refuses mismatched settings/scene identity.
- Surface resolution is decoupled from simulation resolution without changing authored settings.
- Whitewater never consumes an unavailable solver SDF and reports deterministic capacity drops/availability.
- UI suggestions are stale-safe, explicit, and reversible; property edits alone do not replace a solver.
- Gate markers, manifests, current HEAD, and generated report hashes agree.

## 5. Commit and handoff sequence

- [ ] Commit the clean umbrella merge only after conflict review: merge: integrate resolution-quality umbrella.
- [ ] Commit capacity/quality authority reconciliation: refactor(capacity): centralize quality and admission authority.
- [ ] Commit runtime transaction reconciliation: fix(runtime): preserve atomic capacity and source publication.
- [ ] Commit surface/whitewater/source semantics: fix(quality): keep sampling, surface, and whitewater contracts aligned.
- [ ] Commit UI/operator seam: fix(ui): bind capacity and quality controls to owned runtime state.
- [ ] Commit gate/docs updates only after current artifacts are regenerated: test(gate): verify resolution-quality acceptance lanes and docs(merge): record integration evidence.
- [ ] Keep generated demo blends/previews, the session HTML export, bench/, dist/, and archived evidence outside these commits unless a specific acceptance lane requires a newly generated artifact.
- [ ] Before promotion, capture the integration branch HEAD, merge-base, status, focused test output, production-gate manifest, GUI report, and artifact hashes.

## 6. Rollback and stop conditions

- If a conflict requires choosing between two incompatible planner or transaction authorities, stop and compare the branch-specific tests before editing callers.
- If a candidate solver failure releases or mutates the live solver, stop; do not weaken the test or continue to UI/gate work.
- If any current master dirty file cannot be classified as code, test, documentation, or generated output, leave it untouched and record it in the checkpoint manifest.
- If the production gate passes only through a special alias, partial lane, stale report, or missing marker, treat the integration as blocked.
- Rollback is by abandoning the disposable integration worktree/branch after preserving its logs; never reset the user's original master checkout.

## 7. Plan self-review

- **Spec coverage:** capacity planning, source truth, session transactions, quality/sampling, surface/whitewater, UI, cache/playback, and production-gate truth each have a dedicated task and acceptance lane.
- **Duplicate merge prevention:** the umbrella branch is selected from live ancestry; sibling branches are reference slices only.
- **Seam locality:** planning, runtime transaction, UI, and gate entry points are named with current branch line ranges; callers and tests cross the same seams.
- **Evidence discipline:** clean worktrees contribute committed diffs; dirty worktrees contribute tracked and untracked work only after classification; archived experiments are not treated as merge candidates.
- **No destructive step:** all integration work is isolated; current master dirt is checkpointed and preserved before any merge.

