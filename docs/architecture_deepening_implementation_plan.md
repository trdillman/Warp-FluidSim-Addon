# FluidSim Architecture Deepening — Complete Implementation Plan

**Status:** Approved for implementation planning; no production refactor has started.  
**Date:** 2026-08-05  
**Revalidated baseline:** `2db11d22ebc3371b620fc0e20009644f5c98d90f` (`2db11d2`)  
**Architecture vocabulary:** `CONTEXT.md` and the codebase-design terms **module**, **interface**, **depth**, **seam**, **adapter**, **leverage**, and **locality**.  
**Program order:** Simulation Admission → Output Publication → Preview/Cache Playback → Pending Settings.

## 1. Executive summary

This is one linked architecture program, delivered as four independently releasable phases. Each phase deepens one module, moves behavior behind a smaller interface, proves the current runtime contract, and passes a rollback checkpoint before the next phase starts.

The program reduces `fluidSim_warp_blender51/blender/scene_bridge.py` from a 2,016-line mixed owner into the Solver Session module: scene ownership, direct solver synchronization, direct stepping, frame statistics, error storage, and teardown. It must stop owning admission planning, persistent output datablocks, Blender timeline interpretation, deterministic cache baking, and Pending Settings policy.

The program does **not** redesign the solver, sparse leaf grid, viewport playback behavior, backend-status behavior, cache format, or UI. A behavior correction may enter the program only when a reproduced user-visible defect or documented contract violation is captured by a failing regression test first.

Archived native-interop, rendering-API, and ST-FLIP experiment worktrees are excluded. They are not merge candidates or implementation sources.

## 2. Fixed decisions

1. Treat all four candidates as one linked architecture program.
2. Execute in dependency order:
   1. Simulation Admission;
   2. Output Publication;
   3. Preview/Cache Playback and Production Cache Bake;
   4. Pending Settings.
3. Targeted behavior changes require:
   - reproduced user-visible defect or documented contract violation;
   - failing regression test;
   - isolated implementation change, separate from ownership movement.
4. Simulation Admission begins with an authored Blender scene and ends with a running Solver Session.
5. Admission remains Blender-owned; a focused real-Blender background lane is authoritative.
6. Replacement is **preflight-atomic**, not fully swap-atomic:
   - preserve the existing solver through every non-allocating preflight check;
   - release it at Replacement Cutover;
   - if allocation/configuration then fails, leave a clean inactive session with the error preserved.
7. Output Publication owns all persistent simulation-output state: objects, datablocks, geometry, attributes, point capacity/schema, materials, viewport visibility, and render visibility.
8. Diagnostic overlay drawing remains a separate Blender adapter.
9. The refactor adds no GPU synchronization, device-to-host copy, host-frame acquisition, or Blender ID allocation relative to the characterized publication path.
10. Existing structural point-cloud replacement for capacity growth, capacity shrink, or attribute-schema change is preserved and measured; the plan does not falsely claim that all allocation is eliminated.
11. Live and cached inputs produce equivalent published output where their data overlaps. Missing cache data follows explicit, characterized fallback.
12. Preview Playback is non-authoritative visual feedback.
13. Production Cache Bake is deterministic and independent of Blender viewport event delivery.
14. Preserve current playback behavior, catch-up behavior, desynchronization behavior, status strings, and backend-status presentation. Do not introduce lag, progress, recovery, or authority states.

## 3. Truth contract

### 3.1 Core behavior that must remain real

- Simulation remains GPU-resident except for existing publication, cache, profiling, and validation readbacks.
- Live field storage remains fixed `8x8x8` sparse leaves addressed by a flat leaf table.
- Admission captures one coherent evaluated Blender scene and rejects memory, source-integrity, and capacity blockers before Replacement Cutover.
- Initial source and seed admission remains fail-closed and transactional.
- Direct particle injection admits against the refreshed device particle count and reports any short commit as an invariant failure rather than claiming success.
- Successful admission installs sources, colliders, force fields, height source, snippets, initial particles, and initially enabled output state before interactive stepping.
- Preview Playback and Cache Playback create real Blender output datablocks and attributes.
- Production Cache Bake advances frames sequentially and writes full authoritative cache data without consuming preview-decimated output.
- Persistent Blender handlers remain deduplicated, persistent across save/reopen, and safely removed.
- Pending reset/reinitialize state remains per-scene and preserves current undo, script, depsgraph, and rewind behavior.
- Existing statuses and playback side effects remain unchanged unless the targeted-change rule is satisfied.

### 3.2 Allowed non-authoritative surfaces

- Preview Playback is visual feedback and testing, not production-cache proof.
- A cache tagged `source="live_simulation"` remains distinct from deterministic output tagged `source="simulation"`.
- HTML reviews, screenshots, demos, previews, benchmarks, and policy reports are review/evidence artifacts. They do not certify core behavior by themselves.

### 3.3 Prohibited drift

- Do not replace authoritative Blender admission tests with pure mocks.
- Do not infer behavior from module names, UI labels, shared helper calls, or non-crash tests.
- Do not route Production Cache Bake through viewport-decimated particles, retained preview surfaces, display colors, or frame-handler cadence.
- Do not claim that publication performs no allocation under all conditions; preserve and disclose the existing growth fallback.
- Do not leave shallow compatibility wrappers after all internal callers migrate.
- Do not modify or extract code from archived experimental worktrees.

## 4. Claim matrix

| Claim | Required evidence | Current implementation | Status | Required move |
|---|---|---|---|---|
| One Simulation Admission module owns authored scene through running solver | Source ownership plus Blender fault-injection acceptance | `_plan_scene()`, `initialize()`, reset, seed commit, and output prewarm are coordinated in `scene_bridge.py` | Partial | Deepen behind `simulation_admission.py` |
| Admission uses one coherent Blender evaluation | Snapshot identity/frame/depsgraph/device/domain assertions | Sources, plans, colliders, forces, height, and seeds are gathered separately | Partial | Introduce one Admission Snapshot and forbid a second gather before cutover |
| Replacement is preflight-atomic | Fault injection before and after cutover | Current ordering largely preserves preflight but is not isolated | Partial | Name phases and prove the cutover contract |
| Output Publication owns persistent Blender output state | Source inspection plus live/cache Blender acceptance | `viewport.py` mostly owns it but mixes extraction, publication, preparation, materials, and overlay | Partial | Deepen publication; separate payload adapters and overlay |
| Refactor adds no hot-path sync/copy/ID allocation | Instrumented baseline and profile comparison | Existing prewarm helps; residual point growth may reallocate | Partial | Freeze baseline; reject deltas; preserve documented fallback |
| Live/cache output is equivalent where data overlaps | Normalized payload and real Blender comparisons | Paths converge late and differ in cadence/fallback details | Partial | Normalize shared facts before publication and test differences explicitly |
| Production Cache Bake is independent of preview policy | Source inspection, repeat-bake hashes, cache acceptance | Deterministic export exists as a broad `Session` method | Partial | Move to `cache_bake.py` without changing output |
| Playback/status behavior remains unchanged | Exact transition-matrix characterization | Behavior is real but spread across session methods, globals, and handlers | Conformant but fragile | Freeze before moving ownership |
| One Pending Settings module owns transition policy | Exhaustive registry plus Blender Phase 4 acceptance | Classification, snapshots, timers, operators, and presentation are distributed | Partial | Deepen behind `pending_settings.py` |
| Architecture report proves implementation | No such evidence is sufficient | Temporary HTML report exists | Review artifact only | Use only as design context |

## 5. Target module topology and dependency direction

### 5.1 `blender/simulation_admission.py` — deep module

Owns the authored-scene-to-running-solver transaction:

- one Admission Snapshot;
- scene capture and source compilation needed for preflight;
- memory/capacity/integrity planning;
- optional capacity replan;
- Replacement Cutover;
- solver allocation/configuration;
- source, collider, force, height, snippet, and seed installation;
- initial Output Preparation;
- applied-state capture;
- reset admission;
- deterministic cleanup on every failure.

Its interface exposes admission, reset, and non-mutating capacity refresh. Intermediate descriptors, compiler objects, footprints, blockers, retry details, and transaction phases stay behind the seam.

### 5.2 `blender/solver_settings.py` — shared projection module

Owns Blender-authored-value projection into owned solver values:

- domain bounds;
- field mappings and aliases;
- enum conversion;
- vector/rotation ownership conversion;
- gravity projection;
- selected-field application;
- registry validation.

Simulation Admission and Pending Settings are the two real consumers that justify this seam. Lifecycle tier policy does not belong here.

### 5.3 `blender/output_publication.py` — deep module

Owns persistent simulation-output state and has two internal paths:

- **Output Preparation:** object/datablock/material creation, initial schema/capacity, property-triggered enablement, and allowed growth handling;
- **Output Publication:** mutation from one already prepared Publication Payload.

It also owns live-solver and cache payload adapters, but payload preparation and datablock mutation remain separately testable so publication cannot reacquire solver data.

Initial enabled outputs are prepared during Simulation Admission. An output enabled later is prepared from the existing property/timer path outside frame handling. Existing structural point-cloud replacement for growth, shrink, or attribute-schema change remains owned here and behaves exactly as characterized; no additional allocation path may be introduced.

### 5.4 `blender/viewport_overlay.py` — narrow Blender adapter

Owns diagnostic overlay registration and drawing only. It may read presentation facts but cannot create, resize, remove, or publish output datablocks.

### 5.5 `blender/playback.py` — deep module

Owns current Preview/Cache Playback policy:

- live/cache selection and exclusivity;
- playback-pre admission;
- frame-change interpretation;
- frame-handler re-entry protection;
- render-restore suppression;
- rewind/reset dispatch;
- current bounded catch-up and desynchronization behavior;
- cache-frame playback;
- handler registration/removal;
- exact current status/error side effects.

Solver Session remains responsible for direct synchronization and stepping. Playback is non-authoritative, but its current behavior is frozen.

### 5.6 `blender/cache_bake.py` — deep deterministic runner

Owns sequential production frame advancement, cache writes, metadata, scene-frame restoration, temporary live-state changes, and failure cleanup. It cannot depend on frame-change delivery or Output Publication state.

### 5.7 `blender/pending_settings.py` — deep module

Owns:

- `SceneUpdate` lifecycle classification;
- complete setting-tier registry;
- Applied Settings Snapshot;
- per-scene Pending Settings state;
- reset/reinitialize signatures and reasons;
- applicability and blocked reasons;
- safe live/reset/output application;
- reset/reinitialize/revert transitions;
- disposable-start debounce;
- undo/script/depsgraph refresh;
- presentation facts and state cleanup.

Properties, operators, panels, overlay, and playback remain adapters and must not reconstruct policy.

### 5.8 Dependency rule

```text
Blender adapters (properties, operators, panels, handlers)
    → pending_settings / playback / simulation_admission / cache_bake
    → solver_settings / output_publication / Solver Session
    → sim and io modules
```

No new deep module may import a UI adapter. Bidirectional imports between the new deep modules are prohibited; use owned values, narrow callbacks, or lazy imports at the existing Blender seam.

## 6. Phase 0 — baseline, vocabulary, and characterization

### Task 0.1 — Persist vocabulary and plan

**Files**

- New `CONTEXT.md`
- New `docs/architecture_deepening_implementation_plan.md`

**Atomic actions**

1. Define the domain terms used by the four modules.
2. Record the sparse-grid, authority, replacement, and behavior-change invariants.
3. Mark this file as the authoritative implementation plan.
4. Do not create an ADR: no `docs/adr/` decision conflicts with this program.

**Checkpoint**

- New modules/tests use `CONTEXT.md` terms consistently.
- No concrete implementation claim is certified by the architecture HTML report.

### Task 0.2 — Protect the implementation-start checkout

At plan revalidation, tracked files were clean at `2db11d2`; only this plan and `CONTEXT.md` were untracked. Other processes committed source-capacity and policy-evidence work while planning was in progress, so implementation must not assume that status remains static.

Immediately before implementation:

1. record `HEAD`, `git status --porcelain=v2`, and hashes for every dirty path;
2. treat every pre-existing dirty path as user-owned;
3. do not reset, stash, clean, overwrite, broadly format, or broadly stage it;
4. re-check `HEAD`, status, and hashes at every phase boundary;
5. stop and revalidate the affected phase if another process advances `HEAD` or changes an overlapping path.

Architecture commits contain only intentional architecture files.

### Task 0.3 — Capture baseline runtime evidence

Run before adding characterization tests:

```text
python -m pytest -q tests/test_scene_bridge_cache_playback.py tests/test_phase4_pending_interaction.py tests/test_scene_bridge_capacity_initialization.py tests/test_frame_output.py tests/test_frame_change_catchup.py tests/test_operator_panel_behavior.py tests/test_viewport_pointcloud_prewarm.py tests/test_performance_hotpath_contracts.py
```

Also capture the focused user-path profile using the verified command in `AGENTS.md`:

```text
FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py
```

Store baseline counts/timings in an architecture-specific artifact. Do not refresh authoritative policy reports at this stage.

### Task 0.4 — Add characterization tests before ownership movement

**Files**

- `tests/test_scene_bridge_capacity_initialization.py`
- `tests/test_scene_bridge_cache_playback.py`
- `tests/test_frame_change_catchup.py`
- `tests/test_viewport_pointcloud_prewarm.py`
- `tests/test_phase4_pending_interaction.py`
- `tests/test_operator_panel_behavior.py`
- `tests/blender_handler_persistence_gate.py`

**Characterize**

- preflight failure preserves the old solver;
- every current post-cutover failure leaves a clean inactive session;
- one scene identity, frame, depsgraph generation, device, and domain transform feed one admission;
- no second gather occurs between preflight success and cutover;
- initially enabled outputs are prepared before frame-handler publication;
- disabled outputs remain unallocated until the current enable/update path runs;
- every current structural point-cloud replacement case—capacity growth, capacity shrink, and attribute-schema change—remains allowed and separately counted;
- two-scene behavior for shared and distinct output collection names, tagged-object lookup, publication isolation, visibility, unlinking, and scene-scoped removal;
- live/cache exclusivity, lazy initialization, playback-pre initialization, rewind behavior, bounded catch-up, over-bound desynchronization, `last_frame`, exact status text, render restore, nested-handler suppression, and teardown order;
- Pending Settings action, reasons, blocked reason, timers, and current IDProperty presentation.

**Checkpoint**

- Perturbing any characterized call order or status side effect causes a focused test failure.
- No production behavior changes in Phase 0.

## 7. Phase 1 — Simulation Admission

### Task 1.1 — Extract solver-settings projection

**Files**

- New `fluidSim_warp_blender51/blender/solver_settings.py`
- `fluidSim_warp_blender51/blender/scene_bridge.py`
- New `tests/test_solver_settings_projection.py`

**Atomic actions**

1. Move domain-bounds calculation, field mappings, aliases, enum conversion, vector ownership, gravity projection, selected-field application, and target validation.
2. Preserve lazy imports required by Blender/Warp loading.
3. Ensure no RNA/mathutils proxy reaches `SolverSettings`.
4. Keep reset/reinitialize tier classification in its current owner until Phase 4.
5. Migrate `build_settings()` callers.

**Checkpoint**

- Every projected `SolverSettings` value equals baseline output.
- Invalid mappings fail closed.
- The new module has two real consumers planned: Admission and Pending Settings.

### Task 1.2 — Introduce one Admission Snapshot

**Files**

- New `fluidSim_warp_blender51/blender/simulation_admission.py`
- `fluidSim_warp_blender51/blender/scene_inputs.py`
- `fluidSim_warp_blender51/blender/scene_bridge.py`
- `fluidSim_warp_blender51/blender/sources.py`
- `fluidSim_warp_blender51/sim/source_regions.py`

**Atomic actions**

1. Capture scene identity, current frame, depsgraph generation, configured device, domain transform/bounds, settings, sources, source plans, colliders, forces, height source, and seeds once.
2. Convert Blender proxy values into owned descriptors before the snapshot reaches solver configuration.
3. Prevent a second scene gather between successful preflight and Replacement Cutover.
4. Derive every auto-growth candidate assessment from the original Admission Snapshot's owned settings, compiled footprints, and demand facts; do not re-gather Blender data or recompile against a different evaluation.
5. Put `SourceCompiler.release()` in `finally` so compile/planner failures cannot leak transient resources.
6. Give one owner responsibility for disposing transient `SceneInputs`, compiler, mesh-cache, and planning resources for failed, rejected, or superseded candidate plans as well as successful commit.
7. Keep source compilation used for capacity estimation separate from runtime source ownership while guaranteeing both derive from the same snapshot.

**Checkpoint**

- Snapshot-coherence tests detect scene/frame/device/generation mixing.
- No Blender RNA/mathutils proxy escapes into the running solver.

### Task 1.3 — Move preflight behind Simulation Admission

**Atomic actions**

1. Move `ScenePlanningSnapshot`, scene planning, capacity blocker filtering, capacity messaging, source footprints, memory assessment, and auto-growth retry planning behind the new module.
2. Keep capacity refresh non-mutating.
3. Recompute auto-growth capacity verdicts from the same Admission Snapshot without calling scene gather or source compilation a second time.
4. Make initialize and reset consume the same preflight implementation.
5. Preserve current source/collider/force/height/seed/snippet ordering.
6. Keep initial seed admission fail-closed.
7. Preserve refreshed-device-count admission and short-commit invariant reporting from `SourcePipeline.inject_particles()`.
8. Remove source-footprint/capacity-planner knowledge from Solver Session.

**Checkpoint**

- Solver Session calls one admission interface and does not construct capacity plans.
- All authored-input rejection occurs before Replacement Cutover.

### Task 1.4 — Make Replacement Cutover explicit

Internal transaction phases:

1. capture;
2. compile/estimate;
3. validate;
4. optional preflight replan;
5. Replacement Cutover;
6. allocate;
7. configure;
8. commit seeds/snippets;
9. initial Output Preparation;
10. commit Solver Session presentation.

**Required behavior**

- Failures in phases 1–4 preserve the old solver and its state.
- Release the old solver immediately before phase 6.
- Failures in phases 6–10 release the partial replacement, clear active runtime state, preserve the actionable error, and do not resurrect the old GPU solver.
- Full swap atomicity and two-solver staging are out of scope.

### Task 1.5 — Add authoritative Blender admission acceptance

**Files**

- New `tests/blender_simulation_admission_acceptance.py`
- `tools/run_production_gate.py`
- `AGENTS.md`

**Acceptance cases**

- real successful solver admission;
- one coherent evaluated scene snapshot;
- sources, colliders, force fields, height source, snippets, and seeds installed;
- initially enabled outputs prepared before live stepping;
- capacity and memory rejection preserve the active solver;
- injected allocation/configuration/seed/output-preparation failures after cutover leave no active solver;
- transient compiler/input resources and CUDA memory are released;
- marker: `SIMULATION-ADMISSION-ACCEPTANCE-PASS`.

**Commands**

```text
python -m pytest -q tests/test_scene_bridge_capacity_initialization.py tests/test_capacity_planner.py tests/test_memory_safety_recovery.py tests/test_sourcing_pipeline.py tests/test_source_emission_lifecycle.py tests/test_particle_capacity_growth.py tests/test_solver_settings_projection.py
blender --background --factory-startup --python tests/blender_simulation_admission_acceptance.py
```

**Rollback point:** commit Phase 1 only after both lanes pass. Do not start Phase 2 while admission acceptance fails.

## 8. Phase 2 — Output Publication

### Task 2.1 — Freeze preparation, payload, and publication costs

**Files**

- `tests/test_scene_bridge_cache_playback.py`
- `tests/test_frame_output.py`
- `tests/test_viewport_pointcloud_prewarm.py`
- `tests/test_performance_hotpath_contracts.py`

**Instrument separately**

1. live payload preparation/readback;
2. cache payload preparation;
3. Output Preparation;
4. Output Publication.

**Assertions**

- one logical host-frame acquisition per live publication;
- device-side particle stride remains in use;
- no additional sync or `.numpy()` call relative to baseline;
- a publisher-only test fails if it accesses a solver, Warp array, `surface_mesh()`, host-frame acquisition, or synchronization operation;
- no new object/datablock creation on a prewarmed first write;
- `ensure_outputs`-equivalent preparation does not write geometry;
- initially disabled outputs remain unallocated;
- current property-triggered enablement remains outside frame handling;
- structural replacement for growth, shrink, and attribute-schema changes is preserved, separately counted, and never hidden;
- two-scene shared/distinct output-collection behavior is characterized before migration;
- materials/visibility refresh without inventing geometry.

### Task 2.2 — Define Publication Payload equivalence

**Atomic actions**

1. Normalize geometry, attributes, visibility facts, and surface-publication cadence before datablock mutation.
2. Preserve current live surface retention cadence.
3. Preserve device-side live particle decimation and current cache decimation behavior without applying either twice.
4. Specify current fallbacks for absent/malformed cache attributes.
5. Characterize schema transitions when named attributes appear/disappear.

**Equivalence matrix**

- particle positions and shared attributes;
- whitewater positions and shared attributes;
- surface vertices, triangles, and shared attributes when both paths publish a surface;
- material and visibility state;
- display scale/radius;
- explicit differences: missing cache fields and live retained-surface cadence.

### Task 2.3 — Create the deep Output Publication module

**Files**

- New `fluidSim_warp_blender51/blender/output_publication.py`
- `fluidSim_warp_blender51/blender/viewport.py`
- `fluidSim_warp_blender51/blender/materials.py`
- `fluidSim_warp_blender51/sim/frame_output.py`

**Atomic actions**

1. Move output collection/object lookup and current cross-scene collection behavior unchanged.
2. Move object/datablock creation, point schema/capacity, mesh/point publication, attributes, materials, radius, visibility, and removal.
3. Replace `_ViewportFrame` with an owned internal Publication Payload model.
4. Pass the already acquired `HostFrame` into live adaptation; never reacquire it during publication.
5. Route cache adaptation through the same normalized facts where data overlaps.
6. Keep Output Preparation distinct from frame publication inside the module.
7. Route initially enabled outputs through Admission preparation.
8. Route outputs enabled later through the existing safe property/timer update path.
9. Preserve existing structural replacement for growth, shrink, and schema change exactly; do not add another allocator.
10. Preserve current global collection/tag lookup and scene-link/removal behavior until the focused two-scene Blender tests authorize any separate correction.

**Checkpoint**

- All persistent simulation-output mutation has one owner.
- Ordinary callers cannot invoke private swap/allocation helpers.
- Publisher-only tests prove no solver/readback access.

### Task 2.4 — Separate overlay drawing

**Files**

- New `fluidSim_warp_blender51/blender/viewport_overlay.py`
- `fluidSim_warp_blender51/__init__.py`
- `fluidSim_warp_blender51/blender/render.py`
- `fluidSim_warp_blender51/blender/scene_bridge.py`
- `fluidSim_warp_blender51/operators.py`

**Atomic actions**

1. Move overlay registration/drawing into the adapter.
2. Migrate render visibility and all output mutation to Output Publication.
3. Update add-on registration order.
4. Migrate internal callers and tests.
5. Delete `blender/viewport.py` after migration rather than retaining a shallow adapter, unless an external compatibility contract is discovered and documented before deletion.

**Checkpoint**

- Overlay cannot mutate persistent output state.
- Output Publication does not import overlay/UI drawing.
- No active production import references deleted viewport internals.

### Task 2.5 — Real-Blender and performance acceptance

```text
python -m pytest -q tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_viewport_pointcloud_prewarm.py tests/test_performance_hotpath_contracts.py
blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py
blender --background --factory-startup --python tests/blender_attribute_viz_acceptance.py
blender --background --factory-startup --python tests/blender_cache_export_pipeline.py
FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py
```

Extend `tests/blender_viewport_2frame_gate.py` with two-scene cases covering shared and distinct output collection names, tagged-object resolution, publication isolation, visibility changes, scene unlinking, and removal. Preserve the characterized behavior; changing it requires separate defect evidence and a red Blender test.

Reject Phase 2 if instrumentation shows any additional host-frame acquisition, synchronization, device-to-host copy, Blender ID creation, depsgraph relation rebuild, or structural point-cloud replacement relative to the matching baseline case—even if average time remains below a broad threshold.

**Rollback point:** commit Output Publication and overlay separation only after focused Blender acceptance and profile comparison pass.

## 9. Phase 3 — Preview/Cache Playback and Production Cache Bake

### Task 3.1 — Freeze current playback and backend-status semantics

**Files**

- `tests/test_frame_change_catchup.py`
- `tests/test_scene_bridge_cache_playback.py`
- `tests/test_cfl_clamp_reporting.py`
- `tests/blender_handler_persistence_gate.py`

**Characterize exactly**

- live-disabled no-op;
- cache playback bypasses solver initialization and stepping;
- lazy initialization and playback-pre initialization;
- rewind reset and Pending Settings application on rewind;
- bounded catch-up and over-bound desynchronization;
- current `last_frame` updates;
- existing status text and error retention/clearing;
- current `live` side effects;
- active Solver Session ownership;
- step failure release behavior;
- render-restore one-shot behavior;
- handler re-entry refusal and teardown.

Do not add preview-lag, progress, recovery, or authority states.

### Task 3.2 — Create the deep Playback module

**Files**

- New `fluidSim_warp_blender51/blender/playback.py`
- `fluidSim_warp_blender51/blender/scene_bridge.py`
- `fluidSim_warp_blender51/blender/render.py`
- `fluidSim_warp_blender51/properties.py`

**Atomic actions**

1. Move live/cache enable transitions, cache-frame lookup, frame interpretation, playback-pre admission, re-entry state, render-restore state, and handler lifecycle.
2. Keep persistent callback names where acceptance relies on them.
3. Keep Solver Session responsible for direct `sync_scene` and direct step execution.
4. Make Blender handlers thin adapters to Playback.
5. Preserve bare-timeline-scrub fallback initialization.
6. Preserve every characterized status, error, catch-up, desynchronization, and `last_frame` behavior.

**Checkpoint**

- `scene_bridge.py` contains no timeline-gap interpretation, render-restore global, or playback handler ownership.
- Playback uses Simulation Admission and Output Publication through their established seams.

### Task 3.3 — Separate deterministic Production Cache Bake

**Files**

- New `fluidSim_warp_blender51/blender/cache_bake.py`
- `fluidSim_warp_blender51/blender/scene_bridge.py`
- `fluidSim_warp_blender51/operators.py`
- `fluidSim_warp_blender51/io/cache.py`
- `tests/blender_cache_export_pipeline.py`

**Preserve**

- sequential frame iteration;
- exact frame indices;
- temporary live disable and restoration;
- per-frame scene synchronization;
- full, non-preview-decimated frame data;
- settings records and scene key;
- metadata source `"simulation"`;
- frame and state restoration in `finally`;
- live cache metadata source `"live_simulation"` unchanged.

**Add evidence**

- two identical production bakes produce identical frame payload hashes and semantically identical metadata, excluding intentionally variable timestamps;
- changing preview stride, display attributes, retained-surface cadence, or visibility cannot change production cache hashes;
- Production Cache Bake does not call frame-change handlers or Output Publication.

### Task 3.4 — Rewire handler ownership

**Files**

- `fluidSim_warp_blender51/__init__.py`
- `fluidSim_warp_blender51/blender/playback.py`
- `fluidSim_warp_blender51/blender/render.py`
- `tests/blender_handler_persistence_gate.py`

**Rules**

- Playback solely owns `frame_change_post` and `animation_playback_pre`.
- Render keeps current `render_init` ownership.
- Registration remains deduplicated and callbacks remain `@persistent`.
- Remove handlers before Solver Session teardown.
- Add to `HANDLER_LISTS` only if a new handler list is introduced.

**Acceptance**

```text
python -m pytest -q tests/test_frame_change_catchup.py tests/test_scene_bridge_cache_playback.py tests/test_cfl_clamp_reporting.py
blender --background --factory-startup --python tests/blender_handler_persistence_gate.py
blender --background --factory-startup --python tests/blender_cache_export_pipeline.py
blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py
```

**Rollback point:** commit Playback and Cache Bake extraction only after handler persistence, exact behavior characterization, and repeat-bake determinism pass.

## 10. Phase 4 — Pending Settings

### Task 4.1 — Build an exhaustive, disjoint transition registry

**Files**

- New `tests/test_pending_settings_registry.py`
- `tests/test_phase4_pending_interaction.py`
- `tests/test_operator_panel_behavior.py`

**Atomic actions**

1. Enumerate every relevant authored simulation property.
2. Classify each intentionally as live, sync, output, surface, whitewater, reset, reinitialize, or authored-next-run.
3. Assert no accidental omission.
4. Assert no conflicting tier unless an explicit documented exception requires overlap.
5. Preserve particle-capacity special handling.
6. Freeze current reasons, blocked reasons, pending labels, IDProperty keys, timer behavior, and disposable-start behavior.

**Checkpoint**

- Adding a new property without an explicit policy decision fails the registry test.

### Task 4.2 — Create the deep Pending Settings module

**Files**

- New `fluidSim_warp_blender51/blender/pending_settings.py`
- `fluidSim_warp_blender51/blender/scene_bridge.py`
- `fluidSim_warp_blender51/blender/solver_settings.py`

**Move behind the seam**

- `SceneUpdate` and tier registries;
- per-scene interaction state;
- Applied Settings Snapshot;
- reset/reinitialize signatures;
- pending refresh and presentation caching;
- live/reset/surface/whitewater application;
- apply/revert operations;
- debounce timer;
- undo/script/depsgraph refresh;
- state cleanup.

Use Solver Settings for value projection, Simulation Admission for reset/reinitialize, and Output Publication for output refresh. Preserve current dictionary-shaped presentation until every adapter migrates; remove temporary scene-bridge wrappers before final completion.

### Task 4.3 — Convert UI and playback callers into adapters

**Files**

- `fluidSim_warp_blender51/properties.py`
- `fluidSim_warp_blender51/operators.py`
- `fluidSim_warp_blender51/panels.py`
- `fluidSim_warp_blender51/blender/viewport_overlay.py`
- `fluidSim_warp_blender51/blender/playback.py`
- `fluidSim_synth/ui_contract.py`

**Atomic actions**

1. Property callbacks forward declared update intent without reimplementing policy.
2. Operators ask Pending Settings for current eligibility and invoke transitions.
3. Panels/overlay render cached presentation facts only.
4. Playback delegates pending-on-rewind behavior.
5. Preserve property names, operator identifiers, labels, panel placement, status strings, and current visible behavior.
6. Ensure draw paths do not inspect geometry or recompute signatures.

### Task 4.4 — Move pending-state handlers to their owner

- Pending Settings owns `depsgraph_update_post` and `undo_post`.
- Coordinate `load_pre` so state clears and Solver Session releases exactly once.
- Keep callbacks persistent and registration-order independent.

**Acceptance**

```text
python -m pytest -q tests/test_phase4_pending_interaction.py tests/test_pending_settings_registry.py tests/test_operator_panel_behavior.py
blender --background --factory-startup --python tests/blender_phase4_live_interaction_acceptance.py
blender --background --factory-startup --python tests/blender_handler_persistence_gate.py
```

**Rollback point:** commit Pending Settings only after Phase 4 and handler persistence acceptance pass.

## 11. Phase 5 — deletion test, documentation, and production validation

### Task 5.1 — Remove shallow remnants

**Atomic actions**

1. Remove migrated functions, globals, field sets, imports, and compatibility wrappers from `scene_bridge.py`.
2. Keep only Solver Session ownership, direct synchronization/stepping, frame statistics, teardown, and errors.
3. Remove temporary old-path wrappers unless a documented external compatibility requirement exists.
4. Document dependency direction in module headers.
5. Verify no circular import is required during registration, handlers, Blender background runs, or system-Python tests.

**Deletion test**

Deleting any proposed deep module must spread its complexity back across multiple adapters. If deleting a new file merely changes import paths, that file is shallow and must be folded into its natural owner.

### Task 5.2 — Update truthful documentation

**Files**

- `CONTEXT.md`
- `fluidSim_warp_blender51/docs/technical_report.md`
- `fluidSim_warp_blender51/docs/user_guide.md`
- `AGENTS.md`

**Required claims**

- module ownership and dependency direction;
- Preview Playback is visual feedback;
- Production Cache Bake is deterministic and separate;
- current playback/backend status remains unchanged;
- focused acceptance commands;
- review artifacts do not certify core behavior.

### Task 5.3 — Focused deep-module regression lane

```text
python -m pytest -q tests/test_sourcing_pipeline.py tests/test_mesh_source_pipeline.py tests/test_pressure_projection_regressions.py tests/test_memory_safety_recovery.py tests/test_performance_hotpath_contracts.py tests/test_snippet_library.py
```

### Task 5.4 — Cache/live, sparse-grid, truthfulness, and viscosity lane

```text
python -m pytest -q tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_snippet_library.py tests/test_production_gate_truthfulness.py tests/test_sparse_leaf_grid_contract.py tests/test_viscosity_active_work_regressions.py
```

### Task 5.5 — Full Python and standalone Blender acceptance

```text
python -m pytest -q
blender --background --factory-startup --python tests/blender_simulation_admission_acceptance.py
blender --background --factory-startup --python tests/blender_cache_export_pipeline.py
blender --background --factory-startup --python tests/blender_memory_safety_gate.py
blender --background --factory-startup --python tests/blender_handler_persistence_gate.py
blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py
blender --background --factory-startup --python tests/blender_background_acceptance.py
blender --background --factory-startup --python tests/blender_attribute_viz_acceptance.py
blender --background --factory-startup --python tests/blender_phase4_live_interaction_acceptance.py
```

### Task 5.6 — Production gate and policy-drift validation

```text
python tools/run_production_gate.py
python tools/run_production_gate.py --include-gui
python tools/validate_policy_drift_reports.py
```

Run policy-drift validation only after a GUI-inclusive pass and intentional report refresh. Re-read implementation-start ownership before refreshing reports; if another process owns or changes policy-validator/report paths, stop and reconcile before writing.

### Completion criteria

- Default headless production gate passes.
- GUI-inclusive production gate passes.
- Policy-drift validation passes against the validated HEAD and checked-in artifacts.
- Repeat Production Cache Bake hashes match.
- No added publication sync/copy/host acquisition/ID allocation relative to baseline.
- Exact current playback/backend-status characterization remains green.
- Implementation-start dirty-file hashes remain unchanged unless the user explicitly reassigns those files.
- `scene_bridge.py` is materially smaller and no longer owns the four deepened concerns.

## 12. Dependencies and critical path

```text
Vocabulary + dirty-tree protection + characterization
  → Solver Settings projection
  → Admission Snapshot + preflight + Replacement Cutover
  → authoritative Blender admission acceptance
  → publication cost/equivalence characterization
  → Output Preparation + Output Publication + overlay separation
  → Blender/profile publication acceptance
  → playback/status characterization
  → Playback extraction + deterministic Cache Bake
  → handler/cache acceptance
  → exhaustive Pending Settings registry
  → Pending Settings extraction + UI adapters
  → Phase 4/handler acceptance
  → deletion test + documentation
  → full regression + production gate + policy validation
```

No later phase begins while the preceding rollback point is failing.

## 13. Parallel opportunities

- Phase 0 vocabulary and characterization tests can proceed in parallel after dirty-file hashes are recorded.
- During Phase 1, solver-settings projection tests and Blender admission acceptance scaffolding can proceed in parallel; production writes to `scene_bridge.py` stay serialized.
- After the Publication Payload facts are frozen, live/cache equivalence tests and overlay extraction can proceed in parallel.
- Cache Bake determinism tests can be prepared alongside Playback characterization, but production integration is serialized because both touch `scene_bridge.py` and handlers.
- Pending registry tests and UI adapter preparations can proceed in parallel after the registry vocabulary is fixed.
- Documentation can be drafted during each phase but finalized only after runtime evidence exists.

Do not parallelize overlapping writes to:

- `fluidSim_warp_blender51/blender/scene_bridge.py`
- `fluidSim_warp_blender51/operators.py`
- `fluidSim_warp_blender51/properties.py`
- `fluidSim_warp_blender51/panels.py`
- `fluidSim_warp_blender51/__init__.py`
- `tools/run_production_gate.py`
- handler registration code.

## 14. Main files

### New production files

- `fluidSim_warp_blender51/blender/solver_settings.py`
- `fluidSim_warp_blender51/blender/simulation_admission.py`
- `fluidSim_warp_blender51/blender/output_publication.py`
- `fluidSim_warp_blender51/blender/viewport_overlay.py`
- `fluidSim_warp_blender51/blender/playback.py`
- `fluidSim_warp_blender51/blender/cache_bake.py`
- `fluidSim_warp_blender51/blender/pending_settings.py`

### New test files

- `tests/test_solver_settings_projection.py`
- `tests/test_pending_settings_registry.py`
- `tests/blender_simulation_admission_acceptance.py`

### Principal modified files

- `fluidSim_warp_blender51/blender/scene_bridge.py`
- `fluidSim_warp_blender51/blender/scene_inputs.py`
- `fluidSim_warp_blender51/blender/sources.py`
- `fluidSim_warp_blender51/blender/materials.py`
- `fluidSim_warp_blender51/blender/render.py`
- `fluidSim_warp_blender51/blender/viewport.py` (migrate, then delete if no external contract)
- `fluidSim_warp_blender51/sim/frame_output.py`
- `fluidSim_warp_blender51/io/cache.py`
- `fluidSim_warp_blender51/__init__.py`
- `fluidSim_warp_blender51/properties.py`
- `fluidSim_warp_blender51/operators.py`
- `fluidSim_warp_blender51/panels.py`
- `fluidSim_synth/ui_contract.py`
- `tools/run_production_gate.py`
- focused Python and Blender acceptance tests listed in each phase.

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Concurrent commits or dirty user work invalidate the plan baseline | Record HEAD/status/hashes at implementation start and every phase; stop and revalidate overlapping paths; no reset/stash/clean/broad stage |
| Admission mixes Blender evaluations | One Admission Snapshot; no second gather before cutover |
| Source compiler/input resources leak | `finally` cleanup and fault injection |
| Full swap atomicity is accidentally implied | Explicit Replacement Cutover; no two-solver staging |
| Output refactor adds readback or host acquisition | Separate payload preparation from publisher; instrument calls |
| Output refactor adds ID/depsgraph churn | Baseline each structural case; initial/property preparation outside handler; preserve existing growth, shrink, and schema-change fallbacks only |
| Live/cache equivalence hides real cadence differences | Equivalence matrix with explicit retained-surface and missing-data exceptions |
| Production cache consumes preview data | Repeat-bake hashes; prohibit Output Publication dependency |
| Status behavior drifts during extraction | Exact transition characterization before Phase 3 |
| Pending registry misses a property | Exhaustive/disjoint registry test with explicit exceptions |
| Two scenes share output collection state unexpectedly | Characterize shared/distinct collection names, publication, visibility, unlinking, and removal in focused Blender tests; preserve results unless defect evidence + red test authorizes change |
| Handler move drops callbacks or double-releases | Persistence gate, save/reopen cycle, registration-order tests |
| New files become shallow wrappers | Final deletion test; fold shallow modules into natural owner |
| Import cycles appear | One-way dependency rule, owned values, narrow adapters/lazy imports |
| Experimental worktrees contaminate design | Explicit exclusion; active main checkout only |

## 16. Assumptions and non-goals

- Solver algorithms, source semantics, pressure projection, viscosity, ST-FLIP, surface reconstruction, and sparse-grid storage are unchanged.
- Property names, operator identifiers, panel placement, cache format, current status strings, and current playback behavior remain compatible.
- This plan does not add asynchronous preview recovery, lag/progress state, a new authority status, or a new backend-status model.
- This plan does not eliminate Blender 5.1 structural point-cloud replacement for capacity growth, capacity shrink, or attribute-schema change; it prevents regression and centralizes ownership.
- Current cross-scene output-collection behavior is preserved unless separately proven defective.
- No archived experiment is reactivated, integrated, or treated as pending work.
- Any newly discovered defect is deferred unless it satisfies the targeted-change evidence rule.
