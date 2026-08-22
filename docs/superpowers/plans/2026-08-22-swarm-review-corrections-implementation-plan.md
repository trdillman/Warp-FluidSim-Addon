# FluidSim Swarm Review Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:test-driven-development`. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Correct every independently validated swarm-review claim without
disturbing pre-existing dirty work and prove the resulting add-on behavior.

**Architecture:** Extend established deep owners instead of adding a workflow
manager. Scene lifecycle remains in `scene_bridge.py`, playback policy in
`playback.py`, production-bake ownership in `production_bake.py`, Blender cache
identity in a narrow new adapter, and pure serialization/hash behavior in `io`.

**Tech Stack:** Python 3, NumPy, Blender 5.1 Python API, Warp CUDA, pytest,
Blender background acceptance scripts.

**Spec:** `docs/superpowers/specs/2026-08-22-swarm-review-corrections-design.md`

## Global constraints

- Preserve the captured pre-loop hashes in
  `reviews/swarm-review-2026-08-22-luna-terra/AUTONOMOUS_CORRECTIONS_CHECKPOINT.md`.
- Never reset, stash, clean, or overwrite a touched dirty file; patch around the
  current live content.
- No task commits: several target files contain pre-existing user work that
  cannot be safely separated by broad staging. Use focused diffs and checkpoints.
- For every behavior change, write the test first and run it to observe the
  expected failure before editing production code.
- Do not weaken existing tests or replace runtime behavior with source-grep proof.
- Do not claim FUNCTION execution support or literal multi-file crash atomicity.
- Exclude archived experimental checkouts from all implementation and proof.

---

### Task 1: Isolate registry-mutating tests (TEST-004)

**Files:**
- Modify: `tests/test_addon_core_regressions.py`

**Interfaces:**
- Produces: an autouse/local fixture that snapshots and restores
  `KERNEL_REGISTRY` and `SOURCE_ALIASES` around registry-mutating tests.

- [ ] **Step 1: Write the failing cleanup assertion**

Add a fixture-backed test whose literal expectation is that both dictionaries
equal their pre-test snapshots after duplicate and multi-name registration
cases. The mutation caught is “removing only the kernel entry leaves a dangling
source alias.”

```python
before_registry = dict(KERNEL_REGISTRY)
before_aliases = dict(SOURCE_ALIASES)
run_duplicate_registration_case()
assert KERNEL_REGISTRY == before_registry
assert SOURCE_ALIASES == before_aliases
```

- [ ] **Step 2: Verify RED**

Run:
`python -m pytest -q tests/test_addon_core_regressions.py -k 'duplicate_kernel or multi_name'`

Expected: failure showing an extra `paradigmRegression...` alias.

- [ ] **Step 3: Implement minimal test cleanup**

Use a pytest fixture that snapshots both maps and restores them with
`clear(); update(snapshot)` in `finally`. Remove one-off cleanup that restores
only part of the global state.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command and expect all selected tests to pass.

---

### Task 2: Fail closed at pure data boundaries (TYPE-001, TEST-001/002/003)

**Files:**
- Modify: `fluidSim_warp_blender51/sim/source_regions.py`
- Modify: `fluidSim_warp_blender51/io/cache.py`
- Modify: `fluidSim_warp_blender51/diagnostics/benchmark.py`
- Modify: `fluidSim_warp_blender51/io/export.py`
- Modify: `fluidSim_warp_blender51/operators.py`
- Modify: `tests/test_source_region_compilation.py`
- Modify: `tests/test_point_source_pipeline.py`
- Modify: `tests/test_cache_io.py`
- Modify: `tests/test_addon_core_regressions.py`
- Modify: `tests/test_io_path_policy.py`

**Interfaces:**
- Produces: `_owned_array(..., shape_tail=())` with exact-rank validation.
- Produces: collision-preserving metadata normalization before file creation.
- Produces: structured benchmark `result="fail"` for malformed numeric fields.
- Produces: finite OBJ/PLY validation before opening an output path.

- [ ] **Step 1: Write exact-shape failing tests**

Parameterize malformed `(3,)`, `(1, 3, 2)`, `(N, 1)`, and wrong-tail arrays.
Assert `ValueError`; retain literal success expectations for `(0, 3)`, `(N, 3)`,
and `(N,)` scalar attributes.

- [ ] **Step 2: Verify shape RED**

Run:
`python -m pytest -q tests/test_source_region_compilation.py tests/test_point_source_pipeline.py -k 'shape or rank or malformed'`

Expected: malformed arrays are accepted/reshaped instead of raising.

- [ ] **Step 3: Implement exact-rank ownership**

Validate `array.ndim == 1 + len(shape_tail)` and exact trailing dimensions;
for scalar point attributes require rank one. Copy as C-contiguous, set
read-only, and do not reshape.

- [ ] **Step 4: Write metadata/benchmark/export failing tests**

Add literal cases:

```python
with pytest.raises(ValueError, match="Duplicate settings record for frame 1"):
    write_sequence_metadata(path, 1, 1, settings_records={1: first, "1": second})
assert not (path / "fluidsim_sequence.json").exists()

assert compare_benchmark(record(timeStepAverage=None), baseline)["result"] == "fail"
assert compare_benchmark(record(timeStepAverage=float("nan")), baseline)["result"] == "fail"

with pytest.raises(ValueError, match="finite"):
    export_points_ply(np.array([[np.nan, 0, 0]], np.float32), target)
assert not target.exists()
```

Cover OBJ vertices and PLY position/velocity NaN, positive infinity, and
negative infinity. Add operator coverage that malformed baseline JSON returns
`CANCELLED` with status text rather than raising.

- [ ] **Step 5: Verify boundary RED**

Run:
`python -m pytest -q tests/test_cache_io.py tests/test_addon_core_regressions.py tests/test_io_path_policy.py -k 'duplicate_normalized or malformed_benchmark or non_finite'`

Expected: metadata silently keeps one record, benchmark conversion raises, and
non-finite export writes a file.

- [ ] **Step 6: Implement minimal fail-closed behavior**

Preserve raw metadata keys through range filtering; validate finite real JSON
numbers (not bool) before benchmark arithmetic; return structured failure;
catch baseline decode/read/schema failures in the operator; convert export
arrays to output dtype and validate finiteness before opening files.

- [ ] **Step 7: Verify GREEN**

Run both focused commands from Steps 2 and 5 and expect all selected tests to pass.

---

### Task 3: Restore custom snippet contract fidelity (TYPE-002/003)

**Files:**
- Modify: `fluidSim_warp_blender51/properties.py`
- Modify: `fluidSim_warp_blender51/blender/snippets.py`
- Modify: `fluidSim_warp_blender51/sim/custom.py`
- Modify: `tests/test_snippet_library.py`
- Modify: `tests/test_operator_panel_behavior.py`

**Interfaces:**
- Produces: `CustomArgument.field_permission: FieldPermission`.
- Produces: `CustomSnippet.snippet_type: CustomKernelType` defaulting KERNEL.
- Produces: conservative permission-effect validation before code generation.
- Produces: admission-time durable rejection for FUNCTION snippets.

- [ ] **Step 1: Write permission propagation/effect tests**

Add real validator tests for direct subscripts and atomics:

```python
assert has_error(validate(read_arg, "velocity[tid] = 0.0"), "READ")
assert has_error(validate(write_arg, "x = velocity[tid]"), "WRITE")
assert has_error(validate(write_arg, "velocity[tid] += 1.0"), "WRITE")
assert not errors(validate(read_write_arg, "velocity[tid] += 1.0"))
```

Also round-trip permission through PropertyGroup adapter/preset serialization.

- [ ] **Step 2: Verify permission RED**

Run:
`python -m pytest -q tests/test_snippet_library.py -k 'permission or read_field or write_field'`

Expected: fields have no permission metadata and forbidden effects validate.

- [ ] **Step 3: Implement conservative permission analysis**

Persist enum values, convert to `FieldPermission`, and walk the validated AST.
Reject bound-field aliases, unknown mutation calls, READ stores/atomics, WRITE
loads/augmented assignments, and unmodelled effects. Keep existing sandbox rules.

- [ ] **Step 4: Write FUNCTION fail-closed tests**

Assert legacy snippets default to KERNEL; FUNCTION survives adapter/preset
round-trip, produces a durable diagnostic during build/admission, and never
reaches `CompiledSnippet` or manager launch.

- [ ] **Step 5: Verify FUNCTION RED**

Run:
`python -m pytest -q tests/test_snippet_library.py tests/test_operator_panel_behavior.py -k 'function_snippet or snippet_type'`

Expected: type is absent or silently narrowed to KERNEL.

- [ ] **Step 6: Implement typed admission**

Add the RNA/DTO field with KERNEL default. Reject FUNCTION in the adapter/build
boundary, set `last_error`/validity state, and leave compiler/dispatcher kernel-only.

- [ ] **Step 7: Verify GREEN**

Run both focused commands and the complete two-file suite.

---

### Task 4: Split owner-scoped release from forced teardown (ARCH-001)

**Files:**
- Modify: `fluidSim_warp_blender51/blender/scene_bridge.py`
- Modify: `fluidSim_warp_blender51/blender/simulation_admission.py`
- Modify: `fluidSim_warp_blender51/blender/production_bake.py`
- Modify: `fluidSim_warp_blender51/__init__.py`
- Modify: `tests/test_scene_bridge_cache_playback.py`
- Modify: `tests/test_simulation_admission.py`

**Interfaces:**
- Produces: `Session.release(scene) -> bool` owner-scoped.
- Produces: `Session.force_release() -> bool` or module-level equivalent for
  explicit global lifecycle teardown.

- [ ] **Step 1: Write owner/force failing tests**

Assert foreign-scene release returns false and preserves solver identity;
owner release clears it; forced release clears it regardless of owner. Cover
load/unregister and admission/bake fatal cleanup call-site intent.

- [ ] **Step 2: Verify RED**

Run:
`python -m pytest -q tests/test_scene_bridge_cache_playback.py tests/test_simulation_admission.py -k 'release or foreign_owner or teardown'`

Expected: foreign release clears the solver or forced API is absent.

- [ ] **Step 3: Implement and migrate call sites**

Keep `_clear_runtime()` private. Owner-scoped paths call `release(scene)`;
load-pre, unregister, and proven fatal/global cleanup call `force_release()`.
Do not infer force from `scene is None` at adapter call sites.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command and expect all selected tests to pass.

---

### Task 5: Guard playback/operators and isolate cursors (UI-001/002/003/004)

**Files:**
- Modify: `fluidSim_warp_blender51/blender/playback.py`
- Modify: `fluidSim_warp_blender51/operators.py`
- Modify: `fluidSim_warp_blender51/panels.py`
- Modify: `tests/test_scene_bridge_cache_playback.py`
- Modify: `tests/test_operator_poll_and_guards.py`
- Modify: `tests/test_operator_panel_behavior.py`

**Interfaces:**
- Produces: per-scene cache cursor storage owned and cleared by `playback.py`.
- Produces: owner-aware session operator guard usable by poll and execute.
- Produces: Release poll independent of runtime readiness.

- [ ] **Step 1: Write cross-scene and cursor failing tests**

Create scene A live-owner and scene B cache contexts. Assert B cannot activate
cache playback over A, B cannot release/apply snippets/export A, and cache frame
events never change `session.last_frame` for A.

- [ ] **Step 2: Verify RED**

Run:
`python -m pytest -q tests/test_scene_bridge_cache_playback.py tests/test_operator_poll_and_guards.py -k 'foreign or owner or cursor or cache_playback'`

- [ ] **Step 3: Implement ownership and cursor isolation**

Use owner-scoped release, restore rejected RNA transitions under the transition
guard, add execute-time defenses, and store cache cursors by stable scene
identity. Clear cursor storage on load/unregister.

- [ ] **Step 4: Write runtime-failed Release test**

With an active owned session and failed runtime probe, assert Release poll is
true when bake idle and false when bake active.

- [ ] **Step 5: Verify Release RED, implement dedicated poll, verify GREEN**

Run:
`python -m pytest -q tests/test_operator_poll_and_guards.py tests/test_operator_panel_behavior.py -k 'release'`

Override Release poll without `_runtime_poll`; keep ownership and bake-idle checks.
Run both focused commands and expect pass.

---

### Task 6: Protect active production cache paths (IO-001)

**Files:**
- Modify: `fluidSim_warp_blender51/blender/production_bake.py`
- Modify: `fluidSim_warp_blender51/blender/cache_manager.py`
- Modify: `fluidSim_warp_blender51/operators.py`
- Modify: `fluidSim_warp_blender51/panels.py`
- Modify: `tests/test_scene_bridge_cache_playback.py`
- Modify: `tests/test_operator_panel_behavior.py`

**Interfaces:**
- Produces: `production_bake.cache_path_is_active(path) -> bool` using resolved
  backend `base_dir` from the existing active-controller registry.
- Produces: shared actionable guard for cache mutators.

- [ ] **Step 1: Write path-lease failing tests**

Register a RUNNING controller for cache A. Assert Free, Move, Rename, and Remove
Orphans reject resolved aliases of A in poll and direct execute; cache B remains
mutable; terminal controller state releases the guard.

- [ ] **Step 2: Verify RED**

Run:
`python -m pytest -q tests/test_scene_bridge_cache_playback.py tests/test_operator_panel_behavior.py -k 'cache_path or free_cache or move_cache or rename_cache or orphan'`

- [ ] **Step 3: Implement one authoritative path query**

Inspect the existing registry; compare `Path.resolve()` values; do not add a
second lease map. Apply the guard at operator poll/execute and the deep mutator
seam so scripts fail with the same actionable error.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command and expect pass.

---

### Task 7: Stage and rollback current-frame exports (IO-002)

**Files:**
- Modify: `fluidSim_warp_blender51/io/export.py`
- Modify: `tests/test_io_path_policy.py`
- Modify: `tests/blender_background_acceptance.py`

**Interfaces:**
- Produces: an internal bundle transaction that stages all requested artifacts,
  backs up existing finals, publishes, and rolls back on failure.

- [ ] **Step 1: Write late-failure preservation tests**

Seed literal prior OBJ/PLY files, inject failure on the final sibling publish,
and assert prior bytes are restored, no new partial final exists, and no
staging/backup files remain.

- [ ] **Step 2: Verify RED**

Run:
`python -m pytest -q tests/test_io_path_policy.py -k 'bundle or rollback or late_failure'`

Expected: earlier finals are replaced/created before the injected failure.

- [ ] **Step 3: Implement bounded transaction**

Capture/validate host data first. Write same-directory unique staging files,
flush/fsync, backup finals, replace staged files, and restore backups on any
publication error. Clean only transaction-owned paths in `finally`.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command plus full `tests/test_io_path_policy.py`.

---

### Task 8: Restore cache dependency direction (ARCH-002)

**Files:**
- Create: `fluidSim_warp_blender51/blender/cache_identity.py`
- Modify: `fluidSim_warp_blender51/io/cache.py`
- Modify: `fluidSim_warp_blender51/blender/simulation_admission.py`
- Modify: `fluidSim_warp_blender51/blender/production_bake.py`
- Modify: `fluidSim_warp_blender51/blender/playback.py`
- Create: `tests/test_io_dependency_contract.py`
- Modify: `tests/test_scene_bridge_cache_playback.py`

**Interfaces:**
- Produces: Blender adapter functions `scene_key(scene)`,
  `source_signature(scene)`, and `signature_bundle(scene)`.
- Produces: pure IO hash helpers consuming plain settings/source records.

- [ ] **Step 1: Write dependency and behavior failing tests**

Import `fluidSim_warp_blender51.io.cache` with Blender modules unavailable and
assert success. Exercise the moved adapter with literal scene/settings/source
fixtures and assert signatures match pre-move values.

- [ ] **Step 2: Verify RED**

Run:
`python -m pytest -q tests/test_io_dependency_contract.py tests/test_scene_bridge_cache_playback.py -k 'cache_identity or io_dependency or signature'`

- [ ] **Step 3: Extract adapter and migrate callers**

Move Blender imports/scene traversal to `blender/cache_identity.py`. Keep pure
digest/serialization helpers in `io/cache.py`; update admission, playback, and
production-bake imports. Do not move identity into `scene_bridge.py`.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command and complete cache/playback focused suites.

---

### Task 9: Make pending/view/snippet UI truthful (UI-005/006/007)

**Files:**
- Modify: `fluidSim_warp_blender51/blender/pending_settings.py`
- Modify: `fluidSim_warp_blender51/panels.py`
- Modify: `fluidSim_warp_blender51/properties.py`
- Modify: `fluidSim_warp_blender51/operators.py`
- Modify: `tests/test_operator_panel_behavior.py`
- Modify: `tests/test_phase4_pending_interaction.py`
- Modify: `tests/blender_background_acceptance.py`

**Interfaces:**
- Produces: pending presentation with authoritative action-specific copy.
- Produces: `HIDDEN` liquid view mapping to both output toggles false.
- Produces: snippet mutation polls plus execute-time cancellation.

- [ ] **Step 1: Write pending/view/no-op failing tests**

Assert reset-tier presentation contains reset (not restart/reinitialize),
reinitialize-tier presentation names reinitialize, both false visibility maps
to HIDDEN and round-trips, and invalid snippet mutations poll false/directly
return `CANCELLED` without state changes.

- [ ] **Step 2: Verify RED**

Run:
`python -m pytest -q tests/test_operator_panel_behavior.py tests/test_phase4_pending_interaction.py -k 'pending or liquid_view or snippet'`

- [ ] **Step 3: Implement authoritative presentation**

Have `pending_settings` provide action-specific labels/details consumed by the
panel. Add HIDDEN enum/callback mapping. Add actionable polls and execute-time
guards for remove snippet/add argument/remove argument.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command and the full two-file suite.

---

### Task 10: Integration and final proof

**Files:**
- Update: `reviews/swarm-review-2026-08-22-luna-terra/AUTONOMOUS_CORRECTIONS_CHECKPOINT.md`
- Update: `reviews/swarm-review-2026-08-22-luna-terra/CONSOLIDATED_CODEBASE_REVIEW.md`
- Update: `fluidSim_warp_blender51/CHANGELOG.md` only if the existing user edit
  can be extended without overwriting unrelated content.

**Interfaces:**
- Produces: current evidence mapping every claim to code, regression, and gate.

- [ ] **Step 1: Run focused pure-Python gates**

```powershell
python -m pytest -q tests/test_source_region_compilation.py tests/test_point_source_pipeline.py tests/test_mesh_source_pipeline.py tests/test_cache_io.py tests/test_addon_core_regressions.py tests/test_io_path_policy.py
python -m pytest -q tests/test_snippet_library.py tests/test_operator_panel_behavior.py
python -m pytest -q tests/test_scene_bridge_cache_playback.py tests/test_operator_poll_and_guards.py tests/test_phase4_pending_interaction.py tests/test_pending_settings_registry.py tests/test_simulation_admission.py
```

- [ ] **Step 2: Run Blender background gates**

```powershell
blender --background --factory-startup --python tests/blender_cache_export_pipeline.py
blender --background --factory-startup --python tests/blender_handler_persistence_gate.py
blender --background --factory-startup --python tests/blender_background_acceptance.py
```

- [ ] **Step 3: Run broad regression**

```powershell
python -m pytest -q
python tools/run_production_gate.py
```

Run `--include-gui` only when the owned Blender process can satisfy the project
add-on/runtime proof without disturbing unrelated profile state.

- [ ] **Step 4: Audit diff and pre-existing work**

Compare final hashes/status to the checkpoint; explain every intended overlap;
confirm demos/images and unrelated dirty files are unchanged by this loop.

- [ ] **Step 5: Independent review**

Dispatch a fresh read-only deep reviewer over the spec, plan, live diff, RED/GREEN
receipts, runtime outputs, and all 18 claim mappings. Completion requires PASS;
PASS_WITH_FIXES returns to the relevant task and repeats its gates.

## Self-review

- Spec coverage:

| Claim | Task | Closure |
|---|---:|---|
| ARCH-001 | 4 | Owner-scoped release plus explicit forced teardown. |
| ARCH-002 | 8 | Blender cache-identity adapter; pure `io.cache`. |
| TYPE-001 | 2 | Exact source-plan array ranks/shapes. |
| TYPE-002 | 3 | Persisted field permission plus AST effect enforcement. |
| TYPE-003 | 3 | Persisted snippet type and fail-closed FUNCTION admission. |
| IO-001 | 6 | Active production-bake resolved-path guard. |
| IO-002 | 7 | Prevalidated staged bundle with rollback. |
| UI-001 | 4-5 | Safe core release plus rejected foreign cache transition. |
| UI-002 | 4-5 | Shared owner guard and execute-time defenses. |
| UI-003 | 5 | Release poll independent of runtime readiness. |
| UI-004 | 5 | Per-scene cache cursor isolation. |
| UI-005 | 9 | Action-specific pending presentation. |
| UI-006 | 9 | Explicit HIDDEN liquid view state. |
| UI-007 | 9 | Snippet mutation poll/execute truthfulness. |
| TEST-001 | 2 | Collision-preserving legacy metadata writer. |
| TEST-002 | 2 | Structured malformed benchmark handling. |
| TEST-003 | 2 | Finite OBJ/PLY boundary validation. |
| TEST-004 | 1 | Complete registry/alias test cleanup. |

Shared ownership IDs map to Tasks 4-5 without duplicate core implementations.
- Placeholder scan: no TBD/TODO or unspecified error-handling steps remain.
- Type consistency: owner release, force release, cache path query, cache identity
  adapter, permission/type fields, and HIDDEN view state are defined once and
  consumed by later tasks under the same names.
- Scope: no new workflow manager, function compiler, or generation-directory
  export subsystem is introduced.
