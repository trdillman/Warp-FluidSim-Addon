# Phase 4 — FILE GATE A: Initial-Review Follow-Up Independent Review (2026-07-27)

**Gate required by:** `reviews/archive/phases/PHASE_4_LIVE_INTERACTION_PLAN_2026-07-27.md` → "FILE GATE A — Initial review follow-up"
**Findings under disposition:** `reviews/archive/phases/PHASE_4_INITIAL_INDEPENDENT_REVIEW_2026-07-27.md` (`S1–S6`, `T1–T6`)
**Worktree reviewed:** `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_phase4_interaction`
(branch `phase4/live-property-interaction`, HEAD `695fad9`, Phase-4 work uncommitted in the working tree)
**Original Phase-3/baseline worktree:** `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port` (branch `master`, HEAD `451f483`)
**Review method:** fresh read of the post-fix source and tests, `git diff` of the Phase-4 worktree against its own HEAD, independent re-run of the focused and full non-Blender suites, independent registry cross-check and bounded reconcile timing. No implementation or test file was modified by this review.

**Verdict: PASS.** No blocker and no high finding. All five non-deferrable correctness findings (`S1`, `S2`, `S3`, `T1`, `T2`) are fixed with real behavioral tests. One **new Medium performance** finding (`N1`) is recorded; it is not a correctness defect and does not block P2, but it must be closed before P4 performance acceptance.

All paths below are relative to the Phase-4 worktree unless stated otherwise.

---

## 1. Finding-by-finding disposition

| ID | Original severity | Disposition | Primary evidence |
|---|---|---|---|
| S1 | Medium (correctness) | **Fixed** | `scene_bridge.py:845-858`, `:810-843`, `:986-989`, `:180-182` |
| S2 | Medium (correctness) | **Fixed** | `scene_bridge.py:502-503`, `:830-835`; `sim/solver.py:972-1038` |
| S3 | Medium (correctness) | **Fixed** | `scene_bridge.py:754-765`, `:298`, `:342`; `operators.py:205` |
| S4 | Low | **Fixed** | `scene_bridge.py:423`, `:967-981` |
| S5 | Low | **Fixed** | `scene_bridge.py:986-989` (no status write on success) |
| S6 | Note | **Deferred (allowed)** | `scene_bridge.py:917` — unchanged pre-existing behavior, owned by P2 |
| T1 | Medium (standards) | **Fixed** | `tests/test_operator_panel_behavior.py:665-710` |
| T2 | Medium (standards) | **Fixed** | `tests/test_operator_panel_behavior.py:598-618` |
| T3 | Low | **Fixed** | regex assertion removed; AST introspection at `tests/test_operator_panel_behavior.py:669-685` |
| T4 | Low | **Fixed** | `scene_bridge.py:526-575` (memoized), measured below |
| T5 | Low | **Fixed** | `scene_bridge.py:860-870` (single union pass) |
| T6 | Note | **Deferred (allowed)** | `reviews/` still untracked and not gitignored |

### S1 — LIVE error containment — FIXED

- A single bounded boundary now exists: `_apply_live_settings_safely` (`fluidSim_warp_blender51/blender/scene_bridge.py:845-858`) catches any exception, calls `sess.set_error(scene, ...)`, writes `"Live settings error: ..."` into `scene.fluidsim.status`, tags a redraw, and returns `False` instead of propagating.
- Both authoritative callers use it: the RNA update callback path `request_scene_update` → `SceneUpdate.LIVE` (`scene_bridge.py:986-989`) and the per-step reconcile `Session.prepare_step` (`scene_bridge.py:180-182`) invoked from `step_frame` (`scene_bridge.py:757`).
- The apply itself is now **atomic**: `_apply_realtime_solver_settings` snapshots every target it may write (`scene_bridge.py:820-826`, including `gravity` and `_st_reference_mass_cached`) and restores them on any exception before re-raising (`scene_bridge.py:837-843`). I verified independently that the snapshot set covers all 57 live targets and that all 57 exist on `SolverSettings` (script output in §5).
- Playback is no longer killed by a bad hot value: `handle_frame_change` still steps after a failed reconcile, keeping `fs.live` true and the session alive — asserted at `tests/test_scene_bridge_cache_playback.py:703-727` (`stepped == [True]`, `scene.fluidsim.live is True`, `sess.solver is solver`, previous enum value preserved, status starts with `Live settings error:`).
- Callback path asserted at `tests/test_operator_panel_behavior.py:546-567` (bad enum leaves `sample_type`/`flip_ratio` untouched, session retained, error recorded and visible).

### S2 — ST-FLIP derived reference mass — FIXED

- `particles_per_voxel` is reclassified into an explicit `LIVE_REFRESH` tier: `_LIVE_REFRESH_SETTING_FIELDS = frozenset({"particles_per_voxel"})` (`scene_bridge.py:502`), unioned into `_LIVE_SETTING_FIELDS` (`:503`).
- The recalibration hook fires only on a real change and only in the automatic-mass ST configuration: `scene_bridge.py:830-835` (`old_ppv != new_ppv and spacetime_enabled and spacetime_reference_mass <= 0.0`), assigning `solver._st_reference_mass_cached = solver._calibrate_st_reference_mass()`.
- The target is the real derived-state owner: `sim/solver.py:972` defines `_calibrate_st_reference_mass`, it is the same function used by `Solver.__init__` (`sim/solver.py:664-666`) and `Solver.reset` (`:729-731`), and `_st_reference_mass()` reads exactly that cache (`sim/solver.py:1030-1038`).
- Invariant test: `tests/test_operator_panel_behavior.py:569-596` proves the cached reference mass moves with the live density control (`4.0 → 24.0`) while frame-statistics status is preserved.
- Explicit `spacetime_reference_mass`/`spacetime_enabled`/`spacetime_seed` remain reset-tier (`scene_bridge.py:505-508`), matching the plan's instruction to keep restart semantics for the manual value.
- Residual: see `N1` (cost of the recalibration) and `N2` (recalibration is unit-tested against a stub callable, not against a real `FluidSolver`; real-solver evidence is P4 runtime work).

### S3 — playback / manual / export divergence — FIXED

- One pre-step seam: `step_frame` (`scene_bridge.py:754-765`) calls `sess.prepare_step(scene)` before `sess.step(...)` and surfaces a contained failure into status.
- All three authoritative entry points route through it:
  - interactive playback — `Session.handle_frame_change` → `step_frame(scene, self)` (`scene_bridge.py:342`);
  - cache export/bake — `Session.export_cache_sequence` → `step_frame(scene, self, write_cache=False)` (`scene_bridge.py:298`), and it is called **after** `frame_set(timeline_frame)` + `sync_scene` (`:293-297`), so animated/driven RNA is evaluated at the intended timeline frame before reconciliation;
  - manual `Step` operator — `fluidSim_warp_blender51/operators.py:205` (`bridge.step_frame(scene, sess)`), no direct `sess.step` call remains outside the seam (verified by grep: the only `self.step(`/`sess.step(` sites are inside `step_frame` itself).
- Differential test: `tests/test_scene_bridge_cache_playback.py:729-793` drives the *same* frame-keyed `(flip_ratio, cfl, gravity)` sequence through export, manual `step_frame`, and `_on_frame_change`, and asserts all three observe the identical triple sequence from inside `step`.

### S4 — inactive status truthfulness — FIXED

`_SCENE_UPDATE_REASONS[SceneUpdate.LIVE]` is now `"Live setting stored; applies on initialization"` (`scene_bridge.py:423`), written by the inactive branch of `request_scene_update` (`:967-981`). Asserted verbatim at `tests/test_operator_panel_behavior.py:620-628`.

### S5 — status thrash on every slider tick — FIXED

The LIVE branch no longer writes status on success (`scene_bridge.py:986-989`, with an explicit comment); only the contained error path replaces it. Three separate tests pin this: status stays `""` (`test_operator_panel_behavior.py:517`... assertion at `:545`), stays `"Frame 4"` (`:596`), stays `"stats"` (`:618`).

### S6 — per-scene isolation of deferred work — DEFERRED (allowed; Note severity)

`_flush_pending_rebuild` still resolves the scene from `bpy.context.scene` (`scene_bridge.py:917`). This is unchanged pre-existing behavior on the REBUILD/RESET timer path, not the LIVE path (`_apply_live_settings_safely`/`_apply_realtime_solver_settings` are scene-parameterized throughout, `scene_bridge.py:810-870`). The original review classified it as a Note already tracked by the plan; plan section P2.2/P2.4 owns removing this path entirely. **Risk if deferred:** with two scenes open, a coalesced rebuild could target the active context scene rather than the edited one — but this defect predates Phase 4 and P2 replaces the mechanism outright, so it does not block P2. It is *not* one of the non-deferrable findings.

### T1 — mechanical registry↔property coupling — FIXED

`tests/test_operator_panel_behavior.py:665-710` parses `FLUIDSIM_PG_settings` with `ast` (not regex) and asserts:
- exact bidirectional equality `live_props == set(_LIVE_SETTING_FIELDS) | {"gravity", "use_scene_gravity"}` (`:686-687`) — a `_request_live` property missing from the registry now fails, and a registry field without a live property also fails;
- the closed set of declared update callbacks (`:688-694`) — a new undeclared callback fails the test;
- allocation fields (`voxel_size`, `leaf_capacity`, `max_particles`, `max_whitewater`) are disjoint from live ∪ reset (`:697-700`);
- hot ∩ reset = ∅ and hot ∩ surface = ∅ (`:701-702`);
- `_LIVE_REFRESH_SETTING_FIELDS == {"particles_per_voxel"}` (`:703`);
- reset/surface registries are subsets of their declared callback groups (`:704-709`);
- `_validate_setting_registry()` runs (`:710`).
The property census also matches: `properties.py` carries 58 `update=_request_live` declarations (56 hot + `particles_per_voxel` + the two gravity props). See `N3` for the one coupling edge this test still does not assert.

### T2 — gravity branch assertions — FIXED

`tests/test_operator_panel_behavior.py:598-618` exercises both branches on a live session: `use_scene_gravity=True` + `scene.use_gravity=True` → `settings.gravity == (0.0, 0.0, -4.0)`; `use_scene_gravity=False` → `(1.0, 2.0, 3.0)`; and the frame-statistics status survives. The implementation is a single shared helper `_apply_gravity` (`scene_bridge.py:786-794`) reused by both the live path (`:829`) and the reset path (`:868`), so the live and reset gravity rules cannot drift apart.

### T3 — brittle source-regex test — FIXED

No `update=_request_live` regex assertion remains anywhere under `tests/` (grep returns nothing); the check is now AST-based inside the T1 test (`tests/test_operator_panel_behavior.py:669-685`).

### T4 — per-call hot-path overhead — FIXED

Enum class imports and the enum field table are memoized in `_enum_setting_fields` (`scene_bridge.py:526-556`) and target validation is memoized in `_validate_setting_registry` (`scene_bridge.py:559-575`). Measured: registry validation 43.4 µs cold, 0.50 µs memoized; full live reconcile 37.3 µs median (§4).

### T5 — double sweep during reset — FIXED

`_apply_reset_solver_settings` performs one `_apply_scene_settings(..., _LIVE_SETTING_FIELDS | _RESET_SETTING_FIELDS)` pass plus `_apply_gravity` (`scene_bridge.py:860-870`), called once from `Session.reset` before `solver.reset()` (`scene_bridge.py:267-268`). Reset ordering remains asserted by `tests/test_operator_panel_behavior.py:633-663` (snapshot taken *inside* `solver.reset()` equals `(0.2, 77)`).

### T6 — plan artifact untracked — DEFERRED (allowed; Note severity)

`git status --porcelain reviews/` in the Phase-4 worktree still reports `?? reviews/`, and the tracked `.gitignore` does not contain a `reviews/` pattern (lines 25-35 inspected; `git check-ignore` returns an anomalous empty-pattern match at a blank line and is contradicted by `git status`). **Risk if deferred:** the approved hybrid contract and both review artifacts exist only as untracked files on a feature branch and can be lost by a clean/checkout. This is a documentation-custody question, not a correctness one, and the repo's `CONTRIBUTING.md` policy of keeping review output out of Git makes leaving it untracked defensible. It does not gate P2; it must be decided before merge.

---

## 2. New findings introduced by the P1A fixes

### N1 — Medium (performance, not correctness) — the ST recalibration hook is an ~86 ms synchronous call on the RNA callback and per-frame reconcile paths

`scene_bridge.py:830-835` calls `solver._calibrate_st_reference_mass()` inline. That function is a numpy Monte-Carlo estimate (3 MAC face orientations × 27 cells × 32 realizations, `sim/solver.py:972-1028`). Measured in this worktree with a settings-only stub receiver:

```
calibrate first 179.90 ms, mean 85.98 ms, value 7.725084631280045
```

Consequences, both narrow but real:
- With ST-FLIP enabled and automatic reference mass, **each drag tick** of `particles_per_voxel` runs a fresh ~86 ms calibration inside the RNA update callback — the same class of freeze the code already coalesces onto a timer for REBUILD/RESET (`scene_bridge.py:889-899` documents 174-191 ms/tick as "a multi-second freeze over a normal drag gesture").
- If `particles_per_voxel` is keyframed/driven, the per-frame reconcile pays ~86 ms **every frame the value changes**, on top of the GPU step.

Why this is not a blocker: it is bounded, correctness-preserving, gated on an actual value change *and* on ST-FLIP (Beta, off by default) *and* on automatic reference mass; the reconcile itself without recalibration is 37 µs. Recommended fix (P2 timeframe, since P2 already builds debounce/pending infrastructure): route the recalibration through the existing coalescing timer, or cache by `(particles_per_voxel, spacetime_seed)`, or make the estimator cheaper. **This must be closed before the P4 performance/acceptance lane.**

### N2 — Low — recalibration and rollback are proven only against stubs

`tests/test_operator_panel_behavior.py:569-596` injects `solver._calibrate_st_reference_mass = lambda: ...`; no test runs the real `FluidSolver` estimator through the live path. Also, `_apply_realtime_solver_settings` restores `_st_reference_mass_cached` only `if old_reference is not None` (`scene_bridge.py:840-841`), and restores only targets that already existed on `settings` (`:820-822`) — both are safe for a real `SolverSettings` dataclass and a constructed solver, but the guards are untested. Owned by P4 runtime evidence.

### N3 — Low — the tier test does not assert that live/reset fields are actually appliable

The T1 test proves property↔registry coupling but never asserts `_LIVE_SETTING_FIELDS | _RESET_SETTING_FIELDS ⊆ set(_DIRECT_SETTING_FIELDS) | enum-sources` (item 2 of plan P1A.4). A future hot field added to `_HOT_SETTING_FIELDS` but not to `_DIRECT_SETTING_FIELDS` would silently no-op with a green suite — the exact failure mode Phase 4 exists to eliminate. I verified independently that the invariant holds *today* (§5, both difference sets empty). One extra assertion closes it.

### N4 — Note (integration risk) — the Phase-4 worktree is 10 commits behind `master` in the same functions it rewrites

`master` advanced from `695fad9` to `451f483` after this branch's base, including `241b981 fix(live): step the whole dropped-frame gap, release on handler teardown`, `81eb577 fix(cache): refuse to play a cache baked from different scene settings`, `a37ef0e fix(solver): report when the substep cap forces a CFL violation`, and `451f483 fix(prefs): one owner for the device lookup`. Those commits touch precisely the code Phase 4 rewrites: `Session.step(..., timeline_frame=...)`, the catch-up loop in `handle_frame_change`, `unregister_handlers`, and `build_settings` device resolution. A naive merge of the Phase-4 worktree would silently revert them (a `git diff --no-index` between the two worktrees shows them as deletions). Recommendation: rebase/merge `master` into the Phase-4 worktree before P2 lands, and make sure `step_frame` forwards `timeline_frame` when the catch-up loop returns. This is a process/merge risk, not a defect in the reviewed slice.

---

## 3. Commands and results

Focused interaction suite (independent run, this review):

```
$ python -m pytest -q tests/test_operator_panel_behavior.py tests/test_scene_bridge_settings.py \
      tests/test_scene_bridge_cache_playback.py tests/test_performance_hotpath_contracts.py
79 passed in 2.56s
```

Full non-Blender regression suite (independent run, this review):

```
$ python -m pytest -q -p no:randomly
355 passed in 6.25s
```

(Initial-review baselines were 72 and 348; the deltas are the 7 new Phase-4 P1A tests.)

Not run, out of scope for this gate and explicitly deferred by the plan to P4: `tools/run_production_gate.py`, all Blender/CUDA acceptance lanes, `tools/profile_user_path.py`, `tools/profile_gpu_solver.py`. No claim in this report depends on GPU behavior.

---

## 4. Bounded pre-step reconciliation performance result

Measured in this worktree with the repo's own bpy stub, on `_apply_realtime_solver_settings` (the whole `prepare_step` payload: 56 live fields + gravity, ST recalibration not triggered because the value is unchanged), 20 batches × 1000 calls:

```
live fields: 56 (6 enum-sourced fields omitted by the stub scene)
per-call median 37.34 us; min 35.33; max 42.70
registry validate cold 43.4 us, memoized 0.50 us
```

Interpretation: the per-frame reconcile costs ~0.04 ms against a GPU frame step measured in milliseconds — bounded and O(fields), not O(substeps). The repo's own in-suite bound (`tests/test_operator_panel_behavior.py:712-731`, `< 1 ms/call`) is met with ~27× margin. The one unbounded outlier on this path is `N1` (~86 ms, only when `particles_per_voxel` actually changes under ST-FLIP + automatic mass).

---

## 5. Independent registry cross-check

Read-only script (not committed) against the live module, using the same bpy stub:

```
live not appliable:                       []
reset not appliable:                      []
live ∩ reset:                             []
live ∩ surface:                           []
live targets missing on SolverSettings:   []
n live targets:                           57
allocation fields in live|reset:          []
```

i.e. every live and reset field resolves to a real `_DIRECT_SETTING_FIELDS`/enum source and to a declared `SolverSettings` attribute; no allocation-bound field (`voxel_size`, `leaf_capacity`, `max_particles`, `max_whitewater`, `domain_min/max`, `device`) is live- or reset-mutated.

---

## 6. Worktree hygiene confirmation (gate item 7)

```
$ git -C ...\liquid_warp_blender51_phase4_interaction diff --cached --name-only
(empty)
$ git -C ...\liquid_warp_blender51_phase4_interaction status --porcelain
(no entry with a staged status column; Phase-4 work is unstaged ' M' plus '??' inherited items)

$ git -C ...\liquid_warp_blender51_port diff --cached --name-only
(empty)
$ git -C ...\liquid_warp_blender51_port status --porcelain | grep -v '^??'
 M .gitignore
 M demos/crown_splash.blend
 M demos/crown_splash_preview.png
 M demos/dam_break.blend
 M demos/dam_break_preview.png
(no staged entries)

$ grep -rn "SceneUpdate\.LIVE|_HOT_SETTING_FIELDS|_RESET_SETTING_FIELDS|_request_live|_apply_realtime_solver_settings|LIVE_REFRESH|reconcile" \
      --include=*.py fluidSim_warp_blender51 tests tools   # in ...\liquid_warp_blender51_port
(no matches — the original Phase-3/baseline worktree contains zero Phase-4 symbols)
```

Both confirmations hold: **no staged files in either worktree**, and **no Phase-4 symbol leakage into the baseline worktree**.

---

## 7. Reviewer verdict

- Blockers: **none**
- High: **none**
- Unresolved correctness Medium: **none** — `S1`, `S2`, `S3`, `T1`, `T2` are all fixed with path:line evidence and behavioral tests; none is deferred.
- Deferred: `S6` (Note, owned by plan P2.2/P2.4), `T6` (Note, custody decision before merge). Both are Note-severity and outside the non-deferrable set.
- New findings: `N1` Medium **performance** (must close before P4 acceptance; does not block P2), `N2`/`N3` Low, `N4` integration/merge risk.
- Standing limitation, unchanged and correctly scoped by the plan: nothing in this gate is Blender/CUDA runtime evidence. Every "the value reaches the kernel" statement is read from launch sites and stub-driven unit seams. P4 runtime lanes remain mandatory before any capability is promoted to `conformant`.

FILE GATE A is satisfied. P2 may begin.

PHASE4-INITIAL-REVIEW-FOLLOWUP-PASS
