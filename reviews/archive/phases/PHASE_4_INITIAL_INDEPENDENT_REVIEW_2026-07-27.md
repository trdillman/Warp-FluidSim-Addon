# Phase 4 — Initial Live-Interaction Slice: Independent Review (2026-07-27)

**Worktree under review:** `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_phase4_interaction`
(branch `phase4/live-property-interaction`, HEAD `695fad9`)
**Fixed baseline:** working tree of sibling `...\liquid_warp_blender51_port` (branch `master`, HEAD `246fa90`), compared with `git diff --no-index` as instructed (HEADs differ; both carry inherited uncommitted work).

**Verdict: APPROVE the initial slice with follow-ups. No blockers.**
The interaction-truth defect is genuinely repaired for the routed properties: hot values reach `SolverSettings` in place, particle state survives, reset ordering is correct, and no allocation-bound field was declared hot. Findings below are one Medium robustness regression, several Medium spec/coverage gaps, and Low nits. Pending UI / auto-rewind work (plan P2–P4) is explicitly out of scope and is not penalized.

Paths below are worktree-relative to the phase-4 worktree.

---

## Scope reviewed

| File | Delta vs baseline |
|---|---|
| `fluidSim_warp_blender51/blender/scene_bridge.py` | +69 / -1 |
| `fluidSim_warp_blender51/properties.py` | +64 / -58 (58 props re-routed to `_request_live`) |
| `tests/test_operator_panel_behavior.py` | +96 (4 new tests) |
| `tests/test_scene_bridge_cache_playback.py` | +26 (1 new test) |
| Phase 4 live-interaction plan (then at the `reviews/` root; now `reviews/archive/phases/PHASE_4_LIVE_INTERACTION_PLAN_2026-07-27.md`) | new (89 lines, untracked) |

Also read: `AGENTS.md`, `CONTRIBUTING.md`, plan + approved hybrid contract, `sim/solver.py`, `sim/pressure.py`, `sim/surface.py`.

---

## Correct (verified, with evidence)

1. **Live edits truly take effect without clearing state.** `SceneUpdate.LIVE` mutates the *live* `sess.solver.settings` in place (`scene_bridge.py:754-771`, dispatched at `:897-899`) and never touches `solver.particles`/`whitewater`. The solver reads these fields from `self.settings` at launch time every frame — e.g. `sample_type`/`flip_ratio` at `sim/solver.py:1331`, whitewater churn/emission/life/drag at `sim/solver.py:1425-1457`, ST variable projection at `:1264`, `particles_per_voxel` rest count at `sim/pressure.py:1463`. So the copy is observable on the next step with no reinitialize.
2. **Registries are internally consistent — no silent no-ops.** AST check of `_DIRECT_SETTING_FIELDS` (`scene_bridge.py:431-464`), the `enum_fields` sources (`:536-548`), `_HOT_SETTING_FIELDS` (`:478-500`) and `_RESET_SETTING_FIELDS` (`:502-505`):
   - hot fields not appliable: **none**; reset fields not appliable: **none**
   - hot∩reset = ∅, hot∩surface = ∅
   - direct fields claimed by no tier: exactly `voxel_size, leaf_capacity, max_particles, max_whitewater` (all still `update=_request_rebuild`, `properties.py`), plus `whitewater_enabled` (WHITEWATER tier) and `spacetime_resync_surface` (SURFACE tier).
3. **Aliases are correct.** `limit_percent→speed_limit_percent`, `limit_behavior→speed_limit_behavior`, `ww_churn_decay→churn_decay`, `ww_churn_gain→churn_gain` (`:472-477`). The enum branch selects on the *source* name and writes the aliased *target* (`:566-569`), so `limit_behavior` in the hot set resolves to `speed_limit_behavior`. `_apply_scene_settings` also hard-validates all targets against `SolverSettings.__dataclass_fields__` and raises on drift (`:550-556`).
4. **Vector fields are copied by value.** `directional_force` / `omnidirectional_force_center` go through `_VECTOR_SETTING_FIELDS` and are materialized as plain float tuples (`:559-565`), so the new per-frame path cannot latch an invalidated RNA proxy.
5. **Gravity aliasing matches `build_settings`.** `scene_bridge.py:764-770` reproduces `build_settings`'s rule (`:588-591`): scene gravity only when `fs.use_scene_gravity and scene.use_gravity`, else `fs.gravity`; both converted to float tuples. Both `gravity` and `use_scene_gravity` are routed to `_request_live` (`properties.py:619-623`), and both are handled explicitly rather than via the field registry — that is why 58 `_request_live` props map onto 56 hot fields (delta = exactly these two).
6. **Reset ordering is right and does not disturb constructor-bound allocation.** `Session.reset` applies settings *before* `solver.reset()` (`scene_bridge.py:263-264`), which matters because `Solver.reset` re-derives `_st_step_seed` and `_st_reference_mass_cached` from `settings` (`sim/solver.py:723-737`). `_RESET_SETTING_FIELDS` is limited to `topology_expand, open_boundaries, domain_sdf_walls, spacetime_enabled, spacetime_reference_mass, spacetime_seed` — none of which are read by `Solver.__init__` for buffer sizing (`sim/solver.py:602-645` sizes from `domain_min/max, voxel_size, leaf_capacity, max_particles, max_whitewater`, none of which are hot or reset-applied). No constructor-bound allocation value is mutated on a live solver.
7. **Pre-frame reconciliation is bounded and correctly placed.** `handle_frame_change` reconciles once per frame, after the cache-playback early return and after the rewind/reset branch, inside the existing playback try/except (`scene_bridge.py:334-351`). It is O(56 fields) per frame, not per substep.
8. **Existing playback and error behavior is intact.** Cache playback (`:314-321`), rewind→reset (`:331-334`), and the step-failure path that disables `fs.live`, preserves the real error and releases the session (`:343-351`) are unchanged. `_SESSION` singleton, handler registration, and the REBUILD/RESET coalescing timer are untouched.
9. **Tests are behavioral, not structural, and would catch the original defect.**
   - `tests/test_operator_panel_behavior.py:516` asserts settings changed, `solver.particles` identity preserved, `sess.reset` *not* called, and status text — a regression that re-routed LIVE into `_schedule_rebuild` would trip `reset_calls`.
   - `tests/test_operator_panel_behavior.py:544` snapshots `settings` *inside* `solver.reset()` and asserts `(0.2, 77)` — this is exactly the wipe/no-op oracle: on baseline code the snapshot would be the stale `(0.95, 1)`.
   - `tests/test_scene_bridge_cache_playback.py:677` observes settings from inside `step`, catching a missing pre-frame reconcile (the keyframe/driver case).
10. **Plan is honest.** `reviews/archive/phases/PHASE_4_LIVE_INTERACTION_PLAN_2026-07-27.md` marks only the six implemented P1 items `[x]`, states "This is not final Phase 4 acceptance", and its `72 passed` claim reproduces exactly (see Commands).
11. **Hygiene.** No staged files in either worktree; the baseline worktree contains **zero** Phase-4 symbols (`SceneUpdate.LIVE`, `_HOT_SETTING_FIELDS`, `_RESET_SETTING_FIELDS`, `_request_live`, `_apply_realtime_solver_settings`).

---

## Spec / correctness findings

### S1 — Medium — LIVE loses the error containment the old path had
`scene_bridge.py:897-899`. The LIVE branch calls `_apply_realtime_solver_settings` **unguarded** inside an RNA `update` callback. Previously all 58 of these properties routed to `_request_reset` → `_schedule_rebuild` → `_flush_pending_rebuild`, which wraps the work in `try/except`, records `set_error`, writes a user-visible `"Deferred update failed: ..."` status, and still tags a redraw (`:834-845`). Any raise from `_apply_scene_settings` (bad enum key → `KeyError` at `:569`, a `SolverSettings` target drift → `ValueError` at `:555`, a torn-down solver) now escapes into Blender's property-update machinery: no `set_error`, no status, no `_tag_view3d_redraw()`.
The same per-frame call inside `handle_frame_change` (`:340`) *is* guarded, but its handler treats any raise as fatal — it sets `fs.live = False` and releases the session (`:346-350`), so a single bad hot value kills playback instead of being reported and skipped.
Recommendation: wrap the LIVE branch (and ideally the per-frame reconcile) the way `_flush_pending_rebuild` does.

### S2 — Medium — `particles_per_voxel` is hot but its ST-FLIP derived quantity is not invalidated
`particles_per_voxel` is in `_HOT_SETTING_FIELDS` (`scene_bridge.py:491`) and `properties.py:653` is now `_request_live`. But when `spacetime_enabled` is on and `spacetime_reference_mass <= 0`, the reference mass is *derived from* `particles_per_voxel` and cached only in `Solver.__init__` (`sim/solver.py:661-668`) and `Solver.reset` (`:727-733`); `_st_reference_mass()` returns the stale cache (`:1030-1038`). A live `particles_per_voxel` edit therefore changes rest-count/anti-clumping (`sim/pressure.py:1463`) and particle separation (`sim/solver.py:1393`) while ST phase thresholds keep the old calibration until the next reset — an internally inconsistent solver state that no error surfaces.
The plan already lists "Add targeted ST-FLIP recalibration semantics" as pending (P1, unchecked), so this is a known hole; record it concretely. Safe options: move `particles_per_voxel` to the plan's `LIVE_REFRESH` tier with a recalibration hook, or gate it as hot only when `spacetime_enabled` is false.

### S3 — Medium — live playback and baked cache now diverge for animated hot settings
`handle_frame_change` reconciles before every stepped frame (`scene_bridge.py:340`), but `Session.export_cache_sequence` steps in its own loop with no reconcile (`:279-296`) and `step_frame` (`:731-733`, used by the manual step path) also does not. With a keyframed/driven `cfl`, `gravity`, or `flip_ratio`, interactive playback now honours the animation while a bake of the same scene does not — the two produce different results, and cache metadata records nothing about it. Recommend hoisting the reconcile into a single helper used by all three step entry points (cheap and strictly reduces divergence), or explicitly documenting the limitation before P4 cache-metadata work.

### S4 — Low — status string over-claims when no session is active
`scene_bridge.py:889-895`: with `sess.active == False`, LIVE falls into the generic inactive branch and writes `"Live solver settings applied"` (`:423`) even though nothing was applied to any solver. Given this repo's explicit report-truthfulness policy (`backlog/tasks/TASK-POLICY-DRIFT-*`), a wording like "Live setting stored (no active solver)" is preferable. Pre-existing branch, new misleading text.

### S5 — Low — the `LIVE` enum reason doubles as user-facing status on every edit
`request_scene_update` overwrites `scene.fluidsim.status` on each LIVE tick (`:899`), so a slider drag replaces the live particle/step-time readout with a constant string until the next stepped frame restores it. Cosmetic, but worth confirming against the P3 header/overlay design so the pending-state banner does not fight this write.

### S6 — Note — no per-scene isolation yet (already tracked)
`_apply_realtime_solver_settings` operates on the passed `scene` (good), but the deferred rebuild path still resolves `bpy.context.scene` (`:828`). The plan tracks this ("Make deferred work scene-specific"); the new code does not make it worse.

---

## Standards findings

### T1 — Medium — registry↔property coupling is unenforced by tests
The whole point of Phase 4 is to eliminate "the setting silently did nothing". Today the invariants hold (verified by AST above), but nothing prevents a future property from being marked `update=_request_live` while its name is absent from `_HOT_SETTING_FIELDS`/`_DIRECT_SETTING_FIELDS` — it would silently no-op with no test failure. Add a mechanical test asserting: every `_request_live` property name is either in `_HOT_SETTING_FIELDS` or in an explicit allowlist (`gravity`, `use_scene_gravity`), and `_HOT_SETTING_FIELDS | _RESET_SETTING_FIELDS ⊆ _DIRECT_SETTING_FIELDS ∪ enum sources`, with hot∩reset∩surface empty. This also mechanically supports the plan's unchecked "Complete and mechanically validate the property-tier classification".

### T2 — Medium — the newly live-routed gravity path has no assertions
`gravity` and `use_scene_gravity` are the only hot values handled outside the registry (`scene_bridge.py:764-770`), i.e. the highest-risk code. Yet `tests/test_operator_panel_behavior.py:516` uses a `fluidsim` stub *without* a `gravity` attribute (so the branch is skipped entirely), and `tests/test_scene_bridge_cache_playback.py:677` sets gravity but asserts only `flip_ratio`/`cfl`. Neither branch of `use_scene_gravity` is asserted. Add two small assertions (scene-gravity on/off).

### T3 — Low — source-regex test is brittle
`tests/test_operator_panel_behavior.py:578-593` asserts `update=_request_live` by regexing `inspect.getsource(FLUIDSIM_PG_settings)`. It breaks on harmless reformatting and cannot detect the runtime binding. Prefer introspecting the annotation's keyword dict (the bpy stub already fabricates the property objects) or fold this into the T1 mechanical test.

### T4 — Low — per-call overhead in a path that now runs every frame
`_apply_scene_settings` re-imports ten enum classes, rebuilds the `targets` set and re-validates it against `SolverSettings.__dataclass_fields__` on **every** invocation (`:523-556`), now once per frame (`:340`), twice per reset (`:263` → `:776` calls the realtime pass then a second pass), and once per slider tick. Negligible next to a GPU step, but it is avoidable: hoist the validation to module import or memoize it. Worth a glance given `tests/test_performance_hotpath_contracts.py` exists as a lane.

### T5 — Low — `_apply_reset_solver_settings` re-applies the full hot set inside reset
`:773-780` runs the hot pass and then the reset pass, so reset does two full sweeps of `_DIRECT_SETTING_FIELDS`. Correct, just redundant; a single call with `_HOT_SETTING_FIELDS | _RESET_SETTING_FIELDS` is clearer and cheaper.

### T6 — Note — the plan artifact is untracked in this worktree
`reviews/` is untracked here (`git status` shows `?? reviews/`) and is *not* gitignored (`git check-ignore` → not ignored), while the baseline worktree tracks `reviews/*`. `CONTRIBUTING.md` says review/swarm output folders stay out of Git, so leaving it untracked is defensible — but the approved hybrid contract now lives only in an untracked file on a feature branch. Decide deliberately before merge (this is not the `progress.md`-style scratch case; it is the contract of record).

---

## Commands / evidence

```
git -C ...\liquid_warp_blender51_phase4_interaction status --porcelain   # no staged entries
git -C ...\liquid_warp_blender51_phase4_interaction diff --cached --name-only   # empty
git -C ...\liquid_warp_blender51_port status --porcelain                 # only " M .gitignore"; no staged entries
git -C ...\liquid_warp_blender51_port diff --cached --name-only          # empty
grep -rn "SceneUpdate.LIVE|_HOT_SETTING_FIELDS|_RESET_SETTING_FIELDS|_request_live|_apply_realtime_solver_settings" \
    --include=*.py ...\liquid_warp_blender51_port                        # no matches (baseline clean)

git diff --no-index <baseline>/<file> <phase4>/<file>   # for all 4 code/test files (stats in Scope table)

python -m pytest -q tests/test_operator_panel_behavior.py tests/test_scene_bridge_settings.py \
    tests/test_scene_bridge_cache_playback.py tests/test_performance_hotpath_contracts.py
# -> 72 passed in 13.48s   (matches the plan's "72 passed" claim)

python -m pytest -q -x -p no:randomly
# -> 348 passed in 10.27s  (full non-Blender suite; no regressions)
```

AST cross-check of the registries (read-only script, not committed):
```
hot fields not appliable:            []
reset fields not appliable:          []
hot ∩ reset:                         []      hot ∩ surface: []
direct fields in no tier:            ['leaf_capacity', 'max_particles', 'max_whitewater',
                                      'spacetime_resync_surface', 'voxel_size', 'whitewater_enabled']
len(_HOT_SETTING_FIELDS) = 56 ; _request_live properties = 58 (delta = gravity, use_scene_gravity)
```
Property-tier census from `properties.py`: `_request_live` 58, `_request_reset` 14 (height/* , topology/open-boundary/domain-wall, spacetime_enabled/reference_mass/seed), `_request_rebuild` 7 (allocation/domain/profiling), `_request_surface_refresh` 22, `_request_output_refresh` 12, `_request_whitewater_refresh` 1.

Not run (out of scope / requires Blender+CUDA, and the plan defers them): `tools/run_production_gate.py`, the Blender acceptance lanes, `profile_user_path.py`, `profile_gpu_solver.py`.

---

## Residual risks

1. No Blender/GPU runtime evidence yet: every claim above about "the value reaches the kernel" is read from the launch sites, not observed on a live CUDA session. A live-edit acceptance lane (plan P4) is still required.
2. Untested hot-field semantics under real solvers: `pressure_solver`/`viscosity_solver`/`sample_type` enum switches mid-run are plausible-but-unproven (buffers for all paths are allocated in `Solver.__init__`, and the PCG CUDA-graph cache is keyed on `(work_n, batch, parity)` with values read from arrays, so a value change should replay safely — but this was reasoned, not measured).
3. S2's stale ST reference mass is the one live combination that can produce quietly wrong physics rather than a visible error.
4. S3 means a user can bake a cache that does not match what they just watched; cache metadata (plan P4) has not yet recorded settings signatures.
5. The eventual hybrid contract requires per-scene pending state; the current LIVE path is scene-parameterized (safe), but the deferred rebuild path is not — that must not be widened before P2.

None of these makes the approved hybrid contract unsafe to build on: the LIVE tier is additive, bounded, keyed by explicit registries, and never touches allocation-bound values.
