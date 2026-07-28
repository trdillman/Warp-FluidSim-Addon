# Phase 4 — FILE GATE B: Final Independent Interaction Acceptance Review (complete replacement)

- Date (UTC): 2026-07-28
- Worktree: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_phase4_interaction`
- Branch: `phase4/live-property-interaction` @ `695fad96072e` (working tree dirty; Phase 1–4 work unstaged, **no staged files**)
- Reviewer role: fresh independent read-only reviewer. No implementation, test, doc, or tool file was modified by this review; only this report was authored.
- Gate spec: `reviews/PHASE_4_LIVE_INTERACTION_PLAN_2026-07-27.md` → "FILE GATE B — Final interaction acceptance"
- Predecessor gates: `reviews/PHASE_4_INITIAL_REVIEW_FOLLOWUP_GATE_2026-07-27.md` (`PHASE4-INITIAL-REVIEW-FOLLOWUP-PASS`), plus the prior Gate-B pass whose residual notes `N-r1` / `N-r2` this pass re-verifies.

---

## 1. Verdict

**PASS.** No blocker, no high, no unresolved correctness Medium. Every shipped Phase-4 claim is `conformant` against runtime evidence produced in this worktree on real Blender 5.1 + CUDA (RTX 4080 SUPER), not source inspection or screenshots. Both prior residual notes are **closed with code + test + doc evidence**. Remaining items in §8 are Notes and one integration/custody risk; none is a correctness defect and none is gated by the plan.

Independent lanes run by this reviewer in this worktree (all fresh, this session):

| Lane | Command | Result |
|---|---|---|
| Full non-Blender regression | `python -m pytest -q` | **377 passed in 8.88s** |
| Headless production gate | `python tools/run_production_gate.py` | exit 0 → **HEADLESS-GATE-PASS** |
| GUI-inclusive production gate | `python tools/run_production_gate.py --include-gui` | exit 0 → **FULL-PRODUCTION-GATE-PASS** |
| Phase-4 Blender/CUDA acceptance | executed inside both gates; artifact re-emitted | **PHASE4-LIVE-INTERACTION-PASS** (14/14, `artifacts/phase4_live_interaction.json` `"ok": true`) |

Test-count delta versus the previous Gate-B report: 374 → **377** (+3), matching the residual-note regression tests (direct-step immediate detection; baseline/changes metadata; tagged non-finite encoding).

Gate metadata emitted by the tool itself (proves both lanes ran against the same tree, this session):

```
== gate metadata ==                                == gate metadata ==
timestamp_utc: 2026-07-28T07:08:33.981375+00:00    timestamp_utc: 2026-07-28T07:10:54.748743+00:00
git_head: 695fad96072e                             git_head: 695fad96072e
git_dirty: True                                    git_dirty: True
tracked_ignored_files: False                       tracked_ignored_files: False
blender_exe: ...\Blender 5.1\blender.exe           blender_exe: ...\Blender 5.1\blender.exe
gui_acceptance: not-run-headless-gate              gui_acceptance: included
```

---

## 2. Residual-note verification (primary purpose of this pass)

### N-r1 — CLOSED. Immediate headless/scripted edit can no longer step once before detection.

Evidence:

- `fluidSim_warp_blender51/blender/scene_bridge.py:191-201` — `Session.prepare_step(scene, *, refresh_pending=False)` now calls `_refresh_pending_state(scene, self, use_cached=not refresh_pending)`, with an explicit comment naming the two regimes.
- `scene_bridge.py:1049-1063` — `step_frame(..., refresh_pending: bool = True)`: **direct callers conservatively recompute** the reset/reinit signatures.
- `scene_bridge.py:378-381` — the interactive playback path is the single site that opts out (`step_frame(scene, self, refresh_pending=False)`), documented as relying on the persistent RNA/depsgraph/undo handlers that already refreshed the per-scene snapshot.
- `scene_bridge.py:747-765` — `_refresh_pending_state(..., use_cached=False)` only reuses `current_reset_signature`/`current_reinit_signature` when *both* cached values exist; otherwise it recomputes. The cached branch is reachable only under `use_cached=True`, so the conservative direction is preserved.
- Export path is conservative by default: `scene_bridge.py:323` calls `step_frame(scene, self, write_cache=False)` with the `refresh_pending=True` default; the manual operator (`operators.py:293`) and `tools/profile_user_path.py:392` likewise use the default.
- Regression test proving immediate detection without any depsgraph tick: `tests/test_phase4_pending_interaction.py:286-295` (`test_direct_step_refreshes_immediate_scripted_edit_without_depsgraph`) sets `scene.fluidsim.max_particles = 2500` and asserts `pending_presentation(scene)["action"] == "REINIT_PENDING"` immediately after a single `step_frame`. The handler-driven counterpart (`:298-306`) still proves the playback regime.

Verified by execution: both tests are inside the green `377 passed` run, and the Blender/CUDA lane still reports `mid-run allocation edit stays pending → REINIT_PENDING / ('Reinitialize: voxel_size',)` with unchanged solver/particle buffers.

### N-r2 — CLOSED. Non-finite floats and metadata layout are unambiguous and documented.

Evidence:

- `fluidSim_warp_blender51/io/cache.py:43-50` — `_jsonable_setting` maps a non-finite float to the tagged object `{"__fluidsim_nonfinite_float__": "NaN" | "Infinity" | "-Infinity"}`. This is standard JSON (so `json.dumps(..., allow_nan=False)` at `cache.py:68-71` no longer raises) and it is not confusable with an ordinary user string, which was the ambiguity called out by N-r2.
- `cache.py:170-197` — sequence metadata now stores `settings_baseline` (`{signature, settings}` for the first segment) plus `settings_changes` (`{signature: {changed, removed}}` deltas). The former full `settings_snapshots` map is gone; growth is now per-changed-field rather than a full settings object per distinct signature (this also shrinks the prior N2 growth concern).
- Tests: `tests/test_cache_io.py:200-218` asserts the baseline/changes layout **and** `"settings_snapshots" not in metadata`; `tests/test_cache_io.py:220-235` asserts exactly the three tagged encodings round-trip through `write_sequence_metadata`/`read_sequence_metadata`.
- User docs: `fluidSim_warp_blender51/docs/user_guide.md:401-410` describes `settings_baseline` (first complete setting set), `settings_changes` (changed/removed fields only, avoiding repeated full snapshots during animation), and the `__fluidsim_nonfinite_float__` tag "rather than non-standard JSON values or ambiguous ordinary strings".
- Runtime confirmation, not just unit: the CUDA lane check `cache reports animated settings boundary` now prints the real metadata containing `"mixed_settings": true` and a populated `"settings_baseline"` block (gate log line 601).

---

## 3. Separate production markers (plan requirement)

Headless lane (`tools/run_production_gate.py`, no GUI), exit 0:

```text
HEADLESS-GATE-PASS
GUI acceptance not run; use --include-gui for full production proof.
```

GUI-inclusive lane (`--include-gui`), exit 0:

```text
GUI-ACCEPTANCE-PASS (38/38)          (log line 1152)
GUI-PLAYBACK-PROBE-PASS              (1178)
SAVED-DEMO-GUI-PLAYBACK-PASS         (1185)
FULL-PRODUCTION-GATE-PASS            (1187)
```

The separation is enforced in code, not prose: `tools/run_production_gate.py:323-328` prints `FULL-PRODUCTION-GATE-PASS` only when `include_gui` is true, and each Blender lane must emit its own success marker (`tools/run_production_gate.py:88-100`), so a zero exit from a silently-aborted script cannot pass. The GUI lane deletes `artifacts/gui_playback_probe.json` before the probe and validates it with `--expected-blend demos/dam_break.blend` (`tools/run_production_gate.py:271-277`), so a stale artifact cannot satisfy the gate.

Sub-markers observed in the headless log (line numbers): `CACHE-EXPORT-PIPELINE-PASS` (261), `MEMORY-SAFETY-GATE-PASS` (287), `HANDLER-PERSISTENCE-GATE-PASS` (314), `VIEWPORT-2FRAME-GATE-PASS` (353), `BACKGROUND-ACCEPTANCE-PASS` (462), `ATTRIBUTE-VIZ-ACCEPTANCE-PASS` (507), `MESH-SOURCE-ACCEPTANCE-PASS` (540), `FORCE-REGION-ACCEPTANCE-PASS` (566, `delta_x=0.500000` on 67,584 particles), `PHASE4-LIVE-INTERACTION-PASS` (607), `PHASE3-SOURCE-UI-ACCEPTANCE-PASS` (661), `DEMO-ANALYSIS-PASS (18/18)` (767). All reappear in the GUI-inclusive log.

---

## 4. Final claim matrix

"Runtime" = Blender 5.1 + CUDA lane in this session; "unit" = non-Blender behavioral test in the green 377-test run; "source" = implementation seam read at path:line.

| # | Claim | Implementation evidence | Behavioral / runtime evidence | Status |
|---|---|---|---|---|
| 1 | Hot settings update the active solver without state loss | `scene_bridge.py:1109-1160` (`_apply_realtime_solver_settings`, atomic rollback), `:514-538` (`_HOT_SETTING_FIELDS`) | Runtime `hot fields reach active solver` = `(0.35, (0,0,-3), 18, True, 0.02); 1.18 ms`; `live edits preserve particle allocation` (solver and particle buffer identity unchanged) | **conformant** |
| 2 | Reset/reinit rebuild from current RNA in the right order | `scene_bridge.py:1180-1190` applies live+reset in one pass, called at `:253-256` before `solver.reset()` | Runtime `timeline rewind applies pending reinitialize → NONE` with a rebuilt solver at the edited voxel size; unit `test_timeline_reset_reapplies_current_solver_settings` | **conformant** |
| 3 | Live errors are contained, recorded, visible, non-fatal | `scene_bridge.py:1163-1177`, `:1058-1063` (`step_frame` surfaces the error instead of releasing the session) | Unit `test_live_setting_errors_are_atomic_contained_and_visible`; GUI `panel state error visible` (log 1128) | **conformant** |
| 4 | ST-FLIP derived calibration follows the live density control | `scene_bridge.py:1085-1106` (`_cached_st_reference_mass`), `:1140-1148` | Runtime `ST-FLIP automatic mass rescales 7.725084631280045 → 11.587626946920068` (ratio 1.5 for 8→12 ppv) on a real CUDA `FluidSolver` | **conformant** |
| 5 | Animated values behave identically across all step paths | Single seam `Session.prepare_step` (`:191-201`) used by `step_frame` (`:1049-1063`), playback (`:378-381`), manual op (`operators.py:290-293`), export (`scene_bridge.py:323`) | Runtime `manual step evaluates animated live values [(2.0,0.8),(1.25,0.4)]`; `playback applies animated live values`; export feeds the same values into the cache (row 9) | **conformant** |
| 6 | Every property belongs to exactly one mechanically checked tier | `scene_bridge.py:514-566` (LIVE / LIVE_REFRESH / RESET / SURFACE / REINIT), `:806-826` (`_validate_setting_registry`) | Unit `test_solver_property_interaction_tiers_are_mechanically_exhaustive` (AST-parses `FLUIDSIM_PG_settings`, asserts set equality, disjointness, allocation exclusion, appliability) | **conformant** |
| 7 | Gravity scene/custom branches are live | `scene_bridge.py:1063-1071` (`_apply_gravity`) | Runtime gravity `(0,0,-3)` reached the active solver; unit `test_live_gravity_uses_scene_or_custom_branch_without_reset` | **conformant** |
| 8 | Destructive edits stay pending, per scene | `scene_bridge.py:571-593`, `:747-775` (`_refresh_pending_state`), `:1355-1372` | Runtime `mid-run allocation edit stays pending → REINIT_PENDING / ('Reinitialize: voxel_size',)`; `multiple-scene pending state is isolated` (`max_particles` vs `voxel_size` across two real `bpy` scenes) | **conformant** |
| 9 | Rewind applies pending changes | `scene_bridge.py:1263-1268`, wired at `:392-403` | Runtime `frame_set(frame_start)` rebuilt the solver and cleared pending to `NONE` | **conformant** |
| 10 | Explicit operators exist, are context-safe, never save preferences | `operators.py:186-268`, `scene_bridge.py:1216-1249`, `:1268-1300` | Runtime GUI `GUI pending revert action {'FINISHED'}` + `Info: Pending values reverted`; units `test_apply_operators_do_not_destroy_another_scenes_solver`, `test_failed_pending_apply_retains_action_and_reason`, `test_revert_never_changes_device_preferences`; repo-wide grep for `preferences.save`/`is_dirty` in the add-on: zero hits | **conformant** |
| 11 | Always-reachable fixed header control | `panels.py:56-89` (`FLUIDSIM_PT_interaction_popover`, `bl_region_type = "HEADER"`), `:1039-1058` (symmetric `VIEW3D_HT_header.append/remove`) | Runtime GUI `fixed header popover callable outside the sidebar region {'INTERFACE'}` (log 1120 — check renamed, closing prior note N3); headless `fixed header popover registered`; unit `test_header_registration_is_symmetric` | **conformant** |
| 12 | Panel feedback is truthful and draw-safe | `panels.py:24-52` (banner reads only cached presentation), `scene_bridge.py:696-720` (`pending_presentation` performs no geometry or RNA writes) | GUI `panel draw smoke` in four states: 20/21/20/21 panels drew with no draw errors (log 1017, 1041, 1070, 1111), plus `panel draw smoke (diagnostics populated)` 21 panels (1151) | **conformant** |
| 13 | Viewport overlay is informational and subordinate | `blender/viewport.py:826-882` (POST_PIXEL text only), `:884-905` (register/unregister) | Runtime `viewport pending overlay registered`; GUI screenshot with overlay active `2100x1193, 3995/4000 visible samples` | **conformant** |
| 14 | Cache identifies settings changes truthfully and unambiguously | `io/cache.py:43-50` (tagged non-finite), `:62-71` (`settings_record`), `:74-86` (`compact_settings_segments`), `:170-197` (`settings_baseline` / `settings_changes` / `mixed_settings`) | Runtime `cache records settings on every frame` and `cache reports animated settings boundary` → real metadata with `mixed_settings: true`, contiguous segments, distinct sha256 signatures; units `tests/test_cache_io.py:200-235` | **conformant** |
| 15 | Docs state the five tiers, rewind behavior, the no-sticky-sidebar limitation, and the cache metadata contract | `docs/user_guide.md:401-410` (baseline/changes + non-finite tag), `:413-440` (tier table, per-scene pending, rewind, Apply/Reset/Revert, failed-apply semantics, no-preference-save, header popover, explicit sticky-sidebar disclaimer) | Doc text verified row by row against the implemented behavior in this pass | **conformant** |
| 16 | Handlers are persistent and survive save/reopen | `scene_bridge.py:1404-1435` (`@persistent` callbacks; symmetric register/unregister across `frame_change_post`, `depsgraph_update_post`, `undo_post`, `load_pre`) | `HANDLER-PERSISTENCE-GATE-PASS` in both gate lanes (headless log 314); the gate checks every handler list, `@persistent` on every callback, and survival across save/reopen | **conformant** |

---

## 5. Source spot-checks (truth vs. label)

- **Pending is signature-based, not a stale boolean.** `_reset_signature`/`_reinit_signature` (`scene_bridge.py:646-660`) hash real RNA values, object lifecycle fields, domain identity, computed bounds, and the configured device. `_signature_changes` (`:680-687`) treats a missing applied snapshot as *unknown*, not "matching" — the conservative direction, consistent with the N-r1 fix.
- **The two step regimes are explicit and asymmetric on purpose.** The only `refresh_pending=False` caller is interactive playback; every other caller (export, manual operator, profiling tool, tests) uses the conservative default. This is the intended shape of the N-r1 fix and it is commented at both `:194-198` and `:1052-1056`.
- **Scripted/undo/driver edits are still covered on the playback path.** `_on_interaction_state_update` is registered on `depsgraph_update_post` and `undo_post` (`:1391-1400`, `:1412-1418`) and is unit-proven (`test_scripted_signature_change_is_detected_without_rna_callback`, `test_scripted_edit_handler_refreshes_cached_pending_presentation`).
- **Failure semantics are real.** A failed pending apply during rewind stops playback rather than advancing stale state (`:370-381`: sets error, `fs.live = False`, keeps pending), proven by `test_rewind_apply_failure_stops_playback_and_retains_solver`.
- **The ST recalibration shortcut is honest and documented.** `_calibrate_st_reference_mass` (`sim/solver.py:972-1027`) is exactly linear in `particles_per_voxel` in expectation, so the cached coefficient keyed by `spacetime_seed` (`scene_bridge.py:1085-1106`) is an expectation-exact reuse, not a tuned constant. This is what keeps the measured live-apply batch at 1.18 ms instead of the ~86 ms Gate-A finding N1.
- **Cache delta encoding is lossless.** `settings_changes[sig] = {"changed": ..., "removed": ...}` (`io/cache.py:177-186`) records fields whose value differs from the baseline plus fields absent relative to the baseline, so any signature's full settings set is reconstructible from `settings_baseline` + its delta.

---

## 6. Runtime / cache / GUI / performance evidence detail

Fresh `artifacts/phase4_live_interaction.json` (`schema: fluidsim-phase4-live-interaction-v1`, `"ok": true`, 14/14, rewritten by this session's gate runs):

```
PASS CUDA solver initialized                      | Solver ready
PASS hot fields reach active solver               | (0.35, (0,0,-3), 18, True, 0.02); 1.18 ms
PASS live edits preserve particle allocation
PASS ST-FLIP automatic mass rescales              | 7.725084631280045 -> 11.587626946920068
PASS live refresh is bounded                      | 1.18 ms
PASS manual step evaluates animated live values   | [(2.0, 0.8), (1.25, 0.4)]
PASS cache records settings on every frame
PASS cache reports animated settings boundary     | mixed_settings: true, settings_baseline populated
PASS playback applies animated live values
PASS mid-run allocation edit stays pending        | REINIT_PENDING / ('Reinitialize: voxel_size',)
PASS timeline rewind applies pending reinitialize | NONE
PASS multiple-scene pending state is isolated     | first: max_particles, second: voxel_size
PASS fixed header popover registered
PASS viewport pending overlay registered
```

GUI-inclusive lane, interaction-relevant checks (`GUI-ACCEPTANCE-PASS (38/38)`):

```
[PASS] mid-run GUI edit is pending {'action': 'REINIT_PENDING', 'count': 1, 'reasons': ('Reinitialize: voxel_size',)}
[PASS] fixed header popover callable outside the sidebar region {'INTERFACE'}
[PASS] GUI pending revert action {'FINISHED'}
[PASS] panel draw smoke (post-playback manual step) 21 panels drew
[PASS] viewport screenshot 2100x1193, 3995/4000 visible samples
[PASS] render contains visible content 480x320, 4000/4000 visible samples
[PASS] profiler frames in GUI 40
[PASS] benchmark operator in GUI  -> 18.6 ms/frame avg, 3.9M voxels/s
```

Saved-demo timeline playback: `GUI-PLAYBACK-PROBE-PASS` on `demos/dam_break.blend`, validated with `--expected-blend` → `SAVED-DEMO-GUI-PLAYBACK-PASS`.

| Measurement | Value | Source |
|---|---|---|
| Live-apply batch (6 RNA edits incl. ST recalibration) on a live CUDA solver | **1.18 ms** (asserted bound `< 250 ms`) | `artifacts/phase4_live_interaction.json` → `live refresh is bounded` |
| Pre-step reconcile (`_apply_realtime_solver_settings`) | `< 1 ms` per call over 500 iterations after registry warmup | `tests/test_operator_panel_behavior.py::test_live_reconcile_is_bounded_after_registry_warmup` |
| GUI interactive benchmark | 18.6 ms/frame avg, 3.9M voxels/s, 40 profiler frames | GUI lane log 1129-1132 |
| GUI draw cost with overlay + header active | 15.8–17.6 ms per `Draw window and swap` | GUI lane log 1133-1136 |
| GPU profiler smoke | clean in both lanes | headless log §"GPU profiler smoke", `profiler collected frames 10 frames` |
| Memory safety | `MEMORY-SAFETY-GATE-PASS`; safeguard precedes CUDA allocation (`scene_bridge.py:104-118`) | both lanes |

---

## 7. Hygiene

```
$ git diff --cached --name-only          -> (empty)      # no staged files
$ git status --porcelain | grep -c '^[AMDR]'  -> 0       # nothing staged
$ git diff --stat | tail -1              -> 32 files changed, 6706 insertions(+), 676 deletions(-)
$ git ls-files -ci --exclude-standard    -> (empty)      # gate metadata tracked_ignored_files: False
```

Untracked scratch/backlog/review/demo artifacts (`backlog/tasks/…`, `demos/*.blend`, `.pi-subagents/`, progress-style notes) are present and intentionally untracked; they are not flagged as repo noise.

---

## 8. Remaining limitations, notes, and residual risks

**N-r1, N-r2: CLOSED** (§2). No successor defect found.

**N1 — Note (performance, bounded).** `_reset_signature` (`scene_bridge.py:646-660`) walks every scene object and reads five RNA fields per object; it now runs on every direct `step_frame` in addition to every `depsgraph_update_post`. Reviewer measurement from the prior pass: ~7.8 µs/object → ~3.9 ms at 500 objects. Invisible at demo-scene sizes and dwarfed by a GPU frame step, but the N-r1 fix slightly increases how often it runs on scripted/headless step loops. Optional follow-up: cache the object-side signature per depsgraph generation, or restrict the walk to objects with a non-`NONE` FluidSim role.

**N2 — Note (cache metadata format evolution, no version bump).** `CACHE_FORMAT_VERSION` is still `1` (`io/cache.py:21`) while `fluidsim_sequence.json` changed from `settings_snapshots` to `settings_baseline` + `settings_changes`. No in-repo consumer reads either key (grep across `fluidSim_warp_blender51`, `tests`, `tools` shows only `io/cache.py` and `tests/test_cache_io.py`), so nothing breaks today; but a metadata file written by an older build and a new one are structurally different at the same declared version. Also, there is no helper that reconstructs a full settings set from baseline + delta — external tooling must implement the merge itself.

**N3 — Note (API footgun, cosmetic).** Defaults are deliberately asymmetric: `Session.prepare_step(..., refresh_pending=False)` vs `step_frame(..., refresh_pending=True)`. A future caller that reaches for `sess.prepare_step(scene)` directly instead of `step_frame` silently gets the cached-snapshot regime, i.e. exactly the N-r1 behavior. Consider making `Session.prepare_step` require the flag explicitly, or asserting the playback-only opt-out at one call site.

**N4 — Note (ST calibration is an expectation-exact extrapolation).** Documented at `scene_bridge.py:1091-1097`. Only the first calibration per `spacetime_seed` is a real estimator run; later live density changes scale it linearly. Mathematically justified (§5) and the reason the live path is 1.18 ms, but it should remain documented if the ST weighting kernels ever change.

**N5 — Integration/custody risk (unchanged, not a Phase-4 correctness defect).** The entire ~6.7k-line Phase 1–4 change set is uncommitted working-tree content on a branch behind `master` in files `master` also changed (e.g. `preferences.py`). Resolve before merge: commit, rebase onto `master`, then re-run both gate lanes. The plan does not gate on this.

**Standing limitation (correctly scoped).** All CUDA evidence comes from a single machine/driver combination (RTX 4080 SUPER, Blender 5.1.0 `adfe2921d5f3`). Device-transition (`BLOCKED` tier) behavior is proven by the guarded pending/blocked-reason path and unit tests, not by an actual multi-GPU switch during playback.

---

## 9. Gate conditions checklist (plan "FILE GATE B")

| Requirement | Status |
|---|---|
| Residual notes `N-r1`, `N-r2` verified fixed with code + test + doc evidence | ✅ §2 |
| Final claim matrix, every shipped claim `conformant` | ✅ §4 (16/16) |
| Source evidence at path:line | ✅ §2, §4, §5 |
| Runtime evidence (Blender 5.1 + CUDA) | ✅ §3, §6 |
| Cache evidence | ✅ §4 row 14, §6 (`mixed_settings: true`, baseline/changes, tagged non-finite) |
| GUI evidence | ✅ §6 (`GUI-ACCEPTANCE-PASS (38/38)`, popover, revert, screenshots, saved-demo playback) |
| Performance evidence | ✅ §6 (1.18 ms live apply, <1 ms reconcile, 18.6 ms/frame GUI, memory gate) |
| Separate headless and GUI-inclusive markers | ✅ §3 |
| Independent full regression run | ✅ §1 (`377 passed`) |
| Fresh independent verdict, no blocker/high, all correctness mediums resolved | ✅ §1, §8 |
| Explicit remaining limitations | ✅ §8 |
| No implementation/test/doc modified by this review; no staged files | ✅ §7 |

PHASE4-FINAL-INTERACTION-PASS
