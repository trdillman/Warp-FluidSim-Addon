# Artist Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use inline execution by default; do not start a separate task or agent wave without applicable authorization.

**Goal:** Make Preview and ordinary adjustments complete user actions while preserving simulation ownership, lifecycle semantics, effect trust, and cache safety.

**Architecture:** Add small orchestration functions to the existing playback and pending-settings owners, with Blender operators as context/confirmation adapters. Prepare effect drafts inside Simulation Admission and publish their applied evidence at successful installation. Preserve the existing runtime state machine and operator IDs.

**Tech Stack:** Python, Blender 5.1 bpy/RNA/operators/handlers, existing Warp/CUDA runtime, pytest, isolated source-loaded Blender GUI acceptance.

**Spec:** `docs/superpowers/specs/2026-09-12-artist-workflows.md` (read before execution).

## Execution coverage

The detailed checklists below preserve the original planned steps and suggested commit boundaries; actual implementation used reviewed shared-owner checkpoints rather than those exact commit messages. Current completion is tracked here and in `2026-09-13-artist-workflow-final.md`.

| Task | Implementation status | Proof |
|---|---|---|
| 1 — Baseline and behavioral contracts | Complete | Initial baseline, new focused regressions, independent reviews |
| 2 — Effect preparation | Complete | Admission/trust/hot-binding tests and real custom-effect dispatch |
| 3 — Preview/Pause/Resume | Complete | Final real context matrix, saved playback, effect and no-effect GUI passes |
| 4 — Apply/Restart | Complete | Host regressions and actual dialog cancellation, stale-token rejection and confirmation pass |
| 5 — Setup composition | Complete | Scoped rollback tests, real idempotence and setup-preview receipts |
| 6 — Cache/bake safety | Complete | Fault-injection tests, actual cache playback and modal scene switch/delete receipts |
| 7 — Artist-facing panels/docs | Complete | Basic/Advanced source contracts and final broad GUI 55/55 pass |
| 8 — Final integration and handoff | Complete | Installer, 1542 host tests, default headless aggregate, current GUI components and visible custom-effects handoff pass; see final report |

## Global Constraints

- Target Blender 5.1 on Windows with the existing Warp/CUDA runtime; add no dependency.
- Keep simulation GPU-resident and retain the fixed 8x8x8 sparse leaf-grid contract.
- Preserve authoritative object-owned domain settings and the existing scene ownership rules.
- Keep simulation_admission, scene_bridge, playback, pending_settings, output_publication, and cache_bake authoritative in their existing responsibilities.
- Do not change catch-up, desynchronization, frame stepping, cache formats, numerical algorithms, or bake reproducibility claims.
- Preserve unrelated work and archived experiments; do not merge or extract experiments.
- Use temporary source-loaded Blender sessions for development; do not change the normal installed add-on/profile.
- No external publication, automatic cache deletion, or runtime rewrite is part of this work.

## Baseline and critical path

Planning checkout: `C:\Users\Tyler\.codex\worktrees\c6d1\liquid_warp_blender51_port`, clean detached HEAD `08bac736a791eb32be63ff02b11671952dc8c4e7` before these documents. Recheck identity/diff at execution; do not transplant assumptions from the older Downloads checkout.

Critical path: targeted failure checks → transactional effect admission → actual Preview/Pause/Resume → lifecycle-aware Apply → source-bound GUI proof. Setup and bake/cache each form a later independent checkpoint. Panel changes consume working actions, not the other way around.

Source facts checked while planning:

- `effects.build_effect_stack_for_admission` rejects dirty/unapplied stacks via `production_blocking_reason`.
- `SimulationAdmission._configure_candidate` already builds and installs effect candidates; `initialize` prepares replacement before authority cutover, then prepares outputs and captures applied settings. Failures after cutover currently release the new session. Do not promise rollback of an already-released old solver.
- `effects._try_live_parameter_update` already supports direct compatible parameter updates; preserve it.
- `pending_settings._apply_pending` already chooses RESET versus REBUILD and preserves pending state on failure. It is the policy seam, not a template to copy into an operator.
- `FLUIDSIM_OT_toggle_live` and cache start currently arm playback without starting animation.
- `setup_selected` describes itself as a transaction; that description is not rollback proof.
- Modal bake callbacks currently reference `context.scene`; scene-switch behavior needs an executable check.

All new functions and operator IDs below are proposed interfaces, not claims that they already exist. Preserve public legacy behavior unless a task explicitly adds orchestration to it.

## File ownership map

| Files | Responsibility |
|---|---|
| `blender/effects.py`, `blender/simulation_admission.py` | Immutable effect preparation, candidate validation/install, applied evidence |
| `blender/scene_bridge.py` | Compatibility wiring only, no orchestration policy |
| `blender/playback.py` | Preview preparation/eligibility and backend activation |
| `blender/pending_settings.py` | Apply/restart assessment and transition selection |
| `operator_domains/session.py`, `operators.py` | New actions, registration, GUI context, confirmation |
| `blender/setup_wizard.py`, `operator_domains/source_roles.py` | Scoped setup rollback and setup/start composition |
| `blender/cache_bake.py`, `operator_domains/cache.py` | Bake/cache transactional and scene-bound behavior |
| `blender/ui_scene.py`, `blender/ui_view3d.py`, `blender/ui_physics.py`, `blender/effect_ui.py`, `panels.py` | Consistent intent-based presentation |
| `tests/test_simulation_admission.py`, `tests/test_phase4_pending_interaction.py`, `tests/test_scene_bridge_cache_playback.py`, `tests/test_operator_panel_behavior.py` | Existing focused regression harnesses |
| New `tests/blender_artist_workflow_acceptance.py` | GUI timeline, effects, setup, context, and failure acceptance |
| New `docs/ARTIST_WORKFLOWS.md` | User actions, consequences, retained scripting controls |

Paths beginning with `blender/` or `operator_domains/` above are relative to `fluidSim_warp_blender51/`. Tests and docs are relative to the repository root.

## Task 1: Establish the regression boundary

**Files:** Read the owners above; modify `tests/test_simulation_admission.py`, `tests/test_scene_bridge_cache_playback.py`, `tests/test_operator_panel_behavior.py` only for executable missing coverage; create `docs/superpowers/plans/2026-09-12-artist-workflow-evidence.md`.

**Interfaces:** Consumes existing `SimulationAdmission.initialize(scene, *, depsgraph=None, ignore_memory_safeguard=False)`, cache start operator, and modal bake callbacks. Produces an evidence table of hypothesis, caller/sibling path, executable test, observed outcome, and owning task. No new production interface.

- [ ] Recheck `git rev-parse --show-toplevel`, `git rev-parse HEAD`, `git status --short`, nearest instructions, Python and Blender executable resolution. Record actual source import paths in later Blender receipts.
- [ ] Run the focused baseline:

```powershell
python -m pytest -q tests/test_simulation_admission.py tests/test_pending_settings_registry.py tests/test_phase4_pending_interaction.py tests/test_operator_panel_behavior.py tests/test_scene_bridge_cache_playback.py
```

- [ ] Extend the existing admission fault-injection test to snapshot solver identity and applied effect evidence. Use the existing `_scene`, `_owner`, `_Session`, and monkeypatch patterns in that file; inject failure at candidate configuration before cutover. Add these assertions to the existing failure case:

```python
assert owner.session.solver is old_solver
assert scene.fluidsim_effects.applied_signature == old_signature
assert scene.fluidsim_effects.applied_snapshot_json == old_snapshot
```

Capture `old_solver`, `old_signature`, and `old_snapshot` before invoking admission; add `fluidsim_effects` to the existing scene double with those fields. Also verify a successful sibling admission installs the new solver exactly once. Do not replace real RNA checks with only SimpleNamespace tests.
- [ ] Characterize rejected cache activation: invoke the real cache-start execute method in the existing operator harness with cache activation rejected; assert `live` stays false, no live initializer is called, and the cache error remains visible. A passing test closes that hypothesis; a failing test belongs to Task 6.
- [ ] Record setup rollback, restart-before-admission, and modal scene-switch checks as Task 5/6 verification cases. Do not declare them confirmed defects from source inspection alone.
- [ ] Run each added test, retain useful assertions, and update the evidence table. Commit only this task's reviewed paths at an authorized execution checkpoint; never broadly stage existing changes.

## Task 2: Prepare effects inside admission

**Files:** Modify `fluidSim_warp_blender51/blender/effects.py`, `blender/simulation_admission.py`, and compatibility wiring in `blender/scene_bridge.py` only if required; extend `tests/test_simulation_admission.py` and `tests/blender_custom_effects_acceptance.py`.

**Interfaces:** Keep `build_effect_stack_for_admission(scene) -> tuple[EffectInstanceV2, ...]`, but allow eligible draft snapshots instead of requiring an already-applied signature. Add `publish_admitted_effects(scene, instances: tuple[EffectInstanceV2, ...]) -> None`, wrapping existing applied-state publication. Admission must retain both full authoring instances for evidence and effective execution instances for compilation; never publish only the filtered enabled subset against all RNA items.

- [ ] Add a real Blender acceptance case that adds a built-in without calling `apply_stack`, then initializes. Establish failure on the baseline before implementing:

```python
from fluidSim_warp_blender51.blender import effects, scene_bridge

scene = bpy.context.scene  # existing acceptance fixture: real domain/source
effects.clear_effects(scene)
effects.add_builtin_effect(scene, "fluidsim.radial_pulse")
assert not scene.fluidsim_effects.applied_signature
sess = scene_bridge.initialize_session(scene)
assert sess.owns_scene(scene)
assert scene.fluidsim_effects.applied_signature
assert not effects.production_blocking_reason(scene)
```

- [ ] Change stack capture to validate migration and exact-content trust without using dirty/applied equality as an admission prerequisite. Preserve background policy. Capture immutable full/effective snapshots once; reuse candidate validation and sandbox execution in `_configure_candidate`.
- [ ] Add publication at the successful admission boundary using the captured full snapshot:

```python
def publish_admitted_effects(scene, instances):
    _publish_applied_state(scene, instances)
```

The call belongs after successful candidate installation and required initialization, not before allocation or inside snapshot capture. Validate current authoring identity before cutover; reject stale candidates. If post-cutover initialization fails, preserve accurate failure/applied evidence and existing cleanup semantics; do not claim the old solver survived.
- [ ] Test enabled/disabled/muted stacks, removal of the last effect, untrusted imported source, compile/sandbox failure, and stale snapshots. Assert failure cannot advance the applied signature. Keep `effect_identity` strict at bake/resume identity checks; do not weaken it globally to admit drafts.
- [ ] Run focused admission and custom-effects tests; run `blender --background --factory-startup --python tests/blender_custom_effects_acceptance.py` and `tests/blender_simulation_admission_acceptance.py` as separate Blender commands. Record actual markers and errors.
- [ ] Review the diff and checkpoint only these changes with `fix: prepare custom effects transactionally during admission`.

## Task 3: Deliver actual Preview, Pause, and Resume

**Files:** Modify `fluidSim_warp_blender51/blender/playback.py`, `operator_domains/session.py`, `operators.py`; extend `tests/test_operator_panel_behavior.py`, `tests/test_scene_bridge_cache_playback.py`; create `tests/blender_artist_workflow_acceptance.py`.

**Interfaces:** Add `playback.prepare_live_preview(scene, *, depsgraph=None) -> None`; returns only when the owned session/current effects are ready, otherwise raises a specific error. Add operators `fluidsim.preview_start` and `fluidsim.preview_pause`. Existing `toggle_live`, initialize, step, and cache operator IDs remain registered. Start is idempotent, not a toggle.

- [ ] Write failing operator tests for fresh start, resume, duplicate start, missing GUI context, foreign scene owner, active bake, and rejected effect. A resume must not invoke initialize/reset when the owned simulation is compatible.
- [ ] Implement preparation through existing ownership, admission, and pending eligibility. Do not use a chain of `bpy.ops` calls to implement runtime policy. Existing compatible parameter binding remains the fast path. Structural changes use Task 4's restart assessment; until Task 4 lands, reject them clearly when progress would be lost.
- [ ] Resolve a window whose scene is the target scene and a valid animation operator override before changing live state. Use Blender's supported context override and operator poll. The adapter's essential sequence is:

```python
# Inside the validated window/screen/area override:
if not bpy.ops.screen.animation_play.poll():
    raise RuntimeError("No editor can start preview playback")
prepare_live_preview(scene, depsgraph=depsgraph)
scene.fluidsim.live = True
if not window.screen.is_animation_playing:
    result = bpy.ops.screen.animation_play()
    if result != {"FINISHED"}:
        scene.fluidsim.live = False
        raise RuntimeError("Preview was prepared but playback did not start")
```

Catch exceptions as well as non-FINISHED results; leave valid prepared state paused. Never call animation_play when already playing. Pause uses animation_cancel with `restore_frame=False`, disarms preview, and keeps the owned solver and frame. Do not switch a foreign window's scene to obtain context.
- [ ] Build the GUI test on existing acceptance source-registration conventions. Use Blender timers so the event loop advances naturally; do not simulate success with repeated `scene.frame_set`. Add this initial operator assertion:

```python
assert bpy.ops.fluidsim.preview_start() == {"FINISHED"}
assert bpy.context.screen.is_animation_playing
```

On subsequent timer callbacks require increased scene and session frame counters, nonzero particles, changed owned output, and increased production effect execution evidence using the existing custom-effects acceptance accessors. Then pause and resume; compare the same solver object and preserved cadence counters. Include an effect-disabled control or dispatch proof, not just a panel label.
- [ ] Exercise invocation from Scene Properties and View3D, no Timeline editor, already-playing screen, and a second scene/window. Background invocation must report no GUI context cleanly; legacy headless initialize/step remains functional.
- [ ] Run focused tests and the new non-background acceptance via an isolated source-loaded Blender MCP session. Record source path, Blender version, scene, device, operator results, output/dispatch proof, and paused final state.
- [ ] Checkpoint `feat: start and pause live preview in one action` after these checks pass. This is the first independently usable milestone.

## Task 4: Unify changes and restart consequences

**Files:** Modify `fluidSim_warp_blender51/blender/pending_settings.py`, `blender/effects.py`, `operator_domains/session.py`, `operators.py`; extend `tests/test_phase4_pending_interaction.py`, `tests/test_pending_settings_registry.py`, and the new GUI acceptance.

**Interfaces:** Add `pending_settings.apply_changes(scene, *, allow_restart: bool = False, depsgraph=None) -> bool` delegating to `_apply_pending`; returns whether a transition occurred. Add `changes_require_restart(scene) -> bool` using existing disposable-start and registry policy plus structurally incompatible effects. Add `fluidsim.apply_changes` and `fluidsim.restart_simulation`. Do not duplicate setting-tier lists in UI/adapters.

- [ ] Write tests for safe live edit, RESET edit, REBUILD edit, mixed settings/effects, disposable initial state, foreign owner, and failed admission. Verify mixed edits install once and pending reasons clear only on success.
- [ ] Implement the explicit guard before any destructive transition:

```python
if changes_require_restart(scene) and not allow_restart:
    raise RuntimeError("Restart required: current preview progress will be cleared")
```

Use existing `_apply_pending` for setting transitions and admission for structural effects needing replacement. Never reset and then separately reinitialize the same request. Evaluate current state again at execute time; confirmation cannot authorize a different scene, cache deletion, or stale draft.
- [ ] In interactive invocation, show `Restart simulation from frame N? Current preview progress will be cleared.` only for non-disposable progress. Cancellation leaves draft and progress intact. Non-interactive execution defaults to refusing loss unless `allow_restart=True` is explicitly supplied. Restart Simulation is itself an explicit restart action.
- [ ] Preserve playing/paused intent around successful Apply: resume if previously playing; remain paused if previously paused. On failure remain paused with actionable pending/error state. Avoid playback during candidate preparation.
- [ ] When structural effect edits occur during playback, pause before the next step rather than claiming the old stack implements the draft. Keep parameter-only updates direct and verify solver/kernel identities are unchanged. Controller/timing/order changes follow their actual compatibility classification.
- [ ] Run the pending-registry, pending-interaction, and operator suites plus `blender_phase4_live_interaction_acceptance.py`, `blender_handler_persistence_gate.py`, and the expanded GUI acceptance. Add regression evidence for confirmation cancellation and retry after a failed candidate.
- [ ] Checkpoint `feat: apply simulation changes with clear restart consequences`.

## Task 5: Compose safe setup and preview

**Files:** Modify `fluidSim_warp_blender51/blender/setup_wizard.py`, `operator_domains/source_roles.py`; extend `tests/blender_quick_setup_idempotence.py` and the new GUI acceptance.

**Interfaces:** Preserve `setup_selected(context, mode: str, preset_id: str = "") -> SetupResult`. Add a `start_preview` Boolean property to the existing guided-setup operator, default false for scripting compatibility; primary UI supplies true, secondary supplies false. Successful setup invokes the new preview action only after setup commits.

- [ ] Add injected failures after source tagging, domain fitting, and output setup. Capture preexisting object roles/settings/transforms/links and created datablock identities. Assert each failed setup restores touched existing data and removes only datablocks created by that invocation. Include preexisting user material/node modifications.
- [ ] Implement a setup-local rollback journal, not a general transaction framework. Record restoration before each mutation and only delete exact newly created owned datablocks. Treat optional rendering warnings separately from failed domain/source setup.
- [ ] Extend the existing real quick-setup test with both operator options. The preview branch must reach real frame advancement and nonzero liquid; the setup-only branch must remain paused. Repeat setup and verify no duplicate domain/output objects.
- [ ] If playback fails after successful setup, preserve the valid setup and report `Scene configured; preview could not start: <reason>`. Do not roll back a complete setup merely because a window disappeared.
- [ ] Run `blender --background --factory-startup --python tests/blender_quick_setup_idempotence.py`, then GUI setup/preview acceptance. Checkpoint `feat: configure and preview liquid from guided setup`.

## Task 6: Verify and repair cache/bake hazards before simplifying those controls

**Files:** Modify only confirmed defects in `fluidSim_warp_blender51/blender/playback.py`, `blender/cache_bake.py`, `operator_domains/cache.py`, `operator_domains/session.py`; extend `tests/test_scene_bridge_cache_playback.py`, `tests/test_operator_panel_behavior.py`, `tests/blender_cache_export_pipeline.py`, and the GUI acceptance.

**Interfaces:** Preserve production bake controller and cache formats. Modal operator callbacks capture the initiating scene and use it consistently for tick, cancellation, RNA updates, unregistering, and path reservation cleanup. No new public runtime abstraction.

- [ ] Run Task 1's rejected-cache test. If it fails, make activation commit cache/live flags only after successful cache validation. Assert no live initializer or frame-simulation call after rejected activation, including the next real handler event.
- [ ] Add a temporary-cache restart test: save existing file bytes, inject admission failure, attempt restart, compare every existing file byte. Verify reservations are released and a subsequent valid restart works. Never run fault injection against a user's actual cache.

```python
before = {p.name: p.read_bytes() for p in cache_dir.iterdir() if p.is_file()}
# Run the existing restart harness with admission forced to fail.
after = {p.name: p.read_bytes() for p in cache_dir.iterdir() if p.is_file()}
assert after == before
```

- [ ] If restart destroys data before successful preparation, reorder non-destructive validation and admission ahead of replacement. Use recoverable staging of the exact managed cache only if the existing writer requires it; preserve original data until replacement is established, and test failed filesystem promotion/restoration. Do not build a new cache store or weaken identity checks.
- [ ] Add GUI modal bake test: start on scene A, switch the window to B, advance timer events, then cancel. Assert only A advances/receives bake RNA, B's settings/output remain unchanged, and all A reservations/timers are released. Handle initiating-scene deletion by cancellation and cleanup without dereferencing a removed RNA object.
- [ ] Compare sibling resume and ordinary bake-start paths. Record each hypothesis as reproduced/fixed or disproven with test evidence. A disproven hazard requires no production edit.
- [ ] Run cache/playback/operator tests and `blender_cache_export_pipeline.py`, plus the modal GUI case. Checkpoint confirmed changes with `fix: preserve cache and scene ownership across bake actions`.

## Task 7: Present working actions consistently

**Files:** Modify `fluidSim_warp_blender51/blender/ui_scene.py`, `blender/ui_view3d.py`, `blender/ui_physics.py`, `blender/effect_ui.py`, `panels.py`; update `tests/test_operator_panel_behavior.py`; create `docs/ARTIST_WORKFLOWS.md`; update README workflow instructions.

**Interfaces:** Consume new operator IDs and existing shared lifecycle/eligibility presentation. Reuse Basic/Advanced state; do not introduce a second UI mode or mirrored runtime state.

- [ ] Replace normal Initialize/Compile & Apply/armed Start sequences with Preview, Pause/Resume, state-dependent Apply Changes, and Restart Simulation. Keep effect parameters beside effects and quality settings beside simulation quality.
- [ ] Remove structural settings from Live Tuning:

```python
LIVE_TUNING_PROPS = (
    "flip_ratio",
    "particle_display_fraction",
)
```

Retain Voxel Size and Liquid Sampling in quality controls, with pending/restart status derived from pending_settings.
- [ ] Make Set Up & Preview the primary guided action and Set Up Only secondary. Keep live and cached preview clearly labeled; production bake remains separate. Do not change legacy cache arming operators' Python semantics merely to share button labels.
- [ ] Move initialization/release/manual stepping, forced render-node updates/repair, headless bake commands, and detailed telemetry under existing Advanced/Diagnostics. Automatic output preparation must preserve user node/material edits. Essential errors and a repair action remain visible when relevant.
- [ ] Apply the same disclosure in Physics output attribute inventories and View3D controls. Verify operator registration/legacy IDs with existing runtime UI tests; use actual Blender panel drawing rather than only source-string assertions.
- [ ] Write user documentation around adding an effect and pressing Preview, pause/resume/restart consequences, safe versus structural edits, cached playback, production bake, and how to reveal diagnostic controls. State remaining context/ownership limitations plainly.
- [ ] Run the UI/operator suites and inspect Basic/Advanced in an isolated live Blender session. Checkpoint `refactor: organize simulation controls around artist actions`.

## Task 8: Complete runtime proof and handoff

**Files:** Finalize `tests/blender_artist_workflow_acceptance.py`, the evidence document, and `docs/ARTIST_WORKFLOWS.md`. Modify `tools/check_demo_blends.py` or the production gate only when required to wire the new acceptance without changing unrelated demo assets.

**Interfaces:** GUI acceptance emits `ARTIST-WORKFLOW-ACCEPTANCE-PASS` only after all required timer-driven cases complete. Failure exits nonzero and records the failed action. Receipt contains actual source path/HEAD, Blender version, scene/domain identity, GPU, observed frames, output/effect execution evidence, and tested cases.

- [ ] Run `python -m pytest -q` after focused suites pass. Fix regressions without weakening assertions; classify any baseline failures using Task 1 evidence.
- [ ] Run the required admission lane from AGENTS.md when admission changed: capacity initialization/planner, memory safety, sourcing, source lifecycle, particle growth, settings projection, and simulation admission pytest suites; run Blender admission acceptance.
- [ ] Run pending-setting and persistent-handler acceptance after those changes. Run custom-effects acceptance and effect-add invoke acceptance to preserve the earlier popup repair.
- [ ] Reopen `demos/custom_effects_vortex.blend` from this source-loaded add-on, exercise new Preview/Pause/Resume/Restart, and repeat with `demos/dam_break.blend` as the non-effect sibling. Require real domains, nonzero solver state, production effect evidence, changing owned output, and reopen/restart behavior. Do not overwrite tracked demos merely to simplify proof.
- [ ] Run `blender --background --python tools/check_demo_blends.py`. Run the production gate with GUI inclusion once changes are integrated, after inspecting its current side effects and protecting unrelated artifacts. Use the repository's prescribed validation commands for any reports refreshed by that gate; do not fabricate clean-release evidence for a dirty source checkpoint.
- [ ] Run `git diff --check`; inspect the complete diff for duplicated lifecycle policy, swallowed errors, leaked timers/reservations, unconditional node replacement, stale effect evidence, and implicit cache deletion.
- [ ] Update the evidence document with checks actually executed, outcomes, source/runtime identity, remaining limitations, and spec requirement coverage. Record a final scoped checkpoint. Leave the user review scene open and paused only when handing off a visible GUI session; verify the paired runtime and visible scene, not just process creation.

## Execution and review gates

- Execute inline in task order; Task 6 can be independently reviewed after Preview works. Serialize all Blender/CUDA work.
- Each task follows failing behavioral check → smallest owner-level change → focused pass → diff review → scoped checkpoint. Existing passing characterization tests stay passing; they do not require artificial failures.
- A checkpoint may proceed without bake-specific changes when Preview's own checks pass. Full workflow completion still requires Tasks 5–8.
- This plan does not authorize modifying a normal Blender installation, publishing, deleting user caches, or reconciling unrelated experimental branches.
- If a check exposes a second unplanned subsystem, preserve the last verified milestone and record the new scope before expanding.

## Plan self-review

- Coverage: requirements 1–3 → Tasks 2–4; 4 → Task 4; 5 → Tasks 1–2; 6 → Task 5; 7 → Task 6; 8–10 → Task 7; 11 → Tasks 1/6; 12 → Tasks 3/8.
- Interfaces: proposed playback/pending/effect functions are defined in the producing tasks; operator IDs are registered in operators.py and retained by UI consumers.
- Risks remain qualified: candidate/output cutover ordering, setup rollback, cache rejection, bake replacement, and modal scene switching require executable evidence. UI simplification alone closes none of them.
- No runtime changes or test passes are claimed by this planning document.
