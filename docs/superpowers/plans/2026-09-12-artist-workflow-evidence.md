# Artist workflow implementation evidence

Status: source implemented and reviewed; 1,529 host tests pass; not runtime-certified.

Worktree: `C:/Users/Tyler/.codex/worktrees/c6d1/liquid_warp_blender51_port`.
Initial HEAD: `08bac736a791eb32be63ff02b11671952dc8c4e7`.
No regular Blender profile or installed add-on has been changed.

## Implemented behavior

- Eligible effect drafts enter the existing admission transaction automatically.
- Preview prepares and starts real playback; Pause preserves solver/frame; Resume reuses compatible state.
- Pending Settings owns Apply/Restart selection and progress-loss consent.
- Cache Preview validates its frame and identity before releasing a live session; rejection does not initialize live mode.
- Guided Setup and bake restart have scoped failure recovery.
- Normal panels present artist actions; manual lifecycle and effect compilation remain Advanced compatibility controls.

## Executed checks

- Initial focused baseline: 183 passed in 5.63 seconds.
- First full suite during integration: 1497 passed, 9 failed. Failures were stale UI expectations, an incomplete admission test double, and generated ownership-map drift.
- Combined focused integration suite: 290 passed in 24.41 seconds.
- Second full suite: 1522 passed, 3 failed in 153.02 seconds. Two remaining UI-copy assertions were updated to require the new automatic-preparation message while retaining inspection-error visibility; their suite now passes 10 tests. Ownership map regeneration remains required after final source edits.
- Bake cleanup fault regression and sibling suites: implementation lane reports 74 passed, plus 2 operator bake tests. No GPU claims derive from these host tests.
- Python compilation of the new timer-driven artist acceptance script passed. The script has not yet run in Blender.

## Independent review

Core runtime source review found and verified repairs for cache/live coexistence and revoked-effect-trust bypass. Final core source verdict: no remaining source-level blocker in effects, admission, scene bridge, playback, pending settings, or session operators. This is not GUI/CUDA certification.

Setup/bake review identified overbroad rollback ownership, fatal post-admission backup cleanup, and empty-baseline cache restoration. All received shared-owner repairs and regressions. Final setup/bake source verdict: ALLOW. Setup tracks exact created IDs/touched keys; bake cleanup warns and retains recoverable backup on deletion failure while allowing the accepted bake to finish.

Final host verification before runtime: `python -m pytest -q --tb=short` completed with **1529 passed in 77.12 seconds**. The required focused admission/pending/effect/Preview/bake group passed 182 tests. Ownership-map check and git whitespace check passed. The third full run's only failure was generated-map drift; it was regenerated before this passing full run.

## Required remaining proof

- Final source review and generated ownership map; full pytest pass.
- Frozen source identity; isolated source-loaded Blender launch and paired runtime gate.
- Real Preview/Pause/Resume/Apply/Restart, effect dispatch, changing owned output, parameter kernel reuse, and non-effect sibling.
- Setup rollback/idempotence; modal bake scene switching/deletion; admission, pending, handler, and custom-effect acceptance.
- All-demo checks and production validation with GUI launch routed through the canonical isolated launcher. Do not infer aggregate production certification from separate partial checks.
- Final receipt paths, exact runtime/source identities, and remaining limitations.

## Live findings and repairs

At `f39c6ab`, Blender exposed unavailable Preview/Apply RNA operators despite true direct Python polls. Their registered subclasses were interfering with the parents. `21e4db2` separates each registered operator from shared unregistered mixins; a failing MRO regression became passing. The fresh isolated GUI then passed 21 artist checks: frames 5 and 8, 800 particles, 7 Radial Pulse dispatches, actual edited parameter binding with retained kernel/solver, Pause/Resume, and structural-edit consent/Apply/Restart. Receipt: `artifacts/artist_workflow_effects_gui_21e4db2.json`.

Modal cancellation while viewing another scene exposed a shared property-callback bug: restoring scene A's live flag changed scene B's status. `artifacts/artist_bake_gui_diagnostic.json` captures the before/after. Both live and cache mode callbacks now resolve their actual owning Scene from `id_data`; independent source review allows this repair. Main combined callback/preview/bake/playback checks: 86 passed.

Real setup rollback exposed callback side effects during restoration and preset application before the material baseline was captured. Forward setup and rollback now use a public, nesting-safe pending callback suppressor and capture existing outputs before property/preset writes. Non-Nodes modifiers' unsupported ID-property access is handled explicitly. The parent reran the Blender test: `artifacts/artist_setup_background.log` contains `QUICK-SETUP-IDEMPOTENCE-PASS`, with no traceback. The source lane reports 99 related tests passed and independent review ALLOW. Live Setup & Preview composition remains to be tested separately.

All failed receipts are retained as diagnostic history, not passing certification. No tracked demo was saved during these tests. Owned GUI processes 51924 and 52244 were closed before the next source iteration; regular Blender preferences and installed add-ons were not modified.

At `7836e75`, the full suite passed 1535 tests in 140.41 seconds. Fresh GUI receipts passed core effect playback, cross-scene modal cancel, and deletion of the disposable bake owner (including session/cache reservation cleanup). The no-effect sibling required a test-only correction: an empty effect stack has no applied-effect signature. `f3d1b27` corrected that assertion, and the dam-break sibling passed. Manual deferred GUI composition also proved Set Up & Preview creates a domain and advances a new 512-particle solver to frame 7 with owned output.

The subsequent positive Cache Preview test exposed Blender's instance-form `bl_idname` (`WM_OT_name`, unlike the class registration string `wm.name`). The adapter's string comparisons selected live preview instead of cache and failed to force an explicit restart. Both decisions now use ordinary action flags; two canonical-instance-ID regressions failed before and passed after the repair, with 162 related tests passing. The GUI restart check now requires at least three actual solver frames before checking the restart. Earlier restart-at-start receipts are not sufficient proof of that behavior. Independent source review allows the flag repair; affected GUI proof is being rerun.
