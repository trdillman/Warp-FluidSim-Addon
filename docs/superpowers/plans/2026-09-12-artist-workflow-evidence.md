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
