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

### Updated verification checkpoint: product source `2365aca`

The full host suite passed **1537 tests in 108.43 seconds**. The fresh isolated Blender 5.1 source-copy session (PID 51936, port 9877) passed paired runtime/addon identity checks and the following actual GUI cases before being explicitly closed:

| Receipt under `artifacts/` | Observed proof |
| --- | --- |
| `artist_workflow_effects_gui_2365aca.json` | 24 checks; automatic effect admission, actual Radial Pulse dispatch, hot parameter binding with same kernel/solver, changing output, structural pause/consent, restart after real progress |
| `artist_workflow_dam_break_gui_2365aca.json` | Non-effect startup sibling, then structural edit/apply/restart |
| `artist_cache_preview_gui_2365aca.json` | Real production bake followed by cache playback from frame 1 to 4 without an active live solver; temporary cache removed |
| `artist_bake_gui_2365aca.json` | Real modal bake keeps initiating-scene ownership across scene switch and cancellation; cleanup verified |
| `artist_bake_deleted_scene_2365aca.json` | Deleting a disposable bake owner releases controller/session/cache reservation and preserves original scene |
| `artist_setup_preview_gui_2365aca.json` | Set Up & Preview creates a domain and advances a new 512-particle solver with owned output |
| `artist_effect_add_invoke_2365aca.json` | Actual Add Effect popup invocation returns RUNNING_MODAL; no NoneType return failure |

Launch identity is retained in `artifacts/artist_launch_2365aca.json`. These are targeted workflow proofs, not aggregate production certification. Main tracked demos and the regular Blender installation/profile were not changed.

Independent receipt review allows the targeted workflow claims and explicitly rejects aggregate certification until remaining gates pass. A direct SHA-256 comparison after that review checked all **244 tracked package files** against the retained session `5bedb4d235ff4d87bc911401b4b8f125` temporary add-on: **zero missing and zero mismatched files**, with main product source unchanged at `2365acac2128d26f1ec4fd60e56c5f160410b22e`. Setup's compact GUI receipt records its successful terminal state, not a complete created-ID journal; creation/rollback invariants additionally rely on the dedicated setup test. Negative cache/effect admission and failed restart preservation rely on focused host tests, not those positive GUI receipts.

The broader headless production gate ran in a separate clean proof checkout at the same source (`C:/tmp/fluidsim-artist-proof-20260913`). It failed in the cache-export acceptance observer reading `scene.fluidsim.live_surface_publish_cadence`, a setting now owned by the Domain. Production Output Publication already uses the authoritative domain accessor. The fixture now uses Domain-owned solver settings and retains Scene-owned presentation settings. Its bounded 250,000 whitewater capacity preserves nonzero/no-loss export proof; the real focused Blender run reports `CACHE-EXPORT-PIPELINE-PASS`, and independent review allows the correction. Broad GUI probes also use actual Domain ownership and a verified preloaded add-on route, retaining Basic/Advanced and legacy-action coverage. The full host suite after these validation-input changes passed **1537 tests in 61.92 seconds** (`artifacts/artist_host_suite_validation_inputs.log`). The aggregate gate remains **not passed** pending rerun; its original failure log is `artifacts/artist_headless_gate.log` in the proof checkout. Broad GUI/playback acceptance and all-demo proof are still pending.

- Final source review and generated ownership map; full pytest pass.
- Frozen source identity; isolated source-loaded Blender launch and paired runtime gate.
- Real Preview/Pause/Resume/Apply/Restart, effect dispatch, changing owned output, parameter kernel reuse, and non-effect sibling.
- Setup rollback/idempotence; modal bake scene switching/deletion; admission, pending, handler, and custom-effect acceptance.
- All-demo checks and production validation with GUI launch routed through the canonical isolated launcher. Do not infer aggregate production certification from separate partial checks.
- Final receipt paths, exact runtime/source identities, and remaining limitations.

## Live findings and repairs

The diagnostic suffix at `6fa345d` passed mesh/force sources, live settings, admission, closed-domain stability, custom effects, and source UI. The capacity command initially omitted its required `--json-out` argument (a command error); the corrected invocation passed and wrote `artifacts/artist_capacity_6fa345d.json` in the proof checkout. Attribute visualization exposed a second inherited **shared-path ownership bug**: `prepare_live_payload` resolved Domain settings but passed Scene settings into all three `HostFrame.output_attributes` calls. Missing Scene toggles suppressed live particle/whitewater/surface inventories; selected visualization fallback could still publish speed while omitting velocity. All three calls now consume the Domain. Scene presentation controls remain unchanged. Cache capture already uses solver settings and cache publication reads persisted inventory, so that contrasting path required no change.

The focused output suite passed 39 tests, including poisoned Scene values and actual Domain identity checks for all three consumers. Real Blender attribute acceptance passed: velocity and alias aligned to 8,766 points, Geometry Nodes existence probe 8,766/8,766, and `ATTRIBUTE-VIZ-ACCEPTANCE-PASS` in `artifacts/attribute_viz_acceptance_focused_20260913.log`. Ownership map is current. These are focused results; the final aggregate and GUI proof remain pending.

At `8bc3244`, the aggregate passed cache export, memory safety, handler persistence, viewport stepping, and background acceptance, then found stale migration-test assertions. The current schema intentionally stores its guard on Scene, not Domain. The two migration fixtures now author genuine legacy RNA within `legacy_scene_settings`, use current field ownership, and poison hidden legacy IDProperties for equivalence. Review rejected an intermediate weakening of the exhaustive field comparison; the final acceptance captures **every Domain migration field** before save, verifies exact inventory/value equality after migration, and repeats full equality after the second reopen for both designated and fallback-created Domain branches. Fresh focused Blender logs report `DOMAIN-MIGRATION-ACCEPTANCE-PASS` and `DOMAIN-MIGRATION-EQUIVALENCE-PASS`; the latter matches active-voxel average/minimum/maximum exactly (27306/24576/32768). Independent review allows both inputs. No migration product code changed. The full host suite with the capacity repair passed **1537 tests in 94.57 seconds** in `artifacts/artist_host_suite_capacity_recovery.log`.

The aggregate rerun at validation commit `456a93c` passed cache export and then stopped at the memory-safety fixture, which had no designated Domain. Baseline comparison confirmed that Domain-only initialization already existed at `08bac736`; this was inherited fixture drift. Correcting the fixture exposed an inherited **shared-path product bug**: `FLUIDSIM_OT_apply_capacity_suggestion.execute` obtained recommendations from Domain settings but wrote, read back, and rolled them back through Scene settings. The adapter now targets the authoritative Domain, retains Scene status/domain pointer ownership, and rejects a missing Domain before mutation. This affects capacity suggestions whose changes contain solver fields (including coarser resolution and particle capacity); domain-bounds and fresh-budget/retry paths retain their existing behavior.

Focused tests with poisoned Scene mirrors and silent RNA clamping passed (93 tests), and the actual Blender run reports `MEMORY-GATE-RECOMMENDATION coarser_resolution`, successful stepping, and `MEMORY-SAFETY-GATE-PASS` in `artifacts/memory_safety_gate_focused_20260913.log`. Independent review allows the repair. Eleven other post-migration acceptance fixtures were audited and corrected as one family, preserving intentional migration/equivalence tests and the already-correct demo builder/checker. Their broad runtime proof remains pending until the aggregate rerun; source-only checks are not runtime certification.

At `f39c6ab`, Blender exposed unavailable Preview/Apply RNA operators despite true direct Python polls. Their registered subclasses were interfering with the parents. `21e4db2` separates each registered operator from shared unregistered mixins; a failing MRO regression became passing. The fresh isolated GUI then passed 21 artist checks: frames 5 and 8, 800 particles, 7 Radial Pulse dispatches, actual edited parameter binding with retained kernel/solver, Pause/Resume, and structural-edit consent/Apply/Restart. Receipt: `artifacts/artist_workflow_effects_gui_21e4db2.json`.

Modal cancellation while viewing another scene exposed a shared property-callback bug: restoring scene A's live flag changed scene B's status. `artifacts/artist_bake_gui_diagnostic.json` captures the before/after. Both live and cache mode callbacks now resolve their actual owning Scene from `id_data`; independent source review allows this repair. Main combined callback/preview/bake/playback checks: 86 passed.

Real setup rollback exposed callback side effects during restoration and preset application before the material baseline was captured. Forward setup and rollback now use a public, nesting-safe pending callback suppressor and capture existing outputs before property/preset writes. Non-Nodes modifiers' unsupported ID-property access is handled explicitly. The parent reran the Blender test: `artifacts/artist_setup_background.log` contains `QUICK-SETUP-IDEMPOTENCE-PASS`, with no traceback. The source lane reports 99 related tests passed and independent review ALLOW. Live Setup & Preview composition remains to be tested separately.

All failed receipts are retained as diagnostic history, not passing certification. No tracked demo was saved during these tests. Owned GUI processes 51924 and 52244 were closed before the next source iteration; regular Blender preferences and installed add-ons were not modified.

At `7836e75`, the full suite passed 1535 tests in 140.41 seconds. Fresh GUI receipts passed core effect playback, cross-scene modal cancel, and deletion of the disposable bake owner (including session/cache reservation cleanup). The no-effect sibling required a test-only correction: an empty effect stack has no applied-effect signature. `f3d1b27` corrected that assertion, and the dam-break sibling passed. Manual deferred GUI composition also proved Set Up & Preview creates a domain and advances a new 512-particle solver to frame 7 with owned output.

The subsequent positive Cache Preview test exposed Blender's instance-form `bl_idname` (`WM_OT_name`, unlike the class registration string `wm.name`). The adapter's string comparisons selected live preview instead of cache and failed to force an explicit restart. Both decisions now use ordinary action flags; two canonical-instance-ID regressions failed before and passed after the repair, with 162 related tests passing. The GUI restart check now requires at least three actual solver frames before checking the restart. Earlier restart-at-start receipts are not sufficient proof of that behavior. Independent source review allows the flag repair; affected GUI proof is being rerun.
