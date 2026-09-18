# Artist workflow final verification

Status: complete. All scoped implementation, verification, installer, saved-demo and visible-handoff work is closed.

Source checkout: `C:/Users/Tyler/.codex/worktrees/c6d1/liquid_warp_blender51_port`, branch `codex/artist-workflows-20260913`.
Proof checkout: `C:/tmp/fluidsim-artist-proof-20260913`.

Final product source is `1b95b8ab3373b57d575e26a5f4d4d562308acc76`; validation-only HEAD is `98cc8a17ea61cfd0ed9e67e4d4cc75e83049e1b3`. There are no package changes between them. Later closure commits change documentation only. Current GUI source-copy checks compare all **244 tracked package files** and report zero mismatches.

The user explicitly resumed the autonomous loop to finish all open work. This includes repairing the exposed adhesive demo failure and verifying the original installer complaint in a disposable profile. It does not authorize changing the regular installation/profile or publishing a release.

| Completion gate | Result and evidence |
|---|---|
| Implementation and independent source reviews | **ALLOW**: reviewed core, setup, bake, UI, capacity recovery, output ownership and final background guard. Independent final review found no product-source blocker; historical ledger retains findings and scoped commits. |
| Full host suite | **1542 passed in 94.43s**, `artifacts/artist_host_suite_guard_final.log`; includes installer, demo, profiler-owner and background Preview regressions. |
| Real isolated installer | **PASS** at final product source `1b95b8ab3373b57d575e26a5f4d4d562308acc76`: two installed smoke runs, uninstall, matching ZIP hashes, unchanged normal profile. `artifacts/installer_proof_1b95b8ab3373_report.json`, `_identity.json`, `_profile_integrity.json`, and `.log`. Windows runtime tested; Linux ZIP built, not Linux-runtime certified. |
| Interrupted installer cleanup | **PASS**: first disposable installation removed through isolated extension CLI; original failure retained. `artifacts/installer_proof_d0d449e97b9c_cleanup.json`. |
| Saved adhesive demo | **PASS**, committed `718f9b3`: clean factory-startup build and reopen check retain 1M particles, 250 frames, viscosity A/B and nonzero whitewater from frame 30. `artifacts/adhesive_calender_rebuild_isolated_final.log` and `artifacts/adhesive_calender_saved_check_isolated_final.log`. |
| Full headless aggregate | **HEADLESS-GATE-PASS** at final product source `1b95b8ab3373b57d575e26a5f4d4d562308acc76`; every headless acceptance lane invoked by the default gate, demo analysis 18/18 and all-demo gate pass. Proof checkout `artifacts/artist_headless_gate_guard_final.log`. Proof checkout preserves six generated dam-break/crown-splash outputs; this is not a clean-tree or GUI-inclusive aggregate claim. |
| Current-source broad GUI | **55/55 PASS** with the final background guard, `artifacts/artist_broad_gui_guard_final.json`. Profiler setup uses the authoritative Domain, matching Admission. All 244 tracked package files match the temporary runtime copy: `artifacts/artist_guard_gui_source_identity.json`. Unobstructed viewport screenshot inspected after splash dismissal. |
| Saved-demo GUI playback | **PASS** with the final guard, `artifacts/artist_saved_demo_gui_guard_final.json`, validated immediately against the exact proof dam-break file. |
| Current artist workflow GUI | **24 effect / 21 no-effect checks PASS** with the final guard, `artifacts/artist_workflow_effects_gui_guard_final.json` and `artifacts/artist_workflow_no_effect_gui_guard_final.json`. Historic `2365aca` also covers setup/cache/modal bake. |
| Actual consent and editor contexts | **14 PASS** with the final guard, `artifacts/artist_context_consent_gui_guard_final.json`: Properties Preview, already-playing View3D, foreign scene rejection, real ESC cancellation, stale settings/scene rejection, real RET confirmation. |
| Background no-GUI rejection | **PASS** with the final guard, `artifacts/artist_background_context_verified_final.json/.log`. Blender exposes a pseudo-window in background, so a new early guard rejects both Preview modes before preparation. Initialize, Step, Bake and non-playing Apply are unchanged. |
| Visible custom-effects handoff | **PASS**: exact main `demos/custom_effects_vortex.blend`, PID **41380**, port **9877**, temporary source add-on enabled, all 244 package files equal. Real timeline frames **3 → 19**, **28,672 particles**, **61 Vortex Force calls**, changing owned output, applied stack, CUDA, Basic UI, paused. `artifacts/artist_custom_effects_handoff.json` and `.png`; final paired gate confirms the same process/file/add-on. Window left open for the user. |

## Evidence boundaries

- Source-loaded GUI proves development runtime behavior, not persistent installation; the separate installer lane supplies that proof without changing the regular profile.
- GUI components use the canonical isolated MCP launcher. Do not label separate component passes `FULL-PRODUCTION-GATE-PASS` unless that aggregate actually ran.
- Pre-cutover rejection preserves the old solver; post-cutover output failure releases the candidate. No full rollback claim after old-session release.
- CUDA numerical nondeterminism remains unchanged; no bitwise repeatability claim.
- The initial unisolated adhesive build/check passed, but loaded unrelated installed startup handlers. Its receipts are diagnostic only; factory-startup results supersede them. Command documentation now makes this isolation explicit.
