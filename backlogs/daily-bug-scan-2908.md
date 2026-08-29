# Daily bug scan — 2026-08-29

## Summary

- Repository: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port`
- Branch: `master`
- Scan boundary: commits after the automation boundary `2026-08-26T14:38:25Z`, plus the required ordered coverage pass.
- Initial HEAD: `8c6dbd20f07512195782f953dce3cd3a4a7ccbe7`.
- A concurrent documentation commit advanced HEAD during the scan to `6e65adfc2197b31964b0a6787c8622f50027fac5`; it changes only `AGENTS.md` and was checked separately.
- Pre-existing untracked planning/source files and concurrent unstaged code/demo edits were preserved. Only this report, the targeted loader catch, and its regression fixtures belong to this scan.

## Ordered review coverage

1. `6e65adf` — `AGENTS.md`: the new focused-test command is consistent with the existing test layout; no bug found.
2. `8c6dbd2` — species-v2 reconciliation plan: current implementation/candidate separation is explicit; no implementation bug found.
3. `3fa56cb` — swarm completion plan and claim matrix: reviewed for evidence drift; one report-only finding below.
4. `72733fb` — `pytest.ini` and operator-panel test isolation: package roots and submodule-identity assertion are coherent; no defect found.
5. `fe9c9e4` — all 24 changed production, test, and review/spec paths: one malformed-user-preset regression reproduced and fixed below; no second new defect was established.
6. Whole package inventory: all 252 tracked files under `fluidSim_warp_blender51/` and `fluidSim_synth/` were included in the coverage ledger. All 193 Python files parsed successfully; 19 JSON files parsed successfully; the tracked `.fseffect` archive passed ZIP integrity checking; `compileall` passed. Binary demo assets were preserved and not semantically re-authored.

## Findings, highest severity first

### P2 — fixed: malformed user preset input aborted preset discovery

`fe9c9e4` added per-argument normalization in `fluidSim_warp_blender51/sim/custom.py`, calling `dict(argument)` while loading each JSON preset. The loader caught `OSError` and `ValueError` but not `TypeError`. A JSON preset with `"arguments": [1]` or `"arguments": null` therefore raised uncaught `TypeError` and stopped preset discovery instead of skipping malformed input.

Minimal fix: catch `TypeError` at the existing loader boundary. Added regression fixtures to `tests/test_snippet_library.py` for both malformed shapes.

Evidence: the pre-fix reproduction raised `TypeError: 'int' object is not iterable` at `_normalized_preset_arguments`; after the fix, the malformed files are skipped and valid preset round-trip behavior remains green.

### P2 — report-only: historical completion claims are not pinned to proof

`reviews/swarm-review-2026-08-26-completion/CLAIM_VERIFICATION_MATRIX.md` reports `1136 passed` against mutable “current `master`” without a tested SHA, receipt, or artifact reference. The companion completion plan records full and Blender gate results without those proof anchors. These may have been valid historical results, but they are unsafe to reuse as current release evidence.

### P2 — residual: full-suite contract tests are order-sensitive

The full Python suite reached `1133 passed, 3 failed`. The failures were `test_relax_kernel_caps_per_bin_particle_scan`, `test_frame_identity_advances_before_frame_cadence_surface_build`, and `test_phase_order_is_solver_sdf_then_whitewater`; each passed individually, and the relevant focused files passed together. The package suite passed `397 tests`, and the complete scan-focused command passed `246 tests` after the fix. This is a test-isolation/reload reliability problem or an unproven environment interaction, not a recent production-code regression; it should be diagnosed before treating the full suite as a release gate.

## Recommendations, highest ROI first

1. Pin every review/gate claim to the exact tested commit, command, exit status, and receipt/artifact hash. Refresh the swarm completion artifacts before using them as current proof.
2. Remove the order sensitivity in the full Python suite. Reproduce with a minimal predecessor set, then isolate module-reload tests or make package identity explicit so source-contract tests cannot inspect stale objects.
3. Retain and schedule the existing security backlog cards `TASK-REVIEW-039` (compile-boundary validation for preset code) and `TASK-REVIEW-040` (sanitize user-preset selection paths). They were not introduced by this scan, so no unrelated fix was applied.

## Verification and limits

- Passed: `python -m compileall -q fluidSim_warp_blender51 fluidSim_synth`.
- Passed: `python -m pytest -q fluidSim_warp_blender51/tests` → `397 passed`.
- Passed: complete scan-focused regression set → `246 passed`.
- Passed: `python -m pytest -q tests/test_snippet_library.py tests/test_performance_hotpath_contracts.py` → `53 passed`.
- The full suite was run on a moving worktree and is not a clean exact-HEAD receipt. Its three order-sensitive failures remain reported above.
- Blender/CUDA production gates were not launched because unrelated Blender processes were already active; launching another session would violate the repository's serialized-runtime safety rule. No claim of GUI or production-gate status is made.

## Self-evaluation

The first patch attempt matched an earlier exception site in `custom.py` rather than the loader. I detected this through the failing regression test, re-read the live function, applied the catch with unique loader context, and reran the focused tests. Concurrent edits were preserved; staging was restricted to the exact scan hunk, test fixtures, and this report.
