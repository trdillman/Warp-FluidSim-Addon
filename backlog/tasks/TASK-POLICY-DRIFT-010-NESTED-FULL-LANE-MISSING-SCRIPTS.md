# TASK-POLICY-DRIFT-010: Restore nested full-lane inputs

## State

completed

## What to build

Make the committed nested worktree contain the scripts required by its own
production gate.

## Acceptance criteria

- [x] The nested `python tools/run_production_gate.py --include-gui` reaches
  runtime lanes instead of failing during `py_compile`.
- [x] The required lane scripts exist in this worktree and are covered by the
  same commit as the production runner.
- [x] The nested report uses this worktree as `audit_unit` and records the
  current gate result.

## Evidence

- Nested report: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356\POLICY_DRIFT_REPORT.md`.
- Nested worktree HEAD is `697fe6ef9b0861fafc1c9808176f108b03fe3b37` and its
  status is dirty with this card/report work, the restored input cohort,
  generated demo outputs, and the preserved unrelated
  `backlogs/swarm-review-prk-2026-07-27.md`.
- All 37 test files deleted by `a6965e6` are restored byte-for-byte after line
  ending normalization from `a6965e6^`. The two tracked demo profile inputs
  were restored from that commit to create their missing directories, then
  gate-refreshed; their current metadata is included in the scoped commit plan.
- The nested full gate at `2026-07-28T22:35:30Z` reports
  `FULL-PRODUCTION-GATE-PASS`.

## Blocked by

None - the branch-compatible lane input cohort is restored.

## Verification

- `rtk python -m pytest -q tests/test_pressure_projection_regressions.py tests/test_cache_io.py tests/test_io_path_policy.py`: `44 passed`.
- `python -m pytest -q`: `239 passed`.
- `rtk python tools/run_production_gate.py --include-gui`: `FULL-PRODUCTION-GATE-PASS`.
