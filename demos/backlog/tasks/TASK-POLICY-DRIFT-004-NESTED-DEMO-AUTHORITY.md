# TASK-POLICY-DRIFT-004: Correct nested demo unit authority

## State

completed

## What to build

Bind the nested demo report to its actual worktree path and keep it blocked
until the parent nested production lane has a complete executable input set.

## Acceptance criteria

- [x] `audit_unit` is this worktree's `demos` directory, not the outer checkout.
- [x] The report records the current parent full-lane result as strongest
  evidence and does not substitute a lower-level pytest pass.
- [x] A fresh nested demo artifact is generated only after the parent gate can
  execute its complete lane.

## Evidence

- Nested demos report: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356\demos\POLICY_DRIFT_REPORT.md`.
- Its `audit_unit` is the nested `demos` directory, generation is
  `2026-07-28T22:44:16Z`, and HEAD is
  `697fe6ef9b0861fafc1c9808176f108b03fe3b37`.
- The parent gate reports `FULL-PRODUCTION-GATE-PASS`. The current analyzer
  writes `artifacts/demo_analysis/analysis_report.json` as `pass`, `18/18`,
  SHA-256
  `034c45ee532ca4d61abe84e683858e161be6fdd9396cc013c501b8448c13c496`.
- `artifacts/gui_playback_probe.json` records live passing playback through
  solver frame 3, and the validator reports
  `SAVED-DEMO-GUI-PLAYBACK-PASS`.

## Blocked by

None - `TASK-POLICY-DRIFT-010` is completed and its full gate passes.

## Verification

- Parent nested gate: `FULL-PRODUCTION-GATE-PASS`.
- Nested demo analysis: `DEMO-ANALYSIS-PASS (18/18)`.
- Nested saved-demo playback: `SAVED-DEMO-GUI-PLAYBACK-PASS`.
