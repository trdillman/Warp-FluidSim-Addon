# TASK-POLICY-DRIFT-004: Correct nested demo unit authority

## State

blocked

## What to build

Bind the nested demo report to its actual worktree path and keep it blocked
until the parent nested production lane has a complete executable input set.

## Acceptance criteria

- [x] `audit_unit` is this worktree's `demos` directory, not the outer checkout.
- [x] The report records the parent full-lane failure as the strongest current
  evidence and does not promote the `96`-test lower-level pass.
- [ ] A fresh nested demo artifact is generated only after the parent gate can
  execute its complete lane.

## Evidence

- Nested demos report: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356\demos\POLICY_DRIFT_REPORT.md`.
- Its `audit_unit` is the nested `demos` directory, generation is
  `2026-07-26T15:22:19Z`, and HEAD is
  `a2eaca7f45eb91368b082618e8759b028082ec87`.
- The report records the nested parent gate's `py_compile` failure, the
  subordinate `96 passed` result, and no nested demo or GUI artifacts.

## Blocked by

- `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356\backlog\tasks\TASK-POLICY-DRIFT-010-NESTED-FULL-LANE-MISSING-SCRIPTS.md`
