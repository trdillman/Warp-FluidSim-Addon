# TASK-POLICY-DRIFT-004: Rebind nested demo report authority

## State

completed

## What to build

Keep the nested `demos` report bound to the current parent lane, current git
HEAD, and the actual checked-in/generated evidence relationship.

## Acceptance criteria

- [x] `generated_at`, `git_head`, and `audit_unit` identify this `demos` unit
  and the current audited checkout.
- [x] The report lists the active checked-in-evidence and report-authority
  cards while they remain unresolved.
- [x] The footer does not claim synchronized metrics until the files agree.

## Evidence

- Report: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos\POLICY_DRIFT_REPORT.md`.
- Its frontmatter and footer identify audit unit `...\liquid_warp_blender51_port\demos`,
  generation `2026-07-26T15:22:19Z`, and HEAD
  `a2eaca7f45eb91368b082618e8759b028082ec87`.
- Its claim matrix and footer preserve the checked-in metric mismatch and keep
  both evidence-sync cards active.

## Blocked by

None - acceptance criteria met; no remaining blocker
