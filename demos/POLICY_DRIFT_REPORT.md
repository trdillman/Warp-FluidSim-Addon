---
status: fresh
escalated: false
generated_at: 2026-07-26T15:22:19Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356\demos'
git_head: a2eaca7f45eb91368b082618e8759b028082ec87
---

# Policy Drift Report — nested demos unit

## Truth contract

The nested demos unit must have a parent lane that can generate and validate its
own saved-demo evidence. Outer-checkout artifacts and a partial test suite are
not substitutes for that unit's missing full-lane proof.

## Strongest live evidence

- **FAIL** parent nested `rtk python tools/run_production_gate.py --include-gui`:
  it stops at syntax compilation because required test-lane scripts are absent.
- **PASS but subordinate** nested `rtk python -m pytest -q`: `96 passed in
  42.73s`; no nested full-lane demo-analysis or GUI artifact was produced.
- The prior nested report pointed to the outer checkout's `demos` directory
  and old HEAD, so it was not authoritative for this unit.

## Claim matrix

| Claim | Evidence | Verdict | Gap |
|---|---|---|---|
| Nested demos have current full-lane evidence | Parent nested gate fails before demo analysis | missing | Restore parent lane inputs first |
| Nested report identifies its own unit | Prior `audit_unit` pointed outside this worktree | nonconformant | Corrected here |

## Findings

### PD-NESTED-DEMO-001 — Demo proof is blocked by the nested parent lane

Type: validation drift and documentation drift. Blocking criticality: high.

No nested demo artifact can be treated as current because the parent gate never
reaches demo analysis or GUI playback. The lower-level `96 passed` result is not
the lane-specific proof required by the project instructions.

Card: `backlog/tasks/TASK-POLICY-DRIFT-004-NESTED-DEMO-AUTHORITY.md`.

## Machine-readable audit footer

generated_at: 2026-07-26T15:22:19Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356\demos'
git_head: a2eaca7f45eb91368b082618e8759b028082ec87
freshness: fresh
status: fresh
blocking_criticality: high
human_decision_required: false
strongest_live_verification:
  - 'FAIL: parent nested rtk python tools/run_production_gate.py --include-gui; missing lane scripts during py_compile.'
  - 'PASS but subordinate: nested rtk python -m pytest -q; 96 passed in 42.73s.'
files_that_need_correction:
  - 'POLICY_DRIFT_REPORT.md'
  - '../backlog/tasks/TASK-POLICY-DRIFT-010-NESTED-FULL-LANE-MISSING-SCRIPTS.md'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-004-NESTED-DEMO-AUTHORITY.md'
repair_checklist:
  - '[ ] Restore the nested parent lane inputs.'
  - '[ ] Rerun demo analysis and saved-demo playback in this unit.'
  - '[ ] Refresh this report from those nested artifacts.'
