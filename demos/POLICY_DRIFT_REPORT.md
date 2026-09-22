---
status: fresh
escalated: false
generated_at: 2026-09-21T03:36:27.7252393Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: efb02e4e8cb4a8833e2e0631b5f33eee205ea77c
gate_run_at: 2026-09-21T03:12:48.375188Z
gate_verdict: 'BLOCKED-INTERACTIVE-GUI-NO-RECEIPT'
unit_verdict: 'DEMO-ANALYSIS-PASS-CHECKED-IN-EVIDENCE-STALE'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a
git_dirty: true
---

# Policy Drift Report: demos unit

## Audit conclusion

The current demo analyzer passes `18/18`, so the saved demos continue to
exercise real generated behavior rather than only static preview geometry.
The checked-in evidence is not synchronized to this fresh artifact, and the
parent GUI-inclusive gate did not produce a current completion receipt.

## Findings

### PD-DEMO-001: checked-in evidence is stale against the fresh artifact (validation drift)

The fresh artifact hash is
`ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a`; the
checked-in JSON remains
`afb53b3f847818b77e13e64b320d9d85b134a61855a30dcfb29e4ddb7efa15a1`. Current
dam-break column height is `0.3355760574 m` versus checked-in `0.3376998901 m`,
and current crown height is `0.2811620831 m` versus checked-in `0.2944921553 m`.
The existing local card is
`backlog/tasks/TASK-POLICY-DRIFT-007-CHECKED-IN-EVIDENCE-RECURRENCE.md`.

## Scope boundary

The demo analyzer's `18/18` pass is current evidence for this unit. It does
not certify the parent GUI lane, whose current receipt is missing; that gate
finding remains owned by the root report.

`gate_verdict` is bound to the gate run, not to this unit:
`tools/validate_policy_drift_reports.py` compares it against `gate_verdict` in
`artifacts/production_gate_manifest.json`, so every unit report sharing one
`gate_run_at` must carry the identical string, and it must read
`FULL-PRODUCTION-GATE-PASS` once a passing manifest exists
(`tools/run_production_gate.py` writes no other value). The unit-scoped rollup
belongs in `unit_verdict`, which is free-form and may differ per unit.

## Machine-readable audit footer

```yaml
generated_at: 2026-09-21T03:36:27.7252393Z
generated_at_utc: 2026-09-21T03:36:27.7252393Z
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: efb02e4e8cb4a8833e2e0631b5f33eee205ea77c
gate_run_at: 2026-09-21T03:12:48.375188Z
gate_verdict: 'BLOCKED-INTERACTIVE-GUI-NO-RECEIPT'
unit_verdict: 'DEMO-ANALYSIS-PASS-CHECKED-IN-EVIDENCE-STALE'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/analyze_demos.py -> DEMO-ANALYSIS-PASS (18/18)'
  - 'artifacts/demo_analysis/analysis_report.json -> fresh mtime 2026-09-21T03:27:36Z'
  - 'parent rtk python tools/run_production_gate.py --include-gui -> no current receipt'
files_that_need_correction:
  - 'analysis_report.json'
  - 'ANALYSIS.md'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-007-CHECKED-IN-EVIDENCE-RECURRENCE.md'
repair_checklist:
  - 'Copy one stable generated JSON into the checked-in demo evidence.'
  - 'Refresh ANALYSIS.md and preserve the exact artifact hash.'
  - 'Run the policy-drift validator after synchronization.'
```
