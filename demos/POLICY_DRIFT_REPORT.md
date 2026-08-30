---
status: fresh
escalated: false
generated_at: 2026-08-30T09:47:34.449493+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: 41a02c843c79081a79514f08861cab46dea2d46c
gate_run_at: 2026-08-30T09:47:34.449493+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: ddbd3eaa843df1dcd88bb0ff1f82bf6fd6591f9c7e5329fa4ae7eeb4b33dc0b4
git_dirty: true
---

# Policy Drift Report: demos unit

## Audit conclusion

The saved demo scenes are backed by real runtime behavior, not static preview
data. The current GUI-inclusive parent gate used Blender 5.1, external Warp
1.16.0, and CUDA on the RTX 4080 SUPER; it rebuilt and checked the demos,
validated live playback, and finished with `FULL-PRODUCTION-GATE-PASS` and
`SAVED-DEMO-GUI-PLAYBACK-PASS`.

The generated 18/18 artifact is now copied byte-for-byte into the checked-in
JSON, while the Markdown and nested technical report promote the same metrics
and gate provenance. The report-authority and evidence mismatches found at the
start of the audit are repaired; the recurrence card remains visible.

## Strongest live evidence

- `artifacts/production_gate_manifest.json` records proof HEAD
  `41a02c843c79`, `GUI-ACCEPTANCE-PASS (40/40)`,
  `SAVED-DEMO-GUI-PLAYBACK-PASS`, and `FULL-PRODUCTION-GATE-PASS`.
- The generated and checked-in JSON both have SHA-256
  `ddbd3eaa843df1dcd88bb0ff1f82bf6fd6591f9c7e5329fa4ae7eeb4b33dc0b4`.
- The rebuilt `demos/*.blend` files passed the all-demo gate and real saved
  timeline playback at the same proof HEAD.

## Findings

### PD-DEMO-001: checked-in demo evidence recurrence repaired (validation drift)

The synchronized evidence promotes dam-break front speed `1.0111631359`, dam
column height `0.3348927498`, crown correlation `0.9714044559`, and crown
height `0.2806054652`. The JSON equality and shared metric tokens are enforced
by the parent validator. Recurrence tracking remains in
`demos/backlog/tasks/TASK-POLICY-DRIFT-007-CHECKED-IN-EVIDENCE-RECURRENCE.md`.

### PD-DEMO-002: report provenance was stale at audit start (documentation drift)

The prior report named temporary root `C:\tmp\fluidsim-main-release\demos`,
old HEAD `bb6d3fc5...`, and an old gate timestamp. This report now names the
actual demos directory and current gate provenance. The completed local card
is `demos/backlog/tasks/TASK-POLICY-DRIFT-008-REPORT-AUTHORITY-RECURRENCE.md`.

## Scope boundary and non-findings

No demos-unit evidence supports calling the live solver or playback path a
narrow heuristic. `tools/analyze_demos.py` is a parent-root tool; its separate
uncalibrated paddle threshold is reported and tracked by the root unit. The
known first full-gate GUI exit was followed by a focused GUI pass and a full
retry pass, so it is tracked at the root validation workflow rather than
misattributed to these saved scenes.

## Machine-readable audit footer

```yaml
generated_at: 2026-08-30T09:47:34.449493+00:00
generated_at_utc: 2026-08-30T09:47:34.449493Z
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: 41a02c843c79081a79514f08861cab46dea2d46c
gate_run_at: 2026-08-30T09:47:34.449493+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: ddbd3eaa843df1dcd88bb0ff1f82bf6fd6591f9c7e5329fa4ae7eeb4b33dc0b4
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/run_production_gate.py --include-gui -> artifacts/production_gate_manifest.json'
  - 'blender --background --python tools/check_demo_blends.py -> included in full-gate pass'
  - 'rtk python tools/analyze_demos.py -> DEMO-ANALYSIS-PASS (18/18)'
  - 'artifacts/production_gate_manifest.json -> SAVED-DEMO-GUI-PLAYBACK-PASS at 41a02c843c79'
files_that_need_correction: []
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-007-CHECKED-IN-EVIDENCE-RECURRENCE.md'
repair_checklist:
  - 'Keep generated JSON, promoted metrics, and exact artifact hash synchronized.'
  - 'Run the parent policy-drift validator after every evidence refresh.'
```
