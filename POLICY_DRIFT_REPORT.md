---
status: fresh
escalated: false
generated_at: 2026-09-21T03:36:27.7252393Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: efb02e4e8cb4a8833e2e0631b5f33eee205ea77c
gate_run_at: 2026-09-21T03:12:48.375188Z
gate_verdict: 'BLOCKED-INTERACTIVE-GUI-NO-RECEIPT'
unit_verdict: 'NON-GUI-EVIDENCE-CURRENT-GUI-RECEIPT-MISSING'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a
git_dirty: true
---

# Policy Drift Report: root unit

## Audit conclusion

Current non-GUI evidence still exercises the real FluidSim/Warp path: the
fresh demo analyzer passed `18/18`, and package verification reported
`release_ready: true`. No source evidence found a narrow heuristic solver or
fabricated runtime output. The end-to-end claim is not closed for this run:
`tools/run_production_gate.py --include-gui` reached the interactive Blender
lane, produced no fresh GUI receipt or gate manifest, and was stopped after the
Blender child became non-responsive.

## Findings

### PD-ROOT-001: checked-in demo evidence diverges from the fresh artifact (validation drift)

The current artifact was generated at `2026-09-21T03:27:36Z` and passes
`18/18`, but its SHA-256 is
`ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a`, while
the checked-in JSON is `afb53b3f847818b77e13e64b320d9d85b134a61855a30dcfb29e4ddb7efa15a1`.
The current artifact reports dam-break column height `0.3355760574 m` and
crown height `0.2811620831 m`; checked-in evidence reports `0.3376998901 m`
and `0.2944921553 m`. The existing repair card is
`backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md`.

### PD-ROOT-002: full GUI gate has no current completion receipt (validation drift)

The gate was launched at the current HEAD, but its interactive Blender child
became non-responsive and no fresh `artifacts/production_gate_manifest.json`
was written. The existing repair card is
`backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md`.
The prior `phase3_gui_result.json` and `gui_playback_probe.json` are from
2026-09-19 and are historical for this audit.

### PD-ROOT-003: paddle spin-up threshold remains explicitly uncalibrated (validation drift)

`tools/analyze_demos.py` still labels `PADDLE_SPINUP_MOMENTUM_MARGIN = 0.3`
as a placeholder. The opt-in cache required to run the current paddle
analyzer is absent, so no current `paddle_stir` pass is promoted here. The
existing repair card is
`backlog/tasks/TASK-POLICY-DRIFT-016-PADDLE-SPINUP-THRESHOLD-CALIBRATION.md`.

## Scope boundary and non-findings

Intentional numerical clamping, material fallbacks, effect fallbacks, and
test doubles were not promoted as heuristic-runtime findings. The fresh
`18/18` analyzer and package release verification are concrete current
evidence, but they do not substitute for the missing GUI-inclusive receipt.

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
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: efb02e4e8cb4a8833e2e0631b5f33eee205ea77c
gate_run_at: 2026-09-21T03:12:48.375188Z
gate_verdict: 'BLOCKED-INTERACTIVE-GUI-NO-RECEIPT'
unit_verdict: 'NON-GUI-EVIDENCE-CURRENT-GUI-RECEIPT-MISSING'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a
freshness: fresh
status: fresh
blocking_criticality: high
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/analyze_demos.py -> DEMO-ANALYSIS-PASS (18/18), artifact mtime 2026-09-21T03:27:36Z'
  - 'rtk python tools/release/build_extension.py --verify-only -> release_ready=true, version 1.0.255'
  - 'rtk python tools/run_production_gate.py --include-gui -> no receipt; interactive Blender child non-responsive'
  - 'rtk python -m pytest -q -> interrupted after bounded wait; no verdict'
files_that_need_correction:
  - 'demos/analysis_report.json'
  - 'demos/ANALYSIS.md'
  - 'fluidSim_warp_blender51/docs/technical_report.md'
  - 'tools/analyze_demos.py'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md'
  - 'backlog/tasks/TASK-POLICY-DRIFT-016-PADDLE-SPINUP-THRESHOLD-CALIBRATION.md'
  - 'backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md'
repair_checklist:
  - 'Refresh checked-in JSON, Markdown, and technical-report metrics from one stable generated artifact.'
  - 'Capture a bounded, reproducible GUI receipt or preserve the diagnostic failure context.'
  - 'Calibrate paddle spin-up with current real positive and negative cases.'
```
