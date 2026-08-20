---
status: fresh
escalated: false
generated_at: 2026-08-20T23:14:10.544662+00:00
audit_unit: 'C:\tmp\fluidsim-1.0.252-merge-worktree\demos'
git_head: f9613f65cb619ca52972b0f2d6989b8c7ee50f0e
gate_run_at: 2026-08-20T23:14:10.544662+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: f13efd30d309f880ef09e2808126521f0493806d0b0b88a36ab2a03549dc24db
---

# Policy Drift Report — demos unit

## Strongest live evidence

- The parent full gate passed with real Blender/Warp execution and wrote the
  current manifest: `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (40/40)`,
  and `SAVED-DEMO-GUI-PLAYBACK-PASS`.
- `demos/analysis_report.json` is synchronized byte-for-byte with
  `artifacts/demo_analysis/analysis_report.json`; the same SHA-256 and proof HEAD
  are recorded in this report and the root report.
- `demos/ANALYSIS.md` and
  `fluidSim_warp_blender51/docs/technical_report.md` promote exact metrics
  from that artifact and identify the same gate run and source path.

## Finding

### PD-DEMO-001 — Resolved: checked-in demo evidence recurrence

The checked-in JSON, Markdown, and technical report now describe one current
generated run. `tools/validate_policy_drift_reports.py` verifies the equality,
provenance, proof HEAD, full-gate verdict, and active-card state. Evidence-only
descendants remain valid; non-evidence drift makes validation fail closed.

## Machine-readable audit footer

generated_at: 2026-08-20T23:14:10.544662+00:00
audit_unit: 'C:\tmp\fluidsim-1.0.252-merge-worktree\demos'
git_head: f9613f65cb619ca52972b0f2d6989b8c7ee50f0e
gate_run_at: 2026-08-20T23:14:10.544662+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: f13efd30d309f880ef09e2808126521f0493806d0b0b88a36ab2a03549dc24db
freshness: fresh
status: fresh
blocking_criticality: none
human_decision_required: false
strongest_live_verification:
  - 'PASS: parent full gate; DEMO-ANALYSIS-PASS (18/18).'
  - 'PASS: GUI-ACCEPTANCE-PASS (40/40) and SAVED-DEMO-GUI-PLAYBACK-PASS.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Synchronize checked-in demo JSON with the current artifact.'
  - '[x] Synchronize ANALYSIS.md and technical report metrics/provenance.'
  - '[x] Validate the refreshed report and technical wording.'
