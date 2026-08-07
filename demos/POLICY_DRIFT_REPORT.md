---
status: fresh
escalated: false
generated_at: 2026-08-07T19:55:00.500341+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_architecture_resolution_integration_20260807\demos'
git_head: 524960a9f5e5def6fca75a93dd70d690e8479649
gate_run_at: 2026-08-07T19:55:00.500341+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 18819facff53e167f2eb16acb61c6a93571b720fe8b79aafa40f5a3ad7da42d5
---

# Policy Drift Report — demos unit

## Strongest live evidence

- The integrated full gate passed with real Blender/Warp execution and wrote
  `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (38/38)`, and
  `SAVED-DEMO-GUI-PLAYBACK-PASS`.
- `demos/analysis_report.json` is synchronized with
  `artifacts/demo_analysis/analysis_report.json`; the same SHA-256 and proof
  HEAD are recorded in this report and the root report.
- `demos/ANALYSIS.md` and
  `fluidSim_warp_blender51/docs/technical_report.md` promote exact metrics
  from that artifact and identify the same gate run and source path.

## Finding

### PD-DEMO-001 — Resolved: checked-in demo evidence recurrence

The checked-in JSON, Markdown, technical report, rebuilt demo scenes, and
previews describe one current integrated run. `tools/validate_policy_drift_reports.py`
verifies equality, provenance, proof ancestry, full-gate verdict, and active-card
state; it fails closed when any binding is stale or misbound.

## Machine-readable audit footer

generated_at: 2026-08-07T19:55:00.500341+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_architecture_resolution_integration_20260807\demos'
git_head: 524960a9f5e5def6fca75a93dd70d690e8479649
gate_run_at: 2026-08-07T19:55:00.500341+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 18819facff53e167f2eb16acb61c6a93571b720fe8b79aafa40f5a3ad7da42d5
freshness: fresh
status: fresh
blocking_criticality: none
human_decision_required: false
strongest_live_verification:
  - 'PASS: integrated parent full gate; DEMO-ANALYSIS-PASS (18/18).'
  - 'PASS: GUI-ACCEPTANCE-PASS (38/38) and SAVED-DEMO-GUI-PLAYBACK-PASS.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Synchronize checked-in demo JSON with the current artifact.'
  - '[x] Synchronize ANALYSIS.md and technical report metrics/provenance.'
  - '[x] Validate the refreshed report and technical wording.'
