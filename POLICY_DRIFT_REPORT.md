---
status: fresh
escalated: false
generated_at: 2026-08-05T15:23:58.988102+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_architecture_deepening_20260805'
git_head: afbdf0e0f902a3e6205420aca3e31b04defe9749
gate_run_at: 2026-08-05T15:23:58.988102+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: d1a0cd12233a3a7c92ecac177fa9b3c1c1f3514dcc1ea464c798c6d153310242
---

# Policy Drift Report — root unit

## Strongest live evidence

- `python tools/run_production_gate.py --include-gui` passed at HEAD
  `afbdf0e0f902a3e6205420aca3e31b04defe9749`; the run wrote the full-gate
  manifest, `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (38/38)`,
  `CAPACITY-PLANNER-ACCEPTANCE-PASS`, `DEMO-ANALYSIS-PASS (18/18)`, and
  `SAVED-DEMO-GUI-PLAYBACK-PASS`.
- The GUI fixture keeps tagged sink/force regions inside the documented
  `[-6, 6] x [-2, 2] x [0, 6]` domain. The direct regression preserves
  the primary out-of-domain `ValueError` and leaves the failed session without
  a solver, so no secondary `None.solver.frame` crash can replace it.
- Current generated demo analysis and checked-in `demos/analysis_report.json`
  are synchronized at `artifacts/demo_analysis/analysis_report.json`, SHA-256
  `d1a0cd12233a3a7c92ecac177fa9b3c1c1f3514dcc1ea464c798c6d153310242`.

## Findings

### PD-ROOT-001 — Resolved: GUI playback world-box contract

The acceptance fixture does not feed an intentionally out-of-domain UI region
into the live solver. The shared sparse-grid rejection remains strict, and
`tests/test_scene_bridge_cache_playback.py` covers preservation of its
actionable diagnostic.

### PD-ROOT-002 — Resolved: report authority recurrence

The full-gate manifest and `tools/validate_policy_drift_reports.py` bind both
reports, the checked-in demo JSON, promoted documentation metrics, active
cards, artifact hash, verdict, and proof HEAD. A missing, failed, or non-GUI
gate clears the manifest and validation fails closed.

## Machine-readable audit footer

generated_at: 2026-08-05T15:23:58.988102+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port_architecture_deepening_20260805'
git_head: afbdf0e0f902a3e6205420aca3e31b04defe9749
gate_run_at: 2026-08-05T15:23:58.988102+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: d1a0cd12233a3a7c92ecac177fa9b3c1c1f3514dcc1ea464c798c6d153310242
freshness: fresh
status: fresh
blocking_criticality: none
human_decision_required: false
strongest_live_verification:
  - 'PASS: python tools/run_production_gate.py --include-gui; FULL-PRODUCTION-GATE-PASS.'
  - 'PASS: GUI-ACCEPTANCE-PASS (38/38) and SAVED-DEMO-GUI-PLAYBACK-PASS.'
  - 'PASS: CAPACITY-PLANNER-ACCEPTANCE-PASS and DEMO-ANALYSIS-PASS (18/18).'
  - 'PASS: python -m pytest -q; 513 passed.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Preserve the out-of-domain diagnostic contract.'
  - '[x] Bind current reports to the full-gate manifest and artifact hash.'
  - '[x] Synchronize checked-in demo evidence and promoted documentation.'
