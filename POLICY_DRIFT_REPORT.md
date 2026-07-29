---
status: fresh
escalated: false
generated_at: 2026-07-28T17:04:58.639694+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: f9c7b366baaa31d171ff3b6773f39eb0125d110c
gate_run_at: 2026-07-28T17:04:58.639694+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: d1c41549ac490071522b59a380b7c2ad16e6be3414aae12fb5d3d15750dfe32f
---

# Policy Drift Report — root unit

## Strongest live evidence

- `rtk python tools/run_production_gate.py --include-gui` passed at HEAD
  `f9c7b366baaa31d171ff3b6773f39eb0125d110c`; the run wrote the full-gate
  manifest, `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (35/35)`, and
  `SAVED-DEMO-GUI-PLAYBACK-PASS`.
- The GUI fixture now keeps tagged sink/force regions inside the documented
  `[-6, 6] x [-2, 2] x [0, 6]` domain. The direct regression still preserves
  the primary out-of-domain `ValueError` and leaves the failed session without
  a solver, so no secondary `None.solver.frame` crash can replace it.
- Current generated demo analysis and checked-in `demos/analysis_report.json`
  are byte-for-byte synchronized at
  `artifacts/demo_analysis/analysis_report.json`, SHA-256
  `d1c41549ac490071522b59a380b7c2ad16e6be3414aae12fb5d3d15750dfe32f`.

## Findings

### PD-ROOT-001 — Resolved: GUI playback world-box contract

The acceptance fixture no longer feeds an intentionally out-of-domain UI
region into the live solver. The shared sparse-grid rejection remains strict,
and `tests/test_scene_bridge_cache_playback.py` covers preservation of its
actionable diagnostic.

### PD-ROOT-002 — Resolved: report authority recurrence

The full-gate manifest and `tools/validate_policy_drift_reports.py` now bind
both reports, the checked-in demo JSON, promoted documentation metrics, active
cards, artifact hash, verdict, and current HEAD. A missing/failed/non-GUI gate
clears the manifest and validation fails closed.

## Machine-readable audit footer

generated_at: 2026-07-28T17:04:58.639694+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: f9c7b366baaa31d171ff3b6773f39eb0125d110c
gate_run_at: 2026-07-28T17:04:58.639694+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: d1c41549ac490071522b59a380b7c2ad16e6be3414aae12fb5d3d15750dfe32f
freshness: fresh
status: fresh
blocking_criticality: none
human_decision_required: false
strongest_live_verification:
  - 'PASS: rtk python tools/run_production_gate.py --include-gui; FULL-PRODUCTION-GATE-PASS.'
  - 'PASS: GUI-ACCEPTANCE-PASS (35/35) and SAVED-DEMO-GUI-PLAYBACK-PASS.'
  - 'PASS: rtk python -m pytest -q; 388 passed.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Repair the out-of-domain GUI fixture and preserve the primary error.'
  - '[x] Bind current reports to the full-gate manifest and artifact hash.'
  - '[x] Synchronize checked-in demo evidence and promoted documentation.'
