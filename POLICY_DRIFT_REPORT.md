---
status: fresh
escalated: false
generated_at: 2026-08-21T00:07:25.518125+00:00
audit_unit: 'C:\tmp\fluidsim-1.0.252-merge-worktree'
git_head: 459dac4cbf9885d66bf44e30a9412022c24621ac
gate_run_at: 2026-08-21T00:07:25.518125+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: b782a6f7386b5c044af7515743d3de543f505fa6e25d936e459dcb1cb7964365
---

# Policy Drift Report — root unit

## Strongest live evidence

- `FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py --include-gui`
  passed at proof HEAD `459dac4cbf9885d66bf44e30a9412022c24621ac`;
  the run wrote the full-gate
  manifest, `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (40/40)`,
  `GUI-PLAYBACK-PROBE-PASS`, and `SAVED-DEMO-GUI-PLAYBACK-PASS`.
- The generated demo analysis is `result: pass`, 18/18 bounded checks, and the
  checked-in `demos/analysis_report.json` is byte-for-byte synchronized with
  `artifacts/demo_analysis/analysis_report.json`, SHA-256
  `b782a6f7386b5c044af7515743d3de543f505fa6e25d936e459dcb1cb7964365`.
- The measured user path passed at 7.871 ms mean viewport publication and
  21.009 ms mean total, with no added output-path readbacks, synchronization,
  Blender ID creation/removal, depsgraph updates, or structural replacement.
- The four promoted metrics moved within their bounded tolerances relative to
  the previous proof; the shift is GPU floating-point nondeterminism across
  bakes, not a behavioral change, and every bounded check still passes.

## Findings

### PD-ROOT-001 — Resolved: GUI playback world-box contract

The acceptance fixture no longer feeds an intentionally out-of-domain UI
region into the live solver. The shared sparse-grid rejection remains strict,
and `tests/test_scene_bridge_cache_playback.py` covers preservation of its
actionable diagnostic.

### PD-ROOT-002 — Resolved: report authority recurrence

The full-gate manifest and `tools/validate_policy_drift_reports.py` now bind
both reports, the checked-in demo JSON, promoted documentation metrics, active
cards, artifact hash, verdict, and proof HEAD. Evidence-only descendants remain
valid; any intervening non-evidence commit makes validation fail closed.

### PD-ROOT-003 — Resolved: 1.0.253 integration proof staleness

The previous report predated the 1.0.253 candidate integration. This report
binds the merged implementation to the fresh GUI-inclusive gate at `459dac4`
and to this isolated merge worktree.

## Known non-blocking exceptions

- `C:\Python314\python.exe -m pytest -q` passed all 1,028 tests before the committed-HEAD
  production run. There is no repository-wide Python exception in this proof.
- `demos/paddle_stir` is an opt-in moving-collider demo outside the canonical
  18/18 gate set. Fresh standalone rebuilds on both baseline and integration
  score 10/10; the documented pi-crossing velocity failure did not reproduce.

## Machine-readable audit footer

generated_at: 2026-08-21T00:07:25.518125+00:00
audit_unit: 'C:\tmp\fluidsim-1.0.252-merge-worktree'
git_head: 459dac4cbf9885d66bf44e30a9412022c24621ac
gate_run_at: 2026-08-21T00:07:25.518125+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: b782a6f7386b5c044af7515743d3de543f505fa6e25d936e459dcb1cb7964365
freshness: fresh
status: fresh
blocking_criticality: none
human_decision_required: false
strongest_live_verification:
  - 'PASS: FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py --include-gui; FULL-PRODUCTION-GATE-PASS.'
  - 'PASS: GUI-ACCEPTANCE-PASS (40/40) and SAVED-DEMO-GUI-PLAYBACK-PASS.'
  - 'PASS: generated demo analysis result=pass, 18/18 bounded checks.'
  - 'PASS: user-path profile; view mean 7.871 ms, total mean 21.009 ms.'
  - 'PASS: C:\Python314\python.exe -m pytest -q; 1028 passed.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Repair the out-of-domain GUI fixture and preserve the primary error.'
  - '[x] Bind current reports to the full-gate manifest and artifact hash.'
  - '[x] Synchronize checked-in demo evidence and promoted documentation.'
  - '[x] Rebind the stale proof and correct the misbound audit_unit.'
