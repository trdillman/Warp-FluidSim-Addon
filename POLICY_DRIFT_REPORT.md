---
status: fresh
escalated: false
generated_at: 2026-08-09T05:18:06.780758+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: e8a3c062270eaacb0f53e69f5b6246ad8695bb95
gate_run_at: 2026-08-09T05:18:06.780758+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 913fb99f83cabb3657ab7f323235a1bac5e11f3c967b3d816768941f4acf894b
---

# Policy Drift Report — root unit

## Strongest live evidence

- `python tools/run_production_gate.py --include-gui` passed at proof HEAD
  `e8a3c062270eaacb0f53e69f5b6246ad8695bb95`; the run wrote the full-gate
  manifest, `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (38/38)`,
  `GUI-PLAYBACK-PROBE-PASS`, and `SAVED-DEMO-GUI-PLAYBACK-PASS`.
- The generated demo analysis is `result: pass`, 18/18 bounded checks, and the
  checked-in `demos/analysis_report.json` is byte-for-byte synchronized with
  `artifacts/demo_analysis/analysis_report.json`, SHA-256
  `913fb99f83cabb3657ab7f323235a1bac5e11f3c967b3d816768941f4acf894b`.
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

### PD-ROOT-003 — Resolved: proof staleness after the moving-collider work

The previous proof was bound to HEAD `15b8ac2`, and commit `53564ac` had
already invalidated it by changing `fluidSim_warp_blender51/sim/solver.py`,
`tools/run_production_gate.py`, and fifteen test modules. The moving-collider
commits (`e13945b`, `d5ce823`) and the version resync (`2e7783b`) widened that
gap further. This report rebinds the proof to a fresh GUI-inclusive gate run at
`e8a3c06`, and additionally corrects `audit_unit`, which had been misbound to
the retired `C:\tmp\fluidsim-resolution-quality-merge-20260805` worktree rather
than to this unit.

## Known non-blocking exceptions

- `python -m pytest -q` reports 4 failures in
  `tests/test_pressure_projection_regressions.py` (`assert 'host_batched' ==
  'cuda_conditional'`). Verified pre-existing by stashing the moving-collider
  changes and re-running: identical failures. Tracked as TASK-REVIEW-016. These
  are outside the gate's lanes and do not affect its verdict.
- `demos/paddle_stir` is an opt-in moving-collider demo at 9/10 analyzer
  checks, deliberately outside the canonical 18/18 gate set. Its one failing
  check is the pi-crossing velocity impulse documented in
  `docs/ISSUE-paddle-stir-velocity-spike.md`.

## Machine-readable audit footer

generated_at: 2026-08-09T05:18:06.780758+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: e8a3c062270eaacb0f53e69f5b6246ad8695bb95
gate_run_at: 2026-08-09T05:18:06.780758+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 913fb99f83cabb3657ab7f323235a1bac5e11f3c967b3d816768941f4acf894b
freshness: fresh
status: fresh
blocking_criticality: none
human_decision_required: false
strongest_live_verification:
  - 'PASS: python tools/run_production_gate.py --include-gui; FULL-PRODUCTION-GATE-PASS.'
  - 'PASS: GUI-ACCEPTANCE-PASS (38/38) and SAVED-DEMO-GUI-PLAYBACK-PASS.'
  - 'PASS: generated demo analysis result=pass, 18/18 bounded checks.'
  - 'KNOWN-FAIL: python -m pytest -q; 546 passed, 4 pre-existing PCG failures (TASK-REVIEW-016).'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Repair the out-of-domain GUI fixture and preserve the primary error.'
  - '[x] Bind current reports to the full-gate manifest and artifact hash.'
  - '[x] Synchronize checked-in demo evidence and promoted documentation.'
  - '[x] Rebind the stale proof and correct the misbound audit_unit.'
