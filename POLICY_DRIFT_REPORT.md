---
status: fresh
escalated: false
generated_at: 2026-09-01T05:31:42.102923+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: 69a51c86954b61d7c8ce5f0d0e4bf1cead2fd528
gate_run_at: 2026-09-01T05:31:42.102923+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 6b379e9d15ac2a86ffa1a0b49d698b4456911e932f74ebdb07da060936608495
git_dirty: true
---

# Policy Drift Report: root unit

## Audit conclusion

The current executable FluidSim/Warp path is not supported as a narrow
heuristic implementation. A fresh GUI-inclusive production gate ran against
HEAD `69a51c86954b61d7c8ce5f0d0e4bf1cead2fd528`; gate metadata was dirty only
because of an unrelated untracked design document,
used Blender 5.1 with external Warp 1.16.0 on CUDA 12.9 / RTX 4080 SUPER,
advanced live playback, emitted visible surface and whitewater output, and
finished with `FULL-PRODUCTION-GATE-PASS`. The nested package suite and
package verification also pass independently.

The promoted demo evidence and nested technical report are synchronized to the
same gate artifact. The opt-in paddle acceptance threshold remains explicitly
marked uncalibrated. Earlier gate attempts exposed and led to fixes for stale
particle-pool ownership, structured injection receipts, Windows output
encoding, and a retired GUI test operator; the final stable-HEAD gate passed.

## Strongest live evidence

- `artifacts/production_gate_manifest.json` records gate time
  `2026-09-01T05:31:42.102923+00:00`, proof HEAD `69a51c86954b`,
  `FULL-PRODUCTION-GATE-PASS`, GUI inclusion, and generated demo artifact hash
  `6b379e9d15ac2a86ffa1a0b49d698b4456911e932f74ebdb07da060936608495`.
- The gate recorded `GUI-ACCEPTANCE-PASS (40/40)` and
  `SAVED-DEMO-GUI-PLAYBACK-PASS` at the same proof HEAD.
- The current root Python suite passed `1177` tests before the final gate.

## Findings

### PD-ROOT-001: stale report authority was recurring (documentation drift)

Before this audit, this report named `C:\tmp\fluidsim-main-release`, HEAD
`bb6d3fc5...`, and an old artifact hash. The report has been refreshed to this
checkout, current HEAD, and current gate metadata. The completed local card is
`backlog/tasks/TASK-POLICY-DRIFT-014-REPORT-AUTHORITY-RECURRENCE.md`.

### PD-ROOT-002: checked-in demo evidence recurrence repaired (validation drift)

The generated artifact and checked-in `demos/analysis_report.json` now match
byte-for-byte at SHA-256
`6b379e9d15ac2a86ffa1a0b49d698b4456911e932f74ebdb07da060936608495`.
`demos/ANALYSIS.md` and the nested technical report promote the same four
metrics. Recurrence tracking remains in
`backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md`.

### PD-ROOT-003: paddle spin-up acceptance uses an uncalibrated margin (validation drift)

`tools/analyze_demos.py` labels `PADDLE_SPINUP_MOMENTUM_MARGIN = 0.3` as
`UNCALIBRATED` and a placeholder. The current real `paddle_stir` bake passes
10/10 over 96 frames and 158,220 particles, but the acceptance boundary has
not been measured across scale or degraded-motion cases. Repair is tracked by
`backlog/tasks/TASK-POLICY-DRIFT-016-PADDLE-SPINUP-THRESHOLD-CALIBRATION.md`.

### PD-ROOT-004: full GUI gate was intermittently non-reproducible (validation drift)

Earlier full-gate attempts reproduced distinct failures in capacity retry,
demo receipt compatibility, output encoding, and stale GUI test setup. Each
was fixed without weakening the checks. The final gate passed `40/40` GUI
acceptance and saved-demo playback. Recurrence tracking remains in
`backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md`.

## Scope boundary and non-findings

The dirty worktree contains only the evidence files promoted after the proof
commit. Current real
runtime evidence shows live GPU stepping, surface publication, whitewater
objects, GUI panel draws, pending-setting behavior, rendering, and saved-demo
playback. Static fallback terms found in materials, custom-effect paths, and
test fixtures are intentional or test-only and have no contradictory runtime
evidence. The nested package's stale technical report is owned by that unit's
report and backlog, not silently folded into this root unit.

## Machine-readable audit footer

```yaml
generated_at: 2026-09-01T05:31:42.102923+00:00
generated_at_utc: 2026-09-01T05:31:42.102923Z
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: 69a51c86954b61d7c8ce5f0d0e4bf1cead2fd528
gate_run_at: 2026-09-01T05:31:42.102923+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 6b379e9d15ac2a86ffa1a0b49d698b4456911e932f74ebdb07da060936608495
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/run_production_gate.py --include-gui -> artifacts/production_gate_manifest.json'
  - 'rtk python -m pytest -q -> 1177 passed during this audit'
  - 'rtk python tools/analyze_demos.py paddle_stir -> DEMO-ANALYSIS-PASS (10/10)'
  - 'artifacts/production_gate_manifest.json -> FULL-PRODUCTION-GATE-PASS at 69a51c86954b'
files_that_need_correction:
  - 'tools/analyze_demos.py'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md'
  - 'backlog/tasks/TASK-POLICY-DRIFT-016-PADDLE-SPINUP-THRESHOLD-CALIBRATION.md'
  - 'backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md'
repair_checklist:
  - 'Synchronize generated demo metrics and hashes into every promoted report.'
  - 'Calibrate paddle spin-up using repeated real scale and negative cases.'
  - 'Make full GUI gate failures reproducible and preserve diagnostic context.'
```
