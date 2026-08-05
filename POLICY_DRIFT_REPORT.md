---
status: fresh
escalated: false
generated_at: 2026-08-05T11:17:25Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356'
git_head: 0b47a0535f3970a3c8b6426370e63d754ea96486
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 86406b0a7b3f0e75cf337e40a75ed800d2e0d5e490e8f551ded9155a93115cc5
gui_playback_artifact: 'artifacts/gui_playback_probe.json'
gui_playback_artifact_sha256: 8c6c8e3147abb4c4ac25eee45a3f814fbb32d6f149696da522f9b4214329a5f5
---

# Policy Drift Report — particle-surface-rendering-apis worktree

## Strongest live evidence

- `python tools/run_production_gate.py --include-gui` ran at HEAD
  `0b47a0535f3970a3c8b6426370e63d754ea96486` and reported
  `FULL-PRODUCTION-GATE-PASS`.
- Interactive acceptance passed `35/35`; demo analysis passed `18/18`;
  saved-demo playback and its report validator passed. The saved-demo probe
  ran live on `cuda:0` at `600,000` particles, `281,088` voxels, reaching
  solver frame `7` with the viewport confirmed changed.
- `python -m pytest -q`: `239 passed in 9.42s`. This is the host suite, not a
  substitute for the lane above.

## Artifact pinning is rerun-fragile

The `*_sha256` fields below pin generated GPU-simulation artifacts. Those are
not bit-reproducible across runs: this rerun moved `column_height_end` from
`0.0814825` to `0.0768738` on the same inputs, so both hashes changed while
every acceptance check still passed. A hash mismatch here therefore means
"regenerated", not "regressed", and re-pinning is the expected maintenance —
which is what keeps producing report-freshness churn (`TASK-POLICY-DRIFT-009`,
demos `TASK-POLICY-DRIFT-003`). Pinning the pass/fail verdict and the metric
tolerances instead of the byte hash would end that churn; recorded here rather
than changed, since the hash fields are the audit scheme's own contract.

## Resolved finding

### PD-NESTED-001 — Clean nested unit cannot execute its authoritative lane

Type: validation drift. Blocking criticality: high.

The complete test cohort deleted by `a6965e6` and required by this worktree's
runner is restored from its immediate parent. The tracked demo profile inputs
were restored from the same source commit to re-establish their directories,
then refreshed by the passing nested gate and remain scoped commit inputs.

## Machine-readable audit footer

generated_at: 2026-08-05T11:17:25Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356'
git_head: 0b47a0535f3970a3c8b6426370e63d754ea96486
freshness: fresh
status: fresh
blocking_criticality: low
human_decision_required: false
strongest_live_verification:
  - 'PASS: python tools/run_production_gate.py --include-gui; FULL-PRODUCTION-GATE-PASS.'
  - 'PASS: GUI acceptance 35/35; demo analysis 18/18; saved-demo playback report; host pytest 239 passed.'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 86406b0a7b3f0e75cf337e40a75ed800d2e0d5e490e8f551ded9155a93115cc5
gui_playback_artifact: 'artifacts/gui_playback_probe.json'
gui_playback_artifact_sha256: 8c6c8e3147abb4c4ac25eee45a3f814fbb32d6f149696da522f9b4214329a5f5
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Restore the complete branch-compatible full-lane input cohort.'
  - '[x] Rerun the nested full production gate before making a production claim.'
  - '[x] Generate nested demo and GUI artifacts, then refresh both nested reports.'
