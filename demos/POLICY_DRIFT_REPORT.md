---
status: fresh
escalated: false
generated_at: 2026-08-05T18:02:06.351953+00:00
audit_unit: 'C:\tmp\fluidsim-resolution-quality-merge-20260805\demos'
git_head: cc25ed001ae0a0664cc6f43b605bc7c1cfa04f1b
gate_run_at: 2026-08-05T18:02:06.351953+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 74f846dd208251538b1491801ea43dd387736d13361b7bff08a0da5b129ffb45
---

# Policy Drift Report — demos unit

## Strongest live evidence

- The parent full gate passed with real Blender/Warp execution and wrote the
  current manifest: `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (38/38)`,
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

generated_at: 2026-08-05T18:02:06.351953+00:00
audit_unit: 'C:\tmp\fluidsim-resolution-quality-merge-20260805\demos'
git_head: cc25ed001ae0a0664cc6f43b605bc7c1cfa04f1b
gate_run_at: 2026-08-05T18:02:06.351953+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 74f846dd208251538b1491801ea43dd387736d13361b7bff08a0da5b129ffb45
freshness: fresh
status: fresh
blocking_criticality: none
human_decision_required: false
strongest_live_verification:
  - 'PASS: parent full gate; DEMO-ANALYSIS-PASS (18/18).'
  - 'PASS: GUI-ACCEPTANCE-PASS (38/38) and SAVED-DEMO-GUI-PLAYBACK-PASS.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Synchronize checked-in demo JSON with the current artifact.'
  - '[x] Synchronize ANALYSIS.md and technical report metrics/provenance.'
  - '[x] Validate the refreshed report and technical wording.'
