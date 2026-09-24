---
status: fresh
escalated: false
generated_at: 2026-09-24T14:17:05.104120+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: 19864c3f30ef1a99153f1d8e4bd88df40a35c264
gate_run_at: 2026-09-24T14:17:05.104120+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'DEMO-ANALYSIS-PASS-EVIDENCE-CURRENT'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: a32872b81b2538034819d8eec13466678714be94872e3f14bb0fb39bd9250926
git_dirty: true
---

# Policy Drift Report: demos unit

## Audit conclusion

The current demo analyzer passes `18/18`, so the saved demos continue to
exercise real generated behavior rather than only static preview geometry. The
checked-in evidence is synchronized to this gate artifact as of this commit,
and the parent GUI-inclusive gate produced a current completion receipt. This
unit has no open finding.

## Findings

### PD-DEMO-001: checked-in evidence is stale against the fresh artifact (resolved this run)

`analysis_report.json` now mirrors the gate artifact byte for byte, SHA-256
`a32872b81b2538034819d8eec13466678714be94872e3f14bb0fb39bd9250926`, and `ANALYSIS.md` is
regenerated from it with a provenance block naming that hash, the proof HEAD,
and the gate run time. Current dam-break column height is `0.3388671875 m` and
current crown height is `0.0883056670 m`; both documents promote those values.
Since `19864c3`, crown height is measured over the crown phase only (frames 2-5,
peak at frame 4). It no longer comes from the frame-15 jet spray, which gave
`0.2607910037` and swung 20% under a 1e-6 nudge to the drop speed. The new
value is identical under the same nudges. Every other value in the artifact is
unchanged from the `cb9ad00` gate. The gate's rebuilt blends and previews were
not re-committed: the previews are pixel-identical (only the PNG `Date` and
`RenderTime` text differs), the blends differ only in save bytes, and
`meta.json` differs only in line endings.

`backlog/tasks/TASK-POLICY-DRIFT-007-CHECKED-IN-EVIDENCE-RECURRENCE.md` stays
active, because it owns the recurrence rather than this one synchronization.

The earlier caveat, that crown height was a single sampled run, no longer
applies. Before `d2a6b6d`, repeat runs of one commit ranged `0.2494970709` to
`0.2904101610`. That commit made the demo bakes bitwise deterministic, and two
consecutive gate runs (at `d2a6b6d` and at `cb9ad00`) produced a byte-identical
`analysis_report.json`. A future mismatch on any promoted key is therefore a
real change, not variance, as the deliberate crown-phase change at `19864c3`
shows. The root report records this as `PD-ROOT-004`,
now resolved.

## Scope boundary

The demo analyzer's `18/18` pass is current evidence for this unit. The parent
GUI lane now has a current receipt of its own; that gate finding is still owned
by the root report rather than here, as is the resolved `max_crown_height`
reproducibility finding.

`gate_verdict` is bound to the gate run, not to this unit:
`tools/validate_policy_drift_reports.py` compares it against `gate_verdict` in
`artifacts/production_gate_manifest.json`, so every unit report sharing one
`gate_run_at` must carry the identical string, and it must read
`FULL-PRODUCTION-GATE-PASS` once a passing manifest exists
(`tools/run_production_gate.py` writes no other value). The unit-scoped rollup
belongs in `unit_verdict`, which is free-form and may differ per unit.

## Machine-readable audit footer

```yaml
generated_at: 2026-09-24T14:17:05.104120+00:00
generated_at_utc: 2026-09-24T14:17:05.104120+00:00
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: 19864c3f30ef1a99153f1d8e4bd88df40a35c264
gate_run_at: 2026-09-24T14:17:05.104120+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'DEMO-ANALYSIS-PASS-EVIDENCE-CURRENT'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: a32872b81b2538034819d8eec13466678714be94872e3f14bb0fb39bd9250926
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'gate demo lane -> DEMO-ANALYSIS-PASS (18/18)'
  - 'artifacts/demo_analysis/analysis_report.json -> sha256 a32872b8...250926, mirrored byte for byte into demos/analysis_report.json'
  - 'vs the cb9ad00 gate (artifacts/gate_repro/det_cb9ad00/analysis_report.json): only crown_splash max_crown_height (0.2607910037 -> 0.0883056670) and its check detail differ'
  - 'parent rtk python tools/run_production_gate.py --include-gui -> FULL-PRODUCTION-GATE-PASS with a current manifest'
  - 'rtk python tools/validate_policy_drift_reports.py -> POLICY-DRIFT-REPORT-VALIDATION-PASS'
files_that_need_correction: []
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-007-CHECKED-IN-EVIDENCE-RECURRENCE.md'
repair_checklist:
  - 'Done this run: one stable generated JSON copied into the checked-in demo evidence.'
  - 'Done this run: ANALYSIS.md refreshed with the exact artifact hash preserved.'
  - 'Done this run: the policy-drift validator was run after synchronization and passed.'
  - 'Still open on TASK-POLICY-DRIFT-007: make the synchronization automatic so the drift cannot recur silently.'
```
