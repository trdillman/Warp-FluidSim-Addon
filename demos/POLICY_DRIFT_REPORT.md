---
status: fresh
escalated: false
generated_at: 2026-09-22T16:08:07.118697+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: a015e953bbb4ac44cc069b3971cc956ac399eaa7
gate_run_at: 2026-09-22T16:08:07.118697+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'DEMO-ANALYSIS-PASS-EVIDENCE-CURRENT'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 57b48bfb81a1c1512504e333127a1552ef5453f3463cb5481538536c36b6c59c
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
`453b814b588c854c4f9529ba30c121564b1d8a21f36b4e30c96ac7d935fee3b9`, and `ANALYSIS.md` is
regenerated from it with a provenance block naming that hash, the proof HEAD,
and the gate run time. Current dam-break column height is `0.3361816406 m` and
current crown height is `0.2547363043 m`; both documents promote those values.

`backlog/tasks/TASK-POLICY-DRIFT-007-CHECKED-IN-EVIDENCE-RECURRENCE.md` stays
active, because it owns the recurrence rather than this one synchronization.

One caveat for whoever closes that card: the crown height above is a single
sampled run. Repeat runs of this same commit ranged `0.2494970709` to
`0.2904101610`, so a future mismatch on that key alone is expected variance
rather than proof of stale evidence. The root report records this as
`PD-ROOT-004`.

## Scope boundary

The demo analyzer's `18/18` pass is current evidence for this unit. The parent
GUI lane now has a current receipt of its own; that gate finding is still owned
by the root report rather than here, as is the `max_crown_height`
reproducibility observation.

`gate_verdict` is bound to the gate run, not to this unit:
`tools/validate_policy_drift_reports.py` compares it against `gate_verdict` in
`artifacts/production_gate_manifest.json`, so every unit report sharing one
`gate_run_at` must carry the identical string, and it must read
`FULL-PRODUCTION-GATE-PASS` once a passing manifest exists
(`tools/run_production_gate.py` writes no other value). The unit-scoped rollup
belongs in `unit_verdict`, which is free-form and may differ per unit.

## Machine-readable audit footer

```yaml
generated_at: 2026-09-22T16:08:07.118697+00:00
generated_at_utc: 2026-09-22T06:23:28.891512+00:00
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: a015e953bbb4ac44cc069b3971cc956ac399eaa7
gate_run_at: 2026-09-22T16:08:07.118697+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'DEMO-ANALYSIS-PASS-EVIDENCE-CURRENT'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 57b48bfb81a1c1512504e333127a1552ef5453f3463cb5481538536c36b6c59c
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'gate demo lane -> DEMO-ANALYSIS-PASS (18/18)'
  - 'artifacts/demo_analysis/analysis_report.json -> sha256 453b814b...fee3b9, mirrored byte for byte into demos/analysis_report.json'
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
