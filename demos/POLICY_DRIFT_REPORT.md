---
status: fresh
escalated: false
generated_at: 2026-09-24T10:10:12.810378+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: cb9ad0082fa27392ffdae1bff6fd481e998e8aa7
gate_run_at: 2026-09-24T10:10:12.810378+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'DEMO-ANALYSIS-PASS-EVIDENCE-CURRENT'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 98379db74ef635b73c284b4d0a50aeec0f258603763b8a0ef00a3a8ae884b5c8
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
`98379db74ef635b73c284b4d0a50aeec0f258603763b8a0ef00a3a8ae884b5c8`, and `ANALYSIS.md` is
regenerated from it with a provenance block naming that hash, the proof HEAD,
and the gate run time. Current dam-break column height is `0.3388671875 m` and
current crown height is `0.2607910037 m`; both documents promote those values.
The rebuilt `dam_break`/`crown_splash` blends, previews and `meta.json` files
from this gate run land in the same commit.

`backlog/tasks/TASK-POLICY-DRIFT-007-CHECKED-IN-EVIDENCE-RECURRENCE.md` stays
active, because it owns the recurrence rather than this one synchronization.

The earlier caveat, that crown height was a single sampled run, no longer
applies. Before `d2a6b6d`, repeat runs of one commit ranged `0.2494970709` to
`0.2904101610`. That commit made the demo bakes bitwise deterministic, and two
consecutive gate runs (at `d2a6b6d` and at `cb9ad00`) produced a byte-identical
`analysis_report.json`. A future mismatch on any promoted key is therefore a
real change, not variance. The root report records this as `PD-ROOT-004`,
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
generated_at: 2026-09-24T10:10:12.810378+00:00
generated_at_utc: 2026-09-24T10:10:12.810378+00:00
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: cb9ad0082fa27392ffdae1bff6fd481e998e8aa7
gate_run_at: 2026-09-24T10:10:12.810378+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'DEMO-ANALYSIS-PASS-EVIDENCE-CURRENT'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 98379db74ef635b73c284b4d0a50aeec0f258603763b8a0ef00a3a8ae884b5c8
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'gate demo lane -> DEMO-ANALYSIS-PASS (18/18)'
  - 'artifacts/demo_analysis/analysis_report.json -> sha256 98379db7...84b5c8, mirrored byte for byte into demos/analysis_report.json'
  - 'same sha256 from the preceding gate run at d2a6b6d (artifacts/gate_repro/det_d2a6b6d/analysis_report.json)'
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
