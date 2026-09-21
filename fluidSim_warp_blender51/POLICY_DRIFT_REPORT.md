---
status: fresh
escalated: false
generated_at: 2026-09-21T03:36:27.7252393Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: efb02e4e8cb4a8833e2e0631b5f33eee205ea77c
gate_run_at: 2026-09-21T03:12:48.375188Z
gate_verdict: 'BLOCKED-INTERACTIVE-GUI-NO-RECEIPT'
unit_verdict: 'PACKAGE-VERIFY-PASS-PARENT-GUI-BLOCKED'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a
git_dirty: true
---

# Policy Drift Report: fluidSim_warp_blender51 unit

## Audit conclusion

The package verifier is current and reports `release_ready: true` for
`fluidsim_warp`, version `1.0.255`. The current demo analyzer also passes
`18/18`, so this audit found no evidence that the package's real solver path is
only a narrow heuristic. The package technical report is nevertheless stale
against the current generated artifact in three distinct ways - promoted
provenance and metrics, an unattributed suite count, and a self-declared
incomplete lane tabulated as current verification - and the parent GUI gate has
no current completion receipt.

All three findings are documentation drift in one file,
`docs/technical_report.md`, and all three are owned by the single existing card
`backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md`. They are
separated here so that closing the card requires satisfying each of its
acceptance criteria rather than only the metrics refresh.

## Findings

### PD-PKG-001: technical-report provenance and metrics lag the current artifact (documentation drift)

`docs/technical_report.md` still promotes proof HEAD `01c4bf...`, artifact hash
`afb53b3f847818b77e13e64b320d9d85b134a61855a30dcfb29e4ddb7efa15a1`, and
older column/crown metrics. The current generated artifact is
`ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a` with
column height `0.3355760574 m` and crown height `0.2811620831 m`.

The refresh is machine-checked, not free-form: `_check_doc_metrics` in
`tools/validate_policy_drift_reports.py` requires this document to contain each
promoted metric as a literal `<key>=<value formatted %.10f>` token derived from
the gate artifact. For the current artifact those four tokens are
`front_speed_over_c0=1.0113339117`, `column_height_end=0.3355760574`,
`crown_sqrt_t_corr=0.9713680253`, and `max_crown_height=0.2811620831`. The
document currently carries the first token but not the other three.

The generated artifact at `../artifacts/demo_analysis/analysis_report.json` is
untracked, and `tools/analyze_demos.py` reads an equally untracked
`artifacts/demo_cache/`. Promoting that hash on its own would restate the same
drift class at a newer hash, because no reader could regenerate or verify it
from HEAD. The validator's own design supplies the fix: `demos/analysis_report.json`
is the committed mirror it compares against the gate artifact, and both that
file and the demo inputs appear in `ALLOWED_EVIDENCE_PATTERNS`, so committing
them does not invalidate the gate proof. The refresh must therefore land the
artifact mirror and the document edit in the same commit.

### PD-PKG-002: the promoted suite count does not distinguish the nested package suite from the parent repository suite (documentation drift)

`docs/technical_report.md` line 477 promotes `| Full Python suite | **1,043
passed** |` with no attribution to either the nested package suite or the
parent repository suite, and this audit executed no suite that would refresh
it. The root-unit audit for the same gate run records `rtk python -m pytest -q`
as interrupted after a bounded wait with no verdict, so no current run backs
the number from either side. This is acceptance criterion 1 of the card.

### PD-PKG-003: a self-declared incomplete lane is tabulated as current verification (documentation drift)

Section 12 does correctly label its table as a retained baseline captured at
proof HEAD `bb6d3fc5a5be9b944d4316c46aeb9ed47f011409`, so this finding is
row-level rather than table-level. The `Installed headless no-CUDA exit` row at
line 486 reports a prior packaging proof and then states that the `final 1.0.255
artifact rerun remains required`, while still being presented as a gate result
in a verification table. It must be marked historical or pending rather than
read as current proof. This is acceptance criterion 3 of the card.

## Relationship to the root-unit finding

PD-PKG-001 and root-unit `PD-ROOT-001` describe the same divergence: the same
stale hash `afb53b3f...` and the same two metric values. The root unit's
`files_that_need_correction` already claims
`fluidSim_warp_blender51/docs/technical_report.md`, so this file is jointly
owned by `TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY` in this unit and
`TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE` at the root. One
refresh closes the document half of both; neither card should be resolved
without checking the other, or one will be left falsely red.

## Scope boundary

The package verification pass is current package evidence. It does not prove
the parent interactive GUI lane, which was blocked without a fresh receipt;
that remains a root-unit finding. This unit's report is now validated by
`tools/validate_policy_drift_reports.py` alongside the root and `demos`
reports; the findings above were confirmed by direct inspection of the
document, the generated artifact, and the validator's own checks.

`gate_verdict` is bound to the gate run, not to this unit:
`tools/validate_policy_drift_reports.py` compares it against `gate_verdict` in
`artifacts/production_gate_manifest.json`, so every unit report sharing one
`gate_run_at` must carry the identical string, and it must read
`FULL-PRODUCTION-GATE-PASS` once a passing manifest exists
(`tools/run_production_gate.py` writes no other value). The unit-scoped rollup
belongs in `unit_verdict`, which is free-form and may differ per unit.

## Machine-readable audit footer

```yaml
generated_at: 2026-09-21T03:36:27.7252393Z
generated_at_utc: 2026-09-21T03:36:27.7252393Z
amended_at: 2026-09-21T10:19:23.7950210Z
amendment_basis: 'same audit evidence; findings re-scoped to the card acceptance criteria, no new execution'
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: efb02e4e8cb4a8833e2e0631b5f33eee205ea77c
gate_run_at: 2026-09-21T03:12:48.375188Z
gate_verdict: 'BLOCKED-INTERACTIVE-GUI-NO-RECEIPT'
unit_verdict: 'PACKAGE-VERIFY-PASS-PARENT-GUI-BLOCKED'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/release/build_extension.py --verify-only -> release_ready=true, version 1.0.255'
  - 'rtk python tools/analyze_demos.py -> DEMO-ANALYSIS-PASS (18/18)'
  - 'parent rtk python tools/run_production_gate.py --include-gui -> no current receipt'
  - 'parent rtk python -m pytest -q -> interrupted, no verdict; the promoted suite count is unbacked this run'
files_that_need_correction:
  - 'docs/technical_report.md'
  - '../demos/analysis_report.json'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md'
related_findings:
  - 'root PD-ROOT-001 -> backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md (same divergence, same file)'
  - 'root PD-ROOT-002 -> backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md (the blocked parent GUI lane)'
repair_checklist:
  - 'Regenerate one artifact, copy it to demos/analysis_report.json, and commit that mirror in the same commit as the technical-report edit; never promote a hash that exists only as an untracked file.'
  - 'Write all four promoted metrics as literal <key>=<%.10f> tokens so _check_doc_metrics passes: front_speed_over_c0=1.0113339117, column_height_end=0.3355760574, crown_sqrt_t_corr=0.9713680253, max_crown_height=0.2811620831 (recompute if the artifact is regenerated).'
  - 'Attribute the promoted suite count to the nested package suite or the parent repository suite, and record the count from a run executed for this proof.'
  - 'Mark the installed headless no-CUDA row historical or pending until the final 1.0.255 artifact rerun exists.'
  - 'Keep package verification and parent GUI evidence labeled by execution date.'
  - 'Do not promote the stale 2026-09-19 GUI receipts as current proof.'
  - 'Close TASK-POLICY-DRIFT-001 and root TASK-POLICY-DRIFT-015 together, or state explicitly which half remains open.'
```
