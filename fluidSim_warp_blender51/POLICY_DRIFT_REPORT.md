---
status: fresh
escalated: false
generated_at: 2026-09-24T10:10:12.810378+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: cb9ad0082fa27392ffdae1bff6fd481e998e8aa7
gate_run_at: 2026-09-24T10:10:12.810378+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'PACKAGE-VERIFY-PASS-PARENT-GATE-GREEN'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 98379db74ef635b73c284b4d0a50aeec0f258603763b8a0ef00a3a8ae884b5c8
git_dirty: true
---

# Policy Drift Report: fluidSim_warp_blender51 unit

## Audit conclusion

The package verifier is current and reports `release_ready: true` for
`fluidsim_warp`, version `1.0.255`. The current demo analyzer also passes
`18/18`, so this audit found no evidence that the package's real solver path is
only a narrow heuristic. The parent GUI-inclusive gate completed and wrote a
current receipt. Of the three technical-report drifts recorded here - promoted
provenance and metrics, an unattributed suite count, and a self-declared
incomplete lane tabulated as current verification - the first is resolved in
this commit; the other two remain open.

All three findings are documentation drift in one file,
`docs/technical_report.md`, and all three are owned by the single existing card
`backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md`. They are
separated here so that closing the card requires satisfying each of its
acceptance criteria rather than only the metrics refresh. That separation is
load-bearing this run: the metrics refresh has landed and the other two
criteria have not, so the card must stay open.

## Findings

### PD-PKG-001: technical-report provenance and metrics lag the current artifact (resolved this run)

Resolved in this commit. `docs/technical_report.md` now promotes proof HEAD
`cb9ad0082fa27392ffdae1bff6fd481e998e8aa7` and artifact hash
`98379db74ef635b73c284b4d0a50aeec0f258603763b8a0ef00a3a8ae884b5c8`, with column height
`0.3388671875 m` and crown height `0.2607910037 m`.

The refresh is machine-checked, not free-form: `_check_doc_metrics` in
`tools/validate_policy_drift_reports.py` requires this document to contain each
promoted metric as a literal `<key>=<value formatted %.10f>` token derived from
the gate artifact. For the current artifact those four tokens are
`front_speed_over_c0=1.0082030232`, `column_height_end=0.3388671875`,
`crown_sqrt_t_corr=0.9713672681`, and `max_crown_height=0.2607910037`. The
document now carries all four, and the validator passes.

The earlier caveat, that `max_crown_height` did not reproduce to its promoted
precision, is withdrawn. Repeat runs of one commit used to range `0.2494970709`
to `0.2904101610` (root `PD-ROOT-004`). `d2a6b6d` made the demo bakes bitwise
deterministic, and two consecutive gate runs (at `d2a6b6d` and `cb9ad00`)
produced the same artifact hash. The exact-token contract now separates a real
change from variance on every promoted key, and the document says so.

The generated artifact at `../artifacts/demo_analysis/analysis_report.json` is
untracked, and `tools/analyze_demos.py` reads an equally untracked
`artifacts/demo_cache/`. Promoting that hash on its own would have restated the
same drift class at a newer hash, because no reader could regenerate or verify
it from HEAD. The validator's own design supplied the fix:
`demos/analysis_report.json` is the committed mirror it compares against the
gate artifact, and both that file and the demo inputs appear in
`ALLOWED_EVIDENCE_PATTERNS`, so committing them does not invalidate the gate
proof. The refresh accordingly landed the artifact mirror and the document edit
in one commit, and the document names that in-tree mirror so the promoted hash
is verifiable from HEAD.

### PD-PKG-002: the promoted suite count does not distinguish the nested package suite from the parent repository suite (documentation drift)

`docs/technical_report.md` line 477 promotes `| Full Python suite | **1,043
passed** |` with no attribution to either the nested package suite or the
parent repository suite, and this audit executed no suite that would refresh
it. Still open. The gate's own pytest lane is scoped rather than exhaustive: it
runs `tests/test_pressure_projection_regressions.py`, `tests/test_cache_io.py`,
and `tests/test_io_path_policy.py`, reporting `79 passed`. That is current
evidence for those regressions and cannot substitute for the promoted
four-figure count, so no current run backs the number from either side. This is
acceptance criterion 1 of the card.

### PD-PKG-003: a self-declared incomplete lane is tabulated as current verification (documentation drift)

Section 12 does correctly label its table as a retained baseline captured at
proof HEAD `bb6d3fc5a5be9b944d4316c46aeb9ed47f011409`, so this finding is
row-level rather than table-level. The `Installed headless no-CUDA exit` row at
line 486 reports a prior packaging proof and then states that the `final 1.0.255
artifact rerun remains required`, while still being presented as a gate result
in a verification table. It must be marked historical or pending rather than
read as current proof. Still open; nothing in this gate run reruns the
installed no-CUDA packaging proof against the final 1.0.255 artifact. This is
acceptance criterion 3 of the card.

## Relationship to the root-unit finding

PD-PKG-001 and root-unit `PD-ROOT-001` described the same divergence: the same
stale hash `afb53b3f...` and the same two metric values. Both are resolved by
the single refresh landing in this commit, exactly as anticipated below. The
root unit's
`files_that_need_correction` already claims
`fluidSim_warp_blender51/docs/technical_report.md`, so this file is jointly
owned by `TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY` in this unit and
`TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE` at the root. The
refresh in this commit closes the document half of both. Neither card is closed
here: TASK-POLICY-DRIFT-001 still owns the suite-count and no-CUDA-row criteria,
and TASK-POLICY-DRIFT-015 still owns the recurrence.

## Scope boundary

The package verification pass is current package evidence. The parent
interactive GUI lane now has a fresh receipt of its own; that lane remains a
root-unit finding rather than a package one. This unit's report is validated by
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
generated_at: 2026-09-24T10:10:12.810378+00:00
generated_at_utc: 2026-09-24T10:10:12.810378+00:00
amended_at: 2026-09-21T10:19:23.7950210Z
amendment_basis: 'same audit evidence; findings re-scoped to the card acceptance criteria, no new execution'
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: cb9ad0082fa27392ffdae1bff6fd481e998e8aa7
gate_run_at: 2026-09-24T10:10:12.810378+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'PACKAGE-VERIFY-PASS-PARENT-GATE-GREEN'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 98379db74ef635b73c284b4d0a50aeec0f258603763b8a0ef00a3a8ae884b5c8
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/release/build_extension.py --verify-only -> release_ready=true, version 1.0.255'
  - 'gate demo lane -> DEMO-ANALYSIS-PASS (18/18); artifact sha256 98379db7...84b5c8, identical to the preceding gate run at d2a6b6d'
  - 'parent rtk python tools/run_production_gate.py --include-gui -> FULL-PRODUCTION-GATE-PASS with a current manifest'
  - 'gate pytest lane -> 79 passed (pressure-projection, cache-io, io-path-policy only); the promoted four-figure suite count remains unbacked this run'
  - 'rtk python tools/validate_policy_drift_reports.py -> POLICY-DRIFT-REPORT-VALIDATION-PASS'
files_that_need_correction:
  - 'docs/technical_report.md'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md'
related_findings:
  - 'root PD-ROOT-001 -> backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md (same divergence, same file)'
  - 'root PD-ROOT-002 -> backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md (closed; 3/3 passes, and the artifact hash now reproduces across runs)'
  - 'root PD-ROOT-004 -> resolved by d2a6b6d (demo bakes bitwise deterministic; promoted tokens reproduce exactly)'
repair_checklist:
  - 'Attribute the promoted suite count to the nested package suite or the parent repository suite, and record the count from a run executed for this proof; the gate pytest lane covers only 79 targeted regressions and does not supply it (PD-PKG-002, open).'
  - 'Mark the installed headless no-CUDA row historical or pending until the final 1.0.255 artifact rerun exists (PD-PKG-003, open).'
  - 'Keep package verification and parent GUI evidence labeled by execution date.'
  - 'Done this run: one artifact regenerated, mirrored to demos/analysis_report.json, and committed alongside the technical-report edit, so the promoted hash is verifiable from HEAD.'
  - 'Done this run: all four promoted metrics written as literal <key>=<%.10f> tokens - front_speed_over_c0=1.0082030232, column_height_end=0.3388671875, crown_sqrt_t_corr=0.9713672681, max_crown_height=0.2607910037 - and _check_doc_metrics passes. Recompute only if the solver or analyzer changes; the values reproduce run to run since d2a6b6d (root PD-ROOT-004, resolved).'
  - 'Done this run: the stale 2026-09-19 GUI receipts are superseded by a current manifest and are not promoted as proof.'
  - 'TASK-POLICY-DRIFT-001 stays open on criteria 1 and 3; root TASK-POLICY-DRIFT-015 stays open on the recurrence. The shared document half is closed.'
```
