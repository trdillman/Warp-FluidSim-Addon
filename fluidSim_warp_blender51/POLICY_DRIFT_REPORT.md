---
status: fresh
escalated: false
generated_at: 2026-09-22T06:23:28.891512+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: 7fa865984b7c0772e3c3d452d5d4b9f522b94858
gate_run_at: 2026-09-22T06:23:28.891512+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'PACKAGE-VERIFY-PASS-PARENT-GATE-GREEN'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 453b814b588c854c4f9529ba30c121564b1d8a21f36b4e30c96ac7d935fee3b9
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
incomplete lane tabulated as current verification - all three are now
resolved: the first in the prior commit, the other two in a same-day
GPU-less cloud session that attributed the suite count to two reproducible
current scopes and relabeled the no-CUDA row as pending.

All three findings are documentation drift in one file,
`docs/technical_report.md`, and all three are owned by the single existing card
`backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md`. They were
separated here so that closing the card required satisfying each of its
acceptance criteria rather than only the metrics refresh. All three are now
satisfied, so the card is closed. Closing a documentation-attribution finding
does not require GPU/Blender hardware, unlike PD-PKG-001's artifact
regeneration or the root/demos findings below, which still do.

## Findings

### PD-PKG-001: technical-report provenance and metrics lag the current artifact (resolved this run)

Resolved in this commit. `docs/technical_report.md` now promotes proof HEAD
`7fa865984b7c0772e3c3d452d5d4b9f522b94858` and artifact hash
`453b814b588c854c4f9529ba30c121564b1d8a21f36b4e30c96ac7d935fee3b9`, with column height
`0.3361816406 m` and crown height `0.2547363043 m`.

The refresh is machine-checked, not free-form: `_check_doc_metrics` in
`tools/validate_policy_drift_reports.py` requires this document to contain each
promoted metric as a literal `<key>=<value formatted %.10f>` token derived from
the gate artifact. For the current artifact those four tokens are
`front_speed_over_c0=1.0119600894`, `column_height_end=0.3361816406`,
`crown_sqrt_t_corr=0.9713680253`, and `max_crown_height=0.2547363043`. The
document now carries all four, and the validator passes.

One caveat is recorded alongside them in the document itself:
`max_crown_height` does not reproduce to its promoted precision across repeat
runs of the same commit, ranging `0.2494970709` to `0.2904101610` (root
`PD-ROOT-004`). The exact-token contract still catches an unrefreshed document,
but for that one key it cannot separate variance from regression.

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

### PD-PKG-002: the promoted suite count does not distinguish the nested package suite from the parent repository suite (resolved 2026-09-22, cloud session)

`docs/technical_report.md` line 477 promoted `| Full Python suite | **1,043
passed** |` with no attribution to either the nested package suite or the
parent repository suite. That historical row cannot be retroactively
attributed (the original Windows run recorded neither the scope nor a
reproducible command), so it is now explicitly labeled unattributed
historical baseline data instead. A new "Current attributed Python suite
counts" subsection immediately after the results table adds two current,
reproducible, distinctly-scoped counts from a GPU-less cloud session at
proof HEAD `df1868bfe301751866434c621bd8d9133740c4f0` (2026-09-22T11:21:11Z):
nested package suite (`fluidSim_warp_blender51/tests`) **427 passed**;
parent repository suite (repo-root `pytest`) **1,443 passed, 115 skipped**
(skips are CUDA/`bpy`-only tests this environment can't run, not failures).
The gate's own scoped pytest lane (79 passed) is noted as real-GPU evidence
for those specific regressions, not a substitute for either count. This
satisfies acceptance criterion 1 of the card.

### PD-PKG-003: a self-declared incomplete lane is tabulated as current verification (resolved 2026-09-22, cloud session)

Section 12 does correctly label its table as a retained baseline captured at
proof HEAD `bb6d3fc5a5be9b944d4316c46aeb9ed47f011409`, so this finding was
row-level rather than table-level. The `Installed headless no-CUDA exit` row
now opens with `**PENDING — not current proof.**` and states plainly that
it's historical evidence of the contract, predating the final 1.0.255
artifact, not verification of the shipped build. Nothing in this session
reran the installed no-CUDA packaging proof itself — that still needs a real
final-1.0.255-artifact rerun on GPU/Blender hardware — but the row can no
longer be misread as current proof. This satisfies acceptance criterion 3 of
the card.

## Relationship to the root-unit finding

PD-PKG-001 and root-unit `PD-ROOT-001` described the same divergence: the same
stale hash `afb53b3f...` and the same two metric values. Both are resolved by
the single refresh landing in this commit, exactly as anticipated below. The
root unit's
`files_that_need_correction` already claims
`fluidSim_warp_blender51/docs/technical_report.md`, so this file is jointly
owned by `TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY` in this unit and
`TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE` at the root. The
refresh closes the document half of both, and this unit's own
suite-count/no-CUDA-row criteria (PD-PKG-002, PD-PKG-003) are now also
resolved, so `TASK-POLICY-DRIFT-001` is closed. `TASK-POLICY-DRIFT-015` at
root is a separate concern (checked-in demo evidence recurrence) and is not
touched by this session — it stays open on its own merits.

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
generated_at: 2026-09-22T06:23:28.891512+00:00
generated_at_utc: 2026-09-22T06:23:28.891512+00:00
amended_at: 2026-09-22T11:21:11+00:00
amendment_basis: 'PD-PKG-002 and PD-PKG-003 resolved in a GPU-less cloud session: docs/technical_report.md edited to attribute the suite count and relabel the no-CUDA row; no gate re-execution, no new demo/GUI evidence'
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: 7fa865984b7c0772e3c3d452d5d4b9f522b94858
gate_run_at: 2026-09-22T06:23:28.891512+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'PACKAGE-VERIFY-PASS-PARENT-GATE-GREEN'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 453b814b588c854c4f9529ba30c121564b1d8a21f36b4e30c96ac7d935fee3b9
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/release/build_extension.py --verify-only -> release_ready=true, version 1.0.255'
  - 'gate demo lane -> DEMO-ANALYSIS-PASS (18/18); artifact sha256 453b814b...fee3b9'
  - 'parent rtk python tools/run_production_gate.py --include-gui -> FULL-PRODUCTION-GATE-PASS with a current manifest'
  - 'gate pytest lane -> 79 passed (pressure-projection, cache-io, io-path-policy only); scoped regression evidence, not the promoted suite count'
  - 'cloud session python3 -m pytest fluidSim_warp_blender51/tests -q -> 427 passed (2026-09-22T11:21:11Z, HEAD df1868b)'
  - 'cloud session python3 -m pytest -q (repo root) -> 1443 passed, 115 skipped (2026-09-22T11:21:11Z, HEAD df1868b)'
  - 'rtk python tools/validate_policy_drift_reports.py -> POLICY-DRIFT-REPORT-VALIDATION-PASS'
files_that_need_correction: []
active_backlog_tasks: []
related_findings:
  - 'root PD-ROOT-001 -> backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md (same divergence, same file; separate finding, not touched this session)'
  - 'root PD-ROOT-002 -> backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md (parent GUI lane; receipt captured this run, reproducibility still open)'
  - 'root PD-ROOT-004 -> no card yet (max_crown_height is promoted to a precision it does not reproduce)'
repair_checklist:
  - 'Done 2026-09-22 cloud session: attributed the promoted suite count to two current, reproducible scopes (PD-PKG-002, resolved).'
  - 'Done 2026-09-22 cloud session: marked the installed headless no-CUDA row PENDING/historical (PD-PKG-003, resolved).'
  - 'Keep package verification and parent GUI evidence labeled by execution date.'
  - 'Done this run: one artifact regenerated, mirrored to demos/analysis_report.json, and committed alongside the technical-report edit, so the promoted hash is verifiable from HEAD.'
  - 'Done this run: all four promoted metrics written as literal <key>=<%.10f> tokens - front_speed_over_c0=1.0119600894, column_height_end=0.3361816406, crown_sqrt_t_corr=0.9713680253, max_crown_height=0.2547363043 - and _check_doc_metrics passes. Recompute if the artifact is regenerated, and expect max_crown_height to move (root PD-ROOT-004).'
  - 'Done this run: the stale 2026-09-19 GUI receipts are superseded by a current manifest and are not promoted as proof.'
  - 'TASK-POLICY-DRIFT-001 is closed (all three criteria satisfied). Root TASK-POLICY-DRIFT-015 stays open on the checked-in demo evidence recurrence — a separate, unrelated card.'
```
