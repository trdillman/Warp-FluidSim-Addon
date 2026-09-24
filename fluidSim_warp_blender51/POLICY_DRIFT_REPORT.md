---
status: stale
escalated: false
generated_at: 2026-09-24T15:09:27.444825+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: 21d4d37ff210ef0e25f0854423502dd5fbd51fce
gate_run_at: 2026-09-24T15:09:27.444825+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'PACKAGE-VERIFY-PASS-PARENT-GATE-GREEN'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: a32872b81b2538034819d8eec13466678714be94872e3f14bb0fb39bd9250926
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
incomplete lane tabulated as current verification - all three are resolved:
the first on Tyler's machine in the commit that landed this proof HEAD, and
the other two in a GPU-less cloud session merging that history forward (same
resolution as a 2026-09-22 pass at the prior proof HEAD, re-applied here
after this run's regeneration reset the findings to "open" without knowledge
of that branch).

All three findings are documentation drift in one file,
`docs/technical_report.md`, and all three are owned by the single existing card
`backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md`. The
content fixes for all three are correct and stand on their own merits, but
the card itself is **reopened (`blocked`)**, not closed — see the Caveat
below for why a content fix isn't the same as a fresh proof.

## Caveat: this branch's own non-evidence changes are already past the proof HEAD

This report's `git_head` (`21d4d37`) is the gate's real proof commit on
Tyler's machine, and `gate_verdict: FULL-PRODUCTION-GATE-PASS` is true for
that tree. The branch carrying this file (`claude/project-thread-jardkj`,
PR #7) has since added its own commits with real, non-evidence changes that
postdate that proof and are **not** in `ALLOWED_EVIDENCE_PATTERNS`: two
test-guard fixes (`tests/test_artist_effect_admission.py`,
`tests/test_operator_panel_behavior.py`), a packaging-exclusion edit
(`fluidSim_warp_blender51/blender_manifest.toml`), and this unit's own
`backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md`. None of
these change solver or runtime behavior, but `_non_evidence_drift` in
`tools/validate_policy_drift_reports.py` compares the proof HEAD against
*current* HEAD, not HEAD at merge time — so this is **already true on this
branch right now**, not something that only starts once PR #7 merges into
`master`. Anyone running the validator against this branch's HEAD, with
Tyler's cached `artifacts/production_gate_manifest.json` from the `21d4d37`
run in place, will get exactly the failure `_non_evidence_drift` is designed
to produce. `status: fresh` above would be false; this report says `stale`.
**A fresh `tools/run_production_gate.py --include-gui` run on Tyler's
machine, at or after this branch's HEAD, is needed** before re-asserting
`FULL-PRODUCTION-GATE-PASS`/`status: fresh`. Flagged by an automated PR
review (`chatgpt-codex-connector[bot]`, P1, 2026-09-24, on both this report
and the task card); this caveat and the reopened card are the response, not
a rerun this GPU-less cloud session can perform.

## Findings

### PD-PKG-001: technical-report provenance and metrics lag the current artifact (resolved this run)

Resolved in this commit. `docs/technical_report.md` now promotes proof HEAD
`21d4d37ff210ef0e25f0854423502dd5fbd51fce` and artifact hash
`a32872b81b2538034819d8eec13466678714be94872e3f14bb0fb39bd9250926`, with column height
`0.3388671875 m` and crown height `0.0883056670 m`.

The refresh is machine-checked, not free-form: `_check_doc_metrics` in
`tools/validate_policy_drift_reports.py` requires this document to contain each
promoted metric as a literal `<key>=<value formatted %.10f>` token derived from
the gate artifact. For the current artifact those four tokens are
`front_speed_over_c0=1.0082030232`, `column_height_end=0.3388671875`,
`crown_sqrt_t_corr=0.9713672681`, and `max_crown_height=0.0883056670`. The
document now carries all four, and the validator passes.

The earlier caveat, that `max_crown_height` did not reproduce to its promoted
precision, is withdrawn. Repeat runs of one commit used to range `0.2494970709`
to `0.2904101610` (root `PD-ROOT-004`). `d2a6b6d` made the demo bakes bitwise
deterministic, and two consecutive gate runs (at `d2a6b6d` and `cb9ad00`)
produced the same artifact hash. The exact-token contract now separates a real
change from variance on every promoted key, and the document says so.

`19864c3` then redefined `max_crown_height`. It is now the crown-phase peak
(frames 2-5), not the frame-15 jet spray, which was deterministic but swung
20% under a 1e-6 nudge to the drop speed. That is a deliberate analyzer change,
so the token and the artifact hash moved. The other three tokens did not
move. The gate at `21d4d37`, after the non-demo scenes were removed,
reproduced that artifact byte for byte.

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

### PD-PKG-002: the promoted suite count does not distinguish the nested package suite from the parent repository suite (resolved, cloud session)

`docs/technical_report.md` line 477 promoted `| Full Python suite | **1,043
passed** |` with no attribution to either the nested package suite or the
parent repository suite. That historical row cannot be retroactively
attributed and is now explicitly labeled unattributed historical baseline
data. A "Current attributed Python suite counts" subsection immediately
after the results table gives two current, reproducible, distinctly-scoped
counts, refreshed 2026-09-24 after merging master's `21d4d37`: nested
package suite (`fluidSim_warp_blender51/tests`) **426 passed**; parent
repository suite (repo-root `pytest`) **1,440 passed, 116 skipped** (skips
are CUDA/`bpy`-only tests this environment can't run, not failures). The
gate's own scoped pytest lane (79 passed: pressure-projection, cache-io,
io-path-policy) remains noted as real-GPU evidence for those specific
regressions, not a substitute for either count. This satisfies acceptance
criterion 1 of the card.

### PD-PKG-003: a self-declared incomplete lane is tabulated as current verification (resolved, cloud session)

Section 12 does correctly label its table as a retained baseline captured at
proof HEAD `bb6d3fc5a5be9b944d4316c46aeb9ed47f011409`, so this finding was
row-level rather than table-level. The `Installed headless no-CUDA exit` row
now opens with `**PENDING — not current proof.**` and states plainly that
it's historical evidence of the contract, predating the final 1.0.255
artifact, not verification of the shipped build. The installed no-CUDA
packaging proof itself still hasn't been rerun against the final 1.0.255
artifact — that still needs GPU/Blender hardware — but the row can no
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
refresh closes the document half of both, and this unit's own suite-count/
no-CUDA-row criteria (PD-PKG-002, PD-PKG-003) are content-resolved, but
`TASK-POLICY-DRIFT-001` itself is reopened (`blocked`) pending a fresh gate
run — see the Caveat above. `TASK-POLICY-DRIFT-015` at root is a
separate concern (checked-in demo evidence recurrence) untouched by this
session — it stands on its own merits.

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
generated_at: 2026-09-24T15:09:27.444825+00:00
generated_at_utc: 2026-09-24T15:09:27.444825+00:00
amended_at: 2026-09-24T16:30:00+00:00
amendment_basis: 'PD-PKG-002 and PD-PKG-003 content fixes stand (docs/technical_report.md attributes the suite count and relabels the no-CUDA row). TASK-POLICY-DRIFT-001 reopened to blocked: an automated PR review (chatgpt-codex-connector[bot], P1, 2026-09-24) correctly identified that this branch already carries non-evidence commits past the 21d4d37 proof HEAD (two test-guard fixes, blender_manifest.toml, this task card), so _non_evidence_drift already fails against this branch HEAD, not only after merge. status set to stale; card blocked on a fresh --include-gui gate run on Tyler machine.'
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: 21d4d37ff210ef0e25f0854423502dd5fbd51fce
gate_run_at: 2026-09-24T15:09:27.444825+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'PACKAGE-VERIFY-PASS-PARENT-GATE-GREEN'
demo_artifact: '../artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: a32872b81b2538034819d8eec13466678714be94872e3f14bb0fb39bd9250926
freshness: stale
status: stale
blocking_criticality: medium
human_decision_required: true
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/release/build_extension.py --verify-only -> release_ready=true, version 1.0.255'
  - 'gate demo lane -> DEMO-ANALYSIS-PASS (18/18); artifact sha256 a32872b8...250926, byte-identical to the 19864c3 gate'
  - 'parent rtk python tools/run_production_gate.py --include-gui -> FULL-PRODUCTION-GATE-PASS with a current manifest'
  - 'gate pytest lane -> 79 passed (pressure-projection, cache-io, io-path-policy only); scoped regression evidence, not the promoted suite count'
  - 'cloud session python3 -m pytest fluidSim_warp_blender51/tests -q -> 426 passed (2026-09-24, post-merge of 21d4d37)'
  - 'cloud session python3 -m pytest -q (repo root) -> 1440 passed, 116 skipped (2026-09-24, post-merge of 21d4d37)'
  - 'rtk python tools/validate_policy_drift_reports.py -> POLICY-DRIFT-REPORT-VALIDATION-PASS'
files_that_need_correction: []
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md (blocked on a fresh --include-gui gate run; PD-PKG-002/003 content fixes stand)'
related_findings:
  - 'root PD-ROOT-001 -> backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md (same divergence, same file)'
  - 'root PD-ROOT-002 -> backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md (closed; 3/3 passes, and the artifact hash now reproduces across runs)'
  - 'root PD-ROOT-004 -> resolved by d2a6b6d (demo bakes bitwise deterministic; promoted tokens reproduce exactly)'
  - 'root PD-ROOT-004 follow-up -> max_crown_height rebound to the crown phase in 19864c3 (deterministic but chaotic before; now stable under 1e-6/1e-5 drop-speed nudges)'
repair_checklist:
  - 'Done in a GPU-less cloud session, 2026-09-24: attributed the promoted suite count to two current, reproducible scopes, refreshed for 21d4d37 (PD-PKG-002, resolved).'
  - 'Done in a GPU-less cloud session, 2026-09-24: marked the installed headless no-CUDA row PENDING/historical (PD-PKG-003, resolved).'
  - 'Keep package verification and parent GUI evidence labeled by execution date.'
  - 'Done this run: one artifact regenerated, mirrored to demos/analysis_report.json, and committed alongside the technical-report edit, so the promoted hash is verifiable from HEAD.'
  - 'Done this run: all four promoted metrics written as literal <key>=<%.10f> tokens - front_speed_over_c0=1.0082030232, column_height_end=0.3388671875, crown_sqrt_t_corr=0.9713672681, max_crown_height=0.0883056670 - and _check_doc_metrics passes. Recompute only if the solver or analyzer changes; the values reproduce run to run since d2a6b6d (root PD-ROOT-004, resolved).'
  - 'Done this run: the stale 2026-09-19 GUI receipts are superseded by a current manifest and are not promoted as proof.'
  - 'TASK-POLICY-DRIFT-001 reopened to blocked 2026-09-24: this branch already carries non-evidence commits past the 21d4d37 proof HEAD, so a fresh --include-gui gate run on Tyler machine is needed before re-closing, even though PD-PKG-002/003 content fixes are correct as written. Root TASK-POLICY-DRIFT-015 stays open on the checked-in demo evidence recurrence - a separate, unrelated card, untouched by this session.'
```
