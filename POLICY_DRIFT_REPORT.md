---
status: fresh
escalated: false
generated_at: 2026-09-22T06:23:28.891512+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: 7fa865984b7c0772e3c3d452d5d4b9f522b94858
gate_run_at: 2026-09-22T06:23:28.891512+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'FULL-GATE-PASS-EVIDENCE-CURRENT'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 453b814b588c854c4f9529ba30c121564b1d8a21f36b4e30c96ac7d935fee3b9
git_dirty: true
---

# Policy Drift Report: root unit

## Audit conclusion

Current evidence exercises the real FluidSim/Warp path end to end.
`tools/run_production_gate.py --include-gui` completed every lane at this HEAD
and ended `FULL-PRODUCTION-GATE-PASS`: the interactive Blender lane returned
`GUI-ACCEPTANCE-PASS (53/53)`, the saved-demo timeline lane returned
`SAVED-DEMO-GUI-PLAYBACK-PASS`, the demo analyzer passed `18/18`, and a fresh
`artifacts/production_gate_manifest.json` was written with `include_gui: true`.
Package verification reports `release_ready: true`. No source evidence found a
narrow heuristic solver or fabricated runtime output.

Two prior findings are resolved by this run and by the evidence refresh landing
in the same commit. One remains open, and one is newly observed.

## Findings

### PD-ROOT-001: checked-in demo evidence diverges from the fresh artifact (resolved this run)

Resolved in the same commit as this report. `demos/analysis_report.json` now
mirrors the gate artifact byte for byte, SHA-256
`453b814b588c854c4f9529ba30c121564b1d8a21f36b4e30c96ac7d935fee3b9`, and `demos/ANALYSIS.md` and
`fluidSim_warp_blender51/docs/technical_report.md` promote that artifact's four
metric tokens. The current artifact reports dam-break column height
`0.3361816406 m` and crown height `0.2547363043 m`. All three files are in
`ALLOWED_EVIDENCE_PATTERNS`, so landing them does not invalidate the proof.

`backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md`
stays active: it owns the recurrence, and one synchronization does not close
that.

### PD-ROOT-002: full GUI gate has no current completion receipt (resolved this run)

The gate ran to completion at proof HEAD `7fa865984b7c0772e3c3d452d5d4b9f522b94858`
and wrote `artifacts/production_gate_manifest.json` carrying
`gate_verdict: FULL-PRODUCTION-GATE-PASS`, `include_gui: true`, and
`demo_artifact_sha256: 453b814b588c854c4f9529ba30c121564b1d8a21f36b4e30c96ac7d935fee3b9`.
The receipt this finding asked for now exists and is current, so the stale
2026-09-19 `phase3_gui_result.json` and `gui_playback_probe.json` are no longer
load-bearing for any claim here.

`backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md` stays
active, because the card is about reproducibility rather than about a single
receipt. An earlier attempt at this same HEAD hung before reaching the GUI lane
and had to be killed; a second attempt launched while the first was still alive
contended on the shared Warp kernel and `__pycache__` directories and hung
within seconds. The passing run differs only in being launched detached, alone,
with its log written outside any transient directory. One pass is a receipt,
not yet evidence that the lane is reliable, and the two failure modes above are
worth recording on the card as known hazards.

### PD-ROOT-003: paddle spin-up threshold remains explicitly uncalibrated (validation drift)

`tools/analyze_demos.py` still labels `PADDLE_SPINUP_MOMENTUM_MARGIN = 0.3`
as a placeholder. The opt-in cache required to run the current paddle
analyzer is absent, so no current `paddle_stir` pass is promoted here. The
existing repair card is
`backlog/tasks/TASK-POLICY-DRIFT-016-PADDLE-SPINUP-THRESHOLD-CALIBRATION.md`.

### PD-ROOT-004: a promoted metric is not reproducible to its promoted precision (newly observed)

`max_crown_height` is promoted as a ten-decimal constant and is machine-checked
as an exact token, but it is a max-over-frames of a 98th-percentile particle
height on a sparse, fast-moving splash sheet. Three runs of this commit produced
`0.2494970709`, `0.2547363043`, and `0.2904101610`, a spread of roughly 16% of
the mean; the previously promoted `0.2944921553` sits at the top of that
envelope rather than outside it. Its bounded acceptance check is only `> 0.03`,
which every run clears by about an order of magnitude, and
`crown_sqrt_t_corr` repeats to five decimals across the same runs. So nothing
regressed, and the instability is specific to this one statistic.

The consequence is narrow but real: the exact-token contract in
`_check_doc_metrics` still reliably detects an unrefreshed document, but for
this key it cannot distinguish expected variance from a genuine regression, and
a future mismatch on it alone should not be read as either. No backlog card
owns this yet.

## Scope boundary and non-findings

Intentional numerical clamping, material fallbacks, effect fallbacks, and
test doubles were not promoted as heuristic-runtime findings. The `18/18`
analyzer pass, the package release verification, and the GUI-inclusive gate
receipt are all current for this HEAD.

The gate's own pytest lane is scoped, not exhaustive: it runs
`tests/test_pressure_projection_regressions.py`, `tests/test_cache_io.py`, and
`tests/test_io_path_policy.py`, which reported `79 passed`. That is current
evidence for those regressions only. No full-suite run was executed for this
proof, so the suite count promoted in the package technical report remains
unbacked this run; that stays open as package finding `PD-PKG-002`.

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
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: 7fa865984b7c0772e3c3d452d5d4b9f522b94858
gate_run_at: 2026-09-22T06:23:28.891512+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'FULL-GATE-PASS-EVIDENCE-CURRENT'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 453b814b588c854c4f9529ba30c121564b1d8a21f36b4e30c96ac7d935fee3b9
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/run_production_gate.py --include-gui -> FULL-PRODUCTION-GATE-PASS; wrote artifacts/production_gate_manifest.json (include_gui=true)'
  - 'gate GUI lane -> GUI-ACCEPTANCE-PASS (53/53); saved-demo lane -> SAVED-DEMO-GUI-PLAYBACK-PASS'
  - 'gate demo lane -> DEMO-ANALYSIS-PASS (18/18); artifact sha256 453b814b...fee3b9 mirrored to demos/analysis_report.json'
  - 'gate pytest lane (pressure-projection, cache-io, io-path-policy) -> 79 passed; full suite not run this proof'
  - 'rtk python tools/release/build_extension.py --verify-only -> release_ready=true, version 1.0.255'
  - 'rtk python tools/validate_policy_drift_reports.py -> POLICY-DRIFT-REPORT-VALIDATION-PASS'
files_that_need_correction:
  - 'tools/analyze_demos.py'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md'
  - 'backlog/tasks/TASK-POLICY-DRIFT-016-PADDLE-SPINUP-THRESHOLD-CALIBRATION.md'
  - 'backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md'
repair_checklist:
  - 'Calibrate paddle spin-up with current real positive and negative cases (PD-ROOT-003, still open).'
  - 'Make the GUI-inclusive gate reproducible rather than once-passing; record the two observed hang modes (contending second run, session-owned child) on TASK-POLICY-DRIFT-017.'
  - 'Decide how max_crown_height should be promoted given its run-to-run spread (PD-ROOT-004); it currently has no card.'
  - 'Done this run: checked-in JSON, ANALYSIS.md, and technical-report metrics refreshed from the gate artifact.'
  - 'Done this run: a bounded GUI receipt was captured as artifacts/production_gate_manifest.json.'
```
