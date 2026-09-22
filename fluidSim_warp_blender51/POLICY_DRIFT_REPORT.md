---
status: fresh
escalated: false
generated_at: 2026-09-12T18:32:21.911773+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: 01c4bf837bcec9b580bcf42511b87f3ed9b15502
gate_run_at: 2026-09-12T18:32:21.911773+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: afb53b3f847818b77e13e64b320d9d85b134a61855a30dcfb29e4ddb7efa15a1
git_dirty: true
---

# Policy Drift Report: nested package unit

## Audit conclusion

The nested `fluidSim_warp_blender51` package is an executable add-on/runtime
unit, not a report-only or heuristic facade. Its package-local suite passed
`427` tests as of a 2026-09-22 cloud-session refresh (no GPU/Blender host
available there), and `tools/release/build_extension.py --root
fluidSim_warp_blender51 --verify-only` reported `release_ready: true` for
extension `fluidsim_warp`, version `1.0.255`, with external Warp runtime
metadata. The parent GUI-inclusive production gate last exercised the real
Blender/Warp package path and passed at proof HEAD `01c4bf837bcec9b5`
(2026-09-12); that lane needs a real GPU/Blender host and was not rerun in
the 2026-09-22 refresh.

The promoted section in `docs/technical_report.md` and this report's proof
HEAD/demo metrics/artifact hash now agree with the current parent proof
(`01c4bf837bcec9b5`, 2026-09-12). This report's own `git_head` was previously
pinned to an older parent proof (`69a51c86954b`, 2026-09-01) even after the
parent gate re-ran on 2026-09-12 — that mismatch is fixed here. The
2026-09-22 package-local suite/verify-only numbers below are current
evidence from that session; the GUI/production-gate lane remains the
2026-09-12 result until Tyler reruns it on real hardware.

## Strongest live evidence

- 2026-09-22 cloud-session package-local suite (no GPU/Blender host):
  `python3 -m pytest fluidSim_warp_blender51/tests -q` -> `427 passed` in
  `23.96s`.
- 2026-09-22 cloud-session verify-only: `python3 tools/release/build_extension.py
  --root fluidSim_warp_blender51 --verify-only` -> `release_ready: true`,
  extension `fluidsim_warp`, version `1.0.255`, external Warp
  `warp_lang-1.16.0+cu12` wheel.
- Parent `artifacts/production_gate_manifest.json` (Tyler's local machine,
  2026-09-12) records proof HEAD `01c4bf837bcec9b5`, real CUDA/Warp
  execution, GUI `40/40`, saved-demo playback, and
  `FULL-PRODUCTION-GATE-PASS`. That file is git-ignored local evidence, so
  `tools/validate_policy_drift_reports.py` can only pass on a machine that
  has just produced it — it correctly reports the manifest as missing in a
  fresh clone or a GPU-less cloud session.

## Findings

### PD-PKG-001: technical report proof snapshot repaired (documentation drift) — resolved 2026-09-22

The promoted section now names parent proof HEAD `01c4bf837bcec9b5`, the
current four demo metrics, and generated artifact hash
`afb53b3f847818b77e13e64b320d9d85b134a61855a30dcfb29e4ddb7efa15a1`, matching
`docs/technical_report.md` and the root/`demos/` reports. Package-local
suite and verify-only results above are current as of 2026-09-22; the
GUI-inclusive gate itself remains the 2026-09-12 real-hardware run.
Recurrence tracking is in
`fluidSim_warp_blender51/backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md`.

## Scope boundary and non-findings

The current nested dirty files are promoted evidence only. Current tests,
package verification, and the 2026-09-12 parent runtime proof do not
support a claim that the solver, package builder, or add-on lifecycle is a
narrow heuristic. Intentional material/custom-effect fallbacks and synthetic
validation fixtures are not runtime fallbacks. Root and demos report their
own report-authority, checked-in demo-evidence, paddle-calibration, and GUI
gate-reproducibility findings.

## Machine-readable audit footer

```yaml
generated_at: 2026-09-12T18:32:21.911773+00:00
generated_at_utc: 2026-09-12T18:32:21.911773Z
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: 01c4bf837bcec9b580bcf42511b87f3ed9b15502
gate_run_at: 2026-09-12T18:32:21.911773+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: afb53b3f847818b77e13e64b320d9d85b134a61855a30dcfb29e4ddb7efa15a1
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - '2026-09-22 cloud session: python3 -m pytest fluidSim_warp_blender51/tests -q -> 427 passed in 23.96s'
  - '2026-09-22 cloud session: python3 tools/release/build_extension.py --root fluidSim_warp_blender51 --verify-only -> release_ready=true'
  - '2026-09-12 Tyler local machine: tools/run_production_gate.py --include-gui -> FULL-PRODUCTION-GATE-PASS at 01c4bf837bcec9b5'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - 'Keep parent proof HEAD, promoted metrics, and artifact hash synchronized.'
  - 'Keep package-local and installed/no-CUDA lanes labeled by execution date.'
  - 'Rerun the GUI-inclusive production gate on real GPU/Blender hardware before the next release and refresh this report from that manifest.'
```
