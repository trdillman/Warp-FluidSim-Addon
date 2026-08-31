---
status: fresh
escalated: false
generated_at: 2026-08-31T18:41:50.058126+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: 07963107f422a1d1399c7fd150c6c2314e563cad
gate_run_at: 2026-08-31T18:41:50.058126+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 371403b4bb55813c195917a3e9357f1d67ed5a8bb9d9d5b6b74b1b64e960864c
git_dirty: false
---

# Policy Drift Report: nested package unit

## Audit conclusion

The nested `fluidSim_warp_blender51` package is an executable add-on/runtime
unit, not a report-only or heuristic facade. Its current package-local suite
passed `398` tests, and `release/build_extension.py --verify-only` reported
`release_ready: true` for extension `fluidsim_warp`, version `1.0.255`, with
external Warp runtime metadata. The parent GUI-inclusive production gate also
exercised the real Blender/Warp package path and passed.

The promoted section in `docs/technical_report.md` is synchronized to the
current parent proof HEAD, demo metrics, and artifact hash. Older package-local
suite and verify-only results remain historical evidence rather than claims
that those lanes were rerun during this refresh.

## Strongest live evidence

- Historical package-local evidence records `398` tests passed in `15.74s`.
- `rtk python release/build_extension.py --verify-only --json-report
  ..\artifacts\package_verify_only.json` returned `release_ready: true`,
  extension `fluidsim_warp`, version `1.0.255`, and external Warp 1.16.0
  runtime metadata.
- Parent `artifacts/production_gate_manifest.json` records proof HEAD
  `07963107f422`, real CUDA/Warp execution, GUI `40/40`, saved-demo playback,
  and `FULL-PRODUCTION-GATE-PASS`.

## Findings

### PD-PKG-001: technical report proof snapshot repaired (documentation drift)

The promoted section now names parent proof HEAD `07963107f422`, the current
four demo metrics, and generated artifact hash
`371403b4bb55813c195917a3e9357f1d67ed5a8bb9d9d5b6b74b1b64e960864c`.
Package-local results remain explicitly historical. Recurrence tracking is in
`fluidSim_warp_blender51/backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md`.

## Scope boundary and non-findings

The current nested dirty files are promoted evidence only. Current tests,
historical package verification, and parent runtime proof do not
support a claim that the solver, package builder, or add-on lifecycle is a
narrow heuristic. Intentional material/custom-effect fallbacks and synthetic
validation fixtures are not runtime fallbacks. Root and demos report their
own report-authority, checked-in demo-evidence, paddle-calibration, and GUI
gate-reproducibility findings.

## Machine-readable audit footer

```yaml
generated_at: 2026-08-31T18:41:50.058126+00:00
generated_at_utc: 2026-08-31T18:41:50.058126Z
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: 07963107f422a1d1399c7fd150c6c2314e563cad
gate_run_at: 2026-08-31T18:41:50.058126+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 371403b4bb55813c195917a3e9357f1d67ed5a8bb9d9d5b6b74b1b64e960864c
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python -m pytest -q -> 398 passed in 15.74s'
  - 'rtk python release/build_extension.py --verify-only -> artifacts/package_verify_only.json, release_ready=true'
  - 'parent rtk python tools/run_production_gate.py --include-gui -> FULL-PRODUCTION-GATE-PASS'
files_that_need_correction: []
active_backlog_tasks:
  - 'fluidSim_warp_blender51/backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md'
repair_checklist:
  - 'Keep parent proof HEAD, promoted metrics, and artifact hash synchronized.'
  - 'Keep package-local and installed/no-CUDA lanes labeled by execution date.'
```
