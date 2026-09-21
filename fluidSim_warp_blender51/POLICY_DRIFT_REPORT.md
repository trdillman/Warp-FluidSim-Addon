---
status: fresh
escalated: false
generated_at: 2026-09-21T03:36:27.7252393Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: efb02e4e8cb4a8833e2e0631b5f33eee205ea77c
gate_run_at: 2026-09-21T03:12:48.375188Z
gate_verdict: 'PACKAGE-VERIFY-PASS-PARENT-GUI-BLOCKED'
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
against the current generated artifact, and the parent GUI gate has no current
completion receipt.

## Findings

### PD-PKG-001: technical-report provenance and metrics lag the current artifact (documentation drift)

`docs/technical_report.md` still promotes proof HEAD `01c4bf…`, artifact hash
`afb53b3f847818b77e13e64b320d9d85b134a61855a30dcfb29e4ddb7efa15a1`, and
older column/crown metrics. The current generated artifact is
`ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a` with
column height `0.3355760574 m` and crown height `0.2811620831 m`. The existing
local card is
`backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md`.

## Scope boundary

The package verification pass is current package evidence. It does not prove
the parent interactive GUI lane, which was blocked without a fresh receipt;
that remains a root-unit finding.

## Machine-readable audit footer

```yaml
generated_at: 2026-09-21T03:36:27.7252393Z
generated_at_utc: 2026-09-21T03:36:27.7252393Z
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\fluidSim_warp_blender51'
git_head: efb02e4e8cb4a8833e2e0631b5f33eee205ea77c
gate_run_at: 2026-09-21T03:12:48.375188Z
gate_verdict: 'PACKAGE-VERIFY-PASS-PARENT-GUI-BLOCKED'
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
files_that_need_correction:
  - 'docs/technical_report.md'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-001-TECHNICAL-REPORT-AUTHORITY.md'
repair_checklist:
  - 'Refresh technical-report HEAD, hash, and promoted metrics from one stable artifact.'
  - 'Keep package verification and parent GUI evidence labeled by execution date.'
  - 'Do not promote the stale 2026-09-19 GUI receipts as current proof.'
```
