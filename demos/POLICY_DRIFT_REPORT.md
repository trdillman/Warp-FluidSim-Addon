---
status: fresh
escalated: false
generated_at: 2026-07-27T16:43:42Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: 77a4931fb6d16f749628141b5f25aa33017cde5f
---

# Policy Drift Report — demos unit

## Truth contract

This unit owns saved scenes and demo analysis evidence. Its checked-in JSON,
Markdown, and dependent technical wording must identify the same bounded run
as the current parent production gate.

## Strongest live evidence

- **PASS** the parent gate's headless/demo lanes at HEAD
  `77a4931fb6d16f749628141b5f25aa33017cde5f`: `DEMO-ANALYSIS-PASS (18/18)`
  and demo-blend checks.
- **PASS** parent interactive GUI acceptance and saved-demo playback probe:
  `GUI-ACCEPTANCE-PASS (35/35)` and `SAVED-DEMO-GUI-PLAYBACK-PASS`.
- **PASS** current generated `artifacts/demo_analysis/analysis_report.json`:
  dam-break front ratio `1.0117893136`, column height `0.0798046589 m`, and
  crown height `0.3577636480 m`.
- **PASS** checked-in `demos/analysis_report.json` and `demos/ANALYSIS.md` now
  match the selected generated artifact, bounded profiles, and run provenance;
  the dependent technical report matches as well.

## Finding

### PD-DEMO-001 — Resolved: checked-in demo evidence matches generated output

Type: validation drift and documentation drift. Blocking criticality: medium.

The selected bounded run passes all 18 checks. The checked-in JSON and Markdown
now match `artifacts/demo_analysis/analysis_report.json` exactly and identify
gate run `2026-07-27T16:39:08Z`, HEAD
`77a4931fb6d16f749628141b5f25aa33017cde5f`; the dependent technical report
uses the same source artifact and metrics. The prior broad card was
superseded by the narrower recurrence card, which is completed.

## Guarded leads not promoted to findings

The parent gate exercised real CUDA simulation and saved-demo playback. The
18-check analyzer is bounded evidence, not arbitrary-scale validation, but
there is no current evidence here that it is a fake solver backend.

## Machine-readable audit footer

generated_at: 2026-07-27T16:43:42Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: 77a4931fb6d16f749628141b5f25aa33017cde5f
freshness: fresh
status: fresh
blocking_criticality: low
human_decision_required: false
strongest_live_verification:
  - 'PASS: parent gate headless/demo lanes; DEMO-ANALYSIS-PASS (18/18); selected metrics above.'
  - 'PASS: GUI-ACCEPTANCE-PASS (35/35); SAVED-DEMO-GUI-PLAYBACK-PASS.'
  - 'PASS: parent artifacts/gui_playback_probe.json; live CUDA; solver_frame=5; viewport_changed=true.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Select and record one current generated run as the review authority.'
  - '[x] Synchronize checked-in JSON and Markdown with that artifact.'
  - '[x] Synchronize dependent technical wording, then rerun parent analysis.'
