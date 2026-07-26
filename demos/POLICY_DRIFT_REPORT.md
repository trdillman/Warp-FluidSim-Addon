---
status: fresh
escalated: false
generated_at: 2026-07-26T16:45:03Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: 566a4276a4f3e7ba599519091a8a5c20b108da4b
---

# Policy Drift Report — demos unit

## Truth contract

This unit owns saved demo scenes and their analysis evidence. The evidence must
be generated from the real parent FluidSim/Warp run, remain bounded to the
executed scenes and frame windows, and agree across generated JSON, checked-in
JSON/Markdown, and dependent technical wording.

## Strongest live evidence

- **PASS** parent `rtk python tools/run_production_gate.py --include-gui`:
  `FULL-PRODUCTION-GATE-PASS`, GUI acceptance, saved-demo playback, and
  `DEMO-ANALYSIS-PASS (18/18)`.
- **PASS** current generated `artifacts/demo_analysis/analysis_report.json`:
  dam-break front-speed ratio `0.9758125585`, `Z(T=2)=2.5895752608`, column
  height `0.0952136219 m`; crown height `0.4082177579 m`; peak whitewater
  `4,280`; Worthington jet height `177 mm`.
- **PASS** direct saved-demo GUI probe and validator:
  `artifacts/gui_playback_probe.json`, live CUDA, solver frame `5`, `600,000`
  particles, viewport changed, `passed=true`.

## Claim matrix

| Claim | Evidence | Verdict | Gap |
|---|---|---|---|
| Saved demos exercise the real runtime | Parent full lane and direct Blender probe pass | conformant for this profile | Bounded to current scenes and frame windows |
| Checked-in JSON/Markdown mirror current generated evidence | Checked-in JSON/Markdown exactly match the current generated artifact | conformant for this profile | Bounded to the executed scenes and frame windows |
| A passing analyzer proves arbitrary-scale physical validity | Analyzer is 18 bounded checks | partial | It is not a scale/convergence proof |

## Findings

### PD-DEMO-001 — Nested report authority is current

Type: documentation drift and validation drift. Blocking criticality: low.

This report previously named an older run and hid the evidence-sync work. It
now binds to the parent gate run at `2026-07-26T16:45:03Z`, HEAD
`566a4276a4f3e7ba599519091a8a5c20b108da4b`, with no unresolved cards.

Card: `demos/backlog/tasks/TASK-POLICY-DRIFT-004-REPORT-AUTHORITY-RECURRENCE.md`.

### PD-DEMO-002 — Checked-in demo evidence is synchronized

Type: validation drift and documentation drift. Blocking criticality: low.

The final generated artifact reports column height `0.0952136219 m` and crown
height `0.4082177579 m`; checked-in JSON/Markdown match it exactly and report
`18/18`. The earlier delta is documented as run variance under the same
bounded source profiles, not a promoted heuristic runtime.

Card: `demos/backlog/tasks/TASK-POLICY-DRIFT-005-CHECKED-IN-EVIDENCE-RECURRENCE.md`.

## Machine-readable audit footer

generated_at: 2026-07-26T16:45:03Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: 566a4276a4f3e7ba599519091a8a5c20b108da4b
freshness: fresh
status: fresh
blocking_criticality: low
human_decision_required: false
strongest_live_verification:
  - 'PASS: parent rtk python tools/run_production_gate.py --include-gui; FULL-PRODUCTION-GATE-PASS; DEMO-ANALYSIS-PASS (18/18).'
  - 'PASS: artifacts/demo_analysis/analysis_report.json; current generated metrics recorded above.'
  - 'PASS: artifacts/gui_playback_probe.json; live=true; device=cuda:0; solver_frame=5; viewport_changed=true.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Resolve expected run variance versus changed source profile.'
  - '[x] Regenerate checked-in evidence and dependent wording from one current run.'
  - '[x] Rerun parent analyzer and saved-demo playback, then refresh this footer.'
