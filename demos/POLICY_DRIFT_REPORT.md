---
status: fresh
escalated: false
generated_at: 2026-07-25T16:53:31Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: a6965e679ac3fd276c0c81cbfc1ee67257585739
---

# Policy Drift Report

## Truth contract

This nested unit owns demo blends, previews, checked-in analysis, and local
review tasks. Its evidence must come from the parent Blender/CUDA runtime and
must not be promoted beyond the executed frame windows, capacities, and
metrics.

## Verdict

The current demo runtime is substantive within the exercised review boundary.
The parent full lane built and analyzed the demos with real CUDA solver steps,
passed all 18 analyzer checks, opened the saved demo, and observed live GUI
playback through solver frame 5 with 600,000 particles. No current artifact
supports a fabricated or narrow demo backend finding.

The checked-in JSON and Markdown now match the generated analysis artifact's
18/18 result and numerical dam-break metrics from the same bounded source run.

## Strongest live evidence

- **PASS** parent `python tools/run_production_gate.py --include-gui`:
  `FULL-PRODUCTION-GATE-PASS`, demo analysis `18/18`, demo blend checks pass,
  GUI acceptance `35/35`, and saved-demo playback pass.
- **PASS** `artifacts/demo_analysis/analysis_report.json`: current generated
  `pass`, `18/18`; dam-break front-speed ratio `0.9758126`, `Z(T=2)=2.5895753`,
  and end column height `0.0814825 m`.
- **PASS** checked-in `demos/analysis_report.json` and `demos/ANALYSIS.md`:
  same 18/18 result and synchronized rounded metrics.
- **PASS** `artifacts/gui_playback_probe.json`: live CUDA playback, device
  `cuda:0`, solver frame `5`, viewport changed, `600,000` particles.

## Findings

### PD-DEMO-001 — Stale nested report authority and active-task footer

Type: documentation drift and validation drift
Status: corrected in this report and completed in
`demos/backlog/tasks/TASK-POLICY-DRIFT-003-REPORT-FRESHNESS.md`.

The prior nested report was generated at `2026-07-24T15:13:41Z` and bound to
HEAD `7fc3f522...`, not the current HEAD `a6965e679ac3`. It also listed no
active tasks while the nested freshness and evidence cards were
`ready-for-agent`. This report is now bound to the current audit and names the
the completed freshness cards explicitly.

### PD-DEMO-002 — Checked-in demo evidence is stale against the current run

Type: validation drift and documentation drift
Status: corrected by
`demos/backlog/tasks/TASK-POLICY-DRIFT-002-CHECKED-IN-EVIDENCE.md` and the
parent `backlog/tasks/TASK-POLICY-DRIFT-006-DEMO-EVIDENCE-SYNC.md`.

The checked-in JSON/Markdown are internally passing and now describe the same
simulation result as the current full-lane artifact. This correction resolves
the authority failure; it is not proof that the live solver is heuristic. The
evidence remains bounded to the executed source profiles and frame windows.

## Heuristic-implementation assessment

No heuristic or fallback implementation finding was promoted. The live parent
lane exercised the actual CUDA path, demo scene construction, surface and
whitewater outputs, GUI playback, and saved artifacts. Static proxy metadata
and validation test doubles remain bounded mechanisms in their respective
contracts. The demo evidence still does not certify arbitrary scale,
convergence, or long-run physical fidelity.

## Machine-readable audit footer

generated_at: 2026-07-25T16:53:31Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos'
git_head: a6965e679ac3fd276c0c81cbfc1ee67257585739
freshness: fresh
status: fresh
blocking_criticality: low
human_decision_required: false
strongest_live_verification:
  - 'PASS: parent python tools/run_production_gate.py --include-gui; FULL-PRODUCTION-GATE-PASS; DEMO-ANALYSIS-PASS (18/18).'
  - 'PASS: artifacts/demo_analysis/analysis_report.json and demos/analysis_report.json; current generated and checked-in values match; 18/18.'
  - 'PASS: artifacts/gui_playback_probe.json; live=true; device=cuda:0; solver_frame=5; viewport_changed=true.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Regenerate demos/analysis_report.json and demos/ANALYSIS.md from artifacts/demo_analysis/analysis_report.json or a documented rerun.'
  - '[x] Update the parent technical report to the same values and source profile.'
  - '[x] Rerun demo analysis and saved-demo playback, then verify nested and parent authority surfaces agree.'
