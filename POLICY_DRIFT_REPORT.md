---
status: fresh
escalated: false
generated_at: 2026-07-25T16:53:31Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: a6965e679ac3fd276c0c81cbfc1ee67257585739
---

# Policy Drift Report

## Truth contract

This parent unit claims a real Blender 5.1 FluidSim add-on with GPU-resident
Warp simulation, interactive viewport playback, evaluated mesh sources, sparse
8x8x8 leaf-grid storage, and truthful production gates. The `demos` directory
is a nested review unit; its files are outputs of this runtime, not substitutes
for it.

## Verdict

The current runtime is real within the exercised acceptance boundary. The
current Blender/CUDA full lane and the independent host regression suite pass.
The live lane exercised registered Warp kernels, CUDA solver steps, mesh-source
acceptance, GUI redraw/render output, saved-demo playback, and generated demo
analysis. No current runtime artifact establishes a fabricated fallback,
shallow backend, or heuristic substitute for the solver.

The prior current drift was evidence authority: the checked-in demo evidence
and both policy reports were stale or overstated their freshness relative to
the current rerun. The checked-in demo JSON/Markdown now match the generated
artifact's 18/18 result and metrics from the same bounded source run.

## Strongest live evidence

- **PASS** `python tools/run_production_gate.py --include-gui`: current HEAD
  `a6965e679ac3`, `FULL-PRODUCTION-GATE-PASS`, GUI acceptance `35/35`,
  `MEMORY-SAFETY-GATE-PASS`, mesh-source acceptance, demo blend checks, and
  saved-demo GUI playback pass.
- **PASS** `python -m pytest -q`: `256 passed in 4.59s`.
- **PASS** `artifacts/demo_analysis/analysis_report.json`: result `pass`,
  `18/18`; current dam-break metrics are front-speed ratio `0.9758126`,
  `Z(T=2)=2.5895753`, and end column height `0.0814825 m`.
- **PASS** `artifacts/gui_playback_probe.json`: `passed=true`, `live=true`,
  `device=cuda:0`, solver frame `5`, viewport changed, and status reports
  `600,000` particles.

## Claim matrix

| Claim | Current result | Boundary |
|---|---|---|
| CUDA solver and registered Warp kernels are real | PASS | Full gate smoke, solver, mesh-source acceptance, and GUI lane |
| Interactive playback produces changing visible output | PASS | GUI acceptance and saved-demo playback artifact |
| Demo metrics are current and reviewable | PASS | Generated artifact and checked-in JSON/Markdown match |
| Arbitrary scale, convergence, and long-run physical fidelity | UNPROVEN | Current lane does not certify those broader claims |

## Findings

### PD-ROOT-001 — Stale report authority and active-task footer

Type: documentation drift and validation drift
Status: corrected in this report and completed in
`backlog/tasks/TASK-POLICY-DRIFT-009-REPORT-FRESHNESS.md`.

The prior root report was generated at `2026-07-24T15:13:41Z` and bound to
HEAD `7fc3f522...`, while this audit ran at HEAD `a6965e679ac3`. Its footer also
listed no active tasks even though the root freshness card and nested freshness
card were `ready-for-agent`. This report is rebound to the current run and
its footer now records the completed freshness cards instead of promoting the
old completion claim.

### PD-ROOT-002 — Current generated demo evidence disagrees with checked-in evidence

Type: validation drift and documentation drift
Status: corrected by
`backlog/tasks/TASK-POLICY-DRIFT-006-DEMO-EVIDENCE-SYNC.md` and the nested
`demos/backlog/tasks/TASK-POLICY-DRIFT-002-CHECKED-IN-EVIDENCE.md`.

The current full lane and standalone analyzer wrote
`artifacts/demo_analysis/analysis_report.json` with dam-break values
`0.9758126`, `2.5895753`, and `0.0814825 m`. The checked-in
`demos/analysis_report.json` and `demos/ANALYSIS.md` now carry the same 18/18
result and rounded metrics, and the technical report records the same run and
bounded source profiles.

## Heuristic-implementation assessment

Static leads were checked but not promoted to implementation drift because the
current full lane exercised the real CUDA path, registered kernels, mesh
sourcing, GUI playback, and generated outputs. Analytic shape SDFs, display
proxies, input sanitization, and test doubles are explicit mechanisms in their
respective contracts, not evidence of a solver replacement. The current scale
boundary remains limited to the exercised capacities and frame windows; claims
beyond those bounds need new scale/convergence evidence.

## Worktree preservation

The worktree was already dirty. The full gate regenerated demo artifacts; no
reset, cleanup, restore, or unrelated implementation repair was performed.
This audit deliberately changed only the two policy reports and the matching
local task-card authority/state surfaces.

## Machine-readable audit footer

generated_at: 2026-07-25T16:53:31Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: a6965e679ac3fd276c0c81cbfc1ee67257585739
freshness: fresh
status: fresh
blocking_criticality: low
human_decision_required: false
strongest_live_verification:
  - 'PASS: python tools/run_production_gate.py --include-gui; FULL-PRODUCTION-GATE-PASS; GUI-ACCEPTANCE-PASS (35/35); SAVED-DEMO-GUI-PLAYBACK-PASS.'
  - 'PASS: python -m pytest -q; 256 passed in 4.59s.'
  - 'PASS: artifacts/demo_analysis/analysis_report.json and demos/analysis_report.json; result=pass; 18/18; metrics synchronized.'
  - 'PASS: artifacts/gui_playback_probe.json; live=true; device=cuda:0; solver_frame=5; viewport_changed=true.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Regenerate checked-in demo JSON and Markdown from the current full-lane artifact.'
  - '[x] Update the technical report to the same run, metrics, and source profile.'
  - '[x] Rerun the full gate and pytest, then verify all authority surfaces share one HEAD and timestamp.'
