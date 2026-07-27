---
status: fresh
escalated: false
generated_at: 2026-07-27T16:43:42Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: 77a4931fb6d16f749628141b5f25aa33017cde5f
---

# Policy Drift Report — root unit

## Truth contract

This unit claims a real Blender 5.1 FluidSim/Warp runtime: CUDA-resident
simulation, live viewport playback, real source/collider/force paths, and
review evidence bound to the run that produced it. A passing bounded lane is
not evidence of arbitrary-scale physical validity, and checked-in summaries
must not outrank current generated artifacts.

## Strongest live evidence

- **PASS** the headless/demo portion of `rtk python
  tools/run_production_gate.py --include-gui` at HEAD
  `77a4931fb6d16f749628141b5f25aa33017cde5f`: real CUDA solver, mesh-source,
  force-region, memory-safety, viewport, source-UI, demo-build, demo-analysis,
  and demo-blend lanes passed; `DEMO-ANALYSIS-PASS (18/18)`.
- **PASS** interactive GUI acceptance: `GUI-ACCEPTANCE-PASS (35/35)`.
- **PASS** saved-demo GUI probe and validator: `GUI-PLAYBACK-PROBE-PASS` and
  `SAVED-DEMO-GUI-PLAYBACK-PASS`.
- **PASS** `rtk python -m pytest -q`: `343 passed in 9.90s`.
- **PASS** current `artifacts/demo_analysis/analysis_report.json`: `18/18`.
  Selected generated values are dam-break front ratio `1.0117893136`, column
  height `0.0798046589 m`, and crown height `0.3577636480 m`.
- **PASS** `artifacts/gui_playback_probe.json`: `live=true`, `device=cuda:0`,
  `solver_frame=5`, `particle_count=128000`, `viewport_changed=true`.

## Claim matrix

| Claim | Current evidence | Verdict | Gap |
|---|---|---|---|
| The root runtime is a real GPU-backed FluidSim path | Headless production lanes plus separate GUI acceptance and saved-demo playback reached CUDA, GUI, mesh, force, and playback paths | conformant as bounded runtime evidence | Do not generalize to untested scale or convergence |
| Current generated demo analysis passes | `artifacts/demo_analysis/analysis_report.json`, `18/18` | conformant as bounded evidence | Do not generalize to arbitrary scale |
| Checked-in demo/technical evidence describes the selected run | JSON, Markdown, and technical report match the selected artifact and provenance | conformant as bounded evidence | Run variance remains possible between bounded executions |

## Findings

### PD-ROOT-001 — Resolved: generated demo evidence is synchronized

Type: validation drift and documentation drift. Blocking criticality: medium.

The selected generated artifact passes `18/18`, and the checked-in demo JSON,
Markdown, and technical report now match its metrics and provenance: gate run
`2026-07-27T16:39:08Z`, HEAD `77a4931fb6d16f749628141b5f25aa33017cde5f`, and
the bounded `128,000/72` and `259,112/48` scene profiles. The two broad cards
were superseded; the narrower nested evidence and technical-report cards are
completed.

Cards completed: `backlog/tasks/TASK-POLICY-DRIFT-011-DEMO-EVIDENCE-RECURRENCE.md`
and `demos/backlog/tasks/TASK-POLICY-DRIFT-005-CHECKED-IN-EVIDENCE-RECURRENCE.md`.

## Guarded leads not promoted to findings

- `fluidSim_warp_blender51/sim/sourcing.py` contains implemented cone and
  pyramid signed-distance paths, including triangle-distance handling; the
  older approximate-SDF claim is not current source truth.
- `fluidSim_warp_blender51/sim/whitewater.py` and `sim/solver.py` pass active
  count hints through compaction and scatter; the older capacity-only work
  claim is not current source truth.
- The current gate exercised real Warp CUDA and Blender paths. Mocked gate
  tests and synthetic panel-validation rows were not used as runtime proof.

## Machine-readable audit footer

generated_at: 2026-07-27T16:43:42Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: 77a4931fb6d16f749628141b5f25aa33017cde5f
freshness: fresh
status: fresh
blocking_criticality: low
human_decision_required: false
strongest_live_verification:
  - 'PASS: current gate headless/demo lanes; DEMO-ANALYSIS-PASS (18/18); HEAD 77a4931fb6d16f749628141b5f25aa33017cde5f.'
  - 'PASS: GUI-ACCEPTANCE-PASS (35/35); GUI-PLAYBACK-PROBE-PASS; SAVED-DEMO-GUI-PLAYBACK-PASS.'
  - 'PASS: rtk python -m pytest -q; 343 passed in 9.90s.'
  - 'PASS: artifacts/demo_analysis/analysis_report.json; result=pass; 18/18; selected metrics above.'
  - 'PASS: artifacts/gui_playback_probe.json; live CUDA; solver_frame=5; viewport_changed=true.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Select and record one current generated run as the review authority.'
  - '[x] Synchronize checked-in JSON, Markdown, and technical-report metrics/provenance.'
  - '[x] Rerun the required production lanes and verify all evidence surfaces agree.'
