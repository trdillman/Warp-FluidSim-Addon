---
status: fresh
escalated: false
generated_at: 2026-07-26T16:45:03Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: 566a4276a4f3e7ba599519091a8a5c20b108da4b
---

# Policy Drift Report

## Truth contract

The requested unit is a real Blender 5.1 FluidSim/Warp runtime: GPU-resident
simulation, interactive playback, real source/collider/force paths, bounded
active work, and review evidence that remains traceable to the run that made
it. Demo screenshots, labels, mocked unit tests, and stale checked-in metrics
are evidence surfaces, not substitutes for runtime truth.

## Strongest live evidence

- **PASS** `rtk python tools/run_production_gate.py --include-gui`: current root
  lane reached `FULL-PRODUCTION-GATE-PASS`, GUI acceptance, saved-demo playback,
  CUDA, mesh-source, memory-safety, and demo-analysis lanes.
- **PASS** `rtk python -m pytest -q`: `295 passed in 11.74s`.
- **PASS** `artifacts/demo_analysis/analysis_report.json`: current generated
  `pass`, `18/18`; dam-break front-speed ratio `0.9758125585`,
  `Z(T=2)=2.5895752608`, column height `0.0952136219 m`; crown height
  `0.4082177579 m`; peak whitewater `4,280`; jet height `177 mm`.
- **PASS** direct saved-demo probe plus validator: `GUI-PLAYBACK-PROBE-PASS`
  and `SAVED-DEMO-GUI-PLAYBACK-PASS`; the current
  `artifacts/gui_playback_probe.json` reports `live=true`, `device=cuda:0`,
  solver frame `5`, `600,000` particles, viewport changed, and
  `passed=true`.

## Claim matrix

| Claim | Current evidence | Verdict | Gap |
|---|---|---|---|
| Real GPU FluidSim runtime | Full gate and direct Blender probe load real Warp modules on `cuda:0` and advance a saved demo | conformant for the executed lanes | Bounded to the tested scenes, frame windows, and capacities |
| Demo analysis is generated runtime evidence | Current artifact is `pass`, `18/18`, with current cache-derived metrics | conformant as bounded evidence | Do not generalize the profile to arbitrary scale |
| Checked-in demo evidence is authoritative | Checked-in JSON/Markdown exactly match the current generated artifact | conformant for this profile | Bounded to the executed scenes, frame windows, and current run |
| Policy reports describe this audit | Both reports bind to the current gate run and completed evidence cards | conformant | Documentation remains tied to this runtime artifact |
| Nested worktree is a complete production unit | Separate nested lane fails during `py_compile` because required scripts are absent | missing in that unit | See the nested-unit report and card |

## Findings

### PD-ROOT-001 — Report authority is stale against the live checkout

Type: documentation drift and validation drift. Blocking criticality: low.

Before this refresh, the root and nested reports named an older run and hid the
evidence-sync backlog. The current full gate and pytest were rerun under HEAD
`566a4276a4f3e7ba599519091a8a5c20b108da4b`; both reports now bind to that run
and expose no unresolved implementation cards.

Card: `backlog/tasks/TASK-POLICY-DRIFT-010-REPORT-AUTHORITY-RECURRENCE.md`.

### PD-ROOT-002 — Generated demo evidence disagrees with checked-in evidence

Type: validation drift and documentation drift. Blocking criticality: low.

The final generated artifact reports dam-break column height `0.0952136219 m`
and crown height `0.4082177579 m`; checked-in JSON/Markdown and the technical
report now match those values and the 18/18 result. The earlier discrepancy is
classified as run variance under the same bounded source profiles, not as a
heuristic backend or a source-evidence replacement. The nested evidence card
and this dependent integration card are complete.

Card: `backlog/tasks/TASK-POLICY-DRIFT-011-DEMO-EVIDENCE-RECURRENCE.md`.

## Guarded leads not promoted to findings

- `fluidSim_warp_blender51/sim/sourcing.py:374-446` implements cone and
  pyramid signed-distance paths, including triangle-distance handling for the
  pyramid; the current code does not support the older approximate-SDF claim.
- `fluidSim_warp_blender51/sim/whitewater.py:402-460` and
  `fluidSim_warp_blender51/sim/solver.py:1433-1450` pass an active-count hint
  through compaction and grid scatter. The older capacity-sized-work finding
  is not current source truth.
- Mocked `run_production_gate` unit tests prove gate semantics only; they were
  not used as proof of runtime execution.

## Machine-readable audit footer

generated_at: 2026-07-26T16:45:03Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: 566a4276a4f3e7ba599519091a8a5c20b108da4b
freshness: fresh
status: fresh
blocking_criticality: low
human_decision_required: false
strongest_live_verification:
  - 'PASS: rtk python tools/run_production_gate.py --include-gui; FULL-PRODUCTION-GATE-PASS.'
  - 'PASS: rtk python -m pytest -q; 295 passed in 9.21s.'
  - 'PASS: artifacts/demo_analysis/analysis_report.json; result=pass; 18/18; current metrics recorded above.'
  - 'PASS: artifacts/gui_playback_probe.json; direct live CUDA probe; solver_frame=5; particle_count=600000; viewport_changed=true.'
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - '[x] Determine whether the metric delta is expected nondeterminism or a changed source profile.'
  - '[x] Regenerate checked-in JSON/Markdown/technical wording from one documented current run.'
  - '[x] Rerun the full gate, pytest, and direct GUI probe, then refresh all report authority fields.'
