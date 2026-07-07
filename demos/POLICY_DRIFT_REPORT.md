---
status: completed
escalated: false
generated_at: 2026-07-07T22:08:00Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos
---

# Policy Drift Report

## Scope

Audit unit: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos`

This unit contains generated demo `.blend` files, per-frame `.npz` outputs, previews, and the generated analysis report. The executable lane lives at the project root in `tools/build_demo_scenes.py`, `tools/analyze_demos.py`, and `tools/check_demo_blends.py`.

## Strongest Current Evidence

- Current generated outputs exist for `dam_break`, `crown_splash`, `slosh_tank`, and `maelstrom_basin`.
- `python tools/run_production_gate.py` reran the demo cache build, demo analysis, and demo blend checks, ending in `HEADLESS-GATE-PASS`.
- `python tools/analyze_demos.py` passed 30/30 against the current generated per-frame outputs and maelstrom metadata during that gate.
- `blender --background --factory-startup --python tools/check_demo_blends.py` opened all four `.blend` files and passed scene-content checks during that gate.
- The root GUI acceptance lane was not run in this pass; GUI-inclusive proof still requires `--include-gui`.

## Findings

No demo-unit specific unresolved policy drift was found in the current generated artifacts. The current demo analysis and blend checks support the saved outputs inside this unit.

The root report has been closed for this integration branch. Demo evidence remains headless/background proof unless the GUI-inclusive production gate is run.


## Machine-Readable Summary

generated_at: 2026-07-07T22:08:00Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos
freshness: integration-branch
status: completed
blocking_criticality: none
human_decision_required: false
strongest_live_verification:
  - python tools/run_production_gate.py (HEADLESS-GATE-PASS; GUI not run)
  - python tools/analyze_demos.py (DEMO-ANALYSIS-PASS 30/30)
  - blender --background --factory-startup --python tools/check_demo_blends.py (DEMO-BLEND-CHECK-PASS)
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - no demo-unit repair required
  - keep root policy-drift findings linked to solver/design/validation reports
