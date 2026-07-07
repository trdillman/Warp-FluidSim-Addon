---
status: fresh
escalated: false
generated_at: 2026-07-07T02:18:22Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos
git_head: 1fc54e2d751d45fd9a3fa23c8cdb625c3749cd01
---

# Policy Drift Report

## Scope

Audit unit: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos`

This unit contains generated demo `.blend` files, per-frame `.npz` outputs, previews, and the generated analysis report. The executable lane lives at the project root in `tools/build_demo_scenes.py`, `tools/analyze_demos.py`, and `tools/check_demo_blends.py`.

## Strongest Current Evidence

- Current generated outputs exist for `dam_break`, `crown_splash`, `slosh_tank`, and `maelstrom_basin`.
- `python tools/analyze_demos.py` passed 30/30 against the current generated per-frame outputs and maelstrom metadata.
- `blender --background --python tools/check_demo_blends.py` opened all four `.blend` files and passed scene-content checks.
- The root foreground GUI acceptance lane passed `GUI-ACCEPTANCE-PASS (35/35)`.

## Findings

No demo-unit specific unresolved policy drift was found in the current generated artifacts. The current demo analysis and blend checks support the saved outputs inside this unit.

The root report still tracks project-level issues that affect how demo evidence should be interpreted:

- The viewport two-frame gate can fail internally while the aggregate production gate reports pass.
- The sparse-octree product claim is stronger than the flat sparse leaf-table implementation.
- Active-work scaling repair remains tied to fixing the validation gate and rerunning the full evidence set.

## Machine-Readable Summary

generated_at: 2026-07-07T02:18:22Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos
git_head: 1fc54e2d751d45fd9a3fa23c8cdb625c3749cd01
freshness: fresh
status: fresh
blocking_criticality: none
human_decision_required: false
strongest_live_verification:
  - python tools/analyze_demos.py (DEMO-ANALYSIS-PASS 30/30)
  - blender --background --python tools/check_demo_blends.py (DEMO-BLEND-CHECK-PASS)
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - no demo-unit repair required
  - keep root policy-drift findings linked to solver/design/validation reports
