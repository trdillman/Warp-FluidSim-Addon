---
status: fresh
escalated: false
generated_at: 2026-07-03T21:18:03Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos
git_head: 91388c67699aac3c4c693a5dd6545e5951e7d3ce
---

# Policy Drift Report

## Scope

Audit unit: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos`

This unit contains generated demo `.blend` files, per-frame `.npz` outputs, previews, and the generated analysis report. The executable lane lives at the project root in `tools/build_demo_scenes.py`, `tools/analyze_demos.py`, and `tools/check_demo_blends.py`.

## Strongest Current Evidence

- Current generated outputs exist for `dam_break`, `crown_splash`, and `slosh_tank`.
- `python tools/analyze_demos.py` passed 25/25 against the current generated per-frame outputs.
- `blender --background --python tools/check_demo_blends.py` opened all three `.blend` files and passed scene-content checks.
- The root Blender/Warp acceptance lanes also passed on live CUDA/Warp runtime.

## Findings

No demo-unit specific unresolved policy drift was found in the current generated artifacts. The current demo analysis and blend checks support the saved outputs inside this unit.

The root report still tracks two project-level issues that affect how the demo evidence should be interpreted:

- The sparse-octree product claim is stronger than the flat sparse leaf-table implementation.
- Capacity-proportional solver phases are not proven robust under larger allocation envelopes.

## Machine-Readable Summary

generated_at: 2026-07-03T21:18:03Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos
git_head: 91388c67699aac3c4c693a5dd6545e5951e7d3ce
freshness: fresh
status: fresh
blocking_criticality: none
human_decision_required: false
strongest_live_verification:
  - python tools/analyze_demos.py
  - blender --background --python tools/check_demo_blends.py
files_that_need_correction: []
active_backlog_tasks: []
repair_checklist:
  - no demo-unit repair required
  - keep root policy-drift findings linked to solver/design reports
