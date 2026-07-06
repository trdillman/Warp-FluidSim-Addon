---
status: in-progress
escalated: false
generated_at: 2026-07-05T16:22:25Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port
git_head: 27b3024
---

# Policy Drift Report

## Scope

Audit unit: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port`

Primary executable surface: `fluidSim_warp_blender51` Blender 5.1 add-on and its root validation, acceptance, demo-analysis, and demo-blend lanes.

Nested output unit: `demos` has its own current report at `demos/POLICY_DRIFT_REPORT.md`.

## Strongest Current Evidence

- `python -m pytest -q` passed: 69 tests, including fixed-active pressure allocation regression.
- `blender --background --python tests/blender_background_acceptance.py` passed on Warp 1.14.0, CUDA Toolkit 12.9, Driver 13.3, `cuda:0` NVIDIA GeForce RTX 4080 SUPER.
- `blender --background --python tests/blender_attribute_viz_acceptance.py` passed on the same live CUDA/Warp device.
- `blender --python tests/blender_gui_acceptance.py` was rerun twice and failed the viewport screenshot visibility check both times: `GUI-ACCEPTANCE-FAIL (34/35)`, with live playback, visible surface object, render output, profiling, and benchmark operator checks passing.
- `python tools/analyze_demos.py` passed 30/30 against the current generated demo outputs and maelstrom metadata.
- `blender --background --python tools/check_demo_blends.py` passed for `dam_break.blend`, `crown_splash.blend`, `slosh_tank.blend`, and `maelstrom_basin.blend`.

These current lanes supersede stale review claims where source or tests have already changed. Confirmed stale review findings include cache pickle loading, pressure multigrid mapping, panel draw smoke, height-source coverage, and NanoVDB/OpenVDB validation coverage.

## Findings

### PD-001 - Sparse-octree claim is implemented as a flat sparse leaf table

Type: documentation drift

Status: in-progress; pressure/grid capacity repair implemented, GUI viewport-screenshot acceptance still failing

Authoritative evidence:

- `PROMPT.md` and `AGENTS.md` require sparse-octree fields.
- `fluidSim_warp_blender51/__init__.py`, `fluidSim_warp_blender51/docs/user_guide.md`, and `fluidSim_warp_blender51/properties.py` present the runtime as sparse-octree.
- `fluidSim_warp_blender51/sim/fields.py` implements a dense `leaf_table`, `leaf_flags`, and `leaf_coords` indexed by leaf coordinate, with fixed 8x8x8 leaf tiles and no parent/child hierarchy, adaptive levels, or tree traversal.
- `fluidSim_warp_blender51/docs/technical_report.md` calls this structure "octree-equivalent", which is materially weaker than an actual sparse octree and should not be promoted as equivalent without an explicit contract.

Why this is policy drift:

The live runtime is real CUDA/Warp and passes acceptance, but the promoted contract overstates the data structure semantics. A flat sparse leaf table can be a valid design, but it is not the same contract as a sparse octree under scale, resolution changes, or adaptive refinement expectations.

Required corrective action:

- Either implement and validate a real hierarchical sparse octree contract, or revise product/docs/UI wording to say "sparse tiled leaf grid" and explicitly list the non-octree semantics.
- Add a validation check that guards the chosen contract name: either tree hierarchy behavior if octree remains the claim, or no "octree" wording in product/docs/UI if the flat leaf-grid design is accepted.

Backlog task: `backlog/tasks/TASK-POLICY-DRIFT-001-SPARSE-OCTREE-CONTRACT.md`

### PD-002 - Core solver phases scale with allocation capacity, not active work

Type: implementation drift

Status: unresolved

Authoritative evidence:

- Current live lanes pass at the tested sizes, including GUI acceptance at 500,000 particle capacity and generated demos up to 259,112 particles.
- Particle and whitewater launch bounds now use host-side active-count mirrors where active counts are available.
- `fluidSim_warp_blender51/sim/pressure.py` now launches pressure voxel phases over `SparseGrid.active_voxel_count` and the coarse leaf pass over `SparseGrid.active_leaf_count`.
- Existing `review_output/CONSOLIDATED_CODEBASE_REVIEW.md` and JSON findings identified capacity-proportional particle compaction, binning, whitewater compaction, and grid-phase risks. The current runtime should be judged by focused active-work regressions, not stale review text alone.

Why this is policy drift:

The project promotes high-performance interactive GPU liquid simulation. Passing the current scenes is not enough proof that the design holds under reasonable increases in allocation capacity, sparse-domain size, or frame window. Any remaining allocation-wide scans and launches turn unused capacity into frame cost, so the semantics must be guarded by focused active-work scale regressions.

Required corrective action:

- Investigate or stabilize the GUI viewport screenshot lane before marking this report completed.
- Keep benchmark/report wording tied to the tested envelope unless the active-work regression and live acceptance lanes pass.

Backlog task: `backlog/tasks/TASK-POLICY-DRIFT-002-ACTIVE-WORK-SCALING.md`

## Stale Leads Checked

- `review_output/CONSOLIDATED_CODEBASE_REVIEW.md` still contains findings that are now stale against live source and acceptance tests. Do not reopen those from the review text alone.
- `demos/analysis_report.json` and `demos/ANALYSIS.md` were regenerated recently and passed on current analysis. They remain output evidence, not source-of-truth for solver design claims.

## Machine-Readable Summary

generated_at: 2026-07-05T16:22:25Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port
git_head: 27b3024
freshness: fresh
status: in-progress
blocking_criticality: medium
human_decision_required: false
strongest_live_verification:
  - python -m pytest -q
  - blender --background --python tests/blender_background_acceptance.py
  - blender --background --python tests/blender_attribute_viz_acceptance.py
  - blender --python tests/blender_gui_acceptance.py (fails viewport screenshot, 34/35)
  - python tools/analyze_demos.py
  - blender --background --python tools/check_demo_blends.py
files_that_need_correction:
  - fluidSim_warp_blender51/sim/pressure.py
  - fluidSim_warp_blender51/docs/technical_report.md
  - fluidSim_warp_blender51/docs/user_guide.md
  - fluidSim_warp_blender51/properties.py
  - fluidSim_warp_blender51/__init__.py
active_backlog_tasks:
  - backlog/tasks/TASK-POLICY-DRIFT-001-SPARSE-OCTREE-CONTRACT.md
  - backlog/tasks/TASK-POLICY-DRIFT-002-ACTIVE-WORK-SCALING.md
repair_checklist:
  - report is in-progress; original ordering criterion could not be truthfully satisfied because repair edits already existed in the working tree before this pass
  - settle sparse-octree versus sparse-tiled-grid contract
  - add contract guard for chosen wording
  - pressure voxel and coarse leaf phases now use active voxel/leaf bounds
  - allocation-scale regression added for fixed active pressure work
  - resolve repeated GUI viewport screenshot failure before completed status
  - rerun pytest, Blender acceptance, demo analysis, and demo blend checks
