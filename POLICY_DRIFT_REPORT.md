---
status: fresh
escalated: false
generated_at: 2026-07-03T21:18:03Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port
git_head: 91388c67699aac3c4c693a5dd6545e5951e7d3ce
---

# Policy Drift Report

## Scope

Audit unit: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port`

Primary executable surface: `fluidSim_warp_blender51` Blender 5.1 add-on and its root validation, acceptance, demo-analysis, and demo-blend lanes.

Nested output unit: `demos` has its own current report at `demos/POLICY_DRIFT_REPORT.md`.

## Strongest Current Evidence

- `python -m pytest -q` passed: 25 tests.
- `blender --background --python tests/blender_background_acceptance.py` passed on Warp 1.12.1, CUDA Toolkit 12.9, Driver 13.3, `cuda:0` NVIDIA GeForce RTX 4080 SUPER.
- `blender --background --python tests/blender_attribute_viz_acceptance.py` passed on the same live CUDA/Warp device.
- `blender --python tests/blender_gui_acceptance.py` passed 35/35 with interactive playback, visible viewport samples, render output, panel draw smoke, profiling, and benchmark operator proof.
- `python tools/analyze_demos.py` passed 25/25 against the current generated demo outputs.
- `blender --background --python tools/check_demo_blends.py` passed for `dam_break.blend`, `crown_splash.blend`, and `slosh_tank.blend`.

These current lanes supersede stale review claims where source or tests have already changed. Confirmed stale review findings include cache pickle loading, pressure multigrid mapping, panel draw smoke, height-source coverage, and NanoVDB/OpenVDB validation coverage.

## Findings

### PD-001 - Sparse-octree claim is implemented as a flat sparse leaf table

Type: documentation drift

Status: unresolved

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
- `fluidSim_warp_blender51/sim/particles.py` compacts particles by scanning `self.capacity`, then launches reset over `self.capacity`.
- `fluidSim_warp_blender51/sim/solver.py` launches multiple per-frame phases over `self.particles.capacity`, `self.whitewater.capacity`, or `g.voxel_capacity` even when active counts are smaller.
- Existing `review_output/CONSOLIDATED_CODEBASE_REVIEW.md` and JSON findings also identify capacity-proportional particle compaction, binning, and whitewater compaction risks. The current runtime did not contradict those scale risks; it only proved the current acceptance envelope passes.

Why this is policy drift:

The project promotes high-performance interactive GPU liquid simulation. Passing the current scenes is not enough proof that the design holds under reasonable increases in allocation capacity, sparse-domain size, or frame window. Allocation-wide scans and launches turn unused capacity into frame cost, so the semantics do not scale with the active sparse workload the docs advertise.

Required corrective action:

- Bound compaction, binning, advection, whitewater, and grid phases by active particle, active whitewater, or active leaf/voxel ranges where Warp kernels allow it.
- Add a scale regression that compares a fixed active particle/leaf count under low and high allocation capacity and fails on disproportionate frame-time growth.
- Update benchmark/report wording to distinguish current tested envelope from robust capacity-scale proof until that regression exists.

Backlog task: `backlog/tasks/TASK-POLICY-DRIFT-002-ACTIVE-WORK-SCALING.md`

## Stale Leads Checked

- `review_output/CONSOLIDATED_CODEBASE_REVIEW.md` still contains findings that are now stale against live source and acceptance tests. Do not reopen those from the review text alone.
- `demos/analysis_report.json` and `demos/ANALYSIS.md` were regenerated recently and passed on current analysis. They remain output evidence, not source-of-truth for solver design claims.

## Machine-Readable Summary

generated_at: 2026-07-03T21:18:03Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port
git_head: 91388c67699aac3c4c693a5dd6545e5951e7d3ce
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification:
  - python -m pytest -q
  - blender --background --python tests/blender_background_acceptance.py
  - blender --background --python tests/blender_attribute_viz_acceptance.py
  - blender --python tests/blender_gui_acceptance.py
  - python tools/analyze_demos.py
  - blender --background --python tools/check_demo_blends.py
files_that_need_correction:
  - fluidSim_warp_blender51/sim/fields.py
  - fluidSim_warp_blender51/sim/particles.py
  - fluidSim_warp_blender51/sim/solver.py
  - fluidSim_warp_blender51/sim/whitewater.py
  - fluidSim_warp_blender51/docs/technical_report.md
  - fluidSim_warp_blender51/docs/user_guide.md
  - fluidSim_warp_blender51/properties.py
  - fluidSim_warp_blender51/__init__.py
active_backlog_tasks:
  - backlog/tasks/TASK-POLICY-DRIFT-001-SPARSE-OCTREE-CONTRACT.md
  - backlog/tasks/TASK-POLICY-DRIFT-002-ACTIVE-WORK-SCALING.md
repair_checklist:
  - set report status to in-progress before project-file edits
  - settle sparse-octree versus sparse-tiled-grid contract
  - add contract guard for chosen wording
  - replace capacity-wide hot phases or prove bounded scale envelope
  - add allocation-scale regression
  - rerun pytest, Blender acceptance, demo analysis, and demo blend checks
