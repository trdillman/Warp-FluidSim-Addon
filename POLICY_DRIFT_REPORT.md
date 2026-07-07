---
status: completed
escalated: false
generated_at: 2026-07-07T21:55:00Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port
branch: codex/execute-worktree-recommendations
---

# Policy Drift Report

## Scope

Audit unit: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port`

Primary executable surface: `fluidSim_warp_blender51` Blender 5.1 add-on and its root production, acceptance, demo-analysis, and demo-blend lanes.

Nested output unit: `demos` has its own report at `demos/POLICY_DRIFT_REPORT.md`.

## Current Integration State

This branch integrated worktree recommendations after backing up `master` at `87971a7` and preserving `UNMERGED_WORKTREES_REPORT.md`.

Imported surfaces:

- Quick Setup domain/source/collider layout from the dirty detached cherry-pick worktree; `review_2635080.blend` was not imported.
- Slowdown lane behavior was already present through the existing `b5787a0` ancestor and master hardening; no duplicate empty slowdown commit is needed.
- Audit-corrections code/docs/tests were ported without committing generated demo `.blend` or preview PNG artifacts.

## Strongest Current Evidence

- `python -m pytest -q` passed: 113 tests.
- Direct viewport gate passed in Blender 5.1 on Warp 1.12.1 / CUDA `cuda:0`: `blender --background --factory-startup --python tests\blender_viewport_2frame_gate.py` ended in `VIEWPORT-2FRAME-GATE-PASS`.
- Aggregate headless production gate passed: `python tools/run_production_gate.py` ended in `HEADLESS-GATE-PASS`.
- The production gate explicitly did not run GUI acceptance; GUI-inclusive proof still requires `--include-gui` or `FLUIDSIM_FULL_GUI_ACCEPTANCE=1`.

## Findings

### PD-001 - Sparse-octree claim is implemented as a flat sparse leaf table

Type: documentation drift

Status: completed for current source wording

Authoritative evidence:

- Product, docs, UI, and validation wording use the sparse tiled leaf-grid contract.
- The runtime contract remains fixed 8x8x8 leaf tiles addressed through a flat leaf table, without hierarchical tree storage.
- Contract wording guards exist in focused tests.

Required corrective action:

- Keep future product/docs/UI wording on "sparse tiled leaf grid" unless a real hierarchical sparse octree implementation and validation suite are added.

### PD-002 - Active-work scaling repair

Type: implementation drift

Status: completed for current integration branch

Authoritative evidence:

- Particle and whitewater launch/compaction paths use active-count bounds where available.
- Pressure and viscosity grid phases use active voxel/leaf bounds for launches, reductions, and active copies where available.
- Custom snippet dispatch launches over active particle/voxel work dimensions.
- Focused regressions cover active-work bounds for pressure, viscosity, snippets, and compaction signatures.

Required corrective action:

- Keep future benchmark/report wording tied to tested evidence.
- Run GUI-inclusive production proof only when GUI acceptance is required.

### PD-003 - Viewport gate failure must fail closed

Type: validation drift

Status: completed for current integration branch

Authoritative evidence:

- `tools/run_production_gate.py` rejects Blender traceback/fail markers even when Blender exits 0.
- `tests/blender_viewport_2frame_gate.py` catches unexpected exceptions and emits `VIEWPORT-2FRAME-GATE-FAIL` before exiting nonzero.
- The production gate now has explicit headless versus GUI-inclusive proof semantics; GUI proof requires `--include-gui` or `FLUIDSIM_FULL_GUI_ACCEPTANCE=1`.

Required corrective action:

- Keep default headless output distinct from GUI-inclusive full production proof.
- Do not claim GUI acceptance unless the GUI lane is explicitly run.

## Machine-Readable Summary

generated_at: 2026-07-07T21:55:00Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port
branch: codex/execute-worktree-recommendations
freshness: integration-branch
status: completed
blocking_criticality: low
human_decision_required: false
strongest_live_verification:
  - python -m pytest -q (113 passed)
  - blender --background --factory-startup --python tests\blender_viewport_2frame_gate.py (VIEWPORT-2FRAME-GATE-PASS)
  - python tools/run_production_gate.py (HEADLESS-GATE-PASS; GUI not run)
files_that_need_correction: []
active_backlog_tasks:
  - none
repair_checklist:
  - master backup preserved before integration
  - generated binary demo artifacts excluded from commits
  - focused, broad pytest, direct viewport, and headless production gates passed
