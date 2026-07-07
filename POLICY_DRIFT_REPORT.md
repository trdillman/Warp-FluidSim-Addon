---
status: in-progress
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

This branch is integrating worktree recommendations after backing up `master` at `87971a7` and preserving `UNMERGED_WORKTREES_REPORT.md`.

Imported surfaces:

- Quick Setup domain/source/collider layout from the dirty detached cherry-pick worktree; `review_2635080.blend` was not imported.
- Slowdown lane behavior was already present through the existing `b5787a0` ancestor and master hardening; no duplicate empty slowdown commit is needed.
- Audit-corrections code/docs/tests are being ported without generated demo `.blend` or preview PNG artifacts.

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

Status: in-progress until this integration branch finishes verification

Authoritative evidence:

- Particle and whitewater launch/compaction paths use active-count bounds where available.
- Pressure and viscosity grid phases use active voxel/leaf bounds for launches, reductions, and active copies where available.
- Custom snippet dispatch launches over active particle/voxel work dimensions.
- Focused regressions cover active-work bounds for pressure, viscosity, snippets, and compaction signatures.

Required corrective action:

- Run focused integration tests for operator behavior, scene bridge cache playback, snippet dispatch, viscosity/pressure regressions, policy-drift gate semantics, and slowdown telemetry.
- Run the broad available pytest gate before declaring this branch ready.

### PD-003 - Viewport gate failure must fail closed

Type: validation drift

Status: in-progress until integration-branch verification

Authoritative evidence:

- `tools/run_production_gate.py` rejects Blender traceback/fail markers even when Blender exits 0.
- The production gate now has explicit headless versus GUI-inclusive proof semantics; GUI proof requires `--include-gui` or `FLUIDSIM_FULL_GUI_ACCEPTANCE=1`.

Required corrective action:

- Keep default headless output distinct from GUI-inclusive full production proof.
- Do not claim GUI acceptance unless the GUI lane is explicitly run.

## Machine-Readable Summary

generated_at: 2026-07-07T21:55:00Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port
branch: codex/execute-worktree-recommendations
freshness: integration-branch
status: in-progress
blocking_criticality: medium
human_decision_required: false
strongest_live_verification:
  - pending focused integration pytest
  - pending broad available pytest
files_that_need_correction: []
active_backlog_tasks:
  - backlog/tasks/TASK-POLICY-DRIFT-002-ACTIVE-WORK-SCALING.md
  - backlog/tasks/TASK-POLICY-DRIFT-003-VIEWPORT-GATE-FAIL-CLOSED.md
repair_checklist:
  - preserve master backup before integration
  - exclude generated binary demo artifacts unless separately reviewed
  - rerun focused and broad available tests before ready claim
