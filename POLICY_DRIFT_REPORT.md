---
status: fresh
escalated: false
generated_at: 2026-07-07T02:18:22Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port
git_head: 1fc54e2d751d45fd9a3fa23c8cdb625c3749cd01
---

# Policy Drift Report

## Scope

Audit unit: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port`

Primary executable surface: `fluidSim_warp_blender51` Blender 5.1 add-on and its root production, acceptance, demo-analysis, and demo-blend lanes.

Nested output unit: `demos` has its own current report at `demos/POLICY_DRIFT_REPORT.md`.

## Strongest Current Evidence

- Pre-repair, `python tools/run_production_gate.py` printed `PRODUCTION-GATE-PASS` on current head after syntax checks, focused pytest, solver validation, kernel coverage, GPU profiler smoke, Blender background acceptance, attribute-viz acceptance, demo analysis, and demo blend checks; it did not run the foreground GUI acceptance lane as part of that aggregate pass.
- Post-repair, `tools/run_production_gate.py` includes the foreground GUI acceptance lane and rejects Blender-lane tracebacks or explicit fail markers before printing `PRODUCTION-GATE-PASS`.
- The same production-gate output contained a traceback from `tests/blender_viewport_2frame_gate.py`: `RuntimeError: Error: Solver initialization failed: No module named 'warp'`.
- Direct rerun of `blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py` reproduced the same `No module named 'warp'` failure, but PowerShell reported `EXIT:0`.
- Direct rerun of `blender --python tests/blender_gui_acceptance.py` passed: `GUI-ACCEPTANCE-PASS (35/35)`, including viewport screenshot, render output, live playback, panel draw, profiler, and benchmark checks.
- Current generated demo checks passed: `python tools/analyze_demos.py` reported `DEMO-ANALYSIS-PASS (30/30)`, and `blender --background --python tools/check_demo_blends.py` reported `DEMO-BLEND-CHECK-PASS`.

The current full-lane evidence supersedes the older report claim that the foreground GUI viewport screenshot was still failing. The remaining contradiction is in the direct viewport lane itself: it fails internally while Blender returns process exit code 0, but the aggregate production gate now rejects that traceback instead of reporting pass.

## Findings

### PD-003 - Viewport two-frame gate failure is masked by exit-code/pass-summary drift

Type: validation drift

Status: completed

Authoritative evidence:

- Pre-repair, `python tools/run_production_gate.py` reported `PRODUCTION-GATE-PASS` without running `tests/blender_gui_acceptance.py` and while its own output included a traceback from `tests/blender_viewport_2frame_gate.py`.
- Post-repair, `tools/run_production_gate.py` runs `tests/blender_gui_acceptance.py` and fails closed on the same viewport traceback instead of printing `PRODUCTION-GATE-PASS`.
- Direct command: `blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py` printed `RuntimeError: Error: Solver initialization failed: No module named 'warp'` and `EXIT:0`.
- `tests/blender_viewport_2frame_gate.py` calls `main()` before the final `if FAILURES: sys.exit(1)` block, so unexpected exceptions can bypass the script's explicit failure handling. In this environment Blender still returned process exit code 0 for that Python exception.

Why this is policy drift:

The project promotes a strict production gate for live viewport playback. The aggregate gate now fails closed on Blender tracebacks, but the direct viewport script still needs its own fail-closed wrapper and runtime/import repair. This is validation drift, not a solver-performance finding.

Required corrective action:

- Make `tests/blender_viewport_2frame_gate.py` fail closed on unexpected exceptions and emit an unambiguous failure marker.
- Rerun the direct viewport gate and the aggregate production gate after the viewport runtime/import fix.

Backlog task: `backlog/tasks/TASK-POLICY-DRIFT-003-VIEWPORT-GATE-FAIL-CLOSED.md`

### PD-001 - Sparse-octree claim is implemented as a flat sparse leaf table

Type: documentation drift

Status: in-progress

Authoritative evidence:

- `PROMPT.md`, `AGENTS.md`, `fluidSim_warp_blender51/__init__.py`, `fluidSim_warp_blender51/docs/user_guide.md`, `fluidSim_warp_blender51/docs/technical_report.md`, and validation labels now use the sparse tiled leaf-grid contract.
- `fluidSim_warp_blender51/properties.py` describes leaf capacity as "sparse tiled leaf grid", and `fluidSim_warp_blender51/sim/fields.py` documents sparse tiled leaf-grid storage.
- `fluidSim_warp_blender51/docs/technical_report.md` explicitly records fixed 8x8x8 leaf tiles, flat slot-major arrays, and no parent/child hierarchy or adaptive tree traversal.
- `tests/test_sparse_leaf_grid_contract.py` guards the live contract wording and validation registry name.

Why this is policy drift:

The live runtime is real CUDA/Warp and the current acceptance lanes pass. The promoted contract now matches the implemented sparse tiled leaf grid instead of overstating hierarchical tree semantics.

Required corrective action:

- Completed by revising product/docs/UI/test wording to say "sparse tiled leaf grid" and explicitly list the non-hierarchical semantics.
- Completed by adding a focused validation check that guards the chosen contract name.

Backlog task: `backlog/tasks/TASK-POLICY-DRIFT-001-SPARSE-OCTREE-CONTRACT.md`

### PD-002 - Active-work scaling repair is not fully closed by validation evidence

Type: implementation drift

Status: in-progress

Authoritative evidence:

- Current live lanes pass at the tested sizes, including `python tools/run_production_gate.py`, direct foreground GUI acceptance, demo analysis, and demo blend checks.
- Particle and whitewater launch bounds now use host-side active-count mirrors where active counts are available.
- `fluidSim_warp_blender51/sim/pressure.py` now launches pressure voxel phases over `SparseGrid.active_voxel_count` and the coarse leaf pass over `SparseGrid.active_leaf_count`.
- `tests/test_pressure_projection_regressions.py` includes a fixed-active pressure allocation regression comparing low and high leaf capacity.
- `fluidSim_warp_blender51/sim/viscosity.py` now bounds viscosity launches, reductions, and device copies to `SparseGrid.active_voxel_count`.
- `tests/test_viscosity_active_work_regressions.py` includes inactive-tail preservation, deterministic active-bound instrumentation, and a fixed-active viscosity timing regression.
- The current direct viewport two-frame gate failure means production-gate completion remains blocked until that lane is repaired, even though the aggregate gate now fails closed on the traceback.

Why this is policy drift:

The project promotes high-performance interactive GPU liquid simulation. Pressure and viscosity now have focused active-work regressions at the tested envelope, but the validation surface is not yet complete because the direct viewport two-frame lane still fails with `No module named 'warp'`.

Required corrective action:

- Fix PD-003 first so the full production gate fails closed.
- Rerun the fixed production gate, direct GUI acceptance, demo analysis, and demo blend checks.
- Keep benchmark/report wording tied to the tested envelope unless active-work regressions and live acceptance lanes pass under a trustworthy gate.

Backlog task: `backlog/tasks/TASK-POLICY-DRIFT-002-ACTIVE-WORK-SCALING.md`

## Stale Leads Checked

- The previous root report's `blender --python tests/blender_gui_acceptance.py` screenshot blocker is stale on current head; the current rerun passed `GUI-ACCEPTANCE-PASS (35/35)`.
- Older review-output findings remain leads only. Current source and runtime evidence show some prior capacity-scaling and panel-draw findings have already been changed.
- `demos/analysis_report.json` and `demos/ANALYSIS.md` are current generated-output evidence, not source-of-truth for solver design claims.

## Machine-Readable Summary

generated_at: 2026-07-07T02:18:22Z
audit_unit: C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port
git_head: 1fc54e2d751d45fd9a3fa23c8cdb625c3749cd01
freshness: fresh
status: fresh
blocking_criticality: high
human_decision_required: false
strongest_live_verification:
  - python tools/run_production_gate.py (post-repair fail-closed on viewport gate traceback)
  - blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py (No module named 'warp', EXIT:0)
  - blender --python tests/blender_gui_acceptance.py (GUI-ACCEPTANCE-PASS 35/35)
  - python tools/analyze_demos.py (DEMO-ANALYSIS-PASS 30/30)
  - blender --background --python tools/check_demo_blends.py (DEMO-BLEND-CHECK-PASS)
files_that_need_correction:
  - tests/blender_viewport_2frame_gate.py
  - tools/run_production_gate.py
  - fluidSim_warp_blender51/docs/technical_report.md
  - fluidSim_warp_blender51/docs/user_guide.md
  - fluidSim_warp_blender51/__init__.py
  - fluidSim_warp_blender51/sim/validation.py
active_backlog_tasks:
  - backlog/tasks/TASK-POLICY-DRIFT-001-SPARSE-OCTREE-CONTRACT.md
  - backlog/tasks/TASK-POLICY-DRIFT-002-ACTIVE-WORK-SCALING.md
  - backlog/tasks/TASK-POLICY-DRIFT-003-VIEWPORT-GATE-FAIL-CLOSED.md
repair_checklist:
  - fix viewport two-frame gate to fail closed on unexpected exceptions
  - settle sparse-octree versus sparse-tiled-grid contract
  - rerun production gate, direct viewport gate, GUI acceptance, demo analysis, and demo blend checks
