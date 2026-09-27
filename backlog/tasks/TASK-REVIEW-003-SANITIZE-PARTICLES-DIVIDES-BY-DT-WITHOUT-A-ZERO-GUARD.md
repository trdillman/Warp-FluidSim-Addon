# TASK-REVIEW-003: _sanitize_particles divides by dt without a zero-guard, crashing when time_scale=0

## State

completed

## Severity

critical

## What to build

At line 1439, `_sanitize_particles` computes `max_speed = max(s.cfl * g.voxel_size / dt * 2.0, 50.0)` with an unguarded division by `dt`. The companion method `_cfl_speed_limit` (line 1386) explicitly guards this case with `safe_dt = max(float(dt), 1.0e-8)`, but `_sanitize_particles` does not use that guard. When `s.time_scale = 0` (a valid user setting that pauses the simulation), `frame_dt = s.time_scale / s.frame_rate` yields 0.0; then `dt = frame_dt / substeps` is also 0.0, and this line raises `ZeroDivisionError` on the host. There is no early-return for `frame_dt == 0` in `step_frame` before `_sanitize_particles(dt)` is called at line 1752.

**Location:** `fluidSim_warp_blender51/sim/solver.py:1439`
**Dimension:** Execution Engine

## Impact

Setting time_scale to 0 crashes the solver with an unhandled ZeroDivisionError instead of pausing gracefully. This is a hard crash on the Blender main thread.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/sim/solver.py:1439` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Either reuse `_cfl_speed_limit(dt)` to compute `max_speed` (which already has the `safe_dt` guard), or add `safe_dt = max(float(dt), 1.0e-8)` before the division: `max_speed = max(s.cfl * g.voxel_size / safe_dt * 2.0, 50.0)`. Additionally, consider an early-return in `step_frame` when `frame_dt <= 0`.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `EXEC-001` in `reviews/swarm-2026-08-08/findings/exec.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Critical Findings section).
- Identified by the Execution Engine lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed before 1.1.0. `Solver._sanitize_particles` in `fluidSim_warp_blender51/sim/solver.py` now computes `safe_dt = max(float(dt), 1.0e-8)` and divides by `safe_dt`, matching the guard in `_cfl_speed_limit`.

Test: `fluidSim_warp_blender51/tests/test_phase5_surface_tension_source.py::test_zero_time_scale_particle_sanitizer_uses_a_safe_denominator` (source-level check of the guard; no runtime time_scale=0 test).

Fixing commit: `bb6d3fc`.

## Blocked by

None - can start immediately
