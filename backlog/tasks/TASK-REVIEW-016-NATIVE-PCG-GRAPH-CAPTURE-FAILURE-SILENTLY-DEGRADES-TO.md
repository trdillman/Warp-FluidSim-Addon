# TASK-REVIEW-016: Native PCG graph-capture failure silently degrades to host-batched with no diagnostic surface

## State

completed

## Severity

medium

## What to build

At lines 2099-2105, when `_get_pcg_conditional_graphs` throws during CUDA graph capture, the exception is caught and stored in `self.last_pcg_native_error`, and `self._pcg_native_available` is permanently set to `False`. The solver then silently falls back to the host-batched PCG loop. No warning, log, or event is emitted to the UI or profiler. The user sees a significant performance drop (host-batched mode requires per-batch host readbacks) with no indication of why. The `last_pcg_native_error` string is only accessible programmatically; nothing in `step_frame` or `_pressure` checks or surfaces it.

**Location:** `fluidSim_warp_blender51/sim/pressure.py:2099`
**Dimension:** Execution Engine

## Impact

A transient CUDA driver error during graph capture permanently disables the native path and causes silent, ongoing performance degradation. Users cannot diagnose the cause without inspecting internal solver state.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/sim/pressure.py:2099` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: After the except block, emit a warning via the profiler or a solver stats flag (e.g., `self.stats.pcg_native_degraded = True`) that the UI/diagnostics panel can surface. Consider logging the error message once rather than silently storing it.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `EXEC-005` in `reviews/swarm-2026-08-08/findings/exec.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Execution Engine lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed for 2.0 (2026-09-27). A failed CUDA-graph capture is now shown to the user, not only stored.

- `FluidSolver._native_solver_message()` in `sim/solver.py` builds a message from `last_pcg_native_error` on the pressure and viscosity solvers, e.g. "Pressure solve fell back from CUDA graphs to a slower host loop: RuntimeError: ...".
- `_pressure()` and `_viscosity()` write it into the new `FrameStats.native_solver_message` (`"nativeSolverMessage"` in the stats dict).
- `scene_bridge` publishes it to the new read-only `stat_native_solver_message` property, and cache playback clears it.
- The Scene > FluidSim stats panel (`ui_scene.py`) draws it with an ERROR icon whenever it is non-empty.
- The property is excluded from pending-settings detection, like every other `stat_*` readout. The ownership map has been regenerated (226 fields).

Tests:
- `tests/test_pressure_st_and_graph_cache.py::test_native_pcg_capture_failure_is_reported_in_frame_stats` (CUDA): a refused capture falls back to host batches, and the message reaches the frame stats.
- `tests/test_viscosity.py::test_solver_records_viscosity_convergence_the_way_it_records_pressure` also checks the viscosity message.

New user-visible readout; flagged for owner review.

## Blocked by

None - can start immediately
