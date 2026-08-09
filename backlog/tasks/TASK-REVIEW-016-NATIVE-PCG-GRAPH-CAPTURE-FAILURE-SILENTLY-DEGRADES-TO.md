# TASK-REVIEW-016: Native PCG graph-capture failure silently degrades to host-batched with no diagnostic surface

## State

ready-for-agent

## Severity

medium

## What to build

At lines 2099-2105, when `_get_pcg_conditional_graphs` throws during CUDA graph capture, the exception is caught and stored in `self.last_pcg_native_error`, and `self._pcg_native_available` is permanently set to `False`. The solver then silently falls back to the host-batched PCG loop. No warning, log, or event is emitted to the UI or profiler. The user sees a significant performance drop (host-batched mode requires per-batch host readbacks) with no indication of why. The `last_pcg_native_error` string is only accessible programmatically; nothing in `step_frame` or `_pressure` checks or surfaces it.

**Location:** `fluidSim_warp_blender51/sim/pressure.py:2099`
**Dimension:** Execution Engine

## Impact

A transient CUDA driver error during graph capture permanently disables the native path and causes silent, ongoing performance degradation. Users cannot diagnose the cause without inspecting internal solver state.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/sim/pressure.py:2099` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: After the except block, emit a warning via the profiler or a solver stats flag (e.g., `self.stats.pcg_native_degraded = True`) that the UI/diagnostics panel can surface. Consider logging the error message once rather than silently storing it.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `EXEC-005` in `reviews/swarm-2026-08-08/findings/exec.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Execution Engine lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
