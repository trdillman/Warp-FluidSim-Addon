# TASK-REVIEW-015: PCG conditional-graph cache uses blind full eviction instead of LRU

## State

ready-for-agent

## Severity

medium

## What to build

At line 1559, `_get_pcg_conditional_graphs` checks `if len(self._pcg_conditional_graphs) >= 8: self._pcg_conditional_graphs.clear()`. This evicts ALL cached CUDA graphs when the cache reaches 8 entries, rather than evicting the least-recently-used entry. If the solver alternates between different `(work_n, batch, max_iterations, tol_sq)` keys (e.g., due to adaptive topology or user parameter changes), every capacity crossing forces re-capturing all graphs, each requiring a `wp.synchronize_device` call (line 1561). An OrderedDict with `popitem(last=False)` would evict one entry per miss.

**Location:** `fluidSim_warp_blender51/sim/pressure.py:1559`
**Dimension:** Execution Engine

## Impact

Repeated CUDA graph re-captures during normal operation when the active working set slightly exceeds 8 variants. This causes GPU stalls from the synchronize + capture cycle, degrading pressure-solve throughput.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/sim/pressure.py:1559` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Use `collections.OrderedDict` (already imported at line 33). On cache miss when at capacity, call `self._pcg_conditional_graphs.popitem(last=False)` to evict only the oldest entry. Move accessed entries to the end on cache hit with `move_to_end(key)`.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `EXEC-004` in `reviews/swarm-2026-08-08/findings/exec.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Execution Engine lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
