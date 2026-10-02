# TASK-REVIEW-015: PCG conditional-graph cache uses blind full eviction instead of LRU

## State

completed

## Severity

medium

## What to build

At line 1559, `_get_pcg_conditional_graphs` checks `if len(self._pcg_conditional_graphs) >= 8: self._pcg_conditional_graphs.clear()`. This evicts ALL cached CUDA graphs when the cache reaches 8 entries, rather than evicting the least-recently-used entry. If the solver alternates between different `(work_n, batch, max_iterations, tol_sq)` keys (e.g., due to adaptive topology or user parameter changes), every capacity crossing forces re-capturing all graphs, each requiring a `wp.synchronize_device` call (line 1561). An OrderedDict with `popitem(last=False)` would evict one entry per miss.

**Location:** `fluidSim_warp_blender51/sim/pressure.py:1559`
**Dimension:** Execution Engine

## Impact

Repeated CUDA graph re-captures during normal operation when the active working set slightly exceeds 8 variants. This causes GPU stalls from the synchronize + capture cycle, degrading pressure-solve throughput.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/sim/pressure.py:1559` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Use `collections.OrderedDict` (already imported at line 33). On cache miss when at capacity, call `self._pcg_conditional_graphs.popitem(last=False)` to evict only the oldest entry. Move accessed entries to the end on cache hit with `move_to_end(key)`.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `EXEC-004` in `reviews/swarm-2026-08-08/findings/exec.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Execution Engine lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed on 2026-09-29. The flush was real, but it was not the main cost. The bigger cause was the cache key.

**Measurement before the fix.** On the `dam_break` scene, 42 frames made 371 pressure solves. They produced 39 distinct cache keys but only 3 distinct `(work_n, batch, max_iterations)` shapes. The key's fourth field, `tol_sq = tolerance² × active_voxel_count`, changes almost every solve, so no key was ever reused. LRU eviction alone would have saved nothing here: every solve still paid a `wp.synchronize_device` plus a capture of about 5 ms.

**Fix** (`sim/pressure.py`):
- **Tolerance read at run time.** The captured loop no longer has the tolerance baked in. `k_prepare_pcg_conditional_loop` already runs once per solve outside the graph. It now also writes `tol_sq` into a one-element device array, `_pcg_tol_sq`, and `k_finish_pcg_conditional_batch` reads that array. The cache key is now `(work_n, batch, max_iterations)`.
- **LRU instead of flush.** `_pcg_conditional_graphs` is an `OrderedDict`. A hit calls `move_to_end`. A miss at `_PCG_GRAPH_CAPACITY` (8) evicts one entry with `popitem(last=False)`. This matches `_launch_pcg_batch` in the same file and the viscosity solver.

**Result** on `dam_break`, 42 frames, same machine, Warp 1.17:

| | Graph captures | Time spent capturing |
|---|---|---|
| Before | 35 | 158 ms |
| After | 3 | 38 ms |

Particle positions and velocities hash the same before and after, every frame (sha256 `fcbe9584…`). The tolerance value is the same float32, now read from memory instead of from a kernel argument, so the solve is unchanged.

**Tests** (in `tests/test_pressure_st_and_graph_cache.py`, CUDA). Each was seen failing before its fix.
- `test_pcg_conditional_graph_cache_evicts_lru_instead_of_flushing`: a set of frequently used solve shapes plus one new shape per solve. It expects 18 captures for 18 distinct keys, and before the fix got 48. It also checks the cache stays full at 8.
- `test_pcg_conditional_graph_reads_tolerance_at_launch`: loose, then tight, then loose tolerance on one solver. The tight solve must take more iterations, the second loose solve must match the first, and no new capture is allowed. Before the fix it made 1 capture.

Full suite with the 2.0 environment (Warp 1.17): 1649 passed, 3 skipped. The focused pressure, viscosity and CUDA-graph suites: 96 passed.

**Not changed:** the viscosity solver's graph key (`viscosity.py` `_graph_key`) also contains `tol_sq`. It also contains `n` and `dt`, though, so taking the tolerance out would not by itself make keys repeat. That is a separate question.

## Blocked by

None - can start immediately
