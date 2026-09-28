# TASK-REVIEW-036: k_ww_compact_scatter declares out_count as an output parameter but never writes to it

## State

completed

## Severity

low

## What to build

The kernel k_ww_compact_scatter (lines 680-701) lists `out_count: wp.array(dtype=wp.int32)` as a parameter and WhitewaterSystem.compact() passes `self._count2` as this output (line 652). However, the kernel body never assigns to `out_count`. The actual compacted count is written by a separate k_ww_final_count kernel launched afterward (lines 656-662) which overwrites self._count2. The dead output parameter creates a false signal that the scatter kernel owns the count, which could mask a regression if the k_ww_final_count launch is accidentally removed in future refactoring.

**Location:** `fluidSim_warp_blender51/sim/whitewater.py:692`
**Dimension:** Particle Pipeline

## Impact

No runtime impact currently because k_ww_final_count always runs after the scatter and overwrites the value. However, the misleading parameter declaration makes the code harder to maintain and could lead to a silent count-corruption bug if k_ww_final_count is removed without also removing the out_count parameter.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/sim/whitewater.py:692` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Remove the out_count parameter from k_ww_compact_scatter and remove self._count2 from its outputs list. The count is solely owned by k_ww_final_count.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Resolution

Fixed in `2a6eec0` on `claude/2.0-phase1-platform` (2.0 Phase 2, WS5 cleanup).

- `k_ww_compact_scatter` no longer declares `out_count`, and `compact()` no longer
  passes `self._count2` to it; `k_ww_final_count` alone writes the compacted count.

Regression gate: `tests/test_whitewater_kernel_outputs.py` parses `sim/whitewater.py`
and fails if any `@wp.kernel` declares an `out_*` array it never writes.

## Evidence

- Source finding: `PIPE-003` in `reviews/swarm-2026-08-08/findings/pipeline.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Particle Pipeline lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
