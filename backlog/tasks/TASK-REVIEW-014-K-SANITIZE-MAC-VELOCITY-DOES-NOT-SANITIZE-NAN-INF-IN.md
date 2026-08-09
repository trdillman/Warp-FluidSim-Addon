# TASK-REVIEW-014: k_sanitize_mac_velocity does not sanitize NaN/Inf in face weight arrays

## State

ready-for-agent

## Severity

medium

## What to build

In `k_sanitize_mac_velocity` (lines 252-298), when a velocity value is NaN or Inf, the kernel zeroes the velocity (`u[tid] = 0.0`) and then reduces the weight only if `wu[tid] > 1.0e-8`. If `wu[tid]` is itself NaN, the comparison `wu[tid] > 1.0e-8` evaluates to False (NaN comparisons always return False), so the NaN weight is left unchanged. Similarly, if only the weight is NaN but the velocity is finite and large, the `elif wu[tid] > 1.0e-8` branch is skipped and the velocity is neither clamped nor the weight fixed. A NaN face weight poisons the MAC normalization kernel (`k_normalize_mac`) in the next substep, which divides by weight sums.

**Location:** `fluidSim_warp_blender51/sim/solver.py:268`
**Dimension:** Execution Engine

## Impact

A single NaN face weight persists through sanitization and propagates NaN into the MAC velocity field on the next P2G/normalize cycle, potentially destabilizing the simulation after a pressure-projection blowup that the sanitize step was designed to contain.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/sim/solver.py:268` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Add explicit NaN/Inf checks for the weight arrays in the kernel. When `wu[tid]` is NaN or Inf, set it to 0.0 (or a small epsilon). Apply the same check to `wv[tid]` and `ww[tid]`.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `EXEC-003` in `reviews/swarm-2026-08-08/findings/exec.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Execution Engine lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
