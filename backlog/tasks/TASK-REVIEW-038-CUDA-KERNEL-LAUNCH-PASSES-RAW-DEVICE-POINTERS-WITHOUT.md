# TASK-REVIEW-038: CUDA kernel launch passes raw device pointers without array size validation

## State

ready-for-agent

## Severity

low

## What to build

CudaDotProduct.launch() (line 114) wraps a.ptr, b.ptr, and output.ptr as ctypes.c_void_p and passes them directly to wp_cuda_launch_kernel along with a count parameter. There is no validation that arrays a and b have at least 'count' elements, or that output has space for the result. The CUDA kernel iterates 'for (int i = index; i < count; i += stride)' reading a[i]*b[i], so a mismatched count causes out-of-bounds GPU memory reads. The output pointer receives an atomicAdd with no bounds check.

**Location:** `fluidSim_warp_blender51/sim/cuda_reduction.py:114`
**Dimension:** Trust & Safety

## Impact

If callers pass arrays smaller than count, the kernel reads or writes past the end of the GPU allocation, corrupting GPU memory or crashing the context. This is a safety gap in the FFI boundary, exploitable by any code path that miscalculates the count.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/sim/cuda_reduction.py:114` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Add assertions that len(a) >= count, len(b) >= count, and output has at least 1 element before launching. Alternatively, clamp count to min(len(a), len(b)) defensively.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SEC-005` in `reviews/swarm-2026-08-08/findings/security.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Trust & Safety lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
