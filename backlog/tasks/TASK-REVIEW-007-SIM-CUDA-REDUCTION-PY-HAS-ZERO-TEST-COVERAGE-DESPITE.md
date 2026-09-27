# TASK-REVIEW-007: sim/cuda_reduction.py has ZERO test coverage despite being used by the PCG pressure solver

## State

completed

## Severity

high

## What to build

This 175-line module performs CUDA JIT compilation via ctypes (build.build_cuda, build.load_cuda), manages a module-level mutable cache dict (_MODULES = {} at line 49), writes/reads .cu and .cubin files to disk (lines 62-100), and exposes CudaDotProduct.launch() which does raw pointer arithmetic via ctypes (lines 115-135). It is imported by pressure.py line 1487 ('from .cuda_reduction import CudaDotProduct') and instantiated at pressure.py:1489 ('self._pcg_dot = CudaDotProduct(self.device)'). No test file in tests/ imports or references cuda_reduction, CudaDotProduct, or _load_kernel. The error paths (compile failure at line 87-99, kernel load failure after 2 retries at line 101, launch failure at line 133) are entirely untested. The _MODULES global cache is never validated for stale entries or device lifecycle.

**Location:** `fluidSim_warp_blender51/sim/cuda_reduction.py:1`
**Dimension:** Test Coverage

## Impact

The PCG pressure solver's inner-product reduction -- a numerically critical operation in the conjugate gradient loop -- runs through this untested CUDA JIT path. A silent numerical error, stale cache entry, or file I/O failure would corrupt simulation results with no test to catch it.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/sim/cuda_reduction.py:1` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Add unit tests for _load_kernel (mock warp build.load_cuda to simulate compile failure, stale cubin, and successful load), CudaDotProduct.__init__ (non-CUDA device rejection, module caching behavior), and CudaDotProduct.launch (launch result code != 0 path). Add a numerical correctness test comparing against wp.utils.array_inner on known vectors.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TEST-001` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (High Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed before 1.1.0. `fluidSim_warp_blender51/sim/cuda_reduction.py` is now covered by `tests/test_cuda_reduction_abi.py` (10 tests: Warp 1.15/1.16 build and launch ABIs, module cache identity and reference leases, idempotent `CudaDotProduct` release, cache reset with active leases, release-failure retry, and kernel-lookup failure unloading the new module).

Numerical correctness of the PCG inner product is checked against `wp.utils.array_inner` in `tests/test_pressure_projection_regressions.py::test_pcg_persistent_inner_product_is_conditional_body_safe` (CUDA). Cache lifecycle is also exercised in `tests/test_pressure_st_and_graph_cache.py`.

Test file added in commit `bb6d3fc`.

## Blocked by

None - can start immediately
