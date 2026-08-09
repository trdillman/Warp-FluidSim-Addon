# TASK-REVIEW-019: CUDA cubin binary loaded from user-writable cache directory without integrity verification

## State

ready-for-agent

## Severity

medium

## What to build

_load_kernel() constructs a cubin cache path under wp.config.kernel_cache_dir / 'fluidSim_cuda_reduction' (line 62) and loads the binary with build.load_cuda(str(cubin_path), device) on line 88. The only integrity check is a filename containing a SHA256 hash of the source identity (line 56-61). The cubin file itself is not cryptographically authenticated. An attacker with write access to the Warp kernel cache directory (which is user-writable by design) can replace the cubin with a modified binary that executes arbitrary GPU-side code when launched. The cubin is loaded without verifying it matches the source hash beyond the filename.

**Location:** `fluidSim_warp_blender51/sim/cuda_reduction.py:62`
**Dimension:** Trust & Safety

## Impact

An attacker with write access to the kernel cache directory can substitute a malicious cubin that runs arbitrary code on the GPU at kernel launch time, potentially corrupting simulation data or crashing the GPU context.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/sim/cuda_reduction.py:62` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: After building the cubin, compute a hash of the binary content and embed it in the filename or a sidecar metadata file. Verify the hash after loading and before use. Alternatively, store the expected cubin hash in the _MODULES cache and reject mismatched cubins.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SEC-002` in `reviews/swarm-2026-08-08/findings/security.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Trust & Safety lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
