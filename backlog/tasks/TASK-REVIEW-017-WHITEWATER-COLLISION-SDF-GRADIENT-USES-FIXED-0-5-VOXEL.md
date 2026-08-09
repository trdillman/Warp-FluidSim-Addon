# TASK-REVIEW-017: Whitewater collision SDF gradient uses fixed 0.5-voxel forward differences that produce garbage normals at sparse-grid boundaries

## State

ready-for-agent

## Severity

medium

## What to build

In k_advect_whitewater, the collision response computes a surface normal via forward differences (lines 378-383): `gx = sample_scalar_trilinear(collision_sdf, ..., cvp + wp.vec3(0.5, 0.0, 0.0), 1.0e6) - csdf`. The offset is exactly 0.5 voxels in voxel-space coordinates. When the sample point (cvp + offset) falls outside the sparse topology, sample_scalar_trilinear returns its fallback value of 1.0e6. This makes the gradient component approximately 1e6 - csdf, which dominates the normal direction and forces it toward a fixed diagonal regardless of actual collider geometry. The normal magnitude check at line 385 (`wp.length(n) > 1.0e-8`) passes because the garbage gradient is huge, so the incorrect normal is used for the push-out `p = p - n * csdf`.

**Location:** `fluidSim_warp_blender51/sim/whitewater.py:378`
**Dimension:** Particle Pipeline

## Impact

Whitewater particles colliding near the edge of the sparse topology boundary (where topology_expand is 0 or insufficient to cover the 0.5-voxel sample offset) receive collision responses in arbitrary diagonal directions instead of along the true collider surface normal. This causes visible deflection artifacts and incorrect bounce trajectories for foam, spray, and bubble particles near collider edges.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/sim/whitewater.py:378` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Use centered differences with a smaller offset (e.g., 1 voxel) and check whether each sample's flat index is valid (>= 0) before incorporating it into the gradient. Fall back to a zero gradient (skip collision response) if any sample is out-of-grid, rather than using the 1.0e6 fallback in the gradient computation.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `PIPE-002` in `reviews/swarm-2026-08-08/findings/pipeline.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Particle Pipeline lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
