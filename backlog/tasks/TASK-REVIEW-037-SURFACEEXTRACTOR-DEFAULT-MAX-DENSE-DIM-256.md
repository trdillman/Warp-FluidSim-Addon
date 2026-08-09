# TASK-REVIEW-037: SurfaceExtractor default max_dense_dim=256 inconsistent with capacity planner and pipeline defaults of 512

## State

ready-for-agent

## Severity

low

## What to build

SurfaceExtractor.__init__ (line 718) defaults max_dense_dim to 256. However, capacity_planner.py _settings_dict (line 110) defaults surface_max_dense_dim to 512, and SurfacePipeline.ensure_resources (line 122) reads the setting with default 512. While ensure_resources updates the extractor's max_dense_dim after construction (line 125: `self.render_extractor.max_dense_dim = max_dense`), the initial constructor default of 256 means that any code path calling extract() before ensure_resources() would use a smaller limit than what the capacity planner estimated memory for. This creates a window where the capacity planner allows a configuration that the extractor would reject at runtime with 'dense span exceeds Surface Max Dense Dim'.

**Location:** `fluidSim_warp_blender51/sim/surface.py:718`
**Dimension:** Particle Pipeline

## Impact

Minimal in the current code because ensure_resources() is always called before extract(). The inconsistency is a latent maintenance hazard: if a future code path calls extract() without ensure_resources() first, it would use 256 instead of the capacity-planner-assumed 512, potentially raising a RuntimeError for fluid extents between 256 and 512 voxels that were budgeted as safe.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/sim/surface.py:718` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Align the SurfaceExtractor.__init__ default to 512 to match capacity_planner.py and SurfacePipeline.ensure_resources. Better yet, make the default a module-level constant imported from a shared location.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `PIPE-004` in `reviews/swarm-2026-08-08/findings/pipeline.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Particle Pipeline lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
