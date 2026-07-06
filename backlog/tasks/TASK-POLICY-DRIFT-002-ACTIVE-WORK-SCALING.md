# TASK-POLICY-DRIFT-002: Bound solver work to active workload

## State

in-progress

## What to build

Replace or prove the capacity-proportional hot paths in particle, whitewater, and sparse-grid phases so interactive performance scales primarily with active particles/leaves rather than configured allocation capacity.

## Acceptance criteria

- [x] `POLICY_DRIFT_REPORT.md` is changed to `in-progress`. Note: the original "before project files are edited" ordering could not be truthfully satisfied because this pass started from an already-edited worktree.
- [x] Particle compaction/binning/advection phases avoid unnecessary full-capacity work where active counts are available.
- [x] Whitewater compaction/advection phases avoid unnecessary full-capacity work where active counts are available.
- [x] Grid phases that must run over allocated voxel capacity are documented with a measured reason or narrowed to active leaves/voxels.
- [x] A scale regression compares fixed active work under low and high allocation capacity and fails on disproportionate frame-time growth.
- [x] Benchmark/report wording distinguishes tested envelope from robust scale proof until the regression passes.
- [ ] Relevant pytest, Blender acceptance, demo analysis, and demo blend checks pass. Current exception: `blender --python tests/blender_gui_acceptance.py` failed twice at viewport screenshot visibility (`GUI-ACCEPTANCE-FAIL (34/35)`); pytest, background Blender acceptance, attribute-viz acceptance, demo analysis, and demo blend checks pass.
- [ ] `POLICY_DRIFT_REPORT.md` is updated to `completed` only after verified repair.

## Blocked by

None - can start immediately
