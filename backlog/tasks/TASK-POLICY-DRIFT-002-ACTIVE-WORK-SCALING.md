# TASK-POLICY-DRIFT-002: Bound solver work to active workload

## State

ready-for-agent

## What to build

Replace or prove the capacity-proportional hot paths in particle, whitewater, and sparse-grid phases so interactive performance scales primarily with active particles/leaves rather than configured allocation capacity.

## Acceptance criteria

- [ ] `POLICY_DRIFT_REPORT.md` is changed to `in-progress` before project files are edited.
- [ ] Particle compaction/binning/advection phases avoid unnecessary full-capacity work where active counts are available.
- [ ] Whitewater compaction/advection phases avoid unnecessary full-capacity work where active counts are available.
- [ ] Grid phases that must run over allocated voxel capacity are documented with a measured reason or narrowed to active leaves/voxels.
- [ ] A scale regression compares fixed active work under low and high allocation capacity and fails on disproportionate frame-time growth.
- [ ] Benchmark/report wording distinguishes tested envelope from robust scale proof until the regression passes.
- [ ] Relevant pytest, Blender acceptance, demo analysis, and demo blend checks pass.
- [ ] `POLICY_DRIFT_REPORT.md` is updated to `completed` only after verified repair.

## Blocked by

None - can start immediately
