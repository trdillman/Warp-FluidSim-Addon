# TASK-POLICY-DRIFT-002: Bound solver work to active workload

## State

completed

## What to build

Replace or prove the capacity-proportional hot paths in particle, whitewater, and sparse-grid phases so interactive performance scales primarily with active particles/leaves rather than configured allocation capacity.

## Acceptance criteria

- [x] `POLICY_DRIFT_REPORT.md` is changed to `in-progress`. Note: the original "before project files are edited" ordering could not be truthfully satisfied because this pass started from an already-edited worktree.
- [x] Particle compaction/binning/advection phases avoid unnecessary full-capacity work where active counts are available.
- [x] Whitewater compaction/advection phases avoid unnecessary full-capacity work where active counts are available.
- [x] Pressure and viscosity grid phases are narrowed to active leaves/voxels where active counts are available; any remaining allocated-capacity grid phases need measured justification before being claimed as active-work proof.
- [x] Fixed-active regressions compare low and high allocation capacity for pressure and viscosity; viscosity also has deterministic launch/reduction/copy bound coverage.
- [x] Benchmark/report wording distinguishes tested envelope from robust scale proof until active-work regressions and trustworthy live gates pass.
- [x] Custom snippet dispatch launches over active particle/voxel work dimensions.
- [x] Liquid particle compaction has direct count-hint regression coverage proving inactive allocation tails are ignored.
- [x] A scale regression compares fixed active work under low and high allocation capacity and fails on disproportionate frame-time growth.
- [x] Benchmark/report wording distinguishes tested envelope from robust scale proof until active-work regressions and trustworthy live gates pass.
- [x] Relevant focused pytest and available gates pass in the integration branch: full `python -m pytest -q` passed 113 tests, direct viewport gate passed, and `python tools/run_production_gate.py` ended in `HEADLESS-GATE-PASS`.
- [x] `POLICY_DRIFT_REPORT.md` is updated truthfully after integration-branch verification.

## Reviewable artifact

Inspect `POLICY_DRIFT_REPORT.md` for the PD-002 closeout. Rerun `python tools/run_production_gate.py --include-gui` only when GUI-inclusive full production proof is required.

## Blocked by

None - can start immediately
