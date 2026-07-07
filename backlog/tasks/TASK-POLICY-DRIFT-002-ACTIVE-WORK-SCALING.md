# TASK-POLICY-DRIFT-002: Bound solver work to active workload

## State

in-progress

## What to build

Replace or prove the capacity-proportional hot paths in particle, whitewater, and sparse-grid phases so interactive performance scales primarily with active particles/leaves rather than configured allocation capacity.

## Acceptance criteria

- [x] `POLICY_DRIFT_REPORT.md` is changed to `in-progress`. Note: the original "before project files are edited" ordering could not be truthfully satisfied because this pass started from an already-edited worktree.
- [x] Particle compaction/binning/advection phases avoid unnecessary full-capacity work where active counts are available.
- [x] Whitewater compaction/advection phases avoid unnecessary full-capacity work where active counts are available.
- [x] Pressure and viscosity grid phases are narrowed to active leaves/voxels where active counts are available; any remaining allocated-capacity grid phases need measured justification before being claimed as active-work proof.
- [x] Fixed-active regressions compare low and high allocation capacity for pressure and viscosity; viscosity also has deterministic launch/reduction/copy bound coverage.
- [x] Benchmark/report wording distinguishes tested envelope from robust scale proof until active-work regressions and trustworthy live gates pass.
- [ ] Relevant pytest, Blender acceptance, demo analysis, and demo blend checks pass under a trustworthy gate. Current exception: `blender --python tests/blender_gui_acceptance.py` now passes `GUI-ACCEPTANCE-PASS (35/35)`, but direct `blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py` throws `No module named 'warp'` while returning process exit code 0; the aggregate gate now fails closed on that traceback, and should be rerun after the viewport runtime/import repair.
- [ ] `POLICY_DRIFT_REPORT.md` is updated to `completed` only after verified repair.

## Blocked by

None - can start immediately
