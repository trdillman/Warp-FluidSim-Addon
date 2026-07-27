# TASK-POLICY-DRIFT-010: Restore nested full-lane inputs

## State

ready-for-agent

## What to build

Make the committed nested worktree contain the scripts required by its own
production gate.

## Acceptance criteria

- [ ] The nested `python tools/run_production_gate.py --include-gui` reaches
  runtime lanes instead of failing during `py_compile`.
- [ ] The required lane scripts exist in this worktree and are covered by the
  same commit as the production runner.
- [ ] The nested report uses this worktree as `audit_unit` and records the
  current gate result.

## Evidence

- Nested report: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\.claude\worktrees\particle-surface-rendering-apis-f62356\POLICY_DRIFT_REPORT.md`.
- Nested worktree HEAD is `a2eaca7f45eb91368b082618e8759b028082ec87` and its
  status is clean.
- The current nested full gate exits `1` during `py_compile` because these
  worktree-local files are absent: `tests\blender_cache_export_pipeline.py`,
  `tests\blender_gui_acceptance.py`, and
  `tests\blender_viewport_2frame_gate.py`.
- Its lower-level `python -m pytest -q` passes `96`, but that does not override
  the lane-specific production-gate blocker.

## Blocked by

Missing committed lane scripts in this clean worktree; implementation can
start, but the production gate remains blocked until the files are restored.
