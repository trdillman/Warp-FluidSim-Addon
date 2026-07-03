# Verified Review Remediation Plan

Source review: `C:\Users\Tyler\.codex\attachments\9f95b5d0-12f4-474d-8441-e969c244663f\pasted-text.txt`

Repo: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port`

Created: 2026-07-03

## Problem Statement

The `fluidSim_warp_blender51` add-on has a small set of verified review findings that can affect simulation correctness, IO durability, Blender UX, and maintainability. The risk is correcting from stale generated claims instead of current live evidence. The desired state is a corrected add-on where each still-open VCR finding has a focused regression gate, stale findings remain marked stale, and completion is claimed only after targeted tests, Blender acceptance, and independent review agree.

## Verification Summary

Current live worktree is dirty. Preserve existing user/previous-agent changes:

- Modified: `fluidSim_warp_blender51/io/cache.py`
- Modified: `fluidSim_warp_blender51/sim/validation.py`
- Modified: `tests/blender_gui_acceptance.py`
- Modified: `tests/test_solver_validation.py`
- Modified: `tests/test_validation.py`
- Untracked: `.codex/`, `review_output/`, `tests/test_cache_io.py`

Checks already performed for this plan:

- `python -m pytest -q` -> `7 passed in 2.29s`
- `python -m compileall -q fluidSim_warp_blender51 tests tools` -> passed
- `python tests\test_solver_validation.py` -> `SOLVER-VALIDATION-PASS`, including:
  - `[PASS] height-field sourcing flat/profile behavior`
  - `[PASS] NanoVDB/OpenVDB volume sourcing kernels`

Independent read-only reviews were run against the attached review document:

- Security/serialization/trust reviewer: confirmed VCR-001, VCR-004, VCR-005, VCR-007, VCR-009, VCR-010, VCR-011; marked VCR-002, VCR-003, VCR-006 stale/fixed.
- Simulation/Blender reviewer: same core result; additionally confirmed height and volume validation coverage exists in `fluidSim_warp_blender51/sim/validation.py` and is reached by `tests/test_solver_validation.py`.

## Claim Status

| ID | Status | Evidence |
| --- | --- | --- |
| VCR-001 Pressure multigrid UI setting ignored | Confirmed open | `fluidSim_warp_blender51/blender/scene_bridge.py:120` assigns `s.multigrid`; `SolverSettings` defines and solver reads `pressure_multigrid`. |
| VCR-002 Height-source kernel coverage | Stale/fixed in dirty worktree | `python tests\test_solver_validation.py` passed `height-field sourcing flat/profile behavior`; implementation exists in `fluidSim_warp_blender51/sim/validation.py`. |
| VCR-003 NanoVDB/OpenVDB volume-source coverage | Stale/fixed in dirty worktree | `python tests\test_solver_validation.py` passed `NanoVDB/OpenVDB volume sourcing kernels`; implementation exists in `fluidSim_warp_blender51/sim/validation.py`. |
| VCR-004 Collider mesh descriptors rebuilt each gather | Confirmed open | `fluidSim_warp_blender51/blender/scene_bridge.py` still constructs fresh `wp.Mesh` in `gather_colliders()`. |
| VCR-005 Cache files lack schema marker and atomic writes | Confirmed open | `fluidSim_warp_blender51/io/cache.py` writes direct `np.savez_compressed` with no `format_version` and no same-directory temp replace. |
| VCR-006 Explicit `np.load(..., allow_pickle=False)` | Fixed in dirty worktree | `fluidSim_warp_blender51/io/cache.py:49` uses `allow_pickle=False`; old RCE claim remains rejected for verified runtimes. |
| VCR-007 Frame-change exception surfacing weak | Confirmed open UX gap | Handler sets `sess.error`, `fs.status`, and prints, but no persistent panel-level error affordance/clear action is verified. |
| VCR-008 Panel draw assertions | Partly addressed | GUI acceptance has panel draw probes and injected failure mode; still lacks state-specific label/enabled assertions. |
| VCR-009 Operator undo metadata and export invoke | Confirmed open | Many operators lack appropriate `bl_options`; `FLUIDSIM_OT_export_frame` has `filepath`/`execute()` but no file-browser `invoke()`. |
| VCR-010 Validation/benchmark synchronous UI path | Confirmed open | `run_validation` and benchmark run inside `execute()`; validation hardcodes `cuda:0` instead of configured device. |
| VCR-011 Export/report parent path handling | Confirmed open | Export/report/benchmark writers open paths directly without parent creation or clear preflight failure. |

## Correction Plan

### Lane 1 - Correctness Regression First

Scope: VCR-001.

1. Change `build_settings()` to assign `s.pressure_multigrid = MultiGrid[fs.multigrid]`.
2. Add a focused regression test that creates or stubs a scene settings object with `multigrid != "OFF"`, calls `build_settings()`, and asserts:
   - `settings.pressure_multigrid` equals the requested enum value.
   - no downstream path relies on a dynamic `settings.multigrid` attribute.
3. Verification:
   - Focused mapping test.
   - `python -m pytest -q tests/test_pressure_projection.py tests/test_pressure_projection_regressions.py`
   - `python -m pytest -q`
4. Independent review:
   - Read-only reviewer checks the field-name contract from UI property -> `SolverSettings` -> pressure solve.

### Lane 2 - IO Durability

Scope: VCR-005, VCR-006, VCR-011.

1. Keep `np.load(..., allow_pickle=False)` explicit.
2. Add cache `format_version=1` metadata to new `.npz` writes.
3. Read path rejects unsupported or missing versions with an actionable error while preserving any intentional backward-compatibility decision.
4. Write cache frames through a same-directory temporary file and atomic `os.replace()` / `Path.replace()`.
5. Add parent-path policy for explicit user-selected export/report/benchmark output:
   - either create parent dirs where Blender UX expects creation;
   - or fail before write with a clear message naming the missing parent.
6. Verification:
   - `python -m pytest -q tests/test_cache_io.py`
   - Add/extend tests for cache version round-trip, unsupported version rejection, corrupt cache read, interrupted write behavior where practical, and missing parent path behavior.
   - `python -m pytest -q`
7. Independent review:
   - Security/IO reviewer checks no pickle regression, no partial final cache writes, and no confusing file failure paths.

### Lane 3 - Blender UX and Operator Behavior

Scope: VCR-007, VCR-008, VCR-009, VCR-010.

1. Add persistent, panel-visible simulation error state for frame-change failures, with a clear/reset path.
2. Extend panel tests beyond draw smoke:
   - uninitialized state;
   - initialized state;
   - stepped state;
   - validation result state;
   - benchmark result state;
   - error state.
3. Add `invoke()` for export file selection.
4. Add `UNDO` only to operators that mutate Blender scene or user-visible data in an undoable way. Do not blanket-add `UNDO`.
5. Route validation through the configured device setting instead of hardcoded `cuda:0`.
6. For long validation/benchmark work, prefer modal progress or a background/task pattern that reports status without freezing Blender. If that is too large for the immediate pass, split it behind a separate implementation step and keep the hardcoded-device fix first.
7. Verification:
   - `python -m pytest -q`
   - Blender background acceptance after registration/operator changes.
   - Blender GUI acceptance after panel/operator changes, including injected panel draw failure mode.
   - Manual or scripted proof that export opens a file selector and validation uses configured device.
8. Independent review:
   - Blender/UI reviewer checks operator semantics, panel-state assertions, and no overbroad undo behavior.

### Lane 4 - Collider Mesh Cache Design

Scope: VCR-004.

This is the only confirmed finding that should not be folded into the quick-fix lanes without a small design pass.

1. Define cache key and invalidation contract:
   - Blender object identity;
   - mesh topology revision or evaluated mesh identity;
   - world transform version;
   - explicit invalidation on object deletion/rebuild/release.
2. Implement a narrow collider mesh cache in the scene bridge or session state.
3. Add observability for rebuild versus reuse counts in tests or diagnostics.
4. Verification:
   - Unit-level or Blender-background test showing repeated gather without geometry/transform change reuses mesh.
   - Test showing geometry or transform change rebuilds.
   - Live playback smoke with moving collider if transform changes remain supported.
5. Independent review:
   - Performance/correctness reviewer checks stale collider geometry cannot persist after real Blender edits.

## Execution Order

1. Lane 1 first: smallest correctness bug with clear regression.
2. Lane 2 next: durable IO safety and cache compatibility.
3. Lane 3 next: UI/operator behavior, because it touches Blender acceptance.
4. Lane 4 last or separate: collider caching requires an explicit invalidation design.

Do not rework VCR-002, VCR-003, or VCR-006 except to keep their regression gates passing.

## Completion Gate

Do not call the remediation complete until all of these are true:

- Every open VCR item has either a code fix plus regression test, or a documented deferral with reason.
- `python -m pytest -q` passes.
- `python tests\test_solver_validation.py` passes and still reports height and volume validation passes.
- `python -m compileall -q fluidSim_warp_blender51 tests tools` passes.
- Blender background acceptance passes after registration/operator changes.
- Blender GUI acceptance passes after UI/operator/panel changes.
- Independent review confirms no stale generated-review claims were reintroduced as live problems.
- Final report separates fixed, stale, deferred, and verified-live claims.
