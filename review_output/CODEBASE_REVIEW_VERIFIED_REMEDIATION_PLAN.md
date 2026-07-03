# Verified Review Remediation Plan

Source review: `C:\Users\Tyler\.codex\attachments\9f95b5d0-12f4-474d-8441-e969c244663f\pasted-text.txt`

Repo: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port`

Created: 2026-07-03

## Problem Statement

The `fluidSim_warp_blender51` add-on has a small set of verified review findings that can affect simulation correctness, IO durability, Blender UX, and maintainability. The risk is correcting from stale generated claims instead of current live evidence. The desired state is a corrected add-on where each still-open VCR finding has a focused regression gate, stale findings remain marked stale, and completion is claimed only after targeted tests, Blender acceptance, and independent review agree.

## Initial Verification Summary

Initial live worktree was dirty and had to be preserved while verifying the pasted review claims.

Checks performed before implementation:

- `python -m pytest -q` -> `7 passed in 2.29s`
- `python -m compileall -q fluidSim_warp_blender51 tests tools` -> passed
- `python tests\test_solver_validation.py` -> `SOLVER-VALIDATION-PASS`, including:
  - `[PASS] height-field sourcing flat/profile behavior`
  - `[PASS] NanoVDB/OpenVDB volume sourcing kernels`

Independent read-only reviews were run against the attached review document:

- Security/serialization/trust reviewer: confirmed VCR-001, VCR-004, VCR-005, VCR-007, VCR-009, VCR-010, VCR-011; marked VCR-002, VCR-003, VCR-006 stale/fixed.
- Simulation/Blender reviewer: same core result; additionally confirmed height and volume validation coverage exists in `fluidSim_warp_blender51/sim/validation.py` and is reached by `tests/test_solver_validation.py`.

## Execution Result

Executed on 2026-07-03 with parallel implementer subagents and parent-side verification.

Commits created during remediation:

- `fb7e09c fix: map scene multigrid to pressure setting`
- `5bcc69d fix: harden IO cache durability and path policy`

Additional lane changes were implemented in the live worktree before final integration:

- `fluidSim_warp_blender51/blender/scene_bridge.py`
- `fluidSim_warp_blender51/operators.py`
- `fluidSim_warp_blender51/panels.py`
- `tests/blender_gui_acceptance.py`
- `tests/test_operator_panel_behavior.py`
- `tests/test_scene_bridge_collider_cache.py`

Parent-side verification after implementation:

- `python -m pytest -q tests/test_scene_bridge_collider_cache.py tests/test_scene_bridge_settings.py tests/test_operator_panel_behavior.py` -> `8 passed`
- `python -m pytest -q` -> `25 passed`
- `python tests\test_solver_validation.py` -> `SOLVER-VALIDATION-PASS`, including height-field and NanoVDB/OpenVDB volume-source checks
- `python -m compileall -q fluidSim_warp_blender51 tests tools` -> passed
- `git diff --check -- ...` for remediation files -> passed, with only Git LF/CRLF warnings
- `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe --background --factory-startup --python tests\blender_background_acceptance.py` -> `BACKGROUND-ACCEPTANCE-PASS`
- `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe --factory-startup --python tests\blender_gui_acceptance.py` -> `GUI-ACCEPTANCE-PASS (35/35)`

Independent post-implementation reviews:

- Logic/regression reviewer: no blocking issues; reran focused remediation tests, full pytest, and compileall.
- Plan-conformance reviewer: VCR-001, VCR-004, VCR-005/006/011, VCR-007, VCR-008, VCR-009, and VCR-010 all passed. The reviewer did not run Blender GUI locally, but parent-side Blender GUI acceptance did run and passed.

Residual honesty notes:

- `CLAUDE.md` remains modified as unrelated pre-existing/user work and was intentionally not touched.
- IO cache writes now use same-directory temp files and atomic replacement. They do not add explicit file or directory `fsync`, so crash/power-loss durability beyond atomic replace is not claimed.
- VCR-010 was corrected for configured-device validation routing. Modal/background progress for long validation or benchmark execution remains a possible future UX improvement, not part of this completed remediation.

## Claim Status

| ID | Status | Evidence |
| --- | --- | --- |
| VCR-001 Pressure multigrid UI setting ignored | Fixed and tested | `build_settings()` maps scene `multigrid` to `SolverSettings.pressure_multigrid`; covered by `tests/test_scene_bridge_settings.py` and pressure projection regression tests. |
| VCR-002 Height-source kernel coverage | Stale/fixed in dirty worktree | `python tests\test_solver_validation.py` passed `height-field sourcing flat/profile behavior`; implementation exists in `fluidSim_warp_blender51/sim/validation.py`. |
| VCR-003 NanoVDB/OpenVDB volume-source coverage | Stale/fixed in dirty worktree | `python tests\test_solver_validation.py` passed `NanoVDB/OpenVDB volume sourcing kernels`; implementation exists in `fluidSim_warp_blender51/sim/validation.py`. |
| VCR-004 Collider mesh descriptors rebuilt each gather | Fixed and tested | Session-local collider mesh cache reuses unchanged signatures and rebuilds on geometry/topology/transform/device changes; covered by `tests/test_scene_bridge_collider_cache.py`. |
| VCR-005 Cache files lack schema marker and atomic writes | Fixed and tested | Cache files now include `format_version=1` and write through same-directory temp files with atomic replacement; covered by `tests/test_cache_io.py`. |
| VCR-006 Explicit `np.load(..., allow_pickle=False)` | Fixed in dirty worktree | `fluidSim_warp_blender51/io/cache.py:49` uses `allow_pickle=False`; old RCE claim remains rejected for verified runtimes. |
| VCR-007 Frame-change exception surfacing weak | Fixed and tested | Scene/session error state persists, panel displays the error, and `fluidsim.clear_error` clears it; covered by `tests/test_operator_panel_behavior.py` and GUI acceptance. |
| VCR-008 Panel draw assertions | Fixed and tested | GUI acceptance now includes state-specific panel assertions, including persistent error display and clear action; Blender GUI acceptance passed `35/35`. |
| VCR-009 Operator undo metadata and export invoke | Fixed and tested | Export operator now invokes Blender file selector; `UNDO` is limited to mutating operators; covered by `tests/test_operator_panel_behavior.py`. |
| VCR-010 Validation/benchmark synchronous UI path | Partly fixed, scoped honestly | Validation now uses configured add-on device instead of hardcoded `cuda:0`; modal/background progress remains deferred as a future UX improvement. |
| VCR-011 Export/report parent path handling | Fixed and tested | Export/report/benchmark writers preflight missing parent paths with clear failures; covered by `tests/test_io_path_policy.py`. |

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
