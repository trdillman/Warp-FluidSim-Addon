# Reconcile scene-input source contract

## State

completed

## What to build

Reconcile the new source-emission property contract with the scene-input
translation tests so source and mesh-source gathering are validated through the
same settings shape used by the live Blender add-on. Preserve the real
`emission_mode` behavior and keep the production gate's coverage boundary
explicit.

## Acceptance criteria

- [x] Every scene-input test fixture that represents a source includes the
  required emission setting, or the runtime defines and tests an intentional
  compatibility default.
- [x] The nine failing scene-bridge tests pass without weakening their behavior
  assertions.
- [x] `python -m pytest -q` passes, and the focused scene-bridge test module is
  explicitly rerun.
- [x] `python tools/run_production_gate.py --include-gui` remains passing after
  the contract repair.

## Evidence

- `POLICY_DRIFT_REPORT.md:57-78,130-154` identifies this as the active source-contract finding and records the current evidence boundary.
- Current `python -m pytest -q` result: `244 passed, 9 failed` in
  `tests/test_scene_bridge_collider_cache.py`.
- Current failure seam: `fluidSim_warp_blender51/blender/scene_inputs.py:250`
  reads `pg.emission_mode` while the test `SimpleNamespace` fixtures omit it.
- The narrower production gate passes, so the repair must close the gap rather
  than merely rely on the production-gate result.
- `python -m pytest -q tests/test_scene_bridge_collider_cache.py`: `20 passed`.
- `python -m pytest -q`: `253 passed`.
- `python tools/run_production_gate.py --include-gui`: `FULL-PRODUCTION-GATE-PASS`,
  including `GUI-ACCEPTANCE-PASS (35/35)` and
  `SAVED-DEMO-GUI-PLAYBACK-PASS`.

## Blocked by

None - can start immediately
