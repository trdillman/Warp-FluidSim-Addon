# TASK-POLICY-DRIFT-012: Repair GUI acceptance out-of-domain world box

## State

completed

## What to build

Repair the failing GUI playback world-box contract at the shared scene/runtime
boundary or fixture. Preserve the original actionable `ValueError`; the
acceptance path must not turn a failed step into a secondary
`None.solver.frame` crash.

## Classification

implementation drift and validation drift; high blocking criticality

## Acceptance criteria

- [x] `blender --factory-startup --python tests/blender_gui_acceptance.py` reports `GUI-ACCEPTANCE-PASS (35/35)`.
- [x] `rtk python tools/run_production_gate.py --include-gui` reports `FULL-PRODUCTION-GATE-PASS`.
- [x] The out-of-domain world-box error remains actionable; the path neither silently clamps it nor replaces it with the secondary `None.solver.frame` exception.
- [x] A regression covers the failing GUI playback step and preserves the primary `ValueError` diagnostic.

## Evidence

- `blender --factory-startup --python tests/blender_gui_acceptance.py` passed
  `GUI-ACCEPTANCE-PASS (35/35)` after moving the UI-only sink/force fixtures
  inside the documented domain.
- `rtk python tools/run_production_gate.py --include-gui` passed
  `FULL-PRODUCTION-GATE-PASS` at HEAD `f9c7b366baaa31d171ff3b6773f39eb0125d110c`.
- `rtk python -m pytest -q tests/test_scene_bridge_cache_playback.py tests/test_operator_poll_and_guards.py`
  passed `33`; the playback-init regression asserts the exact world-box
  `ValueError`, preserves it in the scene error, and leaves `sess.solver` as
  `None` instead of dereferencing `None.solver.frame`.

## Blocked by

None - acceptance criteria met; no remaining blocker
