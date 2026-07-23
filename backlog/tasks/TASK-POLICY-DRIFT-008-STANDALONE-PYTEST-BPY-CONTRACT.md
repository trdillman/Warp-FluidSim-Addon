# Make standalone pytest honor the Blender-only viewport boundary

## State

completed

## What to build

Make the repository-wide Python regression command and the viewport-output test
agree on their runtime boundary. The test must either exercise the
Blender-dependent viewport module through the supported Blender lane or keep
its host-Python coverage isolated from the top-level `bpy` import, while
preserving the one-frame caching assertions.

## Acceptance criteria

- [x] `python -m pytest -q` completes without a `ModuleNotFoundError: No module named 'bpy'` collection/runtime failure.
- [x] `tests/test_frame_output.py::test_viewport_cache_and_export_share_one_frame` still proves that viewport, cache, and export share one host frame and does not weaken its assertions.
- [x] The Blender-dependent test boundary is explicit and consistent with `python tools/run_production_gate.py --include-gui`.
- [x] `python tools/run_production_gate.py --include-gui` still ends with `FULL-PRODUCTION-GATE-PASS`.

## Evidence

- Verified `python -m pytest -q`: `256 passed`.
- `tests/test_frame_output.py::test_viewport_cache_and_export_share_one_frame` retains its one-frame assertions.
- `fluidSim_warp_blender51/blender/viewport.py` loads `bpy` only for Blender-backed output and overlay entry points; host frame assembly imports without Blender.
- Verified `python tools/run_production_gate.py --include-gui`: `GUI-ACCEPTANCE-PASS (35/35)`, `SAVED-DEMO-GUI-PLAYBACK-PASS`, and `FULL-PRODUCTION-GATE-PASS`.

## Verification

- `python -m pytest -q`: `256 passed`.
- `python tools/run_production_gate.py --include-gui`: `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (35/35)`, and `SAVED-DEMO-GUI-PLAYBACK-PASS`.
- `tests/test_frame_output.py` includes a host-boundary regression that blocks `bpy` import while retaining the frame-sharing assertions.

## Blocked by

None - resolved
