# TASK-REVIEW-002: Make render_init handler persistent and gate all handler lists

## State

completed

## What to build

`render._on_render_init` was registered without `@persistent`, so Blender freed it on every file load. After opening any `.blend` the add-on stayed enabled but `ensure_render_visibility()` never ran again, letting source/collider helper objects reach final renders and output objects keep stale `hide_render` state.

The existing `tests/blender_handler_persistence_gate.py` was supposed to catch exactly this, but filtered to `_on_frame_change` only. Widen it to cover every add-on handler so the whole class is locked, not just the one instance.

## Acceptance criteria

- [x] `fluidSim_warp_blender51/blender/render.py` decorates `_on_render_init` with `@persistent`, using the same defensive `getattr(handlers, "persistent", lambda fn: fn)` lookup as `scene_bridge.py` so the non-Blender unit lane still imports.
- [x] `tests/blender_handler_persistence_gate.py` enumerates every add-on-owned callback across all handler lists it registers into, not just `frame_change_post`.
- [x] The gate asserts every add-on handler carries `_bpy_persistent`, both after register and after reopening a saved `.blend`.
- [x] The gate asserts `render_init` is actually registered, so the check cannot silently pass by finding nothing.
- [x] The gate asserts no handler list populated before the reopen is empty afterwards.
- [x] `blender --background --factory-startup --python tests/blender_handler_persistence_gate.py` prints `HANDLER-PERSISTENCE-GATE-PASS`.
- [x] Negative test proves the widened gate fails closed when the decorator is absent.

## Evidence

- Original defect, live: `render_init` handlers before opening `demos/dam_break.blend` = `[["_on_render_init", false]]`, after = `[]`, while `@persistent`-decorated `frame_change_post` handlers survived and the add-on stayed enabled (`artifacts/live_review/probe2.json`, key `F1_render_init_after_file_load`).
- After fix: `HANDLER-PERSISTENCE-GATE-PASS`.
- Negative test (decorator temporarily removed): `HANDLER-PERSISTENCE-GATE-FAIL: [after register] add-on app handlers missing @persistent: ['render_init._on_render_init']. Blender frees non-persistent handlers on file load.`
- Regression lanes: host pytest `261 passed`; `BACKGROUND-ACCEPTANCE-PASS`; `VIEWPORT-2FRAME-GATE-PASS`; `CACHE-EXPORT-PIPELINE-PASS`; `MESH-SOURCE-ACCEPTANCE-PASS`.
- Source finding: `reviews/ADDON_REVIEW.md` (HIGH, CONFIRMED) and `reviews/CORRECTION_PLAN.md` §0.

## Notes

Root cause of the escape was the gate's own narrow filter, not a missing gate. Prefer whole-class assertions over single-symbol assertions in runtime gates.

## Blocked by

None - completed
