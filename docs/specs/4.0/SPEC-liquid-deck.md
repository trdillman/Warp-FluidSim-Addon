# SPEC — Liquid Deck (in-viewport interaction layer)

Owner module: new `blender/deck/` (`tokens.py`, `registry.py`, `bar.py`, `palette.py`,
`wheel.py`, `gizmos.py`, `modal.py`). Hooks: `blender/ui_header.py` (toggle + stage),
`blender/viewport_overlay.py` (shared draw registry), `preferences.py` (Deck on/off,
"Classic panels only"). Depends on: nothing in the solver. The Liquid Preview lane
shares the draw registry; land `registry.py` first, in whichever lane is ready.

## Why this shape

The ask is a modern, independent, not-seen-in-Blender interface. The constraints are:
Python cannot add a Space type; area pop-outs were declined 2026-10-02; every setting
has an owner and a Pending Settings class; panels are the accessibility and parity
surface the tests pin (`tests/ui/test_operator_panel_behavior.py`, 4956 lines). The
design that meets all four is a layer drawn *in the 3D viewport* with the `gpu` module,
input-routed by a modal operator and Blender gizmos, where each control is a thin
twin of an existing operator or property.

## Trap this spec prevents

1. **A parallel settings model.** Deck controls never hold state. A dial writes
   `obj.fluidsim.<prop>`; a button calls `bpy.ops.fluidsim.<op>`. Pending Settings
   classification, undo and file compatibility therefore stay unchanged. A
   `tests/ui/test_deck_parity.py` enumerates every Deck control and asserts its
   target operator/property exists and is the same one the panel draws.
2. **Modal operator that eats the viewport.** The Deck modal runs only while the
   pointer is over a Deck hit region (hit-tested each `MOUSEMOVE` in `PASS_THROUGH`
   mode); otherwise events pass. Esc never cancels a bake from the Deck (bake cancel is
   Esc-only *inside the bake modal*, decision D3); the Deck's cancel is a click outside.
3. **Draw-handler leaks across file load / add-on reload.** One registry owns every
   handler and gizmo group; `unregister()` removes all of them in a loop that tolerates
   failures (fixes the 3.0 "unguarded unregister loops" item). A load-post handler
   re-syncs Deck state from the scene, never the reverse.
4. **Theme hostility.** Colours derive from `bpy.context.preferences.themes[0]`
   (view3d background, text, widget accent) through `tokens.py`; one accent, one radius
   scale, one motion timing. Dark and light themes both pass a contrast check in tests.
5. **Gizmos that change simulation results silently.** Dragging an emitter velocity
   arrow writes the property through the same path as the panel; the pending-change
   chip appears exactly as it would from the panel. A test drags a gizmo in a headless
   `--background` session (gizmo `invoke/modal` called directly) and asserts the
   property value and the pending classification.
6. **Blocking the GUI gate.** Deck off must leave all existing UI tests byte-identical.
   The Deck lane in the GUI gate is separate and runs last.

## Components

- **Deck bar** (`bar.py`): bottom-anchored translucent card. Transport (Play/Pause via
  `screen.animation_play`, Step, Restart via `fluidsim.restart_simulation`), live readout
  (particles, substeps, ms/frame sparkline from the session stats already in
  `scene_bridge`), pending chip (reads `pending_settings` presentation: label, class,
  Apply/Revert), bake ring (reads `production_bake.active_bakes`).
- **Command palette** (`palette.py`): hotkey opens a centred search card; fuzzy match
  over registered FluidSim operators (bl_label, description), presets, demos, and
  domain/scene properties with their current value; Enter executes or edits inline via
  a numeric field drawn by the Deck and committed through the property.
- **Radial wheel** (`wheel.py`): custom-drawn radial with 6 slots per stage; the hub
  shows the stage (Setup/Simulate/Look/Bake) and clicking it cycles the stage header.
- **Gizmos** (`gizmos.py`, `bpy.types.GizmoGroup` per role):
  - Emitter: velocity arrow (drag = `emitter_velocity`), rate dial (`emission_rate`),
    shape outline.
  - Collider: friction/grip indicator, **Dynamic** toggle (4.0 rigid coupling).
  - Domain: six face handles; while dragging, draw the padded grid outline and the
    resulting cell count/`resolution_divisions`; commit on release.
- **Tokens** (`tokens.py`): single source of colours, radii, blur fallback (no
  backdrop blur on backends without it; use a 2-tap approximation), timing.

Settings: preference `deck_mode` ∈ {AUTO, ON, OFF} (AUTO = on for files saved by ≥ 4.0,
off for older files); scene flag `deck_shown`. No Deck state is saved per object.

## Acceptance

- `test_deck_parity.py`: every control maps to an existing operator/property; palette
  index equals the registered operator set; no control without a panel twin.
- `test_deck_gizmos_headless.py`: emitter arrow drag sets `emitter_velocity` and the
  pending chip reports the panel's classification; domain handle drag yields the same
  `resolution_divisions` as typing the value.
- `test_deck_lifecycle.py`: register/unregister 50× leaks no handlers or gizmo groups;
  load-post with a 3.2 file leaves Deck off under AUTO.
- GUI gate Deck lane: open crown_splash, toggle Deck, drag the velocity gizmo, confirm
  the chip, Apply, assert the applied value; artifact hash unchanged (UI only).
- Deck OFF: `tests/ui` byte-identical pass.
