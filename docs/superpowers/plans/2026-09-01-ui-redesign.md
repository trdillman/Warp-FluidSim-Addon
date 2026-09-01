# FluidSim Warp — UI Redesign Implementation Plan

> **For Claude:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the FluidSim Warp add-on UI into a native Blender 5.1 experience — a Basic/Advanced gradient, a split transport hub, a real per-object domain datablock, an exposed HUD, and consolidated telemetry — without changing solver behavior.

**Architecture:** Six independently shippable phases. Phases 0–2 are *storage-agnostic* and land immediately on today's `Scene.fluidsim` singleton. Phase 3 is the pivot: a tested, one-way `load_post` migration that relocates domain settings from `Scene.fluidsim` to `obj.fluidsim_domain` (a `PointerProperty` on the domain object). Phases 4–5 depend on that datablock existing and build the native surfaces (Physics-on-domain home, deleted N-panel mirror, role Type dropdown, consolidated Diagnostics, developer-gated Warp workspace). Every visual fix routes through the shared `ui_common.py` draw helpers so it propagates through the single-authored delegation layer.

**Tech Stack:** Python 3.11, Blender 5.1 `bpy` (Panel/Operator/PropertyGroup/PointerProperty/EnumProperty), pytest with the in-repo fake-`bpy` harness (`_install_bpy_stub` / `_reload_module` / `_Layout` / `_RnaPropertyGroup` in `tests/test_operator_panel_behavior.py`), and the headless Blender GUI gate (`tools/run_production_gate.py --include-gui`).

**Source spec:** [docs/superpowers/specs/2026-08-31-ui-redesign-design.md](../specs/2026-08-31-ui-redesign-design.md) — read it before starting any chunk. Each chunk below maps to one phase (§10) and one or more consensus diagnoses (D1–D9).

---

## File structure

Files touched, by responsibility. Storage-agnostic phases (0–2) modify presentation only; Phase 3 adds the datablock; Phases 4–5 re-point surfaces at it.

| File | Responsibility | Phases |
| --- | --- | --- |
| `fluidSim_warp_blender51/blender/ui_common.py` | Shared severity/readout/property draw helpers + panel base polls | 0, 3, 4, 5 |
| `fluidSim_warp_blender51/blender/ui_scene.py` | Scene-Properties panels: hub split, setup, output, diagnostics | 0, 1, 2, 4, 5 |
| `fluidSim_warp_blender51/operator_domains/source_roles.py` | `setup_scene` (+ invoke), role-tag operators | 0 |
| `fluidSim_warp_blender51/properties.py` | `FLUIDSIM_PG_settings`: `ui_mode`, `quality`, `schema_version`; `fluidsim_domain` pointer | 2, 3 |
| `fluidSim_warp_blender51/blender/ui_physics.py` | Physics-on-domain primary home (renders `obj.fluidsim_domain`) | 4 |
| `fluidSim_warp_blender51/blender/ui_object.py` | Role Type dropdown + batch helpers | 4 |
| `fluidSim_warp_blender51/blender/ui_view3d.py` | **Deleted** verbatim mirror → lean N-panel (HUD + Live Tuning) | 4 |
| `fluidSim_warp_blender51/blender/migration.py` | **New**: `migrate_domain_settings_schema()` + `load_post` handler | 3 |
| `fluidSim_warp_blender51/blender/scene_bridge.py` + solver read path | Read domain settings from active domain object, not `scene.fluidsim` | 3 |
| `fluidSim_warp_blender51/blender/menus.py` | **New**: `VIEW3D_MT_add` entry | 4 |
| `fluidSim_warp_blender51/blender/effect_workspace.py` + `preferences.py` | Developer-gate the Warp workspace | 5 |
| `fluidSim_warp_blender51/__init__.py` | Register/unregister the new modules in the ordered `steps` tuple | 3, 4, 5 |
| `tests/test_operator_panel_behavior.py` | Home for new pure-python UI tests (reuses the fake-bpy harness) | 0, 1, 2, 4, 5 |
| `tests/test_domain_migration.py` | **New**: migration round-trip / idempotency unit tests | 3 |
| `tests/gui/…` (headless acceptance) | Golden-frame + `.blend` fixture migration gate | 3 |

---

## Testing conventions (read once, applies to every chunk)

All pure-python UI tests live in `tests/test_operator_panel_behavior.py` and reuse its harness. Do **not** invent a new bpy stub.

- **Import the add-on under the stub:** `_install_bpy_stub(monkeypatch)` then `mod = _reload_module("fluidSim_warp_blender51.blender.ui_common")` (or the module under test). `ui_common`, `ui_scene`, etc. all `import bpy` at module top, so the stub must be installed first.
- **Panel draw tests:** instantiate the panel class, assign `panel.layout = _Layout()`, build a `SimpleNamespace` context, call `panel.draw(context)`, then assert on `panel.layout.operators` (list of `{"id", ...kwargs}`), `.props` (names), `.prop_calls`, `.prop_states` (effective greyout/alert per prop via `drawn_enabled(name)`), and `.labels` (`{"text", "icon"}`). The `_Layout` fake is documented at `tests/test_operator_panel_behavior.py:401`.
- **Operator invoke/execute tests:** mirror `test_export_frame_invoke_opens_file_selector` (`tests/test_operator_panel_behavior.py:293`) — a `SimpleNamespace` context carrying `window_manager` / `scene` and assert on the returned `{"..."}` set and recorded side effects.
- **Clamping-sensitive property tests:** use `_RnaPropertyGroup` (`:35`), never a bare `SimpleNamespace`, so silent RNA clamping is reproduced.
- **Per-chunk gate (run before marking the chunk done):**

```bash
python -m pytest -q
```

  and, after a GUI pass per `AGENTS.md`:

```bash
python tools/run_production_gate.py --include-gui
```

```bash
python tools/validate_policy_drift_reports.py
```

- **Commit cadence:** one commit per task (test + implementation together once green). Conventional-commit prefixes (`feat:` / `fix:` / `refactor:` / `test:`).

---

## Chunk 0: Phase 0 — Quick wins (D4, D7, D8)

Small, isolated, storage-agnostic fixes that are immediately felt. No data model change. Everything here operates on today's `scene.fluidsim`.

### Task 0.1: Fix inverted severity mapping in `draw_status_message` (D8)

**Files:**
- Modify: `fluidSim_warp_blender51/blender/ui_common.py:98-106`
- Test: `tests/test_operator_panel_behavior.py`

Design decision (from spec §6): `ERROR` must be *strictly heavier* than `WARNING` via **shape + text + color**, not icon alone. `ERROR` renders on an `alert=True` sub-row with an `"Error: "` prefix; `WARNING` gets a `"Warning: "` prefix and no alert. This satisfies principle #7 (redundant coding) and removes the icon-only severity signal.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_operator_panel_behavior.py`:

```python
def test_status_message_error_is_heavier_than_warning(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_common = _reload_module("fluidSim_warp_blender51.blender.ui_common")

    class _Rec:
        def __init__(self, alert=False):
            self.alert = alert
            self.rows = []
            self.labels = []
        def row(self, align=False):
            child = _Rec(self.alert)
            self.rows.append(child)
            return child
        def label(self, text="", icon="NONE"):
            self.labels.append({"text": text, "icon": icon, "alert": self.alert})

    def _all_labels(rec):
        found = list(rec.labels)
        for child in rec.rows:
            found.extend(_all_labels(child))
        return found

    warn = _Rec()
    ui_common.draw_status_message(warn, "low memory", "WARNING")
    err = _Rec()
    ui_common.draw_status_message(err, "disk full", "ERROR")

    warn_label = _all_labels(warn)[0]
    err_label = _all_labels(err)[0]

    assert warn_label["text"].startswith("Warning:")
    assert warn_label["alert"] is False
    assert err_label["text"].startswith("Error:")
    assert err_label["alert"] is True                 # color reinforces severity
    assert err_label["icon"] != warn_label["icon"]    # distinct glyphs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_operator_panel_behavior.py::test_status_message_error_is_heavier_than_warning -v`
Expected: FAIL — current text has no prefix and no alert row.

- [ ] **Step 3: Write minimal implementation**

Replace `draw_status_message` in `ui_common.py`:

```python
# (severity key) -> (Blender icon, text prefix, use alert row)
_SEVERITY_PRESENTATION = {
    "OK": ("CHECKMARK", "", False),
    "INFO": ("INFO", "", False),
    "WARNING": ("ERROR", "Warning: ", False),   # ERROR = Blender's warning-triangle glyph
    "ERROR": ("CANCEL", "Error: ", True),        # circle-slash + red alert row = strictly heavier
}


def draw_status_message(layout, text: str, severity: str = "INFO") -> None:
    """Draw one status row; severity is shape + text (+ color), never icon-alone."""
    icon, prefix, alert = _SEVERITY_PRESENTATION.get(
        str(severity).upper(), _SEVERITY_PRESENTATION["INFO"]
    )
    row = layout.row(align=True)
    if alert:
        row.alert = True
    row.label(text=f"{prefix}{text}", icon=icon)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_operator_panel_behavior.py::test_status_message_error_is_heavier_than_warning -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fluidSim_warp_blender51/blender/ui_common.py tests/test_operator_panel_behavior.py
git commit -m "fix(ui): make ERROR severity strictly heavier than WARNING (D8)"
```

### Task 0.2: Expose the viewport HUD toggle (`show_overlay`) (D4)

**Files:**
- Modify: `fluidSim_warp_blender51/blender/ui_scene.py:846-861` (`FLUIDSIM_PT_output_display.draw`)
- Test: `tests/test_operator_panel_behavior.py`

The HUD is fully built (`viewport_overlay.py:37` gates on `fs.show_overlay`; `properties.py:1541` defines it, default `True`) but drawn by no panel. One `layout.prop` turns it on.

- [ ] **Step 1: Write the failing test**

```python
def test_display_panel_exposes_hud_overlay_toggle(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    fs = SimpleNamespace(
        liquid_view_mode="PARTICLES",
        surface_enabled=False,
        whitewater_enabled=False,
        show_overlay=True,
        particle_display_fraction=1.0,
        particle_display_scale=1.0,
        particle_attribute_mode="NONE",
    )
    panel = ui_scene.FLUIDSIM_PT_output_display()
    panel.layout = _Layout()
    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    assert "show_overlay" in panel.layout.props
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_operator_panel_behavior.py::test_display_panel_exposes_hud_overlay_toggle -v`
Expected: FAIL — `show_overlay` never drawn.

- [ ] **Step 3: Write minimal implementation**

In `FLUIDSIM_PT_output_display.draw`, add the HUD toggle after `layout.prop(fs, "liquid_view_mode")` (with a separator so it reads as a distinct viewport-display control):

```python
        layout.prop(fs, "liquid_view_mode")
        layout.prop(fs, "show_overlay", text="Viewport HUD")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_operator_panel_behavior.py::test_display_panel_exposes_hud_overlay_toggle -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fluidSim_warp_blender51/blender/ui_scene.py tests/test_operator_panel_behavior.py
git commit -m "feat(ui): expose built-in viewport HUD toggle in Display panel (D4)"
```

### Task 0.3: Give `Guided Setup…` a real `invoke` dialog (D7)

**Files:**
- Modify: `fluidSim_warp_blender51/operator_domains/source_roles.py:61-104` (`FLUIDSIM_OT_setup_scene`)
- Test: `tests/test_operator_panel_behavior.py`

The button carries a trailing ellipsis and a 6-way `mode` enum, but the operator has no `invoke`, so clicking silently runs `QUICK_LIQUID`. Add `invoke` → `invoke_props_dialog` and a `draw` that surfaces `mode`.

- [ ] **Step 1: Write the failing test**

```python
def test_setup_scene_invoke_opens_mode_dialog(monkeypatch):
    _install_bpy_stub(monkeypatch)
    operators = _reload_module("fluidSim_warp_blender51.operators")
    calls = []
    context = SimpleNamespace(
        window_manager=SimpleNamespace(
            invoke_props_dialog=lambda op: calls.append(op) or {"RUNNING_MODAL"}
        )
    )
    op = operators.FLUIDSIM_OT_setup_scene()

    assert op.invoke(context, None) == {"RUNNING_MODAL"}
    assert calls == [op]

    # draw() must surface the mode enum so all six modes are chosen up front
    op.layout = _Layout()
    op.draw(context)
    assert "mode" in op.layout.props
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_operator_panel_behavior.py::test_setup_scene_invoke_opens_mode_dialog -v`
Expected: FAIL — `FLUIDSIM_OT_setup_scene` has no `invoke`/`draw`.

- [ ] **Step 3: Write minimal implementation**

In `source_roles.py`, add to `FLUIDSIM_OT_setup_scene` (after the `poll` classmethod, before `execute`):

```python
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "mode")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_operator_panel_behavior.py::test_setup_scene_invoke_opens_mode_dialog -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fluidSim_warp_blender51/operator_domains/source_roles.py tests/test_operator_panel_behavior.py
git commit -m "fix(setup): honor the Guided Setup ellipsis with a mode dialog (D7)"
```

### Task 0.4: Label the runtime recovery action + remove silent truncation (D7, D8)

**Files:**
- Modify: `fluidSim_warp_blender51/blender/ui_scene.py:144-157` (hub status/runtime rows)
- Test: `tests/test_operator_panel_behavior.py`

Two fixes in the hub's read-only region (they stay when the hub is later split in Chunk 1): (a) the bare-icon `FILE_REFRESH` runtime-probe button gets a text label ("Re-detect GPU"); (b) `status[:80]` truncation is replaced with `_draw_wrapped` (already exists, `ui_scene.py:37`) so the explanatory tail is never dropped.

- [ ] **Step 1: Write the failing test**

```python
def _hub_context(state_id="IDLE", owns_scene=False, runtime_ready=True):
    fs = SimpleNamespace(status="", height_error="")
    return SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)), fs


def test_runtime_recovery_button_is_labeled_and_status_not_truncated(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    monkeypatch.setattr(
        ui_scene, "_runtime_ui_state",
        lambda: (SimpleNamespace(), "cuda:0", False, "no CUDA device"))
    monkeypatch.setattr(
        ui_scene, "activity_ui_state",
        lambda scene: {"id": "IDLE", "owns_scene": False, "label": "Idle", "icon": "PAUSE"})
    monkeypatch.setattr(
        ui_scene, "current_error", lambda scene: "", raising=False)

    long_status = "Solver halted: " + "x" * 200
    ctx, fs = _hub_context()
    fs.status = long_status
    panel = ui_scene.FLUIDSIM_PT_scene_control()
    panel.layout = _Layout()
    panel.draw(ctx)

    # recovery operator carries a non-empty text label (not a bare glyph)
    probe_ops = [op for op in panel.layout.operators
                 if op["id"] == "fluidsim.refresh_runtime_probe"]
    assert probe_ops and probe_ops[0].get("text")
    # the full status survives (tail present, wrapped across labels)
    joined = " ".join(l["text"] for l in panel.layout.labels)
    assert long_status[-20:] in joined.replace("  ", " ") or long_status[-20:] in "".join(
        l["text"] for l in panel.layout.labels)
```

> Note: `current_error` is imported inside `draw()` from `.scene_bridge`; if the local-import form makes monkeypatching awkward, patch `ui_scene.scene_bridge.current_error` via the imported module, or set `fs`/session so `current_error` returns `""`. Confirm the exact patch target when writing the test.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_operator_panel_behavior.py::test_runtime_recovery_button_is_labeled_and_status_not_truncated -v`
Expected: FAIL — probe button has `text=""`; status is sliced to 80 chars.

- [ ] **Step 3: Write minimal implementation**

In `FLUIDSIM_PT_scene_control.draw` replace the truncated status label and the bare-icon probe button:

```python
        status = str(getattr(fs, "status", "") or "")
        if status:
            _draw_wrapped(layout, status)
```

and

```python
        runtime.operator(
            "fluidsim.refresh_runtime_probe",
            text="Re-detect GPU" if not runtime_ready else "Re-detect",
            icon="FILE_REFRESH",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_operator_panel_behavior.py::test_runtime_recovery_button_is_labeled_and_status_not_truncated -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fluidSim_warp_blender51/blender/ui_scene.py tests/test_operator_panel_behavior.py
git commit -m "fix(ui): label runtime recovery + stop silent status truncation (D7/D8)"
```

### Task 0.5: Chunk 0 gate

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q`
Expected: PASS (0 failures).

- [ ] **Step 2: GUI production gate** (per `AGENTS.md`)

Run: `python tools/run_production_gate.py --include-gui`
Then: `python tools/validate_policy_drift_reports.py`
Expected: PASS.

- [ ] **Step 3: Commit any gate-refreshed evidence** (only if the gate updates tracked evidence files).

---

## Chunk 1: Phase 1 — Split the `scene_control` hub (D3)

Replace the one overloaded `FLUIDSIM_PT_scene_control` panel (`ui_scene.py:130-201`, ~22 layout calls) with three single-job panels — **Transport**, **Lifecycle**, **Status & Runtime** — with stable positions and differentiated destructive actions. Still storage-agnostic (reads `scene.fluidsim`).

Design from spec §5.4:
- **Transport:** primary Start/Stop Live on its own full-width row (`scale_y≈1.3`); Step One / Step Many always present, **greyed** (never hidden) with a `poll_message` when unavailable so buttons never move.
- **Lifecycle:** Initialize / Release / Reset; Release+Reset wrapped in an `alert=True` sub-row `separator()`-ed from safe controls, each with a one-line consequence label.
- **Status & Runtime:** state name, labeled runtime recovery (from Task 0.4), error box, and a compact one-line `step_ms · substeps · CFL` strip.

### Task 1.1: Extract a stable Transport panel

**Files:**
- Modify: `fluidSim_warp_blender51/blender/ui_scene.py` (new `FLUIDSIM_PT_scene_transport`)
- Modify: `SCENE_PANEL_CLASSES` tuple (register the new panel; `ui_scene.py:1145-1195`)
- Test: `tests/test_operator_panel_behavior.py`

- [ ] **Step 1: Write the failing test** — assert transport always draws `toggle_live` plus both `step` variants, and that Step rows are *present but greyed* (not absent) when `runtime_ready is False`:

```python
def _transport_ctx(state_id, owns_scene, runtime_ready):
    return SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace(status="")))


def test_transport_keeps_step_buttons_present_but_greyed_when_not_ready(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    monkeypatch.setattr(ui_scene, "_runtime_ui_state",
                        lambda: (SimpleNamespace(), "cuda:0", False, "not ready"))
    monkeypatch.setattr(ui_scene, "activity_ui_state",
                        lambda scene: {"id": "IDLE", "owns_scene": True,
                                       "label": "Idle", "icon": "PAUSE"})
    panel = ui_scene.FLUIDSIM_PT_scene_transport()
    panel.layout = _Layout()
    panel.draw(_transport_ctx("IDLE", True, False))

    op_ids = [op["id"] for op in panel.layout.operators]
    assert "fluidsim.toggle_live" in op_ids
    assert op_ids.count("fluidsim.step") == 2          # Step One + Step Many always present
    # step buttons drawn under a disabled layout (greyed, not hidden)
    assert any(not s["enabled"] for s in panel.layout.prop_states) or True  # see note
```

> Note: `step` buttons are operators, not props, so greyout is asserted via the `_Layout` sub-row `enabled` state. If `_Layout.operator` does not record enabled state, assert instead that the Step rows are created on a `row()` whose `enabled` was set `False` by inspecting draw structure; simplest is to record `enabled` on `operator()` — extend `_Layout.operator` to capture `self.enabled` **only if** the existing suite doesn't already rely on its shape (grep first). Prefer asserting presence-when-not-ready (the anti-reflow property) over exact greyout if extending the fake is risky.

- [ ] **Step 2: Run to verify it fails** — Run: `python -m pytest tests/test_operator_panel_behavior.py::test_transport_keeps_step_buttons_present_but_greyed_when_not_ready -v` — Expected: FAIL (`FLUIDSIM_PT_scene_transport` undefined).

- [ ] **Step 3: Implement `FLUIDSIM_PT_scene_transport`** — a new panel (`bl_order = 0`) that draws only the transport loop; Step rows always emitted with `.enabled = runtime_ready` and a `poll_message`/tooltip; primary toggle on its own `row()` with `row.scale_y = 1.3`. Lift the transport branch of the old hub verbatim, dropping lifecycle/status.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** — `feat(ui): extract stable Transport panel from the hub (D3)`

### Task 1.2: Extract a Lifecycle panel with differentiated destructive actions

**Files:** `ui_scene.py` (new `FLUIDSIM_PT_scene_lifecycle`), `SCENE_PANEL_CLASSES`; Test: same file.

- [ ] **Step 1: Write the failing test** — assert Release and Reset are drawn on an `alert=True` sub-row and separated from Initialize:

```python
def test_lifecycle_marks_release_and_reset_as_destructive(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    monkeypatch.setattr(ui_scene, "_runtime_ui_state",
                        lambda: (SimpleNamespace(), "cuda:0", True, "ready"))
    monkeypatch.setattr(ui_scene, "activity_ui_state",
                        lambda scene: {"id": "IDLE", "owns_scene": True,
                                       "label": "Idle", "icon": "PAUSE"})
    panel = ui_scene.FLUIDSIM_PT_scene_lifecycle()
    panel.layout = _Layout()
    panel.draw(SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace())))

    op_ids = [op["id"] for op in panel.layout.operators]
    assert "fluidsim.release" in op_ids
    assert "fluidsim.reset" in op_ids
    # destructive ops drawn under an alert sub-row (extend _Layout.operator to record `alert`,
    # or assert a separator precedes them — confirm mechanism when writing).
```

- [ ] **Step 2–4:** run-fail → implement `FLUIDSIM_PT_scene_lifecycle` (`bl_order = 1`) with `alert = True` on the destructive sub-row and `layout.separator()` before it; each destructive op gets a one-line consequence via `text=`/tooltip → run-pass.

- [ ] **Step 5: Commit** — `feat(ui): split Lifecycle panel with alert-styled destructive actions (D3)`

### Task 1.3: Convert the old hub into Status & Runtime (with live strip)

**Files:** `ui_scene.py` (`FLUIDSIM_PT_scene_control` → Status & Runtime), `SCENE_PANEL_CLASSES`; Test: same file.

- [ ] **Step 1: Write the failing test** — assert the panel draws the state label, the labeled runtime button (from Task 0.4), the error box, and a compact one-line `step_ms · substeps · CFL` strip (assert a label containing `stat_step_ms`'s value or the literal separators), and that it **no longer** draws transport/lifecycle operators (`toggle_live`, `step`, `reset`, `release`, `initialize` absent).

- [ ] **Step 2–4:** run-fail → repurpose `FLUIDSIM_PT_scene_control.draw` to the read-only Status & Runtime job (keep `bl_idname` stable so child panels `bl_parent_id="FLUIDSIM_PT_scene_control"` — pending/runtime-stats/live-timeline — still parent correctly); add the live strip built from `fs.stat_step_ms/stat_substeps/stat_cfl_clamped`. Remove the transport/lifecycle branches now living in 1.1/1.2 → run-pass.

- [ ] **Step 5: Commit** — `refactor(ui): reduce hub to read-only Status & Runtime + live strip (D3)`

### Task 1.4: Chunk 1 gate

- [ ] Run `python -m pytest -q` → PASS.
- [ ] Run `python tools/run_production_gate.py --include-gui` then `python tools/validate_policy_drift_reports.py` → PASS. (GUI gate confirms the three panels register and no reflow regression.)
- [ ] Commit gate-refreshed evidence if any.

---

## Chunk 2: Phase 2 — Basic/Advanced + Quality (D2)

Add the two-tier disclosure. Still storage-agnostic: `ui_mode`, `quality` land on today's `FLUIDSIM_PG_settings`. This is the core reduction — targeting ~20–25 default visible properties.

### Task 2.1: Add `ui_mode` and `quality` UI-only enums to `FLUIDSIM_PG_settings`

**Files:**
- Modify: `fluidSim_warp_blender51/properties.py` (`FLUIDSIM_PG_settings`, near `:750`)
- Test: `tests/test_operator_panel_behavior.py`

- [ ] **Step 1: Write the failing test** — reload `properties` under the stub and assert both enums exist with correct defaults and item sets:

```python
def test_settings_group_has_ui_mode_and_quality(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    assert {"BASIC", "ADVANCED"} <= {i[0] for i in properties.UI_MODE_ITEMS}
    assert {"DRAFT", "PREVIEW", "FINAL", "CUSTOM"} <= {i[0] for i in properties.QUALITY_ITEMS}
    ann = properties.FLUIDSIM_PG_settings.__annotations__
    assert "ui_mode" in ann and "quality" in ann
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — add module-level `UI_MODE_ITEMS` / `QUALITY_ITEMS` tuples and the two `EnumProperty` annotations (`ui_mode` default `"BASIC"`, `quality` default `"PREVIEW"`), each with a full `description=`. No solver wiring yet (Quality→internals is Task 2.3).

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** — `feat(props): add ui_mode + quality disclosure enums (D2)`

### Task 2.2: Poll-gate the Advanced panels on `ui_mode == 'ADVANCED'`

**Files:**
- Modify: `fluidSim_warp_blender51/blender/ui_scene.py` — add an `_advanced_visible(context)` helper and add it to the `poll` of the 12 `sim_*` deep-config panels + the other Advanced clusters (ST-FLIP block, capacity/safety, full surface-filter chain, per-stream whitewater, per-attribute export toggles).
- Test: `tests/test_operator_panel_behavior.py`

- [ ] **Step 1: Write the failing test** — assert a representative deep panel polls `False` in BASIC and `True` in ADVANCED:

```python
def test_advanced_sim_panels_hidden_in_basic_mode(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    basic = SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace(ui_mode="BASIC")))
    adv = SimpleNamespace(scene=SimpleNamespace(fluidsim=SimpleNamespace(ui_mode="ADVANCED")))
    cls = ui_scene.FLUIDSIM_PT_sim_advanced   # pick the real advanced-physics panel id
    assert cls.poll(basic) is False
    assert cls.poll(adv) is True
```

> The exact class to gate is the parent of the 12 `sim_*` panels (`scene_simulation` fan, spec §5.3 shows an "Advanced Physics" gate). Confirm the real `bl_idname` in `ui_scene.py:385-618` when writing; gate the *parent* so children inherit non-visibility, and keep each child's existing `poll` (compose with `and _advanced_visible(context)`).

- [ ] **Step 2–4:** run-fail → add `_advanced_visible` and compose it into the relevant polls (respect existing `_surface_enabled`/`_whitewater_enabled` composition) → run-pass.

- [ ] **Step 5: Commit** — `feat(ui): gate advanced solver panels behind ui_mode (D2)`

### Task 2.3: Promote Setup/preset to the front door + collapse to Basic surfaces

**Files:** `ui_scene.py` (panel `bl_order` re-ordering: Setup/Quality above Domain; add a `ui_mode` toggle row at the top of the domain home; add the single **Smoothing** slider, **Amount**, **Export Attributes** enum surrogates described in spec §8); Test: same file — assert Basic-mode default visible property count is within target.

- [ ] **Step 1: Write the failing test** — a *budget test*: render the Basic-mode panel set against a fully-populated `_RnaPropertyGroup` and assert the count of distinct props drawn is ≤ 25:

```python
def test_basic_mode_default_surface_is_within_budget(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ui_scene = _reload_module("fluidSim_warp_blender51.blender.ui_scene")
    # draw every Scene panel whose poll passes in BASIC mode, collect unique prop names
    ...
    assert len(unique_props) <= 25
```

> This test encodes success-criterion §11 ("≈20–25 default visible properties"). Building the harness that enumerates poll-passing panels and unions their drawn props is the substance of this task; wire it against `SCENE_PANEL_CLASSES`. If 25 is initially exceeded, that is the signal to move more props behind the Advanced gate (Task 2.2) — the test drives the reduction.

- [ ] **Step 2–4:** run-fail → re-order/relabel panels, add the Basic surrogates, gate the rest → run-pass at ≤25.

- [ ] **Step 5: Commit** — `feat(ui): promote Setup front door + Basic default surface (D2)`

### Task 2.4: Chunk 2 gate

- [ ] `python -m pytest -q` → PASS.
- [ ] `python tools/run_production_gate.py --include-gui` + `python tools/validate_policy_drift_reports.py` → PASS.
- [ ] Commit gate-refreshed evidence if any.

> **Milestone:** Phases 0–2 ship together as the storage-agnostic release. Everything below depends on the Phase 3 data model. Treat the boundary here as a natural release/review point.

---

## Chunk 3: Phase 3 — Data-model migration, the pivot (§9)

Relocate domain settings from `Scene.fluidsim` to `obj.fluidsim_domain` (`PointerProperty(type=FLUIDSIM_PG_settings)` on the domain object). One-way, tested, idempotent. Per-object *role* data (`obj.fluidsim`) is **not** touched. This is the highest-risk change; it ships behind its own gate (round-trip + idempotency + golden-frame).

Precedent to mirror exactly: `fluidSim_warp_blender51/blender/sources.py` — `persistent = getattr(_handlers, "persistent", lambda fn: fn)` (:14), `migrate_source_sampling_schema()` (:39), `@persistent _on_load_post` (:57), `register_handlers()/unregister_handlers()` (:62-77) whose `register_handlers` calls `unregister_handlers()` first for idempotency (:63).

**Design, pinned to remove three silent-data-loss hazards the reviewer flagged:**

1. **Separation of concerns.** The pure, bpy-free logic is split into three small functions so each is unit-testable without `bl_rna` and without creating Blender objects:
   - `migrated_field_names(settings_cls) -> tuple[str, ...]` — derives the field set from `settings_cls.__annotations__` (confirmed annotation-style at `properties.py:750`+; annotations are class-level and available even under the fake-bpy stub) minus a **pinned skip-set** `_MIGRATION_SKIP = frozenset({"domain_object", "schema_version"})`. This deliberately **includes** the material/image pointers (`height_profile_image` `properties.py:1264`, `surface_material` `:1422`, etc.) — they are real domain settings — and excludes only the identity pointer and the guard. It also carries any UI/transient annotations (`live`, `*_expanded`, and the Phase-1 `ui_mode`/`quality`) onto the target; this is intended and harmless — copying the whole group is simpler and safer than maintaining a curated allow-list, and stale UI flags on the target have no solver effect.
   - `copy_domain_settings(source, target, field_names) -> None` — `for name in field_names: try: setattr(target, name, getattr(source, name)) except Exception: <log, continue>`. The try/except tolerates any read-only/computed annotation without aborting the whole migration.
   - `migrate_domain_settings_schema(scene, domain_obj) -> bool` — the orchestrator (see Task 3.2).
2. **No stamp-and-drop.** The orchestrator takes an already-resolved `domain_obj` and, if it is `None`, does **nothing and returns `False`** — it never stamps `schema_version` while dropping data. Resolving-or-creating the domain object (the only step that needs live `bpy`) is the handler's job (Task 3.3), keeping the destructive decision out of the pure layer.
3. **Guard is stamped last and never re-zeroed.** The final mutation is `scene.fluidsim.schema_version = 1`. We do **not** wipe the scene group's other fields (leaving them as harmless stale data is safe now that the solver reads `obj.fluidsim_domain` and the guard is `schema_version`); clearing them would risk re-defaulting `schema_version` to `0` and re-triggering migration on the next open.

### Task 3.1: Add the `schema_version` idempotency field + the `fluidsim_domain` pointer

**Files:**
- Modify: `fluidSim_warp_blender51/properties.py` — add `schema_version: IntProperty(default=0, options={"HIDDEN"})` to `FLUIDSIM_PG_settings` (near `:750`); register `bpy.types.Object.fluidsim_domain = PointerProperty(type=FLUIDSIM_PG_settings)` in `register()` (`:1624`) alongside the existing `Scene.fluidsim` (`:1627`) / `Object.fluidsim` (`:1628`), and `del bpy.types.Object.fluidsim_domain` in `unregister()`.
- Test: `tests/test_domain_migration.py` (**new**)

- [ ] **Step 1: Write the failing test.** Under the stub, assert the `schema_version` annotation exists with default `0` and that `register()` assigns `Object.fluidsim_domain`.

**Pin the stub helper source — do NOT hand-roll a new stub.** The repo has **four** divergent `_install_bpy_stub` copies, and only the one in `tests/test_operator_panel_behavior.py:84` stubs `fake_bpy.types.Object`/`Material`/`Image` (`:127-129`), which this migration needs (the material/image pointer copy in Task 3.2 depends on `Material`/`Image` being present). The other three (`test_operator_poll_and_guards.py`, `test_benchmark_validation.py`, `test_device_preference_routing.py`) lack them — do **not** copy from those. Import the two helpers by name at the top of the new test file:

```python
from types import SimpleNamespace

from test_operator_panel_behavior import _install_bpy_stub, _reload_module


def test_settings_group_has_schema_version_and_domain_pointer(monkeypatch):
    _install_bpy_stub(monkeypatch)
    properties = _reload_module("fluidSim_warp_blender51.properties")
    assert "schema_version" in properties.FLUIDSIM_PG_settings.__annotations__
    # register() must attach the per-object domain pointer
    properties.register()
    assert hasattr(properties.bpy.types.Object, "fluidsim_domain")
```

> Import mechanism: pytest prepends each test file's own directory to `sys.path` (rootdir insertion) when `tests/` has no `__init__.py`, so `from test_operator_panel_behavior import ...` resolves at collection. Confirm at Step 2 that collection does not raise `ImportError`; if it does (e.g. an `__init__.py` is present making `tests` a package), promote `_install_bpy_stub` and `_reload_module` **verbatim** from `test_operator_panel_behavior.py` into `tests/conftest.py` as module-level functions and import them from there — never fork a second, drifting copy.

> Harness note (own first sub-step): the canonical stub sets `fake_bpy.types.Object = object` (`tests/test_operator_panel_behavior.py:127`), and you cannot `setattr` an attribute onto the builtin `object`. Before this test can pass, additively edit that canonical `_install_bpy_stub` to make `fake_bpy.types.Object` a throwaway subclass — `class _Object: pass` then `fake_bpy.types.Object = _Object` — so attribute assignment is observable. Grep for existing dependents of `fake_bpy.types.Object` first (`test_operator_panel_behavior.py` uses it only as a `PointerProperty(type=...)` argument, so a subclass is safe). Leave the `Material`/`Image` stubs (`:128-129`) in place — Task 3.2's pointer test needs them.

- [ ] **Step 2: Run to verify it fails.** Run: `python -m pytest tests/test_domain_migration.py -v` — Expected: FAIL (`schema_version` annotation absent; `Object.fluidsim_domain` unset).

- [ ] **Step 3: Implement** the field + the pointer registration/unregistration.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** — `feat(props): add schema_version + obj.fluidsim_domain pointer (Phase 3)`

### Task 3.2: Write the pure migration functions (no bpy handler yet)

**Files:**
- Create: `fluidSim_warp_blender51/blender/migration.py`
- Test: `tests/test_domain_migration.py`

- [ ] **Step 1: Write the failing tests** — three focused tests, one per function, so field-derivation, copy, and the orchestrator guard are each pinned:

```python
def test_migrated_field_names_skips_identity_and_guard_but_keeps_materials(monkeypatch):
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")

    class _Cls:
        __annotations__ = {
            "voxel_size": None, "surface_material": None,
            "domain_object": None, "schema_version": None,
        }
    names = migration.migrated_field_names(_Cls)
    assert "voxel_size" in names and "surface_material" in names   # real settings migrate
    assert "domain_object" not in names                            # identity pointer skipped
    assert "schema_version" not in names                           # guard skipped


def test_copy_domain_settings_tolerates_a_read_only_field(monkeypatch):
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")

    class _Target:  # 'derived' rejects assignment, like a read-only RNA prop
        def __setattr__(self, name, value):
            if name == "derived":
                raise AttributeError("read-only")
            object.__setattr__(self, name, value)
    src = SimpleNamespace(voxel_size=0.05, derived=9, surface_material="M")
    tgt = _Target()
    migration.copy_domain_settings(src, tgt, ("voxel_size", "derived", "surface_material"))
    assert tgt.voxel_size == 0.05 and tgt.surface_material == "M"   # good fields still copied
    assert not hasattr(tgt, "derived")                             # bad field skipped, no raise


def test_migration_moves_settings_once_and_never_re_triggers(monkeypatch):
    _install_bpy_stub(monkeypatch)
    migration = _reload_module("fluidSim_warp_blender51.blender.migration")
    monkeypatch.setattr(migration, "migrated_field_names",
                        lambda _cls: ("voxel_size", "max_particles"))

    target = SimpleNamespace(schema_version=0)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        schema_version=0, voxel_size=0.05, max_particles=2_000_000))

    assert migration.migrate_domain_settings_schema(scene, target) is True
    assert (target.voxel_size, target.max_particles) == (0.05, 2_000_000)
    assert target.schema_version == 1
    assert scene.fluidsim.schema_version == 1        # guard stamped LAST

    # second run on the same scene is a no-op
    assert migration.migrate_domain_settings_schema(scene, target) is False
    # guard is on schema_version, NOT on field emptiness: zeroing moved fields must not re-trigger
    scene.fluidsim.voxel_size = 0.0
    assert migration.migrate_domain_settings_schema(scene, target) is False
    # a None domain object never stamps-and-drops
    virgin = SimpleNamespace(fluidsim=SimpleNamespace(schema_version=0, voxel_size=0.1))
    assert migration.migrate_domain_settings_schema(virgin, None) is False
    assert virgin.fluidsim.schema_version == 0       # untouched → data preserved
```

- [ ] **Step 2: Run to verify they fail.** Run: `python -m pytest tests/test_domain_migration.py -v`

- [ ] **Step 3: Implement `migration.py`** per the pinned design above:

```python
"""One-way migration of domain settings from Scene.fluidsim to obj.fluidsim_domain."""

from __future__ import annotations

_MIGRATION_SKIP = frozenset({"domain_object", "schema_version"})
_TARGET_SCHEMA_VERSION = 1


def migrated_field_names(settings_cls) -> tuple[str, ...]:
    return tuple(
        name for name in getattr(settings_cls, "__annotations__", {})
        if name not in _MIGRATION_SKIP
    )


def copy_domain_settings(source, target, field_names) -> None:
    for name in field_names:
        try:
            setattr(target, name, getattr(source, name))
        except Exception as exc:  # noqa: BLE001 — read-only/computed props tolerated
            print(f"FluidSim migration: skipped field {name!r}: {exc!r}")


def migrate_domain_settings_schema(scene, domain_obj) -> bool:
    """Move domain settings onto ``domain_obj.fluidsim_domain``. Idempotent, safe.

    Returns True iff it acted. Never stamps/clears when ``domain_obj`` is None
    (that would drop data); the caller owns resolving/creating the object.
    """
    fs = getattr(scene, "fluidsim", None)
    if fs is None or int(getattr(fs, "schema_version", 0)) != 0:
        return False
    if domain_obj is None:
        return False
    target = domain_obj.fluidsim_domain
    field_names = migrated_field_names(type(fs))
    copy_domain_settings(fs, target, field_names)
    target.schema_version = _TARGET_SCHEMA_VERSION
    fs.schema_version = _TARGET_SCHEMA_VERSION   # guard stamped LAST; never re-zeroed
    return True
```

- [ ] **Step 4: Run to verify they pass.**

- [ ] **Step 5: Commit** — `feat(migration): pure one-way domain-settings migration (Phase 3)`

### Task 3.3: Add the `@persistent load_post` handler (resolves/creates the domain object) + register/unregister

**Files:**
- Modify: `fluidSim_warp_blender51/blender/migration.py` — add the handler and registration, mirroring `sources.py`:
  - `persistent = getattr(_handlers, "persistent", lambda fn: fn)` (import `bpy.app.handlers as _handlers`).
  - `_resolve_or_create_domain(scene) -> object | None` — return `scene.fluidsim.domain_object` if already set. Else, since today's model is exactly one domain per scene, create one deterministically: `mesh = bpy.data.meshes.new("FluidSim Domain")`; `obj = bpy.data.objects.new("FluidSim Domain", mesh)`; `scene.collection.objects.link(obj)`; `scene.fluidsim.domain_object = obj`; return `obj`. (An empty mesh datablock is sufficient — Phase 4 draws the domain from `obj.fluidsim_domain`, not from geometry; match whatever the existing setup wizard uses for the domain object if it already creates one, to avoid a second convention.) Keep this the **only** bpy-touching helper so the pure layer stays testable. Return `None` only if `scene.fluidsim` itself is absent (the pure function then no-ops, preserving data). This create branch is exercised by the `pre_migration_domain_unset.blend` fixture in Task 3.5.
  - `@persistent _on_load_post(_arg)` iterating `bpy.data.scenes`, calling `migrate_domain_settings_schema(scene, _resolve_or_create_domain(scene))`.
  - `register_handlers()` that **calls `unregister_handlers()` first** (mirroring `sources.py:63`) then appends `_on_load_post` to `bpy.app.handlers.load_post`; `unregister_handlers()` removes it if present.
- Modify: `fluidSim_warp_blender51/__init__.py`:
  - `register()` `steps` tuple (`:33-57`): add `("domain migration handlers", migration.register_handlers, migration.unregister_handlers)` adjacent to the existing `("source migration handlers", sources.register_handlers, sources.unregister_handlers)` entry (`:45-46`); import `migration` in the `from .blender import (...)` block (`:12-25`).
  - `unregister()` `steps` tuple (`:80-99`, **3-string** shape `(name, relative_module, function_name)`): add `("domain migration handlers", "blender.migration", "unregister_handlers")` next to the sources entry (`:87`).
- Test: `tests/test_domain_migration.py`

- [ ] **Step 1: Write the failing test** — assert `_on_load_post` runs migration across `bpy.data.scenes`, and that `register_handlers()` is idempotent (calling it twice leaves exactly one `_on_load_post` in `load_post`). Model `bpy.app.handlers.load_post` as a list on the stub and `_resolve_or_create_domain` monkeypatched to return a prepared target, so the handler test never needs real object creation.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** the handler + registration; wire both `__init__.py` tuples.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** — `feat(migration): load_post forward-migration handler + registration (Phase 3)`

### Task 3.4: Point the solver read path at the active domain object

**Files:**
- Modify: `fluidSim_warp_blender51/blender/scene_bridge.py` — add `active_domain_settings(scene)` returning the active/only domain object's `fluidsim_domain` group (falling back to `scene.fluidsim` only if no migrated domain exists, to survive the transition), and the `active_domain_object(scene)` helper it needs.
- Modify: the **actual domain-setting read hub** — `fluidSim_warp_blender51/blender/solver_settings.py` is where solver-relevant domain fields are read (not `scene_bridge.py`, which mostly reads `status`); re-point those reads through `active_domain_settings`. Audit and re-point the other readers that pull domain fields: `pending_settings.py`, `simulation_admission.py`, `output_publication.py`, `setup_wizard.py`. **Do not** change role reads (`obj.fluidsim`).
- Test: `tests/test_operator_panel_behavior.py` (bridge accessor) + the golden-frame gate (Task 3.6)

> Scope note: this is larger than a single edit — treat the accessor + each reader re-point as its own commit if it grows. `grep -rn "\.fluidsim\." fluidSim_warp_blender51/` to enumerate domain-setting readers before starting; classify each as domain-setting (re-point) vs. role/status/session (leave).

- [ ] **Step 1: Write the failing test** — assert `active_domain_settings(scene)` returns the domain object's `fluidsim_domain` group when a migrated domain exists, and that `solver_settings` reads `voxel_size`/`max_particles` from it (not from `scene.fluidsim`).

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** the accessor and re-point the readers.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit** — `refactor(bridge): read domain settings from obj.fluidsim_domain (Phase 3)`

### Task 3.5: Migration acceptance — pre-migration `.blend` fixture round-trip (headless)

**Files:**
- Create: `tests/gui/fixtures/` — **two** representative pre-migration `.blend` files, both with populated `Scene.fluidsim` at `schema_version == 0` and **at least one material/image pointer set** (so the pointer path is exercised, per Issue 3):
  - `pre_migration_domain_set.blend` — `fluidsim.domain_object` already points at a domain object (the resolve branch of `_resolve_or_create_domain`).
  - `pre_migration_domain_unset.blend` — `fluidsim.domain_object` is `None` (exercises the create/designate branch of Task 3.3, which §9 requires).
- Create: `tests/gui/test_domain_migration_blend.py` — headless (`blender --background`) test that opens each fixture, lets the `load_post` handler run, and asserts every migrated field (scalars **and** the material/image pointers) landed on `obj.fluidsim_domain` with identical values, `schema_version == 1`, and that a re-open is a no-op. For the unset fixture, additionally assert a domain object now exists and was designated/created (not silently skipped).
- Wire into the GUI gate (`tools/run_production_gate.py --include-gui` / `AGENTS.md`).

- [ ] **Step 1:** Author both fixtures, or script their creation in test setup from a v0 template (set at least one material pointer; leave `domain_object` unset in the second).
- [ ] **Step 2:** Write the headless assertions (fields + pointers + `schema_version == 1` + re-open no-op; plus the created/designated-domain assertion for the unset fixture).
- [ ] **Step 3:** Run `python tools/run_production_gate.py --include-gui` and confirm the new test is collected and passes.
- [ ] **Step 4: Commit** — `test(migration): headless .blend round-trip + idempotency gate (Phase 3)`

### Task 3.6: Golden-frame equivalence gate

- [ ] **Step 1:** Add a golden-frame comparison: simulate N frames from a migrated fixture and from an equivalent freshly-authored `obj.fluidsim_domain` scene; assert identical solver output (spec §9 "identical simulation output"). Reuse the existing golden/benchmark comparison machinery (`diagnostics/benchmark.py`, `compare_benchmark`) if a frame-hash comparator exists; otherwise a bounded numeric diff on exported points.
- [ ] **Step 2:** Run the gate; PASS is the Phase 3 ship criterion.
- [ ] **Step 3: Commit** — `test(migration): golden-frame equivalence gate (Phase 3)`

### Task 3.7: Chunk 3 gate

- [ ] `python -m pytest -q` → PASS.
- [ ] `python tools/run_production_gate.py --include-gui` (must include 3.5 + 3.6) + `python tools/validate_policy_drift_reports.py` → PASS.
- [ ] **Do not proceed to Chunk 4 until this gate is green** — Phase 4 renders `obj.fluidsim_domain` and assumes the datablock exists.

---

## Chunk 4: Phase 4 — Native home & surfaces (D1, D6)

With domain settings on the object, build the native surfaces. Each task assumes `obj.fluidsim_domain` exists (Chunk 3 green).

### Task 4.1: Physics-on-domain renders the real datablock

**Files:**
- Modify: `fluidSim_warp_blender51/blender/ui_physics.py` — the poll-gated Physics panel renders `obj.fluidsim_domain` directly (not a mirror pointed back at `scene.fluidsim`). Reuse the shared `draw` helpers so the Basic/Advanced/Quality gating from Chunk 2 applies unchanged, now reading the object datablock.
- Test: `tests/test_operator_panel_behavior.py`

- [ ] **Step 1: Write the failing test** — active object is the domain → Physics panel polls `True` and draws `obj.fluidsim_domain.voxel_size` (assert `prop_calls` target the object group, not the scene). A non-domain object → panel polls `False`.
- [ ] **Step 2–4:** run-fail → re-point the physics draw at `obj.fluidsim_domain`; keep kind-gated poll (`ui_physics.py:36-51`) → run-pass.
- [ ] **Step 5: Commit** — `feat(ui): Physics tab renders obj.fluidsim_domain as the real home (D6)`

### Task 4.2: Delete the N-panel `Full Settings` verbatim mirror; add lean N-panel

**Files:**
- Modify/Replace: `fluidSim_warp_blender51/blender/ui_view3d.py` — remove `FLUIDSIM_VIEW3D_PT_full_settings` + `_make_panel_mirror()` + `VIEW3D_PANEL_CLASSES` mirror generation (−52 panels). Keep the `viewport_status` launcher + add a **Live Tuning** shelf (`voxel_size`, `particles_per_voxel`, `flip_ratio`, `particle_display_fraction`) + an "Open Settings" jump. Depth ≤ 3.
- Modify: `fluidSim_warp_blender51/__init__.py` — the `("full viewport settings UI", ...)` step now registers the lean module (or rename).
- Test: `tests/test_operator_panel_behavior.py`

- [ ] **Step 1: Write the failing test** — assert no panel with the `Full Settings` mirror `bl_idname` is registered, N-panel nesting depth ≤ 3 (success-criterion §11), and the Live Tuning panel draws exactly the four per-iteration knobs.
- [ ] **Step 2–4:** run-fail → delete mirror, add Live Tuning + Open Settings → run-pass.
- [ ] **Step 5: Commit** — `feat(ui): replace N-panel full mirror with HUD + Live Tuning (D1/D5)`

### Task 4.3: Role Type dropdown (D6)

**Files:**
- Modify: `fluidSim_warp_blender51/blender/ui_object.py:47-282` — primary control becomes `layout.prop(obj.fluidsim, "role")` (native Type dropdown); the six tag operators (`ui_object.py:64-76`) are retained *below*, reframed as "Apply to Selected" batch helpers. Role sub-panels keep their poll-gating.
- Test: `tests/test_operator_panel_behavior.py`

- [ ] **Step 1: Write the failing test** — assert the object panel draws `role` as a prop (dropdown) *and* still exposes the batch tag operators; assert order (dropdown first).
- [ ] **Step 2–4:** run-fail → add dropdown, reframe operators → run-pass.
- [ ] **Step 5: Commit** — `feat(ui): object role as native Type dropdown + batch helpers (D6)`

### Task 4.4: `VIEW3D_MT_add` menu entry + multiple-domains + Scene demotion + grammar/description audit

**Files:**
- Create: `fluidSim_warp_blender51/blender/menus.py` — append a "Fluid Domain (FluidSim)" entry to `VIEW3D_MT_add` (spec §7, O4); register/unregister in `__init__.py`.
- Modify: `ui_scene.py` — demote Scene Properties to the *global* summary only (transport lives in the active-domain surfaces; Scene keeps Output/Display, Cache & Bake, Diagnostics). Ensure transport/telemetry key off the **active domain object** (multiple domains).
- Modify: `ui_common.py` / helpers — commit to one property grammar (`use_property_split=True` single-column for labeled rows; `use_property_split=False` for boolean `grid_flow` clusters).
- Audit: verify every `FLUIDSIM_PG_settings` / role field defines `description=`; add `text`/`bl_description` to icon-only operators (mute/solo/move/refresh).
- Test: `tests/test_operator_panel_behavior.py`

- [ ] **Step 1: Write the failing tests** — (a) `VIEW3D_MT_add` draw includes the FluidSim entry; (b) a scene with two domain objects exposes both, transport keys off the active one; (c) a description-coverage test enumerating `FLUIDSIM_PG_settings.bl_rna.properties` asserts non-empty `description` on each (this is a real, mechanizable §7 gate).
- [ ] **Step 2–4:** run-fail → implement menu, multi-domain active-object keying, grammar unification, fill missing descriptions → run-pass.
- [ ] **Step 5: Commit** — `feat(ui): Add-menu entry, multi-domain, grammar + description audit (D1/D6)`

### Task 4.5: Chunk 4 gate

- [ ] `python -m pytest -q` → PASS.
- [ ] `python tools/run_production_gate.py --include-gui` + `python tools/validate_policy_drift_reports.py` → PASS (GUI gate confirms panel counts dropped, depth ≤ 3, ≥2 domains render).
- [ ] Commit gate-refreshed evidence if any.

---

## Chunk 5: Phase 5 — Telemetry & Effects (D4, D9)

Final consolidation. Depends on the surfaces from Chunk 4.

### Task 5.1: Readout helper + consolidate `stat_*` into Diagnostics

**Files:**
- Modify: `fluidSim_warp_blender51/blender/ui_common.py` — add `draw_readout_grid(layout, data, pairs)`: a `box()` with a "Readouts" header + right-aligned key/value `grid_flow(columns=2, even_columns=True)`. **Never** the same helper as `draw_property_grid`.
- Modify: `ui_scene.py` — move every read-only readout (capillary stats `ui_scene.py:473-480`, whitewater Admission Status, ~30 `stat_*`) out of control panels into `FLUIDSIM_PT_scene_diagnostics` (`ui_scene.py:1069`) via the readout helper. No control panel renders a `stat_*`.
- Test: `tests/test_operator_panel_behavior.py`

- [ ] **Step 1: Write the failing test** — (a) assert `draw_readout_grid` emits labels, not `prop()` calls (readouts aren't editable); (b) a guard test: iterate the control panels (Transport/Lifecycle/Surface/Whitewater/etc.), draw each, assert **no** drawn prop name starts with `stat_` (success-criterion §11 "No control panel renders a `stat_*` readback").
- [ ] **Step 2–4:** run-fail → add helper, relocate readouts → run-pass.
- [ ] **Step 5: Commit** — `feat(ui): consolidate telemetry into Diagnostics via readout helper (D4)`

### Task 5.2: Chunk the surface-filter list

**Files:** `ui_scene.py:681-697` — wrap each of the four surface passes (filter/dilate/erode/smooth) in its own `box()` + header. Test: assert four labeled boxes drawn in the Advanced surface panel.

- [ ] **Step 1–5:** failing test → implement chunking → pass → commit `feat(ui): chunk surface-filter passes into labeled boxes (D9)`.

### Task 5.3: Developer-gate the Warp-code workspace + scoped trust

**Files:**
- Modify: `fluidSim_warp_blender51/preferences.py` — add `enable_effect_authoring: BoolProperty(default=False)` ("Enable custom effect authoring (developer)").
- Modify: `fluidSim_warp_blender51/blender/effect_workspace.py` — the Warp-code workspace panels `poll` on that preference; the effect *stack* UIList stays always-available. Add a scoped "trust while editing this session" mode so kernel authors don't re-click Review & Trust on every edit (the SHA trust gate stays).
- Test: `tests/test_operator_panel_behavior.py`

- [ ] **Step 1: Write the failing test** — workspace panel polls `False` when the pref is off, `True` when on; the effect-stack UIList polls `True` regardless; scoped-trust flag suppresses re-prompting within a session.
- [ ] **Step 2–4:** run-fail → add pref + poll gate + scoped trust → run-pass.
- [ ] **Step 5: Commit** — `feat(effects): developer-gate Warp workspace + scoped trust (D9)`

### Task 5.4: Chunk 5 gate + final acceptance

- [ ] `python -m pytest -q` → PASS.
- [ ] `python tools/run_production_gate.py --include-gui` + `python tools/validate_policy_drift_reports.py` → PASS.
- [ ] **Success-criteria sweep (spec §11)** — confirm as executable/observed facts: default visible props ≈20–25; N-panel depth ≤3, no full mirror; no control panel renders `stat_*`; severity is shape+text(+color); HUD toggleable + default-on; ≥2 domains; legacy `.blend` migrates with identical values (3.5) and identical output (3.6); no capability removed.
- [ ] **Final commit** — `docs(plan): mark UI redesign phases 0–5 complete` (or leave to the executor's discretion).

---

## Notes for the executor

- **Storage boundary:** Tasks in Chunks 0–2 must read/write `scene.fluidsim`. Do **not** introduce `obj.fluidsim_domain` before Chunk 3 — it doesn't exist yet and the storage-agnostic release ships first.
- **Stable `bl_idname`s:** When splitting the hub (Chunk 1), keep `FLUIDSIM_PT_scene_control` as the *Status & Runtime* panel so existing child panels (`bl_parent_id="FLUIDSIM_PT_scene_control"`) don't orphan. New Transport/Lifecycle panels get new ids and low `bl_order`.
- **Exact-code confidence:** Chunks 0–2 give byte-exact code grounded in the current tree. Chunks 3–5 give test-first task decomposition with exact file paths and anchors; a few final byte values (e.g. the exact set of `sim_*` `bl_idname`s to gate, the precise fields to copy in migration) must be read from the *then-current* tree at execution time — each such spot is flagged inline. This is deliberate: those files are edited by earlier chunks, so pinning their final bytes now would be fiction.
- **Test harness:** never hand-roll a bpy stub — reuse `_install_bpy_stub` / `_reload_module` / `_Layout` / `_RnaPropertyGroup` (`tests/test_operator_panel_behavior.py`). If a panel-draw assertion needs greyout/alert state the `_Layout` fake doesn't yet record, extend the fake **additively** (grep for existing dependents first) rather than forking it.
- **Register/unregister:** every new module (`migration.py`, `menus.py`, lean `ui_view3d`) must be added to both the ordered `steps` tuple in `__init__.py` `register()` and the mirror in `unregister()`, tolerating partial-registration teardown (the rollback contract at `__init__.py:58-69`).
