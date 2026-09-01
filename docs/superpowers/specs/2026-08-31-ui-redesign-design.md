# FluidSim Warp — UI Redesign Design Spec

**Date:** 2026-08-31
**Status:** Draft for review
**Scope:** UI/IA redesign of the Blender 5.1 add-on (`fluidSim_warp_blender51/`), **including the data-model migration required to make it fully native**. The solver and runtime behavior are unchanged; the storage location of settings changes (Scene-singleton → per-object modifier) because that migration unlocks the correct architecture and is explicitly in-scope.
**Method:** Six-member expert "council" critique, each grounded in the actual `draw()` code (see §2 / Appendix B).

---

## 1. Problem statement

The add-on exposes a very large, capable feature set through a UI that has grown organically into a wall of settings:

| Metric | Current |
| --- | --- |
| Registered `Panel` instances | **158** (71 authored + 87 auto-generated mirrors) |
| Properties surfaced | **402** (≈228 on `Scene.fluidsim` alone; counts approximate, from the UI-map scan) |
| Operators exposed as buttons | **71** |
| Editors the same settings appear in | **4** (Scene, Object, Physics Properties + 3D-View N-panel) |
| Max panel nesting | **5 levels** (3D-View sidebar) |
| Longest single `draw()` | 46 layout calls (`effect_workspace.py:131`) |

The engineering underneath is strong — a single-authored draw layer, a clean activity-state model, kind-gated Physics polls, a role-based object model, an already-built viewport HUD, and a preset/wizard system. **The failure is almost entirely surfacing and information architecture, not capability.** The same authoritative panel list is mirrored verbatim into three editors, so every clutter defect is triplicated; there is no basic→advanced gradient, so a first-time user faces the same 228-property surface as an expert; and several already-built conveniences (the HUD, the preset front door, the wizard's mode choices) are hidden.

### Goals

1. **Native architecture.** The fluid domain **is an object** (a modifier on it), matching Mantaflow — not a scene singleton. This unlocks multiple domains, per-object roles, drivers, and library linking.
2. **One obvious home per task.** Collapse the four overlapping surfaces into purpose-built roles.
3. **A real on-ramp.** A Basic mode / Quality preset path that shows ~20–25 properties by default, with full control one gate away.
4. **A legible daily loop.** Transport, live telemetry, and the handful of per-iteration knobs prominent and stable.
5. **Native feel throughout.** Match Blender's physics/fluid conventions (home, Type dropdown, entry points, asset-browser presets, layout grammar).
6. **Correct, accessible presentation.** Primary/destructive action clarity, non-inverted severity, readouts distinct from controls, tooltips and descriptions, no silent truncation.

### Non-goals

- Rewriting the solver or changing runtime/simulation behavior. The migration relocates *where settings are stored*; it does not change what the solver computes.
- Removing any capability. Everything reachable today stays reachable; much of it simply moves behind a gate, preset, or the Diagnostics home.
- Custom OpenGL/GPU-drawn UI. Everything here is achievable in Blender's native `layout` toolkit (the existing `viewport_overlay.py` HUD is the one intentional exception and already exists).
- A backward-compatible dual-storage mode. We migrate old `.blend` files forward on load (§9); we do not maintain the Scene-singleton store in parallel.

---

## 2. The council

Six independent reviewers, each dispatched as a subagent that read the real `draw()` code through one lens:

1. **Blender-native UX conventions** — how native fluid/physics/GN UIs are built; what feels foreign.
2. **Simulation TD / power-user workflow** — the real setup→preview→tune→bake→render loop.
3. **Information architecture** — grouping, hierarchy, progressive disclosure, naming.
4. **Visual & interaction design** — hierarchy, density, affordance, severity, icons (native toolkit only).
5. **Onboarding, discoverability, error recovery, accessibility** — first-run, empty states, color-blind/localization.
6. **YAGNI / ruthless reduction** — what to cut, hide, merge, or default.

---

## 3. Consensus diagnosis (prioritized)

Findings are ranked by how many lenses independently raised them and by user impact. Evidence is `file:line` in `fluidSim_warp_blender51/`.

### D1 — Triplication: one hierarchy forced onto three incompatible contexts *(IA, Native, YAGNI)*
The authoritative Scene panel list is auto-mirrored into Physics Properties (kind-gated) and the 3D-View sidebar (re-parented verbatim under one `Full Settings` node, 5 levels deep — `ui_view3d.py:39-91`), plus a fourth compact launcher (`panels.py:34`). DRY at the *code* layer (both mirrors delegate to one `draw()`), but it forces a single hierarchy designed for a wide Properties editor into a narrow sidebar and a per-object context, and triplicates every layout defect.
**Impact:** "Where do I go?" paralysis; the sidebar's fast-tweak promise is defeated by 4–5 disclosure clicks to reach one value.

### D2 — No basic→advanced gradient; 228 properties presented undifferentiated *(IA, YAGNI, Onboarding)*
Six top-level roots are `DEFAULT_CLOSED`; `scene_simulation` fans into **12 flat sibling sub-panels** (`ui_scene.py:385-618`). Disclosure is binary (open/closed), never "show the 3 knobs 90% touch, hide the 200 they never do." The reduction machinery **already exists but is buried** — `product_preset` + Apply/Save (`ui_scene.py:294-297`) and the `setup_scene` Guided Setup wizard (`ui_scene.py:283`) are demoted to child panels while raw solver internals get top billing.
**Impact:** reads as an expert console with no on-ramp; high first-run abandonment risk.

### D3 — `scene_control` hub overloaded; primary/destructive actions undifferentiated *(all six)*
One panel carries: state label, status string, height error, runtime probe, one-of-four lifecycle actions, Initialize/Release, Step-once, Step-many, Reset, and an error box — ~22 layout calls, ~7 conditional branches, all equal-weight sibling rows (`ui_scene.py:135-201`). "Release Solver" (free GPU) and "Reset Simulation" (re-run from frame 1) render identically and adjacent to benign controls; no `alert`, no `scale_y`, no separators. Buttons appear/disappear in place as hidden state changes, so controls move under the cursor.
**Impact:** the most-used panel is the least self-explanatory; Reset/Release are one mis-click from Step.

### D4 — Telemetry is hidden and mixed into control panels *(TD, IA, Visual)*
- The **viewport HUD is fully built but unreachable**: `show_overlay` is drawn by `viewport_overlay.py:37` but exposed by **no panel** (`properties.py:1541` defines it; nothing renders it). One-line fix, large payoff.
- Key stats (`stat_particles/voxels/step_ms/substeps/cfl_clamped`) sit in a `DEFAULT_CLOSED`, ownership-gated panel (`ui_scene.py:232-259`) — collapsed exactly when you're tuning against them.
- Read-only stats are interleaved with controls (capillary stats box inside Surface Tension — `ui_scene.py:473-480`) and pushed through the same `draw_property_grid` as editable fields, so a reported number looks editable.
**Impact:** the tune→observe→tune rhythm is broken; ~30 scattered `stat_*` readbacks never read as a dashboard.

### D5 — Frequency-of-use vs prominence inverted *(TD, IA)*
`flip_ratio` — the single most look-defining FLIP knob — lives in `sim_time` (`ui_scene.py:394`), a child of the `DEFAULT_CLOSED` `scene_simulation`, next to near-once knobs (`count_stiffness`, `trace_mode`). Meanwhile `voxel_size` is correctly prominent. The per-iteration knobs are scattered across the deep tree.
**Impact:** a navigation tax on every tuning pass.

### D6 — Non-native idioms *(Native)*
- Authoritative home is **Scene Properties**; Blender's own fluid (Mantaflow) has no scene-level fluid UI — the domain *is* an object and its physics live in **Physics Properties** on that object. A user's instinct (select domain → Physics tab) lands on a *mirror* (`ui_physics.py`) while the real datablock is on the Scene.
- Role assignment is a **column of six operators** (`ui_object.py:64-76`) though `role` is already an `EnumProperty` (`properties.py:438`); native pattern is a single Type dropdown.
- **Zero** menus/headers/pie/Add-menu entries — the add-on is invisible to muscle-memory discovery paths.
- `use_property_split=True` combined with 2-col `grid_flow` (`ui_common.py:120-131`) cramps fields and misaligns labels.

### D7 — Onboarding gaps and a real bug *(Onboarding)*
- **`Guided Setup…` lies:** the operator has no `invoke` (`source_roles.py:83-104`) yet the button carries a trailing ellipsis (`ui_scene.py:283`) and a 6-way `mode` enum; clicking silently runs only the default `QUICK_LIQUID` — the other 5 modes are reachable only via the F9 redo panel a newcomer doesn't know exists.
- Empty scene shows all panels (poll only checks `scene.fluidsim` exists — `ui_common.py:16`) with no "start here" primer; lifecycle actions enable on `runtime_ready` alone, not on whether a domain exists.
- Silent string truncation (`status[:80]`, reasons `[:5]`, blockers `[:4]`) drops the item that explains the failure, with no "…"/"+N more" — a localization and diagnosis trap.

### D8 — Visual/accessibility defects *(Visual, Onboarding)*
- **Inverted severity mapping:** `draw_status_message` maps `WARNING → "ERROR"` (red triangle) and `ERROR → "CANCEL"` (calmer circle-slash) — the higher severity gets the quieter glyph (`ui_common.py:100-105`). Severity is icon-only (no text prefix, no `alert` coloring).
- Icon overloading: `MOD_FLUIDSIM` = both "Initialize" and "Field Sources"; `FILE_REFRESH` = Reset / Recalculate / Refresh / Apply-Pending; `CHECKMARK` is both an OK *state* and an action verb.
- Two property-presentation grammars (Scene/Object split-layout vs Effects hand-built rows) make Effects look like a separate add-on.
- Icon-only recovery controls (runtime `FILE_REFRESH`) with no surfaced label; property `description=` coverage across 400+ fields unverified (that text is the only hover-help / screen-reader source).

### D9 — Over-exposed solver internals & a code IDE in the artist UI *(YAGNI, Visual)*
Solver internals presented as artist sliders: `pressure_iterations`/`tolerance`/`threshold`, `viscosity_iterations`/`tolerance`, the 16-property ST-FLIP "Spatiotemporal Processing" block (`seed`, `phase_eta`, `reference_mass`, `min_density`…), `leaf_capacity`, `count_stiffness`, `band_voxels`, the flat 10-prop surface-filter chain, per-stream whitewater multipliers, and **33 boolean per-attribute export toggles**. The custom-effects **Warp-code workspace** (compiler front-end, data-access permission matrix, Field API browser, compile/runtime evidence — the 46-call `effect_workspace.py` panel) is a developer console living in the artist UI.

### Strengths to preserve (consensus — do **not** regress these)
- **Single-authored draw layer** — keep the delegation mechanism; change only *what* each surface mirrors.
- **Kind-gated Physics polls** (`ui_physics.py:36-51`) — the most native-feeling surface; make it primary, not a mirror.
- **Role-based, poll-gated Object model** (`ui_object.py`).
- **`activity_snapshot()` state model** (`playback.py:71-142`) — legible transport model; the panel, not the model, is the problem.
- **The viewport HUD** (`viewport_overlay.py:57-83`) — just expose it.
- **Effect-stack UIList row** (enabled/mute/solo/name/cost — `effect_ui.py:215-227`) — the most scannable widget; template for others.
- **Runtime failure messages** (`warp_runtime.py:110-147`) — actionable full sentences; keep verbatim.
- **Pending/reinit honesty** (`ui_scene.py:216-229`) — tells the user when a change re-allocates.
- **Central status vocabulary** (`draw_status_message`) — keep the pattern; fix the inversion.
- **Preset + Guided Setup primitives** — promote them to the front door.

---

## 4. Design principles

1. **One home per task.** A setting has exactly one authoring home; other surfaces link to it, never re-render it.
2. **Progressive disclosure by expertise, not just by collapse.** Basic shows the ~10% that changes the result an artist sees; Advanced is one gate away; solver internals default to a Quality preset.
3. **Frequency drives prominence.** Per-iteration knobs are open and co-located; per-project knobs are deeper; per-install knobs live in Preferences.
4. **Controls and readouts are different modes.** Editable settings and solver telemetry never share a panel or a visual grammar.
5. **Stable, differentiated actions.** The primary verb is always in the same place; destructive verbs are visually distinct (`alert`, separated); state never reflows controls under the cursor.
6. **Native by default.** Adopt Blender's home, dropdown, entry-point, and layout conventions unless there's a documented reason not to.
7. **Redundant coding.** Severity, state, and affordance are conveyed by shape *and* text (and color where available) — never icon-alone.
8. **Migrate cleanly, once.** Relocating domain settings off the `Scene.fluidsim` singleton is in-scope and one-way; legacy files migrate forward on load, verified by test gates (see §9). Per-object role data already lives on the object and is unaffected.

---

## 5. Proposed architecture

### 5.1 Data model — the domain settings move onto the domain object

Today the large domain-settings group (`FLUIDSIM_PG_settings`) is a **scene singleton** (`Scene.fluidsim`, `properties.py`), while **per-object role data already lives on the object** (`Object.fluidsim` / `FLUIDSIM_PG_object` with `role` + collider/force params, `properties.py:438`, `:1628`; tag operators already write `obj.fluidsim.role`, e.g. `operator_domains/colliders.py`). So the object side of native storage is **already done** — only the domain-settings singleton needs to move.

The migration relocates domain settings from `Scene.fluidsim` to **a `PointerProperty` on the domain object**: `obj.fluidsim_domain` (`PointerProperty(type=FLUIDSIM_PG_settings)`).

> **Feasibility note (resolved):** Blender modifier *types* are C-level and cannot be registered from a pure-Python add-on, so a literal `obj.modifiers["FluidSim"]` (à la Mantaflow's compiled modifier) is **not** available to us. We therefore use a `PointerProperty` on the object (`obj.fluidsim_domain`) and surface it under the Physics Properties tab via a `poll`-gated panel. This is *functionally* the Mantaflow model (domain settings owned by the domain object, edited in the Physics tab) without claiming a real modifier datablock.

This is the root change the remaining native improvements depend on:

- **Physics Properties on the domain object becomes the real home** — not a mirror pointed back at the Scene. Select the domain → Physics tab → the actual `obj.fluidsim_domain` datablock. Functionally the Mantaflow layout.
- **Multiple independent domains per scene** become possible (impossible with a scene singleton). Live/transport state and telemetry key off the *active* domain object.
- **Drivers & animation** on domain settings get clean per-object datapaths (`obj.fluidsim_domain.…`) instead of the single scene-singleton path.
- **Presets become asset-browser assets** (§5.2) rather than a bespoke enum.
- **Roles are already object-owned** (§5.6) and need no migration — they already travel with the object through linking/appending.

### 5.1b Surface strategy — collapse four editors into three purpose-built roles

| Surface | New role | Change |
| --- | --- | --- |
| **Physics Properties** (on domain object) | **Primary authoring home** — the real datablock | Render `obj.fluidsim_domain` directly via a poll-gated Physics panel. Select domain → Physics tab; functionally the Mantaflow layout. |
| **Object Properties** | **Per-object roles** (emitter/sink/collider/force/field) | Native Type dropdown driving the per-object role group (§5.6). |
| **Scene Properties** | **Global session + output + bake summary** | Scene-level panels cover only what is genuinely global/session-scoped: active-domain transport, output/display, cache & bake, diagnostics. No per-domain physics lives here. |
| **3D-View N-panel** | **Lean HUD + controls, depth ≤ 3** | **Delete the `Full Settings` verbatim mirror** (`ui_view3d.py:39-91`, −52 panels). Keep `viewport_status` launcher + a curated "Live Tuning" shelf (§5.5) + "Open Settings" jump. |

### 5.2 Two-tier disclosure — Basic / Advanced + a Quality preset

Add two UI-only controls to the settings group:

- **`ui_mode` enum: `BASIC` (default) / `ADVANCED`** — a single toggle at the top of the domain physics home. Advanced-only panels `poll` on `ui_mode == 'ADVANCED'`. Nothing is removed; it's hidden by a poll gate.
- **`quality` enum: `DRAFT` / `PREVIEW` / `FINAL`** — drives the solver-internal trade-offs (pressure/viscosity iterations & tolerances, substep caps, surface cadence). Basic users pick a word; the raw knobs live in Advanced and, when `quality != CUSTOM`, are driven/greyed by the preset.

The **front door** is the promoted Setup panel: preset picker + Guided Setup, shown first, above the domain settings. With per-object storage, presets become **Blender asset-browser assets** (a domain object + settings marked as an asset, dragged in from the FluidSim asset catalog) rather than a bespoke `product_preset` enum — native, extensible by users, and shareable. The existing enum remains as a fallback quick-picker in the Setup header.

### 5.3 Panel map — Basic mode (the default surface)

Target: **~20–25 visible properties** (from 228). Everything else is one gate away.

```
PHYSICS PROPERTIES (domain object) ─ "FluidSim"
  ▸ Setup                 product_preset + Guided Setup (mode dialog)      [front door]
  ▸ Domain                domain_object, voxel_size, max_particles         (3)
  ▸ Quality               quality enum, substeps cap                       (2)
  ▸ Physics (Basic)       gravity/scene-gravity, flip_ratio                (2–3)
      └ [Advanced Physics]  ← poll: ui_mode==ADVANCED (the 12 sim_* panels)
  ▸ Surface               surface_enabled + Smoothing slider               (2)
      └ [Advanced]          full filter chain, resolution/cadence
  ▸ Whitewater            whitewater_enabled + Amount                      (2)
      └ [Advanced]          classification, lifetime, per-stream dynamics

OBJECT PROPERTIES (per selected object)
  ▸ FluidSim Role         Type dropdown + role params (~4)                 (§5.6)

SCENE PROPERTIES (global)
  ▸ Simulation Control    Transport | Lifecycle | Status (split — §5.4)
  ▸ Output & Display      view mode, show_overlay (HUD), Export enum       (3)
  ▸ Cache & Bake          path, range, Bake                                (~4)
  ▸ Diagnostics           all telemetry, collapsed by default              (§5.7)

3D-VIEW N-PANEL ("FluidSim", depth ≤ 3)
  ▸ FluidSim              status, runtime, Transport, "Open Settings"
  ▸ Live Tuning           voxel_size, flip_ratio, particle_display_fraction (§5.5)
```

### 5.4 Split the `scene_control` hub (D3)

Replace the one overloaded panel with three, each with a single job and a stable shape:

1. **Transport** — the per-run loop: primary Start/Stop Live (own full-width row, `scale_y≈1.3`), Step One, Step Many. Always present; Step is greyed (not hidden) with a `poll_message` when unavailable, so it never moves.
2. **Lifecycle** — Initialize / Release (allocation), Reset (re-run from frame 1). Release/Reset wrapped in an `alert=True` sub-row and separated from Transport; each carries a one-line consequence tooltip ("Release: free GPU memory, keep settings" / "Reset: re-run from frame 1").
3. **Status & Runtime** — read-only: state name, runtime readiness with a **labeled** recovery action ("Re-detect GPU" when not ready, not a bare icon), the error box, and a compact one-line `step_ms · substeps · CFL` strip (so live telemetry is visible without opening Diagnostics).

### 5.5 Live Tuning shelf (D5) + HUD (D4)

- New **"Live Tuning"** panel (open by default) in the N-panel and mirrored intent in the domain home: `voxel_size`, `particles_per_voxel`, `flip_ratio`, `particle_display_fraction` — the per-iteration knobs, co-located.
- **Expose `show_overlay`** in Output & Display (`layout.prop(fs, "show_overlay")`). The property already defaults on (`properties.py:1541`) and the HUD already renders particles/voxels/step_ms/substeps/CFL + pending + per-phase timers — it is simply drawn by no panel today, so this is a one-line fix that turns the built HUD on.

### 5.6 Object roles as a native dropdown, backed by per-object data (D6)

Role and its parameters are **already stored on the object** (`obj.fluidsim` / `FLUIDSIM_PG_object`, `properties.py:438`) — no migration needed here. The only change is presentation: the primary control becomes a native Type dropdown, `layout.prop(obj.fluidsim, "role")` (None / Emitter / Sink / Collider / Force / Field), matching Mantaflow's Type enum, instead of the current column of six tag operators (`ui_object.py:64-76`). Those six operators are retained *below* the dropdown, reframed as batch helpers ("Apply to Selected") for multi-object tagging. Role sub-panels keep their existing poll-gating.

### 5.7 Consolidate telemetry into Diagnostics (D4)

Move every read-only readout — capillary stats (`ui_scene.py:473-480`), whitewater Admission Status, the ~30 `stat_*` fields — out of control panels into the existing **Diagnostics** root (plus the compact strip in Status, §5.4). Render readouts through a dedicated helper: a `box()` with a "Readouts" header and a `grid_flow(columns=2, even_columns=True)` right-aligned key/value layout — never the same helper that draws editable props.

### 5.8 Custom Effects: split artist stack from developer workspace (D9)

- **Ship** the effect *stack* (the UIList: add/enable/mute/solo/duplicate/move, pick from library) — legitimate artist surface, already the best widget.
- **Gate** the Warp-code workspace (kernel editor, data-access permission matrix, Field API browser, compile/runtime evidence) behind an add-on Preference: *"Enable custom effect authoring (developer)."* The trust gate stays (it's a security control) but only appears when authoring is enabled. Add a **scoped "trust while editing this session"** mode so kernel authors don't re-click Review & Trust on every source edit.

---

## 6. Visual & interaction standards

Applied uniformly (mostly via the shared helpers in `ui_common.py`, so fixes propagate through the delegation layer):

- **Severity (fix the inversion, D8):** `ERROR` strictly heavier than `WARNING`; add a text prefix ("Warning:" / "Error:"); render error rows on an `alert=True` sub-row so color reinforces shape. Rename the mapping keys so `WARNING` no longer maps to an icon named `ERROR`.
- **Primary action:** one obvious primary verb per panel, its own full-width row, slightly larger (`scale_y`).
- **Destructive actions:** always `alert=True` and separated by `layout.separator()` from safe actions (applies to Transport/Lifecycle and Cache management's "Free Entire Cache").
- **Readouts ≠ controls:** dedicated readout helper (§5.7); never route `stat_*` through `draw_property_grid`.
- **Iconography:** reserve `CHECKMARK`/`ERROR`/`CANCEL`/`LOCKED` for *state*; give actions distinct verb icons (Reset ≠ Recalculate ≠ Refresh; Compile & Apply uses a verb icon, not the OK-state `CHECKMARK`).
- **One property grammar:** commit to `use_property_split=True` single-column for labeled numeric/enum rows everywhere (including Effects); reserve 2-col `grid_flow` for boolean toggle clusters and turn `use_property_split` **off** for those.
- **No silent truncation:** replace `[:N]` slicing on messages with `_draw_wrapped` (already exists, `ui_scene.py:37`); append "+N more" when a list is capped.
- **Chunk the filter list:** wrap each of the four surface passes (filter/dilate/erode/smooth) in its own `box()` + header.
- **Tame the warp_code wall:** move Data Access into its own `DEFAULT_CLOSED` child; render Evidence and Field-API as 2-col `grid_flow` with a header row.

---

## 7. Onboarding, discoverability & accessibility

- **Fix Guided Setup (D7):** give `setup_scene` an `invoke` → `invoke_props_dialog` so the "…" is honored and all 6 modes are chosen up front (or drop the ellipsis and render `mode` as an enum row before the run button).
- **Entry points (D6):** register a menu entry so the tool is reachable from muscle memory — append to `VIEW3D_MT_add` ("Fluid Domain (FluidSim)") and/or an `Object ▸ Quick Effects`-style item; ensure lifecycle operators carry clean `bl_label`s for F3 search (mostly already true).
- **Empty state:** in Setup, when no fluid objects exist, draw a one-line primer ("Select a mesh object, then run Guided Setup"); gate lifecycle actions on "a domain exists," not only `runtime_ready`, or show an inline "Run Guided Setup first" hint.
- **Labeled recovery:** the runtime-error row shows a text button ("Re-detect GPU"), not a bare `FILE_REFRESH` glyph.
- **Descriptions audit:** verify every one of the 400+ PropertyGroup fields defines `description=` (the only hover-help / screen-reader text); add `text`/`bl_description` to icon-only operators (mute/solo/move/refresh).
- **Trust copy:** reword artist-facing trust text away from "SHA-256/digest" jargon ("trusts this exact code; re-editing re-locks it") and keep the hash in the tooltip.

---

## 8. The "diet" — what moves, and the target budget

Nothing is deleted from capability; items move behind a gate/preset/preferences or to the Diagnostics home.

| Item | Today | After |
| --- | --- | --- |
| 3D-View `Full Settings` mirror (52 panels) | verbatim 5-deep clone | **removed**; N-panel = HUD + Live Tuning |
| 12 `sim_*` deep-config panels | flat, `DEFAULT_CLOSED` | behind **Advanced Physics** gate; driven by **Quality** |
| 16-prop ST-FLIP block | artist sliders | Advanced/expert only; `spacetime_enabled` checkbox in Advanced |
| 10-prop surface filter chain | flat list | one **Smoothing** slider (Basic); full chain in Advanced, chunked |
| 33 per-attribute export toggles | 33 checkboxes | one **Export Attributes** enum (Minimal/Standard/Everything); toggles in Advanced |
| Capacity & Safety (`leaf_capacity`…) | artist knobs | auto-derived; read-only "estimated memory" line; knobs in Advanced |
| Whitewater per-stream multipliers | 3 boxes of sliders | on/off + **Amount** (Basic); multipliers in Advanced |
| Warp-code IDE | in artist UI | behind **developer** Preference |
| ~30 `stat_*` readbacks | scattered/interleaved | consolidated in **Diagnostics** + Status strip |

**Target default surface:** ~20–25 visible properties (≈10% of today), with 100% of capability reachable via Basic→Advanced, Quality, presets, or Preferences.

---

## 9. Compatibility & migration

**Scope of the migration:** only the domain-settings singleton moves — `Scene.fluidsim` (`FLUIDSIM_PG_settings`) → `obj.fluidsim_domain` on the domain object. Per-object role data (`obj.fluidsim`) already lives on objects and is **not** migrated. This is a real, tested, one-way migration.

- **New schema field.** Add `schema_version: IntProperty(default=0)` to `FLUIDSIM_PG_settings`. `0` = pre-migration (data still authored on `Scene.fluidsim`); the current target version (e.g. `1`) stamped on the group once it lives on `obj.fluidsim_domain` marks "already migrated." The migration only acts on groups it finds at version `0` on the scene, making it idempotent.
- **Forward migration on load.** A `@persistent load_post` handler (registered alongside the existing `load_post` handlers in `operator_domains/sources.py` / `effects.py`) runs on file open: if `scene.fluidsim` holds a populated, unmigrated (`schema_version == 0`) settings group, it identifies the domain object (the existing `fs.domain_object` pointer, `ui_physics.py:43`; if unset, create/designate one), copies every field onto `obj.fluidsim_domain`, stamps `schema_version = 1`, and clears the scene group. Today's model is one implicit domain per scene, so the mapping is unambiguous.
- **One-way, no dual store.** We do not keep `Scene.fluidsim` alive in parallel (a non-goal). The `schema_version` guard prevents re-running on an already-migrated file (including one re-saved and reopened).
- **Round-trip test is a gate.** A migration acceptance test loads representative pre-migration `.blend` fixtures headless, runs the handler, and asserts every domain setting landed on `obj.fluidsim_domain` with identical values, that `schema_version == 1`, and that a second load is a no-op (idempotency). Part of the production gate before Phase 3 ships.
- **Solver untouched.** The runtime reads domain settings from `obj.fluidsim_domain` instead of `scene.fluidsim`; values and semantics are unchanged, so simulation output is identical pre/post migration (verified by a golden-frame comparison).
- **Quality preset** maps to existing solver-internal properties; when `quality == CUSTOM` the raw Advanced knobs are live, otherwise driven/greyed. No new solver parameters.
- **Delegation layer.** The single-`draw()` mechanism is preserved for the visual fixes (severity, readout helper, grammar land once). What changes is the *data source* each domain panel reads — `obj.fluidsim_domain`, not `scene.fluidsim` — and the deletion of the N-panel full mirror.

---

## 10. Phased roadmap

Ordered by value-to-effort; each phase is independently shippable and testable (headless + `--include-gui`).

- **Phase 0 — Quick wins (hours, high payoff):** expose `show_overlay` (D4); fix severity inversion (D8); fix `Guided Setup` invoke/ellipsis (D7); labeled runtime recovery button; remove silent truncation. Small, isolated, immediately felt.
- **Phase 1 — Split the hub (D3):** Transport / Lifecycle / Status+strip; destructive-action differentiation. Highest daily-loop impact.
- **Phase 2 — Basic/Advanced + Quality (D2):** add `ui_mode` + `quality`; poll-gate the 12 `sim_*` panels and the other Advanced clusters; promote Setup/preset front door. The core reduction.
- **Phase 3 — Data-model migration (§9), the pivot:** move domain settings from `Scene.fluidsim` to `obj.fluidsim_domain` (roles already live on the object — no role migration); add the `schema_version` field; write the `load_post` forward migration + round-trip/idempotency and golden-frame gates; point the solver at `obj.fluidsim_domain`. Largest change; unlocks Phases 4–5. Ship behind the migration gate.
- **Phase 4 — Native home & surfaces (D1/D6):** Physics-on-domain renders `obj.fluidsim_domain`; delete the N-panel `Full Settings` mirror; add Live Tuning shelf; demote Scene to global summary; role Type dropdown; `VIEW3D_MT_add` entry; presets as asset-browser assets; multiple-domains support; property-grammar unification; description/tooltip audit.
- **Phase 5 — Telemetry & Effects (D4/D9):** consolidate `stat_*` into Diagnostics + readout helper; chunk filter list; developer-gate the Warp workspace; scoped effect trust.

> **Sequencing note:** Phases 0–2 are storage-agnostic and ship first for immediate value. Phase 3 (migration) is the gate everything native depends on; treat it as its own reviewed, test-gated change with the migration fixtures in place before Phase 4 begins.

Each phase must pass `python -m pytest -q`, `python tools/run_production_gate.py --include-gui`, and `python tools/validate_policy_drift_reports.py` after a GUI pass, per `AGENTS.md`.

---

## 11. Success criteria

- Default visible property count on a fresh domain ≈ 20–25 (measurable via a panel-render test).
- N-panel nesting depth ≤ 3; no verbatim full-tree mirror registered.
- A new user reaches first splash via: select mesh → Guided Setup (mode chosen) → Play, with a visible primer at the empty state.
- Transport/Step buttons hold a fixed position across all activity states; destructive actions are `alert`-styled and separated.
- The HUD is toggleable from the UI and on by default during live preview.
- No control panel renders a `stat_*` readback; severity is conveyed by shape + text (+ color).
- No capability removed: every property reachable via Basic→Advanced/Quality/preset/Preferences.
- **Domain settings live on the domain object** (Physics tab reads the real datablock); a scene supports **≥ 2 independent domains**; a legacy `.blend` migrates forward on load with identical values (round-trip test) and identical simulation output (golden-frame test).

---

## 12. Open decisions (for approval)

- **O1 — Authoritative home: RESOLVED.** Migration is in-scope, so settings move onto the domain-object modifier and Physics-on-domain becomes the real home (§5.1, §9). This is the full-fidelity native path; it also unlocks multiple domains, per-object roles, drivers, and asset-browser presets. The former low-risk "presented-primary over `Scene.fluidsim`" compromise is dropped.
- **O2 — Quality vs CUSTOM:** should `quality` hard-drive (grey out) the Advanced solver knobs, or only seed them on change? **Recommendation: seed-on-change + a `CUSTOM` value that unlocks; avoids trapping experts.**
- **O3 — N-panel scope:** HUD + Transport + Live Tuning only, or also a short curated "favorites"? **Recommendation: start minimal (§5.3); add favorites only if requested.**
- **O4 — Menu entry placement:** `VIEW3D_MT_add` submenu vs a dedicated top-level "FluidSim" menu vs Object ▸ Quick Effects. **Recommendation: `VIEW3D_MT_add` submenu (most native).**

---

## Appendix A — Council members & core theses

| Lens | Core thesis |
| --- | --- |
| Blender-native UX | Home belongs in Physics-on-domain; role → Type dropdown; delete sidebar clone; add Add-menu entry. |
| TD workflow | Models are strong; expose the HUD, un-bury `flip_ratio`, surface live telemetry, split the hub. |
| Information architecture | Triplication forces one hierarchy onto three contexts; add basic→advanced gradient; pull telemetry to one home. |
| Visual & interaction | Fix inverted severity; primary/destructive prominence; readouts ≠ controls; one grammar; de-overload icons. |
| Onboarding & accessibility | Guided Setup ellipsis bug; no discovery entry points; empty-state primer; no silent truncation; description audit. |
| YAGNI / reduction | Reduction machinery already exists but is buried; gate solver internals behind Quality; ~20–25 default props; dev-gate the code IDE. |

## Appendix B — Primary evidence index

`fluidSim_warp_blender51/`:
- `blender/ui_scene.py` — hub 135-201; runtime stats 232-259; surface-tension stats 473-480; sim fan 385-618; filters 681-697; setup 273-283; cache mgmt 1053-1066.
- `blender/ui_common.py` — poll 16-18; severity map 98-106; grid helper 120-131; wrapped 37.
- `blender/ui_physics.py` — kind classify 36-51; mirrors 67-118.
- `blender/ui_view3d.py` — verbatim re-parent mirror 17-91.
- `blender/ui_object.py` — role model 47-282; tag operators 64-76.
- `panels.py` (package root) — launcher 34; open_scene_properties 12-31.
- `blender/effect_ui.py` — UIList row 215-227; status/trust 152-188.
- `blender/effect_workspace.py` — warp_code 131-206 (46 calls).
- `blender/viewport_overlay.py` — HUD 37, 57-83.
- `operator_domains/source_roles.py` — setup_scene 61-104 (no invoke).
- `runtime/warp_runtime.py` — probe messages 110-147.
- `blender/playback.py` — activity_snapshot 71-142.
- `properties.py` — role enum 438; show_overlay def 1541; domain 764.
- `preferences.py` — device/runtime 103-149.
