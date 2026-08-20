# FluidSim Properties UI and Custom Effects Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dense FluidSim 3D View sidebar with a field-complete Blender Properties workflow and integrate Custom Effects v2 as a coherent artist/developer experience.

**Architecture:** Add focused common, Scene Properties, and Object Properties UI modules while reducing `panels.py` to a compact viewport launcher and registration aggregator. Retain existing operators, RNA, solver, cache, trust, and effect execution contracts; relocate and reorganize only public presentation and supporting validation.

**Tech Stack:** Blender 5.2 Python API, Python 3.13, NVIDIA Warp 1.16.0+cu12, pytest, Blender background validation, extension build tooling.

**Spec:** `docs/superpowers/specs/2026-08-18-properties-ui-and-custom-effects-design.md`

## Global Constraints

- Base commit is `98a596dd52d703d438160fbb4ca66ef7a66e4fdf`, release `1.0.251`.
- Validation must execute `/mnt/data/blender-5.2.0-linux-x64/blender` and the supplied Warp 1.16.0+cu12 wheel extraction.
- Do not bundle Blender or Warp runtimes.
- Preserve RNA property names, operator IDs, cache/checkpoint/effect signatures, trust policy, and solver behavior.
- `activity_snapshot`, `pending_presentation`, and `session().owns_scene` remain authoritative lifecycle sources.
- Legacy snippet panels stay unregistered; compatibility operators remain migration adapters.
- The release target is `1.0.252`, Linux x64 only in this environment.

---

### Task 1: Define and verify the public panel contract

**Files:**
- Create: `blender/ui_common.py`
- Create: `tests/test_properties_ui_contract.py`
- Modify: `tests/test_effect_ui_source.py`

**Interfaces:**
- Produces: `FluidSimScenePanel`, `FluidSimObjectPanel`, `FluidSimView3DPanel`
- Produces: `activity_ui_state(scene) -> dict[str, object]`
- Produces: `active_object_role(context) -> str`
- Produces: `canonical_region_kind(obj) -> str`
- Produces: `draw_status_message(layout, text: str, severity: str = "INFO") -> None`
- Consumes: `blender.playback.activity_snapshot`, `blender.scene_bridge.session`, `blender.pending_settings.pending_presentation`

- [ ] **Step 1: Add failing metadata and ownership tests**

Create tests that import panel source through AST and assert the future roots, orders, spaces, contexts, parent IDs, and default-closed states. Assert that `live`, `cache_playback_enabled`, `show_surface`, `show_particles`, `particles_per_voxel` in Height Fill, and `verbose` in Preferences are not drawn publicly.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
python -m pytest tests/test_properties_ui_contract.py tests/test_effect_ui_source.py -q
```

Expected: failures because Properties panel modules and metadata do not yet exist.

- [ ] **Step 3: Implement common bases and derived-state helpers**

`FluidSimScenePanel` sets `PROPERTIES/WINDOW/scene`; `FluidSimObjectPanel` sets `PROPERTIES/WINDOW/object`; `FluidSimView3DPanel` sets `VIEW_3D/UI/FluidSim`. `activity_ui_state()` normalizes the existing snapshot to an ID, label, icon, and ownership flags without mutating RNA.

- [ ] **Step 4: Add role and canonical-region helpers**

Use canonical `obj.fluidsim.role`, `source_region_kind`, `sink_region_kind`, and `field_region_kind`. Legacy aliases are not consulted except by existing migration code.

- [ ] **Step 5: Run focused tests**

```bash
python -m pytest tests/test_properties_ui_contract.py tests/test_effect_ui_source.py -q
```

Expected: common-helper tests pass; panel inventory tests remain failing until Tasks 2–5.

- [ ] **Step 6: Commit**

```bash
git add blender/ui_common.py tests/test_properties_ui_contract.py tests/test_effect_ui_source.py
git commit -m "test(ui): define Properties workflow contract"
```

---

### Task 2: Implement Scene Properties workflow panels

**Files:**
- Create: `blender/ui_scene.py`
- Modify: `panels.py`
- Modify: `__init__.py`
- Test: `tests/test_properties_ui_contract.py`

**Interfaces:**
- Produces: `SCENE_PANEL_CLASSES: tuple[type, ...]`
- Produces root panel IDs: `FLUIDSIM_PT_scene_control`, `FLUIDSIM_PT_scene_setup`, `FLUIDSIM_PT_scene_domain`, `FLUIDSIM_PT_scene_height`, `FLUIDSIM_PT_scene_simulation`, `FLUIDSIM_PT_scene_surface`, `FLUIDSIM_PT_scene_whitewater`, `FLUIDSIM_PT_scene_outputs`, `FLUIDSIM_PT_scene_cache`, `FLUIDSIM_PT_scene_diagnostics`
- Consumes: Task 1 common bases/helpers and existing services/operators

- [ ] **Step 1: Extend failing tests for the exact root and child inventory**

Assert Scene root `bl_order` values 0, 10, 20, 30, 40, 60, 70, 80, 90, and 100, reserving 50 for Custom Effects. Assert child relationships and conditions for pending, statistics, capacity, physics, surface, whitewater, outputs, cache, and diagnostics.

- [ ] **Step 2: Run the Scene contract tests and verify failure**

```bash
python -m pytest tests/test_properties_ui_contract.py -q
```

- [ ] **Step 3: Implement Session, Setup, Domain, Height Fill, and Physics panels**

Draw only fields assigned by the spec. Add Surface Tension as a Physics child with Phase 5 authored controls and capillary diagnostics. Use existing operators and capacity helpers; do not duplicate lifecycle policy.

- [ ] **Step 4: Implement Surface, Whitewater, Outputs, Cache, and Diagnostics panels**

Add Phase 4 Classification and typed statistics; include separate whitewater materials and output mode; preserve cache range/bake range separation and existing destructive operator dialogs.

- [ ] **Step 5: Replace dense panel registration with Scene module registration**

`panels.register()` registers the compact View3D panel only; root `__init__.py` registers/unregisters `ui_scene` adjacent to `panels`. The previous dense classes are no longer in any `CLASSES` tuple.

- [ ] **Step 6: Run contract and existing source tests**

```bash
python -m pytest tests/test_properties_ui_contract.py tests/test_phase2_assets.py tests/test_runtime_integration_source.py -q
```

Expected: all pass after updating assertions to the Properties workflow without weakening runtime coverage.

- [ ] **Step 7: Commit**

```bash
git add blender/ui_scene.py panels.py __init__.py tests
git commit -m "feat(ui): move scene workflow into Properties"
```

---

### Task 3: Implement Object Properties role workflow

**Files:**
- Create: `blender/ui_object.py`
- Modify: `__init__.py`
- Test: `tests/test_properties_ui_contract.py`
- Create: `tests/test_object_properties_ui.py`

**Interfaces:**
- Produces: `OBJECT_PANEL_CLASSES: tuple[type, ...]`
- Produces: `role_poll(context, *roles: str) -> bool`
- Produces: `volume_grid_poll(context) -> bool`
- Consumes: Task 1 canonical role/region helpers and existing role operators

- [ ] **Step 1: Write failing role-matrix and field-ownership tests**

Use a lightweight fake context to cover NONE, SOURCE, SINK, COLLIDER, FORCE, and FIELD, including analytic, mesh, points, and volume-grid region variants.

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/test_object_properties_ui.py -q
```

- [ ] **Step 3: Implement role root and diagnostics**

Draw selected-scope role actions, active-object enabled state, migration action, and status fields without mutating untagged objects.

- [ ] **Step 4: Implement conditional region/source/collider/force/field/volume panels**

Hide inapplicable payload controls. Source Lifespan becomes a child of Liquid Source. Field Blending and Field Source are separate role-specific panels.

- [ ] **Step 5: Register Object panels and run tests**

```bash
python -m pytest tests/test_object_properties_ui.py tests/test_properties_ui_contract.py -q
```

- [ ] **Step 6: Commit**

```bash
git add blender/ui_object.py __init__.py tests/test_object_properties_ui.py tests/test_properties_ui_contract.py
git commit -m "feat(ui): add role-aware Object Properties workflow"
```

---

### Task 4: Integrate Custom Effects and Warp Code into Scene Properties

**Files:**
- Modify: `blender/effect_ui.py`
- Modify: `blender/effect_workspace.py`
- Modify: `blender/ui_scene.py`
- Modify: `tests/test_effect_ui_source.py`
- Create: `tests/test_effect_properties_ui.py`

**Interfaces:**
- Produces: `FLUIDSIM_PT_custom_effects` as Scene Properties root at `bl_order = 50`
- Produces: `FLUIDSIM_PT_effect_packages` child panel
- Produces: `FLUIDSIM_PT_effect_warp_code` child panel visible for any active kernel
- Consumes: Custom Effects v2 operators, stack model, trust, compiler, fingerprint, and package library unchanged

- [ ] **Step 1: Write failing Custom Effects placement and workflow tests**

Assert PROPERTIES/WINDOW/scene metadata, absence of public `artist_mode`, four artist workflow groups, package actions in a closed child, Warp Code visible independent of `artist_mode`, and no legacy snippet panel registration.

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/test_effect_properties_ui.py tests/test_effect_ui_source.py -q
```

- [ ] **Step 3: Restructure Custom Effects draw flow**

Keep the existing UIList and operators. Draw stack/library actions first, selected artist parameters second, Test/Apply/Revert/Preview third, and state/trust/divergence last. Move import/export/update/rename/delete/reveal operations to Package & Library.

- [ ] **Step 4: Relocate and ungate Warp Code**

Set Scene Properties metadata and parent ID. Poll for an active effect with kernels, not `not stack.artist_mode`. Retain Text Editor, generated-source, field API, trust/expert, and profiling behavior.

- [ ] **Step 5: Run Custom Effects unit and source tests**

```bash
python -m pytest tests/test_effect_properties_ui.py tests/test_effect_ui_source.py tests/test_effect_workspace_source.py tests/test_effect_source.py -q
```

- [ ] **Step 6: Commit**

```bash
git add blender/effect_ui.py blender/effect_workspace.py blender/ui_scene.py tests
git commit -m "feat(ui): integrate Custom Effects with scene workflow"
```

---

### Task 5: Add compact viewport launcher and simplify Preferences

**Files:**
- Modify: `panels.py`
- Modify: `preferences.py`
- Modify: `tests/test_properties_ui_contract.py`
- Create: `tests/test_viewport_launcher_ui.py`

**Interfaces:**
- Produces: `FLUIDSIM_PT_viewport_status`
- Produces: `FLUIDSIM_OT_open_scene_properties`
- Consumes: Task 1 activity helper and current runtime/session operators

- [ ] **Step 1: Write failing compact-launcher and Preferences tests**

Assert the View3D panel contains status and lifecycle actions but no solver, output, cache, object, or effect settings. Assert Preferences draws device/runtime and conditional override but not `verbose`.

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/test_viewport_launcher_ui.py tests/test_properties_ui_contract.py -q
```

- [ ] **Step 3: Implement launcher and Scene Properties shortcut**

The operator finds a Properties area, changes context to `SCENE`, and reports a concise fallback message when none exists. The panel uses derived activity and runtime state.

- [ ] **Step 4: Simplify Preferences draw**

Use compact Runtime and conditional Developer boxes. Keep the property registered but hidden.

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest tests/test_viewport_launcher_ui.py tests/test_properties_ui_contract.py -q
git add panels.py preferences.py tests
git commit -m "feat(ui): add compact viewport launcher and runtime preferences"
```

---

### Task 6: Add supplied-Blender UI validation and update documentation

**Files:**
- Create: `release/validate_properties_ui_blender.py`
- Create: `tests/test_properties_ui_release_source.py`
- Modify: `release/smoke_test_blender.py`
- Modify: `docs/user_guide.md`
- Modify: `docs/custom_effects.md`
- Modify: `docs/technical_report.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Produces: JSON report with panel metadata, registration lifecycle, representative draw paths, role/effect polls, save/reopen state, and compact-launcher checks
- Consumes: supplied Blender and the installed/source extension module

- [ ] **Step 1: Write failing release-source tests**

Assert the Blender validator is wired to check Scene/Object/Custom Effects/Warp Code panels and that smoke testing requires the new UI inventory.

- [ ] **Step 2: Implement background Blender validator**

Register the source module in factory startup, inspect registered class metadata, create representative scene/object/effect data, exercise poll/draw with a recording layout, save/reopen, unregister/re-register, and write strict atomic JSON.

- [ ] **Step 3: Update user documentation**

Replace the seven N-panel groups with the Scene/Object/Preferences hierarchy, describe the compact launcher, and update Custom Effects navigation and package maintenance location.

- [ ] **Step 4: Run source and supplied-Blender validation**

```bash
python -m pytest tests/test_properties_ui_release_source.py -q
PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external \
FLUIDSIM_EXTERNAL_WARP_ROOT=/mnt/data/warp_lang_1_16_external \
/mnt/data/blender-5.2.0-linux-x64/blender \
  --background --factory-startup --offline-mode --python-use-system-env \
  --python-exit-code 1 --python release/validate_properties_ui_blender.py -- \
  --module fluidsim_ui_release \
  --work-dir /mnt/data/fluidsim_ui_blender_validation \
  --report-json /mnt/data/fluidsim_ui_blender_validation.json
```

- [ ] **Step 5: Commit**

```bash
git add release tests docs CHANGELOG.md
git commit -m "test(ui): validate Properties workflow in Blender"
```

---

### Task 7: Deep regression review, release build, and final artifacts

**Files:**
- Modify: `version.py`
- Modify: `blender_manifest.toml`
- Modify: `CHANGELOG.md`
- Modify: release metadata/tests as defects require

**Interfaces:**
- Produces: release `1.0.252`
- Produces: source ZIP, Linux x64 extension ZIP, validation bundle/report, release summary, Git bundle, command ledger, and SHA-256 file
- Consumes: all prior tasks

- [ ] **Step 1: Run full pure-Python and supplied-Warp suites**

```bash
python -m pytest -q
PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external python -m pytest -q
python -m compileall -q -f .
git diff --check
```

Fix root causes without weakening tests.

- [ ] **Step 2: Run existing Custom Effects real-Warp gate**

Use the repository's current supplied-Warp validator with CPU device and verify built-in positive and negative fixtures still behave identically.

- [ ] **Step 3: Run supplied-Blender UI, Custom Effects, Phase 5, and example regressions**

Execute the new Properties UI validator plus existing Custom Effects, Phase 5, Phase 4/3, examples, and F12 evidence validators that are available on CPU/Blender.

- [ ] **Step 4: Perform source review and fix defects**

Review panel poll conditions, duplicate property ownership, lifecycle registration, failed draw isolation, operator scopes, destructive-action visibility, docs, package contents, and release scripts. Add a regression test for every defect fixed.

- [ ] **Step 5: Bump release identity to 1.0.252**

Update `version.py`, manifest, changelog, docs version references, and expected package names. Run identity tests.

- [ ] **Step 6: Build and install the extension with supplied Blender/Warp**

```bash
PYTHONPATH=/mnt/data:/mnt/data/warp_lang_1_16_external \
FLUIDSIM_EXTERNAL_WARP_ROOT=/mnt/data/warp_lang_1_16_external \
python release/build_extension.py \
  --root "$PWD" \
  --blender /mnt/data/blender-5.2.0-linux-x64/blender \
  --output /mnt/data/fluidsim_ui_final_dist \
  --repository /mnt/data/fluidsim-ui-final-ci \
  --skip-lint \
  --smoke-platform linux-x64 \
  --json-report /mnt/data/fluidsim_ui_final_dist/build-report.json
```

Validate the built archive, run installed smoke twice, and uninstall cleanly.

- [ ] **Step 7: Generate final artifacts and checksums**

Create no-runtime source ZIP, validation bundle, machine-readable validation and release summary, handoff, Git bundle, command ledger, and SHA-256 list. Scan every archive for Blender/Warp runtime payloads and run ZIP CRC checks.

- [ ] **Step 8: Verify completion before claiming success**

Run the verification-before-completion workflow, confirm clean worktree and exact commit, and document CUDA/Windows/Ruff gates that remain genuinely unavailable.

- [ ] **Step 9: Commit release**

```bash
git add .
git commit -m "release: ship Properties workflow 1.0.252"
```
