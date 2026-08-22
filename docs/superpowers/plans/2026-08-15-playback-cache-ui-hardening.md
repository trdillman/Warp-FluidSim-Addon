# FluidSim Playback, Cache, and UI Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore responsive live playback, make Production Bake v2 the sole primary cache-capture workflow, enforce exclusive solver ownership, and replace the confusing legacy cache UI with a compact authoritative activity display.

**Architecture:** Keep the established deep-module boundaries. `playback.py` owns preview/cache-playback policy, `scene_bridge.py` owns stepping, `output_publication.py` owns Blender output publication, and `production_bake.py` owns transactional bake state plus the active-controller registry. `operators.py` and `panels.py` remain adapters that consume authoritative snapshots rather than reconstructing workflow state.

**Tech Stack:** Blender 5.1 Python API, Warp CUDA runtime, NumPy cache payloads, pytest contract tests, Blender background/GUI acceptance scripts.

## Global Constraints

- Keep simulation GPU-resident and preserve interactive viewport playback.
- Preserve exact catch-up, desynchronization, rewind, cache-playback error, and cancellation semantics.
- Production Cache Bake bypasses Preview Playback handlers and Output Publication and writes full authoritative cache data.
- Do not claim bitwise CUDA repeatability.
- Preserve cache-v1 read compatibility, but remove implicit cache-v1 live recording and its visible controls.
- Blender UI draw callbacks must remain read-only and must not perform solver work or cache I/O.
- Preserve the Blender 5.1 detached point-cloud prewarm path; do not resize attached point-cloud data.
- Do not modify unrelated dirty work or archived experimental branches/worktrees.

---

### Task 1: Separate live display attributes from production cache attributes

**Files:**
- Modify: `fluidSim_warp_blender51/blender/output_publication.py`
- Test: `tests/test_frame_output.py`
- Test: `tests/test_performance_hotpath_contracts.py`

**Interfaces:**
- Consumes: `HostFrame.surface_mesh`, `_surface_values(solver, mode, verts, host=host)`.
- Produces: `prepare_live_payload(scene, host, publish_surface=True) -> PublicationPayload` whose `surface_attributes` contains only the selected live display channel, while cache writers continue using `HostFrame.output_attributes(...)`.

- [ ] **Step 1: Write failing live-payload tests**

Add a host spy whose `output_attributes()` raises if called, then assert ordinary live publication succeeds and contains only the selected display attribute:

```python
def test_live_payload_does_not_capture_production_attribute_inventory(monkeypatch):
    host = LiveHostSpy(surface_mode="NONE")
    payload = output_publication.prepare_live_payload(scene, host)
    assert payload.surface_attributes == {}
    assert host.output_attribute_calls == []
```

Add a second test selecting `VELOCITY` and assert the display value is retained without calling the production inventory method.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_frame_output.py tests/test_performance_hotpath_contracts.py
```

Expected: the new spy test fails because `prepare_live_payload()` currently calls `host.output_attributes(..., "surface")`.

- [ ] **Step 3: Implement minimal live adaptation**

Replace the live production-inventory call with a presentation-only map:

```python
surface_values = _surface_values(
    solver, fs.surface_attribute_mode, verts, host=host)
surface_attributes = _attribute_map(
    fs.surface_attribute_mode, surface_values)
```

Do not alter `_frame_payload()` in `io/cache.py`; it remains the full production capture path.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2 and expect all selected tests to pass.

- [ ] **Step 5: Commit the isolated performance fix**

```powershell
git add fluidSim_warp_blender51/blender/output_publication.py tests/test_frame_output.py tests/test_performance_hotpath_contracts.py
git commit -m "perf(viewport): avoid production attributes during live display"
```

---

### Task 2: Publish only the final recovered timeline frame and remove implicit live cache writes

**Files:**
- Modify: `fluidSim_warp_blender51/blender/playback.py`
- Modify: `fluidSim_warp_blender51/blender/scene_bridge.py`
- Test: `tests/test_scene_bridge_cache_playback.py`

**Interfaces:**
- Consumes: `scene_bridge.step_frame(..., publish_outputs: bool)`.
- Produces: one viewport publication per Blender frame-change event; all solver recovery frames still execute in order.
- Produces: `SolverSession.step()` with no implicit `cache.write_frame()` side effect.

- [ ] **Step 1: Write failing catch-up and capture tests**

Record every `step_frame()` call and assert only the final recovery call publishes:

```python
assert calls == [
    {"timeline_frame": 8, "publish_outputs": False},
    {"timeline_frame": 9, "publish_outputs": False},
    {"timeline_frame": 10, "publish_outputs": True},
]
```

Replace the existing legacy-write expectation with a regression test that sets `fs.cache_enabled = True`, steps once, and asserts `cache.write_frame` was not called.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
python -m pytest -q tests/test_scene_bridge_cache_playback.py
```

Expected: intermediate calls publish by default and the legacy cache writer is invoked.

- [ ] **Step 3: Implement final-frame-only publication**

Update the recovery loop:

```python
for offset in range(gap - 1, -1, -1):
    sess.sync_scene(scene, depsgraph)
    bridge.step_frame(
        scene,
        sess,
        refresh_pending=False,
        timeline_frame=frame - offset,
        publish_outputs=(offset == 0),
    )
```

Remove the `if write_cache and fs.cache_enabled:` block from `SolverSession.step()`. Retain the keyword for compatibility until all direct callers are migrated, but document it as ignored/deprecated or remove it only if the call inventory proves safe.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command and expect it to pass.

- [ ] **Step 5: Commit the timeline/cache seam**

```powershell
git add fluidSim_warp_blender51/blender/playback.py fluidSim_warp_blender51/blender/scene_bridge.py tests/test_scene_bridge_cache_playback.py
git commit -m "fix(playback): publish only final recovered frame"
```

---

### Task 3: Move authoritative active-bake ownership into the production-bake module

**Files:**
- Modify: `fluidSim_warp_blender51/blender/production_bake.py`
- Modify: `fluidSim_warp_blender51/operators.py`
- Modify: `fluidSim_warp_blender51/blender/playback.py`
- Test: `tests/test_scene_bridge_cache_playback.py`
- Test: `tests/test_operator_panel_behavior.py`

**Interfaces:**
- Produces: `register_active_bake(scene, controller) -> None`.
- Produces: `active_bake(scene) -> BakeController | None`.
- Produces: `clear_active_bake(scene, controller=None) -> None`.
- Produces: `bake_is_active(scene) -> bool` for statuses `INITIALIZING`, `RUNNING`, or `CANCELLING`.

- [ ] **Step 1: Write failing registry and conflict tests**

Cover registration, wrong-controller clearing, terminal-state clearing, and mode conflict rejection:

```python
production_bake.register_active_bake(scene, controller)
assert production_bake.active_bake(scene) is controller
with pytest.raises(RuntimeError, match="Production Bake"):
    playback.set_live_enabled(scene, True)
```

Also assert enabling Cache Playback while a bake is active does not release the shared solver session.

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
python -m pytest -q tests/test_scene_bridge_cache_playback.py tests/test_operator_panel_behavior.py
```

Expected: registry functions are absent and playback transitions do not reject the active bake.

- [ ] **Step 3: Implement the deep registry and adapter migration**

Use scene identity as the key and keep registry mutation in `production_bake.py`:

```python
_ACTIVE_BAKES: dict[int, BakeController] = {}

def _scene_key(scene) -> int:
    pointer = getattr(scene, "as_pointer", None)
    return int(pointer()) if callable(pointer) else id(scene)
```

Replace `_ACTIVE_BAKE` in `operators.py` with the registry functions. Make `set_live_enabled()` and `set_cache_playback_enabled()` fail before allocating or releasing a solver, set an actionable status, and preserve the current state.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command and expect it to pass.

- [ ] **Step 5: Commit ownership hardening**

```powershell
git add fluidSim_warp_blender51/blender/production_bake.py fluidSim_warp_blender51/blender/playback.py fluidSim_warp_blender51/operators.py tests/test_scene_bridge_cache_playback.py tests/test_operator_panel_behavior.py
git commit -m "fix(bake): centralize active production ownership"
```

---

### Task 4: Add an authoritative activity snapshot and simplify Cache & Bake UI

**Files:**
- Modify: `fluidSim_warp_blender51/blender/playback.py`
- Modify: `fluidSim_warp_blender51/operators.py`
- Modify: `fluidSim_warp_blender51/panels.py`
- Modify: `fluidSim_warp_blender51/properties.py`
- Test: `tests/test_operator_panel_behavior.py`

**Interfaces:**
- Produces: immutable `ActivitySnapshot(state, label, detail, icon, busy, progress)`.
- Produces: `activity_snapshot(scene, *, is_animation_playing: bool = False) -> ActivitySnapshot` with states `IDLE`, `LIVE_READY`, `LIVE_PLAYING`, `CACHE_PLAYBACK`, `BAKING`, `CANCELLING`, `COMPLETE`, and `FAILED`. The panel passes `context.screen.is_animation_playing`; the deep function does not inspect UI areas.
- Produces: operators `fluidsim.cache_playback_start` and `fluidsim.cache_playback_stop`; raw RNA checkbox remains internal.

- [ ] **Step 1: Write failing activity precedence and panel-contract tests**

Assert bake activity outranks cache/live state, cache playback outranks live-ready, terminal bake state remains visible, and the panel source does not expose legacy controls:

```python
assert 'text="Record v1 During Live Playback (Legacy)"' not in panel_source
assert 'prop(fs, "cache_playback_enabled")' not in panel_source
assert 'legacy.label(text="Legacy / Single Frame Export"' not in panel_source
```

Assert the panel contains one activity card plus explicit Play Cache/Stop Cache Playback and Start/Resume/Cancel Bake actions.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest -q tests/test_operator_panel_behavior.py
```

- [ ] **Step 3: Implement activity snapshot and explicit cache-playback actions**

Use `production_bake.active_bake(scene)` as authority and only fall back to RNA mirrors for terminal status presentation. The panel passes the current animation-playing boolean into the snapshot. Keep draw callbacks read-only. Operators call `set_cache_playback_enabled()` and report actionable failures.

- [ ] **Step 4: Replace the visible cache UI**

Rename the panel to `Cache & Bake`. Remove raw legacy/live-record, raw cache-playback checkbox, and legacy single-frame export sections. Retain Cache Utilities in a default-closed child panel containing Verify, Remove Orphans, Open, Move, Rename, and Free.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command and expect it to pass.

- [ ] **Step 6: Commit the activity UI**

```powershell
git add fluidSim_warp_blender51/blender/playback.py fluidSim_warp_blender51/operators.py fluidSim_warp_blender51/panels.py fluidSim_warp_blender51/properties.py tests/test_operator_panel_behavior.py
git commit -m "feat(ui): simplify cache and bake activity"
```

---

### Task 5: Promote core settings and keep true advanced controls advanced

**Files:**
- Modify: `fluidSim_warp_blender51/panels.py`
- Test: `tests/test_operator_panel_behavior.py`

**Interfaces:**
- Produces the panel hierarchy specified by the design without changing RNA setting ownership or solver semantics.

- [ ] **Step 1: Write failing hierarchy tests**

Assert:

```python
assert not getattr(FLUIDSIM_PT_domain, "bl_parent_id", "")
assert not getattr(FLUIDSIM_PT_particles, "bl_parent_id", "")
assert FLUIDSIM_PT_advanced_field_sources.bl_parent_id == "FLUIDSIM_PT_inputs"
assert FLUIDSIM_PT_stflip.bl_parent_id == "FLUIDSIM_PT_advanced"
```

Also assert `min_substeps` is drawn beside `max_substeps`, Maximum Playback Recovery Frames is visible without an Advanced disclosure, and manual leaf/topology capacity controls remain under Advanced.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest -q tests/test_operator_panel_behavior.py
```

- [ ] **Step 3: Reorganize panels without duplicating properties**

Promote Domain and Particles to regular top-level panels. Remove `leaf_capacity` and `topology_expand` from Domain and draw them in the Advanced parent where `min_substeps` used to be. Split the ST-FLIP block into `FLUIDSIM_PT_stflip` under Advanced. Move Field Sources under Inputs and rename its label. Move `min_substeps` to Quality. Move `max_frame_catchup` into the regular Simulation settings. Keep diagnostics, validation, snippets, and low-level surface capacity controls Advanced.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command and expect it to pass.

- [ ] **Step 5: Commit the settings hierarchy**

```powershell
git add fluidSim_warp_blender51/panels.py tests/test_operator_panel_behavior.py
git commit -m "refactor(ui): promote core simulation settings"
```

---

### Task 6: Repair performance-contract expectations and run executable verification

**Files:**
- Modify: `tools/profile_user_path.py`
- Modify: `tests/test_production_gate_truthfulness.py`
- Test: `tests/test_production_gate_semantics.py`

**Interfaces:**
- Produces cadence-derived expected surface extractions/readback entrypoints.
- Preserves the declared viewport and total-time ceilings.

- [ ] **Step 1: Write failing cadence-derived contract tests**

For `frames=12`, cadence `2`, and first-frame publication, assert the expected publication count is derived from the actual measured solver-frame sequence rather than hard-coded to twelve.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest -q tests/test_production_gate_truthfulness.py tests/test_production_gate_semantics.py
```

- [ ] **Step 3: Implement cadence-aware profiler assertions**

Record `publish_surface` per measured frame or derive it with `HostFrame.should_publish_surface()`. Assert one host-frame acquisition and one publication invocation per measured frame, but compare surface extraction and NumPy entrypoints against the derived surface-publication count.

- [ ] **Step 4: Run the complete focused Python contract lane**

```powershell
python -m pytest -q tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_operator_panel_behavior.py tests/test_performance_hotpath_contracts.py tests/test_production_gate_truthfulness.py tests/test_production_gate_semantics.py
```

- [ ] **Step 5: Run Blender background acceptance**

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --factory-startup --python tests/blender_cache_export_pipeline.py
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --factory-startup --python tests/blender_handler_persistence_gate.py
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --factory-startup --python tests/blender_viewport_2frame_gate.py
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --factory-startup --python tests/blender_background_acceptance.py
```

Expected markers include the gate-specific PASS markers and no Blender traceback.

- [ ] **Step 6: Run the user-path profiler**

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --factory-startup --python tools/profile_user_path.py -- --frames 12 --voxel-size 0.08 --max-particles 200000 --collider-subdiv 2 --json-out artifacts/user_path_profile.json
```

Expected: `USER-PATH-PROFILE-PASS`, viewport mean below 80 ms, no added output-path synchronization or structural Blender ID churn.

- [ ] **Step 7: Run the production gate with performance lane**

```powershell
$env:FLUIDSIM_USER_PATH_PROFILE='1'
python tools/run_production_gate.py --include-gui
Remove-Item Env:FLUIDSIM_USER_PATH_PROFILE
```

Expected: `FULL-PRODUCTION-GATE-PASS`.

- [ ] **Step 8: Open the saved demo in the Blender GUI**

Use the profile-safe launcher and verify the activity card visibly transitions through an available Live, Cache Playback, or Production Bake state without hanging Blender.

- [ ] **Step 9: Review and commit final verification adjustments**

```powershell
git add tools/profile_user_path.py tests/test_production_gate_truthfulness.py tests/test_production_gate_semantics.py
git commit -m "test(perf): align user path checks with preview cadence"
```

---

## Final Completion Audit

- [ ] Confirm `git diff --check` passes.
- [ ] Confirm no legacy cache controls remain in `panels.py`.
- [ ] Confirm no implicit `cache.write_frame()` call remains in live stepping.
- [ ] Confirm Production Bake owns active-controller registration.
- [ ] Confirm activity-state precedence and conflict tests pass.
- [ ] Confirm every promoted setting appears exactly once in the intended panel.
- [ ] Confirm the user-path profile is below its declared viewport ceiling.
- [ ] Confirm the full GUI-inclusive production gate passes.
- [ ] Confirm the saved demo is open in Blender with the new UI visible.
- [ ] Confirm the worktree contains only intentional commits and artifacts.
