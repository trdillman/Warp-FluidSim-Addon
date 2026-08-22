# Live Geometry Attributes Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish every enabled FluidSim attribute as a correctly named Blender data attribute during live viewport playback, so it is available in the Spreadsheet and Geometry Nodes without breaking cache compatibility or causing an uncontrolled playback regression.

**Architecture:** Keep `sim/frame_output.py` as the sole owner of authoritative enabled-channel gathering and `blender/output_publication.py` as the Blender adaptation/publication owner. Live, cache, and visualization payloads will use lowercase `ATTRIBUTE_SPECS` keys internally; Blender names, types, and POINT domains remain defined only by `sim/attributes.py`. Only visible outputs and enabled channels may cause host readback, and one `HostFrame` must continue to memoize shared source arrays for the publication.

**Tech Stack:** Python 3, NumPy, NVIDIA Warp, Blender 5.1 Python API, pytest, Blender background acceptance, existing user-path profiler.

**Spec:** `AGENTS.md` (GPU-resident simulation, shared-owner diagnosis, runtime proof), `fluidSim_warp_blender51/sim/attributes.py` (attribute schema), and `fluidSim_warp_blender51/properties.py` (enabled attribute publication promises).

## Global Constraints

- Do not change cache attribute names, Blender attribute names, data types, or POINT domains.
- Simulation state remains GPU-resident; host copies are permitted only for Blender output publication and must be bounded to visible, enabled channels.
- Preserve selected visualization behavior even when its corresponding production attribute toggle is disabled.
- Disabled or hidden outputs must not request their production attributes.
- Reuse one acquired `HostFrame` and its memoized arrays; do not add solver-owned readbacks or synchronization calls.
- Live and cache payloads must use canonical lowercase schema keys; UI enum values may remain uppercase at the property boundary.
- Do not rebuild demos or modify saved `.blend` files for this correction.
- Do not weaken the performance gate to accept arbitrary new readbacks; replace obsolete zero-readback assumptions with exact enabled-channel accounting.

## Truth Contract

| Claim | Required implementation | Required proof |
|---|---|---|
| “Velocity enabled” publishes live surface velocity | `prepare_live_payload()` asks `HostFrame.output_attributes(..., "surface", positions=verts)` and `_swap_mesh()` publishes `fluidsim_velocity` | Blender original and evaluated meshes contain FLOAT_VECTOR/POINT data aligned to surface vertices |
| Geometry Nodes can read the attribute | Attribute exists on evaluated geometry and a Named Attribute node reports it exists | Blender acceptance writes an `Exists` probe through a Geometry Nodes modifier |
| Spreadsheet can show the attribute | Attribute exists on the original output mesh datablock | Blender acceptance inspects `FluidSim Surface.data.attributes` |
| Visualization still works | Selected enum mode is canonicalized and merged without duplicate gathering | Existing varying `display_color` checks continue to pass |
| Cache and live agree | Both payloads carry the same lowercase schema keys | Unit parity test compares live and cache payload inventories |
| Performance remains bounded | Hidden/disabled channels do no work; enabled shared arrays are memoized | Exact call-count tests plus paired user-path profiles |

---

### Task 1: Canonical live attribute payloads

**Files:**
- Modify: `fluidSim_warp_blender51/blender/output_publication.py:96-100,862-936,940-1031,1162-1230`
- Modify: `tests/test_frame_output.py:173-210`

**Interfaces:**
- Consumes: `HostFrame.output_attributes(settings, output_kind, *, positions=None, stride=1) -> dict[str, np.ndarray]`
- Produces: `_canonical_attribute_key(mode) -> str | None` and `_merge_selected_attribute(attributes, mode, values) -> dict[str, np.ndarray]`
- Preserves: `PublicationPayload` fields and lowercase `ATTRIBUTE_SPECS` keys.

- [ ] **Step 1: Replace the obsolete omission test with failing live-contract tests**

Create a recording `LiveHost` whose `output_attributes()` returns distinct lowercase maps for particles, whitewater, and surface. Assert that:

```python
def test_live_payload_gathers_enabled_attributes_for_visible_outputs():
    payload = output_publication.prepare_live_payload(scene, host)

    assert host.attribute_calls[0] == ("particles", {"stride": 2})
    assert host.attribute_calls[1] == ("whitewater", {})
    assert host.attribute_calls[2][0] == "surface"
    np.testing.assert_array_equal(
        host.attribute_calls[2][1]["positions"], host.surface_mesh[0])
    assert set(payload.surface_attributes) == {"velocity", "motion"}
    assert set(payload.particle_attributes) == {"age"}
    assert set(payload.whitewater_attributes) >= {"velocity", "type"}
```

Add three neighboring tests:

```python
def test_selected_visualization_channel_is_canonical_and_not_recomputed_when_enabled():
    assert payload.surface_attribute_mode == "SPEED"
    assert set(payload.surface_attributes) == {"speed"}
    assert host.surface_value_calls == 0


def test_selected_visualization_channel_survives_when_production_toggle_is_disabled():
    assert set(payload.surface_attributes) == {"speed"}
    np.testing.assert_array_equal(payload.surface_attributes["speed"], expected)


def test_hidden_outputs_do_not_request_attributes():
    assert host.attribute_calls == []
    assert payload.surface_attributes == {}
    assert payload.particle_attributes == {}
    assert payload.whitewater_attributes == {}
```

- [ ] **Step 2: Run the new tests and verify the diagnosed failure**

Run:

```powershell
python -m pytest -q tests/test_frame_output.py -k "live_payload or selected_visualization or hidden_outputs"
```

Expected before implementation: failure because `prepare_live_payload()` never calls `output_attributes()` and emits uppercase visualization keys.

- [ ] **Step 3: Add one canonical key/merge seam**

Implement in `output_publication.py`:

```python
def _canonical_attribute_key(mode):
    key = str(mode or "").lower()
    return key if key in ATTRIBUTE_SPECS else None


def _merge_selected_attribute(attributes, mode, values):
    result = {
        str(key).lower(): np.asarray(value)
        for key, value in attributes.items()
        if str(key).lower() in ATTRIBUTE_SPECS
    }
    key = _canonical_attribute_key(mode)
    if key is not None and key not in result and values is not None:
        result[key] = np.asarray(values)
    return result
```

Make `_phase3_attribute_map()` delegate to the same canonical filtering rule. Do not retain `_attribute_map()` as a second casing policy.

- [ ] **Step 4: Gather enabled channels only for visible live outputs**

Refactor `prepare_live_payload()` in this order for each visible output:

```python
particle_attributes = host.output_attributes(
    fs, "particles", stride=step) if fs.show_particles else {}
particle_key = _canonical_attribute_key(fs.particle_attribute_mode)
particle_values = None
if particle_key is not None and particle_key not in particle_attributes:
    particle_values = _particle_values(
        solver, fs.particle_attribute_mode, positions, indices=indices, host=host)
particle_attributes = _merge_selected_attribute(
    particle_attributes, fs.particle_attribute_mode, particle_values)
```

Apply the same policy to:

```python
host.output_attributes(fs, "whitewater")
host.output_attributes(fs, "surface", positions=verts)
```

For whitewater, retain physical `type` when Separate/Both output requires it. For surface, acquire `host.surface_mesh` once, pass those exact vertices to `output_attributes()`, and reuse any enabled selected channel instead of recomputing it.

- [ ] **Step 5: Canonicalize cache and visualization lookups**

In `prepare_cache_payload()`, merge selected visualization values through `_merge_selected_attribute()` instead of assigning uppercase enum keys. In `publish_payload()` and `_publish_whitewater_points()`, retrieve visualization values with `_canonical_attribute_key(payload.*_attribute_mode)` while leaving the stored UI mode unchanged.

- [ ] **Step 6: Add live/cache inventory parity and alignment tests**

Add assertions that live and cache payloads both expose lowercase keys and exact output-length arrays:

```python
assert set(live.surface_attributes) == set(cached.surface_attributes)
assert all(len(value) == len(live.surface_verts)
           for value in live.surface_attributes.values())
assert "SPEED" not in live.surface_attributes
assert "SPEED" not in cached.surface_attributes
```

Also test particle display striding: every production particle attribute must have `len(payload.particles)` entries after `particle_display_fraction` is applied.

- [ ] **Step 7: Run the focused contract lane**

Run:

```powershell
python -m pytest -q tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_performance_hotpath_contracts.py
```

Expected: all pass; no test may assert that enabled live production attributes are omitted.

- [ ] **Step 8: Review and commit Task 1**

Inspect `git diff --check` and the focused diff, then commit only the two files:

```powershell
git add fluidSim_warp_blender51/blender/output_publication.py tests/test_frame_output.py
git commit -m "fix(output): publish enabled live attributes"
```

---

### Task 2: Prove Spreadsheet and Geometry Nodes access in Blender

**Files:**
- Modify: `tests/blender_attribute_viz_acceptance.py:44-133`

**Interfaces:**
- Consumes: `FluidSim Surface.data.attributes` created by Task 1.
- Produces: Blender acceptance checks for original datablock, evaluated geometry, and a real Named Attribute node.

- [ ] **Step 1: Add failing named-attribute assertions before implementation is present**

Enable at least velocity, speed, and motion before initialization:

```python
fs.surface_attribute_velocity = True
fs.surface_attribute_speed = True
fs.surface_attribute_motion = True
```

After stepping, assert exact schema and cardinality:

```python
def _check_attribute(data, name, data_type, domain, count):
    attr = data.attributes.get(name)
    check(f"{name} exists", attr is not None)
    if attr is not None:
        check(f"{name} type", attr.data_type == data_type, attr.data_type)
        check(f"{name} domain", attr.domain == domain, attr.domain)
        check(f"{name} aligned", len(attr.data) == count, len(attr.data))
    return attr


_check_attribute(surf.data, "fluidsim_velocity", "FLOAT_VECTOR", "POINT",
                 len(surf.data.vertices))
_check_attribute(surf.data, "fluidsim_speed", "FLOAT", "POINT",
                 len(surf.data.vertices))
_check_attribute(surf.data, "velocity", "FLOAT_VECTOR", "POINT",
                 len(surf.data.vertices))
```

Run the acceptance against pre-fix code and expect these checks to fail.

- [ ] **Step 2: Add evaluated-mesh preservation checks**

Evaluate the surface through the depsgraph and assert the same three attributes, types, domains, and counts on `surf.evaluated_get(depsgraph).data`. This distinguishes publication failure from a modifier/evaluated-geometry loss.

- [ ] **Step 3: Add a real Geometry Nodes Named Attribute probe**

Create a temporary Geometry Nodes modifier in the acceptance process. Build Group Input → Store Named Attribute → Group Output, add a `GeometryNodeInputNamedAttribute` configured as `FLOAT_VECTOR`, set its name input to `fluidsim_velocity`, and connect its `Exists` output to a BOOLEAN/POINT Store Named Attribute named `fluidsim_gn_velocity_exists`:

```python
named = nodes.new("GeometryNodeInputNamedAttribute")
named.data_type = "FLOAT_VECTOR"
named.inputs["Name"].default_value = "fluidsim_velocity"

store = nodes.new("GeometryNodeStoreNamedAttribute")
store.data_type = "BOOLEAN"
store.domain = "POINT"
store.inputs["Name"].default_value = "fluidsim_gn_velocity_exists"
links.new(named.outputs["Exists"], store.inputs["Value"])
```

Inspect the evaluated `fluidsim_gn_velocity_exists` attribute with `foreach_get("value", values)` and require all live surface points to report true. Keep the modifier temporary; do not save the scene.

- [ ] **Step 4: Preserve and rerun visualization checks**

Keep the existing spatially varying `display_color` and material-binding checks. This proves named production attributes and visualization remain independent, functional surfaces.

Run:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python tests/blender_attribute_viz_acceptance.py
```

Expected marker: `ATTRIBUTE-VIZ-ACCEPTANCE-PASS` with named original, evaluated, and Geometry Nodes probe checks passing.

- [ ] **Step 5: Review and commit Task 2**

```powershell
git add tests/blender_attribute_viz_acceptance.py
git commit -m "test(blender): prove live named attribute access"
```

---

### Task 3: Bound the attribute-enabled viewport cost

**Files:**
- Modify: `tools/profile_user_path.py:77-115,240-294,297-325,519-544`
- Modify: `tests/test_performance_hotpath_contracts.py`

**Interfaces:**
- Consumes: output cost counters and `host_output_operation_counts()`.
- Produces: `--surface-attributes {off,default,all}` profile configuration plus machine-readable attribute/readback inventory.

- [ ] **Step 1: Add source-contract tests for a selectable profile mode**

Require the profiler to expose the mode, record enabled keys, report host output operation counts, and keep the existing no-Warp-synchronization assertion. The source test must fail until the profiler contains all four signals.

- [ ] **Step 2: Add the profile argument and configure settings explicitly**

Add:

```python
ap.add_argument(
    "--surface-attributes",
    choices=("off", "default", "all"),
    default="default",
)
```

After setup, set every `surface_attribute_*` toggle from `OUTPUT_ATTRIBUTE_KEYS["surface"]` deterministically. `off` disables optional fields, `default` restores property defaults, and `all` enables every supported surface channel. Record the resolved enabled key list in the JSON summary.

- [ ] **Step 3: Replace the obsolete zero-readback rule with exact accounting**

Do not simply delete `no added output-path device readbacks`. Assert instead:

- `off` adds no attribute readbacks.
- `default` and `all` request only their resolved enabled keys.
- shared sources such as velocity used by velocity, speed, and motion are read back once per acquired `HostFrame`, not once per derived channel.
- no `wp.synchronize()` or `wp.synchronize_device()` is introduced.
- surface extraction cadence and one-host-frame-per-publication invariants remain unchanged.

Emit both `phase2_costs` and `host_output_operation_counts()` in the JSON report so the gate is auditable.

- [ ] **Step 4: Run matched off/default profiles**

Use identical scene parameters and isolated Warp caches:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python tools/profile_user_path.py -- --frames 12 --voxel-size 0.08 --max-particles 200000 --collider-subdiv 2 --surface-attributes off --json-out artifacts/user_path_attributes_off.json

& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python tools/profile_user_path.py -- --frames 12 --voxel-size 0.08 --max-particles 200000 --collider-subdiv 2 --surface-attributes default --json-out artifacts/user_path_attributes_default.json
```

Compare warm `means.view_ms`, `means.total_ms`, surface vertex counts, particle counts, output cadence, and operation counts. Stop and optimize before release if:

- the `off` path regresses more than 5% from the pre-change matched baseline, or
- the default enabled path adds more than 15% warm viewport time over `off`, or
- either path shows per-channel duplicate readbacks or any explicit synchronization.

These are correction acceptance ceilings, not permission to tune the solver or demos around a failing output path.

- [ ] **Step 5: Run the all-channel stress profile**

Run the same command with `--surface-attributes all` and `--no-validate`. Record its incremental cost and exact readback inventory. It is diagnostic evidence; default-path acceptance remains the release gate.

- [ ] **Step 6: Run source and focused performance tests**

```powershell
python -m pytest -q tests/test_performance_hotpath_contracts.py tests/test_frame_output.py
```

- [ ] **Step 7: Review and commit Task 3**

```powershell
git add tools/profile_user_path.py tests/test_performance_hotpath_contracts.py
git commit -m "test(perf): bound live attribute publication cost"
```

---

### Task 4: Integration, packaging, and installed-runtime handoff

**Files:**
- Modify only if required by existing release workflow: generated machine-readable evidence under `artifacts/`
- Do not modify demos or UI labels unless runtime proof exposes a separate defect.

**Interfaces:**
- Consumes: Tasks 1-3 and the existing release builder/gates.
- Produces: source, packaged, installed, and live-Blender proof for the same commit.

- [ ] **Step 1: Run the complete focused regression set**

```powershell
python -m pytest -q tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_snippet_library.py tests/test_production_gate_truthfulness.py tests/test_sparse_leaf_grid_contract.py tests/test_viscosity_active_work_regressions.py tests/test_performance_hotpath_contracts.py
```

- [ ] **Step 2: Run Blender attribute and two-frame playback acceptance**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python tests/blender_attribute_viz_acceptance.py
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python tests/blender_viewport_2frame_gate.py
```

Expected markers: `ATTRIBUTE-VIZ-ACCEPTANCE-PASS` and the existing two-frame pass marker.

- [ ] **Step 3: Run the production gate with the user-path profile**

```powershell
$env:FLUIDSIM_USER_PATH_PROFILE='1'
python tools/run_production_gate.py
```

If any GUI/playback owner changed indirectly, also run `python tools/run_production_gate.py --include-gui`. Do not refresh policy reports unless the GUI-inclusive gate actually passes for the current HEAD.

- [ ] **Step 4: Build the extension through the existing release owner**

Use `fluidSim_warp_blender51/release/build_extension.py` and its checked-in runtime policy. Validate the resulting ZIP rather than manually copying source into an archive. Confirm the manifest version, packaged platform, required external/bundled runtime policy, and artifact hash all correspond to the tested commit.

- [ ] **Step 5: Install the exact tested package and clean only stale FluidSim copies**

Use the `source-command-blender-install` workflow at execution time. Preserve unrelated add-ons and Blender preferences. Remove only superseded FluidSim installations after resolving their exact paths.

- [ ] **Step 6: Prove the installed add-on and live scene in the same Blender process**

Open `demos/dam_break.blend` without saving it. In the active process prove:

- `addon_utils.check(module_name)[1]` is true;
- the add-on is present in `bpy.context.preferences.addons`;
- the loaded module path matches the newly installed package;
- FluidSim registration is present;
- the scene advances through real handlers;
- `FluidSim Surface` contains `fluidsim_velocity` and `velocity` on the POINT domain;
- the evaluated mesh and Geometry Nodes Exists probe succeed.

If Blender MCP is used, independently prove `blender_mcp` installed/enabled/module path plus `status`, `get_file_state`, `scene_info`, and `execute_code` in that same process.

- [ ] **Step 7: Final diff and residual-risk review**

Confirm no demo, cache schema, solver, force, collision, or UI files changed. Report paired profile deltas and distinguish exact measured overhead from inference. The only acceptable residual risk is machine-to-machine timing variance; missing named attributes, uppercase payload keys, duplicate shared readbacks, or absent live-runtime proof are release blockers.

- [ ] **Step 8: Commit any required release evidence separately**

Stage only evidence paths produced by the approved release workflow and use a release/test-scoped Conventional Commit. Do not fold unrelated dirty work into this correction.

## Self-Review Results

- Spec coverage: live publication, schema ownership, cache parity, Spreadsheet visibility, Geometry Nodes access, visualization compatibility, performance, packaging, and installed-runtime proof each have an owning task.
- Placeholder scan: no deferred implementation steps or unspecified tests remain.
- Type consistency: all payload maps use `dict[str, np.ndarray]`; internal keys are lowercase schema keys; Blender names/types/domains remain owned by `ATTRIBUTE_SPECS`.
- Scope check: no new cache format, attribute schema, UI subsystem, demo rebuild, or solver change is included.

