# Profile and Frame Output Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the user-path profiler and surface-output attribute path easier to read while preserving measurements, output payloads, cache/live parity, and GPU readback behavior.

**Architecture:** First separate the profiler's existing setup, measurement, evidence, and restoration phases without changing their order. Then extract only the surface branch from `HostFrame.output_attributes`; particle and whitewater handling stay in the existing method. The second task is gated by pre/post profile evidence because it touches live GPU-to-host output.

**Tech Stack:** Blender 5.1, Python 3, NVIDIA Warp, CUDA, NumPy, pytest, JSON.

**Spec:** User-authorized simplification direction in this Codex task; repository runtime and ownership contracts in `AGENTS.md`.

## Global Constraints

- Preserve GPU residency, frame cadence, output schemas, cache identity, and all existing JSON keys/verdict semantics.
- Do not add a synchronization, device readback, host upload, nearest-neighbour calculation, or new performance measurement.
- Preserve the profiler's `try`/`finally` restoration of Warp monkey patches, surface attribution hooks, and CSV file handles.
- Preserve `HostFrame.output_attributes()` behavior for particles, whitewater, hidden outputs, empty outputs, and stale host counts.
- Do not route Cache Bake through Preview Playback or Output Publication; these modules retain their ownership boundaries.
- Before execution, create a dedicated implementation worktree using `superpowers:using-git-worktrees`; do not implement on an unrelated dirty checkout.
- Commit each task separately only after its focused tests and `git diff --check` pass.

---

### Task 1: Make profile-user-path phases explicit without changing its report contract

**Files:**
- Modify: `tools/profile_user_path.py:1139-1390`
- Modify: `tests/test_profile_user_path_contract.py:446-499`

**Interfaces:**
- Consumes: parsed CLI args, the initialized scene/session/solver, `_SurfaceTimingProbe`, current Warp monkey-patch targets, and the existing report builders.
- Produces: the same report object and JSON emission from `main()`; `main()` remains the CLI entry point.
- Preserves: warm-frame selection, timing attribution, source digest/runtime identity binding, `wp.array.numpy`/synchronize restoration, surface-attribution restoration, and final GPU-contention snapshot.

- [ ] **Step 1: Add a direct restoration characterization**

Extend `test_surface_instrumentation_wraps_only_observed_boundaries_and_restores` in `tests/test_profile_user_path_contract.py` with an assertion that its original callable identities are restored after the profiled callback completes:

```python
assert wp.array.numpy is original_numpy
assert wp.synchronize is original_synchronize
assert wp.synchronize_device is original_synchronize_device
```

Place the assertions after the existing restoration action in that test. Do not make the test run Blender or CUDA.

- [ ] **Step 2: Run the profiler contract suite before extraction**

Run:

```powershell
python -m pytest -q tests/test_profile_user_path_contract.py tests/test_performance_hotpath_contracts.py tests/test_frame_output.py
```

Expected: all tests pass on the original code. Preserve the precise test counts in the task receipt for later comparison.

- [ ] **Step 3: Extract the existing phases as private helpers**

Introduce private helpers named `_prepare_profile_run`, `_measure_profile_frames`, `_build_profile_report`, and `_restore_profile_instrumentation`. Move current statements into those helpers in their existing execution order. Keep `main()` as the single owner of argument parsing, exit-code selection, and the outer `try`/`finally`. The restoration helper contains the current assignments that restore `wp.array.numpy`, `wp.synchronize`, and `wp.synchronize_device`, then restores the surface attribution callback and closes the CSV handle in their existing order.

- [ ] **Step 4: Run the focused Python regression set after extraction**

Run:

```powershell
python -m pytest -q tests/test_profile_user_path_contract.py tests/test_performance_hotpath_contracts.py tests/test_frame_output.py
```

Expected: every test from Step 2 still passes; the new identity assertions prove restoration remains exact.

- [ ] **Step 5: Produce and validate one focused profile report**

Run:

```powershell
blender --background --factory-startup --python tools/profile_user_path.py -- --frames 6 --voxel-size 0.08 --max-particles 200000 --collider-subdiv 2 --json-out artifacts/user_path_profile.json
```

Expected: exit code 0 and a non-empty `artifacts/user_path_profile.json` with the existing schema and verdict fields. Record the report hash, output fingerprint, and device-readback/synchronization counts as the baseline for Task 2; do not commit this generated artifact unless the repository already tracks that exact artifact.

- [ ] **Step 6: Inspect and commit the profiler-only refactor**

Run:

```powershell
git diff --check
git diff -- tools/profile_user_path.py tests/test_profile_user_path_contract.py
git add -- tools/profile_user_path.py tests/test_profile_user_path_contract.py
git commit -m "refactor: separate user path profiling phases"
```

Expected: one commit containing only profiler code and its contract test.

### Task 2: Extract the surface-output branch while preserving output identity and readback counts

**Files:**
- Modify: `fluidSim_warp_blender51/sim/frame_output.py:496-626`
- Modify: `tests/test_frame_output.py:190-264`
- Modify: `tests/test_scene_bridge_cache_playback.py:239-288`

**Interfaces:**
- Consumes: `HostFrame`, surface mesh data, requested attribute keys, existing host-frame caches, and the GPU nearest-map implementation.
- Produces: a private `_surface_output_attributes` helper called only by `HostFrame.output_attributes()`, using the current enabled keys and optional positions while returning the current surface values, count, and cache information required by the public-method cache write.
- Preserves: public `HostFrame.output_attributes(keys)` signature and return shape, cache-key fields, object-identity guard, early cached-dictionary return, nearest-map reuse, finite-target rejection, zero filling, contiguous arrays, validation, and exact device-readback counter increments.

- [ ] **Step 1: Strengthen the existing cache-identity characterization**

In `tests/test_frame_output.py`, edit `test_opt_in_surface_particle_channels_share_one_gpu_nearest_map` immediately after its existing second `frame.output_attributes` call. It already builds a CPU Warp solver, calls `host_output_operation_counts(reset=True)`, and keeps the repeated surface result in `repeated`. Add this exact assertion before the current byte-comparison assertion:

```python
assert repeated is attributes
```

Keep the existing `host_output_operation_counts()["device_readbacks"] == 7` assertion. Together they prove that a repeat call returns the identical cached dictionary and does not perform an additional readback.

- [ ] **Step 2: Run the targeted frame-output contract tests before extraction**

Run:

```powershell
python -m pytest -q tests/test_frame_output.py -k "surface or output_attributes"
python -m pytest -q tests/test_scene_bridge_cache_playback.py -k "nearest_values or cache_payload or live_payload"
```

Expected: all selected tests pass, including the new identity/readback characterization.

- [ ] **Step 3: Move only the surface branch into a private method**

In `HostFrame`, add a private `_surface_output_attributes` helper and move only the current `elif output_kind == "surface"` branch from `output_attributes` into it. Keep the cached-dictionary early return and cache assignment in `output_attributes`, and keep particle/whitewater branches there as well. The helper must preserve the existing target source selection, target pointer cache key, velocity keys, nearest keys, finite-target rejection, nearest-map reuse, zero fill, contiguous conversion, and device-readback increments. Do not introduce a strategy class, common output-kind abstraction, or a second materialization pass.

- [ ] **Step 4: Run cache/live and performance-hotpath regression suites**

Run:

```powershell
python -m pytest -q tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_profile_user_path_contract.py tests/test_performance_hotpath_contracts.py
```

Expected: all tests pass; no test should observe changed output dictionary identity, output attributes, cache payload contents, or readback counts.

- [ ] **Step 5: Compare focused runtime evidence with the Task 1 baseline**

Run:

```powershell
$env:FLUIDSIM_USER_PATH_PROFILE = '1'
python tools/run_production_gate.py --include-gui
```

Expected: the production gate passes. Compare the refreshed profile report with Task 1's saved baseline: output fingerprint must match; device-readback and synchronization counts must match; schema/verdict fields must match. Treat a timing change by itself as diagnostic, not proof of a behavior change.

- [ ] **Step 6: Inspect, record the comparison, and commit**

Run:

```powershell
git diff --check
git diff -- fluidSim_warp_blender51/sim/frame_output.py tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py
git add -- fluidSim_warp_blender51/sim/frame_output.py tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py
git commit -m "refactor: isolate surface output attributes"
```

Expected: one focused commit. The task receipt records the before/after report hashes, matching output fingerprint, and matching device-readback/synchronization counts.

## Plan Self-Review

- Spec coverage: Task 1 preserves profile generation and restoration; Task 2 preserves the live/cache surface-output contract and requires the GUI-inclusive runtime proof mandated by the project.
- Dependency order: Task 2 uses Task 1's report as its evidence baseline; it must not start until Task 1's focused profile report is saved and validated.
- Scope: this plan excludes playback-policy, cache-bake, solver-kernel, pressure, source-execution, and Blender RNA simplification; they need separate contracts and should not be bundled into an output-path refactor.
- Type consistency: the only new runtime interface is private `HostFrame._surface_output_attributes(self, keys)`, returning the current dictionary value type used by `output_attributes`.
- Completeness: each task has a characterization test, code seam, focused suite, runtime evidence, diff check, and isolated commit.
