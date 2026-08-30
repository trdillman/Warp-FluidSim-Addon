# FluidSim Warp Surface Pipeline Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce complete surface-enabled Blender frame time by eliminating the surface mesh host-to-device round trip and CPU nearest-attribute sampling while preserving the exact surface and publication contract.

**Architecture:** Keep marching cubes and attribute sampling on CUDA until all enabled surface channels are complete. Expose a device mesh from `SurfaceExtractor`, transform vertices to world space on GPU, sample MAC velocity and particle attributes directly into aligned device arrays, then perform one final host materialization for Blender. Treat dense staging reuse as a second independent candidate.

**Tech Stack:** Blender 5.1, Python, NVIDIA Warp 1.16, CUDA, NumPy, pytest.

**Spec:** `docs/REAL_WORLD_PERFORMANCE_SOURCE_20260826.md`

## Global Constraints

- Do not change surface resolution, iso value, topology, vertex/triangle order, smoothing, cadence, dense bounds, or maximum-dimension failure behavior.
- Preserve point-domain attribute alignment and Blender main-thread mutation.
- Whitewater low-resolution SDF dependency remains independent from render-surface visibility.
- Keep cache and production-bake payload schemas unchanged.

---

### Task 1: Surface-only whole-frame attribution

**Files:**
- Modify: `tools/profile_real_world_frame.py`
- Create: `tests/blender_surface_performance_probe.py`
- Create: `tests/test_surface_performance_probe.py`

**Interfaces:**
- Consumes: `SurfacePipeline.build()`, `SurfaceExtractor.extract()`, `HostFrame.surface_mesh`, `HostFrame.output_attributes()`, `output_publication.prepare_live_payload()`.
- Produces: `surface-performance-v1` with SDF, filtering, topology, sparse-to-dense, marching-cubes, mesh readback, attribute adaptation, Blender publication, total handler frame, peak memory, and exact output checkpoints.

- [ ] **Step 1: Add schema tests**

```python
def test_surface_report_binds_target_and_whole_frame():
    report = valid_surface_report()
    assert report["timings_ms"]["surface_total"] > 0.0
    assert report["timings_ms"]["handler_frame_total"] >= report["timings_ms"]["surface_total"]
    assert report["correctness"]["surface_topology"] == "pass"
```

- [ ] **Step 2: Run red tests**

Run: `python -m pytest -q tests/test_surface_performance_probe.py`

Expected: FAIL because the report and probe do not exist.

- [ ] **Step 3: Instrument without phase barriers**

Use Warp timing events and the existing final frame synchronization. Count device readbacks, host-to-device uploads of surface vertices, NumPy nearest-neighbor builds, allocations, and bytes transferred.

- [ ] **Step 4: Measure three real scenes**

Run `dam_break`, `crown_splash`, and `adhesive_calender` with surface enabled, default cadence, every currently enabled attribute, one first-use frame, two warmups, and six measured handler frames.

- [ ] **Step 5: Enforce activation**

Continue only if surface plus publication is at least 5 ms or 15% of handler-frame time. Otherwise record DEFER and stop this plan.

### Task 2: Device-resident surface mesh and velocity attributes

**Files:**
- Modify: `fluidSim_warp_blender51/sim/surface.py`
- Modify: `fluidSim_warp_blender51/sim/frame_output.py`
- Modify: `tests/test_frame_output.py`
- Modify: `tests/test_surface_resolution_feature_contracts.py`

**Interfaces:**
- Produces: `SurfaceDeviceMesh(positions, indices, vertex_count, triangle_count)` and `SurfacePipeline.device_mesh()`.
- Produces: `HostFrame.surface_device_attributes(keys: frozenset[str]) -> dict[str, wp.array]`.
- Preserves: `HostFrame.surface_mesh -> tuple[np.ndarray, np.ndarray]` for existing consumers.

- [ ] **Step 1: Add a device-mesh red test**

```python
def test_device_mesh_materializes_existing_surface_once(cuda_solver):
    device_mesh = cuda_solver.surface.device_mesh()
    verts, tris = cuda_solver.surface_mesh()
    assert np.array_equal(device_mesh.positions.numpy(), verts)
    assert np.array_equal(device_mesh.indices.numpy().reshape(-1, 3), tris)
```

- [ ] **Step 2: Add attribute-alignment red tests**

Cover empty/non-empty surfaces and velocity/speed/motion. Require the candidate arrays to match the existing production path exactly and retain marching-cubes vertex order.

- [ ] **Step 3: Run red tests**

Run: `python -m pytest -q tests/test_frame_output.py tests/test_surface_resolution_feature_contracts.py`

Expected: FAIL because the device-mesh interface does not exist.

- [ ] **Step 4: Transform marching-cubes vertices on GPU**

Add a Warp kernel with this interface:

```python
@wp.kernel
def k_surface_vertices_to_world(
    source: wp.array(dtype=wp.vec3),
    count: int,
    voxel_size: float,
    offset: wp.vec3,
    output: wp.array(dtype=wp.vec3),
): ...
```

Do not modify the marching-cubes buffer in place if Warp reuses it internally. Reuse a capacity-grown world-position buffer owned by `SurfaceExtractor`.

- [ ] **Step 5: Sample velocity-family attributes before host readback**

Feed the device world-position buffer directly into `k_sample_surface_velocity`. Derive speed and motion with one fused GPU kernel so vertex positions are not uploaded from NumPy.

- [ ] **Step 6: Preserve lazy HostFrame behavior**

Materialize device mesh and requested attributes once per solver frame. `HostFrame.surface_mesh` and `output_attributes()` must share the device result and perform only final device-to-host transfers.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest -q tests/test_frame_output.py tests/test_surface_resolution_feature_contracts.py tests/test_color_mixing.py tests/test_operator_panel_behavior.py`

Expected: PASS.

- [ ] **Step 8: Commit the velocity-family candidate**

```powershell
git add fluidSim_warp_blender51/sim/surface.py fluidSim_warp_blender51/sim/frame_output.py tests/test_frame_output.py tests/test_surface_resolution_feature_contracts.py
git commit -m "perf(surface): keep mesh attributes on GPU"
```

### Task 3: GPU nearest-particle surface attributes

**Files:**
- Modify: `fluidSim_warp_blender51/sim/frame_output.py`
- Modify: `fluidSim_warp_blender51/sim/attributes.py`
- Modify: `tests/test_frame_output.py`
- Modify: `tests/test_color_mixing.py`

**Interfaces:**
- Produces: `k_sample_surface_particle_attributes` using the existing particle voxel/bin lookup or a dedicated bounded spatial lookup.
- Consumes: device surface positions and enabled particle arrays.

- [ ] **Step 1: Characterize exact current nearest mapping**

Freeze fixtures for color, age, lifetime, viscosity, density, source ID, and UID. Record current nearest indices and values, including ties and empty particles.

- [ ] **Step 2: Write GPU-versus-production red tests**

```python
@pytest.mark.parametrize("key", ["color", "age", "lifetime", "viscosity", "density", "source_id", "uid"])
def test_gpu_surface_attribute_matches_current_nearest_contract(key, fixture):
    assert np.array_equal(fixture.gpu_values(key), fixture.current_values(key))
```

- [ ] **Step 3: Implement one attribute-independent nearest-index kernel**

Compute the nearest particle index once per surface vertex, then gather all enabled channels with aligned kernels. Preserve the current cell size, distance comparison, and deterministic tie rule. If the current CPU tie rule cannot be matched exactly, this candidate requires a calibrated/approved semantic change and must not silently replace it.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest -q tests/test_frame_output.py tests/test_color_mixing.py tests/test_simulation_output_equivalence.py`

Expected: PASS.

- [ ] **Step 5: Benchmark this candidate separately**

Require eliminated CPU nearest-index construction, no surface-position upload, at least 20% lower attribute-adaptation time, and a measurable whole-frame win. Reject and revert if attribute modes are uncommon or the whole-frame effect is below the program gate.

### Task 4: Exact active dense staging

**Files:**
- Modify: `fluidSim_warp_blender51/sim/surface.py`
- Modify: `tests/test_surface_resolution_feature_contracts.py`
- Modify: `tests/test_memory_safety_recovery.py`

**Interfaces:**
- Produces: `SurfaceDenseBucket` with logical active dimensions distinct from allocated high-water capacity.

- [ ] **Step 1: Add allocation/copy-count tests**

Prove stable bounds reuse allocations, shrinking bounds do not clear/copy unused capacity, growing bounds bucket predictably, and max-dense failures remain exact.

- [ ] **Step 2: Implement logical dimensions over reusable buffers**

Keep the marching-cubes logical dimensions equal to the current padded active box. Do not enlarge the processed volume merely because a prior frame allocated a larger buffer. If Warp `MarchingCubes` cannot separate capacity from logical dimensions, retain the current behavior rather than inventing unsafe slicing.

- [ ] **Step 3: Benchmark separately**

Use expanding then contracting liquid bounds. Require identical mesh envelopes, lower sparse-to-dense/marching-cubes time, and peak memory no worse than baseline.

### Task 5: Surface retention gate

**Files:**
- Create: `artifacts/real_world_performance/surface/comparison.json`

- [ ] **Step 1: Run focused and Blender gates**

Run:

```powershell
python -m pytest -q tests/test_frame_output.py tests/test_surface_resolution_feature_contracts.py tests/test_color_mixing.py tests/test_memory_safety_recovery.py
blender --background --factory-startup --python tests/headless_surface_resolution_pipeline.py -- --device cuda:0 --particles 500000 --frames 24
blender --background --factory-startup --python tests/blender_attribute_viz_acceptance.py
```

- [ ] **Step 2: Run seven fresh-process paired trials**

Require at least 5 ms or 15% lower handler-frame median, CI excluding parity, p95 ratio no worse than 1.02, peak memory no worse than 1.02x, exact enabled attributes/order, and calibrated surface envelope pass.

- [ ] **Step 3: Prove live visible playback**

Open the real demo through the canonical Blender launcher, advance frames through real handlers, and verify visibly changing owned surface output. Do not use cache replay as this proof.
