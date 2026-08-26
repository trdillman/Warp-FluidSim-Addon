# Dynamic Liquid Species Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an object-authored, runtime-dynamic GPU liquid-species system whose conservative color mixing and live surface visualization meet strict solver and viewport performance gates.

**Architecture:** Compile object mixture properties into one immutable stable-ID registry during Simulation Admission. Store concentrations in active `vec4` chunks, execute one bounded conservative exchange per existing solver substep, and derive surface-sized Beer-Lambert colors on the GPU. Keep no-species execution structurally unchanged and make cache, toggle, UI, and acceptance consume the same registry contract.

**Tech Stack:** Blender 5.1 Python/RNA, NVIDIA Warp/CUDA, NumPy, pytest, Blender headless/GUI acceptance.

**Spec:** `docs/superpowers/specs/2026-08-22-dynamic-liquid-species-design.md`

## Global Constraints

- Preserve the dirty worktree. Never reset, clean, stash, amend, overwrite, or broadly stage unrelated files.
- The archived native-interoperability, rendering-API, and ST-FLIP experiments are not merge candidates and must not be read as implementation sources.
- Species storage remains GPU-resident during live simulation.
- The sparse field contract remains fixed 8x8x8 leaf tiles addressed through the flat leaf table.
- No species-authored scene means exact zero species allocation, kernels, topology tax, or readback.
- No separate species substep/max-substep property may remain.
- The performance thresholds in the spec may not be weakened.
- Every runtime claim requires a fresh artifact produced from the current source.
- Build/rebuild of tracked demos must be limited to named species demos; never run a broad demo rebuild in the dirty worktree.

## Execution Graph

```text
Task 0 isolated candidate ─> Task 1 performance truth ─> Task 1A feasibility stop gate
Task 2 registry contract ──> Task 3 source authoring ──> Task 4 dynamic GPU state
                                         │                    │
                                         └────────────────────┴─> Task 5 one-pass mixing
Task 1 ─────────────────────────────────────────────────────────> Task 6 GPU viewport
Tasks 4-6 ──> Task 7 cache/toggle ──> Task 8 UI ──> Task 9 acceptance ──> Task 10 release gate
```

After Task 0, Tasks 1 and 2 may run concurrently in disjoint files. Task 1A is a mandatory feasibility gate before Tasks 2-4 are committed. Task 6 tests may be prepared after Task 1, but its implementation consumes Task 4's dynamic concentration interface. All other overlapping core edits are serialized.

### Task 0: Create a Provenance-Safe Isolated Candidate

**Files:**
- Create outside the primary checkout: a new `codex/species-v2` worktree
- Create: `artifacts/species_v2_execution_manifest.json`
- Create outside the primary checkout: `species-v1-reviewed.patch`

**Interfaces:**
- Produces: a clean candidate worktree containing only audited current species-lane hunks
- Protects: every dirty file and untracked artifact in the primary checkout
- Consumed by: every subsequent task

- [ ] **Step 1: Record the primary checkout without mutation**

Before Task 1, record but do not mutate:

```powershell
git status --short
git branch --show-current
git rev-parse HEAD
Get-FileHash fluidSim_warp_blender51/sim/species.py -Algorithm SHA256
Get-FileHash fluidSim_warp_blender51/sim/solver.py -Algorithm SHA256
```

Create `artifacts/species_v2_execution_manifest.json` containing the base HEAD, source hashes, exact allowed file list from this plan, and protected dirty demo paths. This manifest is evidence, not permission to stage unrelated files.

- [ ] **Step 2: Export the current species candidate patch without shell redirection**

Use `git diff --binary --output=<absolute-temp-path> HEAD -- <explicit source/test/tool allowlist>`. Exclude all `demos/*.blend`, previews, artifacts, research downloads, `.superpowers`, caches, and review backups. Record the patch SHA-256 in the manifest.

Separately inventory allowlisted untracked candidate files with `git ls-files --others --exclude-standard -- <explicit allowlist>`. Record each relative path, size, and SHA-256 in the manifest. Audit every untracked file's full content, then copy only the accepted explicit paths into the isolated worktree with PowerShell `Copy-Item -LiteralPath`; validate both resolved source and destination remain inside their intended checkout roots before copying. Untracked files are never inferred from a directory glob.

- [ ] **Step 3: Audit every patch hunk before reuse**

Classify each hunk as `current species lane`, `direct prerequisite`, or `ambiguous/unrelated`. Remove ambiguous/unrelated hunks from the candidate patch with an explicit `apply_patch` edit; never stage or commit them merely because they share a planned file. If ownership cannot be proven, stop and leave the primary checkout untouched.

- [ ] **Step 4: Create an isolated worktree from the recorded HEAD**

Use `superpowers:using-git-worktrees` to create a clean `codex/species-v2` worktree. Apply the reviewed patch there with `git apply --binary`. Verify `git status --short` contains only the audited allowlist.

- [ ] **Step 5: Establish the candidate baseline commit**

Run the focused current species tests in the isolated worktree, then commit the reviewed patch as `feat(species): preserve audited color mixing baseline`. All later `git add -- <whole-file>` commands in this plan execute only in that clean isolated worktree. Never execute them in the primary checkout.

---

---

### Task 1: Replace Permissive Performance Receipts with Release Gates

**Files:**
- Modify: `tools/profile_species_transport.py`
- Modify: `tools/run_species_transport_profile_repetitions.py`
- Modify: `tools/profile_user_path.py`
- Create: `tools/profile_species_viewport.py`
- Modify: `tests/test_species_acceptance_report.py`
- Modify: `tests/test_performance_hotpath_contracts.py`
- Create: `tests/test_species_viewport_profile_contract.py`

**Interfaces:**
- Produces: `paired_case_metrics(samples: list[dict]) -> dict`
- Produces: solver cases `disabled`, `configured_inactive`, `transport`, `mixing_n1`, `mixing_n2`, `mixing_n4`, `mixing_n8`; cases unsupported by the pre-redesign implementation are recorded as `unsupported` and fail production validation without preventing the diagnostic baseline from being written
- Produces: viewport cases with matching scene/substep/topology invariants
- Consumed by: Tasks 5, 6, 9, and 10 stop gates

- [ ] **Step 1: Write RED tests for the frozen thresholds and adjacent brackets**

```python
def test_species_release_thresholds_are_not_legacy_budgets():
    assert SOLVER_GATES["configured_inactive"] == {
        "median_fraction": 0.01, "p95_delta_ms": 0.25}
    assert SOLVER_GATES["transport"] == {
        "median_fraction": 0.02, "p95_delta_ms": 0.50}
    assert SOLVER_GATES["mixing_n1_n4"] == {
        "median_fraction": 0.05, "p95_delta_ms": 1.00}
    assert VIEWPORT_GATES["clear_dye"] == {
        "median_fraction": 0.05, "p95_delta_ms": 1.00}

def test_each_feature_sample_has_adjacent_disabled_brackets():
    sample = paired_case_metrics([
        {"case": "disabled", "ms": 10.0},
        {"case": "mixing_n2", "ms": 10.4},
        {"case": "disabled", "ms": 10.2},
    ])
    assert sample["mixing_n2"]["baseline_ms"] == 10.1
```

- [ ] **Step 2: Run RED tests**

```powershell
python -m pytest -q tests/test_species_acceptance_report.py tests/test_species_viewport_profile_contract.py
```

Expected: failure because the current profiler still admits 12%/35% and has no viewport species cases.

- [ ] **Step 3: Implement a shared immutable gate table and paired sampling**

```python
SOLVER_GATES = {
    "configured_inactive": {"median_fraction": 0.01, "p95_delta_ms": 0.25},
    "transport": {"median_fraction": 0.02, "p95_delta_ms": 0.50},
    "mixing_n1_n4": {"median_fraction": 0.05, "p95_delta_ms": 1.00},
}
VIEWPORT_GATES = {
    "clear_dye": {"median_fraction": 0.05, "p95_delta_ms": 1.00},
}
```

For every feature measurement, execute `disabled_before -> feature -> disabled_after` and use the mean of the two disabled timings. Report median, scaled MAD, paired p95 delta, active particles/leaves/voxels, solver substeps, active species, chunk count, launch inventory, readback inventory, and CUDA allocation.

- [ ] **Step 4: Implement the deterministic surface viewport profiler**

`profile_species_viewport.py` must build the same two-inflow/turbine geometry for every case, hold particle count, surface cadence, active voxels, and solver substeps equal, and record synchronized `solve_ms`, `view_ms`, and `total_ms`. `view_ms` begins before device surface extraction/render-field/color generation and ends after the Blender mesh/color update; `total_ms` covers scene sync through publication. Fail the report if compared cases differ on those invariants. The release gate applies to both `view_ms` and `total_ms`, so moving work across the solve/publication boundary cannot hide it.

- [ ] **Step 5: Run GREEN contract tests**

```powershell
python -m pytest -q tests/test_species_acceptance_report.py tests/test_species_viewport_profile_contract.py tests/test_performance_hotpath_contracts.py
```

- [ ] **Step 6: Capture the intentionally failing pre-redesign baseline**

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" tools/run_species_transport_profile_repetitions.py --device cuda:0 --frames 60 --warmup 15 --particles 200000 --json-out artifacts/species_v2_baseline_solver.json
blender --background --factory-startup --python tools/profile_species_viewport.py -- --device cuda:0 --frames 60 --warmup 15 --particles 200000 --json-out artifacts/species_v2_baseline_viewport.json
```

Expected: one or more new release gates fail. Preserve these artifacts as the RED performance evidence.

- [ ] **Step 7: Commit only Task 1 files**

```powershell
git add -- tools/profile_species_transport.py tools/run_species_transport_profile_repetitions.py tools/profile_user_path.py tools/profile_species_viewport.py tests/test_species_acceptance_report.py tests/test_performance_hotpath_contracts.py tests/test_species_viewport_profile_contract.py
git commit -m "test(species): enforce production performance gates"
```

---

### Task 1A: Prove the Staged Solver and GPU Surface Architecture at N=2

**Files:**
- Modify: `fluidSim_warp_blender51/sim/species.py`
- Modify: `fluidSim_warp_blender51/sim/solver.py`
- Modify: `fluidSim_warp_blender51/sim/kernels_fluid.py`
- Modify: `fluidSim_warp_blender51/sim/surface.py`
- Modify: `fluidSim_warp_blender51/sim/frame_output.py`
- Modify: `fluidSim_warp_blender51/blender/output_publication.py`
- Modify: `fluidSim_warp_blender51/blender/materials.py`
- Modify: `tests/test_species_diffusion.py`
- Modify: `tests/test_species_output_ui.py`
- Create: `tests/test_species_surface_output_gpu.py`

**Interfaces:**
- Produces: staged `begin_exchange(dt, rate)` and `finish_exchange()` scheduling used unchanged by Task 5
- Produces: `DeviceSurfaceResult` / `HostSurfaceResult` seam used unchanged by Task 6
- Proves: the architecture can meet N=2 solver and viewport gates before registry/source migration

- [ ] **Step 1: Write RED staged-order tests**

```python
def test_species_exchange_respects_global_barriers():
    events = run_one_substep_with_launch_spy()
    assert events == [
        "fluid_p2g_with_species_deposit",
        "species_conductance",
        "species_flux",
        "forces_viscosity_pressure",
        "velocity_g2p_with_species_candidate",
        "species_global_alpha_complete",
        "species_delta_apply",
        "particle_advection",
    ]
```

Assert no local count/loop, exactly one staged exchange per existing solver substep, and no species calls/resources in the disabled sibling.

- [ ] **Step 2: Define the staged protocol explicitly**

`begin_exchange(dt, rate)` runs after source emission and as part of the enabled P2G path. It clears active scratch, deposits represented volume/species mass, completes conductance, and completes grid flux. It stores a frame/substep token but does not gather or mutate particle concentrations.

After forces/viscosity/pressure, the enabled G2P sibling samples velocity and writes per-particle species candidate deltas while reducing `global_alpha`. `finish_exchange(token)` launches the separate apply kernel only after that device-wide reduction completes. Particle advection then consumes the updated velocity and preserved concentrations. Receipt finalization occurs after lifecycle compaction and never synthesizes AFTER values.

- [ ] **Step 3: Define and implement the device surface-result seam**

```python
@dataclass
class DeviceSurfaceResult:
    vertices_world: wp.array
    triangles: wp.array
    vertex_count: int
    triangle_count: int
    colors: wp.array | None

    def to_host(self) -> HostSurfaceResult: ...
```

`SurfaceExtractor.extract_device()` keeps generated vertices alive on device, performs index-to-world transformation on device, optionally derives vertex colors, and owns buffers until `to_host()` performs one synchronized materialization. Empty results use zero-length arrays. Cadence reuse retains the complete geometry/color pair from the same solver frame; colors can never be reused with different geometry.

- [ ] **Step 4: Implement the N=2 GPU surface-color path**

Generate the current post-advection render concentration field inside the timed output request, sample it at `vertices_world`, derive Beer-Lambert color on device, and transfer one color per vertex. Remove live Clear Dye particle position/concentration readback and CPU nearest mapping for this N=2 proof.

- [ ] **Step 5: Run numerical and output GREEN**

```powershell
python -m pytest -q tests/test_species_diffusion.py tests/test_sparse_species_stencil.py tests/test_species_output_ui.py tests/test_species_surface_output_gpu.py tests/test_frame_output.py
```

- [ ] **Step 6: Run the mandatory feasibility gates**

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" tools/run_species_transport_profile_repetitions.py --device cuda:0 --frames 60 --warmup 15 --particles 200000 --json-out artifacts/species_v2_feasibility_solver.json
blender --background --factory-startup --python tools/profile_species_viewport.py -- --device cuda:0 --frames 60 --warmup 15 --particles 200000 --json-out artifacts/species_v2_feasibility_viewport.json
```

Required before Task 2: transport <=2%; active N=2 mixing <=5% and <=1 ms paired p95; Clear Dye `view_ms` and end-to-end `total_ms` <=5% and <=1 ms paired p95; exact disabled zero-work. If any gate fails, stop this plan at the last passing commit and optimize these fixed-N reusable seams. Do not begin registry, source, cache, or UI migration.

- [ ] **Step 7: Commit only after feasibility passes**

```powershell
git add -- fluidSim_warp_blender51/sim/species.py fluidSim_warp_blender51/sim/solver.py fluidSim_warp_blender51/sim/kernels_fluid.py fluidSim_warp_blender51/sim/surface.py fluidSim_warp_blender51/sim/frame_output.py fluidSim_warp_blender51/blender/output_publication.py fluidSim_warp_blender51/blender/materials.py tests/test_species_diffusion.py tests/test_species_output_ui.py tests/test_species_surface_output_gpu.py
git commit -m "perf(species): prove staged mixing and GPU surface output"
```

---

### Task 2: Add the Pure Stable-ID Registry and Mixture Contract

**Files:**
- Create: `fluidSim_warp_blender51/sim/species_schema.py`
- Create: `tests/test_species_authoring_registry.py`
- Modify: `fluidSim_warp_blender51/sim/quality_contract.py`

**Interfaces:**
- Produces: `SpeciesDefinition`, `MixtureComponent`, `MixturePayload`, `SpeciesRegistry`
- Produces: `compile_species_registry(definitions) -> SpeciesRegistry`
- Produces: `SpeciesRegistry.pack(payload) -> np.ndarray[float32]`
- Produces: `absorption_from_display_color(color, reference_depth) -> tuple[float, float, float]`
- Consumed by: Tasks 3, 4, 7, and 8

- [ ] **Step 1: Write RED identity, ordering, and packing tests**

```python
def test_registry_order_is_stable_and_not_color_deduplicated():
    red_a = SpeciesDefinition("object:a", "Red", RED_ABSORPTION)
    red_b = SpeciesDefinition("object:b", "Red", RED_ABSORPTION)
    registry = compile_species_registry((red_b, red_a))
    assert tuple(item.id for item in registry.definitions) == ("object:a", "object:b")
    assert registry.active_count == 2
    assert registry.chunk_count == 1

def test_pack_rejects_unknown_negative_or_nonfinite_components():
    with pytest.raises(ValueError, match="unknown species"):
        REGISTRY.pack(MixturePayload((MixtureComponent("missing", 1.0),)))
```

Also test duplicate IDs, empty IDs, nonfinite absorption, deterministic fingerprint, N=1/2/4/5/8 chunk counts, simplex normalization, and zero-strength payloads.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_species_authoring_registry.py
```

- [ ] **Step 3: Implement the pure immutable schema**

Fingerprint the canonical JSON containing schema version, sorted IDs, names, and absorption coefficients. Amount changes do not alter the registry fingerprint; identity or absorption changes do.

- [ ] **Step 4: Run GREEN and compile**

```powershell
python -m pytest -q tests/test_species_authoring_registry.py
python -m py_compile fluidSim_warp_blender51/sim/species_schema.py
```

- [ ] **Step 5: Commit**

```powershell
git add -- fluidSim_warp_blender51/sim/species_schema.py fluidSim_warp_blender51/sim/quality_contract.py tests/test_species_authoring_registry.py
git commit -m "feat(species): add stable dynamic registry contract"
```

---

### Task 3: Route Object Mixtures Through Every Source Family

**Files:**
- Modify: `fluidSim_warp_blender51/properties.py`
- Modify: `fluidSim_warp_blender51/blender/scene_inputs.py`
- Modify: `fluidSim_warp_blender51/blender/sources.py`
- Modify: `fluidSim_warp_blender51/blender/scene_bridge.py`
- Modify: `fluidSim_warp_blender51/blender/simulation_admission.py`
- Modify: `fluidSim_warp_blender51/sim/source_compat.py`
- Modify: `fluidSim_warp_blender51/sim/source_regions.py`
- Modify: `fluidSim_warp_blender51/sim/sourcing.py`
- Modify: `tests/test_species_source_lifecycle.py`
- Create: `tests/test_scene_bridge_source_plan_reconciliation.py`

**Interfaces:**
- Consumes: Task 2 schema
- Produces: `AdmissionSnapshot.species_registry: SpeciesRegistry | None`
- Produces: every source payload carries `mixture: MixturePayload`
- Produces: `sync_scene()` complete source-plan replacement semantics
- Consumed by: Tasks 4, 7, 8, and 9

- [ ] **Step 1: Write a RED parametrized production-path test**

```python
@pytest.mark.parametrize("source_kind", [
    "analytic", "mesh", "point_direct", "point_rasterized",
    "vdb", "height_fill", "auto_seed", "injection",
])
def test_every_liquid_source_preserves_mixture_to_source_plan(source_kind):
    plan = compile_source_fixture(source_kind, RED_MIXTURE)
    assert plan.payload.mixture == RED_MIXTURE
```

Add the contrasting empty-plan reconciliation test: synchronize one Point/VDB plan, remove it, synchronize again, and assert the solver receives `()` rather than retaining the old plan.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_species_source_lifecycle.py tests/test_scene_bridge_source_plan_reconciliation.py
```

- [ ] **Step 3: Add RNA storage without final panels**

Add persistent component IDs, liquid color, dye strength, and mixture collections to the existing source property groups. Convert picker color to absorption through Task 2's canonical function; treat Dye Strength as emitted amount. Add the corresponding height-fill fields to scene settings. Do not remove legacy two-lane properties until Task 8 migration is ready.

- [ ] **Step 4: Replace fixed vectors at adapter boundaries**

`scene_inputs`, `sources`, `source_compat`, and `source_regions` pass `MixturePayload`; only the Solver Session packs it against the admitted registry. Unknown IDs fail with `Reinitialize required: source mixture is outside the admitted registry`.

- [ ] **Step 5: Make source reconciliation exhaustive**

Change `scene_bridge.sync_scene()` to call the source-plan replacement seam unconditionally, including an empty tuple. Verify analytic/mesh and Point/VDB sibling lanes.

- [ ] **Step 6: Build the registry inside the coherent Admission Snapshot**

Compile all enabled dyed source definitions before memory planning. Store registry and fingerprint in the snapshot used for allocation, settings, source compilation, and cache identity; no downstream module may rescan Blender objects independently.

- [ ] **Step 7: Run focused source and admission checks**

```powershell
python -m pytest -q tests/test_species_authoring_registry.py tests/test_species_source_lifecycle.py tests/test_scene_bridge_source_plan_reconciliation.py tests/test_sourcing_pipeline.py tests/test_simulation_admission.py
```

- [ ] **Step 8: Commit**

```powershell
git add -- fluidSim_warp_blender51/properties.py fluidSim_warp_blender51/blender/scene_inputs.py fluidSim_warp_blender51/blender/sources.py fluidSim_warp_blender51/blender/scene_bridge.py fluidSim_warp_blender51/blender/simulation_admission.py fluidSim_warp_blender51/sim/source_compat.py fluidSim_warp_blender51/sim/source_regions.py fluidSim_warp_blender51/sim/sourcing.py tests/test_species_source_lifecycle.py tests/test_scene_bridge_source_plan_reconciliation.py
git commit -m "feat(species): route object mixtures through all sources"
```

---

### Task 4: Replace Fixed Vec4 Ownership with Active Chunk Storage

**Files:**
- Modify: `fluidSim_warp_blender51/sim/species.py`
- Modify: `fluidSim_warp_blender51/sim/particles.py`
- Modify: `fluidSim_warp_blender51/sim/particle_capacity.py`
- Modify: `fluidSim_warp_blender51/sim/capacity_planner.py`
- Modify: `fluidSim_warp_blender51/sim/memory_safety.py`
- Modify: `fluidSim_warp_blender51/sim/solver.py`
- Modify: `fluidSim_warp_blender51/sim/frame_output.py`
- Modify: `fluidSim_warp_blender51/blender/output_publication.py`
- Modify: `fluidSim_warp_blender51/io/cache.py`
- Modify: `tests/test_species_state_contract.py`
- Modify: `tests/test_particle_capacity_growth.py`
- Modify: `tests/test_capacity_planner.py`
- Modify: `tests/test_species_cache_io.py`
- Modify: `tests/test_species_output_ui.py`

**Interfaces:**
- Consumes: `SpeciesRegistry`
- Produces: `SpeciesTransport.concentration_chunks: wp.array(dtype=wp.vec4)`
- Produces: `SpeciesTransport.chunk_count: int`
- Produces: `SpeciesTransport.export_concentrations(count) -> np.ndarray[(count, active_count)]`
- Produces: exact-fingerprint schema-2 checkpoint/cache serialization sufficient to keep Task 4 independently runnable
- Produces: temporary dynamic CPU output adapter consumed and then optimized away by Task 6
- Produces: exact dynamic allocation inventory
- Consumed by: Tasks 5, 6, and 7

- [ ] **Step 1: Write RED N=1/4/5/8 allocation, compaction, and growth tests**

```python
@pytest.mark.parametrize("active_count,chunks", [(1, 1), (4, 1), (5, 2), (8, 2)])
def test_dynamic_species_storage_uses_only_active_chunks(active_count, chunks):
    transport = make_transport(active_count=active_count, particle_capacity=17)
    assert transport.chunk_count == chunks
    assert len(transport.concentration_chunks) == 17 * chunks
```

Test survivor alignment, dead-particle sink accounting across every active lane, candidate growth rollback, masking of unused final lanes, and zero allocation when registry is `None`.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_species_state_contract.py tests/test_particle_capacity_growth.py tests/test_capacity_planner.py
```

- [ ] **Step 3: Implement flattened vec4 chunk ownership**

Remove `species_concentration` from the generic fixed optional-particle attribute inventory. `SpeciesTransport` owns chunked concentrations, grid mass/delta, represented volume, palette absorption chunks, receipts, compaction, growth, and export.

```python
def chunk_index(owner_index: int, chunk: int, chunk_count: int) -> int:
    return owner_index * chunk_count + chunk
```

All kernels take `active_count` and `chunk_count`; unused tail lanes are written as zero and rejected if restored nonzero.

- [ ] **Step 4: Implement enabled-only compaction and growth**

Use a species-enabled compaction sibling that copies base particle data and loops active chunks in the same survivor traversal. The disabled `ParticleSystem.compact()` kernel and ABI remain unchanged. Growth aliases grid-sized scratch and allocates only the new particle-sized concentration/delta arrays before cutover.

- [ ] **Step 5: Centralize exact allocation accounting**

One pure integer inventory in `species.py` reports particle, grid, palette, receipt, candidate growth, and host-staging bytes. Capacity planner, memory safety, and Simulation Admission consume this inventory rather than reconstructing layout knowledge.

- [ ] **Step 6: Run GREEN and memory checks**

Before running GREEN, update `HostFrame`, the current CPU publication adapter, solver checkpointing, and cache I/O to consume `SpeciesTransport.export_concentrations()` rather than `ParticleSystem.attr("species_concentration")`. Introduce schema 2 with exact registry-fingerprint restore in this task so save/checkpoint/cache behavior remains coherent after fixed vec4 ownership is removed. Task 7 later adds stable-ID reordering, v1 translation, and hot-toggle policy; it does not repair a broken interim cache.

```powershell
python -m pytest -q tests/test_species_state_contract.py tests/test_particle_capacity_growth.py tests/test_capacity_planner.py tests/test_memory_safety_recovery.py tests/test_species_cache_io.py tests/test_species_output_ui.py tests/test_frame_output.py
```

- [ ] **Step 7: Commit**

```powershell
git add -- fluidSim_warp_blender51/sim/species.py fluidSim_warp_blender51/sim/particles.py fluidSim_warp_blender51/sim/particle_capacity.py fluidSim_warp_blender51/sim/capacity_planner.py fluidSim_warp_blender51/sim/memory_safety.py fluidSim_warp_blender51/sim/solver.py fluidSim_warp_blender51/sim/frame_output.py fluidSim_warp_blender51/blender/output_publication.py fluidSim_warp_blender51/io/cache.py tests/test_species_state_contract.py tests/test_particle_capacity_growth.py tests/test_capacity_planner.py tests/test_species_cache_io.py tests/test_species_output_ui.py tests/test_frame_output.py
git commit -m "feat(species): add active chunk GPU storage"
```

---

### Task 5: Integrate One Bounded Mixing Pass into Existing Solver Substeps

**Files:**
- Modify: `fluidSim_warp_blender51/sim/species.py`
- Modify: `fluidSim_warp_blender51/sim/kernels_fluid.py`
- Modify: `fluidSim_warp_blender51/sim/solver.py`
- Modify: `fluidSim_warp_blender51/sim/fields.py`
- Modify: `fluidSim_warp_blender51/blender/solver_settings.py`
- Modify: `fluidSim_warp_blender51/blender/pending_settings.py`
- Modify: `fluidSim_warp_blender51/properties.py`
- Modify: `tests/test_species_diffusion.py`
- Modify: `tests/test_sparse_species_stencil.py`
- Modify: `tests/test_performance_hotpath_contracts.py`

**Interfaces:**
- Consumes: Task 4 chunk storage and Task 3 packed source payloads
- Consumes: Task 1A staged scheduling
- Produces: `SpeciesTransport.begin_exchange(dt: float, mixing_rate: float) -> ExchangeToken`
- Produces: `SpeciesTransport.finish_exchange(token: ExchangeToken) -> None`
- Produces: exactly one begin/finish pair per existing solver substep
- Produces: `SolverStats.species_transport_steps == SolverStats.substeps` while active
- Consumed by: Tasks 6, 7, and 9

- [ ] **Step 1: Write RED scheduling and numerical tests**

```python
def test_exchange_pair_runs_once_per_existing_solver_substep(monkeypatch):
    solver = make_solver(substeps=3, active_species=5)
    begins, finishes = spy_exchange_pair(monkeypatch, solver.species)
    solver.step_frame()
    assert begins == [solver.last_dt] * 3
    assert len(finishes) == 3

def test_high_mixing_rate_remains_finite_conservative_without_inner_steps():
    result = exchange_two_cells(rate=1.0e6, dt=1.0, left=(1, 0), right=(0, 1))
    assert_simplex(result)
    assert_species_mass_conserved(result)
```

Retain collision/air barriers, cross-leaf equivalence, equal/opposite flux, source/sink budget, and N=1/2/4/5/8 cases.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_species_diffusion.py tests/test_sparse_species_stencil.py tests/test_performance_hotpath_contracts.py
```

- [ ] **Step 3: Delete the independent scheduling surface**

Remove `species_max_diffusion_substeps` from RNA, `SolverSettings`, settings projection, pending registry, cache settings, profiler schema, tests, and UI. Rename diagnostics from `diffusion_substeps` to `transport_steps`.

- [ ] **Step 4: Implement the bounded rate mapping and one-pass exchange**

```python
def bounded_mix_fraction(rate: float, dt: float) -> float:
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("mixing rate must be finite and nonnegative")
    return -math.expm1(-rate * dt)
```

The Task 1A staged protocol is generalized across active chunks without changing its barriers: `begin_exchange()` completes deposit/conductance/flux before forces; the enabled G2P sibling later writes particle candidates and completes global-alpha reduction; `finish_exchange()` applies deltas before advection. Neither method contains a local substep loop. Global alpha remains a true all-particle/all-active-lane reduction; never replace it with independent clamps or expected-value accounting.

- [ ] **Step 5: Fuse particle traversals only in enabled siblings**

Add an enabled sibling of `k_particle_set_grid_data` that performs the existing staggered velocity deposit and cell-centred represented-volume/species-mass deposit in one particle traversal. Add an enabled sibling of `k_staggered_field_to_particle` that samples velocity and concentration delta in one traversal. `_p2g()` and `_g2p_advect()` select these siblings only when a registry is active; disabled kernel ABIs remain untouched.

Conductance and flux remain separate because flux depends on fully accumulated outgoing conductance. Delta calculation and apply remain separate because apply depends on the completed global-alpha reduction.

- [ ] **Step 6: Remove the species-only topology tax with proof**

First prove conservation at leaf boundaries with authored `topology_expand=0`. If generic transfer topology lacks stencil support, fix the generic P2G stencil-support contract in `fields.py`; do not retain `species_enabled => topology_expand >= 1`.

- [ ] **Step 7: Run focused GREEN**

```powershell
python -m pytest -q tests/test_species_diffusion.py tests/test_sparse_species_stencil.py tests/test_species_state_contract.py tests/test_species_source_lifecycle.py tests/test_performance_hotpath_contracts.py
```

- [ ] **Step 8: Run the solver performance stop gate**

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" tools/run_species_transport_profile_repetitions.py --device cuda:0 --frames 60 --warmup 15 --particles 200000 --json-out artifacts/species_v2_core_solver.json
```

Required: configured inactive <=1%, transport <=2%, N=1..4 mixing <=5%, p95 limits pass, no disabled work, no capacity-wide work. If any threshold fails, stop before Task 6 and use the profiler's phase/launch/byte attribution to correct Task 5; do not advance or weaken the gate.

- [ ] **Step 9: Commit**

```powershell
git add -- fluidSim_warp_blender51/sim/species.py fluidSim_warp_blender51/sim/kernels_fluid.py fluidSim_warp_blender51/sim/solver.py fluidSim_warp_blender51/sim/fields.py fluidSim_warp_blender51/blender/solver_settings.py fluidSim_warp_blender51/blender/pending_settings.py fluidSim_warp_blender51/properties.py tests/test_species_diffusion.py tests/test_sparse_species_stencil.py tests/test_performance_hotpath_contracts.py
git commit -m "perf(species): integrate bounded mixing into solver passes"
```

---

### Task 6: Replace CPU Surface Coloring with GPU-First Publication

**Files:**
- Modify: `fluidSim_warp_blender51/sim/species.py`
- Modify: `fluidSim_warp_blender51/sim/surface.py`
- Modify: `fluidSim_warp_blender51/sim/frame_output.py`
- Modify: `fluidSim_warp_blender51/blender/output_publication.py`
- Modify: `fluidSim_warp_blender51/blender/materials.py`
- Modify: `fluidSim_warp_blender51/sim/attributes.py`
- Modify: `tests/test_species_output_ui.py`
- Create: `tests/test_species_surface_output_gpu.py`
- Modify: `tests/test_frame_output.py`

**Interfaces:**
- Consumes: Task 4 registry/chunks
- Consumes: Task 1A `DeviceSurfaceResult` / `HostSurfaceResult` lifetime contract
- Produces: `SurfaceFrame.species_colors: np.ndarray[(vertex_count, 4)]`
- Produces: device `derive_surface_species_colors(...)`
- Produces: one Blender POINT attribute named `fluidsim_color`

- [ ] **Step 1: Write RED output-cost and color-truth tests**

```python
def test_live_clear_dye_surface_never_reads_particle_species_or_positions(host):
    host.positions = forbidden("particle position readback")
    host.species_concentrations = forbidden("species readback")
    payload = prepare_live_payload(CLEAR_DYE_SCENE, host)
    assert payload.surface_attributes.keys() == {"color"}

def test_gpu_surface_color_matches_beer_lambert_reference():
    assert_allclose(gpu_color((0.5, 0.5), RED_BLUE_PALETTE), PURPLE_REFERENCE)
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_species_output_ui.py tests/test_species_surface_output_gpu.py tests/test_frame_output.py
```

- [ ] **Step 3: Generalize the proven current-frame render concentration field on GPU**

After the final lifecycle/advection state for the displayed frame, deposit current particle chunks into a render-only cell-centred concentration field. Reuse active topology and active dimensions; never allocation capacity. The field is invalidated whenever particle state advances.

- [ ] **Step 4: Generalize vertex optics across active chunks**

Before `DeviceSurfaceResult.to_host()` transfers vertices, sample the render concentration field at `vertices_world`, accumulate optical depth across active chunks plus carrier remainder, and write one `wp.vec4` per surface vertex. Preserve Task 1A's single synchronized geometry/color materialization, empty-result behavior, and cadence-paired geometry/color ownership.

- [ ] **Step 5: Publish one attribute and one material input**

Remove live Clear Dye calls to `_nearest_aligned_values`, raw particle concentration readback, `display_color`, and CORNER expansion. Publish `fluidsim_color` once on POINT domain. Make the Clear Dye material read `fluidsim_color`. Keep raw species output behind an explicit advanced debug/export switch.

- [ ] **Step 6: Run GREEN and static hot-path checks**

```powershell
python -m pytest -q tests/test_species_output_ui.py tests/test_species_surface_output_gpu.py tests/test_frame_output.py tests/test_performance_hotpath_contracts.py
```

- [ ] **Step 7: Run the viewport performance stop gate**

```powershell
blender --background --factory-startup --python tools/profile_species_viewport.py -- --device cuda:0 --frames 60 --warmup 15 --particles 200000 --json-out artifacts/species_v2_gpu_viewport.json
```

Required: median `view_ms` and synchronized end-to-end `total_ms` <=5%, paired p95 <=1 ms for both, zero particle-species/position readbacks for surface-only Clear Dye, zero CPU nearest calls, one surface-sized color transfer, identical surface cadence/topology/substeps.

- [ ] **Step 8: Commit**

```powershell
git add -- fluidSim_warp_blender51/sim/species.py fluidSim_warp_blender51/sim/surface.py fluidSim_warp_blender51/sim/frame_output.py fluidSim_warp_blender51/blender/output_publication.py fluidSim_warp_blender51/blender/materials.py fluidSim_warp_blender51/sim/attributes.py tests/test_species_output_ui.py tests/test_species_surface_output_gpu.py tests/test_frame_output.py
git commit -m "perf(viewport): derive species surface color on GPU"
```

---

### Task 7: Implement Hot Pause/Resume and Cache Schema 2

**Files:**
- Modify: `fluidSim_warp_blender51/sim/solver.py`
- Modify: `fluidSim_warp_blender51/sim/species.py`
- Modify: `fluidSim_warp_blender51/io/cache.py`
- Modify: `fluidSim_warp_blender51/blender/playback.py`
- Modify: `fluidSim_warp_blender51/blender/pending_settings.py`
- Modify: `tests/test_species_cache_io.py`
- Modify: `tests/test_scene_bridge_cache_playback.py`
- Modify: `tests/test_phase4_pending_interaction.py`
- Modify: `tests/test_simulation_admission.py`

**Interfaces:**
- Consumes: Tasks 2 and 4 registry/storage
- Produces: `species_schema_version = 2`
- Produces: `set_species_physics_enabled(enabled: bool)` as a live setting
- Produces: stable-ID cache column remapping

- [ ] **Step 1: Write RED toggle and migration tests**

```python
def test_physics_pause_preserves_identity_and_concentrations():
    solver = mixed_solver()
    solver_identity = id(solver)
    before_by_uid = concentrations_by_uid(solver)
    solver.set_species_physics_enabled(False)
    solver.step_frame()
    solver.set_species_physics_enabled(True)
    assert id(solver) == solver_identity
    assert concentrations_by_uid(solver) == before_by_uid

def test_cache_reorders_columns_by_stable_id():
    restored = restore_cache(saved_ids=("red", "blue"), current_ids=("blue", "red"))
    assert_array_equal(restored[:, 0], SAVED_BLUE)
```

Also assert paused frames launch no exchange kernels, particles continue moving with composition attached by UID, new births receive the authored mixture, derived display remains concentration-driven, and resumed exchange changes the preserved state. Test added-ID zero initialization after replacement admission, removed-ID rejection, exact v1 two-lane translation, incompatible-v1 failure, and playback requirements.

Add a Replacement Cutover test that edits one object's advanced mixture membership N=4→N=5, observes `REINIT_PENDING`, applies reinitialize, proves solver identity and registry fingerprint change only after successful cutover, and proves an injected allocation/configuration failure preserves the original N=4 solver, concentrations, and pending reason.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_species_cache_io.py tests/test_scene_bridge_cache_playback.py tests/test_phase4_pending_interaction.py
```

- [ ] **Step 3: Separate structural registry admission from live physics**

`species_physics_enabled` becomes LIVE. Toggling it changes only whether `exchange()` runs. Registry/storage/palette and source composition remain intact. Registry membership/fingerprint changes remain REINITIALIZE; height initial-fill composition changes remain RESET.

- [ ] **Step 4: Implement schema 2 and constrained migration**

Serialize `(particle_count, active_count)` host concentrations plus canonical registry JSON/fingerprint. Map exact stable IDs on restore. Translate schema 1 only when its two IDs and palette match exactly; otherwise raise `species resume state is incompatible; regenerate the cache`.

- [ ] **Step 5: Run GREEN**

```powershell
python -m pytest -q tests/test_species_cache_io.py tests/test_scene_bridge_cache_playback.py tests/test_phase4_pending_interaction.py tests/test_pending_settings_registry.py tests/test_simulation_admission.py
```

- [ ] **Step 6: Commit**

```powershell
git add -- fluidSim_warp_blender51/sim/solver.py fluidSim_warp_blender51/sim/species.py fluidSim_warp_blender51/io/cache.py fluidSim_warp_blender51/blender/playback.py fluidSim_warp_blender51/blender/pending_settings.py tests/test_species_cache_io.py tests/test_scene_bridge_cache_playback.py tests/test_phase4_pending_interaction.py tests/test_simulation_admission.py
git commit -m "feat(species): add resumable physics toggle and cache v2"
```

---

### Task 8: Adapt the Existing Add-on Panels and Remove Legacy A/B Controls

**Files:**
- Modify: `fluidSim_warp_blender51/properties.py`
- Modify: `fluidSim_warp_blender51/panels.py`
- Modify: `fluidSim_warp_blender51/operators.py`
- Modify: `fluidSim_warp_blender51/blender/pending_settings.py`
- Modify: `fluidSim_warp_blender51/blender/solver_settings.py`
- Modify: `tests/test_species_output_ui.py`
- Modify: `tests/test_operator_panel_behavior.py`

**Interfaces:**
- Consumes: Tasks 2, 3, and 7
- Produces: basic/advanced object authoring in existing physics contexts
- Produces: domain hot toggle and artistic controls
- Removes: `species_count`, `species_id_a/b`, fixed A/B palette controls, `species_max_diffusion_substeps`

- [ ] **Step 1: Write RED panel/property tests**

Assert emitters, mesh/Point/VDB sources, height fill, and injection surfaces expose Liquid Color, Dye Strength, and Initial Mixture; advanced components are collapsed by default. Assert the domain exposes physics toggle, global mixing, and Color Intensity. Assert the Advanced section exposes Mixing Response, Reference Depth, and Carrier Tint with these exact semantics:

```python
effective_fraction = bounded_mix_fraction(rate, dt) ** mixing_response
tau = optical_depth * (
    sum(c_i * absorption_i for i in active_species)
    + (1.0 - sum(c_i)) * carrier_absorption
)
rgb = exp(-tau)
```

Here `optical_depth` is the basic live Color Intensity multiplier. Reference Depth is not another path multiplier: it is the structural calibration distance passed to `absorption_from_display_color()` when definitions and Carrier Tint are compiled. Require finite positive Mixing Response, Reference Depth, and Color Intensity. Reference Depth changes the registry fingerprint and is REINITIALIZE; Color Intensity is live and never changes concentration or registry identity. Assert no species count or substep control is registered or drawn.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_species_output_ui.py tests/test_operator_panel_behavior.py tests/test_pending_settings_registry.py
```

- [ ] **Step 3: Implement collection operators and context-aware panels**

Add/remove/relink mixture-component operators with persistent IDs. Reuse the add-on's existing advanced foldout/tab conventions; do not add a separate top-level species editor. Adapt control visibility to source role and domain-fill/injection contexts.

Add behavioral tests showing Mixing Response changes exchange monotonically without changing conservation, Color Intensity changes derived color without changing concentration, Reference Depth changes compiled absorption/fingerprint and stages reinitialization, and Carrier Tint affects only the clear remainder. These are artistic mappings over authoritative state, not direct RGB replacement.

- [ ] **Step 4: Remove legacy fixed controls and migrate saved scenes**

On load, translate legacy A/B properties into two stable definitions only when the old feature is enabled. Persist the translated collection and schema version; never retain parallel runtime ownership.

- [ ] **Step 5: Run GREEN and Blender UI acceptance**

```powershell
python -m pytest -q tests/test_species_output_ui.py tests/test_operator_panel_behavior.py tests/test_pending_settings_registry.py
blender --background --factory-startup --python tests/blender_phase3_source_ui_acceptance.py -- --json-out artifacts/species_v2_source_ui.json
```

- [ ] **Step 6: Commit**

```powershell
git add -- fluidSim_warp_blender51/properties.py fluidSim_warp_blender51/panels.py fluidSim_warp_blender51/operators.py fluidSim_warp_blender51/blender/pending_settings.py fluidSim_warp_blender51/blender/solver_settings.py tests/test_species_output_ui.py tests/test_operator_panel_behavior.py
git commit -m "feat(ui): add object-first liquid mixture controls"
```

---

### Task 9: Build General Acceptance Scenes, Not Feature-Specific Machinery

**Files:**
- Modify: `tests/blender_species_transport_acceptance.py`
- Modify: `tools/build_demo_scenes.py`
- Modify: `tools/check_demo_blends.py`
- Modify: `tools/analyze_demos.py`
- Create: `tests/blender_dynamic_species_authoring_acceptance.py`
- Create: `tests/blender_species_toggle_resume_acceptance.py`
- Create: `tests/blender_species_cache_resume_acceptance.py`
- Create: `tests/blender_species_source_family_acceptance.py`
- Create: `demos/color_mixing_turbine.blend` through the named builder only
- Create: `artifacts/species_v2_acceptance/` reports during verification

**Interfaces:**
- Consumes: Tasks 3-8
- Produces: executable proof for object authoring, N=4/N=5, purple mixing, toggle/resume, and cache

- [ ] **Step 1: Rewrite acceptance to author through public properties/operators**

The acceptance script must create ordinary fluid objects, set their public Liquid Color/Dye Strength properties, initialize through Simulation Admission, advance real frame handlers, and inspect authoritative concentrations plus derived viewport color. It must not import or call a bespoke species-demo construction helper.

`blender_species_source_family_acceptance.py` runs each production sibling through public RNA/operators: analytic, evaluated mesh, Point direct append, Point rasterized, VDB, height fill, auto-seed, and injection. For each case it proves the deepest emitted particles carry the authored stable species ID/amount. It then disables/removes the final source and proves a second scene sync installs an empty plan/no further births. Helper-level `compile_source_fixture()` tests remain unit evidence only.

- [ ] **Step 2: Add quantitative purple truth checks**

```python
assert mixed_particle_count > 0
assert mixed_surface_vertex_count > 0
assert purple_region_fraction >= 0.05
assert red_mass_error <= 1.0e-5
assert blue_mass_error <= 1.0e-5
assert shader_reads_attribute == "fluidsim_color"
```

The purple reference comes from the same Beer-Lambert contract evaluated independently on CPU, not a hard-coded shader color.

- [ ] **Step 3: Add N=4 and N=5 authoring cases**

N=4 proves four independent object-authored species in one chunk. N=5 proves a second chunk, storage masking, source routing, color accumulation, and performance/accounting based on active chunks.

- [ ] **Step 4: Add toggle and cache acceptance**

Advance to a mixed state, record solver identity and UID→composition, pause physics, advance with moving particles/new births but zero exchange launches, prove display remains concentration-driven, resume the same solver and observe further exchange, save/reopen, bake/read cache, and restore compatible state. Record registry fingerprint at every boundary. Separately edit registry membership N=4→N=5 and prove successful Replacement Cutover plus failure rollback.

- [ ] **Step 5: Build only the named turbine demo and run executable gates**

```powershell
blender --background --python tools/build_demo_scenes.py -- color_mixing_turbine
blender --background --factory-startup --python tests/blender_dynamic_species_authoring_acceptance.py -- --device cuda:0 --json-out artifacts/species_v2_acceptance/authoring.json
blender --background --factory-startup --python tests/blender_species_source_family_acceptance.py -- --device cuda:0 --json-out artifacts/species_v2_acceptance/source_families.json
blender --background --factory-startup --python tests/blender_species_toggle_resume_acceptance.py -- --device cuda:0 --json-out artifacts/species_v2_acceptance/toggle.json
blender --background --factory-startup --python tests/blender_species_cache_resume_acceptance.py -- --device cuda:0 --json-out artifacts/species_v2_acceptance/cache.json
```

- [ ] **Step 6: Verify the saved demo playback contract**

Register `color_mixing_turbine` as an opt-in tracked scene so the canonical 18-scene analyzer count remains unchanged, while `check_demo_blends.py` still treats it as a playable tracked demo. Run the named demo through the real frame handlers after reopen; prove nonzero solver state, changing output, authoritative species concentrations, and purple surface output. Do not rely on saved preview mesh or preloaded frame data.

- [ ] **Step 7: Commit only named acceptance/demo files**

```powershell
git add -- tests/blender_species_transport_acceptance.py tests/blender_dynamic_species_authoring_acceptance.py tests/blender_species_source_family_acceptance.py tests/blender_species_toggle_resume_acceptance.py tests/blender_species_cache_resume_acceptance.py tools/build_demo_scenes.py tools/check_demo_blends.py tools/analyze_demos.py demos/color_mixing_turbine.blend
git commit -m "test(species): prove dynamic authored color mixing"
```

---

### Task 10: Full Verification, Independent Review, and Live Owner Handoff

**Files:**
- Modify: `tools/run_production_gate.py`
- Modify: `tests/test_production_gate_truthfulness.py`
- Modify: `fluidSim_warp_blender51/CHANGELOG.md`
- Create: `artifacts/species_v2_final/` reports during verification

**Interfaces:**
- Consumes: every preceding task and fresh artifacts only
- Produces: final solver/viewport performance receipts, Blender acceptance, production gate, and live review handoff

- [ ] **Step 1: Wire fail-closed species gates into the production gate**

The gate requires current-HEAD hashes, all species acceptance reports, solver performance, viewport performance, active-N accounting, and no active stale cards. Missing, stale, or failing evidence blocks the gate.

- [ ] **Step 2: Run focused contracts**

```powershell
python -m pytest -q tests/test_species_authoring_registry.py tests/test_species_source_lifecycle.py tests/test_scene_bridge_source_plan_reconciliation.py tests/test_species_state_contract.py tests/test_species_diffusion.py tests/test_sparse_species_stencil.py tests/test_species_cache_io.py tests/test_species_output_ui.py tests/test_species_surface_output_gpu.py tests/test_species_viewport_profile_contract.py tests/test_performance_hotpath_contracts.py
```

- [ ] **Step 3: Run fresh performance gates**

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" tools/run_species_transport_profile_repetitions.py --device cuda:0 --frames 60 --warmup 15 --particles 200000 --json-out artifacts/species_v2_final/solver_perf.json
blender --background --factory-startup --python tools/profile_species_viewport.py -- --device cuda:0 --frames 60 --warmup 15 --particles 200000 --json-out artifacts/species_v2_final/viewport_perf.json
```

- [ ] **Step 4: Run project runtime gates**

Create a fresh integration worktree from the completed feature commits using `superpowers:using-git-worktrees`. Run broad gates there so the production gate cannot rebuild or overwrite the Owner's dirty `dam_break` or `crown_splash` files in the primary checkout.

```powershell
python -m pytest -q
python tools/run_production_gate.py
blender --background --factory-startup --python tests/blender_background_acceptance.py
blender --background --factory-startup --python tests/blender_attribute_viz_acceptance.py
```

Add GUI-inclusive production gate only after the background and performance lanes pass:

```powershell
python tools/run_production_gate.py --include-gui
python tools/validate_policy_drift_reports.py
```

- [ ] **Step 5: Dispatch an independent read-only source/runtime review**

The reviewer receives the spec, plan, exact integrated diff, current source hashes, and fresh artifacts. It must verify object-first authoring, dynamic N, disabled zero-work, performance gates, cache/toggle semantics, and that the demo is not a private implementation path. Findings return to the owning task; the reviewer does not edit.

- [ ] **Step 6: Open a profile-safe live review only after all gates pass**

Launch the named turbine scene with the intended temporary add-on copy. In the same process prove target add-on and `blender_mcp` enabled/module paths, listener, `status`, file state, scene info, Python round-trip, live CUDA solver, advancing frame handlers, red/blue sources, mixed concentrations, purple GPU-derived surface color, and paused stable handoff.

- [ ] **Step 7: Final scoped commit**

```powershell
git add -- tools/run_production_gate.py tests/test_production_gate_truthfulness.py fluidSim_warp_blender51/CHANGELOG.md
git commit -m "feat(species): ship dynamic GPU liquid color mixing"
```

## Stop Conditions

- Stop immediately if a change would require importing from an archived experiment.
- Stop if disabled execution changes kernel inventory, topology, allocation, or readbacks.
- Stop at Task 5 if the solver thresholds fail; do not mask cost with changed substeps or workloads.
- Stop at Task 6 if surface cases read particle concentrations/positions or invoke CPU nearest transfer.
- Stop at Task 7 if toggle/resume cannot preserve concentrations without reinitialization.
- Stop at Task 9 if purple is produced by direct RGB blending rather than authoritative concentration.
- Do not claim completion until Task 10's fresh live runtime proof is open and verified.
