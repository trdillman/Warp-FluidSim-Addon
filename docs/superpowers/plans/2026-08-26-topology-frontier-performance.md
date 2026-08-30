# FluidSim Warp Topology Frontier Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated full-domain topology expansion scans with compact active-frontier work while preserving the flat sparse leaf table and transactional admission.

**Architecture:** Keep the existing full flat lookup tables as authoritative live/candidate storage. Mark requested leaves as today, compact marked linear leaf indices, expand only the compact frontier with duplicate-safe flag writes, compact final membership, and transact that membership into the existing candidate table. Slot order remains non-authoritative; canonical leaf membership, overflow, and world-space field continuity are authoritative.

**Tech Stack:** NVIDIA Warp 1.16, CUDA, Python, pytest, Blender 5.1.

**Spec:** `docs/REAL_WORLD_PERFORMANCE_SOURCE_20260826.md`

## Global Constraints

- Preserve fixed `8x8x8` tiles and the flat leaf table.
- Preserve every activation source and requested expansion radius.
- Live lookup/count arrays may change only after admission proves no overflow.
- Preserve stable live Warp-array identity and churn continuity.
- Do not assume atomic slot order is deterministic.

---

### Task 1: Topology split profiler and materiality gate

**Files:**
- Modify: `tools/profile_real_world_frame.py`
- Create: `tests/test_topology_performance_probe.py`

**Interfaces:**
- Produces: per-frame timings for candidate clear, each marking family, each expansion iteration, allocation, scalar admission reads, cutover copies, churn remap, and active field clears.

- [ ] **Step 1: Add topology-report tests**

Reject reports missing any activation family, overflow state, active/domain leaf ratio, or whole-frame binding.

- [ ] **Step 2: Instrument with Warp events**

Do not add synchronization between passes. Collect events at the existing admission/final frame boundaries.

- [ ] **Step 3: Measure sparse/large and dense contrasts**

Use the same domain capacity with 1%, 10%, 50%, and near-dense active leaves. Include particle-only, static/dynamic compiled source, legacy source, collider, height, whitewater, capillary, and per-substep effect activation.

- [ ] **Step 4: Enforce activation**

Continue only if topology is at least 2 ms or 10% of a production-size whole solver frame, and full-domain clear/expand/allocation is the dominant topology component.

### Task 2: Compact marked-leaf membership

**Files:**
- Modify: `fluidSim_warp_blender51/sim/fields.py`
- Modify: `tests/test_sparse_leaf_grid_contract.py`
- Modify: `tests/test_memory_safety_recovery.py`

**Interfaces:**
- Produces: `SparseGrid.compact_candidate_marks() -> int` device count plus device index array.
- Adds device buffers: `_frontier_indices`, `_frontier_count`, `_next_frontier_indices`, `_next_frontier_count` sized to candidate capacity, not `num_leaf_cells`.

- [ ] **Step 1: Add canonical membership red tests**

```python
def test_compact_marks_match_dense_reference(cuda_grid):
    dense_membership = cuda_grid.reference_candidate_leaf_set()
    compact_membership = cuda_grid.compact_candidate_leaf_set()
    assert compact_membership == dense_membership
```

Cover empty, one leaf, boundaries/corners, duplicate particle/source marks, near-capacity, exact capacity, overflow, and all expansion radii used by production.

- [ ] **Step 2: Run red tests**

Run: `python -m pytest -q tests/test_sparse_leaf_grid_contract.py tests/test_memory_safety_recovery.py`

Expected: FAIL because compact frontier buffers and methods do not exist.

- [ ] **Step 3: Implement duplicate-safe compaction**

Use a scan/compaction primitive or bounded atomic append from the existing `leaf_flags`. The resulting index order is not an output contract. Count may remain device-resident until the existing admission boundary.

- [ ] **Step 4: Preserve overflow semantics**

Compaction must record required membership even when it exceeds capacity. Do not truncate the required count before `TopologyCapacityError(required, capacity)` is produced.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest -q tests/test_sparse_leaf_grid_contract.py tests/test_memory_safety_recovery.py tests/test_performance_hotpath_contracts.py`

Expected: PASS.

### Task 3: Frontier-only halo expansion

**Files:**
- Modify: `fluidSim_warp_blender51/sim/fields.py`
- Modify: `tests/test_sparse_leaf_grid_contract.py`

**Interfaces:**
- Produces: `SparseGrid.expand_frontier(radius: int) -> None`.

- [ ] **Step 1: Add dense-reference equivalence tests**

For radii 0-6, compare canonical expanded coordinates against the existing full-domain `k_expand_leaf_flags` result for isolated, boundary, diagonal, clustered, and random memberships.

- [ ] **Step 2: Implement one frontier iteration**

Add a kernel with this interface:

```python
@wp.kernel
def k_expand_compact_leaf_frontier(
    frontier: wp.array(dtype=wp.int32),
    frontier_count: wp.array(dtype=wp.int32),
    leaves_dim: wp.vec3i,
    membership_flags: wp.array(dtype=wp.int32),
    next_frontier: wp.array(dtype=wp.int32),
    next_count: wp.array(dtype=wp.int32),
): ...
```

Each frontier leaf enumerates its 27 neighbors. Use an atomic compare/exchange or equivalent first-writer test on `membership_flags` so duplicates append once.

- [ ] **Step 3: Iterate exactly the requested radius**

Swap frontier buffers per iteration. Preserve clipping at domain boundaries. Do not use a smaller radius based on observed occupancy.

- [ ] **Step 4: Compare launch/work counts**

Prove the candidate touches `O(active frontier * 27)` rather than `O(num_leaf_cells * 27)` in sparse cases, while dense cases remain bounded and correct.

### Task 4: Transactional candidate table construction

**Files:**
- Modify: `fluidSim_warp_blender51/sim/fields.py`
- Modify: `fluidSim_warp_blender51/sim/solver.py`
- Modify: `tests/test_sparse_leaf_grid_contract.py`
- Modify: `tests/test_memory_safety_recovery.py`

**Interfaces:**
- Produces: `SparseGrid.finalize_compact_topology(sync: bool = True) -> int`.

- [ ] **Step 1: Add live-identity and recovery tests**

Capture Python/Warp identities of live table, coordinates, counts, and persistent fields. Force overflow and assert identities and contents remain unchanged. Admit the next valid topology and assert recovery.

- [ ] **Step 2: Fill the existing candidate flat table from compact indices**

Do not replace the flat table with a hash tree or sorted hierarchy. Populate `_candidate_leaf_table[linear_index] = slot` and `_candidate_leaf_coords[slot]` for every admitted member.

- [ ] **Step 3: Preserve the admission readback boundary**

Packing required count and overflow into one status buffer is allowed only as a separately measured sub-slice. The live cutover still occurs after status is known.

- [ ] **Step 4: Run affected sibling lanes**

Run:

```powershell
python -m pytest -q tests/test_sparse_leaf_grid_contract.py tests/test_memory_safety_recovery.py tests/test_source_emission_lifecycle.py tests/test_force_subsystem_contracts.py tests/test_pressure_projection_regressions.py tests/test_viscosity_active_work_regressions.py
blender --background --factory-startup --python tests/blender_memory_safety_gate.py
```

### Task 5: Optional unchanged-membership cutover skip

**Files:**
- Modify conditionally: `fluidSim_warp_blender51/sim/fields.py`
- Modify conditionally: `fluidSim_warp_blender51/sim/solver.py`

- [ ] **Step 1: Inspect Task 1 cutover/remap timing**

Implement this task only if candidate-to-live copies plus churn remap exceed 0.5 ms.

- [ ] **Step 2: Add canonical membership equality**

Compute a device equality result between admitted candidate membership and live membership. Equality must compare membership, not atomic slot numbers.

- [ ] **Step 3: Retain live slot mapping on equality**

Skip table/coordinate cutover and churn remap only when canonical membership is identical. Keep live arrays and `_nvox` authoritative.

### Task 6: Topology baseline-versus-candidate benchmark and retention gate

**Files:**
- Create: `artifacts/real_world_performance/topology/comparison.json`

- [ ] **Step 1: Run seven paired fresh-process benchmark trials**

Require at least 2 ms or 15% lower whole solver-frame median in the sparse-large-domain target, CI excluding parity, p95 no worse than 1.02, and memory no worse than 1.02x.

- [ ] **Step 2: Bind exact topology sidecars**

For every measured frame compare canonical leaf set/count, required/admitted/overflow, recovery, activation-family coverage, churn continuity, and active voxel count. Compare particles, pressure/divergence, whitewater, and surface through the calibrated output envelope.

- [ ] **Step 3: Run the full production gate**

Run: `python tools/run_production_gate.py`

Expected: `HEADLESS-GATE-PASS`. Run GUI live proof separately before a viewport claim.
