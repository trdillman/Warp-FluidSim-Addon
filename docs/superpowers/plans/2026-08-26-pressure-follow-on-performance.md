# FluidSim Warp Pressure Follow-On Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the remaining 9-10 ms regular pressure phase without repeating the already retained CUDA-graph work or changing convergence quality.

**Architecture:** Preserve the current recorded-command and bounded CUDA-graph batches. Measure remaining kernel/reduction segments, then independently test producer/reduction fusion and exact-signature warm start. ST-PCG graph parity is a separate optional task activated only by an ST workload.

**Tech Stack:** NVIDIA Warp 1.16, CUDA, Python, pytest, Blender 5.1.

**Spec:** `docs/REAL_WORLD_PERFORMANCE_SOURCE_20260826.md`

## Global Constraints

- Preserve pressure tolerance, maximum iterations, 25-iteration checkpoint/stall contract, projection order, and fail-closed behavior.
- Preserve regular PCG graph keys and bounded LRU behavior.
- Do not retry eager preload, same-leaf branching, 128-thread tuning, per-iteration host readback, or the historical rejected configurations.
- Treat changed atomic reduction order as a numerical change requiring calibrated evidence.

---

### Task 1: Remaining pressure-segment attribution

**Files:**
- Modify: `tools/profile_gpu_solver.py`
- Create: `tests/test_pressure_performance_probe.py`

**Interfaces:**
- Produces: assembly, index marking, matrix apply, precondition/update, dot/reduction, graph launch/capture, checkpoint readback, projection, and total pressure timings.

- [ ] **Step 1: Add report tests**

Require cold capture and warm execution to be separate. Bind active leaves/voxels, unknown count, iterations, status, residual, divergence, and whole frame.

- [ ] **Step 2: Instrument existing graph execution with events**

Do not change execution or add per-kernel synchronization. Use graph-boundary and existing checkpoint events.

- [ ] **Step 3: Measure fixed-iteration and normal-tolerance fixtures**

Use at least three pressure-heavy geometries: closed domain, free surface, and moving collider. Continue only if a remaining subsegment can theoretically save at least 2 ms whole-frame.

### Task 2: Producer-plus-reduction fusion

**Files:**
- Modify: `fluidSim_warp_blender51/sim/pressure.py`
- Modify: `fluidSim_warp_blender51/sim/cuda_reduction.py`
- Modify: `tests/test_pressure_projection_regressions.py`
- Modify: `tests/test_pressure_st_and_graph_cache.py`

**Interfaces:**
- Produces separately gated fused kernels for one measured dominant producer/reduction pair.
- Preserves `last_pcg_execution_mode` and graph cache behavior.

- [ ] **Step 1: Freeze numerical baselines**

Capture at least ten baseline replicates of pressure norm, residual path, iteration/status, divergence L1/L2/Linf, projected MAC velocity norms, and full authoritative output.

- [ ] **Step 2: Write failure/status red tests**

```python
@pytest.mark.parametrize("case", ["zero_rhs", "converged", "max_iterations", "stalled", "singular", "non_finite"])
def test_fused_pressure_path_preserves_status_and_projection(case, pressure_fixture):
    assert pressure_fixture.candidate(case).status == pressure_fixture.baseline(case).status
    assert pressure_fixture.candidate(case).project_velocity == pressure_fixture.baseline(case).project_velocity
```

- [ ] **Step 3: Implement only the measured dominant fusion**

Possible interfaces include matrix-apply-plus-dot or update-plus-residual reduction. Do not fuse multiple candidates in one commit. Keep the native reduction graph-capturable and provide the existing fallback on build/load failure.

- [ ] **Step 4: Update recorded commands and graph keys**

Ensure cached graphs cannot replay old kernel layouts. Key/version the command bundle and invalidate existing captures when array identity, active dimension, batch, or fused implementation changes.

- [ ] **Step 5: Run pressure tests**

Run:

```powershell
python -m pytest -q tests/test_pressure_projection.py tests/test_pressure_projection_regressions.py tests/test_pressure_st_and_graph_cache.py
```

- [ ] **Step 6: Benchmark and decide**

Require 20% lower pressure phase, at least 2 ms lower whole frame, CI excluding parity, p95 no worse than 1.02, and calibrated field/output pass. Otherwise revert.

### Task 3: Exact-operator warm start

**Files:**
- Modify: `fluidSim_warp_blender51/sim/pressure.py`
- Modify: `fluidSim_warp_blender51/sim/solver.py`
- Modify: `tests/test_pressure_projection_regressions.py`

**Interfaces:**
- Produces: `PressureOperatorKey` and `PressureSolver.can_warm_start(key) -> bool`.

- [ ] **Step 1: Define the complete operator key**

Include topology membership/generation, active unknown mapping, timestep, density/material fields, boundary/open-wall state, collider state, pressure sources, solver mode, tolerance, and every coefficient affecting the operator or RHS.

- [ ] **Step 2: Add invalidation red tests**

Change each key component independently and require a cold zero initialization. Cover reset, load, rewind, failure, capacity growth, and scene replacement.

- [ ] **Step 3: Preserve the convergence target**

Use prior pressure only as `x0`. Do not change tolerance, iteration cadence, residual definition, or maximum iterations. On any invalid key or previous non-converged status, clear pressure exactly as current production does.

- [ ] **Step 4: Benchmark as a separate candidate**

Report iterations and total pressure time across static/moving topology populations. Retain only if iterations are non-increasing at population level, total pressure and whole-frame time improve, and no invalidation leaks state.

### Task 4: ST-PCG graph parity, conditional

**Files:**
- Modify conditionally: `fluidSim_warp_blender51/sim/pressure.py`
- Modify conditionally: `tests/test_pressure_st_and_graph_cache.py`
- Modify conditionally: `tests/headless_spatiotemporal_validation.py`

**Interfaces:**
- Produces: graph-batched `solve_pcg_st` with explicit `last_pcg_execution_mode` and host-batched fallback.

- [ ] **Step 1: Activate only for a real ST target**

Do not execute this task merely because the source gap exists. Require an actual ST scene where pressure is at least 20% of frame time.

- [ ] **Step 2: Key the graph completely**

Use active `n`, batch size, ST scalar coefficients, device-array identities, topology/material generation, and command-bundle version.

- [ ] **Step 3: Restart before fallback**

Native build/load failure may use the current host-batched path. Capture/launch failure must report the error, restore initialized PCG state, and only then fall back; never continue from partially advanced vectors.

- [ ] **Step 4: Verify and benchmark**

Run:

```powershell
python -m pytest -q tests/test_pressure_st_and_graph_cache.py tests/test_pressure_projection_regressions.py
blender --background --factory-startup --python tests/headless_spatiotemporal_validation.py -- --device cuda:0
blender --background --factory-startup --python tests/headless_stflip_benchmark.py -- --device cuda:0 --frames 6
```

Require 25% lower warm ST pressure and 10% lower ST whole-frame time plus pressure/divergence/MAC/particle/surface envelope pass.

### Task 5: Pressure retention gate

**Files:**
- Create: `artifacts/real_world_performance/pressure/comparison.json`

- [ ] **Step 1: Run seven fresh-process paired trials**

Separate first-use capture/JIT from warm steady state. Record GPU contention, clocks, temperatures, graph hits/misses/captures, and peak memory.

- [ ] **Step 2: Run the full pressure and production suites**

Run:

```powershell
python -m pytest -q tests/test_pressure_projection.py tests/test_pressure_projection_regressions.py tests/test_pressure_st_and_graph_cache.py tests/test_sparse_leaf_grid_contract.py
python tools/run_production_gate.py
```

- [ ] **Step 3: Retain only a real whole-frame win**

The final verdict requires at least 2 ms whole-frame improvement and 20% pressure improvement. A faster isolated reduction with no whole-frame gain is rejected.
