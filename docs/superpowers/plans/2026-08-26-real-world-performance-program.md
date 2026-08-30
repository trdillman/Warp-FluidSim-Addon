# FluidSim Warp Real-World Performance Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver measurable whole-frame improvements to live FluidSim simulation and surface playback without lowering simulation quality.

**Architecture:** Use the accumulated performance source to execute three independent material projects: surface/publication, sparse topology, and remaining pressure work. A short whole-frame attribution pass selects the order; it does not reopen broad discovery. Each project has its own clean worktree, correctness contract, paired benchmark, review gate, and retain/reject decision.

**Tech Stack:** Blender 5.1, Python, NVIDIA Warp 1.16, CUDA, NumPy, pytest, Blender background/GUI acceptance.

**Spec:** `docs/REAL_WORLD_PERFORMANCE_SOURCE_20260826.md`

## Global Constraints

- Preserve GPU residency and the fixed flat `8x8x8` sparse leaf-table contract.
- Preserve all source, collider, force, whitewater, surface, pressure, cache, and lifecycle behavior.
- Do not lower quality settings or output cadence.
- Do not reuse archived experiment branches.
- Use immutable clean master/candidate subjects and fresh processes.
- Retain only changes that clear both the target-phase and whole-frame gates.
- Cache replay, headless simulation, and live GUI playback are separate claims.

---

### Task 1: Freeze the durable baseline and produce one materiality report

**Files:**
- Create: `tools/profile_real_world_frame.py`
- Create: `tests/blender_real_world_performance_probe.py`
- Create: `tests/test_real_world_performance_probe.py`
- Create: `artifacts/real_world_performance/baseline/manifest.json`

**Interfaces:**
- Consumes: `Profiler.collect(step_time_seconds=...)`, solver statistics, `SurfacePipeline`, `output_publication`, three tracked demo files.
- Produces: `real-world-frame-profile-v1` with `frame_total_ms`, `solver_ms`, `topology_ms`, `pressure_ms`, `surface_ms`, `publication_ms`, `synchronization_ms`, first-use cost, peak GPU bytes, correctness checkpoint, and runtime provenance.

- [ ] **Step 1: Write schema and rejection tests**

Add tests that reject missing frame totals, overlapping phase totals, negative unattributed time, cache-playback substitution, static output, wrong add-on root, dirty subject, missing peak memory, and absent first-use samples.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest -q tests/test_real_world_performance_probe.py`

Expected: FAIL because the profile schema and probe do not exist.

- [ ] **Step 3: Implement the non-perturbative profile**

Use Warp timing events inside the existing final frame synchronization. Do not insert phase synchronizations. Record explicit scopes around `_rebuild_topology`, `_pressure`, `surface.after_frame`, HostFrame capture, and Blender output publication. Keep `frame_total` authoritative and report `unattributed` separately.

- [ ] **Step 4: Run one raw pass per scene**

Run the real handler path for `dam_break`, `crown_splash`, and `adhesive_calender`, with first-use separated from six warm measured frames. Require non-empty and visibly changing owned output.

- [ ] **Step 5: Select order without reopening discovery**

Rank only surface, topology, and pressure by measured whole-frame milliseconds. A project activates only when its phase is at least 2 ms and at least 10% of the corresponding whole frame. Save the report; do not implement any smaller seam.

- [ ] **Step 6: Commit**

```powershell
git add tools/profile_real_world_frame.py tests/blender_real_world_performance_probe.py tests/test_real_world_performance_probe.py artifacts/real_world_performance/baseline/manifest.json
git commit -m "perf(diagnostics): profile real-world frame costs"
```

### Task 2: Execute the highest measured material project

**Files:**
- Read: `artifacts/real_world_performance/baseline/manifest.json`
- Read: `docs/superpowers/plans/2026-08-26-surface-pipeline-performance.md`
- Read: `docs/superpowers/plans/2026-08-26-topology-frontier-performance.md`
- Read: `docs/superpowers/plans/2026-08-26-pressure-follow-on-performance.md`

**Interfaces:**
- Consumes: Task 1 ranked phase totals.
- Produces: one independently retained or rejected candidate with paired whole-frame evidence.

- [ ] **Step 1: Choose the largest measured phase from the three plans**

Do not choose by implementation ease. Record the phase milliseconds and theoretical maximum whole-frame improvement.

- [ ] **Step 2: Execute its child plan completely**

Follow every red-test, implementation, focused verification, paired benchmark, and review step in that plan.

- [ ] **Step 3: Record RETAIN or REJECT**

Retain only if both the child plan and common whole-frame gates pass. Revert rejected runtime changes while preserving the benchmark receipt.

### Task 3: Execute the second material project

**Files:**
- Read: remaining child plans from Task 2.

**Interfaces:**
- Consumes: first retained candidate as the new integration baseline, while still measuring its own isolated master-derived candidate.
- Produces: a second independent decision and an integrated two-candidate benchmark.

- [ ] **Step 1: Select the next largest material phase**

Skip any phase below 2 ms or 10% after the first retained optimization changes the frame distribution.

- [ ] **Step 2: Execute the child plan**

Keep its implementation and paired evidence independent from the first candidate.

- [ ] **Step 3: Benchmark the integrated pair**

Require the integrated speedup to be consistent with the individual gains and preserve p95, peak memory, and output envelopes.

### Task 4: Execute remaining regular-PCG work only if still material

**Files:**
- Read: `docs/superpowers/plans/2026-08-26-pressure-follow-on-performance.md`

**Interfaces:**
- Consumes: current regular PCG graph implementation and updated real-world profiles.
- Produces: one remaining pressure candidate or a measured defer decision.

- [ ] **Step 1: Re-read the historical no-retry ledger**

Do not retry eager preload, same-leaf branching, 128-thread tuning, or per-iteration readbacks.

- [ ] **Step 2: Execute only if pressure remains at least 20% of the frame**

If it is smaller, stop. The remaining theoretical win cannot justify numerical risk.

- [ ] **Step 3: Run the pressure child plan**

Prefer producer/reduction fusion before warm start. Treat warm start as a separate candidate.

### Task 5: Final live integration and production proof

**Files:**
- Modify: `docs/REAL_WORLD_PERFORMANCE_SOURCE_20260826.md`
- Create: `artifacts/real_world_performance/final/comparison.json`

**Interfaces:**
- Consumes: every retained candidate and its clean evidence.
- Produces: final integrated whole-frame, memory, correctness, and live-Blender verdict.

- [ ] **Step 1: Run focused subsystem suites**

Run all focused commands from each retained child plan.

- [ ] **Step 2: Run the full Python suite**

Run: `python -m pytest -q`

Expected: PASS with no weakened tests.

- [ ] **Step 3: Run the default production gate**

Run: `python tools/run_production_gate.py`

Expected: `PROFILE-GPU-SOLVER-PASS`, `DEMO-BLEND-CHECK-PASS`, and `HEADLESS-GATE-PASS`.

- [ ] **Step 4: Run GUI-inclusive proof**

Run: `python tools/run_production_gate.py --include-gui`

Expected: real handler-driven changing output and the full GUI marker. If the canonical launcher remains blocked, report the exact blocker and do not substitute cache playback.

- [ ] **Step 5: Publish the integrated comparison**

Report first-use, warm median/p95, whole handler frame, solver, surface, pressure, topology, publication, peak GPU memory, and correctness envelopes. State user-visible impact only for scenarios actually run.
