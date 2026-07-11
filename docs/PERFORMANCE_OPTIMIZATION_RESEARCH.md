# FluidSim Warp Performance Optimization Research

Date: 2026-07-11
Scope: Blender 5.1 add-on, NVIDIA Warp/CUDA, interactive playback
Status: research and approval brief; no production-code changes made by this research pass

## Executive conclusion

The shortest credible route to a 40% improvement is pressure-first. In the
current phase-profile artifact, warm frames spend **29.563 ms (71.5% of solver
time)** in pressure, while the complete warm-frame path averages **56.844 ms**.
Small reductions elsewhere cannot reach the target by themselves.

The first implementation experiments should therefore be:

1. remove repeated pressure-neighbor address work inside the PCG stencil;
2. replay the repeated PCG launch sequence with `wp.Launch`, then CUDA graph
   capture if the timeline proves launch gaps;
3. profile the pressure stencil with Nsight Compute and tune block/layout only
   from measured occupancy and memory behavior.

Those are compatible with the required fixed 8x8x8 leaf tiles and flat leaf
table. Do **not** replace the field layout with an octree or other hierarchical
tree.

A full multigrid preconditioner or compact pressure-unknown index can plausibly
produce the remaining algorithmic gain, but both are larger structural changes
and need approval before implementation.

## Evidence from the current add-on

This section describes the worktree as inspected on 2026-07-11. The research
agent did not rerun the profiler; values below come from the existing
`artifacts/perf_goal_baseline_user_path_phases.json` artifact.

### Measured frame budget

Warm-frame means exclude frame 0, which includes initialization/warm-up work.

| Component | Warm mean | Share |
|---|---:|---:|
| Total interactive path | 56.844 ms | 100% |
| Solve wall time | 44.165 ms | 77.7% of total |
| Solver phase timer | 41.373 ms | 72.8% of total |
| Pressure (`compute_5`) | 29.563 ms | 71.5% of solver timer |
| Viewport update | 11.927 ms | 21.0% of total |
| Misc (`compute_15`) | 2.653 ms | 6.4% of solver timer |
| Sourcing/topology (`compute_0`) | 2.179 ms | 5.3% of solver timer |
| Whitewater (`compute_12`) | 1.893 ms | 4.6% of solver timer |

The warm frames contain 280,576 active voxels, roughly 155k-194k particles,
one or two substeps, and 50 pressure iterations per substep.

### What 40% requires

- A 40% solve-time reduction means `44.165 -> 26.499 ms`.
- If non-pressure solve cost stayed fixed, pressure would need to fall from
  `29.563 -> about 11.897 ms`, a **59.8% pressure reduction**.
- A 40% end-to-end reduction means `56.844 -> 34.106 ms`.
- If every non-pressure cost stayed fixed, pressure alone would need to fall to
  about `6.825 ms`, a **76.9% pressure reduction**.

Therefore pressure optimization is necessary, but pressure plus a smaller
viewport/other-phase win is the safer route to a 40% end-to-end result.

### Hot-path structure

- `sim/pressure.py::solve_pcg()` batches convergence checks every 25 iterations,
  keeps recurrence scalars on the device, and performs one packed readback per
  batch. This is already the right synchronization direction.
- Each PCG iteration still performs an apply-matrix stencil, two inner-product
  reductions, two one-thread scalar kernels, and three full-domain update
  kernels. With 50 iterations, this creates hundreds of GPU operations per
  substep.
- `sim/pressure.py::neighbor_flat()` reconstructs a neighbor coordinate and
  resolves it through `voxel_flat_index()` for every stencil neighbor. Yet the
  leaf-local layout is x-fastest and 8x8x8; for a given axis, 75% of neighbor
  requests remain inside the same leaf and need no leaf-table lookup.
- `sim/solver.py::_rebuild_topology()` performs one 4-byte topology-count
  readback per substep, then bounds subsequent launches by active voxels. That
  readback is deliberate and the measured sourcing/topology phase is only
  5.3% of solver time; it is not the first target.
- `sim/solver.py::step_frame()` synchronizes once before end-of-frame host
  counts and viewport output. This is required by the current Blender data
  handoff and is not a dominant standalone cost (`sync_ms` is under 1 ms in the
  profile).
- `blender/viewport.py` already uses Blender bulk `foreach_set()` and reuses the
  mesh data-block with `clear_geometry()`. Its remaining cost is mainly
  unavoidable GPU-to-host output plus Blender geometry mutation, not a Python
  per-element loop.
- No `wp.ScopedCapture`, `wp.capture_launch`, or reusable `wp.Launch` path was
  found in the add-on.

## Primary-source findings applied to this code

### 1. Keep the current GPU-resident direction

Warp CUDA launches are asynchronous with respect to the Python host. Host-side
`array.numpy()` synchronizes outstanding GPU work and copies data to CPU, so it
should stay at explicit presentation/diagnostic boundaries rather than enter
inner loops. Warp's synchronization guidance also recommends event/stream
ordering rather than broad device synchronization when only a dependency edge
is required.

Application here:

- preserve the current device-resident PCG recurrence;
- do not reintroduce per-iteration scalar `.numpy()` calls;
- keep surface/particle readback at the Blender presentation boundary;
- if independent work is later moved to another stream, use stream waits or
  events and prove that Blender-visible results are complete before mutation.

Sources: [Warp concurrency and synchronization](https://nvidia.github.io/warp/stable/deep_dive/concurrency.html),
[Warp arrays and NumPy readback](https://nvidia.github.io/warp/stable/user_guide/basics.html#arrays).

### 2. CUDA graphs fit the repeated PCG batch, not the whole dynamic frame

Warp `ScopedCapture` records kernel launches, memory copies, and memsets for
lower-overhead replay. NVIDIA's CUDA graph measurements show why this matters
for long chains of short kernels: repeat launch can collapse per-node CPU
submission overhead, but graph creation/first launch has a cost and reuse is
required. Graph capture also forbids host synchronization inside the captured
region.

Application here:

- the natural capture unit is one fixed `check_every=25` PCG batch;
- the packed residual readback remains outside the graph;
- capture must be invalidated or parameter-updated when device, array pointers,
  active launch dimension, solver mode, or kernel topology changes;
- do not capture the entire frame: topology sizes, source/collider state,
  substep count, surface extraction, and Blender handoff are dynamic;
- measure graph creation and first-launch cost separately from steady replay.

Sources: [Warp `ScopedCapture`](https://nvidia.github.io/warp/stable/api_reference/_generated/warp.ScopedCapture.html),
[Warp graph and stream rules](https://nvidia.github.io/warp/stable/deep_dive/concurrency.html#synchronization-and-graph-capture),
[NVIDIA CUDA graph launch measurements](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/),
[NVIDIA CUDA graphs introduction](https://developer.nvidia.com/blog/cuda-graphs/).

### 3. Try reusable Warp launch commands before graph lifecycle machinery

Warp's `Launch` object stores a configured launch so it can be replayed quickly
and allows dimensions/parameters to be updated. It is a smaller experiment than
graph caching and directly tests whether repeated Python launch construction is
material.

Application here:

- prototype reusable launch commands for the PCG kernels only;
- compare identical 50-iteration pressure solves after warm-up;
- keep the change only if pressure phase and total solve time improve outside
  run-to-run noise;
- move to graph capture only if a CUDA timeline still shows submission gaps.

Source: [Warp `Launch`](https://nvidia.github.io/warp/stable/api_reference/_generated/warp.Launch.html).

### 4. Preserve contiguous x-fastest field access and optimize the stencil

Warp documents that contiguous row-major indexing with the last dimension
varying fastest is the preferred coalesced CUDA access pattern. CUDA best
practices identify coalesced global-memory access and minimized host/device
transfers as high-priority optimizations. The add-on's leaf storage is already
contiguous and x-fastest, so replacing it is unnecessary.

The avoidable work is neighbor resolution:

- current pressure kernels decode `flat -> leaf/local -> global ijk`, then each
  of six neighbors performs domain checks, leaf-coordinate reconstruction,
  dense leaf-table indexing, and slot lookup;
- an in-leaf fast path can compute the neighbor flat index by `flat +/- stride`
  for local coordinates 1..6;
- only local coordinate 0 or 7 needs cross-leaf resolution;
- this removes leaf-table reads and integer address arithmetic from about 75%
  of same-axis neighbor queries without adding memory or changing storage.

Sources: [Warp large-array/coalesced indexing](https://nvidia.github.io/warp/stable/user_guide/basics.html#large-array-indexing),
[CUDA Best Practices: coalesced access and transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations).

### 5. Tile/shared-memory work is conditional on measured reuse

Warp's tile API can stage contiguous data in shared memory, and current Warp
uses vectorized 128-bit loads when alignment conditions are satisfied. Its docs
report up to 1.3x higher bandwidth for the vectorized tile-load path. CUDA best
practices likewise recommend shared memory when it eliminates redundant or
uncoalesced global traffic.

Application here:

- the 8x8x8 leaf is an appealing cooperative unit, but a 512-thread block may
  reduce occupancy depending on register use;
- pressure needs halo values and sparse-boundary handling, so tile conversion
  is not automatically faster;
- first run Nsight Compute on `k_apply_matrix`; only build a tiled version if
  memory workload, cache behavior, and achieved occupancy show a real limiter;
- preserve a flat leaf table. A cooperative kernel is an execution strategy,
  not a new storage hierarchy.

Sources: [Warp tiles and vectorized loads](https://nvidia.github.io/warp/stable/user_guide/tiles.html#vectorized-tile-loads),
[CUDA Best Practices: device memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-spaces).

### 6. Allocation tuning is secondary

Warp's stream-ordered mempool can reduce repeated allocation cost, and raising
the pool release threshold can help allocation-heavy applications. The current
solver mostly allocates persistent buffers at initialization; hot-path source
arrays and surface dense storage are cached/reused. There is no evidence that
allocation dominates the measured frame.

Application here:

- use `ScopedMemoryTracker` or allocator metrics before changing thresholds;
- do not spend an optimization try on mempool tuning unless a profile shows
  per-frame allocation/free traffic.

Source: [Warp allocators and release threshold](https://nvidia.github.io/warp/stable/deep_dive/allocators.html#release-threshold).

### 7. Blender output remains a host-side presentation boundary

Blender frame handlers run as part of playback/render frame changes, and
Blender warns that mutating scene data from handlers can race rendering unless
the interface is locked. Application timers are cooperative callbacks on
Blender's event loop, not a way to make expensive work free. The add-on should
therefore avoid background mutation of Blender data and keep all `bpy` writes
on the supported Blender execution path.

Application here:

- GPU simulation may remain asynchronous until output is required;
- mesh/point-cloud mutation must stay on Blender's main-side callback path;
- a presentation cadence option (surface every N solver frames) can lower view
  cost, but it changes visible behavior and needs explicit approval/defaults;
- do not add a Python thread that mutates `bpy` data.

Sources: [Blender application handlers](https://docs.blender.org/api/current/bpy.app.handlers.html),
[Blender application timers](https://docs.blender.org/api/current/bpy.app.timers.html).

### 8. A true multigrid preconditioner is the strongest algorithmic option

The current `coarse_initial_guess()` is an initial guess, not a full multigrid
V-cycle preconditioner. McAdams, Sifakis, and Teran describe a geometric
multigrid cycle used as a PCG preconditioner for irregular voxelized fluid
domains; their reported convergence is essentially grid-size independent and
far faster than a weak preconditioner at high resolution.

Application here:

- a real V-cycle could reduce 50-75 fine-grid PCG iterations enough to move the
  complete frame target;
- the live field-storage contract must remain fixed 8x8x8 sparse leaves in a
  flat table;
- if approved, coarse pressure levels should be independent flat arrays/tables
  owned by the pressure solver, never a hierarchical tree replacing live field
  storage;
- correctness gates must cover free-surface Dirichlet cells, collider Neumann
  faces, singular domains, residual monotonicity, and deterministic active
  voxel counts.

Source: [McAdams, Sifakis, Teran, *A Parallel Multigrid Poisson Solver for Fluids Simulation on Large Grids*](https://diglib.eg.org/items/c865bec5-c9d1-4f54-afb7-129b64005a07).

## Ranked optimization hypotheses

The gain ranges below are hypotheses, not promises. Each must be measured on
the same warm-frame workload and rejected if it fails the acceptance gates.

| Rank | Hypothesis | Expected target | Risk / size | Approval |
|---:|---|---|---|---|
| 1 | Add in-leaf fast neighbor addressing in pressure stencils | Remove most leaf-table lookups and coordinate math from repeated `k_apply_matrix` work | Small/medium, local to field/pressure helpers | Normal optimization; no structural approval if contracts/tests remain unchanged |
| 2 | Reuse configured `wp.Launch` commands in PCG | Reduce Python-side setup for hundreds of repeated launches | Small, reversible | Normal optimization |
| 3 | Capture one 25-iteration PCG batch as a CUDA graph | Collapse repeated launch submission overhead | Medium lifecycle/invalidation complexity | **Approve before durable integration** |
| 4 | Nsight-guided block size and pressure tile experiment | Improve occupancy or memory bandwidth only if profile proves limiter | Medium; hardware/version sensitive | Prototype allowed; approve before replacing canonical kernels |
| 5 | Compact fluid pressure unknowns into an auxiliary active list | Stop running the linear solve over non-fluid voxels in allocated leaves | Large; new index/remap and neighbor contract | **Approval required** |
| 6 | Build a true flat-array geometric multigrid V-cycle preconditioner | Cut fine-grid PCG iteration count, strongest likely algorithmic gain | Large numerical subsystem | **Approval required** |
| 7 | Optional decoupled viewport surface cadence | Reduce the 11.9 ms presentation cost | Visible behavior tradeoff | **Approval required** |
| 8 | Precompute six pressure-neighbor indices per active voxel | Replace all repeated neighbor resolution with direct gathers | Medium/large memory tradeoff | **Approval required**; prefer rank 1 first |

## Larger structural changes: approval brief

### A. CUDA graph cache for PCG batches

Proposed shape:

- cache one graph for a fixed device, active dimension, array identity set,
  solver mode, and 25-iteration batch topology;
- keep initial residual readback and each batch checkpoint outside capture;
- invalidate on solver reset, capacity reallocation, device change, or a
  captured parameter/dimension that cannot be safely updated;
- expose graph enabled/disabled in benchmark metadata for A/B proof.

Why approval is needed: graph lifetime and invalidation become persistent
solver state. A stale graph can silently launch with outdated pointers or
dimensions.

Acceptance gate:

- identical pressure status, iterations, residual tolerance, active voxel
  counts, and deterministic benchmark fields;
- no graph creation in steady-state warm frames;
- at least 10% pressure-phase gain before keeping the complexity.

### B. Compact pressure-unknown indexing

Proposed shape:

- build an auxiliary list of fluid/non-collision flat voxel indices after
  voxel classification;
- launch pressure kernels over unknown count instead of all active-leaf voxels;
- preserve the canonical flat leaf table for fields and coordinate lookup;
- either precompute compact-neighbor mapping or gather through flat indices.

Why approval is needed: it changes the pressure matrix/index contract and adds
topology-build cost. Benefit depends on fluid occupancy fraction.

Go/no-go probe:

- measure `pressure_unknowns / active_voxels` over representative scenes;
- proceed only when the ratio is consistently low enough that saved iterations
  exceed compaction/remap cost (initial threshold: under 0.65).

### C. Full multigrid V-cycle preconditioner

Proposed shape:

- retain the live 8x8x8 flat leaf grid unchanged;
- create pressure-only coarse levels as independent flat arrays;
- implement restriction, smoothing, coarse solve, and prolongation;
- use the V-cycle as the PCG preconditioner, not only as an initial guess.

Why approval is needed: this is a new numerical subsystem with correctness,
memory, and maintenance cost.

Go/no-go proof:

- existing coarse guess versus V-cycle A/B on dam break, slosh, collider, thin
  free-surface, and singular cases;
- target median fine-grid iterations at or below 20 without worse divergence,
  residual, volume stability, or determinism;
- reject if setup/coarse-level cost erases the frame-time gain.

### D. Presentation cadence / asynchronous handoff

Safe minimal option:

- keep solver stepping every playback frame;
- optionally extract/write surface every N frames while particle/diagnostic
  modes retain their documented behavior;
- never mutate Blender data from a worker thread.

Why approval is needed: output can visibly lag simulation and changes the
meaning of interactive playback.

## Recommended under-10-try experiment sequence

One try means one independently measurable causal change. Do not combine tries
until each change has an A/B result.

1. **Lock the baseline.** Repeat the exact warm workload at least three times;
   record median warm total, solve, pressure, viewport, iterations, particles,
   voxels, and residual.
2. **Pressure neighbor fast path.** Optimize only same-leaf neighbor addressing.
3. **Reusable `wp.Launch`.** Apply only to repeated PCG operations.
4. **CUDA graph micro-prototype.** Capture one 25-iteration batch; do not yet
   add a general cache.
5. **Nsight pressure pass.** Use Nsight Systems for launch gaps and Nsight
   Compute for the dominant stencil/reduction; change block size only if the
   report predicts a limiter.
6. **Unknown-count probe.** Measure fluid pressure unknown ratio and compaction
   overhead without changing canonical field storage.
7. **Compact unknown prototype** if try 6 passes its threshold; otherwise skip.
8. **Multigrid V-cycle prototype** only if the combined measured gain is still
   below 40% and structural approval is granted.
9. **Viewport cadence or output-path experiment** only if end-to-end remains
   short after solve reaches its target.

Stop early when the same acceptance workload reaches at least 40% improvement
with stable correctness. Do not spend remaining tries adding complexity.

## Measurement and acceptance protocol

### Measurement

- Use the existing focused user-path profiler and GPU solver profiler.
- Separate frame 0 compilation/setup from warm steady state.
- Use at least three repeated runs and compare medians, not the best run.
- Keep voxel size, source/collider setup, maximum particles, output mode,
  substep policy, pressure tolerance, and device fixed.
- Record both phase timers and wall-clock frame components; synchronized phase
  profiling perturbs execution and must not be compared directly with an
  unsynchronized wall-time baseline.
- Use Warp CUDA event timing for focused GPU intervals and Nsight Systems to
  identify launch gaps. Use Nsight Compute only on selected kernels because
  replay-based metric collection substantially perturbs runtime.

Sources: [Warp profiling guide](https://nvidia.github.io/warp/stable/deep_dive/profiling.html),
[Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

### Correctness gates

Every kept try must preserve:

- CUDA-resident solver buffers except explicit diagnostics/viewport/export;
- fixed 8x8x8 sparse leaves and flat leaf table, no hierarchical tree storage;
- deterministic active voxel fields required by benchmark comparison;
- pressure status, residual validity, projection behavior, and singular-system
  handling;
- particle/whitewater counts, finite positions/velocities, and topology
  overflow handling;
- non-empty changing Blender viewport output;
- production gate and focused regression tests.

### Performance decision rule

Keep a change only when:

- median target metric improves by more than normal run-to-run spread;
- the improvement survives a second scene or scale;
- no correctness field regresses;
- added complexity is proportional to the measured gain.

## Explicit non-recommendations

- Do not replace the flat leaf table with OpenVDB/NanoVDB/octree storage.
- Do not move Blender data mutation to a background Python thread.
- Do not add broad `wp.synchronize_device()` calls for convenience.
- Do not loosen pressure tolerance, reduce solver iterations, lower resolution,
  suppress surface output, or skip FluidSim features and count that as an
  implementation speedup.
- Do not tune memory-pool thresholds without allocation evidence.
- Do not convert kernels to Warp tiles solely because tiles exist; require an
  occupancy/memory profile and an A/B result.
