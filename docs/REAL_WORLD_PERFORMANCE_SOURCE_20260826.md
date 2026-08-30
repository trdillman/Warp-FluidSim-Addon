# FluidSim Warp Real-World Performance Source

Date: 2026-08-26
Current authority: local `master` at `4f4e44a2b9f7e8de36eb4fbb1587ad36c64b07de`
Hardware: NVIDIA GeForce RTX 4080 SUPER, 16 GiB
Runtime: Blender 5.1.0, Warp 1.16.0, CUDA Toolkit 12.9

## Purpose

This document is the durable source for the next performance program. It consolidates the code review, runtime profiles, historical pressure work, nine bounded 2026-08-26 optimization slices, rejected approaches, benchmark infrastructure, correctness constraints, and live-Blender proof requirements. Future work must begin here instead of repeating discovery.

The objective is not a faster microbenchmark. The objective is a materially smoother interactive simulation and lower production-bake time at unchanged simulation quality, with lower avoidable GPU work and memory traffic.

## Truth contract

- Optimize live simulation, surface generation, and production cache bake separately. Cache replay is not live simulation proof.
- Measure the complete handler-driven or production-bake path as well as the targeted GPU phase.
- Do not reduce particles, voxel/surface resolution, substeps, pressure tolerance/iterations, whitewater, collision coverage, precision, output cadence, or cache contents to manufacture a result.
- Keep simulation GPU-resident and preserve the fixed `8x8x8` sparse leaf tiles addressed through the flat leaf table.
- Preserve transactional topology admission, stable live Warp-array identity, exact failure/recovery behavior, cache-v2 integrity checks, and Blender main-thread ownership.
- Regular CUDA math and marching-cubes output are not globally bitwise deterministic. Use exact checks for topology, counts, settings, cache schema, and non-math refactors; use calibrated envelopes for particle/surface/pressure fields.
- A retained optimization must improve the real target path, not merely shift first-use/JIT/initialization work outside the timer.

## What is already optimized in current master

Do not rediscover or reimplement the following regular-PCG work. The July approval report is `docs/PERFORMANCE_OPTIMIZATION_APPROVAL_REPORT.md`.

- Recorded Warp launch commands are reused for repeated full-grid PCG kernels.
- The PCG `x/r` update is fused with Jacobi preconditioning.
- Parity-specific alpha, beta, and packed convergence-state commands are reused.
- Unchanged 25-iteration PCG batches are captured as bounded CUDA graphs keyed by active dimension, batch length, and scalar-buffer parity.

That program improved:

| Measured path | Historical baseline | Retained result | Improvement |
|---|---:|---:|---:|
| 10-frame GPU solver step | 98.95 ms | 25.23 ms | 74.5% |
| GPU pressure phase | 80.87 ms | 9.98 ms | 87.7% |
| Interactive user path, run 1 | 117.66 ms | 60.19 ms | 48.8% |
| Interactive user path, run 2 | 117.66 ms | 61.09 ms | 48.1% |

Rejected historical pressure changes that must not be retried unchanged:

| Attempt | Result | Decision |
|---|---|---|
| Convergence readback every iteration | Appeared fast but failed divergence/convergence gates | Rejected |
| Same-leaf neighbor branch | 117.13 ms; branch overhead erased gain | Rejected |
| Eager Warp module preload | Moved about 830 ms from playback into initialization | Rejected |
| `k_apply_matrix` 128-thread tuning | Inconclusive/slower in paired timing | Rejected |
| Per-iteration readback configuration | Correct but regressed under live load | Rejected |

## Current measured baseline after those pressure improvements

The 2026-08-26 immutable-base profiles used the exact current master and stable aggregate work:

| Run | Mean synchronized frame | Pressure phase | Active voxels | Particles | Mean pressure iterations |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.9506 ms | 8.998 ms | 27,648 | 58,582 | 68.75 |
| 2 | 27.0345 ms | 10.052 ms | 27,648 | 58,582 | 68.75 |
| 3 | 25.3659 ms | 9.424 ms | 27,648 | 58,582 | 68.75 |

Pressure remains roughly 37% of this synchronized solver workload even after the historical 87.7% pressure improvement. That makes remaining pressure work material, but it is a follow-on optimization—not an untouched subsystem.

Manual user-path reports measured 15.7369-19.7536 ms in their scenario, but that profiler manually calls seams after `scene.frame_set()`. It is useful attribution, not handler-driven GUI acceptance.

Historical viewport publication alone measured approximately 22-25.5 ms in the pressure optimization workload. This is older runtime evidence, but it supports treating surface/publication as a whole-frame target rather than a cosmetic afterthought.

## 2026-08-26 bounded execution ledger

Nine bounded slices were executed. Seven runtime candidates were rejected and reverted. Preserve these results to prevent another micro-optimization loop.

| Slice | Target | Result | Decision / lesson |
|---|---|---|---|
| Directional comparator | Evidence correctness | Fixed false rejection of large speedups | Retained infrastructure |
| Live-prefix NumPy transfer | Surface/output readback | 2.77% median; CI crossed parity | Rejected; not enough whole-frame impact |
| Custom Effects particle active bound | Four effects, 20x capacity slack | 0.012 ms absolute; p95 regressed | Rejected |
| Same-frame source-batch reuse | Four-substep analytic source | 0.0423 ms; p95 regressed | Rejected |
| Whole gathered-scene fingerprint | Static/moving scene sync | 1.59x and 1.16x slower | Rejected; hashing after gathering is the wrong seam |
| Count-only particle occupancy | Sparse/dense, relaxation on/off | No case cleared 0.5 ms; fidelity incomplete | Rejected |
| Duplicate whitewater type reduction | 50k-200k whitewater | 0.1995-0.2453 ms | Rejected below materiality floor |
| Liquid compaction count fusion | 50k-200k particles | 0.0230-0.1469 ms | Rejected below materiality floor |
| Cache-v2 scene validation cursor | 60 distinct frames/leg | 28.1819 -> 19.7783 ms; 30.95%, 8.404 ms | Retained for cache playback only; not live-solver proof |

The cache cursor result is valuable for cache review, but it does not improve the GPU liquid solve. Its contention-aware replacement campaign correctly refused approval on the busy Windows WDDM desktop rather than inventing isolation.

## Reusable infrastructure

The following work exists in the isolated evidence clone at `C:\tmp\liquid-warp-addon-performance-20260826` and should be ported or recreated only when execution begins from a fresh worktree:

- `diagnostics/evidence.py`: clean-subject, runtime, scenario, paired-trial, contention, and correctness binding.
- `diagnostics/equivalence.py`: exact and calibrated particle/whitewater/surface comparisons.
- `diagnostics/profiler.py`: Warp event attribution plus authoritative frame total and unattributed time.
- `tools/validate_performance_evidence.py`: fail-closed performance approval.
- `tools/compare_simulation_outputs.py`: correctness comparison CLI.
- Candidate-specific benchmark harnesses and raw reports under `artifacts/performance_review_20260826/`.

The infrastructure is reusable; rejected runtime edits are not.

## Remaining material hot paths

### 1. Surface construction and viewport publication

Current path:

1. `SurfacePipeline.build()` splats every live particle into a sparse SDF.
2. Multiple filter/dilate/erode/smooth kernels traverse all active surface voxels.
3. An optional render-resolution grid rebuilds sparse topology.
4. `SurfaceExtractor` stages the active-leaf AABB into a dense padded grid and runs marching cubes.
5. Surface positions/triangles and enabled attributes are read to host.
6. Some surface attributes are sampled/derived through host NumPy work before Blender mesh publication.

Key files:

- `fluidSim_warp_blender51/sim/surface.py`
- `fluidSim_warp_blender51/sim/frame_output.py`
- `fluidSim_warp_blender51/blender/output_publication.py`

Why users may feel it: surface work and Blender mesh publication happen on visible frames, and historical viewport publication was 22-25.5 ms. Surface-enabled production demos can be much larger than the solver-only baseline.

First candidate: GPU-resident surface attribute sampling and one bounded host transfer per enabled attribute family, without changing surface topology, point order, cadence, smoothing, resolution, or marching-cubes bounds.

Second candidate: topology-preserving active dense staging that uploads only the exact occupied padded AABB and reuses allocations without clearing or copying unused high-water capacity.

Required keep gate: at least 5 ms or 15% lower complete surface-enabled handler-frame time, p95 no worse than 1.02x, exact enabled attribute arrays/order where the math is unchanged, calibrated surface volume/bounds/Chamfer/Hausdorff, and peak GPU memory no worse than 1.02x.

### 2. Sparse topology construction

Current `SparseGrid` allocates full-domain candidate lookup and flag arrays. Every rebuild performs:

- full `num_leaf_cells` clear of candidate lookup and flags;
- particle/source/collider/height/whitewater marking;
- for every expansion radius, a full-domain flag copy and a full-domain 27-neighbor scan;
- full-domain atomic leaf allocation;
- two scalar host reads for count and overflow;
- candidate-to-live table/coordinate cutover;
- persistent `churn` remap;
- active-range clearing of 11 or more voxel fields.

Topology is rebuilt on the first substep of every frame and on every substep when snippets/effects require it.

Key files:

- `fluidSim_warp_blender51/sim/fields.py`
- `fluidSim_warp_blender51/sim/solver.py`

Non-negotiable invariants:

- fixed `8x8x8` tiles and flat leaf table;
- every particle/source/collider/height/whitewater/CFL/capillary/effect activation source remains covered;
- overflow is known before live lookup/count buffers change;
- live Warp-array identity remains stable;
- churn follows the same world-space leaf across topology changes;
- no speedup may come from narrowing expansion coverage.

First candidate: compact marked-leaf frontier expansion. Mark into the existing flag table, compact marked indices, expand from the compact frontier with duplicate-safe marking, compact final membership, then transact into the existing flat candidate table. This removes repeated full-domain neighbor scans when active leaves are sparse.

Second candidate: after full candidate admission, detect unchanged canonical membership and retain the current slot mapping, avoiding table/coordinate cutover and churn remap. This is only worth retaining if the first attribution report shows copies/remap are material.

Required keep gate: at least 2 ms or 15% lower whole solver-frame time in a production-size sparse-domain scenario, exact canonical leaf-coordinate set/count/required/admitted/overflow/recovery/churn continuity, no memory regression above 1.02x, and affected sibling lanes pass.

### 3. Remaining regular pressure cost

Do not rewrite existing graph batching. Remaining general-PCG opportunities are:

- fuse producer-plus-dot/reduction passes where graph capture still launches separate vector kernels;
- keep convergence/residual decisions device-resident for a bounded batch without changing the existing 25-iteration checkpoint contract;
- reuse the previous pressure as a warm start only under an exact operator signature;
- compact pressure-only active unknowns further only if current indexed work still includes a material inactive fraction.

Key files:

- `fluidSim_warp_blender51/sim/pressure.py`
- `fluidSim_warp_blender51/sim/cuda_reduction.py`

Required keep gate: at least 2 ms lower whole solver frame and 20% lower pressure phase on the current 9-10 ms baseline, with unchanged tolerance/iteration limits, finite pressure/MAC fields, divergence envelope, convergence status, topology/substeps, and correct zero/singular/non-finite failure behavior.

### 4. ST-PCG parity

`solve_pcg_st()` still launches multiple kernels and two reductions per iteration from Python. Regular PCG already has recorded commands, native graph-capturable reduction, bounded graph caching, and execution-mode telemetry.

First candidate: bring ST-PCG to feature parity with regular PCG through a separately keyed CUDA-graph batch and explicit host-batched fallback. This only benefits ST-FLIP users, so it follows general surface/topology work unless an actual target scene uses ST-FLIP.

Required keep gate: 25% lower warm ST pressure phase and at least 10% lower ST whole-frame time, with pressure/divergence/MAC/particle/surface envelopes and restart-before-fallback behavior.

### 5. Synchronization and readbacks

Do not remove the required final frame synchronization. Instead:

- attribute every remaining `.numpy()` scalar read and `wp.synchronize*` call to a correctness boundary;
- pack topology count/overflow and pressure convergence state where it removes a real host round trip;
- use Warp events for profiling without adding phase barriers;
- reject any change that only moves JIT, capture, or initialization outside the measured region.

This is folded into the three projects above, not run as another independent micro-optimization campaign.

## Real-world benchmark matrix

Every candidate uses immutable clean master and candidate subjects in fresh processes with counterbalanced ordering.

| Scenario | User-visible purpose | Required measurements |
|---|---|---|
| `dam_break` live preview | Common liquid playback | handler frame total, solver, topology, pressure, surface, publication, p95, visible frame change |
| `crown_splash` live preview | High particle/surface churn | same plus particle/surface correctness envelope |
| `adhesive_calender` | Production-scale high viscosity/particles | frame total, GPU memory, substeps, pressure/viscosity, surface, admission |
| Pressure-heavy fixed fixture | Isolate remaining regular PCG | pressure GPU event, iterations/residual/divergence/MAC, frame total |
| Sparse-large-domain fixture | Expose full-domain topology scaling | topology split timings, leaf set, overflow/recovery, frame total |
| ST-FLIP fixture | ST-PCG only | cold capture, warm pressure, frame total, ST correctness |

For live claims, run the real Blender GUI handler path and prove visible output changes after reopen. Headless and cache playback remain separate evidence lanes.

## Decision protocol that prevents another seven-hour micro-seam loop

1. Use existing evidence and source maps; no repository-wide re-review.
2. Spend at most one focused attribution cycle on a target. If the target is below 10% of whole-frame time or below 2 ms, defer it immediately.
3. Predict the maximum possible whole-frame win before implementation. Do not implement if even removing 100% of the measured seam cannot clear the keep gate.
4. Implement one material vertical slice per clean worktree.
5. Run a raw baseline leg and raw candidate leg before a full paired campaign.
6. Stop after two materially different failed implementations in one subsystem and move to the next ranked whole-frame target.
7. Never bundle rejected micro-slices to manufacture an aggregate pass.

## Execution order

1. Add one non-perturbative whole-frame breakdown that exposes topology, pressure, surface, publication, and synchronization together in the three real scenes.
2. Implement the highest measured of surface GPU attribute/staging or compact topology frontier. Both have concrete plans; the measurement only decides order.
3. Implement the other material project.
4. Execute remaining regular-pressure producer/reduction or exact warm-start work against the already optimized PCG baseline.
5. Execute ST-PCG graph parity only for a real ST target workload.
6. Integrate only candidates that independently pass whole-frame, correctness, live-Blender, memory, and production gates.

## Known verification state

The isolated 2026-08-26 candidate passed `1292` Python tests, `PHASE4-LIVE-INTERACTION-PASS`, `HANDLER-PERSISTENCE-GATE-PASS`, `DEMO-BLEND-CHECK-PASS`, and `HEADLESS-GATE-PASS`. Foreground GUI performance execution was blocked by the canonical Blender MCP launch receipt `WinError 5`, so that specific claim remains unproven.
