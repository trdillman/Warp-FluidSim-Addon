# FluidSim Warp Performance Optimization Approval Report

Date: 2026-07-11
Hardware: NVIDIA GeForce RTX 4080 SUPER, 16 GiB
Runtime: Blender 5.1.0, NVIDIA Warp 1.12.1, CUDA Toolkit 12.9

## Decision summary

The retained production-safe pressure-solver changes exceed the target on both
measured contracts. The canonical six-frame interactive user path improved
from a 117.66 ms baseline to **60.19 ms and 61.09 ms**, or **48.8% and 48.1%
faster** on two consecutive uncontended runs. The 10-frame GPU solver profile
improved from 98.95 ms to 25.23 ms per step, or **74.5% faster**.

Iteration 14 initially appeared to meet the target by eagerly loading Warp
modules during solver initialization. A truth audit measured initialization at
1591.98 ms with preload versus 761.93 ms without it: approximately 830 ms of
latency had merely moved outside the playback timer. That change was rejected
and removed. Iteration 15 instead captured each unchanged 25-iteration PCG
batch as a CUDA graph. Capture occurs inside the measured first solve, so the
accepted result does not move work outside the timer. Iterations 16-17 were not
needed.

Retained production optimizations:

- reuse recorded Warp launch commands for repeated PCG kernels;
- fuse the PCG `x/r` update with Jacobi preconditioning;
- reuse parity-specific recorded commands for scalar alpha, beta, and packed
  convergence-state kernels;
- launch unchanged 25-iteration PCG batches as bounded CUDA graphs keyed by
  active dimension, batch length, and scalar-buffer parity.

No numerical tolerance, iteration cadence, sparse-storage contract, output
cadence, or public setting changed. Larger numerical or storage changes remain
approval-only options and were not required.

## Benchmark contract

Command:

```powershell
blender --background --factory-startup --python tools/profile_user_path.py -- --frames 6 --voxel-size 0.08 --max-particles 200000 --collider-subdiv 2 --json-out <artifact>
```

Protected path:

`repo add-on load -> Quick Setup -> Initialize Solver -> scene sync -> solver frame -> viewport output`

Baseline runs:

| Artifact | Mean total | Mean solve | Mean viewport |
|---|---:|---:|---:|
| `artifacts/perf_goal_baseline_user_path.json` | 116.91 ms | 94.18 ms | 22.00 ms |
| `artifacts/perf_goal_baseline_user_path_repeat.json` | 118.40 ms | 92.13 ms | 25.53 ms |
| Median baseline | **117.66 ms** | - | - |

The 40% target is **70.59 ms or less**.

## Iteration ledger

| Iteration | Change | Mean total | Result |
|---:|---|---:|---|
| 1 | Fuse PCG update and preconditioner | 105.50 ms | 10.3% faster; focused CUDA tests passed |
| 2 | Check convergence every 5 iterations | 83.50 ms | 29.0% faster; focused tests passed, full gate not yet run |
| 3 | Add 128-thread pressure blocks | 81.08 ms warm | 31.1% faster; first run excluded because Warp JIT compiled for 2.45 s |
| 4 | Check convergence every iteration | 65.71 / 64.20 ms | Rejected: full gate failed pressure divergence and PCG convergence |
| 5 | Separate 25-step stall window from frequent convergence checks | 79.77 ms | Correctness restored; 15/15 validation passed; readback latency remained unstable |
| 6 | Same-leaf neighbor shortcut | 117.13 ms | Rejected and reverted: branch overhead erased the gain |
| 7 | Eager Warp module preload | 88.49 ms | Rejected and reverted: shifted latency without useful total gain |
| 8 | Restore correctness-preserving fused configuration | 131.95 ms | Correct, but per-iteration readback configuration regressed under live load |
| 9 | Restore 25-step checkpoint batching | 165.18 ms | Environment contended; cached module loads roughly doubled |
| 10 | Reuse recorded launches for repeated full-grid PCG kernels | 92.72 ms first run | 23.7% solver-step and 29.8% pressure improvement in focused GPU profile; 15/15 validation passed |
| 11 | Fuse `x/r` update with Jacobi preconditioning | 77.52 ms focused GPU step | Retained after uncontaminated repeat; 18/18 focused tests passed |
| 12 | Tune `k_apply_matrix` to 128-thread blocks | inconclusive/slower | Rejected and reverted; paired timing did not support retention |
| 13 | Record parity-specific alpha/beta/checkpoint commands | 84.32 ms paired 20-frame solver mean | 59.9% faster than paired 210.38 ms baseline; retained |
| 14 | Eagerly load all used Warp modules at Initialize Solver | 40.28 / 51.73 ms playback-only | Rejected and removed: init rose from 761.93 ms to 1591.98 ms, exposing latency shifting rather than a total-path win |
| 15 | Capture unchanged 25-iteration PCG batches as bounded CUDA graphs | **60.19 / 61.09 ms** canonical user path | **48.8% / 48.1% faster; retained after focused and full production gates passed** |

Iterations 8-9 ran while another live Blender process (PID 38684) owned GPU
resources. `nvidia-smi` reported the RTX 4080 SUPER in P2 with 2688 MiB used.
Those runs are retained as evidence but are not used to estimate the accepted
change's speedup.

Extended-lane paired evidence:

| Workload | Baseline | Final | Improvement |
|---|---:|---:|---:|
| 10-frame GPU solver step | 98.95 ms | 25.23 ms | 74.5% |
| GPU pressure phase | 80.87 ms | 9.98 ms | 87.7% |
| Interactive user path, run 1 | 117.66 ms | 60.19 ms | 48.8% |
| Interactive user path, run 2 | 117.66 ms | 61.09 ms | 48.1% |

Accepted run artifacts:

- `artifacts/perf_goal_iteration_15_graph.json`
- `artifacts/perf_goal_iteration_15_user_run1.json`
- `artifacts/perf_goal_iteration_15_user_run2.json`

The rejected preload produced playback-only totals of 40.28 ms and 51.73 ms,
and a production-gate playback total of 48.59 ms. These values are not accepted
performance evidence because the measured interval excluded the added 830 ms
initialization cost.

## Correctness evidence

- Final focused pressure and hot-path suite: 18 tests passed, including graph
  reuse, bounded cache size, odd parity, and voxel-array invalidation.
- Full solver validation after repairing the false-stall condition: 15/15
  tests passed, including pressure divergence reduction and Jacobi/PCG
  residual convergence.
- The invalid target-beating configuration was rejected after the production
  gate reported unchanged divergence and a PCG residual larger than the RHS.
- Sparse storage remains the required flat leaf table with fixed 8x8x8 leaves;
  no octree or hierarchical storage was introduced.
- Final post-graph `python tools/run_production_gate.py` passed on 2026-07-11:
  34 CUDA regressions, smoke core
  and solver, 15/15 solver validation, kernel coverage, GPU profiler, cache
  export/playback, two-frame viewport, background acceptance, attribute
  visualization, four rebuilt demo caches, 39/39 demo analysis checks, saved
  blend checks, and the interactive user-path profile.
- Retained interactive runs preserved the same 50 pressure iterations, substep
  sequence, active-voxel count, approximately 169k mean particles, changing
  non-empty viewport output, and finite converged pressure behavior.

## Optional future approval: flat-array geometric multigrid V-cycle

Recommended approval: **yes, as an isolated pressure-only prototype with the
current PCG solver retained as reference and fallback**.

Scope:

1. Add restriction, coarse-grid solve, prolongation, and smoothing levels in
   independent flat arrays; do not add a tree or octree storage hierarchy.
2. Preserve the sparse 8x8x8 leaf table as the live field-storage contract.
3. Keep current PCG as the reference/fallback and compare residual quality,
   divergence reduction, and visual outputs scene by scene.
4. Change one numerical variable at a time and keep deterministic diagnostics.
5. Accept only if the standard user path is at least 40% faster on two
   consecutive uncontended runs and every production gate passes.

Why approval is required:

- this changes the numerical algorithm and convergence behavior;
- coarse boundary conditions must match the fine sparse domain;
- restriction/prolongation can introduce mass or divergence artifacts;
- additional GPU arrays and level management affect memory and lifecycle.

Rollback: remove the multigrid prototype and use the existing PCG loop. No
cache-file or scene migration is required.

## Deferred structural options

### Compact pressure-unknown index

Build a device-side list containing only fluid, non-collision pressure
unknowns and launch PCG over that list. This could remove work on air voxels,
but requires scatter/gather maintenance whenever topology or voxel type
changes. Approve only after profiling proves inactive-voxel work dominates.

### Optional viewport update cadence

Allow surface/attribute display to update every N solver frames while physics
continues every frame. This can improve perceived playback but intentionally
changes presentation cadence, so it must be explicit UI behavior rather than
an invisible optimization.

## Primary-source research

See `docs/PERFORMANCE_OPTIMIZATION_RESEARCH.md` for the source-cited Warp,
CUDA, sparse-grid, Blender, and profiling research supporting these options.
