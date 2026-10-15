# 4.0 performance investigation, 2026-10-15

Status: verifier repaired; allocation correction behavior verified; matched timing rerun BLOCKED by competing Blender/GPU jobs. This is not a release qualification record. Release identity remains 4.0.0.

## Scope and invariants

Starting source: PR #30 head 295c4cdf291cb96d02b3ded75f9e7eae315aed0a.
Isolated branch: fix/performance-budget-20261015.
No particle limits, resolution, physical-time advancement, tolerance, solver iteration limits, substep limits, force laws, integration, or release thresholds have been relaxed.

## Verifier defect and correction

tools/performance_budget.py previously recognized the intended Multigrid change only when pressure_preconditioner was absent from the baseline schema. An explicit JACOBI field in 3.2.1 therefore made that comparison invalid.

The common-field path now recognizes only the stable, exact JACOBI -> MULTIGRID transition using the existing owned-change declaration and neutral legacy value. It reports both values and applies the existing 1.10 timing ceiling. Baseline drift, candidate drift, a reverse/unknown transition, unrelated settings changes, and all existing workload/provenance guards remain enforced. Identical MULTIGRID runs receive the ordinary 1.05 ceiling. Legacy omitted-field comparisons retain their prior behavior.

Tests reproduce the original rejection, cover a valid transition, an over-budget transition, unexpected settings/workload changes, an unknown/reversed baseline, trial drift, identical Multigrid, and order disagreement.

## Re-evaluation of retained 3.2.1 evidence

Input: C:/Users/Tyler/wt/orch/scratch/abba_4_0_vs_3_2_1/summary.json, source dd04571, 8 trials per variant per demo, frames 2-48.

The figures in the original release status are ratios of the median trial medians. The production reducer instead takes the median of within-block ratios. With noisy block timings these differ materially:

| Demo | Reported host ratio (trial medians) | Gate host ratio (paired blocks) | Gate solver ratio | Existing ceiling | Gate verdict |
|---|---:|---:|---:|---:|---|
| crown_splash | 1.229 | 1.2640 | 1.3390 | 1.10 | regression |
| dam_break | 1.106 | 1.1816 | 1.2379 | 1.10 | regression |
| paddle_stir | 1.017 | 0.9937 | 1.0344 | 1.10 | pass |
| chromatic_vortex | 0.975 | 1.0017 | 0.9687 | 1.05 | pass |

The gate's reduction and thresholds were not changed to improve these results. The crown's claimed 1.155 compound allowance is not encoded in this verifier; no extra rigid-body allowance has been added here. It would not rescue the saved crown result under the paired-block reduction.

## Cost attribution from retained evidence

Synchronized stage medians (median of trial medians; stage medians are not additive):
- Crown collision: 1.166 -> 12.636 ms; pressure: 12.380 -> 14.876 ms; synchronized solver: 43.104 -> 56.972 ms. The new collision path dominates the measured increase.
- Dam pressure: 17.694 -> 24.924 ms; collision: 4.320 -> 4.117 ms; solver: 59.869 -> 67.799 ms. Pressure dominates this increase.
- Crown median adaptive substeps: 8 -> 7; dam: 9 -> 9. These are solver outputs, not authorized reductions to authored workload.
- Host time also includes scene synchronization and output publication. The solver-stage timer does not measure the complete Blender frame. The instrumented whole-frame profiler confirms publication, array copies, launch packing, and pressure-index compaction are material costs.

Scope is shared-path: Multigrid construction/V-cycles execute through PressureSolver.build_preconditioner/solve_pcg, called by FluidSolver._pressure in interactive steps and production bakes. Dynamic coupling executes pose/ranked collision rasterization and RigidCoupling.gather/step for every active dynamic collider. The crown is a reproducer, not the owner of the defect. Dam is the pressure sibling; static-cork and Jacobi controls separate activation paths.

## Iteration evidence and diagnostic controls

The budget child now records the returned status of every production project_frame call, including iteration count, residual, convergence, projection validity, execution mode, and actual convergence-check cadence. It does not read GPU arrays or override solve cadence. Previously frame stats retained only the last substep; they cannot establish a frame's complete iteration workload.

Fresh isolated Blender source probes on Warp 1.18.0 / RTX 4080 SUPER, frames 2-48:
- Crown: JACOBI 356 solves, median 25 iterations, total 10,650; MULTIGRID 358 solves, median 5, total 1,790. All returned converged.
- Dam: JACOBI 402 solves, median 50, total 22,550; MULTIGRID 404 solves, median 10, total 3,650. All returned converged.
- Static-cork control: MULTIGRID 348 solves, median 5, total 1,740; all converged.

These are one diagnostic trajectory per configuration, not eight-trial qualification. Both median iteration ratios are 5x; the total ratios are about 5.95x and 6.18x, with slightly different adaptive substep histories. This does not establish the four-demo release iteration requirement.

The deep probes add cProfile and synchronized section boundaries; their wall times cannot be used as release timings. Nested section times overlap. They confirm V-cycle/hierarchy, pressure-index construction, gather/readback, pose/refit, and host integration costs; they do not prove one exact additive decomposition. The static-cork control changes physical behavior and is diagnostic only.

## Allocation correction and behavior verification

The existing coupling profile spends substantial time in Warp copies and repeatedly allocates an output CPU array via out.numpy() for each substep. The candidate reuses a pinned CPU output buffer, copies on the same null-stream boundary as numpy(), waits before host integration, and returns an independent NumPy snapshot. It changes no arithmetic or physical settings. Allocation-reuse and result-ownership tests cover the boundary; The CUDA tests pass (2/2), including an outstanding non-default producer stream. Four fresh crown runs (two original, two corrected) have identical SHA-256 hashes for every exported checkpoint array and identical checkpoint scalars. No independently measured wall-clock speedup is claimed.

The attempted correction ABBA batch was stopped at a waiting boundary after one block because the activity monitor confirmed foreign Blender processes inside timed runs. The first corrected trial had 13 overlap samples from three other process IDs; the monitor began after the first original trial. Neither this block nor the planned 3.2.1 matched rerun is accepted as timing qualification. Completed, failed and contaminated attempts are retained. The sequential driver was stopped; no benchmark is left running.

## Evidence and reproducibility

Owned evidence directory: F:/FluidSim-performance-evidence-20261015.
run_diagnostics.py uses frozen before/after/reference source copies, a private Blender profile/extensions/cache, the same saved blends for every variant, and Warp 1.18.0. It refuses overlapping Blender processes. Deep ablations are separate from uninstrumented ABBA timing.
The first attempt failed at final checkpoint export because the probe called a nonexistent API; it was corrected to export_checkpoint_state. That attempt and resource-contention refusals are retained and do not count as completed trials.

## Release decisions still required

1. Keep the established per-demo timing budgets until an owner decision changes them. Decide explicitly whether and how to encode the separate coupled-body allowance, with body/workload provenance.
2. Choose the authoritative 4.0 baseline: the current production gate still pins historical 1.x references; this investigation's 3.2.1 reference is an explicit comparison, not a silent gate-policy replacement.
3. Require measured iteration counts for the Multigrid claim across all required demos. A timing pass or fewer iterations alone is insufficient.
4. Do not issue 4.0 qualification without the installed-package, output/GUI, export, platform and other mandatory release checks. This task does not approve those decisions or change release identity.


## Final checks and remaining margin

- Full final CPU suite: **2,876 passed, 338 skipped**, 361.69 seconds. This uses system Warp 1.17.0 with CUDA hidden; it does not certify Warp 1.18 CUDA.
- Focused gate/evidence tests: **86 passed**. The original bug tests failed before the correction, then passed.
- Readback CUDA tests on Warp 1.18.0: **2 passed**.
- Independent read-only code review: no actionable findings.
- git diff --check: clean.
- Ruff reports five inherited diagnostics in touched files: import order and two zip strictness findings in rigid_coupling.py, an existing nested-with finding in test_demo_budget_runner.py, and existing implicit-string-concatenation style in performance_budget.py. No new Ruff diagnostic remains. A clean whole-repository lint/package gate is not claimed.
- No production, installed-package, GUI or release gate was run for this change. No 4.0 qualification or publish claim is made.

The only accepted timing margin remains the retained pre-correction evidence. Under the unchanged paired-block reducer:
- Crown: 1.2640 host / 1.3390 solver against 1.10: **16.40 / 23.90 percentage points over** the ceiling. Even a hypothetical 1.155 compound ceiling leaves **10.90 / 18.40 points over**.
- Dam: 1.1816 / 1.2379 against 1.10: **8.16 / 13.79 points over**.
- Paddle: 0.9937 / 1.0344 against 1.10: **10.63 / 6.56 points below**.
- Vortex: 1.0017 / 0.9687 against 1.05: **4.83 / 8.13 points below**.

These are differences between ratios expressed in percentage points, not newly measured corrected results. The original trial-median host figures (22.9% / 10.6%) remain descriptive, but are not the enforced gate reduction.

Next runtime step: reserve an exclusive GPU/Blender window, rerun the correction ABBA blocks and the matched four-demo, eight-trial-per-variant 3.2.1 comparison using F:/FluidSim-performance-evidence-20261015/run_diagnostics.py. Archive the contaminated correction outputs before rerunning; the driver resumes existing completed JSONs and must not reuse contaminated timings. Continue to monitor the full trial interval, rejecting any overlap. Do not relax thresholds or remove the cork to pass the release comparison.

Compact checked evidence and input hashes are in docs/evidence/performance-20261015/. Full raw reports, source snapshots, profiler dumps and retained failed attempts remain in the owned evidence directory.
