# Jacobi-preconditioned PCG: resting-column diagnosis

> **Fixed** on branch `fix/pcg-stall-termination`: a PCG solve now stalls only
> after `PCG_STALL_WINDOW` (100) iterations without a new best residual, in
> both the CUDA-conditional and host-batched paths (`sim/pressure.py`). On the
> tank below, default Jacobi now holds height99 at 0.7421877 with 30/30 solves
> converged (first solve: 75 iterations). Regressions:
> `test_both_preconditioners_hold_the_full_band_column` and
> `test_pcg_stalls_only_after_a_window_without_a_new_best_residual`. The
> probe is `tools/diagnostics/diagnose_resting_column.py`.

## Result and scope

Confirmed shared-path termination defect on 04d732a, not a new narrow-band-only failure and not a 200-iteration budget shortage. Production solver code and FLIP/Jacobi/full-band defaults are unchanged in this task. The diagnostic control is not a proposed production patch or speed optimization.

At a 25-iteration checkpoint the stopping logic treats a non-decreasing Euclidean residual as stalled. A valid PCG residual can increase before later convergence. Both the CUDA-conditional and host-batched branches apply this test, terminate early, and then permit their finite approximate pressure to project velocity. The resulting under-correction repeatedly compresses the column.

The direct production owner is `sim/pressure.py::solve_pcg`; its production projection caller in the same module selects PCG for normal pressure solves. Solver stepping is shared by live preview and Production Cache Bake. Contrasting execution paths were tested explicitly: CUDA conditional versus host batched, full band versus narrow band, Jacobi versus Multigrid. Live viewport and production-bake versions of this particular column were not separately qualified.

## Experiment

Certified Python 3.13 / Warp 1.17.0, RTX 4080 SUPER cuda:0, CUDA driver 13.3. Existing closed 16x16x32 tank, h=1/32, 8 particles/cell, authored fill 0.75, 30 frames. The initial particle-centre height (99th percentile) is 0.7421875; it must not be confused with the authored fill plane. Identical particle initialization and physics settings were used except for the named diagnostic control. The per-solve probes are instrumented and are not timing benchmarks.

| Control | Final height99 | Final particles | Solve outcomes |
|---|---:|---:|---|
| Default Jacobi, limit 200 | 0.5578094125 | 49,152 | 25 stalled / 31 converged |
| Jacobi, limit 800 | 0.5578094125 | 49,152 | 25 stalled / 31 converged |
| Jacobi, host-batched instead of CUDA-conditional | 0.5578094125 | 49,152 | 25 stalled / 31 converged |
| Jacobi, check only at iteration 200 | 0.7421875000 | 49,152 | 30 converged |
| Multigrid preconditioner | 0.7421737313 | 49,152 | 30 converged |
| Jacobi, narrow band | 0.5506955981 | 5,682 | 29 stalled / 22 converged |

Full-band particle count stays 49,152 throughout. Its height99 falls about 24.84%; no particle deletion explains that result. The occupied-cell measure is a geometric proxy, not an exact mass/volume integral. Narrow-band interior particle removal is intentional, so its reduced particle count is not evidence of physical mass loss.

On the first default solve, the initial RHS norm is 0.2043747753. It stops at iteration **25 of 200** (also 25 of 800) with independently recomputed `||b-Ax||=0.2684199512`, against the implemented acceptance norm 0.0009050967. The single-check diagnostic reaches a true residual 0.0000174751, below acceptance, while keeping the column stable. Host-batched output reproduces the CUDA-conditional collapse, ruling out conditional graph replay as the unique cause. The recursive residual at full convergence is smaller than the independently recomputed residual; floating-point recurrence values are not exact-error certificates.

## History and test interpretation

The same non-monotonic-residual stopping predicates exist in both v2.0.0 and v2.1.0 source tags (host and CUDA branches). Thus the predicate predates 3.0. Those release binaries were not replayed against this tank, so this is not a bisect of the first visibly affected release. Git history includes the predicate before the 3.0 narrow-band work.

`tests/solver/test_narrow_band_acceptance.py` describes failure to converge within the iteration budget and explicitly asserts Jacobi height below 0.6. The 800-iteration control disproves that budget explanation for this reproducer. This is a characterization of an existing defect, not a passing hydrostatic correctness gate.

## Recommended subsequent repair

Replace the shared one-check monotonicity assumption with a justified stagnation/breakdown policy that permits valid non-monotonic PCG progress. Preserve finite/NaN/singular safeguards, convergence tolerance and actual iteration limits in both execution branches. Add a regression that fails for premature termination on a non-monotonic yet convergent SPD problem and a real resting-column invariant under default settings. Measure true residual, divergence and height across sibling paths. Do not simply raise the iteration cap, switch the default preconditioner, force 200 iterations in production, relax a correctness gate, or change substeps to claim a speed gain.

## Evidence

Root: `F:/FluidSim-post30-20261010/evidence/`.
`jacobi-diagnostic.json` contains raw frame/solve data; SHA-256 `2d5f46241bf1fcc4f35c188ee91fe1034e7a65dfb10884f505dcd7878442db22`.
`jacobi-diagnostic.log` records actual CUDA module/device execution.
`jacobi-summary.json` extracts reason counts from the original dictionary-string status field without changing raw measurements.
`diagnose_resting_column-initial.py` is the actual measured probe; the checked-in probe only simplifies later status reporting. `jacobi-stop-history.log` records the source-history search. All solver files still match 04d732a.

No independent subagent review was available through the permitted routes; this is a self-reviewed, controlled experimental diagnosis. Further solver repair is deliberately not bundled with the authorized Append/VDB integrations.
