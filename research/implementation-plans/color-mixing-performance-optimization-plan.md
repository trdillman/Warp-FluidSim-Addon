# Color Mixing Performance Optimization Plan

Status: proposed optimization execution plan  
Date: 2026-08-22  
Baseline HEAD: `0d395d78582af1f8876539f0f927110c3eff872d`  
Scope owner: color-mixing lane only

## Decision

Keep the implemented two-species, particle-authoritative clear-dye architecture and optimize it in measured, reversible slices. Do not reverse the lane order and do not broaden this pass into pigments, immiscible density interfaces, or thermodynamic phase transitions.

This remains the correct first lane because it is the smallest truthful visible feature and already exercises the reusable foundations needed later: represented liquid volume, conservative transported state, sparse-leaf stencil traversal, lifecycle accounting, admission, cache/checkpoint persistence, and output publication. Density and thermal work should reuse those seams, not reuse `SpeciesTransport` as a universal physics module.

## Owner goal and truthful boundary

The Owner's broader goal is realistic interaction among colored liquids, different-density fluids, and state transitions. The current delivered slice is intentionally narrower:

- two miscible transparent dyes in one constant-density liquid carrier;
- concentration and represented volume are authoritative particle state;
- diffusion is conservative on the fixed flat 8x8x8 sparse leaf grid;
- display color is derived in scene-linear space with Beer-Lambert absorption.

It does not claim paint/pigment mixing, oil-water interfaces, buoyancy from density, freezing, boiling, steam, or condensation.

## Why this optimization path

Three paths were considered.

1. **Proof-first, semantics-preserving hot-path reduction — selected.** Repair the measurement contract, add a real-solver diffusion-only proof, then remove redundant atomics, launches, and scratch traffic one slice at a time. This is the fastest route to a trustworthy pass while preserving the implemented numerical model.
2. **Immediate diffusion algorithm redesign — deferred.** Keeping mass on-grid across internal diffusion substeps or replacing the global limiter could be faster, but changes the discrete method and numerical behavior. It is only justified if low-risk work cannot meet the immutable budgets.
3. **Benchmark-only tuning — rejected.** Adjusting sampling, denominators, thresholds, or instrumentation without reducing user-path work would not improve the product and would weaken proof.

## Immutable correctness and architecture invariants

- Concentration and represented volume remain authoritative; RGB remains derived.
- `c0 >= 0`, `c1 >= 0`, and `c0 + c1 <= 1`; unused lanes remain zero.
- Owned-face fluxes remain equal and opposite and conservation remains tolerance-based.
- No independent clamping, clamp-then-renormalize, CPU diffusion, or substep readback.
- `D=0` remains a mixing-off identity while lifecycle accounting remains active.
- Particle motion owns advection; species transport owns only miscible diffusion.
- Live storage remains the fixed flat 8x8x8 sparse leaf-table contract.
- There is no diffusion through inactive, air, or solid boundaries.
- Source, sink/removal, growth, compaction, checkpoint, cache, and reopen identity remain complete.
- The disabled feature retains launch, allocation, and readback parity with the pre-feature solver.
- Admission rejects excessive particle, grid, and diffusion-substep demand before replacement cutover.
- Performance budgets remain transport-only `<= 12%` and K=2 `<= 35%`; results are never clamped or selectively discarded.
- CUDA behavior is not claimed bitwise deterministic.

## Contract amendment: diffusion scratch

This plan supersedes the fixed private scratch inventory in `color-mixing-plan.md`. That inventory described one implementation, not externally observable behavior. Algebraically equivalent removal of unused face-delta arrays and recomputable face-conductance arrays is allowed when all numerical, sparse-seam, memory-admission, and lifecycle proofs pass. Tests must assert public numerical behavior rather than private scratch existence.

## Proof contract before optimization claims

The existing profiler is diagnostic only until Task 1 is complete. A performance claim requires:

- an uninstrumented timed pass separated from an untimed work-audit pass;
- raw frame timings and per-repetition results retained in the report;
- exact executable/device, command, timestamp, Git HEAD, dirty-tree fingerprint, and profiler hash;
- measured particle, active-leaf, active-voxel, solver-substep, and species-substep counts;
- explicit diffusion, lifecycle/compaction, source-ledger, receipt, and publication attribution where observable;
- balanced or adjacent-bracketed disabled/enabled order with no best-run selection;
- five independently launched profiler processes, each with at least 15 warm-up frames and 60 measured frames for final proof; an external runner retains every child receipt and validates process/executable identity;
- for every enabled frame, overhead is computed against the arithmetic mean of its immediately preceding and following disabled frames; one process produces the median of those paired frame overheads;
- across the five process medians, let `m = median(r_i)`, `s = 1.4826 * median(abs(r_i - m))`, and the one-sided decision bound be `u = m + 2.5*s`; require `u <= 0.12` for transport-only and `u <= 0.35` for K=2;
- require stability `s <= 0.02` for transport-only and `s <= 0.04` for K=2; these thresholds and the decision formula are frozen before profiling;
- report each process's paired-overhead p95 and absolute incremental p95; require the aggregate worst child absolute incremental p95 to be `<= 2.0 ms` for transport-only and `<= 4.0 ms` for K=2;
- require the median disabled frame time in every child receipt to be `<= 10.0 ms` for this fixed 200,000-particle configuration, preventing a slow denominator from manufacturing a pass;
- validator recomputation of summaries and verdicts from raw data;
- one standalone result and one result in production-gate order;
- unchanged budget thresholds and a baseline ceiling that prevents denominator inflation.

During rapid iteration, a smaller explicitly labelled diagnostic run may decide whether to continue a slice, but it cannot certify the budgets. The interactive user-path profile remains a separate required proof because the species microbenchmark disables Blender publication and surface work.

## Execution loop

Each task follows this loop:

1. Add or tighten the focused regression so the missing behavior fails.
2. Make one bounded runtime change with `apply_patch`.
3. Run the smallest relevant tests and CUDA acceptance.
4. Run an uninstrumented diagnostic profile and compare raw samples.
5. Send the exact diff and evidence to an independent read-only reviewer.
6. Retain the slice only if correctness is unchanged and repeatable user-path cost improves or a proven correctness/proof defect is fixed.
7. If it regresses or is neutral, manually reverse only that slice; never reset, clean, stash, or overwrite the dirty worktree.

No task may raise budgets, omit slow samples, weaken an acceptance test, or synthesize measured receipts from ledger arithmetic.

## Task 0 — Make the existing gate safe in a dirty demo checkout

Files:

- Modify `tools/run_production_gate.py`.
- Modify `tests/test_production_gate_truthfulness.py`.

Work:

- Add a fail-closed `--no-rebuild-demos` option that skips only the canonical demo build step while retaining the existing demo analysis/check and all other default headless lanes.
- Keep the existing default behavior unchanged when the option is absent.
- Emit the selected demo policy in gate output so a receipt cannot hide which path ran.

RED: a focused gate-policy test must prove the current runner attempts demo rebuilding even when a caller needs preservation.

GREEN: the new flag bypasses only the builder; existing demo analysis/check lanes still execute, missing or bad demos still fail, and the default path still executes the builder.

This safety seam is required before the final gate because the checkout contains user-owned modified demo bytes. It is not permission to rebuild, replace, or normalize them.

## Task 1 — Make performance evidence decision-grade

Files:

- Modify `tools/profile_species_transport.py`.
- Add `tools/run_species_transport_profile_repetitions.py` as the external five-process runner/aggregator.
- Modify `tests/test_species_acceptance_report.py`.
- Modify the species performance report validator used by the profiler, if separate.

Work:

- Split the profiler into an unwrapped timed pass and an untimed instrumented audit pass.
- Stop monkeypatching `wp.launch`, synchronization, or Warp array readback during timed samples.
- Use adjacent disabled brackets or a balanced paired schedule and actually use the paired-overhead calculation.
- Emit one versioned child report with raw samples, paired-frame overheads, dispersion, provenance, topology/work counts, and exact check keys.
- Use the external runner to launch and retain all five child reports and derive the aggregate `m`, `s`, `u`, worst-child p95 deltas, and verdict from the frozen formulas above.
- Make validation recompute every summary and verdict and reject missing, stale, non-CUDA, unstable, or internally inconsistent evidence.
- Keep the current artifact and any short run labelled diagnostic; do not promote them as passing proof.

RED:

- A forged report with inconsistent raw timings and summaries must fail validation.
- A report missing provenance, exact checks, CUDA identity, warm-up minimum, or required raw lengths must fail.
- A test must prove timed mode does not install launch/readback wrappers.
- An aggregate receipt with a deleted slow child, mismatched executable/process identity, a disabled median above 10.0 ms, excessive scaled MAD, excessive p95 delta, or a recomputed upper bound above budget must fail.

GREEN:

- Focused profiler/report tests pass.
- A short diagnostic run produces schema-valid raw evidence without timed instrumentation.

Stop condition: do not optimize from percentage deltas until repeated baseline variance and phase attribution are visible. Short diagnostic children may use fewer samples only when their report status is explicitly `diagnostic` and the aggregate validator refuses them as production evidence.

## Task 2 — Prove diffusion in the real solver path

Files:

- Modify `tests/blender_species_transport_acceptance.py`.
- Modify `tests/test_species_acceptance_report.py` and its validator as needed.

Work:

- Add a `FluidSolver.step_frame()` segment with no new source after initialization.
- Prove diffusion alone changes particle concentration distribution and derived particle/surface color distribution.
- Add a long-run real-solver conservation/simplex/receipt check across multiple CFL substeps, including source and removal lifecycle boundaries.
- Preserve the existing direct small transport fixtures as focused numerical oracles, but do not substitute them for the integrated proof.

RED: disabling the diffusion launch must fail the no-new-source mixing metric.

GREEN: the CUDA Blender acceptance reports nonzero mixing change, bounded simplex, closed ledger, and cache/reopen identity through the actual solver lifecycle.

## Task 3 — Correct species growth and resource accounting

This prerequisite runs before allocation-pass claims or scratch-accounting changes.

Files:

- Modify `fluidSim_warp_blender51/sim/particle_capacity.py`.
- Modify `fluidSim_warp_blender51/sim/species.py`.
- Modify `fluidSim_warp_blender51/sim/memory_safety.py` and `fluidSim_warp_blender51/sim/capacity_planner.py` only at the proven shared estimate seam.
- Modify focused growth, rollback, memory-safety, and admission tests.

Sub-slice A — growth accounting:

- Charge every enabled species particle channel and its temporary growth double buffer, including represented volume, concentration, and particle delta.
- Keep host staging separate from device-persistent and transaction-peak bytes so suggestions do not double-count it.
- RED: a species-enabled growth plan using the current fixed byte charge underestimates a worked literal allocation and must fail.
- GREEN: exact integer byte inventory matches the worked allocation; refusal occurs before allocation; rollback preserves the original solver.

Sub-slice B — capacity-axis resize:

- Grow particle-capacity-dependent species arrays without reconstructing or duplicating unchanged voxel scratch.
- RED: an allocation audit must observe duplicated voxel scratch during particle-only growth in the current implementation.
- GREEN: the audit observes only particle-axis replacement state and all growth/compaction/lifecycle tests pass.

The represented-volume-only lifecycle and shared multi-component halo abstraction are explicitly deferred to the density/next-field plan. They require their own complete allocation, source/sink, growth, compaction, cache/checkpoint, restore, teardown, and acceptance matrix; this pass must not create a partial future-lane state.

## Task 4 — Remove avoidable compaction work

Files:

- Modify `fluidSim_warp_blender51/sim/particles.py`.
- Modify `tests/test_species_source_lifecycle.py`.
- Modify `tests/test_particle_capacity_growth.py` if compaction/growth alignment coverage is shared there.

Work:

- Copy represented volume and concentration inside the existing receipt-aware species compaction scatter.
- Skip the two generic optional-channel scatter launches for these fields when the specialized kernel owns them.
- Preserve alignment with every other optional particle channel.
- Replace per-particle AFTER/min/max global receipt contention with a hierarchical reduction only if the fused-copy slice alone is insufficient and the reduction measures actual post-compaction state.

RED:

- Assert mixed optional-channel compaction alignment and sink/after receipts under the existing named floating-point tolerances.
- Assert the audit launch inventory has no duplicate species channel scatters.

GREEN:

- Lifecycle, growth, compaction, cache, and acceptance tests pass.
- The diagnostic profile shows lower D=0 absolute time and launch/traffic counts.

Rollback boundary: copy fusion and hierarchical receipt reduction are separate sub-slices.

## Task 5 — Remove limiter atomics with identical semantics

Files:

- Modify `fluidSim_warp_blender51/sim/species.py`.
- Modify `tests/test_species_diffusion.py`.

Sub-slice A:

- Perform the global-alpha atomic minimum only when the candidate is below one.
- RED: the untimed audit or existing performance-hotpath source contract reports one limiter-min atomic candidate for every alpha=1 particle.
- GREEN: the same audit reports zero alpha=1 atomic candidates while unlimited and genuinely limited numerical cases remain equivalent within existing tolerances.

Sub-slice B:

- Record the already-computed global alpha into the receipt from one thread rather than every particle.
- RED: the untimed audit or performance-hotpath source contract reports one identical receipt write per particle.
- GREEN: it reports one receipt writer per diffusion substep and receipts, bounds, and conservation remain equivalent.

If an audit-only device counter would alter the timed kernel signature or work, use the repository's existing source/IR performance-hotpath contract seam instead; runtime numerical acceptance is still mandatory. The two sub-slices are reviewed and reversible independently.

## Task 6 — Eliminate dead diffusion scratch and traffic

Files:

- Modify `fluidSim_warp_blender51/sim/species.py`.
- Modify `fluidSim_warp_blender51/sim/memory_safety.py` and any capacity/admission estimate derived from the live allocation inventory.
- Modify `tests/test_species_diffusion.py` and focused memory/admission tests.

Sub-slice A:

- Remove `face_delta_x/y/z`, their clears, and their writes because no production kernel consumes them.

Sub-slice B:

- Recompute unchanged owned-face validity/conductance during flux application and remove persistent `face_conductance_x/y/z` arrays.

RED:

- Replace private scratch inspection with a test-owned one-face numerical oracle.
- Assert translated cross-leaf equality, inactive/air/solid barriers, exact allocation estimates, and admission refusal before allocation.

GREEN:

- Numerical acceptance is unchanged and measured active-voxel scratch/traffic falls.
- Each sub-slice independently improves K=2 timing or is reverted.

## Task 7 — Conditional sparse-neighbor fast path

Run only if Tasks 4-6 do not reliably meet both budgets.

Files:

- Modify `fluidSim_warp_blender51/sim/fields.py`.
- Modify exhaustive sparse-grid neighbor tests.

Work:

- For moves that remain inside an 8x8x8 leaf, return the exact flat-index offset directly.
- Consult the leaf table only at leaf seams/domain boundaries.
- Do not cache neighbor indices across topology rebuilds.

RED: an audit-only lookup counter or the existing performance-hotpath source/IR contract must prove that the current interior-neighbor path consults the leaf table. GREEN: it proves interior moves use exact flat offsets while seam moves still consult the leaf table. Also run exhaustive interior, all six seam directions, missing-neighbor, and domain-boundary numerical equivalence followed by every stencil consumer's focused tests.

## Deferred algorithmic options

Only open one of these after Tasks 0-7 and a reviewed measurement showing the budgets still fail:

- keep invariant volume/topology weights on-grid across K internal diffusion substeps and perform G2P once;
- two-stage or spatially local conservative limiter reduction;
- two-lane internal scratch while retaining the four-lane additive persistence schema;
- enabled-only integration with the main liquid transfer kernels.

Each is a numerical or deep-ownership change and requires a separate design review plus Gaussian, cross-leaf, disconnected-region, long-run conservation, lifecycle, cache, and multi-CFL proof. CUDA graph capture, dense storage, hierarchical live grids, and RGB transport remain out of scope.

## Final verification gate

After the last retained slice:

1. Run all focused species, lifecycle, persistence, capacity, memory, admission, and profiler-report tests.
2. Run the real Blender species acceptance and validate its report.
3. Run the external five-process final performance runner with the final schema, plus the same runner in production-gate order; require both immutable budgets, the fixed upper-bound/stability/tail/baseline criteria, and retention of every child receipt.
4. Run `python -m pytest -q`.
5. Run `FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py --no-rebuild-demos` for the required interactive user-path profile and safe default headless gate behavior, using the platform-appropriate environment assignment syntax.
6. Hash the modified tracked demo files before and after the gate and require byte-for-byte identity.
7. Run an independent read-only diff/spec/performance review.
8. Add the species runtime lane to the default production gate only when all proof above is repeatably green; otherwise leave it excluded and report the exact remaining gap.

Completion means lower measured user-path cost with no weakened physics, no missing lifecycle state, no admission regression, and reproducible evidence. It does not mean broadening the color-mixing feature claim.

## Review ownership and cadence

- Orchestrator owns plan decisions, integration, GPU serialization, final evidence, and handoff.
- One implementation owner edits a bounded task at a time.
- A separate read-only reviewer receives each completed diff plus focused evidence before the next dependent slice.
- A final independent reviewer checks the full diff against this plan, the original color-mixing plan, and the Owner's goal.
- No agent stages, commits, cleans, resets, or edits unrelated dirty files.
