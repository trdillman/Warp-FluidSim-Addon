# Task: Viscosity & Surface-Tension Optimization (remaining work)

Continuation of the dev-loop work recorded in
`docs/superpowers/plans/viscosity-surface-tension-slowdown-optimization.md`.
Read that plan first — it holds every measured number, all 14 hypothesis
verdicts, and the raw-JSON evidence paths. Do not re-derive it.

## Where things stand (measured, not assumed)

Viscosity is **done**: the PCG loop now runs inside a CUDA conditional graph.
Paired benchmark, verified: `viscosity_on` -83.44 %, `both_on` -75.98 %, four
controls inside the +/-5 % noise floor. Viscosity's multiplier fell 7.80x ->
1.55x. That work is uncommitted but green (1402 passed, `HEADLESS-GATE-PASS`).

**Surface tension is untouched and is now the dominant cost.** Post-viscosity-fix,
against a 40.63 ms/frame baseline: viscosity adds 22.17 ms/frame, surface
tension adds **54.85 ms/frame**. The SDF rebuild is 87-95 % of the entire ST
delta, driven by a splat radius that grows 2 -> 7 voxels (a 27x per-particle
write-volume increase, measured 12-15x per-call cost) for a particle that is
0.6 voxels across.

## Hard constraints — violating any of these fails the task

- **The worktree is DIRTY with unrelated in-progress UI/effects/demo work.**
  PRESERVE ALL OF IT. Never `git stash`, `git reset`, `git clean`,
  `git checkout --` a file you did not yourself edit, or `git revert`.
- **Never `git add -A` or `git add .`.** Stage only the explicit file list in
  task 2 below.
- **Never merge or cherry-pick from `fix/settings-apply-domain-object`.** That
  requires an explicit user decision that has not been given.
- **One GPU (`cuda:0`).** Never run two benchmark or profiling processes
  concurrently. Benchmarks are strictly sequential.
- **Do not change `DEFAULT_VISCOSITY_TOLERANCE` or `DEFAULT_VISCOSITY_ITERATIONS`.**
  That is plan item 3f, a physics decision reserved for the user.
- **Evidence standard.** Every headline number must be recomputed from raw JSON
  on disk. No speedup may be claimed without a paired before/after benchmark run
  from one binary with a mode flag. Label conclusions Verified / Likely /
  Assumed. If something cannot run, report the exact command and error — never
  estimate, never fabricate.
- **The full suite must stay green.** Baseline is 1402 passed. Run
  `python -m pytest -q` before declaring any task done.

## Reference material

- `.autonomous/visc-surface-tension-opt/reference/surface-py-band-propagation-WIP.patch`
  — a 442-line WIP implementation of task 3 (closest-point band propagation plus
  a ball-cull mode) produced by an earlier interrupted session. It was reverted
  because it left `tests/test_surface_stack.py` red (2 failed). **Treat it as a
  design reference, not as code to reapply blindly.** Its `"splat"` ball-cull
  mode carries a correctness argument worth reusing: cells outside the ball of
  radius `far_distance + particle_radius` can only produce distances
  `>= far_distance`, which can never win the `atomic_min` against a field
  initialised to exactly `far_distance`, so culling them is bit-identical.
- `docs/superpowers/plans/viscosity-surface-tension-slowdown-optimization.md`
  — the living plan. Update its ROI table and add an "iteration N accounting"
  section as you go.
- Benchmark driver: `tools/profile_viscosity_surface_tension.py`
- Raw evidence: `artifacts/visc_st_measurement/`, `artifacts/visc_pcg_residual/`

## Tasks

- [x] 1. Resolve the interrupted `sim/surface.py` rewrite. Save it as a
  reference patch, restore the file to HEAD, confirm `tests/test_surface_stack.py`
  passes. (Done before this list was written: patch saved to
  `reference/surface-py-band-propagation-WIP.patch`, file reverted, 3 passed.)

- [x] 2. Commit the verified viscosity work. Stage **only** these files and
  nothing else:
  `fluidSim_warp_blender51/sim/viscosity.py`,
  `fluidSim_warp_blender51/sim/solver.py`,
  `fluidSim_warp_blender51/sim/surface_tension.py`,
  `fluidSim_warp_blender51/diagnostics/profiler.py`,
  `tests/test_viscosity.py`,
  `tests/test_viscosity_active_work_regressions.py`,
  `tests/test_viscosity_conditional_pcg_cuda.py`,
  `tests/test_profiler_phase_contract.py`,
  `tools/profile_gpu_solver.py`,
  `tools/profile_viscosity_surface_tension.py`,
  `tools/diagnose_viscosity_pcg_convergence.py`,
  `docs/superpowers/plans/viscosity-surface-tension-slowdown-optimization.md`.
  Verify with `git status --short` afterwards that all other modified entries
  are still present and unstaged. Run `python -m pytest -q` first; do not commit
  a red suite.
  NOTE: `sim/solver.py` and `sim/surface_tension.py` may carry unrelated
  pre-existing user edits. Inspect `git diff` on those two files before staging;
  if they mix task work with unrelated work, stage the task hunks only
  (`git add -p`) and say so in `progress.md`.

- [ ] 3. **ROI item 1 — shrink the surface-tension SDF splat cost.** The largest
  remaining win. Two independent sub-steps; land the safe one first.
  - [x] 3a. Implement the ball-cull ("splat") mode: keep the enlarged band
    radius but skip cells outside the ball of radius
    `far_distance + particle_radius`. This is provably output-identical, so it
    needs no accuracy budget — assert bit-identical SDF output against the
    legacy path in a test.
  - [ ] 3b. Implement closest-point band propagation ("propagate"): splat only
    the radius the particle support justifies, recording the closest particle
    per seeded voxel, then propagate outward over active voxels so cost scales
    with active voxels (~121-131k) instead of particles x R^3 (~244k x 3375).
  - [ ] 3c. Accuracy check, mandatory. The band must remain valid out to
    `support_voxels` — which `required_sdf_splat_radius_voxels` currently
    guarantees by construction, and propagation does not. Quantify the SDF error
    against the legacy path and assert a bound in a test. A propagation scheme
    that silently truncates the band is a correctness regression, not a
    speedup.
  - [x] 3d. Paired benchmark, both modes vs legacy, run from one binary via a
    mode flag on `tools/profile_viscosity_surface_tension.py` (mirror the
    existing `--viscosity-convergence-check {current,legacy-host}` pattern).
    Configs that matter: `surface_tension_on`, `both_on`, and the two
    `no_whitewater` variants. Report per-trial ranges, not just medians.

- [x] 4. **ROI item 2a — cut the ST rebuild count with whitewater on.**
  `build_sdf` runs 2.09x/substep with ST + whitewater vs 1.09x with ST alone.
  Determine whether `SurfacePipeline.after_substep()` can reuse the SDF that
  `_forces()` built this substep. **Evaluate, do not assume:** the comment at
  `solver.py:2170-2174` argues whitewater needs the *post-advection* field, so
  this may be genuinely unmergeable. If it is unmergeable, say so with the
  reasoning and close the item — a refuted item is a result, not a gap. Upper
  bound if mergeable: ~28 ms/frame.

- [ ] 5. Re-measure the viscosity budget/tolerance knee (plan item 3f) against
  the **post-conditional-graph** solver. The old table (-21.9 % at tol 3e-4,
  -32.9 % at tol 1e-3) was computed against the pre-fix 251 ms/frame viscosity
  configuration; viscosity now costs 22.17 ms/frame, so the absolute ceiling has
  collapsed and the old percentages no longer apply. **Do not change any
  default.** Produce a corrected decision table for the user, and state plainly
  that the remaining value of 3f is accuracy (the w component still exhausts its
  budget above tolerance in 31 of 39 solves, median overshoot 2.19x), not speed.

- [ ] 6. Final verification and reporting.
  - [ ] 6a. `python -m pytest -q` — full suite green.
  - [ ] 6b. `python tools/run_production_gate.py` — reaches `HEADLESS-GATE-PASS`.
  - [ ] 6c. Update the plan doc's ROI table and hypothesis verdicts to match what
    was actually measured. If a bound turned out wrong, correct it explicitly
    rather than quietly restating it — the plan already does this for
    hypothesis 14 and that convention should continue.
  - [ ] 6d. Commit the surface-tension work (task-only files, same staging
    discipline as task 2).
  - [ ] 6e. Write a final summary in `progress.md`: measured before/after per
    config, what was refuted, what remains open, and the two decisions still
    reserved for the user (the `fix/settings-apply-domain-object` merge, and
    item 3f).
