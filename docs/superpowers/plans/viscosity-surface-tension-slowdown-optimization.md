# Plan: Viscosity & Surface-Tension Slowdown Optimization

Source task: optimize the viscosity and surface-tension slowdowns in the
FluidSim Warp/Blender solver (`sim/viscosity.py`, `sim/surface_tension.py`,
`sim/surface.py`, `sim/solver.py`). This is a living plan for a
self-referential development loop.

## Loop status

| Phase | State |
| --- | --- |
| 1a — profiler attribution fix | **Done and verified** (iteration 2) |
| 1b — paired measurement | **Done** (iteration 2); raw JSON on disk, paths below |
| 2 — plan / ROI ordering | **This document**, re-ranked by iteration 3 |
| 3a — viscosity PCG convergence diagnosis | **Done** (iteration 3); raw JSON on disk, paths below |
| 3b — tighten the convergence-check cadence | **Diagnosed and REJECTED** (iteration 3). Best available cadence is worth -1.36 % of the viscosity-on frame, below the measurement noise floor. Not implemented. |
| 3e — on-device viscosity convergence check | **Done and verified** (iteration 4). First production performance change in this loop. |
| 3 — implementation | **In progress.** One of two headline items done (viscosity 3e). Surface tension (ROI item 1) is untouched. |
| 4 — verification of a speedup | **Done for item 3e only.** Paired benchmark: viscosity_on -83.44 %, both_on -75.98 %, four controls inside noise. No speedup is claimed for surface tension. |

Blocked and explicitly out of scope: quality-tier scaling of
`viscosity_iterations`. `sim/quality_contract.py` does not exist on
`feat/ui-redesign-phases-0-3` or on `master`; it exists only on the unmerged
`fix/settings-apply-domain-object` branch. Pulling from that branch is not
authorized. See "Blocked work" in the ROI list.

## Phase 1a — profiler attribution: implemented and verified

### The instrumentation change

`SurfaceTensionSolver.apply()` (`fluidSim_warp_blender51/sim/surface_tension.py`)
now wraps its solver-SDF dependency in its own phase span:

```python
with self.solver._phase("surface"):
    sdf_current = self.solver.surface.ensure_solver_sdf_current()
```

`apply()` is called from `_forces(dt)` (`solver.py:1635`), which `step_frame()`
wraps in `with self._phase("forces")`. Before the change, the surface-tension
SDF rebuild was invisible inside `compute_3` ("forces"). It now also lands in
`compute_11` ("surface"), the same slot `SurfacePipeline.after_substep()` and
`after_frame()` already use for the identical `build_sdf()` work. No new
profiler slot was added: `TIMER_PHASE_SLOTS` already occupies all sixteen
`compute_0..15` slots and `tests/test_profiler_phase_contract.py` asserts the
slot set is exactly `range(16)`.

**Verified**: the edit executes. `compute_11` is populated in every
surface-tension configuration of both measurement replicates below; it is
~0.2-2.5 ms/frame with ST off and 30-60 ms/frame with ST on.

### The profiler nesting defect, fixed

`Profiler.track()` (`fluidSim_warp_blender51/diagnostics/profiler.py`) was not
nesting-aware. On exit it unconditionally added elapsed time to `timers[slot]`,
`timers["compute"]` and `timers["solver"]`, so the new nested "surface" span
double-counted into the frame aggregates.

`track()` is now nesting-aware via a `_track_depth` counter on the profiler:

- every span still accumulates into **its own** `compute_<slot>` track, so
  per-slot semantics are unchanged and slots remain supersets of the spans
  nested inside them;
- only the **outermost** open span contributes to the aggregate `"compute"` and
  `"solver"` tracks, so overlapping spans can no longer inflate frame totals;
- the depth unwinds in a `finally` block, and `begin_frame()` resets it to 0 so
  a span abandoned by a hard failure cannot bias later frames.

**Sync semantics under nesting (deliberate).** When `sync=True` against a CUDA
device, `track()` still issues `wp.synchronize_device` on entry and on exit of
**every** span, inner spans included. A nested span therefore costs two extra
device syncs that a flat span would not: the inner *entry* sync drains work
issued by the enclosing span and is charged to the enclosing span; the inner
*exit* sync is charged to the inner span. This is the behaviour we want — the
inner slot must not be credited with work the outer span queued — but it means
the `sync=True` lane's **absolute** times are inflated wherever spans nest, and
that lane is for attribution only. All headline wall-clock numbers below come
from the separate `sync=False` lane. Suppressing inner syncs was rejected: it
would let asynchronous GPU work drift across the inner span boundary and
silently mis-attribute it.

**Regression tests** — `tests/test_profiler_phase_contract.py`, five new tests
(file now has 10, all passing):

- `test_nested_phase_spans_do_not_double_count_frame_aggregates` — with
  "surface" nested inside "forces", `compute_3 >= compute_11`,
  `compute == solver == compute_3`, and depth returns to 0.
- `test_sibling_phase_spans_still_sum_into_frame_aggregates` — flat spans still
  sum: `compute == compute_3 + compute_4`.
- `test_nesting_depth_unwinds_on_exception_and_resets_each_frame`.
- `test_nested_spans_keep_per_span_cuda_synchronization` — asserts exactly four
  `synchronize_device` calls for two nested spans, pinning the documented sync
  behaviour above.
- `test_sync_disabled_profiler_never_synchronizes`.

**Rejected alternative (unchanged from iteration 1):** moving
`surface_tension.apply(dt)` out of `_forces()` into its own top-level,
non-nested phase in `step_frame()`. Today ST's grid force is applied *before*
the `max_speed` clamp inside `_forces()` (`solver.py:1621-1650`); moving the
call would let ST's velocity contribution skip that substep's clamp. That is a
physics change requiring its own correctness pass, and the nesting-aware
profiler removes the only reason to consider it.

## Phase 1b — measurement

### Harness

`tools/profile_gpu_solver.py` was extended (bpy-free, already had `Profiler`
wiring and JSON output) rather than `tools/profile_user_path.py` (1583 lines,
bpy-bound). Additions:

- a `dam_break` scene preset mirroring `tools/build_demo_scenes.py:build_dam_break()`
  — the solver-visible configuration of `demos/dam_break.blend`: domain
  `(0, -1.00146496, 0)`-`(5.6, 1.00146496, 1.6)`, `voxel_size 0.03`,
  `max_particles 600000`, substeps 1-16, whitewater on, and the analytic box
  source column (centre `(0.3, 0.3, 0.6)`, half extents `(0.3, 0.3, 0.6)`,
  emit rate 8.0). Explicit CLI flags always override the preset;
- `--viscosity`, `--viscosity-constant`, `--viscosity-iterations`,
  `--viscosity-tolerance` (default: leave the solver default alone),
  `--surface-tension`, `--surface-tension-coefficient`, `--no-whitewater`;
- per-phase means for `compute_3/4/11/12` and a `sdfSplatRadiusVoxels` summary
  field;
- `solver.release()` in a `finally`, so a paired sweep that builds 80+ solvers
  in one process does not accumulate device allocations;
- an `on_frame` callback so external probes can zero their counters at the
  warmup boundary.

`tools/profile_viscosity_surface_tension.py` is the new paired driver. It runs
six configurations — `baseline`, `viscosity_on`, `surface_tension_on`,
`both_on`, `baseline_no_whitewater`, `surface_tension_on_no_whitewater` — in
two lanes, interleaving trials across configurations, and writes raw + aggregate
JSON. It also installs measurement-only monkeypatched probes (production code
untouched) around `ViscositySolver.solve`, `ViscositySolver._solve_component`,
the Warp array `.numpy()` readback, `SurfacePipeline.ensure_solver_sdf_current`
and `SurfacePipeline.build_sdf`, and samples `nvidia-smi` clock/utilisation
before and after every run.

### Two lanes, and why

- **wall lane** (`sync=False`): the headline. `stepTime` is measured in
  `step_frame()` across an explicit `wp.synchronize_device`, so it is true
  wall-clock cost. Per-phase timers in this lane are launch-side only and must
  not be read as GPU time.
- **phase lane** (`sync=True`): attribution only. Per-span device syncs inflate
  absolute times (and more so where spans nest — see Phase 1a). Use it for the
  *shape* of the breakdown, never for headline totals.

### Phase-slot correction — the task brief was wrong on one point

The brief stated "forces-net-of-ST = `compute_3 - compute_11`". **That is not
true**, and measurement caught it: the first smoke run reported
`compute_3 = 17.5 ms` and `compute_11 = 35.9 ms`, impossible if `compute_11`
were a subset of `compute_3`. Three call sites accumulate into `compute_11`:
the ST SDF rebuild inside `_forces()` (nested in `compute_3`),
`SurfacePipeline.after_substep()` (once per substep whenever whitewater is
enabled, **not** nested in forces), and `SurfacePipeline.after_frame()` (also
not nested). Isolation therefore comes from paired deltas, not subtraction:

- `compute_11(ST on) - compute_11(ST off)` = what the ST SDF rebuild adds;
- `compute_3(ST on) - compute_3(ST off)` = the rest of the ST force cost.

### Raw evidence on disk

| Replicate | Raw JSON | Aggregate JSON |
| --- | --- | --- |
| R1 (contended) | `artifacts/visc_st_measurement/visc_st_raw_20260902T163333Z.json` | `artifacts/visc_st_measurement/visc_st_aggregate_20260902T163333Z.json` |
| **R2 (clean, headline)** | `artifacts/visc_st_measurement/visc_st_raw_20260902T164256Z.json` | `artifacts/visc_st_measurement/visc_st_aggregate_20260902T164256Z.json` |

R2 command:

```
python tools/profile_viscosity_surface_tension.py --device cuda:0 --frames 8 \
    --warmup 2 --trials 7 --emit-frames 6 --scene-preset dam_break \
    --out-dir artifacts/visc_st_measurement
```

Device NVIDIA RTX 4080 SUPER (`cuda:0`), Warp 1.16.0, Python 3.14.3. Statistic:
median over trials of the median over that trial's 8 measured frames. Every
number below was recomputed independently from the raw JSON, not copied from
the tool's own printed table.

**R1 is disclosed, not hidden.** R1's per-trial medians drifted downward
mid-run — `baseline` phase-lane trial medians ran 96.8, 92.0, 93.5, 82.5,
38.0 ms with *identical* particle counts, voxel counts, substeps and pressure
iterations, and every phase slot scaled down together. That is GPU contention
on a shared desktop (Chrome, Houdini, two Electron apps and the Windows
compositor are all resident graphics clients), not a workload change. R2 was
run after that contention cleared: its per-trial spread is tight
(`baseline` wall lane 31.06-34.40 ms across 7 trials) and its sampled SM clock
stayed in 2580-2940 MHz. R1's *ratios* were directionally the same but
20-25 % higher for surface tension; R2 is the conservative headline.

### R2 — wall lane, ms/frame

| Config | ms/frame | ms/substep | substeps | particles | voxels |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 32.26 | 5.71 | 6 | 243906 | 131072 |
| viscosity_on | 251.43 | 50.77 | 5 | 230460 | 120832 |
| surface_tension_on | 89.03 | 15.35 | 6 | 243879 | 131072 |
| both_on | 291.93 | 58.99 | 5 | 230302 | 120832 |
| baseline_no_whitewater | 29.89 | 5.29 | 6 | 243937 | 131072 |
| surface_tension_on_no_whitewater | 59.00 | 10.24 | 6 | 243876 | 131072 |

Median of **per-trial paired ratios** (config divided by its control within the
same trial, which cancels drift shared by both):

| Comparison | ratio | range over 7 trials |
| --- | ---: | --- |
| viscosity_on / baseline | **7.80x** | 7.17-8.26 |
| surface_tension_on / baseline | **2.73x** | 2.46-2.89 |
| both_on / baseline | **9.05x** | 7.89-9.68 |
| surface_tension_on_no_whitewater / baseline_no_whitewater | **1.98x** | 1.86-2.13 |

**Confound, disclosed.** Viscosity changes the dynamics: viscosity-on frames
run at ~230.5k particles / 120832 voxels / 5 substeps versus ~243.9k / 131072 /
6 for viscosity-off. The viscosity-on configurations therefore do *less* work
per frame and are still 7.8x slower — the confound understates the viscosity
cost, it does not inflate it. Normalising per substep gives 5.71 -> 50.77 ms
(8.89x), i.e. the same conclusion. Surface-tension configurations show no such
confound: particle count, voxel count and substep count all match their control.

### R2 — phase lane, median of per-trial deltas (ms/frame, attribution only)

| Comparison | `compute_3` forces | `compute_4` viscosity | `compute_11` surface |
| --- | ---: | ---: | ---: |
| viscosity_on - baseline | — | **+223.5** [205.7, 258.4] | — |
| surface_tension_on - baseline | **+29.2** [26.8, 29.8] | — | **+57.6** [56.0, 60.3] |
| surface_tension_on_no_whitewater - baseline_no_whitewater | **+28.4** [26.4, 28.9] | — | **+30.2** [28.1, 30.9] |
| both_on - baseline | +23.0 | +228.7 | +48.5 |

Pressure (`compute_5`) moves -3 ms/frame with viscosity on, consistent with
viscosity pre-smoothing the velocity field and with the smaller active grid.

### Requirement (d) — cost decomposition, from the probe runs

Probes are timed with explicit device syncs around the probed call only.

**Surface tension — the per-substep SDF rebuild is essentially the whole cost.**

| Comparison | step ms/frame | `build_sdf` ms/frame | rebuild share of the ST delta |
| --- | --- | --- | ---: |
| ST off -> on, whitewater **off** | 31.94 -> 63.57 (**+31.63**) | 0.41 -> 30.36 (**+29.96**) | **94.7 %** |
| ST off -> on, whitewater **on** | 37.18 -> 103.27 (**+66.09**) | 2.33 -> 60.09 (**+57.76**) | **87.4 %** |

Two independent multipliers produce that, and the second is the bigger one:

1. **Call rate.** `build_sdf` runs 0.18x/substep with ST and whitewater both
   off, 1.09x/substep with ST only, 1.18x/substep with whitewater only, and
   **2.09x/substep with both**. ST adds exactly one rebuild per substep; with
   whitewater there are two. Call counts are exact:
   `ensure_solver_sdf_current` was called 98 times over 8 frames / 45 substeps
   with ST + whitewater = 45 (forces) + 45 (`after_substep`) + 8 (`after_frame`).
   The cache hit rate is **4.1 %** (4 hits in 98 calls) — the `sim_time`
   component of `solver_sdf_build_key()` does prevent essentially all reuse.
2. **Per-call cost, ~13x.** The same `build_sdf` costs 0.35-0.41 ms/call with ST
   off and **4.96-5.11 ms/call with ST on — 12.2x to 14.6x**. Cause, verified in
   `surface.py:158-202`: when the target is the solver grid and surface tension
   is enabled, the particle splat radius is enlarged to guarantee the
   normal/curvature support band. At the benchmark settings the recorded
   `sdfSplatRadiusVoxels` goes from **2 to 7**
   (`required_sdf_splat_radius_voxels(0.6, band 1.5, smoothing 1, filters 1)`
   = `ceil(0.6 + 4.5 + 1.0)` = 7), so `k_build_surface_sdf` writes a
   15^3 = 3375-voxel cube per particle instead of a 5^3 = 125-voxel cube — a
   **27x increase in per-particle write volume** for a particle whose actual
   radius is 0.6 voxels.

**Viscosity — the three PCG solves are the whole cost; the readbacks are not.**

| Component | ms/frame | share of the +222.97 ms/frame viscosity delta |
| --- | ---: | ---: |
| 3 per-component PCG solves (u/v/w = 76.3 / 76.2 / 75.8) | **228.32** | **102.4 %** |
| residual `.numpy()` readback stalls (456 calls = 12/substep) | **7.84** | **3.5 %** (3.43 % of PCG time) |

The solves measure slightly over 100 % of the delta because the probe's syncs
charge otherwise-overlapped launch work into the measured span and because the
viscosity-on run has fewer substeps than its control. The three components cost
within 0.7 % of each other, as expected for three structurally identical solves.

**The PCG never converges.** Across 114 component solves the measured iteration
count was **exactly 40.0 every time** — the full `DEFAULT_VISCOSITY_ITERATIONS`
budget — at the shipped default `DEFAULT_VISCOSITY_TOLERANCE = 1e-4`. Zero early
exits. Each component solve is therefore ~16.0 ms for 40 iterations at 120832
active voxels, about 0.40 ms per iteration, and each iteration issues ~9 kernel
launches plus 2 `array_inner` reductions — including two `dim=1` kernels
(`k_compute_alpha`, `k_compute_beta`) and a 4-byte `wp.copy`, i.e. ~360 launches
per component solve and ~1080 per substep.

## Phase 3a — why the viscosity PCG convergence criterion never trips

### Harness

`tools/diagnose_viscosity_pcg_convergence.py` (new, measurement-only).
It monkeypatches `ViscositySolver._solve_component` in-process with a replica of
the shipped loop that reads the residual back after **every** iteration instead
of only at `it % 10`. Production code is untouched. Because the shipped path
takes exactly 40 iterations in every solve (iteration 2, 114/114), suppressing
the early exit while recording changes nothing: the recorded 40-iteration runs
are bit-for-bit the shipped trajectory, with extra observation.

| Run | Purpose | Raw JSON |
| --- | --- | --- |
| D1 | first 6 component solves, budget raised to 400 | `artifacts/visc_pcg_residual/visc_pcg_residual_maxit400_20260902T170430Z.json` |
| D2 | 9 steady-state component solves, budget raised to 400 | `artifacts/visc_pcg_residual/visc_pcg_residual_steady_maxit400_20260902T170549Z.json` |
| **D3 (headline)** | **all 120 component solves, unperturbed shipped budget 40** | `artifacts/visc_pcg_residual/visc_pcg_residual_shipped40_all_20260902T170725Z.json` |

D3 command:

```
python tools/diagnose_viscosity_pcg_convergence.py --device cuda:0 --frames 8 \
    --warmup 2 --emit-frames 6 --scene-preset dam_break --max-iterations 0 \
    --skip-solves 0 --record-solves 1000 --tag shipped40_all
```

D3 ran at `activeVoxelsMean = 118080` and 4.625 substeps/frame, against the
iteration-2 benchmark's 120832 voxels and 4.75 substeps/frame — the same regime.

### The criterion, read exactly

`viscosity.py:406-409`:

```python
if it % 10 == 0 or it == iterations:
    wp.utils.array_inner(self.r, self.r, out=self.residual, count=n)
    if float(self.residual.numpy()[0]) < tolerance * tolerance * max(1, n):
        break
```

`r` is the **raw** residual `b - Ax` (not preconditioned, not scaled), `n` is
the active voxel count, so the test is an RMS-per-voxel criterion on a quantity
with units of velocity: `RMS(r) < tolerance`. At the shipped
`DEFAULT_VISCOSITY_TOLERANCE = 1e-4` and `n ≈ 118080` the threshold on `r·r` is
`≈ 1.18e-3`.

`pressure.py:2044/2090` uses the **identical** form,
`tol_sq = tolerance * tolerance * max(1, full_n)`, on its own raw
`b - Ax` residual. The criterion is therefore not mis-scaled relative to the
solver that demonstrably converges.

### Verdict: (iii) — the residual is still decreasing at iteration 40

**Verified**, from D3 (120 component solves, unperturbed) and D2/D1 (extended
budget).

- **Not (i), mis-scaled.** The threshold is reachable. 69 of 120 recorded
  component solves (57.5 %) drive `r·r` below it within the 40-iteration budget.
- **Not (ii), a stall.** The residual falls essentially monotonically and keeps
  falling: in D2 every recorded solve reaches `r·r ≈ 2e-29` — machine zero — by
  iteration 400. Measured decay is clean and geometric, ~5.5 iterations per
  order of magnitude of `r·r`: relative `r·r` reaches 1e-2 at iteration ~10,
  1e-4 at ~21, 1e-6 at ~32, 1e-8 at ~44, 1e-12 at ~67. Jacobi is not failing on
  this system.
- **(iii) confirmed.** Convergence lands at iterations **34-37** in steady
  state, i.e. *just under* the 40-iteration budget, and for one component it
  lands past it. Distribution of the first iteration at which `r·r` is below
  threshold, over the 75 steady-state (`dt = 1/120 s`) component solves in D3:

  | first iteration below threshold | 34 | 35 | 36 | 37 | never within 40 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | solves | 14 | 25 | 4 | 7 | **25** |

  All 25 non-converged steady solves are **component 2 (w)** — every single w
  solve in steady state. Over all 120 recorded solves, component 0 converges in
  30/40, component 1 in 38/40, component 2 in **1/40** (and that one is the
  trivial first frame with an empty domain).

**The shipped result is under-converged, and silently so.** This is an accuracy
finding, not a performance one. The w-component viscosity solve exits at
`max_iterations` with a residual still above the tolerance the settings ask for,
and nothing in the solver reports it: unlike `PressureSolver`, `ViscositySolver`
has **no status field, no convergence flag, no stall/breakdown/NaN guard** — it
only records `last_iterations`. The magnitude is mild: median
`r(40)·r(40) / threshold = 1.206` over the 25 non-converged steady solves
(max 1.909), i.e. the residual **norm** overshoots tolerance by a median 9.8 %
(max 38 %). Mild, but real, undisclosed, and systematically biased to the
gravity-aligned component.

### Why it "never trips" even in the 57.5 % of solves that do converge

Because convergence lands at 34-37 while the check cadence is 10/20/30/40. The
check at 30 always fails; the only checkpoint that can succeed is the terminal
`it == iterations` check at 40 — which saves no work, since the loop was about
to exit anyway. That is the complete explanation of iteration 2's
"`component_iterations = [1520, 1520, 1520]`, zero early exits": not a broken
criterion, a criterion whose reachable checkpoints straddle the answer.

### Why pressure's criterion trips and viscosity's does not

| | pressure | viscosity |
| --- | --- | --- |
| criterion | `r·r < tol² · max(1, n)` on raw `b - Ax` | **identical** |
| default tolerance | 1e-5 (**tighter**) | 1e-4 |
| default budget | 200 (`DEFAULT_PRESSURE_ITERATIONS`) | 40 |
| measured mean iterations | 68.75 of 200 (baseline probe, R2 raw JSON) | 40.0 of 40 |
| initial guess | `x0 = 0` (or a coarse-grid guess) | `x0 = b` — a warm start |
| check cadence | every 25 | every 10 |
| where the check runs | **on device**, via `k_prepare_pcg_conditional_loop` / `k_finish_pcg_conditional_batch` inside a CUDA conditional graph — no host stall | **host readback**, `self.residual.numpy()[0]`, a synchronous stall |
| extra exits | stagnation (`res >= last_res·(1-1e-6)`), breakdown, non-finite | **none** |

The decisive differences are budget headroom (200 vs 40, against ~69 vs ~36
iterations actually needed) and the fact that pressure's check is free while
viscosity's costs a host stall. Two corrections to the task brief, both
recomputed from the raw JSON: pressure's budget is **200**, not 350; and a
viscosity convergence check costs **0.13756 ms**, not ~0.65 ms
(`62.7286 ms / 456 readbacks`, `visc_st_raw_20260902T164256Z.json`).

## Phase 3b — cadence tightening: measured, and rejected

Unit costs, recomputed from `visc_st_raw_20260902T164256Z.json`
(`viscosity_on` probe, 8 frames, 114 component solves, 4560 iterations):

- 16.0223 ms per component solve, of which 4 readbacks × 0.13756 ms = 0.55024 ms;
- **0.38680 ms per PCG iteration** net of its share of checks;
- 14.25 component solves/frame; viscosity_on frame = **251.427 ms** (recomputed
  as median over 7 trials of the per-trial median `stepTime`).

Replaying every one of the 120 recorded D3 residual trajectories against each
candidate cadence, and pricing the result with those measured unit costs:

| cadence `it % k` | mean iterations | mean checks | Δ ms/frame | Δ % of the 251.43 ms viscosity-on frame |
| ---: | ---: | ---: | ---: | ---: |
| 10 (shipped) | 39.25 | 3.92 | 0.00 | — |
| 8 | 39.20 | 4.90 | +1.64 | +0.65 % |
| **6 (best)** | 37.72 | 6.49 | **-3.42** | **-1.36 %** |
| 5 | 37.50 | 7.50 | -2.64 | -1.05 % |
| 4 | 37.67 | 9.42 | +2.04 | +0.81 % |
| 2 | 37.20 | 18.60 | +17.47 | +6.95 % |
| 1 | 36.85 | 36.85 | +51.31 | +20.41 % |

The break-even is exact and unfavourable: one extra check costs 0.13756 ms and
one saved iteration returns 0.38680 ms, so a check must save 0.356 iterations in
expectation just to pay for itself — but convergence is packed into a 4-iteration
window (34-37) immediately below the budget, so there is almost nothing to save.

**Decision: not implemented.** The best cadence is worth -1.36 % of the
viscosity-on frame. Iteration 2's R2 `baseline` per-trial medians spread
31.06-34.40 ms on a 32.26 ms median (±5 %), so a 1.36 % effect cannot be
demonstrated by the paired benchmark this loop requires. Shipping a change whose
win is unprovable would violate the evidence standard. Plan item 3b is closed as
**refuted**, in the same way item 3d was.

## Phase 3e — on-device convergence check: IMPLEMENTED and MEASURED (iteration 4)

This is the first iteration in which production solver code was changed for
performance. Three things were delivered, in the order the brief required.

### 1. The check moved on-device, numerics-neutral

`ViscositySolver` previously evaluated its stopping criterion on the host:
`float(self.residual.numpy()[0])` inside the Python `for it in range(...)`
loop, once every 10 iterations. The whole loop is now capturable: the criterion
is evaluated by `k_viscosity_finish_pcg_conditional_batch` inside a CUDA
conditional graph (`wp.capture_while`), exactly as `PressureSolver` already
does, and the graph is cached per
`(active voxels, dt, voxel size, cadence, budget, tolerance^2, array pointers)`.
A component solve now costs **one** host readback (of a packed 5-float
`pcg_stats` array) instead of one per check.

Supporting changes the port required:

- `k_viscosity_pcg_beta_and_roll` computes `beta = rz_new/rz`, rolls `rz_new`
  into `rz` and clears `rz_new` in one launch. Pressure's captured loop needs an
  even/odd ping-pong over two `rz` slots; rolling on-device removes that, so the
  viscosity graph works for any budget divisible by the cadence.
- The reduction is `cuda_reduction.CudaDotProduct` (`atomicAdd` into a zeroed
  target), not `wp.utils.array_inner`, because `array_inner` overwrites *and
  allocates* and is therefore not capturable inside a conditional child graph.
- `ViscositySolver.release()` now exists and is called from `Solver.release()`,
  so captured graphs are freed with the solver.
- `_solve_pcg_host()` is kept as the reference path and the fallback when CUDA,
  conditional graphs, or a divisible budget are unavailable. It packs its three
  scalars with `k_pack_pcg_stats` so even the fallback reads back once per check
  instead of once per scalar.

Numerics-neutrality is pinned by
`tests/test_viscosity_conditional_pcg_cuda.py::test_device_check_matches_host_check_at_the_shipped_cadence`:
at cadence 10 the device and host loops stop on the same iteration with the same
verdict and agree on u/v/w to `rtol=2e-4`. The comparison is not bit-exact
because an accumulating `atomicAdd` reduction orders differently from warp's
tree reduction.

### 2. Per-iteration testing, and what it changes

`DEFAULT_VISCOSITY_CHECK_EVERY` is now **1**. This *does* change output, and the
direction is: **u and v used to be over-solved and now are not.** Measured in
production over 117 component solves
(`artifacts/visc_pcg_residual/visc_pcg_residual_status_after_20260902T180030Z.json`):

| component | before (cadence 10) | after (cadence 1) |
| --- | ---: | ---: |
| u | 40.00 iterations, always | 36.54 mean, 34/39 converged |
| v | 40.00 iterations, always | 36.59 mean, 36/39 converged |
| w | 40.00 iterations, always | 38.87 mean, 8/39 converged |

Every solve that stops early honours the tolerance it claims: across all 78
converged solves the **maximum** residual/threshold ratio is **0.997547**, so no
converged verdict is ever issued above threshold. u and v are measurably *less*
converged than the shipped build — that is the intended, stated consequence of
honouring the tolerance instead of always burning the full budget.

The **w component's shortfall is unchanged and unfixed**: 31 of 39 w solves
still exhaust the budget above tolerance, median overshoot **2.19x**, max
**12.98x**. That is plan item 3f (budget/tolerance), a user decision that was
deliberately not touched — neither the tolerance default nor the iteration
budget was changed. What *has* changed is that the shortfall is no longer
silent.

### 3. Status reporting and guards

`ViscositySolver` previously had no status field at all. It now reports through
the same shape `PressureSolver` uses — `last_status` / `last_iterations` /
`last_residual` / `last_viscosity_status` / `last_viscosity_failure`, plus
`last_component_statuses` for the per-axis detail — with the same exit reasons
carried across: `converged`, `max_iterations`, `stalled`, `singular`,
`invalid` (non-finite), plus viscosity-specific `skipped` / `jacobi` /
`not_run`. `solve()` reports the **worst** component by a severity-ordered
status code, so one bad axis cannot be hidden by two good ones.

One deliberate deviation from pressure, recorded because it is a judgement call:
pressure's stagnation test compares consecutive checks. At cadence 1 that test
misfires — replayed against the recorded trajectories it fires on **26 of 120**
solves, earliest at iteration 3, with the residual still ~3e5x above tolerance
and still falling geometrically. The viscosity guard therefore compares against
a **window of 10 checks** (`VISCOSITY_STAGNATION_WINDOW`), which fires on **0 of
3366** recorded checks (worst window ratio 0.145). In the 117-solve production
run no `stalled`, `singular` or `invalid` status ever fired: the guards do not
misfire.

### Paired benchmark — measured, not bounded

Identical command both sides, run sequentially on the one GPU. The `before` side
is obtained from the same binary via `--viscosity-convergence-check legacy-host`,
which restores cadence 10 and the host readback:

```
python tools/profile_viscosity_surface_tension.py --device cuda:0 --frames 8 \
  --warmup 2 --trials 7 --emit-frames 6 --scene-preset dam_break \
  --out-dir artifacts/visc_st_measurement
```

- before: `artifacts/visc_st_measurement/visc_st_raw_20260902T175446Z.json`
- after: `artifacts/visc_st_measurement/visc_st_raw_20260902T175731Z.json`

The emulation is verified faithful, not assumed: the `before` probe counters
report exactly 4.00 readbacks per component solve and 40.0/40.0/40.0 iterations,
identical to the clean pre-change runs R2 and R3.

Statistic: per-trial median of per-frame `stepTime`, then the median of the
paired per-trial ratios.

| config | before ms/frame | after ms/frame | ratio | delta |
| --- | ---: | ---: | ---: | ---: |
| baseline (control) | 45.91 | 40.63 | 0.9794 | -2.06 % |
| **viscosity_on** | **378.75** | **62.80** | **0.1656** | **-83.44 %** |
| surface_tension_on (control) | 93.06 | 95.48 | 0.9789 | -2.11 % |
| **both_on** | **415.71** | **100.15** | **0.2402** | **-75.98 %** |
| baseline_no_whitewater (control) | 36.94 | 34.98 | 0.9707 | -2.93 % |
| surface_tension_on_no_whitewater (control) | 64.71 | 66.95 | 0.9544 | -4.56 % |

The four controls move by -2.1 % to -4.6 %, i.e. inside the +/-5 % trial spread
this benchmark was already shown to have; nothing is claimed from them. The two
viscosity configs move far outside it, and their per-trial ranges do not overlap
at all (viscosity_on before 362.08-433.42 ms, after 53.78-65.73 ms). Viscosity's
own cost, contention-matched as `viscosity_on - baseline`, falls from
**332.84 to 22.17 ms/frame**; measured against the clean iteration-2 R2
reference (219.17 ms) it still falls to 22.17. The viscosity multiplier goes
from **7.8-8.6x to 1.55x**.

### Revised hypothesis: the bound was built on the wrong cost

The plan predicted **-21.07 ms/frame (-8.38 %)** from measured unit costs. The
measured result is roughly **6x larger** than that bound, so the model behind the
bound was wrong and is corrected here rather than quietly celebrated.

From the probe counters in the same two files (recomputed):

| | before | after |
| --- | ---: | ---: |
| readbacks per component solve | 4.00 | 1.00 |
| iterations per component (u/v/w) | 40.0 / 40.0 / 40.0 | 36.51 / 34.74 / 40.00 |
| ms per component solve | 23.788 | 1.598 |

Iterations fell by ~10 %, readbacks by 75 %, but per-solve time fell by **93 %**.
Neither of the two costs the bound was built from can explain that. What actually
dominated was **per-iteration host launch and allocation overhead**: ~9
Python-side `wp.launch` calls plus 2 `wp.utils.array_inner` reductions (each of
which *allocates*) per iteration — roughly 400 launches and 120 allocations per
component solve — all of which the captured conditional graph replaces with one
`wp.capture_launch`. The shipped viscosity PCG was **launch-overhead bound, not
readback bound.**

This retro-actively explains hypothesis 3's verdict (readback cost refuted at
3.5 % of the delta — correct, and it stays refuted) and hypothesis 13's (cadence
tightening priced at -1.36 % — also correct *given the shipped loop structure*).
Both were right about their own quantity and both missed the dominant one,
because the cadence model priced iterations and checks and never priced the
launch. The measured 0.38680 ms "per PCG iteration" was mostly host-side, not
device work.

## Hypothesis verdicts

| # | Hypothesis | Verdict |
| --- | --- | --- |
| 1 | `viscosity.py:419-426` runs 3 fully independent PCG solves (u/v/w) | **CONFIRMED.** 1 `solve()`/substep times exactly 3 `_solve_component()` calls, 228.3 ms/frame combined, u/v/w within 0.7 % of each other. |
| 2 | Up to 40 PCG iterations per component, ~7 launches each | **CONFIRMED and worse than stated.** Always 40 of 40, never fewer, at the shipped tolerance. ~9 launches + 2 reductions per iteration. |
| 3 | 12 synchronous `.numpy()` host stalls per substep dominate viscosity cost | **Count CONFIRMED (exactly 12.0/substep, 456 calls). Cost REFUTED: 7.84 ms/frame = 3.5 % of the viscosity delta.** Not worth optimising on its own — and iteration 4 confirmed this from the other side: removing them (plus per-iteration launch overhead) won 6x more than the readback model predicted, so the readbacks were never the dominant cost. |
| 4 | `solver_sdf_build_key()` includes `sim_time`, so the SDF cache never hits with ST on | **CONFIRMED** (4.1 % hit rate, 4 of 98 calls). |
| 5 | `build_sdf()` reruns fully every substep and its particle-splat kernel is the single largest ST cost | **CONFIRMED.** The rebuild is 87-95 % of the whole ST cost. |
| 6 | `solver.py:2174-2175` forces a *second* rebuild per substep when whitewater is also on | **CONFIRMED.** 2.09 `build_sdf` calls/substep with ST + whitewater vs 1.09 with ST alone. |
| 7 | "forces-net-of-ST = `compute_3 - compute_11`" (from the task brief) | **REFUTED.** `compute_11` has three call sites, only one of which nests inside forces. Use paired deltas. |
| 8 | The capillary substep limiter does not bind at ordinary settings | **CONFIRMED, unchanged.** Every measured frame reports `substepLimiter=CFL` and `capillarySubstepsRequired=1`. |
| 9 | ST's enlarged splat radius adds materially to `build_sdf` cost (iteration 1 ranked this "lowest priority, likely small") | **CONFIRMED and promoted to the top of the ROI list.** Splat radius 2 -> 7 = 27x per-particle write volume = a measured 12-15x per-call cost increase. Iteration 1's ranking of this item was wrong. **Iteration 5 correction:** the 27x write-volume model is right about where the cost is, but it implied radius-2 cost was recoverable, and it is not. The radius is not free to shrink: surface tension reads the field across a support band the particle does not reach, so the smallest radius that keeps the field exact where the force *and its stencil* are read is 4, not 2. Two changes against that floor won -35.4 % (bounding the splat to the ball that can win) and a further -34.1 % (propagating closest points out to the band); the residual gap to radius 2 is the band itself and is not an artefact. |
| 10 | The viscosity convergence criterion is mis-scaled and therefore unreachable | **REFUTED** (iteration 3). Identical in form to pressure's, and reached within budget by 69 of 120 recorded component solves. |
| 11 | The residual stalls — Jacobi is inadequate for this system | **REFUTED** (iteration 3). Monotone geometric decay, ~5.5 iterations per order of magnitude of `r·r`, down to machine zero (~2e-29) by iteration 400. |
| 12 | The residual is still decreasing at iteration 40, i.e. the shipped result is under-converged (**partly addressed** in iteration 4: still true for w — 31/39 solves, median overshoot 2.19x — but it is now *reported* rather than silent) | **CONFIRMED** (iteration 3). Convergence lands at iterations 34-37; **every steady-state w-component solve (25 of 25) exhausts the budget still above tolerance**, median residual-norm overshoot 9.8 %, max 38 %. Silent: `ViscositySolver` reports no status. |
| 13 | Tightening the `it % 10` cadence recovers meaningful work (plan item 3b) | **REFUTED** (iteration 3), and the refutation stands: *within the shipped host-checked loop* a tighter cadence costs more in checks than it saves in iterations. Iteration 4 changed the loop structure instead, after which cadence 1 is free. |
| 14 | The viscosity PCG is readback-bound (the model behind plan item 3e's -21.07 ms bound) | **REFUTED by the implementation itself** (iteration 4). Iterations fell 10 %, readbacks 75 %, but per-component-solve time fell 93 % (23.788 -> 1.598 ms). The loop was **launch-overhead bound**: ~400 Python-side `wp.launch` calls and ~120 allocating `array_inner` reductions per component solve, all replaced by one `wp.capture_launch`. |
| — | A quality-tier multiplier exists for `pressure_iterations` that `viscosity_iterations` lacks | **OPEN and BLOCKED.** `sim/quality_contract.py` is not on this branch. Not investigated; pulling from `fix/settings-apply-domain-object` is not authorized. |

## ROI-ordered implementation plan (Phase 3 — not started)

Ordering is derived from the measured shares above, not from code reading.
Nothing here has been implemented, and no speedup is claimed for any item.

**Re-ranking after iteration 3.** The item numbering below is kept stable for
continuity, but the *execution order* changed. Two of viscosity's three
sub-items (3b, 3d) are now refuted, and a new one (3e, on-device convergence
check) replaced them with a larger and better-founded bound. Recommended order:

| rank | item | bound | basis |
| ---: | --- | ---: | --- |
| — | **3e** — on-device viscosity convergence check | **DONE (iteration 4). Measured -83.44 % of the viscosity-on frame, -75.98 % of both_on.** The -21.07 ms bound was a 6x underestimate; see Phase 3e. |
| — | **1** — shrink the ST SDF splat cost | **DONE (iteration 5), in two parts. Measured -35.4 % of `build_sdf` from bounding the splat, then -34.1 % more from band propagation.** Composed (the product of the two measured ratios, since each stage has its own baseline), that is **-57.40 %** of `build_sdf` in `surface_tension_on`. The "~90 % of the rebuild cost" target was not reachable; see below. | paired, order-reversed benchmarks; `artifacts/sdf_splat_bounds/`, `artifacts/sdf_band_propagate/` |
| — | **2a** — cut the ST rebuild count with whitewater on | **CLOSED (iteration 5): genuinely unmergeable.** `_g2p_advect` and `_relax` run between the two builds, so whitewater would be fed pre-advection geometry. A *reverse* candidate fell out of the analysis — reuse substep N's post-advection SDF for substep N+1's forces — which is larger, and is recorded unimplemented because it would require rekeying a fail-closed invalidation. | code analysis; no change made |
| 1 | **3g** — the tolerance-invariant half of the viscosity solve | **NEW (iteration 5).** ~12.1 ms/frame, 50.5 % of the viscosity solve, flat across a full decade of tolerance | measured; see item 3f |
| 2 | **3c** — fuse the three per-axis viscosity solves | unquantified; needs numerics review | structural |
| — | **3f** — viscosity budget/tolerance knee | **RE-MEASURED (iteration 5): -7.0 % of the viscosity solve at tol 1e-3, not the projected -33 %, and invisible in the wall lane.** Now an *accuracy* decision — w hits its 40-iteration cap on every solve at the shipped tolerance | changes simulation output; user decision, not an optimization |
| — | **3b, 3d, 2b** | closed | refuted or not-a-bug |

1. **Shrink the surface-tension SDF splat radius.** Largest measured win per
   unit of risk, and the only cost driver here that is a pure implementation
   artefact rather than a physics requirement. `build_sdf` currently splats
   every particle into a 15^3-voxel cube so the SDF has valid values across the
   normal/curvature support band, even though the particle itself is 0.6 voxels
   across. The band values are a *distance field*, not particle data: they can
   be propagated outward from the narrow band by a fast sweep / jump-flood / a
   few dilate passes over `nvox` (active voxels ~121-131k) instead of by 27x
   redundant per-particle atomic writes over ~244k particles. Target: return
   the per-call cost toward the measured 0.35-0.41 ms of the splat-radius-2
   path, which would address up to ~90 % of the 30-58 ms/frame rebuild cost.

   **Iteration 5: implemented, and that target was wrong — correcting it rather
   than restating it.** The radius-2 cost is not a floor this item can reach.
   Radius 2 is what the *particle* needs; surface tension additionally reads
   the field across the force band and the one further cell its normal/curvature
   stencil touches, and for `dam_break` the smallest radius that keeps those
   values exact is 4. The band is a physics requirement, not the implementation
   artefact the paragraph above assumed; only the *redundancy* around it was.
   Both available reductions were taken and measured paired and order-reversed:
   bounding the splat to the ball that can win (-35.4 % of `build_sdf`, and
   output-identical), then seeding radius 4 and propagating closest points out
   to radius 7 (-34.1 % more, exact inside the seeded window and a strict
   upper bound outside it). Composed — the product of the two measured
   ratios, because each stage was measured against its own baseline in its
   own session — that is **-57.40 %** of `build_sdf` in
   `surface_tension_on`, against the ~90 % the target claimed. The accuracy check the paragraph below demands was done and
   is a derived bound, not an assumption: `tests/test_surface_sdf_band_propagate_cuda.py`.
   **Must be paired with an accuracy check** — the band must remain valid out
   to `support_voxels`, which `required_sdf_splat_radius_voxels` currently
   guarantees by construction.
2. **Cut the surface-tension rebuild count.** Two sub-items, both smaller than
   item 1 but cheap:
   - 2a. With ST + whitewater on, `build_sdf` runs 2.09x/substep. Determine
     whether `after_substep()` can reuse the SDF `_forces()` built this substep.
     The existing comment at `solver.py:2170-2174` argues whitewater needs the
     *post-advection* field, so this may be genuinely unmergeable — evaluate,
     do not assume. Upper bound if mergeable: ~28 ms/frame at these settings.
   - 2b. The 4.1 % cache hit rate is a direct consequence of `sim_time` in
     `solver_sdf_build_key()`. This is *correct* — positions really do change
     every substep — so this is not a bug to fix; it is confirmation that
     per-substep rebuild cadence, not caching, is the lever.
3. **Viscosity PCG.** Largest single line item overall (228 ms/frame, 7.8x
   slowdown). Iteration 3 measured the convergence behaviour in full and closed
   two sub-items:
   - 3a. **DONE.** See "Phase 3a" above. Verdict (iii): the residual is still
     decreasing at iteration 40, convergence lands at 34-37, and every
     steady-state w solve exceeds the budget. The criterion is sound; the budget
     is one notch too small and the cadence straddles the answer.
   - 3b. **REFUTED and closed.** See "Phase 3b" above. -1.36 % at the best
     cadence, below the noise floor. Not implemented.
   - 3c. Combine the three per-axis solves into one interleaved PCG. This is the
     structurally correct fix (the three solves share topology, stencil and
     preconditioner shape) but needs a numerics review and validation against
     the existing viscosity fixtures. Iteration 3 adds a caveat: the three
     components do **not** converge together (u/v at 34-35, w past 40), so a
     fused solve must either keep three independent residual tests or accept
     running all three to the slowest component's iteration count. Fusing on a
     single shared residual would make u and v strictly *more* over-solved than
     today.
   - 3d. **Do not** prioritise batching the residual readbacks. Measured at
     3.5 % of the viscosity delta; the hypothesis is refuted.
   - 3e. **DONE (iteration 4) — see Phase 3e above for what was implemented and
     measured. The bound quoted below was correct arithmetic on the wrong cost
     model and understated the result by ~6x; it is kept verbatim for the
     record.** Original text: **move the viscosity convergence check on-device.**
     `PressureSolver` already does exactly this: `k_prepare_pcg_conditional_loop`
     and `k_finish_pcg_conditional_batch` evaluate the criterion inside a CUDA
     conditional graph, so the check costs no host stall
     (`pressure.py:774-847`, `solve_pcg`). Porting that machinery to
     `ViscositySolver` makes the check free, which simultaneously (a) removes
     the 7.84 ms/frame of readback stalls that item 3d correctly said were not
     worth batching but which *are* worth eliminating when they become free, and
     (b) permits a per-iteration test, which the D3 trajectories say cuts mean
     iterations from 39.25 to 36.85. Upper bound from measured unit costs:
     **-13.23 ms/frame (iterations) -7.84 ms/frame (readbacks) = -21.07 ms/frame,
     -8.38 % of the viscosity-on frame.** That is a *bound derived from measured
     per-unit costs, not a measured speedup* — it must be proven by a paired
     benchmark before any claim is made. Risk: moderate; it is a port of
     existing, tested machinery, and the numerics are unchanged. It should carry
     the pressure solver's stagnation/breakdown/non-finite exits with it, which
     `ViscositySolver` currently lacks entirely.
   - 3f. **NEW — the budget/tolerance knee is a user decision, not an
     optimization. RE-MEASURED in iteration 5; the original projection below is
     superseded.** The first table was a *replay* of recorded D3 trajectories
     against alternative settings, taken before the conditional-graph fix. That
     fix collapsed viscosity's absolute cost, and with it the value of trading
     accuracy for iterations. Measured directly, on `dam_break`, cuda:0, six
     sequential order-reversed runs (1e-4, 3e-4, 1e-3, 1e-3, 3e-4, 1e-4),
     7 trials x 8 frames, two runs pooled per arm, no default changed:

     | tolerance | viscosity solve ms/frame | residual-stall ms | remainder ms | u iters | v iters | w iters |
     | --- | ---: | ---: | ---: | ---: | ---: | ---: |
     | 1e-4 (**shipped**) | 23.940 | 11.845 | 12.095 | 36.38 | 35.13 | **40.00** |
     | 3e-4 | 22.421 | 10.391 | 12.030 | 31.10 | 29.69 | 35.68 |
     | 1e-3 | 22.253 | 9.090 | 13.163 | 25.71 | 23.57 | 29.37 |

     **The speed case for 3f is dead.** Loosening the tolerance a full decade
     buys **-7.04 %** of the viscosity solve (23.940 -> 22.253 ms/frame), not
     the projected -32.9 %. And even that does not reach the wall clock: over
     the same runs the *controls* drifted +8.84 %/+9.91 % (`baseline`) and
     +7.48 %/+7.10 % (`baseline_no_whitewater`), while the spread between the
     two runs of the shipped arm alone was 54.574 vs 60.953 ms/frame (11.7 %).
     Every between-arm wall difference is smaller than the within-arm spread,
     so **the wall lane carries no usable signal for this change** and no wall
     figure should be quoted for it.

     Why the saving is so small: iterations do fall ~27 %, and that shows up
     where expected — the residual host-sync stall, which is where the
     on-device PCG loop's GPU time is actually waited on, drops -12.27 % /
     -23.26 %. But that stall is only 49.5 % of the viscosity solve. The
     remainder (launch and assembly, plus whatever GPU work trails the residual
     read) is ~12.1 ms/frame and is **flat across tolerances** — it does not
     care how many iterations ran. Cutting the iteration count therefore cannot
     reach more than half the cost, and reaches well under that in practice.
     *That flat ~12.1 ms/frame is now the larger of the two viscosity terms and
     is a new optimization target; see ROI item 5 below.*

     **What 3f is still worth deciding — accuracy, not speed.** At the shipped
     tolerance the w component's mean stop iteration is **exactly 40.00**, the
     budget: it hits the cap on every solve in the sample and never converges.
     At 3e-4 the mean falls to 35.68 and at 1e-3 to 29.37, both below the cap.
     So today's default silently under-converges w, and the cost of fixing that
     is now under 7 % of the viscosity solve rather than the ~22 % the old table
     implied. That materially strengthens the case for 3e-4 — but it changes
     simulation output, so it remains a user decision.
     **Whatever is chosen, `ViscositySolver` should report convergence status
     the way `PressureSolver` does, so under-convergence stops being silent.**

     <details><summary>Superseded pre-conditional-graph projection (replay of
     recorded D3 trajectories, retained for the record)</summary>

     | setting | converged | median stop iteration | Δ ms/frame vs today |
     | --- | ---: | ---: | ---: |
     | tolerance 1e-3 | 75/75 | 25 | -82.68 (-32.9 %) |
     | tolerance 3e-4 | 75/75 | 30 | -55.12 (-21.9 %) |
     | tolerance 1e-4 (shipped) | 50/75 | 35 | 0 |
     | tolerance 3e-5 | 10/75 | 40 | — |
     | budget 40 -> 48, tolerance unchanged | 75/75 (expected) | 41 | +14.70 (+5.9 %) |

     These percentages were never wrong for the solver they described; they are
     wrong for the solver that now ships. Raw evidence for the replacement:
     `artifacts/visc_tolerance_knee/run{1..6}_tol*`.
     </details>
4. **Blocked work — quality-tier scaling of `viscosity_iterations`.** Requires
   `sim/quality_contract.py`, which is only on the unmerged
   `fix/settings-apply-domain-object` branch. Needs a user decision on merging
   that branch before any work here.

## Evidence and honesty accounting (iteration 2)

- **Source changed:** `fluidSim_warp_blender51/diagnostics/profiler.py`
  (nesting-aware `track()`, depth reset in `begin_frame()`);
  `fluidSim_warp_blender51/sim/surface_tension.py` (iteration-1 edit verified;
  its stale "not stack-aware / will double-count" comment corrected).
- **Tests changed:** `tests/test_profiler_phase_contract.py` (+5 tests).
- **Tools changed:** `tools/profile_gpu_solver.py` (dam_break preset,
  viscosity/ST/whitewater flags, per-phase and splat-radius summary fields,
  `solver.release()`, `on_frame` callback);
  `tools/profile_viscosity_surface_tension.py` (new paired driver with
  measurement-only probes and GPU-state sampling).
- **No production solver code was changed for performance.** No optimization
  has been implemented and no speedup is claimed.
- **Suite:** `python -m pytest -q` gives **1386 passed** in 137 s.
  `tools/run_production_gate.py` has **not** been run this iteration.
- Every number in this document was recomputed from the raw JSON files listed
  above. Nothing is estimated or extrapolated.

## Evidence and honesty accounting (iteration 3)

- **Source changed:** none. No production file was modified.
- **Tools added:** `tools/diagnose_viscosity_pcg_convergence.py`
  (measurement-only residual-trajectory probe; monkeypatches
  `ViscositySolver._solve_component` in-process, does not touch `sim/`).
- **Raw evidence added:** the three `artifacts/visc_pcg_residual/*.json` files
  listed in Phase 3a.
- **Nothing was implemented and no speedup is claimed.** Plan item 3b was
  investigated, priced against measured unit costs, and rejected because its
  best-case win (-1.36 %) is smaller than the benchmark's demonstrated trial
  spread (±5 %). A refuted item is a result, not a gap.
- **Accuracy finding, stated plainly:** at the shipped defaults the w-component
  viscosity solve does not reach the tolerance the settings ask for, in every
  steady-state substep measured, and the solver does not report it. Median
  residual-norm overshoot 9.8 %, max 38 %. This is a correctness gap, not a
  performance framing.
- **Two corrections to the iteration-3 task brief**, both recomputed from
  `visc_st_raw_20260902T164256Z.json`: a viscosity convergence check costs
  **0.13756 ms** (62.7286 ms / 456 readbacks), not ~0.65 ms; and the pressure
  budget is **200** (`DEFAULT_PRESSURE_ITERATIONS`), not 350 — its measured mean
  is 68.75 of 200.
- **Suite:** see the iteration-3 report; no production code changed, so no
  behavioural regression is possible from this iteration's edits.

## Evidence and honesty accounting (iteration 4)

- **Production source changed (first time in this loop):**
  `fluidSim_warp_blender51/sim/viscosity.py` (conditional-graph PCG loop, three
  new kernels, status/guard machinery, `release()`);
  `fluidSim_warp_blender51/sim/solver.py` (one edit: `Solver.release()` now
  releases the viscosity solver).
- **Tests changed:** `tests/test_viscosity.py` (2 -> 12 tests);
  `tests/test_viscosity_conditional_pcg_cuda.py` (**new**, 6 CUDA tests);
  `tests/test_viscosity_active_work_regressions.py` (1 test extended to also
  count `CudaDotProduct.launch`, because the device path no longer routes
  through `wp.utils.array_inner` and the active-prefix invariant would
  otherwise have gone unchecked).
- **Tools changed:** `tools/profile_viscosity_surface_tension.py`
  (`--viscosity-convergence-check {current,legacy-host}`, so the before/after
  pair runs from one binary); `tools/diagnose_viscosity_pcg_convergence.py`
  (`--probe status`, `--check-every`; the status probe *wraps* the production
  `_solve_component` and records what the solver itself reports, so it requires
  no production-code change to work).
- **Suite:** `python -m pytest -q` gives **1402 passed** (baseline 1386 + 16 new).
- **Gate:** `python tools/run_production_gate.py` reaches `HEADLESS-GATE-PASS`.
  Note on the "pre-existing red": `validate_policy_drift_reports.py` **does not
  exist on this branch** (`find . -name validate_policy_drift_reports.py`
  returns nothing) and the gate has no policy-drift step, so the
  "no new failures versus baseline" criterion is satisfied trivially — the gate
  is fully green here, and that caveat applies to a different branch.
- **What was NOT done, and why:** surface tension is untouched. ROI item 1
  (shrink the SDF splat radius) is still the largest single remaining item and
  is the next action. Plan item 3f (tolerance / iteration budget) was
  deliberately not touched: it changes simulation output and is a user decision.
  Plan item 4 remains blocked on the unmerged `fix/settings-apply-domain-object`
  branch. **The overall task — "optimize the viscosity *and* surface-tension
  slowdowns" — is therefore NOT complete.**
- Every number in the Phase 3e section was recomputed from the raw JSON on disk
  named there. Nothing is estimated.

## Evidence and honesty accounting (iteration 5)

- **Scope:** surface tension (ROI items 1 and 2a) and a re-measurement of the
  viscosity tolerance knee (item 3f). Viscosity's solver was not touched.
- **What shipped:** two commits.
  - `c6ee7f7` *perf(surface): bound the particle-SDF splat to the ball that can
    win* — narrows the splat loop analytically to the cells a particle can
    actually claim. **Output-identical**, so the claim is exactness by
    construction plus a paired benchmark, not a tolerance argument.
  - `4ecc21a` *perf(surface): carry closest points out to the capillary band* —
    seeds only the radius the particle support justifies, records the winning
    particle per voxel, and hands those closest points outward one cell per
    pass. **Not** output-identical, so it ships **opt-in** behind
    `--sdf-band-mode` / `set_default_sdf_band_mode`, with a derived bound rather
    than an assumed one.
- **Measured, paired, order-reversed, one GPU, sequential** (`dam_break`,
  cuda:0, 7 trials x 8 frames; every figure recomputed from the raw per-frame
  records, never from the tool's aggregate):

  | config | stage 1: bound the splat | stage 2: propagate the band | composed |
  | --- | ---: | ---: | ---: |
  | `surface_tension_on` | 61.551 -> 39.779 (**-35.37 %**) | 42.138 -> 27.776 (**-34.08 %**) | -57.40 % |
  | `both_on` | 50.641 -> 32.878 (**-35.08 %**) | 31.594 -> 20.280 (**-35.81 %**) | -58.33 % |
  | `surface_tension_on_no_whitewater` | 32.400 -> 20.743 (**-35.98 %**) | 21.430 -> 14.078 (**-34.30 %**) | -57.94 % |

  The two stages are separate paired experiments
  (`artifacts/sdf_splat_bounds/`, `artifacts/sdf_band_propagate/`), each with
  its own baseline, run on different days. **Their absolute columns are not
  comparable across stages** — stage 2's 42.138 ms/frame baseline is the same
  code path as stage 1's 39.779 ms/frame result, 6 % apart on machine drift
  alone. The composed column is therefore the *product of the two measured
  ratios* (0.6463 x 0.6592 and so on), not an end-to-end measurement: label it
  **Likely**. Each per-stage figure is **Verified**, with non-overlapping run
  ranges between its arms. Arms were split by each run's own recorded `args`,
  not by directory naming.
- **Accuracy of the non-identical change, in the terms that matter:** over an
  11.5k-voxel shell around the interface, normal direction moves 3.0e-5 degrees
  and curvature 2e-6 relative. Inside the exact window the field matches the
  splat to 3.7e-7 voxels — one float32 ulp — and **no voxel anywhere
  underestimates**, at any seed radius tested. The guarantee is derived, not
  hoped for: a voxel is seeded by its *true* nearest particle whenever that
  particle lies within Chebyshev radius `seed` cells, and Chebyshev never
  exceeds Euclidean, so every voxel with `sdf <= seed*h - r` is exact; outside
  it every pass takes a min against another *real* particle distance, never an
  interpolation, so the field is a strict upper bound that only falls with more
  passes.
- **Corrected, not quietly restated:**
  - ROI item 1's target — "return the per-call cost toward the measured
    0.35-0.41 ms of the splat-radius-2 path, which would address up to ~90 % of
    the rebuild cost" — **was not reachable and was wrong in its premise.** The
    band is a physics requirement, not an implementation artefact; the smallest
    exact seed radius here is 4, not 2. Delivered -57.40 % of
    `build_sdf` in `surface_tension_on`, not ~90 % of it.
  - Hypothesis 9's 27x/12-15x model is right about *where* the cost is and
    wrong about what could be recovered; annotated in place.
  - Item 3f's -21.9 %/-32.9 % projections are **superseded**: re-measured
    against the post-conditional-graph solver they are -6.3 %/-7.0 % of the
    viscosity solve, and nothing at all in the wall lane.
- **Refuted, and recorded so it is not retried blind:** fusing the seed pass
  into the splat with a packed 64-bit `(distance, index)` `atomic_min`. It was
  implemented and measured, not reasoned about — on sm_89 the contended int64
  atomic costs **2.9-3.6x more than the 32-bit pass it removes** (seed 4, 200k
  particles: 1.71 ms two-pass vs 5.05 ms fused). Reverted; the refutation lives
  in `k_seed_surface_sdf_closest`'s docstring.
- **A measurement that produced no result, reported as such:** the wall lane in
  the tolerance sweep. The controls drifted +8.84 %/+9.91 % (`baseline`) and
  +7.48 %/+7.10 % (`baseline_no_whitewater`) across the six runs, and the two
  runs of the shipped arm alone differed by 11.7 % (54.574 vs 60.953 ms/frame).
  Every between-arm difference is smaller than the within-arm spread. **No wall
  figure is quoted for item 3f**, in either direction.
- **New item found by measurement, not by reading code:** ~12.1 ms/frame — 50.5 %
  of the viscosity solve — is invariant across a full decade of tolerance. It
  is now the larger of the two viscosity terms and the largest remaining
  viscosity lever (item 3g).
- **Tests changed:** `tests/test_surface_sdf_band_propagate_cuda.py` (**new**,
  23 tests). Its device tests assert the three claims above directly — never an
  underestimate, exact inside the seeded window, monotone in pass count — and
  the margins are non-vacuous: the curvature assertion passes with 500x room.
  Tolerances are 1e-5 voxels (~27 float32 ulps), loose enough that differently
  rounded hardware cannot flake them and still three orders below the ~1e-2
  voxel error the propagation legitimately carries outside the seeded window.
- **Tools changed:** `tools/profile_viscosity_surface_tension.py` gained
  `--sdf-band-mode` and `--sdf-band-seed-radius` (+16 lines), so the
  before/after pair runs from one binary with a mode flag.
- **Suite:** `python -m pytest -q` gives **1436 passed** in 167.53 s
  (1413 + 23 new).
- **Gate:** `python tools/run_production_gate.py` reaches `HEADLESS-GATE-PASS`
  (320 checks passing, 0 failing) plus `DEMO-BLEND-CHECK-PASS`.
- **What was NOT done, and why:**
  - The propagate path is **not the default**. Flipping it changes physics
    output; that is the user's call, not this loop's.
  - No viscosity default changed. `DEFAULT_VISCOSITY_TOLERANCE` and
    `DEFAULT_VISCOSITY_ITERATIONS` are untouched; item 3f is presented as a
    decision, not applied as an optimization.
  - Item 2a is closed as unmergeable; the larger reverse candidate it exposed
    (reuse substep N's post-advection SDF for substep N+1's forces) is recorded
    and **not** implemented, because it would require rekeying a fail-closed
    whitewater invalidation.
  - Plan item 4 remains blocked on the unmerged
    `fix/settings-apply-domain-object` branch. Merging it was not authorized and
    was not done.
- Every number in this section and in item 3f was recomputed from the raw JSON
  named alongside it. Nothing is estimated.
