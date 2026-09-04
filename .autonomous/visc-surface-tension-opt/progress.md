# Progress notes — visc-surface-tension-opt

## Session 0 (initializer) — 2026-09-03

**State at start:** branch `feat/ui-redesign-phases-0-3`, HEAD `cc90ffa`, no
commits made. Worktree dirty with ~24 unrelated UI/effects/demo entries that
must be preserved. Suite was RED (2 failed / 1400 passed).

**Done:**

- Task 1. Diagnosed the red suite: `fluidSim_warp_blender51/sim/surface.py`
  carried +359/-9 lines from an interrupted session — a real, substantive
  implementation of ROI item 1 (closest-point band propagation, plus a
  ball-cull mode with a sound bit-identity argument), but incomplete: it broke
  the kernel-launch-sequence assertions in `tests/test_surface_stack.py`.
  Preserved it verbatim to
  `reference/surface-py-band-propagation-WIP.patch` (442 lines), then restored
  the file to HEAD.
  **Verified:** `python -m pytest -q tests/test_surface_stack.py` -> 3 passed.

**Decision recorded:** the WIP patch was reverted rather than repaired in place
because a red suite is a bad baseline for autonomous work, and because the
patch touches the exact file task 3 must modify. It is preserved as a design
reference so none of the thinking is lost. Recover it any time with
`git apply .autonomous/visc-surface-tension-opt/reference/surface-py-band-propagation-WIP.patch`.

**Next:** task 2 — commit the verified viscosity work with strict per-file
staging.

**Open questions for the user (do not act on these autonomously):**
1. Whether `fix/settings-apply-domain-object` merges to master. Blocks
   quality-tier scaling of `viscosity_iterations` (`sim/quality_contract.py`
   lives only on that branch).
2. Plan item 3f — the viscosity tolerance/budget choice. Changes simulation
   output. Task 5 produces a corrected decision table but changes nothing.

## Session 1 — FAILED TO START (auth)

`bash .autonomous/visc-surface-tension-opt/run-session.sh 1` exited 1 after
73 bytes of output. Exact error, verbatim:

```
Failed to authenticate: OAuth session expired and could not be refreshed
```

The headless CLI (`/c/Users/Tyler/.local/bin/claude`) holds credentials
separately from the Claude Code desktop app, and its token has expired. The
executor never reached step 1; **no project file was read or written by it.**

Repo state is unchanged from the end of session 0: task 1 done, suite green,
HEAD `cc90ffa`, no commits, dirty worktree fully intact.

**Unblock:** the user must re-authenticate the CLI, e.g. run `claude` once
interactively and log in, or `claude setup-token`. Then re-run
`bash .autonomous/visc-surface-tension-opt/run-session.sh 1`.

## Session 2 — in this Claude Code session (user elected in-session execution)

The headless CLI could not authenticate, so the user chose to have the same
task list executed directly in this session. Same guardrails apply.

**Task 2 DONE — commit `bd0d81e`.**

Mixed-work inspection, as the task list required:

- `sim/surface_tension.py` — the whole diff is task work (+24/-1). Staged
  entire. One edit made before committing: the trailing comment asserted that
  `compute_3 - compute_11` yields "forces net of surface tension", which
  hypothesis 7 in the plan explicitly REFUTED (a smoke run reported
  `compute_3 = 17.5 ms` against `compute_11 = 35.9 ms`, impossible if
  compute_11 were a subset). Replaced with the paired-delta method that
  actually isolates ST.
- `sim/solver.py` — MIXED. Two hunks: one is the user's unrelated Custom
  Effects work (`update_custom_effect_runtime_bindings`, ~line 806), one is
  task work (`viscosity_solver.release()`, ~line 1129). `git add -p` is
  unavailable in this environment (interactive flags unsupported), so the
  hunks were split programmatically and the task hunk applied with
  `git apply --cached`. Script reported `total hunks: 2 | keeping: 1 |
  dropping: 1`.

**Verified after commit:** `git status --short` shows all 26 unrelated
modified/untracked entries still present, and `git diff` on `sim/solver.py`
shows exactly the user's Custom Effects hunk, unstaged and intact.

**Verified green before commit:** `python -m pytest -q` -> `1402 passed`
(exit 0), matching the documented baseline. That run started before the
comment edit, so the edit was re-verified separately: `python -m py_compile`
plus `pytest -q tests/test_profiler_phase_contract.py tests/test_surface_stack.py`
-> `13 passed`.

**Next:** task 3 — ROI item 1, the surface-tension SDF splat cost.

### Task 3 — ROI item 1, the surface-tension SDF splat

The splat is the single largest surface-tension cost: at the benchmark's
settings `build_sdf` runs **65.4 ms per frame** in `surface_tension_on`, against
**0.43 ms** in `baseline`. The gap is the splat radius. Surface tension forces
`required_sdf_splat_radius_voxels` up to **7** (particle radius 1.2 voxels +
band 1.5 + smoothing 1 + 1), so each particle writes a 15^3 = 3375-cell cube
instead of the default 5^3 = 125. That is the 27x volume the plan identified.

Only a ball of radius `(far_distance + particle_radius) / h` ~ 6.1 voxels can
change the output. **71.83%** of the 3375 cube cells lie outside it
(Monte-Carlo over 20000 sub-voxel particle offsets). So the headroom is real.

Two ways to skip those cells were implemented and measured. The first was
wrong. Both are output-identical by construction, and both were held to that
standard by test: `tests/test_surface_sdf_splat_cull_cuda.py` asserts **exact
float32 bit equality**, not `allclose`.

#### 3a — per-cell distance cull. REFUTED, and it is a real refutation.

Add `if d >= max_distance: continue` to the loop body, ahead of the leaf-table
lookup, on the theory that the lookup (two dependent global loads) and the
atomic (a contended read-modify-write) were the expensive parts.

Correct but **33% slower**. Independently recomputed from the raw per-frame
records in `artifacts/sdf_splat_cull/` (the analysis never reads the tool's own
aggregate), 4 runs, order-reversed, 7 trials x 8 frames per config:

| config | legacy ms/frame | ball ms/frame | delta |
|---|---|---|---|
| `surface_tension_on` | 65.428 | 86.750 | **+32.59%** |
| `both_on` | 54.827 | 71.731 | **+30.83%** |
| `surface_tension_on_no_whitewater` | 36.207 | 45.860 | **+26.66%** |
| `baseline_no_whitewater` (control) | 0.425 | 0.402 | -5.29% |

The control is the load-bearing part: at the default splat radius of 2 the
whole cube already sits inside the cull ball, so the cull provably cannot fire
there, and it doesn't move. The harness is measuring what it claims to.

Order check on wall-clock `surface_tension_on`, per run: legacy 106.14 / 110.06,
ball 133.09 / 126.15. The two modes do not overlap and swapping execution order
did not move them, so this is not drift.

**Root cause:** the splat is ALU-bound on the `wp.length` sqrt, not
memory-bound. Computing `d` in order to decide whether to skip a cell pays
exactly the cost the skip was supposed to avoid, and pays it on all 3375 cells
instead of only the in-leaf ones. **Culling by distance costs the distance.**
The plan's ROI-item-1 premise — that the atomics are the bottleneck — is
refuted by this measurement.

#### 3a' — analytic loop-bound narrowing. Implemented.

Derive the `dk`, then `dj`, then `di` ranges from the ball radius, so culled
cells are never visited, and restore the loop body verbatim. Two sqrts per `dk`
and per `(dk, dj)` replace 3375 per particle.

Two details that matter:

- The legacy sentinel `SDF_SPLAT_NO_CULL = 1e30` overflows float32 when
  squared (`(1e30/h)^2` ~ 1e63 -> inf -> garbage integer bounds), so `rad` is
  clamped to `2*(r+1)`. No cell in the cube reaches `sqrt(3)*(r+1)`, so the
  clamp excludes nothing and legacy mode still walks the exact full cube.
- Each level widens its integer range by one cell. An excluded cell then
  satisfies `|e|^2 > rad^2 + 2*rx + 1`, i.e. it is outside the ball by a full
  voxel in the squared metric — orders of magnitude beyond any float rounding
  in the sqrt.

**Test gap found and closed:** both modes now share the bounds arithmetic, so
legacy-vs-ball can no longer catch a bug that moves both. Added
`k_reference_full_cube_splat`, a frozen verbatim copy of the pre-optimization
loop body, and pinned **both** modes to it bit-for-bit at splat radii 2, 5, 7
and 9. `atomic_min` is order-independent and introduces no rounding, so
nondeterministic atomic ordering cannot explain away a mismatch. **14 passed.**

Kernel-only microbenchmark (400k particles, splat radius 7, alternating modes,
30 samples each): legacy 17.692 ms, bounded 11.182 ms, **-36.79%**, with no
overlap in range (legacy min 16.895 > ball max 12.762). Full paired benchmark
running; the headline is not settled until that lands.

### Task 4 — reuse the `_forces()` SDF in `after_substep()`. UNMERGEABLE.

Closing this as refuted from the code, not from measurement, because the code
is unambiguous. `sim/solver.py:2170-2178`:

```
# Surface tension may have built the SDF before G2P/advection.  The
# particle positions now describe a new interface; invalidate that
# pre-advection key so whitewater and substep render surfacing can
# never consume stale geometry.
self.surface.invalidate_solver_sdf()
sdf_ready = self.surface.after_substep()
```

`_g2p_advect` and `_relax` run between the two builds and move every particle.
The two SDFs describe different geometry, so the second is not a redundant
rebuild of the first — the invalidation sits there deliberately to stop
exactly the reuse this item proposes. Merging them would feed whitewater
pre-advection geometry. **No change made.**

**A reverse-direction candidate did fall out of the analysis** (recorded, not
implemented, not measured — label: Likely):

Surface tension in substep *N+1* could reuse the SDF that `after_substep()`
built at the end of substep *N*. Particle positions are unchanged between those
two points, and `_clear_grid_fields` does not clear the SDF. That would cut
builds from ~2.09 to ~1.09 per substep — a larger win than this item was ever
worth.

The hazard is why it is a candidate and not a change: `solver_sdf_build_key()`
includes `(solver.frame, round(solver.sim_time, 12), ...)`, and that is
load-bearing beyond particle motion — it incidentally invalidates on topology
rebuilds too. Rekeying on geometry would need new explicit invalidations at
topology rebuild and at every feature-hook particle edit, against a documented
fail-closed whitewater invariant. That is a correctness-sensitive change to a
safety mechanism and it belongs in its own scoped piece of work.

#### 3d — paired benchmark of the loop-bound splat. CONFIRMED.

Same design as the 3a pair: 4 runs, order-reversed (legacy, ball, ball,
legacy), 7 trials x 8 frames per config, one GPU, sequential. Raw records in
`artifacts/sdf_splat_bounds/`; every figure below recomputed from the per-frame
data, not from the tool's aggregate.

**Direct probe — `build_sdf` ms per measured frame.** This is the thing that
actually changed, so it is the primary measurement:

| config | legacy (range) | bounded (range) | delta |
|---|---|---|---|
| `surface_tension_on` | 61.551 (59.377-63.725) | 39.779 (38.730-40.828) | **-35.37%** |
| `both_on` | 50.641 (48.071-53.210) | 32.878 (31.617-34.139) | **-35.08%** |
| `surface_tension_on_no_whitewater` | 32.400 (31.550-33.250) | 20.743 (19.920-21.567) | **-35.98%** |

**No overlap between the legacy and bounded ranges on any of the three.** The
worst bounded run beats the best legacy run in every case.

**Frame time, wall lane** (per-trial medians, both legacy runs vs both ball
runs): `surface_tension_on` -19.53%, `both_on` -13.33%,
`surface_tension_on_no_whitewater` -17.23%.

**One confound, resolved rather than ignored.** The wall-lane controls moved
+2% to +11%, which would normally sink the result. The per-run breakdown says
why, and it is not the mode: `baseline` reads 39.71 (A/legacy) then 47.98,
50.67, 48.96 — run A is a cold-machine outlier and every later run agrees with
every other. Recomputing against the *hot* legacy run only (D vs B+C) leaves
the controls at +1.53% / +6.42% / +1.76% and the ST configs at **-21.13% /
-13.21% / -18.90%**, and `build_sdf` at -33.01% / -31.61% / -34.25%. The win
survives both framings, and the drift, being upward during the ball runs, makes
the wall-lane figures conservative rather than flattering.

**At the default splat radius of 2 the change is within noise, in both
directions.** Across all 8 runs of both experiments `baseline_no_whitewater`
spans 0.348-0.545 ms/frame with no mode pattern (legacy median 0.425, bounded
0.443). A single-run comparison suggested +20%; the full sample shows that was
an artifact of comparing one legacy run against two bounded ones. No regression
is claimed and none is detectable — the bounds arithmetic is a fixed per-particle
cost that only matters relative to a cube this small, and it is under 0.1 ms/frame
in absolute terms.

**Verdict on ROI item 1:** the item is confirmed, but *not* by the mechanism the
plan proposed. The win comes from not visiting cells, not from avoiding atomics;
the atomic-contention model was refuted by 3a. Measured
`-35%` of `build_sdf`, `-13%` to `-21%` of the frame in ST configs.

### Task 3b — band propagation. Implemented, bounded, shipped opt-in;
### made the default in iteration 6.

The splat-bounding win of 3a/3d took the *waste* out of the splat. What was
left was not waste: surface tension reads the SDF across a support band the
particle itself does not reach, and at `dam_break` settings that band forces
radius 7 for a 0.6-voxel particle. Every particle was writing 3375 cells to
support values it could not itself determine.

So: seed only the radius the particle support justifies, record *which*
particle won each seeded voxel, and hand those closest points outward one cell
per pass. Cost then scales with active voxels rather than particles x R^3.

**This is not output-identical, so the bound is derived rather than assumed.**
A voxel is seeded by its *true* nearest particle whenever that particle lies
within Chebyshev radius `seed` cells, and Chebyshev distance never exceeds
Euclidean, so every voxel with `sdf <= seed*h - r` carries the exact value.
`band_seed_splat_radius_voxels` sizes the seed so that window covers the force
band *and the one further cell its normal/curvature stencil reads*. Outside the
window, each pass takes a min against another real particle distance — never an
interpolation — so the field is a strict upper bound that only falls with more
passes. Never an underestimate, at any pass count.

Measured (`artifacts/sdf_band_propagate/`, paired, order-reversed, 7 trials x
8 frames, `build_sdf` ms/frame recomputed from the per-frame records):

| config | splat | propagate | delta |
|---|---:|---:|---:|
| `surface_tension_on` | 42.138 | 27.776 | **-34.08 %** |
| `both_on` | 31.594 | 20.280 | **-35.81 %** |
| `surface_tension_on_no_whitewater` | 21.430 | 14.078 | **-34.30 %** |

No range overlap on any of the three. The controls drifted +3.0 % to +6.6 %
*against* the propagate arm, so the wall-lane deltas (-14.22 % / -7.37 % /
-10.42 %) are conservative rather than flattering.

**Accuracy, measured on a 60k-particle blob-and-sheet scene** (23 tests,
`tests/test_surface_sdf_band_propagate_cuda.py`): max underestimate 3.725e-07
voxels at every seed radius tested — one float32 ulp, i.e. zero real
underestimates. Inside the exact window, error 1.9e-07 to 3.7e-07 voxels.
Outside it, error falls monotonically with passes (0.0786 / 0.0565 / 0.0178 /
0.0000 voxels at seeds 2/3/4/5), confirming convergence from above. At the
derived seed the capillary stencil is untouched: normal direction moves
3.0e-5 degrees, curvature 2e-6 relative — 33x and 500x inside the assertion
bounds, so the test is not vacuous.

**Left opt-in** (`default_sdf_band_mode()` is still `splat`). Flipping it
changes physics output; that is a user decision.

> **Superseded in iteration 6.** The user took that decision on this
> evidence, and `_default_sdf_band_mode` is now `SDF_BAND_MODE_PROPAGATE`.
> The paragraph above is kept because it records what was true when the
> measurement was made, not because it is still true.

**A refutation worth keeping.** A fused single-pass seed using a packed 64-bit
`(distance, index)` `atomic_min` was implemented and measured *first*. On sm_89
the contended int64 atomic costs 2.9-3.6x more than the 32-bit pass it removes
(seed 4, 200k particles: 1.71 ms two-pass vs 5.05 ms fused). Reverted; the
refutation is recorded in `k_seed_surface_sdf_closest`'s docstring so it is not
retried blind.

Commit `4ecc21a`, three files, staged explicitly.

### Task 5 — the viscosity tolerance knee, re-measured. The speed case is dead.

Six sequential order-reversed runs (1e-4, 3e-4, 1e-3, 1e-3, 3e-4, 1e-4),
7 trials x 8 frames, `dam_break`, cuda:0, two runs pooled per arm.
**No default was changed** — the tolerance was passed on the command line only.
Raw: `artifacts/visc_tolerance_knee/run{1..6}_tol*`.

| tolerance | viscosity solve ms/frame | residual-stall ms | remainder ms | u iters | v iters | w iters |
|---|---:|---:|---:|---:|---:|---:|
| 1e-4 (**shipped**) | 23.940 | 11.845 | 12.095 | 36.38 | 35.13 | **40.00** |
| 3e-4 | 22.421 | 10.391 | 12.030 | 31.10 | 29.69 | 35.68 |
| 1e-3 | 22.253 | 9.090 | 13.163 | 25.71 | 23.57 | 29.37 |

**The old table's -21.9 % / -32.9 % are superseded.** Measured against the
post-conditional-graph solver, a full decade of tolerance buys **-7.04 %** of
the viscosity solve. Those projections were never wrong for the solver they
described; they are wrong for the solver that now ships.

**And the wall lane produced no usable signal, which is reported rather than
worked around.** The controls drifted +8.84 %/+9.91 % (`baseline`) and
+7.48 %/+7.10 % (`baseline_no_whitewater`) over the six runs, while the two
runs of the shipped arm alone differed by 11.7 % (54.574 vs 60.953 ms/frame).
Every between-arm difference is smaller than the within-arm spread. No wall
figure is quoted for this item, in either direction.

**Why the saving is small, and the new item that falls out of it.** Iterations
do drop ~27 %, and that lands where expected: the residual host-sync stall —
where the on-device PCG loop's GPU time is actually waited on — falls
-12.27 %/-23.26 %. But that stall is only **49.5 %** of the viscosity solve.
The remainder is ~12.1 ms/frame and is **flat across a full decade of
tolerance**. It does not care how many iterations ran, so no iteration-count
change can reach it. That flat half is now the larger of the two viscosity
terms and the largest remaining viscosity lever (logged as plan item 3g).

**What 3f is still worth deciding: accuracy, not speed.** At the shipped
tolerance the w component's mean stop iteration is **exactly 40.00** — it
exhausts its budget on every solve in the sample and never converges. At 3e-4
the mean is 35.68, at 1e-3 it is 29.37; both converge. Today's default silently
under-converges w, and the cost of fixing that is now under 7 % of the
viscosity solve rather than the ~22 % the old table implied. That strengthens
the case for 3e-4 — but it changes simulation output, so it stays a user
decision.

## Final summary — what this loop delivered

**Two commits, both surface tension. Viscosity's solver was not touched this
session** (its conditional-graph win predates this loop).

| commit | change | output-identical? |
|---|---|---|
| `c6ee7f7` | bound the particle-SDF splat to the ball that can win | **yes**, by construction |
| `4ecc21a` | carry closest points out to the capillary band | no — derived bound; **opt-in then, default since iteration 6** |

**Measured before/after, `build_sdf` ms/frame** (each stage paired and
order-reversed against its own baseline; the composed column is the product of
the two measured ratios, not an end-to-end run, so it is **Likely** where the
per-stage figures are **Verified**):

| config | stage 1: bound | stage 2: propagate | composed |
|---|---:|---:|---:|
| `surface_tension_on` | 61.551 -> 39.779 (-35.37 %) | 42.138 -> 27.776 (-34.08 %) | -57.40 % |
| `both_on` | 50.641 -> 32.878 (-35.08 %) | 31.594 -> 20.280 (-35.81 %) | -58.33 % |
| `surface_tension_on_no_whitewater` | 32.400 -> 20.743 (-35.98 %) | 21.430 -> 14.078 (-34.30 %) | -57.94 % |

**Verification:** `python -m pytest -q` gives **1436 passed** in 167.53 s
(1413 + 23 new). `python tools/run_production_gate.py` reaches
**`HEADLESS-GATE-PASS`** (320 checks passing, 0 failing) plus
`DEMO-BLEND-CHECK-PASS`.

### What was refuted

1. **The atomic-contention model** behind ROI item 1 (3a). Culling per cell by
   distance was *slower*; the win came from not visiting cells at all.
2. **A fused 64-bit packed-key seed pass.** Implemented and measured, not
   argued about: 2.9-3.6x slower than the 32-bit pass it replaces on sm_89.
3. **ROI item 1's "~90 % of the rebuild cost" target.** Unreachable in
   principle, not just in practice — the radius-2 cost it aimed at is what the
   *particle* needs, while surface tension additionally reads a band the
   particle does not reach. The smallest exact seed radius here is 4. Delivered
   -57.40 %.
4. **Item 3f's -21.9 % / -32.9 % projections**, superseded by direct
   measurement at -6.3 % / -7.0 %.
5. **ROI item 2a**, closed as genuinely unmergeable: `_g2p_advect` and `_relax`
   run between the two builds, so merging would feed whitewater pre-advection
   geometry.

### What remains open

- **Plan item 3g: implemented and measured.** See the section below; this
  entry is kept for the investigation that sized it. The flat half is
  host-side `wp.launch` marshalling, not GPU work. Measured three ways on `dam_break`, viscosity on,
  8 frames after 3 warmup, 31 substeps, 93 component solves (11.62 per frame):

  | bucket | ms/frame | % of `_solve_component` |
  |---|---:|---:|
  | wait on the device PCG loop (`pcg_stats.numpy()`) | 10.505 | 55.2 % |
  | **the 9 `wp.launch` calls outside the loop** | **4.911** | **25.8 %** |
  | graph-cache misses (24 of 93 lookups) | 1.233 | 6.5 % |
  | `wp.capture_launch` | 0.730 | 3.8 % |
  | `array.zero_()` x4 | 0.528 | 2.8 % |
  | `wp.copy` write-back | 0.259 | 1.4 % |
  | closing device drain | 0.134 | 0.7 % |
  | unattributed (incl. `_pcg_dot.launch`) | 0.706 | 3.7 % |
  | **total** | **19.030** | 100 % |

  Two nested double-counts were stripped before this table: `array.numpy()`
  calls `wp.copy` internally, and the seven `[capture]` launches run inside
  `graph lookup MISS`. What remains closes to 3.7 %.

  The graph-cache-thrash hypothesis was **refuted**. `_graph_key` does carry
  `int(n)` and `float(dt)`, and the key set does churn -- 24 distinct keys and
  a 25.8 % miss rate over 8 frames, against an LRU that reached its 32-entry
  capacity -- but a miss costs only 0.470 ms and the whole line is 6.5 %.

  The cause is that the closing device drain is **0.134 ms/frame**. The GPU
  is not busy; the host is. Priced out of the simulation on the real kernels
  at N = 80 000 (400 reps x 7 trials):

  | case | median host us/call |
  |---|---:|
  | `wp.launch`, 1 arg, `dim=1` | 27.86 |
  | `wp.launch`, 10 args, `dim=N` | 52.74 |
  | the 9-call setup + write-back sequence | 347.97 |
  | **the same sequence as one captured graph** | **12.92** |

  So the fixed floor is ~28 us per `wp.launch` regardless of the work, and
  argument marshalling roughly doubles it for the wide stencil kernels. This
  is the same disease the conditional graph already cured for the inner loop,
  left untreated on either side of it.

  **The sized fix:** extend the captured region from the PCG loop alone to the
  whole component solve, so setup and write-back are enqueued with the loop
  instead of issued call by call. Isolated, that is 347.97 -> 12.92 us, i.e.
  4.043 -> 0.150 ms/frame at 11.62 solves/frame. The two methods do not cover
  an identical call set -- in situ the nine launches are the seven pre-loop
  ones plus `k_viscosity_prepare_pcg_conditional_loop` and
  `k_viscosity_apply_collision`, totalling 4.911 ms/frame, while the isolated
  sequence swaps the first of those two for the `wp.copy` -- so the two
  corroborate to about 18 % rather than repeat each other. Expect
  roughly 4 to 5 ms/frame, ~25 % of the viscosity solve. `array.zero_()` and
  the `wp.copy` are capturable too, worth another ~0.8 ms/frame.

  **What has to be checked before claiming it:** the capture already keys on
  `n` and `dt`, so setup carries no new key material, but a larger captured
  body makes each of the 24 misses/8 frames more expensive, and that has to be
  paid out of the 4-5 ms before any net figure is honest. `_active_count()`
  and the status readback stay outside the capture. No speedup may be quoted
  without a paired, order-reversed before/after from one binary with a mode
  flag.

## Plan item 3g — implemented, and what it actually bought

The captured region now extends from the PCG loop alone to the whole component
solve. Setup, the conditional preamble, the loop and the write-back are
enqueued as one graph, so a component solve costs one `wp.capture_launch`
where it used to cost sixteen host calls.

`sim/viscosity.py` gains a mode flag, `VISCOSITY_GRAPH_SCOPE_LOOP` and
`VISCOSITY_GRAPH_SCOPE_COMPONENT`, defaulting to `component`, reachable from
the profiler as `--viscosity-graph-scope`. The flag exists so before and after
can be measured from **one binary**, which is what the evidence rule above
demands; it is not a user-facing setting.

### The measurement

Eight runs of `tools/profile_viscosity_surface_tension.py` at its defaults
(dam_break, 8 frames after 3 warmup, 3 trials, all six configurations),
arranged as four order-reversed pairs — loop/comp, comp/loop, comp/loop,
loop/comp — run strictly sequentially on the one GPU.

**Whole-frame step time produced no usable magnitude, and is not quoted as
one.** The same failure as item 3f: across these eight runs the untouched
`baseline` configuration moves by more than the effect, and the three
configurations with viscosity *switched off* — which this flag cannot reach —
moved by up to 25 % within a single pair. Whole-frame time agrees on
**direction** (component faster in 4 of 4 pairs on `both_on`, 3 of 4 on
`viscosity_on`) and nothing more.

**The profiler already records the thing itself**, so that is what is quoted.
`probes[cfg].viscosity.component_ms` is host wall time inside
`_solve_component`, and `component_calls` is exactly `3 x substepsMeasured`
in all eight runs (one solve per velocity axis per substep). Host launch cost
is precisely what item 3g removes and precisely what this probe brackets.

| config | loop | component | change |
|---|---:|---:|---:|
| `viscosity_on`, host ms per component solve | 0.6159 | 0.4212 | **-31.62 %** |
| `both_on`, host ms per component solve | 0.5657 | 0.4246 | **-24.95 %** |

Component is faster in **4 of 4** order-reversed pairs for both configs, per
pair by -29.07/-39.81/-25.09/-30.67 % and -28.67/-25.47/-17.20/-23.94 %. The
twelve loop trials and the twelve component trials **do not overlap at all**:
`viscosity_on` loop spans 0.5575-0.6724 ms and component 0.3736-0.4876 ms.
The four control configurations report `component_calls = 0`, which is the
check that this metric is attributing to the right code.

At 14.25 component solves per frame:

| config | saved/frame | as % of frame | as % of the viscosity increment |
|---|---:|---:|---:|
| `viscosity_on` | 2.775 ms | 4.50 % | 12.34 % |
| `both_on` | 2.011 ms | 2.59 % | 5.22 % |

**Verified** for the per-solve figures — paired, order-reversed, one binary,
non-overlapping distributions. **Likely** for the per-frame translation: it is
the per-solve saving multiplied by the measured solve count, not an end-to-end
frame measurement, which this machine's variance cannot deliver.

Against the sizing that motivated the work: the isolated pricing predicted
4-5 ms/frame. Measured, it is 2.775. The prediction was **optimistic by ~1.6x**
— priced-in-isolation launch costs overstate what the same calls cost inside a
running solver.

### Correctness

Same kernels, same order, same arguments — only where they are enqueued from
differs. Three checks, because a graph capture is easy to get silently wrong:

1. **It really captures.** `array.zero_()` (a memset) and `wp.copy` (a memcpy)
   had never been issued inside a capture in this solver; either being
   rejected would have been swallowed by the fallback and the "speedup" would
   have been the fallback path. On a real GPU run the component arm reports
   `executionMode = "cuda_conditional"` with an empty `nativeError`.
2. **It computes the same thing.** A whole-run digest is *not* a valid oracle
   here — the solver accumulates with `atomicAdd` in the particle scatter and
   in the PCG dot products, so two identical runs already differ in the last
   bits. Instead one mid-simulation state was frozen and the same solve
   replayed four times, loop/loop/component/component. All four arms returned
   `max_iterations`, 40 iterations, residual 2.914e-02, and cross-scope
   disagreement (7.15e-07) is **1.50x** same-scope disagreement (4.77e-07) —
   i.e. within the solver's own nondeterminism.
3. **The two paths cannot drift.** The launch sequences are factored into
   `_record_*` helpers used by both, and a unit test asserts the two scopes'
   *expanded* op sequences are equal — same kernels, same dims, same order.
   `tests/test_viscosity.py` is 25 passed (18 pre-existing + 7 new).

The `component` graph writes `comp` only in its last two ops, so an aborted
capture leaves the velocity component untouched and the host-issued path can
redo the solve from scratch. `_active_count()` and the status readback stay
outside the capture, as required.
- **The reverse SDF-reuse candidate** exposed by task 4: reuse substep N's
  post-advection SDF for substep N+1's forces (~2.09 -> ~1.09 builds/substep,
  larger than item 2a ever was). Recorded, deliberately **not** implemented —
  it requires rekeying a fail-closed whitewater invalidation, which is its own
  scoped piece of work.
- **Plan item 3c**, fusing the three per-axis viscosity solves. Unquantified.
- **Plan item 4**, quality-tier scaling of `viscosity_iterations`. Blocked on
  `sim/quality_contract.py`, which exists only on an unmerged branch.

### Three decisions reserved for the user — two since taken, one still open

1. **Does `fix/settings-apply-domain-object` merge to master?** Plan item 4 is
   blocked behind it. **Still reserved.** Merging or cherry-picking from it was
   not authorized and was not done.
2. **Item 3f: tolerance and iteration budget.** **Taken — and the
   recommendation recorded here was wrong, so it is corrected rather than
   quietly dropped.** 3e-4 does not *cost* ~6 % of the viscosity solve, it
   **saves** 6.34 %: loosening the tolerance makes w stop earlier, which makes
   the delivered residual worse while making the self-report honest. The
   accuracy fix was the opposite move. Measured with the ceiling lifted to 256,
   w converges at 1e-4 in 41-46 iterations and u in 35-41 — the shipped ceiling
   of 40 sat below w's *minimum*, so w was truncated on every solve.
   `DEFAULT_VISCOSITY_ITERATIONS` is now **64**; `DEFAULT_VISCOSITY_TOLERANCE`
   stays at 1e-4, because the fix was to reach the tolerance rather than lower
   the bar to the budget. The status plumbing this entry asked for was done in
   the same change: `FluidSolver._viscosity` was discarding `solve()`'s return
   value entirely, and now records it into `stats` the way `_pressure` does.
3. **Does band propagation become the default?** **Taken.**
   `_default_sdf_band_mode` is `SDF_BAND_MODE_PROPAGATE`; the splat stays
   reachable as the reference path. See the iteration 6 section of the plan.

### Standing constraints honoured

One GPU throughout — every benchmark ran sequentially, never two at once.
Every headline number in this file was recomputed from the raw JSON named
beside it; nothing is estimated. Both commits staged an explicit file list; the
31 unrelated dirty worktree entries were preserved untouched.
