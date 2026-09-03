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
