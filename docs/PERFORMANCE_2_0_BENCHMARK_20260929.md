# FluidSim Warp 2.0 solver benchmark (2026-09-29)

## Answer

**Against the code before the optimization work** (`pre_8c6dbd2`), 2.0 is:

| Workload | Before | 2.0 | 2.0 ÷ before (paired) | Rounds 2.0 was faster |
|---|---|---|---|---|
| Viscosity | 301 ms/frame | 56 ms | 0.177, about **5.6× faster** | 5 of 5 |
| Viscosity + surface tension | 327 ms | 75 ms | 0.221, about **4.5× faster** | 5 of 5 |
| Surface tension | 152 ms | 142 ms | 0.938, **6% faster** | 4 of 5 |
| Plain (neither) | 75 ms | 81 ms | 1.071, **7% slower** | 2 of 5 |

The plain-workload cost is known and deliberate. Since 1.1.0, particle-to-grid
transfer (P2G) uses a sorted gather (`d2a6b6d`) so bakes are bitwise
reproducible. Per substep the gap is 2.4% (0.7% to 10% across rounds). Per
frame the rounds range from 5% faster to 9% slower, so the 7% is the noisier
figure.

**Against 1.1.0, 2.0 has no regression.** Paired ratios:
- plain 0.925 (faster in 4 of 5 rounds)
- viscosity 0.901 (5 of 5)
- surface tension 0.964 (3 of 5)
- viscosity + surface tension 1.040, range 0.96–1.08 (1 of 5). Per substep that
  is 1.026, inside noise and inside the 5% budget.

**master today costs the same per substep as 2.0**: ratios between 0.98 and
1.05. Per frame it is 6–10% higher on three workloads:
- **Plain and surface tension.** The free-slip collider walls (`ce26141`)
  change the flow, and it needs about 5% more substeps (368 against 349, and
  363 against 346). That is a physics change, not slower code.
- **Viscosity.** It reads 1.085 with a range of 1.00–1.17, while master takes
  142 substeps against 141. Treat it as mostly noise.

**The REVIEW-015 fix** cuts pressure-graph captures on this scene from 35 to 3
and saves about 2.9 ms per frame. Results are unchanged. Whole-frame timing
agrees on three of four workloads (plain 0.964), but the effect is at the edge
of this machine's noise. See [REVIEW-015 on its own](#review-015-on-its-own).

## What was measured

One workload, run on five checkouts of this repository:

| Label | Commit | What it is |
|---|---|---|
| `pre_8c6dbd2` | `8c6dbd2` (2026-08-26) | The code before the optimization work started. It is 1.0.255 (`bb6d3fc`) plus 13 commits (liquid color mixing and review fixes). |
| `v1.1.0` | `c5919d6` | Release 1.1.0 |
| `v2.0.0` | `ad769d2` | Release 2.0.0 |
| `master` | `b454012` | master today, without the REVIEW-015 fix |
| `master_fix` | branch `fix/review-015-pcg-graph-cache` | master plus the REVIEW-015 fix |

**Workload.** The `dam_break` demo scene as `tools/profile_gpu_solver.py`
mirrors it:
- 5.6 × 2.0 × 1.6 m domain at a voxel size of 0.03.
- A box source emitting for the first 4 frames, about 206k particles.
- Surface and whitewater on.
- Substeps between 1 and 16.

It is solver-only: it runs `FluidSolver.step_frame()` without Blender. Four
variants:
- `plain`: no viscosity, no surface tension.
- `visc`: viscosity 1.0.
- `st`: surface tension 0.072.
- `visc_st`: both.

**Same work on every version.** master's new Export Attributes switch
defaults to off. Earlier versions always exported the attributes that were
enabled, so the benchmark turns the switch on wherever it exists. A first
attempt without this made master look faster than it is. That run was
discarded.

**Timing.**
- Each trial is a fresh process that runs 2 warm-up frames, then 40 measured
  frames.
- Each frame is timed on the host around `step_frame()` plus a device
  synchronize, so no version's own statistics decide its score.
- A trial's score is its median frame time.
- There are 5 rounds. Each round runs every version on every workload, with
  the order rotated each round so drift on the machine does not always land
  on the same version.
- A ratio is paired: version B divided by version A within the same round,
  then the median over the rounds.

**Machine.**
- NVIDIA GeForce RTX 4080 SUPER, driver 610.47.
- Blender 5.2's Python.
- Warp 1.17.0 for every version, so the Warp version is not a factor.

## Results

Median ms/frame is the median of the five trial medians; the range is the
fastest and slowest trial. ms/substep is total frame time divided by total
substeps, median over trials.

**ms/frame [fastest..slowest trial] and ms/substep**

| Workload | pre_8c6dbd2 | v1.1.0 | v2.0.0 | master | master_fix |
|---|---|---|---|---|---|
| plain | 75.1 [72.1..88.0], 8.62 | 86.6 [74.0..104.1], 9.60 | 81.4 [77.2..84.0], 9.00 | 85.0 [81.9..89.0], 8.74 | 82.9 [79.0..89.5], 8.63 |
| visc | 300.9 [289.7..336.6], 95.34 | 58.0 [50.3..70.3], 17.09 | 56.0 [42.3..59.7], 16.35 | 58.2 [49.6..64.2], 17.25 | 57.9 [47.6..67.6], 16.63 |
| st | 152.3 [141.1..159.6], 16.26 | 143.3 [142.4..149.0], 16.15 | 142.1 [138.2..148.5], 16.20 | 157.3 [143.4..166.9], 16.28 | 157.0 [147.1..157.8], 16.28 |
| visc_st | 326.6 [302.7..373.8], 97.63 | 73.4 [67.7..77.1], 22.64 | 75.2 [70.4..79.5], 22.13 | 71.6 [68.2..74.4], 22.50 | 72.0 [64.0..87.2], 22.04 |

Viscosity frames are cheaper than plain ones on 1.1.0 and later. Thick fluid
moves slowly, so the solver needs far fewer substeps (141 against 349).

**Paired ratios, ms/frame (range over rounds; rounds faster; ms/substep ratio)**

| Comparison | plain | visc | st | visc_st |
|---|---|---|---|---|
| v1.1.0 ÷ pre | 1.115 [1.03..1.18]; 0/5; 1.106 | 0.195 [0.15..0.22]; 5/5; 0.183 | 0.940 [0.93..1.01]; 4/5; 0.994 | 0.231 [0.19..0.24]; 5/5; 0.223 |
| v2.0.0 ÷ pre | 1.071 [0.95..1.09]; 2/5; 1.024 | 0.177 [0.13..0.20]; 5/5; 0.169 | 0.938 [0.89..1.05]; 4/5; 0.977 | 0.221 [0.20..0.26]; 5/5; 0.227 |
| v2.0.0 ÷ v1.1.0 | 0.925 [0.81..1.04]; 4/5; 0.923 | 0.901 [0.84..0.99]; 5/5; 0.910 | 0.964 [0.95..1.04]; 3/5; 0.978 | 1.040 [0.96..1.08]; 1/5; 1.026 |
| master ÷ v2.0.0 | 1.059 [1.00..1.09]; 0/5; 0.992 | 1.085 [1.00..1.17]; 1/5; 1.047 | 1.097 [1.01..1.18]; 0/5; 1.009 | 0.968 [0.89..1.00]; 5/5; 0.980 |
| master_fix ÷ master | 0.964 [0.95..1.05]; 4/5; 0.959 | 0.959 [0.94..1.16]; 3/5; 0.994 | 0.994 [0.94..1.03]; 4/5; 0.987 | 1.036 [0.89..1.17]; 2/5; 1.037 |

**Determinism, as a side result.**
- 1.1.0 and later took the same substeps, pressure iterations and final
  particle count in every trial.
- `pre_8c6dbd2` did not: plain took 342 to 359 substeps across trials.
- master and master_fix match each other exactly on all three counts, so the
  REVIEW-015 fix does not change results.

## REVIEW-015 on its own

The in-process measurement is the real evidence. On dam_break over 42 frames:
- Pressure-graph captures fell from 35 to 3.
- Time spent capturing fell from 158 ms to 38 ms, about 2.9 ms per frame.
- Particle state hashes the same on every frame.

On an 83 ms plain frame, 2.9 ms is about 3.5%. The whole-frame paired median
for plain is 0.964 (faster in 4 of 5 rounds), which matches. Viscosity (0.959)
and surface tension (0.994) point the same way. Viscosity + surface tension
reads 1.036, but its range is 0.89 to 1.17, so it is noise. A saving this
size sits at the edge of what whole-frame timing on this machine can resolve.

## How this fits the earlier 2.0 numbers

The 2.0 release (handoff section 3) measured Blender demo bakes against
1.0.255, in ABBA pairs, with viscosity and surface tension off:
- crown_splash 0.973
- dam_break 1.004
- paddle_stir +4.0% on absolute medians, +5.5% paired
- chromatic_vortex −2.2% against 1.1.0

Evidence is in `C:\Users\Tyler\fluidsim-gate-logs\perf-2.0-20260927\`.

That comparison already answered "no worse than 1.0.255" for the demos. This
benchmark adds four things it could not show:
- **Viscosity and surface tension.** The demos had them off, so the 4.5–5.6×
  viscosity win never appeared there. That work shipped in 1.1.0.
- **One Warp version.** Every version here runs on Warp 1.17. In the earlier
  run, 1.0.255 ran on its own Warp.
- **1.1.0 and master** in the same run as 2.0.
- **The pre-optimization code itself.** `pre_8c6dbd2` is 1.0.255 plus 13
  commits (liquid color mixing and review fixes), none of them performance
  work.

The plain result here (+7% per frame, +2.4% per substep) and dam_break's 1.004
there are consistent. This run's plain range (0.95 to 1.09) covers 1.004. Both
point at the same cost, the deterministic P2G gather, which the stage profiler
measured at 1.78 → 3.42 ms per frame on paddle_stir.

The Blender-side wins from August (for example GPU surface attributes, −88% of
frame time when nearest-particle channels are on) are in
`docs/REAL_WORLD_PERFORMANCE_RESULTS_20260827.md`. This solver-only benchmark
does not measure them.

## Limits

- **Solver only.** Mesh building, attribute writes to Blender, cache I/O and UI
  cost are not in these numbers.
- **One scene, one GPU, a shared workstation.** A single trial's median can
  move 10–25% (v1.1.0 plain ranged from 74 to 104 ms). Only paired ratios
  that point the same way in most rounds count. Five rounds settle the
  viscosity wins but not differences under about 5%.
- **master changed the physics.** Free-slip walls take more substeps, so
  compare master with 2.0 per substep, not per frame.
- **Before the optimization work, viscosity never converged.** Every solve ran
  all 40 PCG iterations. `1635f90` (1.1.0) lets it stop at its tolerance, so
  part of the viscosity speedup is a correctness fix, not only faster code.
- **Open finding: viscosity graph churn.** A probe on master_fix, viscosity
  workload, 42 frames:
  - 435 viscosity solves made 213 fresh graph captures.
  - Capturing took 411 ms, about 10 ms per frame, or roughly 17% of a 57 ms
    frame.
  - The cache key includes the solve size, `dt` and the tolerance (32, 5 and
    32 distinct values). Without them there are only 6 keys.

  This is the same kind of problem as REVIEW-015, but larger. It is not fixed
  here.

## Reproduce

Create one detached worktree per version, then run the tool from any
checkout that has it:

```powershell
git worktree add --detach C:/Users/Tyler/wt/bench_pre 8c6dbd2
git worktree add --detach C:/Users/Tyler/wt/bench_v11 v1.1.0
git worktree add --detach C:/Users/Tyler/wt/bench_v20 v2.0.0
git worktree add --detach C:/Users/Tyler/wt/bench_base b454012

$env:PYTHONPATH = "C:\Users\Tyler\fluidsim-warp\site-1.17.0;C:\Users\Tyler\fluidsim-warp\devsite"
$env:PYTHONNOUSERSITE = "1"
& "C:\Users\Tyler\Downloads\blender-5.2.0-windows-x64\blender-5.2.0-windows-x64\5.2\python\bin\python.exe" `
    tools/benchmark_versions.py `
    --version pre_8c6dbd2=C:/Users/Tyler/wt/bench_pre --version v1.1.0=C:/Users/Tyler/wt/bench_v11 `
    --version v2.0.0=C:/Users/Tyler/wt/bench_v20 --version master=C:/Users/Tyler/wt/bench_base `
    --version master_fix=C:/Users/Tyler/wt/rev015 `
    --trials 5 --frames 40 --warmup 2 --out-dir artifacts/bench_2_0_20260929
```

The run took about 13 minutes. Raw per-trial JSON and `summary.json` are in
`artifacts/bench_2_0_20260929/` in the main checkout, which git ignores.
