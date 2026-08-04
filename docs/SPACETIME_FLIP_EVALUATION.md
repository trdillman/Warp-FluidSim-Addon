# Spatiotemporal FLIP — implementation status and measured evaluation

Reference: Braun, Winchenbach, Bender, Thuerey, *Spatiotemporal FLIP*, ACM TOG 45(4)
Art. 76.

This document records what the ST path implements, what it measurably costs, and
what is still unexplained. It exists because the headline question — "is ST worth
enabling for high-resolution liquid on one consumer GPU?" — has a measured answer
that does not match the paper's scaling claim, and that answer should not have to
be rediscovered.

All numbers below: NVIDIA Warp on `cuda:0`, RTX 4080 SUPER (16 GiB, sm_89), Blender
5.1 bundled Python. Harness `tests/headless_equilibrium_ab.py`, a sealed tank filled
to a known rest depth; `occupiedVolume` is computed in the harness from raw particle
positions on its own analysis grid, deliberately independent of solver voxel
bookkeeping, so clumping shows up as volume error rather than hiding.

---

## 1. Read this first: the solver is not run-to-run deterministic

Three byte-identical repeats, dx 0.025, CFL 1, 10 simulated seconds:

```
ST-off   volErr% = -4.6957 / -5.2754 / -5.9833      spread 1.29 pp,  sd 0.53
         ke      =  0.2348 /  0.1836 /  0.1807      spread 30% of mean
         solve s =  28.11  /  27.19  /  25.65       sd 1.01 s (~3.5%)

ST-on    volErr% = 17.6524 / 18.3015 / 17.4957      spread 0.81 pp,  sd 0.35
         ke      =  0.1339 /  0.0973 /  0.1220      spread 32% of mean
         solve s =  37.63  /  35.06  /  36.94       sd 1.08 s
```

Cause is almost certainly atomic float-add ordering in P2G — float addition is not
associative, and the ordering differs per launch — chaotically amplified over 10
simulated seconds. This is a pre-existing property of the solver, unrelated to any
ST work.

**Consequences for anyone benchmarking this repo:**

- A single-run A/B difference below roughly **1 pp of volume error** is noise. Several
  intermediate conclusions during this evaluation had to be retracted for this reason.
- **`kineticEnergyMean` is unusable at n=1.** ~30% repeat scatter.
- Wall-time sd is ~3.5%, so timing claims need >10% separation or repeats.
- Any test asserting bit-identical output across two runs of the *same* configuration
  cannot pass today. If exact reproducibility is wanted, it needs deterministic
  reduction ordering in P2G, which is a separate piece of work.

---

## 2. Cost: ST is more expensive, and the penalty grows with resolution

Equal simulated duration, CFL 1, ST-off vs ST-on wall clock:

```
    dx     particles    off_s     on_s    on/off
 0.050         0.21M     5.60     6.08      1.09
 0.040         0.41M     6.17     6.70      1.09
 0.032         0.78M     8.18     9.85      1.20
 0.025         1.66M    15.81    19.08      1.21
 0.020         3.24M    30.67    38.86      1.27
 0.016         6.27M    71.27    95.70      1.34
```

The overhead **increases** with scale (1.09× to 1.34×), which is the opposite of the
paper's several-fold-speedup-at-scale claim. Endpoint separation is far above the
~3.5% timing noise, so the trend is real.

Iso-wall-time consequence: at an 80 s budget, ST-off affords dx 0.016 (6.27M
particles) while ST-on affords dx 0.020 (3.24M). **1.9× more particles for the same
wall time with ST off.** There is no tested budget at which ST buys resolution.

### Where the cost goes

Per-phase attribution, dx 0.025, CFL 1, `--sync`, ms/frame, leaf phases only
(`total`/`compute`/`solver` are rollups and excluded):

```
phase        ST off    ST on     delta
pressure      76.81   107.85   +31.04    <- 64% of the whole delta
g2p           42.56    49.32    +6.77
p2g           20.87    22.63    +1.76
collision      2.71     4.68    +1.97
sourcing      13.19    15.04    +1.85
lifecycle     11.80    13.48    +1.67
relax         15.54    16.78    +1.24
misc           8.14     9.02    +0.88
indexing       6.88     7.64    +0.76
forces         1.72     2.02    +0.30    <- sigma = 0, as designed
surface        0.12     0.13    +0.01
viscosity      0.07     0.09    +0.01
rollup       200.73   249.06   +48.33
```

+24% per frame at matched CFL, and **64% of it is `pressure`**. The standard PPE
family is CUDA-graph captured; the ST family is not (`k_apply_matrix_st` launches
`dim=_dim()`, unindexed) and evaluates per-face beta on the fly. That is the single
highest-leverage optimisation if ST is kept.

`forces` staying flat confirms the CSF surface-tension path is genuinely free at the
default sigma = 0.

Note: a profiled run (`--sync`) costs more than an unprofiled one. Never read
`solveSeconds` and the phase split from the same run.

---

## 3. Quality: a real phase-field bias at paper defaults

10 simulated seconds, dx 0.025, expected rest depth 0.8100:

```
  ST  CFL  solve_s  substeps  volErr%   depth      ke
 off    1    28.46        28    -5.15  0.7555  0.1228
 off    2    17.39        18   -13.06  0.6916  0.2915
 off    4    12.45        10   -21.30  0.6246  0.4335
 off    8     6.90         4   -40.31  0.4705  0.0814
  on    1    35.72        37   +17.02  0.9359  0.0850
  on    2    21.96        23   +16.03  0.9279  0.0833
  on    4    14.58        12   +11.28  0.8891  0.0628
  on    8    10.44         7    +1.29  0.8081  0.0462
```

ST-on degrades far more gracefully with CFL than ST-off does — that part matches the
paper. But at the paper's own defaults it holds the liquid roughly 17% too voluminous
(depth 0.94 against an expected 0.81), where the non-ST path loses volume instead.
`particleDrift` is 0.00 and the particle count is fixed, and `occupiedVolume` is
measured off raw particle positions, so the liquid genuinely occupies more space —
this is not a bookkeeping artefact.

### What was ruled out

- **Calibration is correct.** `∫W_S_1d dx = 1.0`, `∫W_T dτ = 1.0000`,
  `W_T(-0.5/0/0.5) = 0.0/0.9229/2.1875`, `particles_per_voxel = 8.0`, calibrated
  m0 = 7.7251 vs analytic 8.0 (−3.4%, and the wrong direction to explain a 22 pp
  swing). eta_phi = 0.5 and the phase threshold 0.5 are both the paper's stated
  defaults (Sec. 3.6 and the parameters block). Nothing is mis-set.
- **No single implemented feature owns it.** Turning each ST sub-feature off in turn
  leaves the error at 14–17%; none returns to the ST-off signature. Worth noting the
  reverse result: disabling variable-density projection makes it *worse* (+26.90%),
  so that stage is recovering roughly 10 pp and earns its cost.
- **eta = 1.0 is scene fitting, not a fix.** It zeroes the error at dx 0.025 / CFL 1
  but sign-flips across the range (−19.60% at CFL 8, −7.60% at dx 0.050). Rejected.
  It is however informative: at eta = 1.0 the ST path beats ST-off at *every* CFL by
  roughly 2× on volume error. The method can carry better quality; the defect is a
  phase-field bias, not the method.
- **Jensen bias in the phase kernel is not the cause.** `C(x) = min(sqrt(x), 1)` is
  concave, so averaging per-face `C` values over-estimates phi relative to applying
  `C` once to the mean face mass (Eq. 13 defines phi at a point). The kernel was
  corrected to the latter on fidelity grounds. Measured, it moved volume error from
  +17.02% to a mean of +17.86% over four runs — the wrong direction and within about
  2 sd of the repeat noise. Kept for correctness, but it explains nothing.

### What remains open

Both the old and corrected kernel use a **6-face mass average as a surrogate for
m(x_c)**; the paper samples the phase field at the cell centre. A cell-centred
occupancy is already available at no cost — `voxel_counts` is filled by
`_index_particles()` immediately before `_p2g()` — but a raw count discards the
slab-integrated smoothing that is the point of the method. A properly smoothed
cell-centred mass needs a new P2G accumulation pass, roughly +7 ms on a 200 ms frame.

This is the leading remaining hypothesis for the +17%, and it is a design decision
rather than a bug fix, so it is recorded here rather than implemented.

Also unresolved: the phase threshold and eta both sweep the error monotonically
through zero (thr 0.2 → +25.45%, 0.5 → +17.02%, 0.8 → −1.47%), which is consistent
with too many marginal cells being retained as liquid, where
`beta = alpha / max(rho(phi), eps_rho)` is largest and the cells are most compliant.

---

## 4. Recommendation

`spacetime_enabled` defaults to **off**, and on this evidence it should stay off:
today ST costs more the larger the simulation gets, and no wall-time budget tested
lets it buy resolution.

Ranked follow-ups, if ST is to be made to pay for itself:

1. **Make the ST pressure operator graph-capturable** (precompute per-face beta once
   per solve rather than on the fly). Targets 64% of the overhead. Cost work.
2. **Cell-centred phase sampling** per Sec. 2 above. Targets the +17% bias. Quality
   work, and the prerequisite for ST being a quality win rather than a cost.
3. **Deterministic P2G reduction ordering.** Not ST-specific, but without it no A/B
   below ~1 pp in this repo can be trusted, including the two items above.

## 5. Known smaller issues

- `sim/validation.py` sets `spacetime_phase_threshold = 0.2` in the ST validation
  test, while the shipped default is 0.5 — the test does not exercise the default
  configuration.
- `max_substeps` has a UI hard maximum of 32 while headless runs use up to 256, so
  fine resolutions cannot reach their required substep count from the UI.

## 6. Reproducing

```bash
blender --background --python tests/headless_spatiotemporal_validation.py -- --device cuda:0
```

A/B harness (`--set FIELD=VALUE` overrides any `SolverSettings` field, repeatable):

```bash
python tests/headless_equilibrium_ab.py --device cuda:0 --spacetime --cfl 1 --voxel-size 0.025 --seconds 10 --output bench/run.json
```

Per-phase attribution:

```bash
python tools/profile_gpu_solver.py --device cuda:0 --sync --json-out bench/phases.json
```

Raw result JSON is written to `bench/`, which is gitignored — it is regenerable and
runs to several MB.
