# Paddle spin-up acceptance calibration

Measured 2026-09-22 at HEAD `f383eae` (Blender 5.1, CUDA). Supersedes the
`UNCALIBRATED` placeholder `PADDLE_SPINUP_MOMENTUM_MARGIN = 0.3` that shipped
in `tools/demos/analyze_demos.py` with no bake behind it.

## What the check measures

Per frame, over all particles, about the tank axis:

```
Lz = sum(rx * vy - ry * vx)
```

a bulk sum with uniform particle mass omitted. Three windows are averaged:
`early` (frames 1-5), `steady` (the meta-declared `spinup_end_frame ..
steady_end_frame`), and `late` (the last 5 frames). Acceptance is now three
clauses, of which the first is new:

```python
driven   = abs(steady_mean_lz / (N * |omega| * r_out**2)) >= 0.005
spins_up = (steady_mean_lz - early_mean_lz) > 0.30 * abs(steady_mean_lz)
decays   = late_mean_lz < steady_mean_lz
```

`driven` nondimensionalises the steady circulation against the solid-body
circulation the paddle *could* impose, so one floor means the same thing at
every resolution, particle count and paddle speed. When `omega` is zero there
is no such scale at all, which is itself a failure rather than a division to
attempt.

## Method

21 real bakes (~28 s each), one per row, built through
`tools/demos/build_demo_scenes.py` with the calibration envelope it now reads from
`FLUIDSIM_PADDLE_*` environment variables. Defaults are byte-preserving: with
no variant set the shipped `paddle_stir.blend` is produced unchanged, and a
variant never writes over it.

```bash
FLUIDSIM_PADDLE_VARIANT=voxel_fine \
FLUIDSIM_PADDLE_VOXEL=0.010 \
FLUIDSIM_PADDLE_SEED_SPACING=0.010 \
FLUIDSIM_PADDLE_RPM=100 \
FLUIDSIM_PADDLE_FRAME_SCALE=1.0 \
blender --background --factory-startup --python tools/demos/build_demo_scenes.py -- paddle_stir
```

Four axes were swept independently and then combined at the worst corner:
resolution (voxel 0.008-0.016), particle count (19k-311k via seed spacing),
duration (48-144 frames), and paddle speed (0-100 RPM). Five repeats were
baked at the shipped settings to separate run-to-run noise from envelope
spread.

## Measurements

`R` is the spin-up ratio `(steady - early) / |steady|`; `Lhat` is the
dimensionless steady circulation; `ppv` is particles per voxel
`(voxel / seed_spacing)^3`; `decay` is `late / steady`.

| label | R | Lhat | ppv | particles | frames | voxel | rad/s | decay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| degraded_rpm0 | -2.0338 | n/a (omega=0) | 1.728 | 158,220 | 96 | 0.012 | 0.000 | 0.5019 |
| corner_low | 0.2457 | 0.00394 | 0.244 | 39,508 | 96 | 0.010 | 10.472 | 0.0285 |
| seed_xsparse | 0.2654 | 0.00450 | 0.216 | 19,149 | 96 | 0.012 | 10.472 | 0.0401 |
| seed_vsparse | 0.4217 | 0.00604 | 0.422 | 38,998 | 96 | 0.012 | 10.472 | 0.2796 |
| seed_sparse | 0.6103 | 0.01037 | 0.787 | 72,807 | 96 | 0.012 | 10.472 | 0.4669 |
| **voxel_fine** | **0.6202** | **0.01524** | 1.000 | 159,948 | 96 | 0.010 | 10.472 | 0.6002 |
| voxel_vfine | 0.7715 | 0.01352 | 0.512 | 162,972 | 96 | 0.008 | 10.472 | 0.7469 |
| degraded_rpm2 | 0.8365 | 0.02731 | 1.728 | 158,220 | 96 | 0.012 | 0.209 | 1.1595 |
| frames_short | 0.8821 | 0.19381 | 1.728 | 158,220 | 48 | 0.012 | 10.472 | 0.7867 |
| rpm_10 | 0.8991 | 0.07654 | 1.728 | 158,220 | 96 | 0.012 | 1.047 | 1.0714 |
| rpm_25 | 0.9263 | 0.11345 | 1.728 | 158,220 | 96 | 0.012 | 2.618 | 0.8308 |
| base_r2 | 0.9432 | 0.17076 | 1.728 | 158,220 | 96 | 0.012 | 10.472 | 0.7628 |
| base_r1 | 0.9433 | 0.17106 | 1.728 | 158,220 | 96 | 0.012 | 10.472 | 0.7708 |
| base_r5 | 0.9434 | 0.17160 | 1.728 | 158,220 | 96 | 0.012 | 10.472 | 0.7731 |
| base_r3 | 0.9436 | 0.17192 | 1.728 | 158,220 | 96 | 0.012 | 10.472 | 0.7649 |
| base_r4 | 0.9439 | 0.17299 | 1.728 | 158,220 | 96 | 0.012 | 10.472 | 0.7675 |
| rpm_50 | 0.9448 | 0.15078 | 1.728 | 158,220 | 96 | 0.012 | 5.236 | 0.8331 |
| frames_long | 0.9639 | 0.16213 | 1.728 | 158,220 | 144 | 0.012 | 10.472 | 0.6223 |
| seed_dense | 0.9679 | 0.31528 | 3.375 | 311,304 | 96 | 0.012 | 10.472 | 0.6161 |
| voxel_coarse | 0.9708 | 0.33446 | 4.096 | 154,224 | 96 | 0.016 | 10.472 | 0.4915 |
| degraded_rpm05 | 19.5890 | 0.00048 | 1.728 | 158,220 | 96 | 0.012 | 0.052 | -26.6108 |

## The supported envelope

Two bounds, both measured. Outside either one a rejection is correct rather
than a false negative, so neither is a threshold to be relaxed.

**At least one seed particle per voxel** (`seed spacing <= voxel size`). `R`
degrades monotonically with under-seeding: 0.943 at ppv 1.728, 0.620 at
ppv 1.000, 0.610 at 0.787, 0.422, 0.265, 0.246. Below ppv 1 FLIP has too few
markers to carry momentum and the pool genuinely does not spin up.

**Paddle speed >= 25 RPM for a 96-frame window.** `rpm_10` is *not* rejected
by either threshold here - its `R` is 0.899 and its `Lhat` is 0.0765, both
comfortably inside. It is rejected by the pre-existing `decays` clause,
because at 10 RPM the pool is still accelerating when the window ends
(late 1962 against steady 1831, decay ratio 1.071). That is an honest reading
of a scene that never reached steady state, not a threshold miscalibration.
`rpm_25` is the lowest speed that equilibrates in time.

## Why two conditions, not one

The relative margin divides by `abs(steady_mean_lz)`. When the paddle barely
turns, that denominator is noise, and the ratio explodes rather than
collapsing. `degraded_rpm05` - 0.5 RPM, a scene with effectively no drive -
measures `R = 19.589`, so **the margin alone accepts it at any value below 19.6**.
No choice of a single constant separates it from a real bake. Its `Lhat` is
0.00048, three orders of magnitude below the shipped 0.171, so the absolute
floor closes the false accept cleanly.

This was a live defect in the shipped check, not a hypothetical: the
placeholder 0.3 passed `degraded_rpm05`.

## Derivation of the two constants

**`PADDLE_STEADY_CIRCULATION_FLOOR = 0.005`.** The worst supported case is
`voxel_fine` at `Lhat = 0.01524`; the worst *rejected* case is
`degraded_rpm05` at 0.00048. The floor sits ~3x below the former and ~10x
above the latter, near the geometric middle of a 32x gap.

**`PADDLE_SPINUP_MOMENTUM_MARGIN = 0.30`.** Over the supported envelope, `R`
runs 0.620 (`voxel_fine`) to 0.971 (`voxel_coarse`). The five shipped-settings
repeats give 0.9433, 0.9432, 0.9436, 0.9439, 0.9434 - a spread of 7e-4, three
orders of magnitude below the envelope spread. The envelope sets this number,
not the noise, and 0.30 is half the worst supported case.

The tight repeat spread is worth contrasting with the old `max_crown_height`.
At calibration time it swung ~40% across bakes of the same commit. `Lz` is a
bulk sum over ~158k particles, where the solver's order-dependent float
atomics averaged out. The old crown height was a tail extremum over ~200
particles of late jet spray, where they did not. A bulk statistic is the
reason this threshold can be calibrated at all.

The solver has since been made bitwise deterministic (`d2a6b6d`). The old
statistic still swung 20% under a 1e-6 relative change in drop speed, so it
was rebound to the crown phase only (see `tools/demos/analyze_demos.py`).

## Verification

`driven and spins_up and decays` was replayed through the shipped
`analyze_paddle_stir` over all 21 caches, monkeypatching only `load_scene`:

- every in-envelope case accepted (`base_r1..r5`, `frames_short`,
  `frames_long`, `rpm_25`, `rpm_50`, `seed_dense`, `voxel_coarse`,
  `voxel_fine`)
- all three degraded cases rejected, each by a different clause:
  `degraded_rpm0` by the omega=0 guard, `degraded_rpm05` by the new
  circulation floor (0.00048 against 0.005) and `degraded_rpm2` by the
  pre-existing `decays` clause (2 RPM does not equilibrate either, decay
  ratio 1.160). Only `degraded_rpm05` needed the new floor; it is also the
  only one the shipped placeholder wrongly accepted.
- out-of-envelope cases are not constrained either way

`base_r1..r5` - the shipped settings - score **10/10** on the full analyzer.
Regression coverage for the accept and reject sides is in
`tests/demos/test_demo_analysis_contract.py`.

## Recheck on master `b5e0050` (2026-09-24)

After the determinism (`d2a6b6d`) and launch-overhead (`1a31f15`) solver
changes, nine cases were re-baked (RTX 4080 SUPER, Blender 5.1) and replayed
through the shipped analyzer the same way. Both constants hold unchanged:

| label | R | Lhat | decay | verdict |
|---|---:|---:|---:|---|
| base_r1 / r2 / r3 | 0.9437 | 0.17241 | 0.7721 | accepted (10/10), bitwise identical repeats |
| rpm_25 | 0.9263 | 0.11346 | 0.8259 | accepted |
| voxel_fine | 0.6294 | 0.01563 | 0.5919 | accepted (worst supported case) |
| seed_sparse | 0.6134 | 0.01039 | 0.4645 | out of envelope, accepted |
| degraded_rpm0 | -0.0608 | 0.00000 | 1.2938 | rejected |
| degraded_rpm05 | 11.1051 | -0.00074 | 24.3107 | rejected (circulation floor) |
| degraded_rpm2 | 0.8115 | 0.02372 | 1.1701 | rejected (`decays`) |

`CALIBRATION-VERIFY-PASS`. The worst supported R is now 0.629 (was 0.620), so
0.30 stays at roughly half of it, and run-to-run spread at the shipped
settings is now exactly zero.
