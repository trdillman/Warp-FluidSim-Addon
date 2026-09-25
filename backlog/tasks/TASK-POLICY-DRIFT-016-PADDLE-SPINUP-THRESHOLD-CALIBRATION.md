# Calibrate paddle spin-up acceptance

## State

done (2026-09-22)

## What to build

Replace the moving-collider analyzer's explicitly uncalibrated fixed angular
momentum margin with a measured, documented acceptance envelope. The current
real 96-frame, 158,220-particle bake passes, but the source admits that the
threshold is a placeholder.

## Acceptance criteria

- [x] Measure repeated real `paddle_stir` bakes across the supported frame,
      resolution, and particle-count envelope.
- [x] Derive and document a threshold from those measurements rather than
      retaining an unexplained magic number.
- [x] Add a negative or degraded-motion case that proves the check rejects
      missing spin-up and still accepts valid scale variants.
- [x] Remove the `UNCALIBRATED` qualification and rerun
      `python tools/analyze_demos.py paddle_stir` with a recorded `10/10`
      result.

## Blocked by

None - can start immediately

## Disposition (2026-09-22)

Calibrated against 21 real bakes at HEAD `f383eae`, sweeping resolution
(voxel 0.008-0.016), particle count (19k-311k), duration (48-144 frames) and
paddle speed (0-100 RPM), with five repeats at the shipped settings.
Full table, derivation and verification: `docs/paddle_spinup_calibration.md`.

**The placeholder hid a real false accept.** The margin divides by the
steady-window mean, so a scene with effectively no drive divides by noise and
the ratio *explodes* rather than collapsing: the 0.5 RPM bake measures a
spin-up ratio of 19.589 and the shipped 0.3 accepted it. No value of a single
relative constant separates that case from a real bake, so the fix is two
conditions, not a retuned one:

- `PADDLE_STEADY_CIRCULATION_FLOOR = 0.005` (new) - steady Lz nondimensionalised
  by the solid-body circulation the paddle could impose, `N*|omega|*r_out^2`.
  Worst supported case 0.0152, shipped 0.171, degraded 0.00048.
- `PADDLE_SPINUP_MOMENTUM_MARGIN = 0.30` (was an unexplained 0.3, now measured)
  - the supported envelope runs 0.620 to 0.971; 0.30 is half the worst case.
  Five repeats spread by 7e-4, three orders of magnitude below the envelope,
  so the envelope sets the number and not the noise.

The supported envelope has two measured bounds, and outside either one the
rejection is correct rather than a false negative: at least one seed particle
per voxel, and paddle speed >= 25 RPM for a 96-frame window (below that the
pool has not equilibrated when the window ends, and the pre-existing `decays`
clause rejects it - `rpm_10` measures late 1962 against steady 1831).

Verification, replaying the shipped `analyze_paddle_stir` over all 21 caches
with only `load_scene` monkeypatched: every in-envelope case accepted, all
three degraded cases rejected, `CALIBRATION-VERIFY-PASS`. Regression coverage
in `tests/test_demo_analysis_contract.py` (4 new tests, 6/6 pass).

Recorded result on the shipped scene:

```
python tools/analyze_demos.py paddle_stir
[PASS] paddle_stir: angular momentum spins up and decays - mean Lz (mass
omitted, uniform particle mass): early(1-5)=2317.4351, steady=41384.3544,
late(last 5)=32091.8230; spin-up ratio 0.9440 (floor 0.3), steady circulation
0.17297 (floor 0.005)
DEMO-ANALYSIS-PASS (10/10)
```

Bakes, harness and raw results are archived under
`artifacts/paddle_calibration/` (gitignored).

## Recheck on master `b5e0050` (2026-09-24)

The solver changed after calibration (bitwise determinism in `d2a6b6d`,
launch-overhead cut in `1a31f15`), so nine of the 21 cases were re-baked on a
GPU machine (RTX 4080 SUPER, Blender 5.1) and replayed through the shipped
`analyze_paddle_stir`. The constants did not need to change.

| label | R | Lhat | verdict |
|---|---:|---:|---|
| base_r1, r2, r3 | 0.9437 (all three bitwise identical) | 0.1724 | accepted, 10/10 |
| rpm_25 | 0.9263 | 0.1135 | accepted |
| voxel_fine (worst supported) | 0.6294 | 0.0156 | accepted |
| seed_sparse (out of envelope) | 0.6134 | 0.0104 | accepted |
| degraded_rpm0 | -0.0608 | 0.0000 | rejected |
| degraded_rpm05 | 11.1051 | -0.0007 | rejected by the circulation floor |
| degraded_rpm2 | 0.8115 | 0.0237 | rejected by `decays` |

Result `CALIBRATION-VERIFY-PASS`. The worst supported R moved from 0.620 to
0.629, so 0.30 is still about half of it, and the repeat spread is now zero
rather than 7e-4. `degraded_rpm05` again shows why the margin alone is not
enough (R 11.1 would pass it); the floor still catches it. Raw results are in
`artifacts/paddle_calibration/recheck_b5e0050.*` (gitignored).

**Incidental finding, not addressed here:** rebuilding `paddle_stir` from
`tools/build_demo_scenes.py` produces a 280 KB `.blend` where the checked-in
one is 5.7 MB. The rebuilt scene analyses 10/10, but the shipped artifact is
evidently not what the builder currently emits. The checked-in `.blend` was
restored; this deserves its own card.
