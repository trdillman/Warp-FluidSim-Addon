# FluidSim Warp — Demo Scene Analysis

Verdict: **PASS** (18/18 checks)

## dam_break

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 128,000..128,000 over 72 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 45 mm) |
| domain containment | PASS | max outside 1.6 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 6.00 m/s (limit 13.724139317275965) |
| front advances monotonically | PASS | min frame-to-frame delta 58.6 mm |
| front speed vs shallow-water theory | PASS | 3.48 m/s = 1.01*sqrt(gH) (accept 0.6-2.2) |
| Martin-Moses front position at T=2 | PASS | Z(T=2) = 2.59 (experimental 2.5-4.0) |
| column collapses | PASS | left-wall height 0.08 m from 1.20 m |
| whitewater emitted by turbulent bore | PASS | peak whitewater 3,736 |

## crown_splash

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 259,112..259,112 over 48 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 15 mm) |
| domain containment | PASS | max outside 0.0 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 7.17 m/s (limit 10.5) |
| crown forms after impact | PASS | peak 10,841 crown particles at frame 4 (impact ~frame 1) |
| crown rises above pool | PASS | max crown height 219 mm above pool |
| crown radius grows ~sqrt(t) | PASS | corr(R, sqrt(t)) = 0.972 over 5 expansion frames |
| Worthington jet rebounds at center | PASS | peak 512 jet particles, height 176 mm |
| crown azimuthal symmetry | PASS | sector-radius CV = 0.007 over 12/12 sectors |

## Evidence provenance

### Generated source evidence

These 18 checks and metrics are synchronized from
`artifacts/demo_analysis/analysis_report.json`, produced by the real GUI-inclusive production
gate at `2026-08-07T20:08:29.062185+00:00` under HEAD `ffd642e5edaf5e3fa8b2b0deff08d9e7f29a60d6`. Exact
promoted metrics are `front_speed_over_c0=1.0133832205`,
`column_height_end=0.0848999023`,
`crown_sqrt_t_corr=0.9715338226`, and
`max_crown_height=0.2191748172`. Artifact SHA-256 is
`f132a9d50033db9d2068c61dceb645b5a2b8b8369b697ed8b516ec9bda38497a`. These bounded run values are evidence, not arbitrary-scale physical
claims.

### Review evidence

`GUI-ACCEPTANCE-PASS (38/38)` and `SAVED-DEMO-GUI-PLAYBACK-PASS` prove real
interactive execution separately; review evidence does not replace generated
source metrics.
