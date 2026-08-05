# FluidSim Warp — Demo Scene Analysis

Verdict: **PASS** (18/18 checks)

## dam_break

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 128,000..128,000 over 72 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 45 mm) |
| domain containment | PASS | max outside 1.6 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 5.77 m/s (limit 13.724139317275965) |
| front advances monotonically | PASS | min frame-to-frame delta 57.1 mm |
| front speed vs shallow-water theory | PASS | 3.47 m/s = 1.01*sqrt(gH) (accept 0.6-2.2) |
| Martin-Moses front position at T=2 | PASS | Z(T=2) = 2.59 (experimental 2.5-4.0) |
| column collapses | PASS | left-wall height 0.09 m from 1.20 m |
| whitewater emitted by turbulent bore | PASS | peak whitewater 3,938 |

## crown_splash

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 259,112..259,112 over 48 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 15 mm) |
| domain containment | PASS | max outside 0.0 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 8.13 m/s (limit 10.5) |
| crown forms after impact | PASS | peak 10,838 crown particles at frame 4 (impact ~frame 1) |
| crown rises above pool | PASS | max crown height 285 mm above pool |
| crown radius grows ~sqrt(t) | PASS | corr(R, sqrt(t)) = 0.972 over 5 expansion frames |
| Worthington jet rebounds at center | PASS | peak 537 jet particles, height 164 mm |
| crown azimuthal symmetry | PASS | sector-radius CV = 0.007 over 12/12 sectors |

## Evidence provenance

### Generated source evidence

These 18 checks and metrics are synchronized from
`artifacts/demo_analysis/analysis_report.json`, produced by the real GUI-inclusive production
gate at `2026-08-05T15:23:58.988102+00:00` under HEAD `afbdf0e0f902a3e6205420aca3e31b04defe9749`. Exact
promoted metrics are `front_speed_over_c0=1.0113339117`,
`column_height_end=0.0854480267`,
`crown_sqrt_t_corr=0.9715337333`, and
`max_crown_height=0.2854394317`. Artifact SHA-256 is
`d1a0cd12233a3a7c92ecac177fa9b3c1c1f3514dcc1ea464c798c6d153310242`. These bounded run values are evidence, not arbitrary-scale physical
claims.

### Review evidence

`GUI-ACCEPTANCE-PASS (38/38)` and `SAVED-DEMO-GUI-PLAYBACK-PASS` prove real
interactive execution separately; review evidence does not replace generated
source metrics.
