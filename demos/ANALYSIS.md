# FluidSim Warp — Demo Scene Analysis

Verdict: **PASS** (18/18 checks)

## dam_break

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 128,000..128,000 over 72 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 45 mm) |
| domain containment | PASS | max outside 1.6 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 5.79 m/s (limit 13.724139317275965) |
| front advances monotonically | PASS | min frame-to-frame delta 56.6 mm |
| front speed vs shallow-water theory | PASS | 3.47 m/s = 1.01*sqrt(gH) (accept 0.6-2.2) |
| Martin-Moses front position at T=2 | PASS | Z(T=2) = 2.59 (experimental 2.5-4.0) |
| column collapses | PASS | left-wall height 0.07 m from 1.20 m |
| whitewater emitted by turbulent bore | PASS | peak whitewater 3,233 |

## crown_splash

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 259,112..259,112 over 48 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 15 mm) |
| domain containment | PASS | max outside 0.0 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 7.07 m/s (limit 10.5) |
| crown forms after impact | PASS | peak 10,842 crown particles at frame 4 (impact ~frame 1) |
| crown rises above pool | PASS | max crown height 242 mm above pool |
| crown radius grows ~sqrt(t) | PASS | corr(R, sqrt(t)) = 0.972 over 5 expansion frames |
| Worthington jet rebounds at center | PASS | peak 563 jet particles, height 203 mm |
| crown azimuthal symmetry | PASS | sector-radius CV = 0.007 over 12/12 sectors |

## Evidence provenance

### Generated source evidence

These 18 checks and metrics are synchronized from the selected generated
source artifact at `artifacts/demo_analysis/analysis_report.json`, produced by
the headless/demo portion of the real `python tools/run_production_gate.py
--include-gui` run. The gate run started at `2026-08-05T06:49:33Z` under HEAD
`3bf6da6aa6f1ab043cb2dd3a125ad9b416fa9d5a`.
The bounded source profiles are the checked-in `demos/dam_break/meta.json`
and `demos/crown_splash/meta.json`: 128,000 particles over 72 frames for
`dam_break`, and 259,112 particles over 48 frames for `crown_splash`.
This selected run is bound to gate run `2026-08-05T06:49:33.968820+00:00`,
HEAD `3bf6da6aa6f1ab043cb2dd3a125ad9b416fa9d5a`, and artifact SHA-256
`65ec3d69463d150c0a19df231e2f33697543fd1e2d273aad451fdfd0c57a472a`. Exact
selected metrics are `front_speed_over_c0=1.0113339117`,
`column_height_end=0.0743652880`, `crown_sqrt_t_corr=0.9715359144`, and
`max_crown_height=0.2423486263`; the run passes all 18 bounded checks. Values
are bounded run evidence, not arbitrary-scale physical claims.

### Review evidence

Runtime review evidence is separate from the generated metric table:
`artifacts/gui_playback_probe.json` records live CUDA playback on `cuda:0`
through solver frame 7 with 128,000 particles, a changed viewport, and
`passed=true`; its validator reports `SAVED-DEMO-GUI-PLAYBACK-PASS`. The
interactive GUI lane also reports `GUI-ACCEPTANCE-PASS (38/38)`. Review
evidence validates execution and playback but does not replace the generated
source metrics.
