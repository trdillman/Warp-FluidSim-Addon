# FluidSim Warp — Demo Scene Analysis

Verdict: **PASS** (18/18 checks)

## dam_break

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 128,000..128,000 over 72 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 45 mm) |
| domain containment | PASS | max outside 1.6 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 6.48 m/s (limit 13.724139317275965) |
| front advances monotonically | PASS | min frame-to-frame delta 57.1 mm |
| front speed vs shallow-water theory | PASS | 3.49 m/s = 1.02*sqrt(gH) (accept 0.6-2.2) |
| Martin-Moses front position at T=2 | PASS | Z(T=2) = 2.59 (experimental 2.5-4.0) |
| column collapses | PASS | left-wall height 0.08 m from 1.20 m |
| whitewater emitted by turbulent bore | PASS | peak whitewater 3,278 |

## crown_splash

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 259,112..259,112 over 48 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 15 mm) |
| domain containment | PASS | max outside 0.0 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 6.58 m/s (limit 10.5) |
| crown forms after impact | PASS | peak 10,840 crown particles at frame 4 (impact ~frame 1) |
| crown rises above pool | PASS | max crown height 225 mm above pool |
| crown radius grows ~sqrt(t) | PASS | corr(R, sqrt(t)) = 0.972 over 5 expansion frames |
| Worthington jet rebounds at center | PASS | peak 554 jet particles, height 183 mm |
| crown azimuthal symmetry | PASS | sector-radius CV = 0.007 over 12/12 sectors |

## Evidence provenance

### Generated source evidence

These 18 checks and metrics are synchronized from the selected generated
source artifact at `artifacts/demo_analysis/analysis_report.json`, produced by
the headless/demo portion of the real `python tools/run_production_gate.py
--include-gui` run. The gate run completed at `2026-08-11T00:35:06.917651+00:00`
under proof HEAD `aa40a3cd0c762503e886d52a9f1f427e0fb298e8`.
The bounded source profiles are the checked-in `demos/dam_break/meta.json`
and `demos/crown_splash/meta.json`: 128,000 particles over 72 frames for
`dam_break`, and 259,112 particles over 48 frames for `crown_splash`.
This selected run is bound to gate run `2026-08-11T00:35:06.917651+00:00`,
proof HEAD `aa40a3cd0c762503e886d52a9f1f427e0fb298e8`, and artifact SHA-256
`eb4c3f0e2a7104ec4da2129ad837052f76e9862a2d9f920f6deed121888c1ea4`. Exact
selected metrics are `front_speed_over_c0=1.0132693700`,
`column_height_end=0.0841247439`, `crown_sqrt_t_corr=0.9715353735`, and
`max_crown_height=0.2764160037`; the run passes all 18 bounded checks. Values
are bounded run evidence, not arbitrary-scale physical claims.

### Review evidence

Runtime review evidence is separate from the generated metric table:
`artifacts/gui_playback_probe.json` records live CUDA playback on `cuda:0`
through solver frame 6 with 128,000 particles, a changed viewport, and
`passed=true`; its validator reports `SAVED-DEMO-GUI-PLAYBACK-PASS`. The
interactive GUI lane also reports `GUI-ACCEPTANCE-PASS (38/38)`. Review
evidence validates execution and playback but does not replace the generated
source metrics.
