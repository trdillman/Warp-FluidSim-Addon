# FluidSim Warp — Demo Scene Analysis

Verdict: **PASS** (18/18 checks)

## dam_break

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 128,000..128,000 over 72 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 45 mm) |
| domain containment | PASS | max outside 1.6 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 5.54 m/s (limit 13.724139317275965) |
| front advances monotonically | PASS | min frame-to-frame delta 68.4 mm |
| front speed vs shallow-water theory | PASS | 3.35 m/s = 0.98*sqrt(gH) (accept 0.6-2.2) |
| Martin-Moses front position at T=2 | PASS | Z(T=2) = 2.59 (experimental 2.5-4.0) |
| column collapses | PASS | left-wall height 0.10 m from 1.20 m |
| whitewater emitted by turbulent bore | PASS | peak whitewater 4,280 |

## crown_splash

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 259,112..259,112 over 48 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 15 mm) |
| domain containment | PASS | max outside 0.0 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 7.48 m/s (limit 10.5) |
| crown forms after impact | PASS | peak 10,840 crown particles at frame 4 (impact ~frame 1) |
| crown rises above pool | PASS | max crown height 408 mm above pool |
| crown radius grows ~sqrt(t) | PASS | corr(R, sqrt(t)) = 0.972 over 5 expansion frames |
| Worthington jet rebounds at center | PASS | peak 537 jet particles, height 177 mm |
| crown azimuthal symmetry | PASS | sector-radius CV = 0.007 over 12/12 sectors |

## Evidence provenance

### Generated source evidence

These 18 checks and metrics are synchronized from the current generated
source artifact at `artifacts/demo_analysis/analysis_report.json`, produced by
the real `python tools/run_production_gate.py --include-gui` run. The artifact
was written at `2026-07-26T16:47:25Z`; the full production gate recorded that
run at `2026-07-26T16:45:03Z` under HEAD
`566a4276a4f3e7ba599519091a8a5c20b108da4b`.
The bounded source profiles are the checked-in `demos/dam_break/meta.json`
and `demos/crown_splash/meta.json`: 128,000 particles over 72 frames for
`dam_break`, and 259,112 particles over 48 frames for `crown_splash`.
The delta from the prior checked-in values is classified as run variance under
the same checked-in source profiles; this rerun still passes all 18 bounded
checks.

### Review evidence

Runtime review evidence is separate from the generated metric table:
`artifacts/gui_playback_probe.json` records live CUDA playback on `cuda:0`
through solver frame 5 with 600,000 particles, a changed viewport, and
`passed=true`; the full production lane also reports
`FULL-PRODUCTION-GATE-PASS`. Review evidence validates execution and playback
but does not replace the generated source metrics.
