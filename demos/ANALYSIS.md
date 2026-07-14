# FluidSim Warp — Demo Scene Analysis

Verdict: **PASS** (18/18 checks)

## dam_break

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 128,000..128,000 over 72 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 45 mm) |
| domain containment | PASS | max outside 0.1 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 13.49 m/s (limit 13.724139317275965) |
| front advances monotonically | PASS | min frame-to-frame delta 61.0 mm |
| front speed vs shallow-water theory | PASS | 3.22 m/s = 0.94*sqrt(gH) (accept 0.6-2.2) |
| Martin-Moses front position at T=2 | PASS | Z(T=2) = 2.49 (experimental 2.5-4.0) |
| column collapses | PASS | left-wall height 0.23 m from 1.20 m |
| whitewater emitted by turbulent bore | PASS | peak whitewater 2,699 |

## crown_splash

| Check | Result | Detail |
|---|---|---|
| mass conservation | PASS | count 259,112..259,112 over 48 frames |
| no floor penetration | PASS | max below-floor 0.0 mm (limit 15 mm) |
| domain containment | PASS | max outside 0.0 mm |
| velocity bounded (p99.9) | PASS | p99.9 speed 6.01 m/s (limit 10.5) |
| crown forms after impact | PASS | peak 10,872 crown particles at frame 4 (impact ~frame 1) |
| crown rises above pool | PASS | max crown height 265 mm above pool |
| crown radius grows ~sqrt(t) | PASS | corr(R, sqrt(t)) = 0.971 over 5 expansion frames |
| Worthington jet rebounds at center | PASS | peak 531 jet particles, height 202 mm |
| crown azimuthal symmetry | PASS | sector-radius CV = 0.007 over 12/12 sectors |
