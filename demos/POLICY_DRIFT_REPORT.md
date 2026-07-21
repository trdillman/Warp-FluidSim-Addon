---
status: completed
escalated: false
generated_at: 2026-07-13T00:00:00-04:00
audit_unit: demos
---

# Policy Drift Report

## Scope

This unit contains the retained `dam_break` and `crown_splash` demo assets and their checked-in analysis evidence. The executable lane lives in `tools/build_demo_scenes.py`, `tools/analyze_demos.py`, and `tools/check_demo_blends.py`.

## Strongest Current Evidence

- Checked-in analysis records 18/18 passing Dam Break and Crown Splash checks.
- `python tools/run_production_gate.py` rebuilds, analyzes, and checks both retained scenes.
- The GUI-inclusive gate validates saved Dam Break timeline playback.

## Findings

No demo-unit-specific policy drift is recorded for the two retained scenes.

## TASK-POLICY-DRIFT-007 — enforcement fixed; production acceptance blocked

The Martin-Moses dam-break front-position check now uses its reported
authoritative experimental range `2.5-4.0` for both enforcement and reporting.
The focused regression rejects the formerly contradictory `Z(T=2)=2.24` case.

On 2026-07-21, `python tools/analyze_demos.py` rejected the rebuilt Dam Break
cache at `Z(T=2)=2.24`. The requested
`python tools/run_production_gate.py --include-gui` likewise stopped at demo
analysis with `PRODUCTION-GATE-FAIL`; its GUI acceptance and saved-demo
playback lanes did not run after that fail-closed result. Bringing the demo
itself within `2.5-4.0` is a separate simulation/acceptance change, not a
reason to widen this check again.
