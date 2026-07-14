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
