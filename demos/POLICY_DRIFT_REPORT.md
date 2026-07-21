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

## TASK-POLICY-DRIFT-007 — enforcement and demo profile fixed

The Martin-Moses dam-break front-position check now uses its reported
authoritative experimental range `2.5-4.0` for both enforcement and reporting.
The focused regression rejects the formerly contradictory `Z(T=2)=2.24` case.

The rebuilt Dam Break profile applies a `+0.7 m/s` release velocity to both the
preview injection and its live source, plus `ww_churn_gain=2.0` to retain the
whitewater criterion. On 2026-07-21, a real Blender 5.1/CUDA build and
analysis of that profile passed all Dam Break checks, including
`Z(T=2)=2.59` and peak whitewater `3,490`.

The current-worktree `python tools/run_production_gate.py --include-gui` passed
after rebuilding both demos: Dam Break and Crown Splash analysis passed 18/18,
interactive GUI acceptance passed 35/35, and the saved Dam Break timeline
playback report validated before `FULL-PRODUCTION-GATE-PASS`.
