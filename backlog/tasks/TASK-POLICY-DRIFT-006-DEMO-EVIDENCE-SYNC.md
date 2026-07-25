# Synchronize technical demo evidence

## State

completed

## What to build

Regenerate the checked-in demo JSON/Markdown evidence and the dependent
technical-report wording from the same reproducible run as the parent full
production gate. Keep the parent report explicit about the difference between
current generated artifacts and checked-in review evidence.

## Acceptance criteria

- [x] `demos/analysis_report.json` matches the current generated analysis artifact.
- [x] `demos/ANALYSIS.md` reports the same metrics, source run, and 18/18 result.
- [x] `fluidSim_warp_blender51/docs/technical_report.md` uses the same metrics and run provenance.
- [x] The parent and nested policy reports cite the synchronized source artifact and remain bounded to its executed scale.

## Evidence

- Current full lane: `python tools/run_production_gate.py --include-gui` passed at
  `2026-07-25T16:37:34Z`, HEAD `a6965e679ac3fd276c0c81cbfc1ee67257585739`,
  with `FULL-PRODUCTION-GATE-PASS` and `DEMO-ANALYSIS-PASS (18/18)`.
- Current generated artifact: `artifacts/demo_analysis/analysis_report.json`
  has dam-break values `0.9758126`, `2.5895753`, and `0.0814825 m`.
- Checked-in `demos/analysis_report.json` and `demos/ANALYSIS.md` now match the
  generated artifact's metrics and 18/18 result; the technical report records
  the same source profiles and run provenance.
- Current `python -m pytest -q` passed `256` tests; synchronization corrected
  evidence/documentation drift without changing the solver lane.

## Blocked by

None - can start immediately
