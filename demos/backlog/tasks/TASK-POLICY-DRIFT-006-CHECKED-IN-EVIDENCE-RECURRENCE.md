# TASK-POLICY-DRIFT-006: Resynchronize checked-in demo evidence

## State

completed

## What to build

Update the demos audit unit's checked-in JSON, Markdown, and technical report to
the same generated run and provenance used by the parent production gate.

## Classification

validation drift and documentation drift; medium blocking criticality

## Acceptance criteria

- [x] `demos/analysis_report.json` matches the selected current generated artifact.
- [x] `demos/ANALYSIS.md` and `fluidSim_warp_blender51/docs/technical_report.md` use the same metrics.
- [x] Provenance identifies this unit, the source artifact, and current HEAD.
- [x] After the GUI blocker is repaired, the parent gate and report validator pass for the synchronized evidence.

## Evidence

- `demos/analysis_report.json` is byte-for-byte synchronized with
  `artifacts/demo_analysis/analysis_report.json` from the final full-gate run;
  both report the same `18/18` result and SHA-256.
- `demos/ANALYSIS.md` and `fluidSim_warp_blender51/docs/technical_report.md`
  contain the same exact promoted metric tokens and run provenance.
- `rtk python tools/validate_policy_drift_reports.py` passed after the GUI
  blocker repair and synchronized evidence refresh.

## Blocked by

None - dependency met; acceptance criteria met; no remaining blocker
