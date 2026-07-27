# TASK-POLICY-DRIFT-005: Resynchronize nested checked-in evidence

## State

completed

## What to build

Own only the checked-in demo JSON and Markdown. Regenerate them from the
current parent analysis artifact and preserve that run's provenance; the
dependent technical-report wording belongs to the root integration card.

## Acceptance criteria

- [x] `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos\analysis_report.json`
  matches `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\artifacts\demo_analysis\analysis_report.json`
  for all metrics and the 18/18 result.
- [x] `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos\ANALYSIS.md`
  reports the same metrics, source run, and bounded profile.
- [x] The checked-in files explicitly distinguish generated source evidence
  from review evidence; technical-report wording is handled by the root
  integration card.

## Evidence

- Demos report: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos\POLICY_DRIFT_REPORT.md`.
- Selected generated source evidence reports dam-break front ratio
  `1.0117893136`, column height `0.0798046589 m`, and crown height
  `0.3577636480 m`; checked-in JSON and Markdown now match the artifact.
- The current gate's demo-analysis lane reports `DEMO-ANALYSIS-PASS (18/18)`
  under HEAD `77a4931fb6d16f749628141b5f25aa33017cde5f`.

## Scope and relationship

This card owns only `demos\analysis_report.json` and `demos\ANALYSIS.md`.
`backlog/tasks/TASK-POLICY-DRIFT-011-DEMO-EVIDENCE-RECURRENCE.md` owns the
dependent `fluidSim_warp_blender51\docs\technical_report.md` integration.

## Blocked by

None - the selected generated artifact, checked-in JSON, and Markdown now
match for all 18 checks and record the same bounded-run provenance. No human
decision is required.
