# TASK-POLICY-DRIFT-005: Resynchronize nested checked-in evidence

## State

completed

## What to build

Own only the checked-in demo JSON and Markdown. Regenerate them from the
current parent analysis artifact after the metric delta is classified; the
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
- Final generated source dam-break column height is `0.0952136219 m` and
  crown height is `0.4082177579 m`; checked-in JSON and Markdown match the
  generated artifact exactly.
- The full gate rerun under the same checked-in profiles changed the derived
  values between runs while preserving `DEMO-ANALYSIS-PASS (18/18)`, so the
  earlier delta is classified as runtime variance rather than a source-profile
  change.

## Scope and relationship

This card owns only `demos\analysis_report.json` and `demos\ANALYSIS.md`.
`backlog/tasks/TASK-POLICY-DRIFT-011-DEMO-EVIDENCE-RECURRENCE.md` owns the
dependent `fluidSim_warp_blender51\docs\technical_report.md` integration.

## Blocked by

None - the delta was classified as run variance, the checked-in
JSON now exactly matches the generated artifact, and the Markdown records the
bounded source and separate review evidence.
