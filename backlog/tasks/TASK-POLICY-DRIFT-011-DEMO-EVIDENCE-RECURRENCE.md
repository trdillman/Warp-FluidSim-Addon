# TASK-POLICY-DRIFT-011: Integrate current demo evidence into technical report

## State

completed

## What to build

Update only the dependent technical-report metrics and provenance after the
demos evidence card establishes the current checked-in JSON and Markdown.

## Acceptance criteria

- [x] `fluidSim_warp_blender51/docs/technical_report.md` matches the current
  generated artifact for the documented metrics and 18/18 result.
- [x] The technical report records the source artifact, current run/HEAD, and
  bounded scene/frame profile without repeating stale checked-in values.
- [x] The technical report distinguishes current generated evidence from
  review evidence and does not call stale values synchronized.

## Evidence

- Root report: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\POLICY_DRIFT_REPORT.md`.
- Generated source: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\artifacts\demo_analysis\analysis_report.json`;
  final dam-break column height is `0.0952136219 m` and crown height is
  `0.4082177579 m`, with `DEMO-ANALYSIS-PASS (18/18)`.
- The technical report records the full-gate run at `2026-07-26T16:45:03Z`,
  HEAD `566a4276a4f3e7ba599519091a8a5c20b108da4b`, the 72/48-frame bounded
  profiles, and separate GUI review evidence. The prior metric delta is
  documented as run variance; stale values are not called synchronized.

## Scope and relationship

`demos/backlog/tasks/TASK-POLICY-DRIFT-005-CHECKED-IN-EVIDENCE-RECURRENCE.md`
owns the checked-in demo JSON and Markdown. This card owns only the dependent
technical-report integration and must follow that card.

## Depends on

`C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos\backlog\tasks\TASK-POLICY-DRIFT-005-CHECKED-IN-EVIDENCE-RECURRENCE.md`

## Blocked by

None - the demos evidence card is complete, the technical report matches the
final generated artifact, and the full production gate plus pytest pass.
