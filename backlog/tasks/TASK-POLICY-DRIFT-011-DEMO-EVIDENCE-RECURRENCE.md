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
- The selected generated source artifact reports dam-break front ratio
  `1.0117893136`, column height `0.0798046589 m`, crown height `0.3577636480 m`,
  and `DEMO-ANALYSIS-PASS (18/18)`.
- The technical report now records that artifact, the gate run
  `2026-07-27T16:39:08Z`, HEAD `77a4931fb6d16f749628141b5f25aa33017cde5f`, the
  bounded scene/frame profiles, and separate GUI review evidence.

## Scope and relationship

`demos/backlog/tasks/TASK-POLICY-DRIFT-005-CHECKED-IN-EVIDENCE-RECURRENCE.md`
owns the checked-in demo JSON and Markdown. This card owns only the dependent
technical-report integration and must follow that card.

## Depends on

`C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port\demos\backlog\tasks\TASK-POLICY-DRIFT-005-CHECKED-IN-EVIDENCE-RECURRENCE.md`

## Blocked by

`demos/backlog/tasks/TASK-POLICY-DRIFT-005-CHECKED-IN-EVIDENCE-RECURRENCE.md`
completed first and established the synchronized source artifact. No human
decision is required.
