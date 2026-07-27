# Superseded: synchronize checked-in demo analysis evidence

## State

superseded

## Superseded by

This card is superseded by
`TASK-POLICY-DRIFT-005-CHECKED-IN-EVIDENCE-RECURRENCE.md` for the checked-in
demo JSON and Markdown. Its technical-report acceptance criterion belongs to
the root
`backlog/tasks/TASK-POLICY-DRIFT-011-DEMO-EVIDENCE-RECURRENCE.md` card. Do not
execute this card in parallel with those owners.

## What to build

Regenerate the nested checked-in JSON and Markdown analysis from the current
parent full-lane run, preserving the exact source profile, timestamp, metrics,
and bounded review scope.

## Acceptance criteria

- [ ] `demos/analysis_report.json` matches the current generated analysis artifact.
- [ ] `demos/ANALYSIS.md` reports the same values, provenance, and 18/18 result.
- [ ] The nested report distinguishes current demo analysis from parent-only production-gate claims.
- [ ] The parent technical report is updated to the same synchronized evidence.

## Evidence

- Current generated artifact: `artifacts/demo_analysis/analysis_report.json` is
  `pass`, `18/18`, with dam-break front ratio `1.0121877903`, column height
  `0.0852106214 m`, and crown height `0.2748437524 m`.
- Checked-in `demos/analysis_report.json` and `demos/ANALYSIS.md` still report
  the prior run's values, so this card is reopened.
- Parent full lane passes, but the checked-in evidence is not current proof.

## Blocked by

Superseded; use the narrower owners above.
