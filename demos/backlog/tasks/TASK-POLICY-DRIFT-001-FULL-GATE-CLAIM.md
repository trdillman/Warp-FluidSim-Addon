# Keep nested report aligned with the current full-gate claim

## State

completed

## What to build

Update the nested demo report so it reflects the current parent production-lane pass while keeping the standalone 18/18 demo analysis and the failing repository-wide pytest result separate.

## Scope

This is the completed nested report-hygiene slice only. It may record the parent production-lane, GUI-acceptance, and saved-demo-playback passes when the current parent evidence proves them; it does not claim a clean repository-wide pytest run.

## Acceptance criteria

- [x] The nested report records the current parent `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (35/35)`, and `SAVED-DEMO-GUI-PLAYBACK-PASS`, plus the separate repository-wide pytest failure.
- [x] The nested report keeps the standalone `DEMO-ANALYSIS-PASS (18/18)` distinct from the parent full-gate result and does not promote it into a clean repository-wide claim.
- [x] The report remains `fresh` and current at the parent run's git head.

## Evidence

- `demos/POLICY_DRIFT_REPORT.md:10-38` records the current nested verdict, parent full-lane pass, standalone demo pass, and separate pytest failure.
- `demos/POLICY_DRIFT_REPORT.md:42-57` records the remaining active evidence-sync finding.
- `demos/POLICY_DRIFT_REPORT.md:68-88` records the current footer, verification, and active card.

## Blocked by

None — this nested report-hygiene slice is complete. The remaining checked-in evidence work is tracked by `TASK-POLICY-DRIFT-002-CHECKED-IN-EVIDENCE.md`.
