# Refresh policy drift report authority

## State

completed

## What to build

Keep the root policy-drift report bound to this checkout and to the most recent
lane-specific production evidence. The prior report named a temporary audit
root, an old commit, and an old demo artifact while the live checkout had
moved on.

## Acceptance criteria

- [x] Report frontmatter names the real root audit unit and current git HEAD.
- [x] Report records the current full-gate timestamp and artifact provenance.
- [x] Report records unresolved evidence drift as local backlog tasks instead
      of presenting stale metrics as current proof.
- [x] `python tools/validate_policy_drift_reports.py` is rerun after report
      refresh and its result is recorded.

## Blocked by

None - completed during the 2026-08-29 audit
