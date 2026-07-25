# Synchronize checked-in demo analysis evidence

## State

completed

## What to build

Regenerate the nested checked-in JSON and Markdown analysis from the current
parent full-lane run, preserving the exact source profile, timestamp, metrics,
and bounded review scope.

## Acceptance criteria

- [x] `demos/analysis_report.json` matches the current generated analysis artifact.
- [x] `demos/ANALYSIS.md` reports the same values, provenance, and 18/18 result.
- [x] The nested report distinguishes current demo analysis from parent-only production-gate claims.
- [x] The parent technical report is updated to the same synchronized evidence.

## Evidence

- Current generated artifact: `artifacts/demo_analysis/analysis_report.json` is
  `pass`, `18/18`, with dam-break values `0.9758126`, `2.5895753`, and
  `0.0814825 m`.
- Checked-in `demos/analysis_report.json` and `demos/ANALYSIS.md` now report the
  same values, provenance, and 18/18 result.
- The nested report cites the local demo analysis separately from the parent
  full-gate claims, and the parent technical report uses the same source run.
- Parent full lane and independent pytest both pass; this is synchronized
  evidence, not evidence of a heuristic solver backend.

## Blocked by

None - can start immediately
