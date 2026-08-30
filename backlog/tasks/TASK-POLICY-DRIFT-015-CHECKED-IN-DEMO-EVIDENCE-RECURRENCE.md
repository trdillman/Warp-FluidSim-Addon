# Synchronize checked-in demo evidence

## State

ready-for-agent

## What to build

Make the checked-in demo report and its human-readable analysis reflect the
current generated result from the lane-specific production gate. The current
gate artifact passes, but the checked-in JSON, Markdown, and artifact hash
still describe an older bake.

## Acceptance criteria

- [ ] Run `python tools/run_production_gate.py --include-gui` at a stable
      checkout and use its generated demo artifact as the only authority.
- [ ] Update `demos/analysis_report.json` and `demos/ANALYSIS.md` together,
      including all promoted metrics and the exact artifact SHA-256.
- [ ] Verify no stale metric or hash remains in the root or nested technical
      report surfaces.
- [ ] Run `python tools/validate_policy_drift_reports.py` and record a pass
      after the synchronized evidence is in place.

## Blocked by

None - can start immediately
