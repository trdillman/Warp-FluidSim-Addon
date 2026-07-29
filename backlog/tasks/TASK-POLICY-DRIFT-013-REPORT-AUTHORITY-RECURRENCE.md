# TASK-POLICY-DRIFT-013: Enforce current report authority

## State

completed

## What to build

Make the policy report and its promoted demo claims derive from the same current
full-lane run, HEAD, generated artifacts, and active local cards so a completed
report-freshness task cannot silently become stale on the next checkout.

## Classification

documentation drift and validation drift; medium blocking criticality

## Acceptance criteria

- [x] The report HEAD and generated timestamp match the audited checkout and run.
- [x] A failed authoritative gate cannot be reported as a passing full production lane.
- [x] Root and demos reports point to the same current generated demo artifact and provenance.
- [x] The workflow fails closed when report HEAD, gate verdict, or artifact provenance is stale or misbound.

## Evidence

- `artifacts/production_gate_manifest.json` records the successful GUI-inclusive
  run `2026-07-28T17:04:58.639694+00:00`, full HEAD
  `f9c7b366baaa31d171ff3b6773f39eb0125d110c`, verdict, and demo-artifact hash.
- `tools/validate_policy_drift_reports.py` validates both report frontmatter and
  footers, artifact/checked-in JSON equality, promoted metric tokens, current
  HEAD, gate verdict, and active card states; missing or stale manifests fail
  closed.
- `rtk python tools/validate_policy_drift_reports.py` passed
  `POLICY-DRIFT-REPORT-VALIDATION-PASS` after the synchronized report/card
  refresh.

## Blocked by

None - acceptance criteria met; no remaining blocker
