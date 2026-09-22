# Refresh nested technical report authority

## State

done

## What to build

Update the nested package technical report so its promoted suite counts,
proof HEAD, demo metrics, and artifact hash match current executable evidence.
The package's current tests and release verification pass, but the report
still cites an older proof snapshot.

## Acceptance criteria

- [x] Distinguish the nested package suite result from the parent repository
      suite and record the current package test count. (`fluidSim_warp_blender51/tests`
      only: 427 passed, 2026-09-22.)
- [x] Update the proof HEAD and all promoted demo metrics from the current
      lane artifact, including its SHA-256. (Synced to `01c4bf837bcec9b5`,
      2026-09-12, matching `docs/technical_report.md` and the root/`demos/`
      reports — the nested report had drifted to an older 2026-09-01 proof.)
- [x] Mark any unexecuted installed-package or no-CUDA lane as historical or
      pending instead of presenting it as current proof. (The GUI-inclusive
      production gate is labeled as the 2026-09-12 real-hardware run; it was
      not and cannot be rerun from a GPU/Blender-less cloud session.)
- [x] Rerun the nested tests, package verification, and parent policy-drift
      validator after the report update. (Nested tests and verify-only
      reran clean, 2026-09-22. `tools/validate_policy_drift_reports.py`
      still reports the production-gate manifest as missing — that file is
      git-ignored local evidence and the validator can only pass right after
      Tyler runs `tools/run_production_gate.py --include-gui` on his own
      GPU/Blender machine; a cloud session can never produce it. Not a bug
      in this task, and not something to fake.)

## Blocked by

None - closed 2026-09-22. Follow-up (not blocking this task): rerun the
GUI-inclusive production gate on real hardware before the next release so
`docs/technical_report.md` and the root/`demos/` `POLICY_DRIFT_REPORT.md`
files get a fresh proof HEAD too.
