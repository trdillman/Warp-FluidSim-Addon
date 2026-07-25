# TASK-POLICY-DRIFT-009: Refresh stale root policy report authority

## State

completed

## What to build

Keep the root policy-drift report's frontmatter, evidence, and machine-readable
footer bound to the same current runtime rerun and git HEAD as the audit.

## Acceptance criteria

- [x] The report records the current UTC generation time, audit-unit root, and
  git HEAD.
- [x] The report cites the current full production lane, host regression suite,
  demo analyzer, and saved-demo playback artifact.
- [x] Stale prior-run metadata is identified as documentation/validation drift
  rather than treated as current runtime evidence.

## Evidence

- Historical audit evidence: the prior report was bound to
  `7fc3f52298dfe7a260c3950a7e57af44296b64ce` and had no active backlog tasks.
- Live checkout evidence: `git rev-parse HEAD` is
  `a6965e679ac3fd276c0c81cbfc1ee67257585739`; the refreshed report now binds
  both frontmatter and footer to that HEAD and the current UTC report run.
- Current evidence: `python tools/run_production_gate.py --include-gui` passed
  at `2026-07-25T16:37:34Z`; host pytest passed 256 tests; the analyzer passed
  18/18; and `artifacts/gui_playback_probe.json` reports live CUDA playback at
  600,000 particles, solver frame 5, and `passed=true`.

## Blocked by

None - the full production lane, host regression suite, demo analyzer, and
saved-demo playback were rerun and the report binding was refreshed.
