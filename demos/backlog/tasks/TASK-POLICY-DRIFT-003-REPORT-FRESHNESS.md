# TASK-POLICY-DRIFT-003: Refresh stale nested policy report authority

## State

completed

## What to build

Keep the nested demo policy report's frontmatter, checked-in evidence claims,
and machine-readable footer bound to the same current parent runtime rerun and
git HEAD as the audit.

## Acceptance criteria

- [x] The nested report records the current UTC generation time, audit-unit root,
  and git HEAD.
- [x] Current analyzer output, checked-in demo evidence, and saved-demo playback
  are cited separately from parent-only validation results.
- [x] Stale prior-run metadata is identified as documentation/validation drift,
  not promoted as current demo evidence.

## Evidence

- Historical audit evidence: the prior nested report was bound to
  `7fc3f52298dfe7a260c3950a7e57af44296b64ce` and had no active backlog tasks.
- Live checkout evidence: `git rev-parse HEAD` is
  `a6965e679ac3fd276c0c81cbfc1ee67257585739`; the refreshed nested report now
  binds frontmatter and footer to that HEAD and the current UTC report run.
- Current nested evidence: the analyzer and synchronized checked-in JSON/Markdown
  both report 18/18; `artifacts/gui_playback_probe.json` reports live CUDA
  playback at 600,000 particles, solver frame 5, and `passed=true`.

## Blocked by

None - the parent analyzer and saved-demo playback were rerun, checked-in demo
evidence was synchronized, and the nested report binding was refreshed.
