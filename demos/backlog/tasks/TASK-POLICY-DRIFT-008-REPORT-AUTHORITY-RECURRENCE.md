# Refresh demos report authority

## State

completed

## What to build

Keep the demos-unit policy report bound to the real local demos directory,
current git HEAD, and the latest parent lane artifact instead of an old
temporary checkout.

## Acceptance criteria

- [x] Report frontmatter names the actual demos audit unit and current HEAD.
- [x] Report records the current gate timestamp, artifact hash, and dirty
      worktree caveat.
- [x] Any remaining checked-in metric mismatch is represented by the active
      local evidence task.

## Blocked by

None - completed during the 2026-08-29 audit
