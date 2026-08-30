# Refresh nested technical report authority

## State

ready-for-agent

## What to build

Update the nested package technical report so its promoted suite counts,
proof HEAD, demo metrics, and artifact hash match current executable evidence.
The package's current tests and release verification pass, but the report
still cites an older proof snapshot.

## Acceptance criteria

- [ ] Distinguish the nested package suite result from the parent repository
      suite and record the current package test count.
- [ ] Update the proof HEAD and all promoted demo metrics from the current
      lane artifact, including its SHA-256.
- [ ] Mark any unexecuted installed-package or no-CUDA lane as historical or
      pending instead of presenting it as current proof.
- [ ] Rerun the nested tests, package verification, and parent policy-drift
      validator after the report update.

## Blocked by

None - can start immediately
