# Harden full GUI gate reproducibility

## State

ready-for-agent

## What to build

Investigate the first current full-gate GUI subprocess exit `4294967295`.
The same dirty checkout passed the focused GUI lane and a bounded full-gate
retry, so the issue is intermittent validation reliability rather than a
proven deterministic solver defect.

## Acceptance criteria

- [ ] Preserve a complete log and process/runtime context for any nonzero GUI
      exit, including cases where the GUI JSON verdict is not written.
- [ ] Reproduce the failure or establish a bounded repeated-pass envelope in
      fresh Blender processes without weakening the acceptance checks.
- [ ] Ensure the full production gate reports a clean exit only when all GUI,
      saved-demo playback, and artifact checks complete.
- [ ] Record the final disposition in the policy-drift report and rerun the
      full lane at a stable checkout.

## Blocked by

None - can start immediately
