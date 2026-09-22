# Harden full GUI gate reproducibility

## State

done (2026-09-22)

## What to build

Investigate the first current full-gate GUI subprocess exit `4294967295`.
The same dirty checkout passed the focused GUI lane and a bounded full-gate
retry, so the issue is intermittent validation reliability rather than a
proven deterministic solver defect.

## Acceptance criteria

- [x] Preserve a complete log and process/runtime context for any nonzero GUI
      exit, including cases where the GUI JSON verdict is not written.
- [x] Reproduce the failure or establish a bounded repeated-pass envelope in
      fresh Blender processes without weakening the acceptance checks.
- [x] Ensure the full production gate reports a clean exit only when all GUI,
      saved-demo playback, and artifact checks complete.
- [x] Record the final disposition in the policy-drift report and rerun the
      full lane at a stable checkout.

## Blocked by

None - can start immediately

## Disposition (2026-09-22)

Three consecutive `tools/run_production_gate.py --include-gui` runs at HEAD
`f383eae` on an unmodified tree all reported `FULL-PRODUCTION-GATE-PASS`.
Evidence archived as `artifacts/gate_repro/gate_run{1,2,3}.log` and
`artifacts/gate_repro/manifest_run{1,2,3}.json`. Each log is exactly 1916
lines with the same 29 `[PASS]` markers in the same order and zero `[FAIL]`;
of the 686 differing lines, 671 are numeric-only on an identical text skeleton
and the 15 structural diffs are per-run temp directory names, `bpy_struct`
pointer addresses, Python set iteration order, timestamps and diagnostic
hashes. Recorded in `POLICY_DRIFT_REPORT.md` under PD-ROOT-002.

The original exit `4294967295` was a process fault, not a gate defect. Two
launch hazards are now known and must be avoided:

1. Launching non-detached: the run hangs before reaching the GUI lane.
2. Launching a second run while another is alive: the two contend on the
   shared Warp kernel cache and `__pycache__` directories and hang within
   seconds.

Correct procedure: one run at a time, detached, `python -u`, log written to a
durable path outside any transient directory.

Note for future verification: the manifest's `demo_artifact_sha256` changes on
every run (`c0888aca…`, `b3a1cf8c…`, `a0b2348c…`) because the bake is not
bitwise deterministic (PD-ROOT-004). Compare `gate_verdict` and the marker
sequence; never compare that hash.

