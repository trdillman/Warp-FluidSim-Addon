# Refresh nested technical report authority

## State

done

## What to build

Update the nested package technical report so its promoted suite counts,
proof HEAD, demo metrics, and artifact hash match current executable evidence.
The package's current tests and release verification pass, but the report
still cites an older proof snapshot.

## History

A 2026-09-22 cloud-session pass first marked this card `done` without
pulling `master`, which already carried a more thorough 2026-09-21T03:36
audit (Tyler + another Claude session) showing the card needed to stay open
on three separate findings (PD-PKG-001/002/003). That audit's version was
taken as authoritative and the card reopened. Tyler then ran the real
GUI-inclusive production gate for real (commit `f383eae`, 2026-09-22),
which resolved PD-PKG-001 (proof HEAD, metrics, and the `demos/
analysis_report.json` mirror). A same-day GPU-less cloud session then
resolved the remaining two, which needed no GPU/Blender hardware:

## Acceptance criteria

- [x] **PD-PKG-001**: proof HEAD, artifact hash, and the four promoted
      metric tokens synced to the passing gate; `demos/analysis_report.json`
      mirror committed alongside. (Resolved in `f383eae`, Tyler's machine,
      2026-09-22.)
- [x] **PD-PKG-002**: the promoted `| Full Python suite | **1,043 passed** |`
      row is now explicitly labeled unattributed historical data (it can't be
      retroactively attributed), and a "Current attributed Python suite
      counts" subsection gives two current, reproducible, distinctly-scoped
      counts. (Resolved in a GPU-less cloud session, 2026-09-22, HEAD
      `df1868b`; refreshed 2026-09-24 for master's `21d4d37` determinism fix
      and demo cleanup, which changed the test count to 426/1,440+116 skipped.)
- [x] **PD-PKG-003**: the `Installed headless no-CUDA exit` row now opens
      with `**PENDING — not current proof.**` and states it predates the
      final 1.0.255 artifact. The rerun itself still needs GPU/Blender
      hardware and has not happened — only the row's honesty about that is
      fixed. (Resolved in a GPU-less cloud session, 2026-09-22; unaffected
      by the 2026-09-24 refresh.)

Full detail is in `fluidSim_warp_blender51/POLICY_DRIFT_REPORT.md`.
`docs/technical_report.md` is also referenced by root's
`TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE`, but that card
tracks a separate concern (`demos/analysis_report.json` sync, already
handled in `f383eae`) and doesn't gate this one closing — see the "Relationship
to the root-unit finding" section of the report above.

## Second reopen scare (2026-09-24)

Master's own 2026-09-24 audit regeneration (`e599369`, landing the
`d2a6b6d`/`19864c3`/`21d4d37` determinism fix) re-derived this file's
`POLICY_DRIFT_REPORT.md` from a basis that predates the 2026-09-22 PD-PKG-002/
003 fix — because that fix only ever existed on this PR branch, never merged
into master, master's own audit had no way to know about it and correctly
(from its own vantage point) marked the card open again. Re-applied the same
resolution on top of master's newer, correct PD-PKG-001 data after merging.
No actual regression; just two histories that hadn't met yet. Worth merging
this PR into master soon so master's own audit stops rediscovering already-
closed findings.

## Blocked by

None - closed 2026-09-22, reconfirmed 2026-09-24.
