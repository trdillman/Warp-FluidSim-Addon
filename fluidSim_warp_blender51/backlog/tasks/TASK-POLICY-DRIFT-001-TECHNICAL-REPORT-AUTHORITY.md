# Refresh nested technical report authority

## State

ready-for-agent

## What to build

Update the nested package technical report so its promoted suite counts,
proof HEAD, demo metrics, and artifact hash match current executable evidence.
The package's current tests and release verification pass, but the report
still cites an older proof snapshot.

## Correction (2026-09-22)

A prior pass in this thread marked this card `done` against a 2026-09-22
cloud-session refresh that synced the report to the 2026-09-12 proof HEAD
(`01c4bf837bcec9b5`). That refresh was made without first pulling `master`,
which already carried a newer, more thorough 2026-09-21T03:36 audit (done by
Tyler with another Claude session) showing that proof itself is stale and the
parent GUI gate is currently **blocked with no receipt**
(`gate_verdict: BLOCKED-INTERACTIVE-GUI-NO-RECEIPT`). That audit's version of
`fluidSim_warp_blender51/POLICY_DRIFT_REPORT.md` is authoritative; this card
is reopened to match it, per that report's `active_backlog_tasks` list.

## Acceptance criteria (per the 2026-09-21T03:36 audit, PD-PKG-001/002/003)

- [ ] **PD-PKG-001**: `docs/technical_report.md` still promotes proof HEAD
      `01c4bf...` and artifact hash `afb53b3f...`; the current generated
      artifact hash is `ef62e6506d4e4131a20804d3d67dfecdbdf20517196d204fe79eb515c3051c2a`
      with `column_height_end=0.3355760574` and `max_crown_height=0.2811620831`.
      Needs all four promoted metric tokens updated
      (`front_speed_over_c0=1.0113339117`, `column_height_end=0.3355760574`,
      `crown_sqrt_t_corr=0.9713680253`, `max_crown_height=0.2811620831`) **and**
      a regenerated `demos/analysis_report.json` committed in the same commit
      (never promote a hash that exists only as an untracked artifact).
- [ ] **PD-PKG-002**: the promoted `| Full Python suite | **1,043 passed** |`
      row isn't attributed to either the nested package suite or the parent
      repository suite, and no run currently backs that number.
- [ ] **PD-PKG-003**: the `Installed headless no-CUDA exit` row at line 486
      must be marked historical/pending, not tabulated as current proof.

Full detail and the repair checklist are in
`fluidSim_warp_blender51/POLICY_DRIFT_REPORT.md`. Note the cross-link there:
this card and root's `TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE`
jointly own `docs/technical_report.md` — don't close one without checking the
other.

## Blocked by

Regenerating `demos/analysis_report.json` and the promoted metrics needs a
real demo run (`tools/analyze_demos.py` against real `.blend` output), which
needs GPU/Blender hardware. Not reproducible from a cloud session — this is
Tyler's to run locally, same as the parent GUI gate rerun.
