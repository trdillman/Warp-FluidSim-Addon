---
status: fresh
escalated: false
generated_at: 2026-07-21T20:06:18Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: 231911d59d6083cd439ca9ca6039b05182198977
---

# Policy Drift Report

## Verdict

The safe-settings Blender RNA boundary is resolved. The direct assignment seam
now uses the destination RNA type to normalize integer recommendations, and the
full GUI-inclusive production lane passed on Blender 5.1 / `cuda:0`.

PD-ROOT-002 remains open: checked-in demo evidence and the technical report
still need synchronization from one current reproducible profile. That
documentation task does not block the now-passing production lane.

## Audit units

- Root unit: this directory, containing the Blender 5.1 add-on and production
  lane.
- Nested unit: `demos`, with its own report and checked-in analysis artifacts.

## Strongest live evidence

- **PASS** `python tools/run_production_gate.py --include-gui`, started at
  `2026-07-21T20:00:16.660750+00:00`, exit code `0`, ending in
  `FULL-PRODUCTION-GATE-PASS`.
- **PASS** the production lane's live recovery: it stopped the unsafe 24.4 GiB
  allocation before CUDA allocation, then applied safe settings, initialized
  the solver on `cuda:0`, and emitted `MEMORY-SAFETY-GATE-PASS`.
- **PASS** GUI and saved-demo proof: `GUI-ACCEPTANCE-PASS (35/35)` and
  `SAVED-DEMO-GUI-PLAYBACK-PASS`.
- **PASS** the same run's CUDA and demo lanes: 44 focused regressions, 16/16
  validation tests, kernel coverage, GPU profiler smoke, cache export/playback,
  mesh-source acceptance, and demo analysis (18/18).
- `artifacts/current_policy_drift_full_gate.log` remains historical and was
  excluded from current authority.

## Findings

### PD-ROOT-001 — Safe-settings recovery violates Blender's integer property contract

Status: resolved 2026-07-21

Type: implementation drift, with validation coverage at the integration seam

Evidence:

- `fluidSim_warp_blender51/properties.py:513` declares
  `particles_per_voxel` as a Blender `IntProperty`.
- `fluidSim_warp_blender51/sim/memory_safety.py:235` stores the recommendation
  as a float and `:256,262` scale it fractionally.
- `fluidSim_warp_blender51/operators.py:148-155` now inspects the destination
  RNA definition and applies `int(value)` before assigning an `INT` property.
- `tests/blender_memory_safety_gate.py` verifies the recovered Blender RNA
  property is an `int` after applying safe settings, then advances the solver.
- The full GUI-inclusive production gate passes that recovery and every later
  lane.

Required correction: completed.

### PD-ROOT-002 — Technical report preserves stale demo metrics

Type: documentation drift and validation drift

Evidence:

- `fluidSim_warp_blender51/docs/technical_report.md` still reports the demo
  front speed as approximately `0.94*sqrt(gH)` and preserves stale validation
  counts.
- The current generated artifact reports dam-break
  `front_speed_over_c0=0.975812558470745`, `z_at_T2=2.589575260779174`, and
  `column_height_end=0.09686404466629028`.
- Checked-in `demos/analysis_report.json` and `demos/ANALYSIS.md` still report
  `0.9370464662808852`, `2.4854085941125073`, and `0.234130859375` from their
  2026-07-13 run.

Required correction: regenerate and synchronize the checked-in demo evidence
and technical report from one current reproducible profile.

## Scope assessment

- The current CUDA stages exercised real Warp modules and registered kernels;
  no additional shallow-backend or fabricated-fallback finding was opened from
  lexical markers alone.
- The prior approximate Cone/Pyramid and capacity-sized whitewater findings
  were not reopened: current dirty source and focused coverage contain the
  corresponding repairs, and the current full lane did not contradict them
  during the full production gate.
- The former stale full-gate authority claims were repaired in the reports;
  their report-hygiene cards are marked completed. The root production lane is
  now authoritative; the remaining follow-up is checked-in evidence sync.

## Worktree preservation

The pre-existing dirty solver and performance-test edits, generated demo
binaries/previews and metadata, untracked `.agents/`, and existing audit cards
remain preserved. This repair adds the RNA-seam change, its focused Blender
regression, this root report update, and the TASK-005 status update; the
requested production gate refreshed its normal demo outputs.

## Machine-readable audit footer

generated_at: 2026-07-21T20:06:18Z
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: 231911d59d6083cd439ca9ca6039b05182198977
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification:
  - 'PASS: python tools/run_production_gate.py --include-gui; exit 0; FULL-PRODUCTION-GATE-PASS'
  - 'PASS: MEMORY-SAFETY-GATE-PASS; unsafe allocation stopped, safe retry initialized on cuda:0'
  - 'PASS: GUI-ACCEPTANCE-PASS (35/35); SAVED-DEMO-GUI-PLAYBACK-PASS'
files_that_need_correction:
  - 'fluidSim_warp_blender51/docs/technical_report.md'
  - 'demos/analysis_report.json'
  - 'demos/ANALYSIS.md'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-006-DEMO-EVIDENCE-SYNC.md'
repair_checklist:
  - 'Regenerate and synchronize checked-in demo analysis and technical-report metrics.'
  - 'Refresh the remaining demo-evidence report from current runtime proof.'
