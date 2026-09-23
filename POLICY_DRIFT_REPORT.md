---
status: fresh
escalated: false
generated_at: 2026-09-22T16:08:07.118697+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: a015e953bbb4ac44cc069b3971cc956ac399eaa7
gate_run_at: 2026-09-22T16:08:07.118697+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'FULL-GATE-PASS-EVIDENCE-CURRENT'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 57b48bfb81a1c1512504e333127a1552ef5453f3463cb5481538536c36b6c59c
git_dirty: true
---

# Policy Drift Report: root unit

## Audit conclusion

Current evidence exercises the real FluidSim/Warp path end to end.
`tools/run_production_gate.py --include-gui` completed every lane at this HEAD
and ended `FULL-PRODUCTION-GATE-PASS`: the interactive Blender lane returned
`GUI-ACCEPTANCE-PASS (53/53)`, the saved-demo timeline lane returned
`SAVED-DEMO-GUI-PLAYBACK-PASS`, the demo analyzer passed `18/18`, and a fresh
`artifacts/production_gate_manifest.json` was written with `include_gui: true`.
Package verification reports `release_ready: true`. No source evidence found a
narrow heuristic solver or fabricated runtime output.

Two prior findings are resolved by this run and by the evidence refresh landing
in the same commit. One remains open, and one is newly observed.

## Findings

### PD-ROOT-001: checked-in demo evidence diverges from the fresh artifact (resolved this run)

Resolved in the same commit as this report. `demos/analysis_report.json` now
mirrors the gate artifact byte for byte, SHA-256
`453b814b588c854c4f9529ba30c121564b1d8a21f36b4e30c96ac7d935fee3b9`, and `demos/ANALYSIS.md` and
`fluidSim_warp_blender51/docs/technical_report.md` promote that artifact's four
metric tokens. The current artifact reports dam-break column height
`0.3361816406 m` and crown height `0.2547363043 m`. All three files are in
`ALLOWED_EVIDENCE_PATTERNS`, so landing them does not invalidate the proof.

`backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md`
stays active: it owns the recurrence, and one synchronization does not close
that.

### PD-ROOT-002: full GUI gate reproducibility (resolved — 3/3 consecutive passes)

The gate ran to completion three consecutive times at proof HEAD
`f383eae8` on an unmodified tree, each launched detached and alone, each
writing `artifacts/production_gate_manifest.json`. All three recorded
`gate_verdict: FULL-PRODUCTION-GATE-PASS` and `include_gui: true`. Logs and
manifests are archived as `artifacts/gate_repro/gate_run{1,2,3}.log` and
`artifacts/gate_repro/manifest_run{1,2,3}.json`.

The three runs are structurally identical, not merely identically verdicted:

* each log is exactly 1916 lines;
* each carries the same 29 `[PASS]` markers in the same order, and zero
  `[FAIL]` markers;
* of the 686 differing lines, 671 differ only in numeric fields on an
  otherwise identical text skeleton, and the remaining 15 are benign
  environmental noise — per-run temp directory names
  (`fluidsim_cache_pipeline_*`, `fluidsim-handler-gate-*`,
  `fluidsim_migration_*`, `fluidsim_equivalence_*`), `bpy_struct` pointer
  addresses, Python set iteration order, timestamps, and diagnostic hashes.

One manifest field is *not* stable: `demo_artifact_sha256` differs on every
run (`c0888aca…`, `b3a1cf8c…`, `a0b2348c…`), as does `gate_run_at`;
`gate_verdict`, `git_head`, `include_gui` and `demo_artifact` are identical.
That instability is the bake nondeterminism of PD-ROOT-004 surfacing in the
receipt, and it means the manifest hash is a per-run receipt rather than a
reproducibility anchor. The gate already takes this position itself — it
asserts `exact payload hashes recorded as diagnostics only — no cross-bake
hash or numerical equality is an acceptance condition` and `CUDA derived
surface nondeterminism reported without equality claim`. Any future
verification should compare verdicts and marker sequences, never manifest
hashes.

The lane is therefore reliable, and the two earlier failures were process
faults rather than gate faults: an attempt launched non-detached hung before
reaching the GUI lane, and a second attempt launched while the first was still
alive contended on the shared Warp kernel cache and `__pycache__` directories
and hung within seconds. Both hazards, and the detached / `python -u` /
durable-log launch procedure that avoids them, belong on
`backlog/tasks/TASK-POLICY-DRIFT-017-FULL-GUI-GATE-REPRODUCIBILITY.md`, which
can now be closed.

### PD-ROOT-003: paddle spin-up threshold was uncalibrated (resolved — and it hid a false accept)

`tools/analyze_demos.py` labelled `PADDLE_SPINUP_MOMENTUM_MARGIN = 0.3` a
placeholder with no bake behind it. Calibrated 2026-09-22 against 21 real
bakes at HEAD `f383eae` over resolution (voxel 0.008-0.016), particle count
(19k-311k), duration (48-144 frames) and paddle speed (0-100 RPM), with five
repeats at the shipped settings. Full derivation:
`docs/paddle_spinup_calibration.md`.

The placeholder was not merely unjustified, it was wrong. The margin divides
by the steady-window mean, so a scene with effectively no drive divides by
noise and the ratio *explodes* instead of collapsing: a 0.5 RPM bake measures
a spin-up ratio of 19.589, and the shipped 0.3 accepted it. No single relative
constant separates that from a real bake, so the check now carries two
conditions — a new absolute floor on the steady circulation nondimensionalised
by `N*|omega|*r_out^2` (`PADDLE_STEADY_CIRCULATION_FLOOR = 0.005`; worst
supported case 0.0152, shipped 0.171, degraded 0.00048) alongside the now-measured
relative margin (`0.30`, half the worst supported case in an envelope running
0.620 to 0.971).

Run-to-run spread over five repeats is 7e-4, three orders of magnitude below
the envelope spread, so the envelope sets the threshold and not the noise.
That contrast with PD-ROOT-004 is structural rather than coincidental: `Lz` is
a bulk sum over ~158k particles, where the order-dependent float atomics
average out, while `max_crown_height` is a 6-particle tail extremum, where
they do not.

Verified by replaying the shipped `analyze_paddle_stir` over all 21 caches
(`CALIBRATION-VERIFY-PASS`): every in-envelope case accepted, all three
degraded cases rejected. Regression coverage added in
`tests/test_demo_analysis_contract.py`. Recorded result on the shipped scene:
`DEMO-ANALYSIS-PASS (10/10)`. Card
`backlog/tasks/TASK-POLICY-DRIFT-016-PADDLE-SPINUP-THRESHOLD-CALIBRATION.md`
is closed.

One incidental finding was raised here and has since been diagnosed:
rebuilding `paddle_stir` from `tools/build_demo_scenes.py` emits a 280 KB
`.blend` where the checked-in one is 5.7 MB. The cause is compression, not
content. The checked-in file begins `42 4c 45 4e 44 45 52` (`BLENDER`,
uncompressed); the rebuild and every other demo blend begin `28 b5 2f fd`
(Zstandard). Both files were inspected headlessly and hold the same
datablocks, including the same 158,220-vertex `Paddle Stir Pool Seed`, so
`paddle_stir.blend` is simply the only uncompressed demo blend in the
repository. It loads, plays back and passes the gate as-is, so this is
repository weight and reproducibility rather than correctness. Tracked as
`backlog/tasks/TASK-POLICY-DRIFT-018-PADDLE-STIR-BLEND-UNCOMPRESSED.md`.

### PD-ROOT-004: promoted metrics are not reproducible to their promoted precision (root cause found)

**Root cause: the bake is not bitwise deterministic.** The analyzer is a pure
function of the cache — re-running it on the preserved gate cache reproduces
`0.2904101610` exactly — so all variance is upstream, in the solver. Two
order-dependent mechanisms in `fluidSim_warp_blender51/sim/kernels_fluid.py`
are responsible: `k_scatter_particles_to_bins` assigns slots via
`starts[voxel] + wp.atomic_add(cursor, voxel, 1)`, so within-voxel particle
order varies per run, and the P2G scatter accumulates with float
`wp.atomic_add`, which is non-associative, so a different arrival order changes
the last bits of every grid velocity. Measured amplification between two runs
(sorted-z max|Δ|, order-invariant): ~6e-5 m at frames 1-6, 2.4e-4 at impact,
9.6e-3 by frame 10, 3.2e-2 m by frame 15 — a 300-500x chaotic blow-up.

**The instability is not specific to one statistic.** Over 10 fresh
`crown_splash` bakes and 3 `dam_break` bakes at this HEAD, none of the four
promoted metrics reproduces to its promoted 10 decimals:

| metric | observed range over re-bakes |
| --- | --- |
| `crown_sqrt_t_corr` | 0.971368 – 0.971430 (5th decimal) |
| `front_speed_over_c0` | 1.01031 – 1.01133 (~0.1%) |
| `column_height_end` | 0.33365 – 0.33862 (~1.5%) |
| `max_crown_height` | 0.21599 – 0.32566, mean 0.27109, sd 0.03070 (range 40.5% of mean, CV 11.3%) |

The earlier "roughly 16%" figure for `max_crown_height` was a 4-sample
underestimate. That metric screams loudest because it is the max over frames of
`percentile(z[crown_mask], 98)` and the argmax frame carries only 39-324 masked
particles — effectively the ~5th-highest particle out of 259,112 at a single
frame. No re-estimation rescues it: over 6 bakes, max-p90 spreads 38.3%,
max-p75 39.7%, p98 gated to frames with n>=500 23.3%, mean-p98 over frames
10-20 24.6%. Two candidates were perfectly stable but are attained at frame 4,
the pre-impact ballistic phase — they measure the falling droplet, not the
crown. Note also that cached positions are stored as float16 (~0.24 mm ulp at
these magnitudes), so 10-decimal promotion was never physically meaningful.

Nothing regressed. Every bounded acceptance check still passes by a wide
margin — crown height is checked only as `> 0.03`, which every run clears by
about an order of magnitude — and the gate itself already asserts that
`exact payload hashes recorded as diagnostics only — no cross-bake hash or
numerical equality is an acceptance condition`.

The consequence is that the exact-token contract in `_check_doc_metrics`
(`tools/validate_policy_drift_reports.py`) contradicts that stated policy: it
only ever holds because the documents are regenerated from whichever gate run
last ran, and for these keys it cannot distinguish expected variance from a
genuine regression. A mismatch on any of them is variance, not regression. The
open decision — per-metric tolerance bands, dropping the exact-token contract
for these keys, or making P2G deterministic via a gather-style pass over the
sorted bins the counting sort already produces — is not taken here. `CHANGELOG`
already lists CUDA solver determinism as an open release gate, so this is a
known gap rather than a new defect. No backlog card owns it yet.

## Scope boundary and non-findings

Intentional numerical clamping, material fallbacks, effect fallbacks, and
test doubles were not promoted as heuristic-runtime findings. The `18/18`
analyzer pass, the package release verification, and the GUI-inclusive gate
receipt are all current for this HEAD.

The gate's own pytest lane is scoped, not exhaustive: it runs
`tests/test_pressure_projection_regressions.py`, `tests/test_cache_io.py`, and
`tests/test_io_path_policy.py`, which reported `79 passed`. That is current
evidence for those regressions only. No full-suite run was executed for this
proof, so the suite count promoted in the package technical report remains
unbacked this run; that stays open as package finding `PD-PKG-002`.

`gate_verdict` is bound to the gate run, not to this unit:
`tools/validate_policy_drift_reports.py` compares it against `gate_verdict` in
`artifacts/production_gate_manifest.json`, so every unit report sharing one
`gate_run_at` must carry the identical string, and it must read
`FULL-PRODUCTION-GATE-PASS` once a passing manifest exists
(`tools/run_production_gate.py` writes no other value). The unit-scoped rollup
belongs in `unit_verdict`, which is free-form and may differ per unit.

## Machine-readable audit footer

```yaml
generated_at: 2026-09-22T16:08:07.118697+00:00
generated_at_utc: 2026-09-22T06:23:28.891512+00:00
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: a015e953bbb4ac44cc069b3971cc956ac399eaa7
gate_run_at: 2026-09-22T16:08:07.118697+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'FULL-GATE-PASS-EVIDENCE-CURRENT'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 57b48bfb81a1c1512504e333127a1552ef5453f3463cb5481538536c36b6c59c
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/run_production_gate.py --include-gui -> FULL-PRODUCTION-GATE-PASS; wrote artifacts/production_gate_manifest.json (include_gui=true)'
  - 'gate GUI lane -> GUI-ACCEPTANCE-PASS (53/53); saved-demo lane -> SAVED-DEMO-GUI-PLAYBACK-PASS'
  - 'gate demo lane -> DEMO-ANALYSIS-PASS (18/18); artifact sha256 453b814b...fee3b9 mirrored to demos/analysis_report.json'
  - 'gate pytest lane (pressure-projection, cache-io, io-path-policy) -> 79 passed; full suite not run this proof'
  - 'rtk python tools/release/build_extension.py --verify-only -> release_ready=true, version 1.0.255'
  - 'rtk python tools/validate_policy_drift_reports.py -> POLICY-DRIFT-REPORT-VALIDATION-PASS'
files_that_need_correction:
  - 'tools/analyze_demos.py'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md'
  - 'backlog/tasks/TASK-POLICY-DRIFT-018-PADDLE-STIR-BLEND-UNCOMPRESSED.md'
repair_checklist:
  - 'Decide how max_crown_height should be promoted given its run-to-run spread (PD-ROOT-004); it currently has no card.'
  - 'Re-save demos/paddle_stir.blend compressed so the checked-in file matches what the builder emits (TASK-POLICY-DRIFT-018).'
  - 'Done this run: all three unit reports and both promoted-metric surfaces rebound to the 2026-09-22T16:08:07Z GUI-inclusive gate at a015e95; the validator passes.'
  - 'Done this run: the paddle_stir .blend size divergence was diagnosed as an uncompressed save and carded as TASK-POLICY-DRIFT-018.'
  - 'Done this run: PD-ROOT-002 closed - the GUI-inclusive gate passed 3/3 consecutive detached runs at f383eae with structurally identical logs (TASK-POLICY-DRIFT-017 closed).'
  - 'Done this run: PD-ROOT-003 closed - paddle spin-up acceptance calibrated against 21 real bakes, and a false accept the placeholder allowed was fixed with a second, absolute condition (TASK-POLICY-DRIFT-016 closed).'
  - 'Done earlier: checked-in JSON, ANALYSIS.md, and technical-report metrics refreshed from the gate artifact.'
  - 'Done earlier: a bounded GUI receipt was captured as artifacts/production_gate_manifest.json.'
```
