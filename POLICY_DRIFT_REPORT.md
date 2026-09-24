---
status: fresh
escalated: false
generated_at: 2026-09-24T10:10:12.810378+00:00
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: cb9ad0082fa27392ffdae1bff6fd481e998e8aa7
gate_run_at: 2026-09-24T10:10:12.810378+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'FULL-GATE-PASS-EVIDENCE-CURRENT'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 98379db74ef635b73c284b4d0a50aeec0f258603763b8a0ef00a3a8ae884b5c8
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

All four findings are now resolved. This run confirms PD-ROOT-004, the bake
nondeterminism, is fixed. The gate ran twice, at `d2a6b6d` and at this proof
HEAD, and both runs produced a byte-identical demo artifact. The evidence refresh
for this run lands in the same commit as this report.

## Findings

### PD-ROOT-001: checked-in demo evidence diverges from the fresh artifact (resolved this run)

Resolved in the same commit as this report. `demos/analysis_report.json` now
mirrors the gate artifact byte for byte, SHA-256
`98379db74ef635b73c284b4d0a50aeec0f258603763b8a0ef00a3a8ae884b5c8`. `demos/ANALYSIS.md` and
`fluidSim_warp_blender51/docs/technical_report.md` promote that artifact's four
metric tokens. The current artifact reports dam-break column height
`0.3388671875 m` and crown height `0.2607910037 m`. All three files match
`ALLOWED_EVIDENCE_PATTERNS`, so landing them does not invalidate the proof. So do
the rebuilt demo blends, the previews, and `demos/*/meta.json`. `meta.json` was
added to the allowlist in `cb9ad00`, because `tools/build_demo_scenes.py`
rewrites it on every gate demo build.

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

At `f383eae`, one manifest field was *not* stable: `demo_artifact_sha256`
differed on every run (`c0888aca…`, `b3a1cf8c…`, `a0b2348c…`), as did
`gate_run_at`. That instability was the bake nondeterminism of PD-ROOT-004
surfacing in the receipt. Since `d2a6b6d` it is gone. Two consecutive GUI-inclusive
runs, at `d2a6b6d` and at `cb9ad00`, wrote the same
`demo_artifact_sha256` (`98379db7…`). The analysis artifacts are byte-identical,
and so are the dam_break and crown_splash `meta.json` files. Both logs are 1916
lines with the same 383 `[PASS]` lines and zero `[FAIL]` lines. Run 1 is
archived under `artifacts/gate_repro/det_d2a6b6d/`.

The gate still asserts `exact payload hashes recorded as diagnostics only`,
and equality is still not an acceptance condition. The artifact hash is now a
usable reproducibility check on one device. Other devices, or Warp/driver
upgrades, may still change it.

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
The contrast with the pre-`d2a6b6d` behaviour of PD-ROOT-004 was structural,
not coincidental. `Lz` is a bulk sum over ~158k particles, so the
order-dependent float atomics averaged out. `max_crown_height` was a
6-particle tail extremum, so they did not. `paddle_stir` was not in the
determinism proof, so its bakes may still vary in the last bits.

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

### PD-ROOT-004: promoted metrics were not reproducible to their promoted precision (resolved by `d2a6b6d`)

**Root cause: the bake was not bitwise deterministic.** The analyzer is a pure
function of the cache, so all variance came from the solver. Over 10
`crown_splash` bakes and 3 `dam_break` bakes at one HEAD, none of the four
promoted metrics reproduced to 10 decimals:

* `max_crown_height` had a range of 40.5% of its mean.
* `column_height_end` varied by about 1.5%.
* `front_speed_over_c0` varied by about 0.1%.
* `crown_sqrt_t_corr` varied in the 5th decimal.

A stage-hash bisection (`artifacts/determinism/probe_stage_hashes.py` +
`diff_runs.py`) found four order-dependent sites on the demo path:

1. `fluid_dot_reduce` (`sim/cuda_reduction.py`), the PCG dot product, combined
   block partials with float `atomicAdd`. It now uses an ordered last-block
   combine.
2. The P2G float-atomic scatter. It is now a gather over the counting-sort bins.
3. Shape emission reserved particle slots with an arrival-order `atomic_add`.
   That permuted the particle arrays, which the ordered gather then summed in a
   different order. It now reserves by demand kernel, then
   `wp.utils.array_scan`, then a scanned-slot commit.
4. The whitewater free list was pushed in arrival order inside advection. It is
   now a scan-ordered push over `ww.dead` flags.

**Resolution evidence:**

* Two full bakes of each scene are byte-identical in every `frame_*.npz`.
  Their analysis JSON is identical, and `series.json` differs only in timing
  fields.
* This gate run and the preceding one produced the same artifact SHA-256, and
  all four promoted tokens repeat exactly: `front_speed_over_c0=1.0082030232`,
  `column_height_end=0.3388671875`, `crown_sqrt_t_corr=0.9713672681`,
  `max_crown_height=0.2607910037`.
* Full pytest passed 1145 tests at `d2a6b6d`. No consistent step-time
  regression was measured.

This resolves the conflict between the exact-token contract in
`_check_doc_metrics` and the gate's "diagnostics only" hash policy. On the
promoted demo path, a token mismatch now indicates a solver or analyzer change,
not run-to-run variance.

**Still out of scope:**

* Mesh and compiled-source emitters still reserve slots with arrival-order
  cursors.
* Float atomics in color mixing, the `kernels_core` splat, the pressure
  `leaf_rhs` sums and the legacy scatter P2G are unaudited.
* Marching-cubes surface output was not tested for determinism.
* There is no claim of determinism across devices or Warp versions.

None of these affect the four promoted metrics.

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
generated_at: 2026-09-24T10:10:12.810378+00:00
generated_at_utc: 2026-09-24T10:10:12.810378+00:00
escalated: false
audit_unit: 'C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_port'
git_head: cb9ad0082fa27392ffdae1bff6fd481e998e8aa7
gate_run_at: 2026-09-24T10:10:12.810378+00:00
gate_verdict: 'FULL-PRODUCTION-GATE-PASS'
unit_verdict: 'FULL-GATE-PASS-EVIDENCE-CURRENT'
demo_artifact: 'artifacts/demo_analysis/analysis_report.json'
demo_artifact_sha256: 98379db74ef635b73c284b4d0a50aeec0f258603763b8a0ef00a3a8ae884b5c8
freshness: fresh
status: fresh
blocking_criticality: medium
human_decision_required: false
strongest_live_verification_commands_artifacts:
  - 'rtk python tools/run_production_gate.py --include-gui -> FULL-PRODUCTION-GATE-PASS; wrote artifacts/production_gate_manifest.json (include_gui=true)'
  - 'gate GUI lane -> GUI-ACCEPTANCE-PASS (53/53); saved-demo lane -> SAVED-DEMO-GUI-PLAYBACK-PASS'
  - 'gate demo lane -> DEMO-ANALYSIS-PASS (18/18); artifact sha256 98379db7...84b5c8 mirrored to demos/analysis_report.json'
  - 'cmp artifacts/demo_analysis/analysis_report.json artifacts/gate_repro/det_d2a6b6d/analysis_report.json -> identical (gate runs at d2a6b6d and cb9ad00)'
  - 'gate pytest lane (pressure-projection, cache-io, io-path-policy) -> 79 passed; full suite not run this proof'
  - 'rtk python tools/release/build_extension.py --verify-only -> release_ready=true, version 1.0.255'
  - 'rtk python tools/validate_policy_drift_reports.py -> POLICY-DRIFT-REPORT-VALIDATION-PASS'
files_that_need_correction:
  - 'tools/analyze_demos.py'
active_backlog_tasks:
  - 'backlog/tasks/TASK-POLICY-DRIFT-015-CHECKED-IN-DEMO-EVIDENCE-RECURRENCE.md'
  - 'backlog/tasks/TASK-POLICY-DRIFT-018-PADDLE-STIR-BLEND-UNCOMPRESSED.md'
repair_checklist:
  - 'Re-save demos/paddle_stir.blend compressed so the checked-in file matches what the builder emits (TASK-POLICY-DRIFT-018).'
  - 'Done this run: all three unit reports and both promoted-metric surfaces rebound to the 2026-09-24T10:10:12Z GUI-inclusive gate at cb9ad00; the validator passes.'
  - 'Done this run: PD-ROOT-004 closed - d2a6b6d made the demo bakes bitwise deterministic; two consecutive gate runs produced the same artifact SHA-256 and identical promoted tokens.'
  - 'Done earlier: all three unit reports rebound to the 2026-09-22T16:08:07Z gate at a015e95.'
  - 'Done this run: the paddle_stir .blend size divergence was diagnosed as an uncompressed save and carded as TASK-POLICY-DRIFT-018.'
  - 'Done this run: PD-ROOT-002 closed - the GUI-inclusive gate passed 3/3 consecutive detached runs at f383eae with structurally identical logs (TASK-POLICY-DRIFT-017 closed).'
  - 'Done this run: PD-ROOT-003 closed - paddle spin-up acceptance calibrated against 21 real bakes, and a false accept the placeholder allowed was fixed with a second, absolute condition (TASK-POLICY-DRIFT-016 closed).'
  - 'Done earlier: checked-in JSON, ANALYSIS.md, and technical-report metrics refreshed from the gate artifact.'
  - 'Done earlier: a bounded GUI receipt was captured as artifacts/production_gate_manifest.json.'
```
