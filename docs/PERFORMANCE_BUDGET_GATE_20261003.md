# D7 performance-budget gate verification (2026-10-03)

Implementation base: master `11aa45a73766d9073eb51b8d544168ab75decfc5`.
Solver, physics, substep settings and tracked demos are unchanged. The exact
reference/runtime/frozen-scene contract is in `RELEASE_2_1_HANDOFF.md` section 3.

## Verification

- Focused budget, benchmark-validation and production-gate truthfulness/coverage
  suite after all implementation changes: **82 passed** (2.01 seconds).
- Extended suite including profiler-phase and performance-hotpath checks:
  **105 passed** (140.11 seconds), before the final workload-evidence tests.
- Ruff on changed/new tools and focused tests, compilation, and diff checks passed.
- Tests cover sustained regressions, excluded warmup, outliers, ordering drift,
  minimum trial count, missing/duplicate/malformed evidence, changed settings,
  stage evidence, legacy storage adaptation, inactive defaults, real advancement,
  particle loss, reduced adaptive work, failure propagation and stale PASS removal.
- Independent review's runtime-workload finding was addressed and re-reviewed.

## GPU result: PASS

Blender 5.2 LTS, RTX 4080 SUPER; references on isolated Warp 1.16.0 and
candidate on Warp 1.17.0. All **128 fresh-process trials** completed: eight
reference/candidate pairs per demo in each of timing and synchronized-stage lanes.
Each trial advances 48 simulation frames, scoring the median of frames 2-48.
The following ratios are medians of per-pair candidate/reference trial medians;
values below 1 are faster. Work ratios use actual total substeps in frames 2-48.

| Demo | Timing ratio | Stage solver ratio | Substep ratios (timing / stage) | Verdict |
|---|---|---|---|---|
| dam_break | 0.6289 | 0.6860 | 1.0549 / 1.0105 | PASS |
| crown_splash | 0.6326 | 0.7377 | 0.9871 / 0.9829 | PASS |
| paddle_stir | 0.6272 | 0.7672 | 0.9691 / 0.9691 | PASS |
| chromatic_vortex | 0.6616 | 0.6545 | 1.0122 / 1.0122 | PASS |

Both lanes are below 1.05 for every demo. Per-frame particle counts agree in
all pairs; actual solver frames/time advance and every frame has positive work.
All effective settings, frozen asset hashes and devices match across the cohort.
The measured P2G medians remain visible: dam_break 2.781 -> 5.090 ms and
paddle_stir 1.664 -> 2.814 ms, despite faster solver totals. No phase regression
is hidden by omitting its profile; the budget is for complete frames/solver totals.

Initial setup attempts were rejected for runtime-path or Scene/Domain storage
incompatibility and are excluded. The final vortex asset matches the archived
`vortex_2x.blend` byte-for-byte, and 1.0.255 kernel module hashes match the
original archived stage logs. The execution connection dropped after 118 trials;
the original process had stopped. Recovery retained only complete successful
pairs whose code HEAD, asset hash, Warp version and profiling mode still matched,
then ran the ten remaining trials through the same rotating fresh-process driver.
The final evaluator revalidated all 128 raw records and returned
`PERFORMANCE-BUDGET-PASS` (exit 0); independent recomputation agrees.

Durable evidence: `evidence/performance-budget-20261003/budget.json`,
`runs.json.gz` (full per-frame timings/stages/workload records), and
`implementation.sha256` (UTF-8/LF measurement implementation hashes).
The SHA-256 of the decompressed raw JSON bytes is
`1e11023088d68828778feba59c10148ba2cf719c75fccfc380edcd4ff6370248`.
Individual process logs remain in `artifacts/performance_budget_final/`.

A full GUI production gate was not run for this tooling change. This focused
comparison does not create a full-production manifest. Normal gate invocations
always run fresh trials; this interrupted-run recovery is not a production
resume/cache option.
