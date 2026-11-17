# Warp 1.19.0 production gate evidence (17 November 2026)

Result: `HEADLESS-GATE-PASS` (exit 0), including `PERFORMANCE-BUDGET-PASS`.

- Candidate: origin/master `45824b891c0234e6ffa39abdcbb08b39a629099b`, clean checkout.
- Command: `python tools/run_production_gate.py --measure-budget`, run with Blender's bundled
  Python and an exclusive blender-gate GPU lease.
- Runtime: Blender 5.2.0 LTS (official Windows x64 zip, archive SHA-256
  `2d184b62...318adc4`, matches the published `blender-5.2.0.sha256`), Warp 1.19.0 (CUDA
  Toolkit 13.4, driver 13.5), NVIDIA GeForce RTX 4080 SUPER, Windows 11.
- Not run: the foreground GUI lane (`--include-gui`). Certification is therefore scoped to the
  headless production gate on Windows x64 / Blender 5.2.0. Linux is unqualified.

## Paired performance budget (candidate / release 4.0.1, commit `46cdf3ce67b1`)

Both sides ran Multigrid with rigid coupling; base ratio limit 1.05, residual guard unchanged.
Ratios below 1.0 mean the candidate was faster.

| Demo | Trials (base/cand) | Stage | Host ratio | Solver ratio | Pressure residual ratio |
|---|---|---|---|---|---|
| crown_splash | 8/8 | full | 0.9965 | 0.9837 | 1.00 |
| dam_break | 4/4 | early-stop pass | 0.9762 | 0.9852 | 1.00 |
| paddle_stir | 4/4 | early-stop pass | 0.9831 | 0.9930 | 1.00 |
| chromatic_vortex | 4/4 | early-stop pass | 0.9895 | 1.0086 | 1.00 |

## Files

- `production-gate-run3.txt`: full gate log (stored as `.txt`; `*.log` is git-ignored).
- `performance-budget-summary.compact.json`: the budget summary without per-stage and per-order
  detail (the raw 6.6 MB report stays in the ignored `artifacts/` directory of the run).
- `gate-timing.json`: step wall times.
- `blender-provenance.json`: the distribution record the gate required. `reviewed_by` names the
  automated session that checked the archive hash against the official checksum, not a human
  reviewer.

Note: the log's "runs" before this one (run1) was cut off by a session restart and run2 refused to
start on dirty generated demo files; neither produced a verdict. Run3 is the single complete run.
