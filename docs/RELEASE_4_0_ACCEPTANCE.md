# 4.0 acceptance ledger

Reconciles four working sessions from 2026-10-14/15 into one decision per task
and open question: the v4.0.0 release plan (PR #30), the performance-gate
refactor, the chromatic_vortex GPU colour transfer, and the release/4.0
worktree reconciliation. Two Codex branches on the F: drive are also reviewed.
Every verdict is checked against live git on `release/4.0`, not against a
transcript's own claim.

Verdicts: **ACCEPT** (already on release/4.0), **ACCEPT after move**
(cherry-picked onto release/4.0), **REJECT** (superseded; archived as an
`archive/*` tag), **DECIDED** (Tyler's call, 2026-10-15).

## Work items

| Item | Where it was | Verdict | Evidence |
|---|---|---|---|
| Lanes W/D/V/K/L-usd/N/R (the 4.0 features) | release/4.0 by `d1e5039` | ACCEPT | Merged before this reconciliation; PR #30 head `295c4cd` contains them. |
| Gate judges order-balanced host ratio vs 3.2.1, with a pressure-accuracy guard | `60b3685` | ACCEPT | `tests/gate/test_performance_budget.py`; INTENDED_CHANGES owns JACOBI→MULTIGRID (+10%) and rigid coupling (+5%). |
| Gate early stop plus overlapped non-budget lanes (29 → 16 min) | `074bd1e`, `726481b`, `cdaf7f6` | ACCEPT | Gate session: Tyler said "keep it as is". The first gate on it matched demo hash `74dada72…`. |
| crown_splash rigid-coupling cost (1.324 → 1.153 vs 1.155) | `9661fb0` (from fix/crown-rigid-cost) | ACCEPT | Bitwise parity tests `tests/solver/test_rigid_integrate_parity.py`; 2585-test suite green. |
| Installer test reads the Warp pin from the wheel lock | `0b14756` | ACCEPT | 7/7 real-Blender installer tests. |
| chromatic_vortex live colour lookup on Warp (775 → 112 ms/frame) | perf/nearest-values-gpu `604e92b` | ACCEPT after move → `703dd70` | Bitwise identical to the NumPy path (fixture test). It runs only on the live path, so bakes and the demo hash are unchanged. |
| Refuse cache export while a live solver is owned (PR #30 review r4278308803, a data-loss bug) | Codex fix/export-live-session-20261015 `be8019e`, `425deb0` | ACCEPT after move → `48bcc88`, `3f05d97` | The two commits cherry-picked cleanly. 245 focused tests pass (interchange lifecycle, cache playback, USD capacity, gate semantics and budget). The new `EXPORT_LIVE_SESSION_OK` acceptance script runs in the gate. |
| Pinned readback for rigid impulses plus an explicit JACOBI→MULTIGRID budget patch | Codex fix/performance-budget-20261015 `b29b968` | REJECT (superseded) | `9661fb0` removed the per-substep readback entirely, and `60b3685` INTENDED_CHANGES already owns the transition. Its per-substep pressure recorder is diagnostic only. Archived as a tag. |
| Learned pressure initial guess | lane N | ACCEPT as hidden/off | The ship gate was not met: the oracle bound is ≤1.2× on demos and wall clock is 0.968×. See `RELEASE_4_0_STATUS.md`. No weights ship. |

## Open questions

| Question | Decision | Follow-through |
|---|---|---|
| dam_break 1.09–1.11× vs 3.2.1 against the 1.10 Multigrid allowance | **DECIDED:** loosen Multigrid's stop condition until dam_break is ≤ 1.0× vs 3.2.1. Keep the 10% allowance. Check every demo visually over full runs for volume loss or jitter. Then run one full gate and update its stored records. | Done: check every iteration, 4 coarse sweeps, stop at half the tolerance (the full tolerance lost chromatic_vortex volume). dam_break reached about 1.03x, not 1.0x; see "Multigrid stop rule" below. |
| Liquid Deck crash risks (context.area None, double-invoke, Ctrl+Z swallowed, gizmo with no active object, stale running flag after load) | **DECIDED:** fix before 4.0 | Done: `2abe4cb`, `1b6d0ae`, `0ca7eb1`, `4dabb4e`, `b7203ce` (from fix/deck-risks-40; the last one keeps plain keys such as Delete in the palette). |
| USD export omits Dynamic colliders (Alembic carries them) | **DECIDED:** fix before 4.0 | Done: `58746c2` (from fix/usd-carried-40). |
| Deck wheel binding: Alt+RMB in code, plain RMB in spec | **DECIDED:** keep Alt+RMB, fix the spec | Done: `69e077d`. |
| Linux CI and Linux USD certification | **DECIDED:** defer to 4.1; 4.0 ships on the Windows GUI gate | Done: CHANGELOG 4.0.0, "Known limitations". |
| Plan open decisions 1–4 (driver floor, APIC default, Deck default state, learned-guess weights) | Plan defaults stand | No change requested. |

## Multigrid stop rule

**Rule shipped:** a MULTIGRID solve checks its residual after every iteration
(`MG_CHECK_EVERY_CANDIDATES = (1,)`, was `(5, 3)`) and stops at half the
configured pressure tolerance (`MG_TOLERANCE_SCALE = 0.5`). Its coarsest level
runs 4 damped-Jacobi sweeps (`MG_COARSE_SWEEPS = 4`, was 16). All three
constants are in `sim/pressure.py`. Jacobi solves are unchanged.

Why each part:
- Before, a solve that met the tolerance after 6 V-cycles ran on to iteration
  10, the next multiple of 5. Checking every iteration removes that overshoot.
- Each coarse sweep is a latency-bound graph node (~6.5 us), so checking every
  iteration with 16 sweeps cost more than it saved.
- Stopping at the full tolerance (shipped in `ccfbefa`) was faster, but over
  a full run it visibly lowered chromatic_vortex's liquid level (below). Half
  the tolerance costs about one V-cycle and restores it.

**Speed.** Paired runs vs 3.2.1, order-balanced host ratio. The first two
columns are 8-trial ABBA runs (`benchmark_versions.py --demo-budget
--full-schedule`; `orch\scratch\mg\pair1`, `pair2`). The half-tolerance
column is a 4-round alternating screen (`screen4_<demo>.log`); the gate below
is the binding measurement.

| Demo | Before (check 5, 16 sweeps) | Check 1, 4 sweeps, full tolerance | **Shipped: half tolerance** | Budget |
|---|---|---|---|---|
| dam_break | 1.09-1.11x | 1.033x | **1.026x** (7 iterations) | 1.10 |
| crown_splash | 1.153x | 1.098x | **1.001x** (5 iterations) | 1.155 (Multigrid x rigid body) |
| paddle_stir | not paired | 1.023x | **0.998x** (6 iterations) | 1.10 |
| chromatic_vortex | not paired | 0.190x (gate on `ccfbefa`) | **0.190x** (8 iterations) | 1.05 |

chromatic_vortex's 0.19x includes the GPU colour lookup (`703dd70`).

**dam_break did not reach 1.0x.** Of the remaining ~3%, the pressure stage
accounts for about +0.5 ms/frame. The rest is p2g (+0.4 ms) and ~1 ms of host
time outside the timed stages, which no Multigrid stop rule reaches.

Stopping earlier was measured and rejected:
- A 4x looser tolerance (4 iterations) visibly lost paddle_stir volume
  (final 0.030 vs 0.0446).
- The full tolerance lost chromatic_vortex volume (below).
- One pre/post smoothing sweep needed 8 iterations and was slower.
- Three levels instead of four was slower; so was 3-iteration checking with
  4 sweeps (dam_break 1.144x).

The next lever is warm-starting from the previous frame's pressure (4.1).

**Volume and jitter over full demo runs.** Every frame, surface volume and
second-difference jitter (`fullrun.py`, `analyze_full.py`; runs in
`orch\scratch\mgull1`-`full5`). "Before" is the old rule (check every 5th,
16 sweeps).

| Demo | Variant | Iterations (median) | Volume drift, first to last frame | Jitter median / p95 (% of mean volume) | Volume diff vs before, mean / max |
|---|---|---|---|---|---|
| dam_break | before | 5 | +6.94% | 0.160 / 1.50 | |
| dam_break | full tolerance | 6 | +6.53% | 0.175 / 1.53 | 0.58% / 2.4% |
| dam_break | **half tolerance** | 7 | +6.13% | 0.147 / 1.61 | 0.70% / 2.3% |
| crown_splash | before | 5 | +10.36% | 1.02 / 6.56 | |
| crown_splash | full tolerance | 4 | +10.85% | 1.21 / 5.43 | 4.2% / 14.8% |
| crown_splash | **half tolerance** | 5 | +7.93% | 1.27 / 5.90 | 2.3% / 8.5% |
| paddle_stir | before | 5 | -70.17% | 0.113 / 2.47 | |
| paddle_stir | full tolerance | 4 | -69.68% | 0.154 / 2.11 | 1.0% / 2.2% |
| paddle_stir | **half tolerance** | 5 | -70.35% | 0.107 / 3.97 | 0.93% / 4.5% |
| chromatic_vortex | before | 10 | -4.18% | 1.13 / 7.08 | |
| chromatic_vortex | full tolerance | 7 | **-10.46%** | 0.885 / 5.91 | 4.4% / 27.5% |
| chromatic_vortex | **half tolerance** | 8 | +0.99% | 0.809 / 5.31 | 4.4% / 26.3% |

- **chromatic_vortex lost liquid under the full tolerance.** Its mean surface
  volume over the run was 2.9% under the old rule and the fill is visibly
  lower at frames 96 and 120 (`full3\cmp_chromatic_vortex.png`). At half the
  tolerance the run mean is within 0.75% of the old rule and the end volume is
  above it. The per-frame max differences (~26%) are the stirred surface
  moving to different places, chaotic in every variant including checking
  every 5th with 4 sweeps (15.6%).
- paddle_stir's drain is the demo's sink and behaves the same in all three;
  it ends at 0.0444 vs 0.0446.
- crown_splash differences are the chaotic splash; half tolerance is closer
  to the old rule than the full tolerance was.
- Jitter stays within run-to-run variation everywhere.

Rendered frames (`mg_half_tolerance_check.png`: dam_break and crown_splash at
frames 48 and 120, chromatic_vortex at 96 and 120) look the same as the old
rule, including chromatic_vortex's fill level. The paddle_stir render camera
only sees the outside of the bowl, so its evidence is the volume curve.

**Accuracy guard.** The old rule overshot the tolerance: 3.2.1's Jacobi checks
every 25th iteration and its Multigrid every 5th. Stopping near the tolerance
puts the median residual ratio vs 3.2.1 above the gate's general 1.10x guard.
At half the tolerance it measures dam_break 3.7x, chromatic_vortex 5.4x
(Multigrid in 3.2.1 too), paddle_stir 0.6x and crown_splash 0.4x.

The first full gate on `ccfbefa` failed on exactly this: chromatic_vortex
`less_accurate` at 10.30x, because the residual allowance was keyed only on
the JACOBI->MULTIGRID transition and chromatic_vortex already ran Multigrid in
3.2.1. Per Tyler ("target speed, not 3.2.1's error level"), a candidate
Multigrid solve now carries its own residual limit of 6x
(`MULTIGRID_RESIDUAL_RATIO_LIMIT` in `tools/performance_budget.py`):
- through the JACOBI->MULTIGRID intended change; or
- where both versions run Multigrid, through the `multigridStopRule` entry,
  which adds no time allowance (budget ratio 1.0).

The limit rejects the full-tolerance rule, which measured dam_break 7.2x and
chromatic_vortex 10.3x and is the one that lost volume. Runs without
Multigrid keep the 1.10x guard. Tests:
`tests/gate/test_performance_budget.py` (606 passed with `tests/solver`).

## Gate and release

**Full GUI production gate at `f4c38cd`: PASS** (2026-10-15, RTX 4080 SUPER,
Blender 5.2.0 LTS, Warp 1.18.0, 773.9 s; log
`C:\Users\Tyler\wt\orch\scratch\r40-final\gate.log`).
- FULL-PRODUCTION-GATE-PASS, DEMO-ANALYSIS-PASS (21/21).
- Analysis report sha256 `b20d6606ea3a10f34feb99aeca48bcd87592a977e36c7b634e91f0a63300bcf8`.
  It differs from `74dada72...` (gate at `479b227`) because the Multigrid stop
  rule changes every Multigrid bake. The technical report's promoted values
  and the rebuilt crown_splash and dam_break files are rebound to this run.
- Paired budget vs 3.2.1, 8 trials, 48 frames (`PERFORMANCE-BUDGET-PASS`):

| Demo | Stage | Host ratio (order spread) | Budget | Residual ratio (limit 6x) | Pressure iterations, 3.2.1 -> 4.0 |
|---|---|---|---|---|---|
| dam_break | early stop | **0.869x** (0.065) | 1.10 | 3.68x | 50 -> 7 |
| crown_splash | full | **1.132x** (0.098) | 1.155 | 0.44x | 25 -> 5 |
| paddle_stir | full | **1.041x** (0.190) | 1.10 | 0.60x | 50 -> 6 |
| chromatic_vortex | early stop | **0.189x** (0.021) | 1.05 | 5.42x | 10 -> 8 |

Host ratios move by several percent between gate runs of the same code
(dam_break: 1.081x when the `ccfbefa` gate was re-scored, 0.869x here;
paddle_stir's order spread is 0.19). The screen and ABBA figures under
"Multigrid stop rule" are the better point estimates. Every demo is inside
its budget in all of them.

Attempts before this one:
- Attempt 1 stopped on a tracked `.npz` probe file, which was removed.
- Attempt 2, at `ccfbefa`, failed the residual guard on chromatic_vortex (see
  "Accuracy guard").
