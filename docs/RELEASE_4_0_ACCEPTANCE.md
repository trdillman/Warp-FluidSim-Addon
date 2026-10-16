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
| dam_break 1.09–1.11× vs 3.2.1 against the 1.10 Multigrid allowance | **DECIDED:** loosen Multigrid's stop condition until dam_break is ≤ 1.0× vs 3.2.1. Keep the 10% allowance. Check every demo visually over full runs for volume loss or jitter. Then run one full gate and update its stored records. | Done: check every iteration, 4 coarse sweeps. dam_break reached 1.033x, not 1.0x; see "Multigrid stop rule" below. |
| Liquid Deck crash risks (context.area None, double-invoke, Ctrl+Z swallowed, gizmo with no active object, stale running flag after load) | **DECIDED:** fix before 4.0 | Done: `2abe4cb`, `1b6d0ae`, `0ca7eb1`, `4dabb4e`, `b7203ce` (from fix/deck-risks-40; the last one keeps plain keys such as Delete in the palette). |
| USD export omits Dynamic colliders (Alembic carries them) | **DECIDED:** fix before 4.0 | Done: `58746c2` (from fix/usd-carried-40). |
| Deck wheel binding: Alt+RMB in code, plain RMB in spec | **DECIDED:** keep Alt+RMB, fix the spec | Done: `69e077d`. |
| Linux CI and Linux USD certification | **DECIDED:** defer to 4.1; 4.0 ships on the Windows GUI gate | Done: CHANGELOG 4.0.0, "Known limitations". |
| Plan open decisions 1–4 (driver floor, APIC default, Deck default state, learned-guess weights) | Plan defaults stand | No change requested. |

## Multigrid stop rule

**Rule shipped:** a MULTIGRID solve checks its residual after every iteration
(`MG_CHECK_EVERY_CANDIDATES = (1,)`, was `(5, 3)`), and its coarsest level runs
4 damped-Jacobi sweeps (`MG_COARSE_SWEEPS = 4`, was 16). The pressure
tolerance is unchanged. Before, a solve that met the tolerance after 6
V-cycles ran on to iteration 10, the next multiple of 5. Each coarse sweep is
a latency-bound graph node (~6.5 us), so checking every iteration with 16
sweeps cost more than it saved.

**Speed.** Paired 8-trial ABBA runs vs 3.2.1, order-balanced host ratio
(`benchmark_versions.py --demo-budget --full-schedule`; evidence in
`C:\Users\Tyler\wt\orch\scratch\mg\pair1` and `pair2`):

| Demo | Before (check 5, 16 sweeps) | Check 1, 16 sweeps | **Shipped (check 1, 4 sweeps)** | Budget |
|---|---|---|---|---|
| dam_break | 1.09-1.11x | 1.084x | **1.033x** (order spread 0.009) | 1.10 |
| crown_splash | 1.153x | 1.106x | **1.098x** | 1.155 (Multigrid x rigid body) |
| paddle_stir | not paired | 1.037x | **1.023x** | 1.10 |
| chromatic_vortex | 0.907x (Jacobi, unaffected) | | | 1.05 |

**dam_break did not reach 1.0x.** Of the remaining ~3%, the pressure stage
accounts for +0.5 ms/frame. The rest is p2g (+0.4 ms) and ~1 ms of host time
outside the timed stages, which no Multigrid stop rule reaches.

Stopping earlier was measured and rejected:
- A 4x looser tolerance (4 iterations) visibly lost paddle_stir volume
  (final 0.030 vs 0.0446).
- One pre/post smoothing sweep needed 8 iterations and was slower.
- Three levels instead of four was also slower.

The next lever is warm-starting from the previous frame's pressure (4.1).

**Accuracy guard.** Multigrid now stops at the configured tolerance, while
3.2.1's Jacobi, checked every 25th iteration, overshoots it. So the median
residual ratio vs 3.2.1 sits above the gate's general 1.10x guard:
dam_break 7.17x, paddle_stir 1.29x, crown_splash 1.12x.

Per Tyler ("target speed, not 3.2.1's error level"), the JACOBI->MULTIGRID
intended change in `tools/performance_budget.py` now carries its own residual
limit of 10x (`MULTIGRID_RESIDUAL_RATIO_LIMIT`).
- A doubled tolerance measures 16x and still fails.
- Runs without the preconditioner change keep the 1.10x guard.
- Re-scoring pair2 with the new reducer, all four demos pass.

**Volume and jitter over full demo runs.** Every frame, surface volume and
second-difference jitter (`fullrun.py`; `full1/` before, `full2/` shipped,
side-by-side frames in `mg_before_vs_shipped.png`):

| Demo | Variant | Volume drift, first to last frame | Jitter median / p95 (% of mean volume) | Volume diff vs before, mean / max |
|---|---|---|---|---|
| dam_break | before | +6.94% | 0.160 / 1.50 | |
| dam_break | shipped | +6.53% | 0.175 / 1.53 | 0.58% / 2.4% |
| crown_splash | before | +10.36% | 1.02 / 6.56 | |
| crown_splash | shipped | +10.85% | 1.21 / 5.43 | 4.2% / 14.8% (chaotic splash; check 1 alone: 4.2% / 13.5%) |
| paddle_stir | before | -70.17% | 0.113 / 2.47 | |
| paddle_stir | shipped | -69.68% | 0.154 / 2.11 | 1.0% / 2.2% |

No demo loses volume against the old rule: paddle_stir ends at 0.0454 vs
0.0446, and its drain is the demo's sink, the same in both. Jitter stays
within run-to-run variation.

Rendered frames 48 and 120 of dam_break and crown_splash look the same. Only
the crown_splash cork drifts to a different spot, which is chaotic and moves
with the check-every-iteration change alone. The paddle_stir render camera
only sees the outside of the bowl, so its evidence is the volume curve.

## Gate and release

To be filled in with the gate log, the demo hash and per-demo ratios, the version
bump, the build, and the smoke install.
