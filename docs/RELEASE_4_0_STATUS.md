# 4.0 release status

Measured results for 4.0 items whose ship decision depends on a gate. Numbers
here are measurements, not targets; each one names its method.

## Learned pressure initial guess (lane N)

Spec: `docs/specs/4.0/SPEC-learned-pressure-guess.md`.

**Verdict: the ship gate is not met.** The feature ships hidden and off, and no
weights file ships. `pressure_learned_guess` has no UI, defaults to False, and
with no weights file it falls back to the zero guess. The Live Readout then
shows "Learned guess: off (weights missing)". When the setting is off, results
and cache identity are unchanged: the field is left out of every settings hash
at its default. The training and evaluation tooling stays in
`tools/learned_guess/` and has tests.

| Gate line | Required | Measured | Pass |
|---|---|---|---|
| Demos, median PCG iterations over frames 2-48 | >= 2x fewer | at most 1.2x, even for a perfect guess (oracle) | no |
| Held-out scenes, median PCG iterations | >= 1.5x fewer | 1.00x (trained model) | no |
| Net wall-clock win | > 1.0x | 0.968x, so guess-on is about 3% slower | no |
| Residual <= tolerance on every guess-on solve | all | all converged (12 held-out scenes, both check cadences) | yes |
| Guess off reproduces current results bitwise | identical | identical (see below) | yes |

### Why 2x cannot be reached: oracle bound on the four demos

The gate is decided by an upper bound, not by the model. In the 4.0 solver,
the multigrid-preconditioned PCG already converges in a handful of
iterations. Even the best possible initial guess removes almost none of them.

Method:
- Each demo ran in Blender 5.2 for frames 2-48.
- Every pressure solve ran twice, from the same state:
  - once with the zero guess;
  - once seeded with an oracle, meaning that solve's own converged pressure
    restricted to the coarse per-leaf grid and injected back to voxels. This
    is exactly the information a perfect coarse-grid model could supply.
- Iterations were counted exactly, checking convergence every iteration
  (`check_every=1`).
- "Production" re-quantises the counts to the shipped convergence-check cadence
  (`check_every=5`).

| Demo | Solves | Zero-guess median | Oracle median | Median ratio | Total-iteration ratio | Production median (zero -> oracle) |
|---|---|---|---|---|---|---|
| dam_break | 404 | 6 | 5 | 1.20x | 1.149 | 10 -> 5 (totals 1.39x) |
| crown_splash | 335 | 4 | 4 | 1.00x | 0.959 (oracle worse) | 5 -> 5 |
| paddle_stir | 141 | 8 | 7 | 1.14x | 1.087 | 10 -> 10 |
| chromatic_vortex | 522 | 7 | 7 | 1.00x | 1.069 | 10 -> 10 |

A plain warm start is also not enough. Seeding each solve with the previous
solve's coarse pressure gave the same counts as the oracle on all four demos.
The lost iterations are fine-scale error, and a coarse guess cannot carry
fine-scale error.

### Trained model on the held-out set

The model was trained once, end to end, with the shipped tooling.

**Data**
- `export_dataset.py` on training seeds 0-47, 24 frames each: 3907 samples in
  13 coarse-grid shapes.
- The generator draws pool, column and drop scenes in domains 16-40 cells per
  axis.
- The held-out seeds (1000+) are two separated blocks, a shape that never
  appears in training.

**Training (`train.py`)**
- CPU PyTorch 2.9.1.
- Width 16, 4 modes, 3 layers, a 32-unit projection.
- 40 epochs. Final normalised masked MSE: 3.0e-4.

**Export (`export_weights.py`)**
- 1,134,457 bytes, under the 5 MB limit.
- The file stays in scratch and is not committed.

**Evaluation (`evaluate.py`)**
- Held-out seeds 1000-1011, 48 frames each.
- Medians are taken over frames 2-48.
- Wall clock was measured without the recording hooks, as interleaved
  ABBA-paired runs: 2 blocks of off/on/on/off per scene, frames 2-48 timed,
  frame 1 (compile and capture) excluded.
- Hardware: RTX 4080 SUPER, driver 610.88, Warp 1.18.0.

| Held-out result | Exact (`check_every=1`) | Production cadence |
|---|---|---|
| Median of per-scene median ratios | 1.00x (range 1.00-1.17) | 1.00x |
| Total-iteration ratio (off/on) | 0.998 | 0.986 |
| Wall-clock ratio, median of ABBA pairs | 0.968 (range 0.912-0.991) | |
| Guess-on solves converged | all 12 scenes | all 12 scenes |

The model learns the target well. That still cannot help here, because the
oracle shows the coarse guess has almost nothing to remove. The cost of
inference then makes the solve a few percent slower.

### Guess off is bitwise unchanged

- `tests/solver/test_learned_pressure_guess.py` covers the following:
  - guess forced off is bitwise identical to the zero-guess path;
  - fixture weights still converge to tolerance;
  - missing weights or a schema mismatch give a zero guess plus a reason,
    with no per-frame warning;
  - inference keeps graph capture working.
- An independent probe compared base `2cff18f` with this branch. It ran:
  - two tanks, narrow band off and on, 12 frames each;
  - a thrown column, 20 frames.

  Both trees hashed `98a53ff6...481f`.
- Cache identity is pinned by `tests/cache_io/test_learned_guess_cache_identity.py`.
  The existing settings-signature pins are unchanged.

### Not done or not proven

- The trained model was not run on the four demos. The oracle bound makes the
  demo line fail whatever the model does. The demo numbers above are the
  oracle bound, not a model measurement.
- The spec's dataset was the four demos plus generated scenes with obstacle
  boxes and viscosity on and off. The generator here has no obstacles and no
  viscosity, and the demos were kept out of training.
- The spec's architecture was 24 channels, 16 modes and 4 layers. The coarse
  grids are 6-24 leaves per axis, so 16 modes per axis is more than the grid
  holds. The trained model is smaller, and the tests show the weights format
  accepts a 24/4/4 model under 5 MB.
- The settings signature includes `pressure_learned_guess` when it is on, but
  not a weights schema version or digest.
- When a bake resumes, the previous-pressure input channel starts from zero.
  Results with the guess on depend on iteration count, but every solve stays
  within tolerance.
- The torch parity test skips in the add-on test runner, which has no torch.
  It was run by hand with CPython 3.14 and torch 2.9.1 (max abs error about
  5e-7).
- No GUI production gate was run for this lane.
