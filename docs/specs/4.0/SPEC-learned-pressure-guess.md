# SPEC — Learned pressure initial guess (gated neural component)

Owner module: new `sim/learned_guess.py` (weights loader, Warp tile inference kernels,
schema check), `sim/pressure.py` (the `use_coarse_guess` seam in `solve_pcg` and
`project_frame`; today it can be seeded by `coarse_initial_guess`, which production never
calls), new `tools/learned_guess/` (dataset export from bakes, PyTorch trainer,
exporter to `.npz`, evaluator). Depends on: `SPEC-defaults-4.md` (training bakes use the
4.0 defaults).

## Why an initial guess and not a surrogate

PCG runs to the same tolerance whatever the starting point, so a learned guess can
only change iteration count and wall time; the dossier's "surrogate predicts the
velocity field" would change answers outside its training distribution. This is the
one place a neural component can enter a deterministic production solver without a
trust argument the project cannot make.

## Trap this spec prevents

1. **PyTorch in the extension.** Blender's Python has no torch, and a torch wheel is
   gigabytes. Inference is Warp-native: `wp.tile_fft`/`wp.tile_ifft` (Warp ≥ 1.14,
   libmathdx path for arbitrary sizes, power-of-two fallback otherwise), `wp.tile_matmul`
   for the channel mixing, GELU in a kernel. Training is a developer tool only.
2. **Breaking graph capture.** Inference must be fixed-shape per session (coarse grid
   dims fixed at admission, leaf-level 1-cell-per-leaf grid, padded to the next
   power of two) and captured inside the existing pressure graph. No host sync, no
   `.numpy()`, no allocation in the step.
3. **Changing results beyond tolerance.** The guess feeds `x0` only. Acceptance proves
   that forcing the guess off reproduces the beta hash, and that with the guess on the
   final residual meets the same tolerance. The settings signature includes the weight
   schema version so a cache records which weights seeded it (diagnostic only; the
   cache does not depend on it).
4. **Overfitting to the four demos.** The dataset is the four demos plus a randomised
   scene generator (`tools/learned_guess/scenes.py`: tank sizes, emitter positions,
   obstacle boxes, viscosity on/off) with a held-out split; the gate reports held-out
   iteration reduction separately.
5. **Silent fallback.** If weights are missing, the schema mismatches, or the device
   lacks the tile FFT path, the solver uses the zero guess and the Live Readout shows
   "Learned guess: off (reason)". Never a warning-per-frame.

## Design

- Inputs on the coarse grid (one cell per leaf, like the electrostatic aggregate):
  divergence sum, fluid-voxel fraction, solid fraction, previous frame's coarse `q`.
  Output: coarse `q` guess, prolonged to voxels by injection (same prolongation as the
  multigrid spec) as `x0`.
- Architecture: 3D FNO, 4 layers, 16 modes per axis, 24 channels; ~1.5 M parameters,
  fp32 weights `.npz` < 5 MB, schema `learned_guess_v1`.
- Settings: `pressure_learned_guess` (bool, default decided by the gate),
  OUTPUT-neutral class = RESET (changes session fixed shapes).
- Tooling: `export_dataset.py` (reads bake checkpoints and per-frame coarse fields; may
  need a bake-time "dump coarse fields" diagnostic flag), `train.py` (PyTorch, Adam,
  relative L2 on `q`), `export_weights.py`, `evaluate.py` (iterations/wall, held-out).

## Acceptance (ship-on gate)

- ≥ 2× fewer PCG iterations at the default tolerance on all four demos (median over
  frames 2–48) and ≥ 1.5× on the held-out set.
- Net wall-clock win per solve on the gate GPU (inference + fewer iterations < baseline).
- Max-iteration cap never reached where it was not reached before.
- Guess forced off → beta demo hash. Guess on → residual ≤ tolerance on every frame.
- If any line fails: ship with the setting hidden and off; keep the tool and record
  the numbers in `docs/RELEASE_4_0_STATUS.md`.
