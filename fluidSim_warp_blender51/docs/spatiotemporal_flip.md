# Spatiotemporal FLIP (ST-FLIP) beta

This add-on includes an opt-in beta ST-FLIP path based on Braun, Winchenbach, Bender, and Thuerey, *Spatiotemporal FLIP for Fast Free-Surface and Two-Phase Simulation With Very Large Time Steps*. Phase three will extend reference-error, conservation, and multi-scene validation; default FLIP remains the production path.

## What is implemented

- Per-particle time residual `time_offset = t_global - t_particle`.
- Stochastic staggered-time advection with residual carryover:
  `dt_act = clamp(dt + time_offset + gamma * xi * dt, 0, 2 * dt)` and
  `time_offset = dt + time_offset - dt_act`.
- Adaptive jitter attenuation using `smoothstep(0, 1; local_cfl)`.
- Locally sub-stepped particle advection for high global CFL collision robustness.
- 4D-to-3D P2G deposition with a separable kernel:
  - spatial separable poly6, `(35/32) * max(0, (1-r^2)^3)` per axis;
  - temporal one-sided shifted poly6, `(35/16) * max(0, (1-(tau-1/2)^2)^3)` for `tau <= 1/2`.
- Space-time phase field from P2G weight accumulators using
  `phi = min(sqrt(m / (eta_phi * m0)), 1)`.
- Free-surface pressure activation from the space-time phase field.
- Optional variable-coefficient projection from the same face accumulators.
- Temporary particle re-synchronization for surface SDF construction so output surfaces are not built from time-jittered positions.

## UI controls

Open **FluidSim > Particles > Enable ST-FLIP**.

Important controls:

- **Jitter**: base `gamma`. `1.0` matches the paper default.
- **Adaptive Jitter**: attenuates jitter in calm regions.
- **Phase Eta**: `eta_phi`; lower values sharpen the phase transition.
- **Phase Threshold**: cell phase threshold for free-surface activation.
- **Reference Mass**: `m0`; `0` derives from Particles / Voxel.
- **Variable Projection**: uses phase-derived face coefficients in pressure projection.
- **Local CFL / Max Local Substeps**: particle trajectory substepping inside each global step.
- **Resync Surface**: un-jitters particles for surface extraction only.

## Headless release validation

Run the installed-package NVIDIA GPU gate from the source checkout:

```bash
python release/build_extension.py \
  --blender /absolute/path/to/blender \
  --output dist \
  --gpu-regression-platform linux-x64 \
  --json-report dist/gpu-build-report.json
```

This installs the built archive into an isolated Blender profile, executes all tests registered by `sim.validation`, including `spatiotemporal FLIP jitter, phase, and residual boundedness`, and advances a ten-frame preview simulation.

## Historical version 1.0.231 calibration and acceptance additions

The ST-FLIP phase reference mass `m0` is now calibrated at solver construction when `spacetime_reference_mass <= 0`. The calibration follows the paper's practical prescription: estimate the expected space-time mass at a MAC face under a uniform particle-filled patch, with `tau` uniformly sampled in `[-1/2, 1/2]`, using the exact host equivalents of the Warp spatial and temporal kernels. A positive manual `spacetime_reference_mass` still overrides calibration for experiments.

The historical benchmark evidence is retained in `demo_outputs/stflip_headless_benchmark.json` and checks substep reduction, high-CFL status, ST residual boundedness, phase/weight activation, particle retention, and finite pressure. `demo_outputs/stflip_timeline_viewport_acceptance.json` records the earlier timeline viewport acceptance. These files are excluded from shipping archives; the installed-package GPU gate above is now the release acceptance path.
