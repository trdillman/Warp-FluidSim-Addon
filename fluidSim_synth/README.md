# Synthesized FluidSim Contracts

`fluidSim_synth` is the distilled fidelity contract for FluidSim 1.0.226. It records enums, kernel inventory, dispatch and custom-kernel contracts, UI behavior, profiling expectations, and validation targets used to guide and check the Blender/Warp port.

This package is validation and reference input. It is not imported by the production Blender add-on and is not a second simulation runtime.

Current direct consumers include:

- `tests/test_kernel_coverage.py`, which imports the synthesized kernel manifest;
- `tests/test_contract_wording.py`, which checks wording in the synthesized validation plan.

These contracts preserve source fidelity, but they are historical evidence. When a contract disagrees with current executable behavior, live source, runtime output, and current tests are authoritative; investigate and reconcile the stale contract rather than treating it as runtime truth.
