# Lane: Test Coverage
- **Dimension name:** Test Coverage
- **ID prefix:** TEST
- **Output path:** C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port/reviews/swarm-2026-08-08/findings/tests.json

Follow `_common.md` in this same directory.

## Scope
- tests/** (all ~104 test files)
- tools/** test runners and harnesses (e.g., run_production_gate.py, blender_*_acceptance.py, headless_stflip_benchmark.py)
- Cross-reference against source modules in fluidSim_warp_blender51/ to map coverage.

## Dimension Focus
- Coverage gaps: which source modules (sim/solver, pressure, kernels, sourcing, capacity_planner, memory_safety, scene_bridge, cache, operators, panels) have NO or WEAK tests? Name them.
- Test quality: do tests assert real behavioral outcomes, or merely "it ran without exception"? Flag shallow/tautological assertions.
- Missing negative/failure-path tests: are error paths, capacity overflow, NaN propagation, invalid inputs, and fail-closed memory paths tested?
- Isolation & flakiness: shared mutable global state, test-order dependence, reliance on GPU/hardware (are GPU/warp paths stubbed or untested?).
- Mock/stub correctness: are mocks hiding real bugs (e.g., mocking the very thing under test)?
