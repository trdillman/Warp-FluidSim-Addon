# TASK-REVIEW-004: solve_jacobi_st hardcodes last_residual = -1.0, hiding divergence diagnostics

## State

completed

## Severity

high

## What to build

At lines 1827-1828, `solve_jacobi_st` sets `self.last_residual = -1.0` and `self.last_residual_squared = -1.0` after running a fixed iteration budget, then unconditionally reports `PRESSURE_STATUS_JACOBI` with `pressure_valid=True`. The non-ST variant `solve_jacobi` (lines 2001-2012) was explicitly fixed to compute the real residual via `self.measure_residual(voxel_type)`, and its comment states: 'the previous hardcoded last_residual = -1.0 actively hid whether that fixed budget was actually enough: diagnostics/UI/validation could not distinguish a well-converged Jacobi solve from a wildly divergent one.' The ST variant was never given the same fix. `_set_pressure_status` (line 1691) uses `self.last_residual` to build the status dict, so `_pressure` in solver.py stores a negative residual into `self.stats.pressure_residual`.

**Location:** `fluidSim_warp_blender51/sim/pressure.py:1827`
**Dimension:** Execution Engine

## Impact

In spacetime mode, the UI/diagnostics show a meaningless -1.0 residual for ST Jacobi solves. A wildly divergent pressure field is indistinguishable from a converged one, and the solver always projects it (pressure_valid=True). This can mask numerical instability in ST-FLIP simulations.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/sim/pressure.py:1827` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Port the residual computation from `solve_jacobi` into `solve_jacobi_st`: replace lines 1827-1828 with `residual = self.measure_residual(voxel_type); self.last_residual = residual if math.isfinite(residual) else math.inf; self.last_residual_squared = (residual * residual if math.isfinite(residual) else math.inf)`.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `EXEC-002` in `reviews/swarm-2026-08-08/findings/exec.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (High Findings section).
- Identified by the Execution Engine lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed for 2.0 (2026-09-27). `solve_jacobi_st` in `fluidSim_warp_blender51/sim/pressure.py` now reports the real residual instead of `-1.0`. The card's suggested `measure_residual()` was not used as-is: it applies the constant-coefficient operator, while the ST solve uses the variable-coefficient `k_apply_matrix_st`. A new `_measure_residual_st()` measures `|rhs - A_st * p|` against the operator the solve actually used, with the same single readback and the same non-finite -> `inf` mapping as `solve_jacobi`.

Test: `tests/test_pressure_st_and_graph_cache.py::test_solve_jacobi_st_reports_its_real_residual` (CUDA). Zero iterations give exactly `|rhs|`, 20 iterations must more than halve it, and `last_residual_squared` must match.

## Blocked by

None - can start immediately
