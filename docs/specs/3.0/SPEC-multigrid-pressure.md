# SPEC — Multigrid-preconditioned CG for pressure

Owner module: `sim/pressure.py` (`PressureSolver`, `k_apply_preconditioner`,
`_get_pcg_conditional_graphs`, `_record_pcg_iterations`, the "Electrostatic leaf
aggregates" kernels), `sim/fields.py: SparseGrid`.
Depends on: nothing. Can land independently of the other 3.0 specs.

## Trap this spec prevents

1. **`k_sample_down_residual` uses `wp.atomic_add`.** It is marked "used by the V-cycle
   path" and it is non-deterministic. The deterministic bake guarantee (user guide §12,
   gate 3/3) dies the moment it is used per iteration. Restriction must be a **gather**:
   one thread per coarse cell sums its 8 (or 512) fine children in fixed order. Same
   for `k_build_leaf_aggregates`.
2. **A PCG preconditioner must be symmetric positive definite.** A V-cycle is SPD only
   if pre- and post-smoothing use the same smoother with the same count, the smoother is
   symmetric (damped Jacobi is; lexicographic Gauss–Seidel is not), and prolongation is
   the transpose of restriction (piecewise-constant restriction ↔ injection
   prolongation, or full-weighting ↔ trilinear). Violating this makes PCG stall or
   diverge intermittently, which looks like "pressure sometimes fails to converge" and
   is miserable to debug.
3. **No host synchronisation inside the captured loop.** The conditional-graph machinery
   (`k_prepare_pcg_conditional_loop`, `k_finish_pcg_conditional_batch`) is the reason
   viscosity got 5×. The entire V-cycle must be capturable: fixed launch dims (use the
   `num_leaves`-on-device pattern from the viscosity graph-key work), no Python-side
   `if` on device values, no `.numpy()`.
4. **The coarse operator must see solids.** The existing leaf Jacobi treats a leaf as
   fluid if any voxel is fluid and ignores collision fraction. For a *preconditioner*
   that is tolerable (it only has to approximate), but the coarse diagonal must still
   count only fluid neighbours or the preconditioner becomes indefinite near walls.

## Design

Levels, all on the existing sparse leaf layout:

| Level | Cell size | Storage | Smoother |
|---|---|---|---|
| L0 | h | fine unknowns (existing `x`, `r`, `z`) | damped Jacobi ω=0.67, 2 pre / 2 post |
| L1 | 2h | per leaf 4³ = 64 cells → `num_leaves × 64` array | damped Jacobi, 2/2 |
| L2 | 4h | per leaf 2³ = 8 cells | damped Jacobi, 2/2 |
| L3 | 8h | one per leaf (existing `leaf_rhs/leaf_pot` arrays) | Jacobi, 16 sweeps (coarse "solve") |

Intra-leaf levels index as `leaf_slot * 64 + local` so the leaf table is reused and
neighbour lookups across leaves go through `leaf_table` exactly as `k_apply_matrix` does.
Operators at L1–L3 are rediscretised 7-point Laplacians with a per-cell fluid flag =
"any fine child is fluid and not collision"; the diagonal counts fluid neighbours only.

Restriction: piecewise-constant average over fluid children (gather). Prolongation:
injection. They are transposes up to the fluid-count scaling; apply the scaling on the
restriction side so the pair stays symmetric.

Hook point: `k_apply_preconditioner` currently does `z = inv_diag * r`. Add
`PressureSolver.preconditioner: "JACOBI" | "MULTIGRID"` and a `_apply_mg_vcycle(r → z)`
that records into the same CUDA graph the PCG batch is captured in. The graph key gains
the preconditioner kind. Keep JACOBI as the fallback and the gate's reference until the
hash is re-baselined.

Reuse: `coarse_initial_guess` becomes "one V-cycle on the RHS" so there is one coarse
code path, not two.

## Settings
- Domain group: `pressure_preconditioner: Enum` {Jacobi, Multigrid}, default Multigrid
  for new domains; migration keeps Jacobi on 2.x domains (the plan's hash-attribution
  rule). Tier LIVE (no state layout change).
- Expose nothing else. Smoothing counts and ω are constants until a measurement says
  otherwise.

## Acceptance
- Unit (`tests/solver/`): on a 32³ box the preconditioner is symmetric:
  `⟨M r1, r2⟩ == ⟨r1, M r2⟩` to 1e-6 relative, for random `r1, r2`. Add a negative test
  with 1 pre / 3 post smoothing that fails it.
- dam_break Final, tolerance 1e-5: median PCG iterations drop ≥3× vs Jacobi; wall time per
  pressure solve drops ≥1.5× (the V-cycle is heavier per iteration). Report both; gate on
  both.
- Converged pressure fields from Jacobi and Multigrid agree to the solve tolerance
  (relative L2 of the projected velocity ≤ 1e-4), so the demo hash *does* change but the
  visual result does not. Re-baseline deliberately in beta.
- Bitwise determinism 3/3 with Multigrid on; grep the diff for `atomic_add` in
  `pressure.py` — none may remain on any per-iteration path.
- No `.numpy()`, `wp.synchronize`, or host branch inside `_apply_mg_vcycle` (assert via a
  unit test that captures it into a graph and replays twice with identical output).
