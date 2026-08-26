# Thermal State and Stationary Freezing/Melting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development` for each task and `superpowers:verification-before-completion` before every commit. Execute tasks in order; do not start the deferred vapor/gas program from this plan.

**Goal:** Ship the first truthful GPU-resident thermal state transition: mass- and energy-accounted heat transport with reversible, stationary freezing/melting on the existing flat 8x8x8 sparse leaf grid.

**Architecture:** A new bpy-free `ThermalState` deep module owns particle/grid thermal state, matrix-free implicit diffusion, enthalpy closure, and conservation receipts. `FluidSolver.step_frame()` remains the shared orchestrator; Blender, cache, output, admission, and profiling code adapt this interface without owning physics. Frozen material remains represented by the same particles and becomes stationary through phase-aware pressure/advection mechanics; it is not a moving rigid or elastic solid.

**Tech stack:** Python 3.11, Warp kernels on the selected GPU, NumPy only for tiny CPU references/tests and frame-boundary serialization, pytest, Blender 5.1 acceptance scripts.

**Specs:** `research/implementation-seams/RECOMMENDATION.md`, `research/implementation-seams/state-transitions.md`, `research/state-transitions/REPORT.md`.

**Live baseline inspected:** dirty worktree on `codex/fluidsim-1.0.248-merge-prep`; base reference `0d395d78582af1f8876539f0f927110c3eff872d` is historical context only. Re-read live files before execution and preserve unrelated changes.

## Truth contract and scope

- The current `particles.attr("temperature")`, `FluidSolver.temperature`, temperature source controls, `thermal_buoyancy` custom preset, and output coloring are artistic attribute scaffolding. They do not prove advection, diffusion, energy conservation, latent heat, freezing, melting, vapor, or gas pressure.
- Slice F1 in this plan may claim only: **conserved enthalpy transport, implicit heat diffusion, and stationary freezing/melting with a bounded mushy interval** after all F1 gates pass.
- F1 must not claim moving ice, fracture, nucleation, supercooling, boiling, evaporation, condensation, steam, bubbles, or gas pressure.
- Existing artistic `temperature` remains source- and cache-compatible and is labeled **Artistic Temperature**. Physical temperature is a separate derived field named `thermodynamic_temperature_k`.
- No particle is created, deleted, or converted by freezing/melting. Particle mass is immutable in F1; freeze, melt, and refreeze change specific enthalpy and derived liquid fraction only.
- A white particle deletion effect, whitewater emission, hidden particle kill, display-only volume, or deletion-driven “steam” is explicitly rejected. Those paths cannot satisfy any phase-transition acceptance test.
- Disabled-feature parity is mandatory: `thermal_enabled=False` launches no thermal kernels, allocates no thermal arrays, does not alter liquid motion or existing output attributes, and reads legacy caches exactly as before.

## Live production path and affected callers

`FluidSolver.step_frame()` in `fluidSim_warp_blender51/sim/solver.py:2067` currently runs topology/clear, collision, indexing, sourcing, P2G, forces, viscosity, collision, pressure, G2P/advection, lifecycle, relaxation, surface, and whitewater. The live thermal gaps are shared-path, not demo-specific:

- `_clear_grid_fields()` clears `self.temperature` every substep (`solver.py:1350`).
- `_p2g()` transfers velocity and optional viscosity, but not temperature or energy (`solver.py:1520`).
- `_forces()` performs velocity diffusion and independent exponential particle-attribute dissipation; neither is heat conduction (`solver.py:1595`).
- `_pressure()` calls `PressureSolver.project_frame(...)` without phase mobility (`solver.py:1742`).
- `_g2p_advect()` transfers only velocity and always advects live particles (`solver.py:1799`).
- `_lifecycle()` kills lifespan/sink particles and compacts all enabled channels (`solver.py:1848`).

Production callers that must remain coherent are:

1. preview/timeline playback: `blender/playback.py:handle_frame_change()` -> `blender/scene_bridge.py:step_frame()` -> `Session.step()` -> `FluidSolver.step_frame()`;
2. direct/session stepping: operators and scripts -> `scene_bridge.step_frame()`;
3. authoritative sequential bake: `blender/production_bake.py` -> `scene_bridge.step_frame(..., write_cache=False, publish_outputs=False)` -> cache-v2 frame plus checkpoint write;
4. output/cache/profiling siblings: `output_publication` and `io/cache.py:_frame_payload()` share `sim/frame_output.py:HostFrame`; diagnostics call the same solver entry point.

One reproducer is insufficient. Every F1 runtime fix must verify the common seam plus at least preview and production-bake/cache-reopen sibling lanes.

## Exact physical representation

Create these bpy-free frozen dataclasses in `fluidSim_warp_blender51/sim/thermal.py`:

```python
@dataclass(frozen=True)
class ThermalMaterial:
    reference_temperature_k: float
    solidus_temperature_k: float
    liquidus_temperature_k: float
    specific_heat_solid_j_kg_k: float
    specific_heat_liquid_j_kg_k: float
    conductivity_solid_w_m_k: float
    conductivity_liquid_w_m_k: float
    latent_heat_fusion_j_kg: float
    reference_density_kg_m3: float

@dataclass(frozen=True)
class ThermodynamicState:
    temperature_k: float
    liquid_fraction: float
    apparent_heat_capacity_j_kg_k: float

@dataclass(frozen=True)
class ThermalStepResult:
    iterations: int
    relative_residual: float
    particle_mass_kg: float
    sensible_energy_j: float
    latent_energy_j: float
    boundary_energy_j: float
    source_energy_j: float
    transfer_correction_j: float
    stationary_momentum_removed_kg_m_s: float
    finite: bool
```

Public pure functions:

```python
def specific_enthalpy_from_state(
    temperature_k: float, liquid_fraction: float, material: ThermalMaterial
) -> float: ...

def state_from_specific_enthalpy(
    specific_enthalpy_j_kg: float, material: ThermalMaterial
) -> ThermodynamicState: ...
```

Use the following continuous, monotone piecewise closure, with `Ts < Tl`, `cp_m = (cp_s + cp_l) / 2`, `hs = cp_s * (Ts - Tref)`, `c_app = cp_m + Lf / (Tl - Ts)`, and `hl = hs + c_app * (Tl - Ts)`:

- `h < hs`: `T = Tref + h/cp_s`, `f_l = 0`;
- `hs <= h <= hl`: `T = Ts + (h-hs)/c_app`, `f_l = clamp((T-Ts)/(Tl-Ts), 0, 1)`;
- `h > hl`: `T = Tl + (h-hl)/cp_l`, `f_l = 1`.

`specific_enthalpy_j_kg` is authoritative. `thermodynamic_temperature_k` and `liquid_fraction` are derived and may be cached only after `state_from_specific_enthalpy`. Never use the existing `density`, ST-FLIP phase mask, `voxel_type`, or `alive` as thermodynamic phase state.

When F1 is enabled, `ParticleSystem` owns aligned optional float channels:

```text
mass_kg                      extensive, immutable during F1
specific_enthalpy_j_kg       authoritative intensive state
thermodynamic_temperature_k  derived cache/output
liquid_fraction              derived cache/mechanics/output, [0, 1]
```

The grid owns `thermal_mass_kg`, `enthalpy_sum_j`, `specific_enthalpy_j_kg`, `temperature_k`, `liquid_fraction`, `conductivity_w_m_k`, `heat_capacity_j_k`, `cell_active`, and matrix-free PCG/Picard scratch. All arrays remain on `solver.device` through a substep.

## Sparse implicit heat solve

`ThermalState.advance(particles, velocity, voxel_type, dt) -> ThermalStepResult` performs:

1. mass-weighted P2G: atomically accumulate `m_i` and `m_i h_i`; divide only where thermal mass is positive;
2. close `h -> (T, f_l, c_app, k)` on-grid;
3. solve backward Euler with a six-neighbor finite-volume stencil and harmonic face conductivity:
   `(C_i + dt*sum(G_ij))*T_i - dt*sum(G_ij*T_j) = C_i*T_i_old + dt*Q_i + boundary_rhs`, where `C_i = m_i*c_app_i` and `G_ij = k_face*dx` for cubic cells;
4. run Picard iterations until both temperature relative residual and total-energy residual meet tolerance, then convert solved `T` through the exact closure back to grid enthalpy;
5. G2P interpolate specific enthalpy, then apply a mass-weighted per-leaf correction so grid energy equals particle energy within tolerance; recompute particle temperature/fraction from corrected enthalpy;
6. reduce the `ThermalStepResult` ledger on-device and read it only at the existing frame boundary.

Create `fluidSim_warp_blender51/sim/kernels_thermal.py` for kernels and `fluidSim_warp_blender51/sim/thermal.py` for ownership/orchestration. Stencils must use `fields.voxel_flat_index()` and the existing `leaf_table`/`leaf_coords`; no hierarchical tree or dense-domain fallback is permitted.

Default inactive/unallocated-neighbor behavior is explicit `ADIABATIC`. Supported F1 boundaries are `ADIABATIC` and `FIXED_TEMPERATURE`; a missing neighbor may never silently mean zero kelvin. `FluidSolver._rebuild_topology()` must retain particle-occupied frozen leaves and request one leaf of thermal support when F1 is enabled. `simulation_admission` must charge that halo before replacement cutover.

## Stationary freezing mechanics

Create face mobility from adjacent liquid fractions:

`M_face = min(f_l_left, f_l_right)^3 / (min(f_l_left, f_l_right)^3 + mushy_drag*(1-min(f_l_left, f_l_right))^2 + eps)`.

Modify `PressureSolver.project_frame(...)` with the backward-compatible keyword-only interface:

```python
def project_frame(
    self, velocity, voxel_type, collision_velocity, voxel_counts, settings,
    *, max_gradient, reference_mass=None, pressure_source=None,
    face_mobility: tuple[wp.array, wp.array, wp.array] | None = None,
) -> dict: ...
```

When `face_mobility is None`, retain byte-for-byte launch selection and existing numerical behavior. When present, the matrix and projection use `div(M grad q)` and `u_new = M*(u_star - grad(q))`; a zero-mobility face is stationary. After G2P, particles with `liquid_fraction <= frozen_fraction_epsilon` retain position and zero velocity; mushy particles advect with phase-limited velocity. Melting reuses the retained particle and its exact mass/enthalpy. Do not emulate solidification with extreme viscosity, particle deletion, collision bits, or a hidden animated mesh.

## File map

**Create:**

- `fluidSim_warp_blender51/sim/thermal.py` — material/closure/state/ledger and substep owner.
- `fluidSim_warp_blender51/sim/kernels_thermal.py` — P2G/G2P, closure, stencil, PCG/Picard, mobility, reductions.
- `tests/test_thermal_state.py` — units, closure, allocation, disabled parity.
- `tests/test_thermal_transport.py` — sparse diffusion/reference/convergence.
- `tests/test_phase_change.py` — latent plateau, Stefan, freeze-thaw, stationary mechanics.
- `tests/blender_state_transition_acceptance.py` — Blender adapter, playback, output, cache/reopen runtime gate.
- `tools/profile_thermal_state.py` — repeatable enabled/disabled GPU profile and JSON budget verdict.

**Modify:**

- `sim/particles.py` — optional thermal float channels, resize/compact/checkpoint alignment.
- `sim/sourcing.py`, `sim/source_regions.py`, `blender/sources.py`, `blender/scene_inputs.py` — initialize mass/enthalpy from owned sources without reinterpreting artistic temperature.
- `sim/solver.py` — settings, conditional allocation, topology support, phase order, pressure mobility, stationary advection, stats.
- `sim/pressure.py` and its kernel module(s) — optional face-mobility operator/projection.
- `sim/memory_safety.py`, `sim/capacity_planner.py`, `blender/simulation_admission.py` — thermal/halo peak accounting and cutover cleanup.
- `sim/constants.py`, `diagnostics/profiler.py`, `tests/test_profiler_phase_contract.py` — `thermal_p2g`, `thermal_solve`, and `phase_state` timing slots.
- `io/checkpoint.py`, `io/cache.py`, `io/cache_schema.py` — checkpoint migration and physical-state metadata/payload.
- `sim/frame_output.py`, `sim/attributes.py`, `blender/output_publication.py` — aligned physical temperature/fraction attributes and ledger diagnostics.
- `properties.py`, `blender/solver_settings.py`, `blender/pending_settings.py`, `panels.py` — authored material/boundary settings, exhaustive lifecycle ownership, honest labels.
- Existing focused tests listed in the tasks below; never weaken their assertions.

## TDD task sequence and incremental commits

### Task 1: Freeze the fidelity and closure contracts

- [ ] Add RED tests in `tests/test_thermal_state.py` for invalid SI material parameters, monotone closure, exact solid/mushy/liquid inversions, `[0,1]` fraction bounds, and `1e-5` relative round-trip error away from endpoints.
- [ ] Add a RED source-contract test proving artistic `temperature` and physical `thermodynamic_temperature_k` are distinct names.
- [ ] Run `rtk proxy python -m pytest -q tests/test_thermal_state.py`; expect collection/import failure for missing `sim.thermal`, then behavioral failures until the closure is implemented. A pass caused only by string/symbol assertions is invalid.
- [ ] Create `sim/thermal.py` with the exact dataclasses/functions above and no Warp allocation yet; make the focused tests GREEN.
- [ ] Commit: `feat(thermal): define enthalpy and phase closure`.

### Task 2: Add conditional aligned particle thermal state

- [ ] RED: extend `tests/test_particle_capacity_growth.py`, `tests/test_source_emission_lifecycle.py`, and `tests/test_point_source_pipeline.py` to prove thermal channels compact/grow/source together, mass is positive and stable, and disabled solvers allocate none.
- [ ] RED command: `rtk proxy python -m pytest -q tests/test_thermal_state.py tests/test_particle_capacity_growth.py tests/test_source_emission_lifecycle.py tests/test_point_source_pipeline.py`; expect missing optional-float-channel/injection support.
- [ ] In `particles.py`, add `optional_float_channels=()` to `ParticleSystem.__init__`, a generic float scatter kernel, and aligned resize/compact/checkpoint loops. Keep the five existing `PHYSICAL_ATTR_NAMES` unchanged.
- [ ] Source interfaces consume physical `initial_temperature_k` only when thermal is enabled; compute initial `specific_enthalpy_j_kg` through the closure. Compute `mass_kg = reference_density_kg_m3 * rest_particle_volume_m3` from resolved quality, never from the existing artistic `density` attribute.
- [ ] GREEN the focused tests and prove legacy source behavior is unchanged with F1 off.
- [ ] Commit: `feat(thermal): add conservative particle energy state`.

### Task 3: Build the sparse implicit thermal solver

- [ ] RED in `tests/test_thermal_transport.py`: constant-field preservation; two-cell heat exchange; adiabatic total-energy closure; fixed-temperature boundary energy receipt; equal result for an interior stencil and the same stencil split across a leaf face; CPU-reference equality on a tiny domain; monotonic error reduction at three voxel sizes.
- [ ] RED in `tests/test_sparse_leaf_grid_contract.py`: fixed `8x8x8`/512 addressing, one-leaf thermal support, explicit missing-neighbor boundary policy, and no dense/hierarchical allocation.
- [ ] Run `rtk proxy python -m pytest -q tests/test_thermal_transport.py tests/test_sparse_leaf_grid_contract.py`; expect missing `ThermalState`/kernels and cross-leaf failures before implementation.
- [ ] Implement `kernels_thermal.py` and `ThermalState.allocate(grid, particle_capacity, settings, device) -> ThermalState`; matrix-free PCG uses diagonal preconditioning and reports residual/iteration count. Preview defaults: tolerance `1e-4`, 30 PCG iterations, 3 Picard iterations. Production defaults: tolerance `1e-6`, 80 PCG iterations, 8 Picard iterations. Failure to converge raises `ThermalConvergenceError`; it must not silently accept the last iterate.
- [ ] Implement conservative G2P correction and require total energy error <= `1e-5` for deterministic tiny cases and <= `0.1%` for atomic production cases.
- [ ] GREEN both suites.
- [ ] Commit: `feat(thermal): solve heat on sparse leaf grid`.

### Task 4: Integrate the shared solver phase

- [ ] RED in `tests/test_thermal_state.py` and `tests/test_profiler_phase_contract.py`: exact F1 order is source/index -> velocity P2G -> `thermal_p2g` -> forces -> `thermal_solve`/phase closure -> viscosity -> collision -> phase-aware pressure -> G2P/stationary advection. Assert disabled mode has zero thermal launches, arrays, timers, and output changes.
- [ ] Run focused RED; expect missing `thermal_p2g`, `thermal_solve`, and `phase_state` phases.
- [ ] In `FluidSolver.__init__`, allocate `self.thermal` only when `settings.thermal_enabled`; in `_rebuild_topology` retain thermal support; in `step_frame` call the new phases at the asserted order. Do not route physical heat through snippets or `_forces()` dissipation.
- [ ] Add `FrameStats` thermal ledger/convergence fields and profiler slots. Do not add a substep `.numpy()`, `wp.synchronize_device`, or host reduction.
- [ ] GREEN focused tests plus `rtk proxy python -m pytest -q tests/test_particle_grid_transfer.py tests/test_profiler_phase_contract.py`.
- [ ] Commit: `feat(thermal): integrate conserved heat substeps`.

### Task 5: Make frozen material stationary and melting reversible

- [ ] RED in `tests/test_phase_change.py`: fully frozen quiescent particles do not translate under gravity/pressure; liquid contrasting path still moves; mushy velocity decreases monotonically with fraction; freeze-thaw preserves particle count/UID/mass; refreeze returns stationary without energy creation.
- [ ] RED in `tests/test_pressure_projection_regressions.py`: optional mobility makes solid faces impermeable, reduces divergence in liquid cells, and `face_mobility=None` reproduces the existing projection path.
- [ ] Run both suites; expect motion through frozen cells and missing mobility parameters.
- [ ] Implement the exact optional `PressureSolver.project_frame` signature, face operator, and stationary G2P/advection rule. Accumulate removed momentum as a diagnostic; do not add it to the thermal-energy ledger.
- [ ] GREEN both suites and existing viscosity/ST-FLIP projection regressions.
- [ ] Commit: `feat(thermal): add stationary freeze melt mechanics`.

### Task 6: Prove latent heat and conservation

- [ ] RED in `tests/test_phase_change.py`: latent plateau; one-phase Stefan front at three resolutions with monotonic convergence and finest error below one cell; closed adiabatic mass/energy drift below `0.1%`; heat/cool reversible cycle mass and energy drift below `0.1%`; hysteresis (if later enabled) changes onset only and never the ledger.
- [ ] Add contrasting-path tests with latent heat zero and thermal disabled. Expected RED is wrong plateau/front/ledger behavior, not missing labels.
- [ ] Run `rtk proxy python -m pytest -q tests/test_phase_change.py tests/test_thermal_transport.py`; make only numerical/core changes needed for GREEN. Do not tune a demo around a failing reduced case.
- [ ] Commit: `test(thermal): gate Stefan and reversible conservation`.

### Task 7: Wire admission, persistence, UI, and output

- [ ] RED admission/memory tests: thermal estimate charges exactly `32 bytes * particle_capacity + 64 bytes * voxel_capacity + reduction/halo overhead`; measured allocations must be <= estimate and within 5% after fixed allocator overhead is excluded; refusal leaves the old solver alive; candidate failure cleans thermal allocations.
- [ ] RED cache tests: bump `CHECKPOINT_FORMAT_VERSION` from 1 to 2; cache-v2 manifest records the checkpoint version and `thermal_schema_version=1`; round-trip mass, enthalpy, material, derived fields, ledger, and slot-ordered topology. Old checkpoints load only with thermal disabled; attempting thermal resume from missing state fails with “reinitialize/regenerate cache” rather than inventing enthalpy.
- [ ] RED adapter tests: `thermal_enabled` and all material/boundary settings are present in `blender/solver_settings.py` and the exhaustive `pending_settings.py` registry as reinitialize-required; panel copy says “Stationary Freezing/Melting”; source control says “Physical Initial Temperature (K)”; legacy control says “Artistic Temperature.”
- [ ] RED output tests: `HostFrame` returns aligned `thermodynamic_temperature_k` and `liquid_fraction`; output publication/cache preserve dtype/length; no sync occurs until HostFrame/frame boundary.
- [ ] Run `rtk proxy python -m pytest -q tests/test_memory_safety_recovery.py tests/test_simulation_admission.py tests/test_cache_io.py tests/test_frame_output.py tests/test_pending_settings_registry.py tests/test_operator_panel_behavior.py`; expect schema/registry/attribute failures.
- [ ] Implement adapters only after RED. Keep `simulation_admission.py`, `pending_settings.py`, `output_publication.py`, and `production_bake.py` ownership boundaries intact.
- [ ] GREEN focused tests, then commit: `feat(thermal): persist and expose stationary phase state`.

### Task 8: Runtime acceptance and performance budget

- [ ] Create `tests/blender_state_transition_acceptance.py` with two live lanes: preview handler advancement and authoritative production bake/cache reopen. Each reopens at `frame_start`, initializes non-empty owned particles, crosses the latent interval, publishes changing physical temperature/fraction, and verifies mass/energy receipts.
- [ ] RED run: `blender --background --factory-startup --python tests/blender_state_transition_acceptance.py`; expect `STATE-TRANSITION-ACCEPTANCE-FAIL` before the adapter/core is complete. Final marker is `STATE-TRANSITION-ACCEPTANCE-PASS` only after both lanes pass.
- [ ] Create `tools/profile_thermal_state.py`. On the same GPU/settings, compare 30 measured frames after 5 warmups. Budgets: disabled F1 median step time within 2% of baseline and zero thermal allocations/launches; enabled F1 has no substep host round-trip, no convergence failures, peak allocation <= admission estimate, and median total step time <= 1.5x disabled. Report thermal phase milliseconds, PCG/Picard iterations, active/halo leaves, particles, VRAM, and ledger drift. A budget miss is RED, not a warning.
- [ ] Run the profile through Blender 5.1 bundled Python and the repository GPU profile command from `AGENTS.md`.
- [ ] Run final focused gate:

  ```powershell
  rtk proxy python -m pytest -q tests/test_thermal_state.py tests/test_thermal_transport.py tests/test_phase_change.py tests/test_sparse_leaf_grid_contract.py tests/test_pressure_projection_regressions.py tests/test_memory_safety_recovery.py tests/test_simulation_admission.py tests/test_cache_io.py tests/test_frame_output.py tests/test_pending_settings_registry.py tests/test_scene_bridge_cache_playback.py tests/test_profiler_phase_contract.py
  blender --background --factory-startup --python tests/blender_state_transition_acceptance.py
  rtk proxy python tools/run_production_gate.py
  rtk proxy python -m pytest -q
  ```

- [ ] After a GUI-inclusive gate/report refresh, run `rtk proxy python tools/validate_policy_drift_reports.py` before release claims.
- [ ] Commit: `test(thermal): accept stationary freezing melting`.

## Deferred programs: not part of F1

### F2 — evaporation/condensation with passive vapor

Requires separate approval and plan. It may create `sim/phase_conversion.py`, `sim/kernels_phase_conversion.py`, and `sim/gas.py` only after immutable/variable particle mass, residual conversion reservoirs, thermal interface area, and F1 ledgers are proven. The required interface is a single transaction:

```python
PhaseConversion.apply(
    liquid, thermal, passive_vapor, interface, dt
) -> ConversionLedger
```

RED must prove liquid mass loss equals vapor gain plus declared open-boundary loss; latent debit equals converted mass times `L_v`; insufficient capacity carries a residual rather than deleting mass; remote condensation transports and deposits vapor before claiming condensation. A display-only “steam” volume may exist only as explicitly non-conservative artistic output and cannot close F2.

### F3 — low-Mach coupled gas

Requires a separate `GasState` with sparse gas density/species/temperature, MAC velocity, pressure, and phase-change divergence source. Only F3 may claim pressure-driven vapor expansion, boiling, bubbles, or two-way steam coupling. RED must include pressure/divergence closure and a classical bubble-growth convergence benchmark. Passive vapor, whitewater, deletion, and an incompressible liquid projection without volume source are rejected substitutes.

### Later research

Moving frozen bodies need rigid clusters, elasticity, or MPM coupling; nucleation/supercooling need a separately parameterized model. Neither is cleanup for F1-F3.

## Rollback boundaries

- Each task is one independently reviewable commit. Revert only the latest failing task; never reset or discard unrelated dirty work.
- Tasks 1-3 are bpy-free and can be rolled back without cache/UI migration.
- Task 4 is the first shared runtime seam; keep it feature-gated so reverting removes all thermal launches.
- Task 5 changes pressure/advection; merge only with `face_mobility=None` parity proof.
- Task 7 is the persistence boundary. Once checkpoint v2 thermal caches are published, do not silently downgrade; retain explicit unsupported-version errors.
- Do not merge any F2/F3 scaffold into F1 merely because a class, label, or output proxy exists.

## Final review checklist

- [ ] Every named existing path/symbol is revalidated against the live worktree; every new path above is marked Create.
- [ ] Source behavior, runtime behavior, convergence/conservation evidence, then Blender review evidence are all present in that order.
- [ ] Preview and production-bake/cache-reopen callers pass; one demo or one direct solver test is not closure.
- [ ] No UI label, screenshot, cache key, or non-crash test is used as proof of phase physics.
- [ ] No host readback/synchronization exists inside thermal substeps.
- [ ] Disabled F1 is allocation-, launch-, motion-, cache-, and output-compatible.
- [ ] F1 claims remain stationary freezing/melting only; vapor/gas/moving-solid claims remain deferred.
