# State-transition implementation seams

## Executive recommendation

Treat the current `temperature` channels as reusable storage and presentation scaffolding, not thermodynamics. The smallest truthful progression is:

1. preserve today's scalar as **artistic temperature** and add explicit temperature transport plus optional Boussinesq lift;
2. introduce a deep `sim/thermal.py` module whose small interface advances conserved liquid enthalpy on the existing flat sparse leaf grid, derives temperature and liquid fraction, and applies a stationary porous/velocity-lock freezing model;
3. add one atomic phase-conversion transaction that exchanges liquid particle mass, enthalpy, and vapor species with a conservation ledger;
4. initially advect vapor passively on a separate sparse gas state; add remote condensation only after that state has transport and deposition;
5. claim coupled boiling/bubble growth only after a low-Mach gas velocity/pressure/divergence solve exists;
6. defer moving frozen bodies/MPM and nucleation. They are separate solvers, not refinements of viscosity or whitewater.

This agrees with `research/state-transitions/REPORT.md:12-19` and `research/SYNTHESIS.md:15-24`, but makes the live seam explicit: the solver's substep orchestration is the shared caller, while thermal state and conversion accounting belong behind new deep modules. `scene_bridge.py` must remain only the Solver Session adapter; Blender properties, cache, output, and admission adapt the new interfaces rather than owning physics.

## Current truth

### Real core behavior

- Sources carry per-particle temperature, viscosity, and density (`sim/sourcing.py:95-125`, `129-158`). Analytic and mesh particle emitters write all three (`780-817`, `943-981`), and direct injection does the same (`1097-1139`).
- Particle storage includes `temperature` in the always-present physical attributes (`sim/particles.py:16-18`). Compaction explicitly scatters temperature with age, life, viscosity, and density (`50-80`), so deletion preserves alignment. Capacity growth clones particle state transactionally through `ParticleSystem`/`ParticleBins`; it does not add or remove mass semantics.
- The solver allocates sparse grid `temperature`, `viscosity`, and `density` fields (`sim/solver.py:710-722`). The storage contract is a flat leaf table with 8x8x8 leaves and 512 scalar entries per leaf (`sim/constants.py:47-52`; `sim/fields.py:1-10,74-94,469-475`).
- Temperature can be stamped after P2G by compiled scalar field plans (`sim/sourcing.py:1749-1829`), accessed by custom kernels (`sim/custom.py:40-44,881-910`), exponentially moved toward a target on particles (`sim/solver.py:1660-1667`; `sim/kernels_fluid.py:1386-1407`), cached, and published as a Blender particle attribute (`io/cache.py:227-269,333-380`; `sim/frame_output.py:199-240`; `blender/output_publication.py:1-5,736-748`).
- The UI truthfully describes temperature as an emitted particle attribute used by visualization/field operations (`properties.py:663-665`), exposes dissipation (`889-908`), and offers temperature coloring (`65-70,1218-1222`; `panels.py:696-704,1065-1069`).

### Reusable scaffolding, not physics

- The grid temperature allocation and custom-kernel binding are useful GPU-resident scalar infrastructure.
- Sparse neighbor lookup and topology expansion are suitable for thermal stencils, provided thermal support leaves and halos are made explicit (`sim/fields.py:74-94,357-390,529-540`).
- Existing source compilation, cache schema validation, HostFrame lazy readback, output publication, Pending Settings, and Simulation Admission are adapter seams to extend.
- The `thermal_buoyancy` snippet is an artistic particle-velocity edit, not a grid energy or gas buoyancy model (`sim/custom.py:671-686`). The age-to-temperature preset is explicitly visualization (`691-701`).
- Whitewater is churn-based secondary-particle output driven by liquid velocity/SDF (`sim/solver.py:1894-1950`); it is not vapor mass, steam, or bubbles.

### Absent physics, proven by production caller tracing

The production substep is topology/clear -> collision -> index/source -> P2G -> forces -> viscosity -> pressure -> G2P/advection -> lifecycle/compaction -> surface/whitewater (`sim/solver.py:2067-2155`). Within that path:

- `_clear_grid_fields()` zeros grid temperature every substep (`1350-1363`).
- `_p2g()` transfers particle velocity and optionally viscosity; it never reads particle temperature and never writes grid temperature (`1520-1587`). `_g2p_advect()` transfers only velocity (`1799-1845`). Thus there is no standard temperature advection cycle.
- `force_diffusion` smooths MAC velocity, not temperature (`1646-1659`). Temperature "dissipation" is independent exponential attribute relaxation, not conduction or an energy sink ledger (`1660-1667`).
- Pressure uses folded constant-density units `q = p*dt/rho`; its ordinary path receives velocity, voxel type, collision velocity and particle counts, not thermal state (`sim/pressure.py:4-12`; `sim/solver.py:1742-1758`). ST-FLIP's phase/density interpolation is a projection variant, not thermodynamic phase fraction or material density.
- Lifecycle kills particles by lifespan/sink and compacts them (`1848-1865`); no removed mass or energy is transferred to another species. Reseeding/top-up emission is occupancy-driven and assigns authored attributes, not conserved particle mass.

Repository-wide symbol/test inspection finds no production energy equation, heat capacity, conductivity, enthalpy, latent heat, thermodynamic phase fraction, vapor mass/species, gas velocity/pressure, thermal divergence source, solidification mechanics, condensation, boiling closure, or nucleation model. Existing tests prove attribute sourcing/alignment, custom writes, cache schema, and UI presentation (for example `tests/test_mesh_source_pipeline.py:132-170`, `tests/test_frame_output.py:22-25,234-245`, `tests/blender_background_acceptance.py:288-301`), not those absent claims.

## Caller/ownership map

```text
Blender properties/panels
  -> pending_settings.py (setting lifecycle)
  -> simulation_admission.py (snapshot, memory/capacity preflight, cutover)
  -> solver_settings.py / scene_inputs.py (adapters)
       -> FluidSolver.step_frame() shared production caller
            -> ThermalState.advance(...)          [new sim/thermal.py]
            -> PhaseConversion.apply(...)         [new sim/phase_conversion.py]
            -> GasState.advance/couple(...)       [later sim/gas.py]
            -> existing pressure/G2P/lifecycle/surface
       -> HostFrame/cache/output adapters
            -> cache_bake.py authoritative sequence
            -> playback.py non-authoritative presentation
            -> output_publication.py persistent Blender IDs
```

Affected production callers are interactive preview through `scene_bridge.py`/`playback.py`, sequential production bake through `cache_bake.py`, direct solver tests/profiles, saved-demo playback, checkpoint/cache export, and output publication. This is a **shared-path** design; a demo-only or snippet-only implementation would not close it.

Ownership decisions:

- `sim/thermal.py` owns the thermal-state interface, state invariants, sparse diffusion, particle/grid transfer, temperature/enthalpy conversion, and thermal ledger. This produces leverage across preview, bake, tests, and future gas coupling, with locality for numerical changes.
- `sim/phase_conversion.py` owns one transaction interface: consume pre-conversion liquid/gas/thermal state and return device-resident deltas plus ledger totals. It alone may create/delete phase mass.
- `sim/gas.py` owns vapor species, gas velocity/pressure and later low-Mach coupling. Passive-vapor and coupled-gas implementations are two real adapters at this seam.
- `sim/solver.py` orders these deep modules but must not reproduce their equations.
- Blender modules remain adapters under their documented ownership. `simulation_admission.py` budgets all new resident/scratch arrays; `pending_settings.py` owns setting transitions; `output_publication.py` owns persistent output; `cache_bake.py` owns authoritative cache generation; `playback.py` owns preview policy; `scene_bridge.py` owns the Solver Session only.

## Claim matrix

| Claim | Current implementation/evidence | Status | Minimum evidence before claim |
|---|---|---:|---|
| Temperature attribute authoring/display | Sourced, dissipated, custom-writable, cached, published | conformant | Existing source/cache/UI tests |
| Temperature transport | No standard temperature P2G/G2P; grid reset each substep | missing | Manufactured advection + diffusion convergence and sparse-leaf parity |
| Thermal buoyancy | Optional custom particle lift only | proxy | Boussinesq force from transported grid temperature with reference-state test |
| Freezing/melting | No enthalpy, liquid fraction, or velocity lock | missing | Stefan convergence, latent plateau, reversible-cycle ledger |
| Artistic steam | Whitewater/material/custom output can look vapor-like | proxy | Label explicitly "artistic steam"; visual/runtime output test only |
| Conservative evaporation | Particle deletion/emission has no phase mass/energy transfer | missing | Interface-flux convergence; liquid+vapor mass and enthalpy closure |
| Remote condensation | No transported vapor species | missing | Vapor advection/diffusion plus remote deposition and conservation test |
| Coupled boiling | No gas momentum/pressure or divergence source | missing | Low-Mach gas coupling and bubble-growth reference case |
| Moving frozen solids | No rigid/elastic/MPM solid state | missing | Separate solid solver, coupling, momentum/energy tests |
| Nucleation | No site/barrier/stochastic model | missing | Validated nucleation law, resolution/rate study, seeded reproducibility policy |

## Deep seam design

Recommended small interfaces (illustrative, not a requirement to preserve these Python spellings):

```python
thermal = ThermalState.allocate(grid, particle_capacity, settings, device)
thermal.advance(liquid, velocity, voxel_type, dt) -> ThermalStepResult
conversion.apply(liquid, thermal, gas, interface, dt) -> ConversionLedger
gas.advance(thermal, conversion, velocity, voxel_type, dt) -> GasStepResult
```

The interface includes invariants: all arrays remain on the selected Warp device; topology is the existing flat 8x8x8 leaf table; conversions are conservative transactions; derived temperature/phase fraction are not independently authoritative; every stencil declares support/halo needs; inactive/unallocated neighbors have an explicit boundary condition; and results report finite-value, mass, sensible-energy, latent-energy, and source/sink totals.

Suggested state:

- liquid particles: `specific_enthalpy` (authoritative), optional cached/derived `temperature`; retain current temperature as an artistic channel until migration is explicit;
- liquid grid: `enthalpy_sum`, `thermal_weight`, `enthalpy`, `temperature`, `liquid_fraction`, `thermal_rhs`, `thermal_diag`, solver scratch;
- stationary freezing: `porosity`/liquid fraction and a velocity-lock coefficient derived from it; do not overload viscosity or voxel activity;
- interface conversion: per-cell `mass_flux`, `energy_flux`, and bounded particle emission/removal requests with stable conversion IDs;
- passive gas: vapor mass/species and temperature on its own leaf support; coupled tier later adds gas MAC velocity, pressure, divergence source, density/state closure.

Collisions to prevent:

- Do not repurpose `density`: in ordinary FLIP it is mostly attribute/scaffolding and in ST-FLIP `self.density` stores a projection phase mask (`solver.py:1552-1562`). Thermodynamic density needs a distinct name and contract.
- Do not use `VOXEL_ACTIVE` or particle `alive` as phase fraction; those drive topology/lifecycle and compaction.
- Do not silently change occupancy top-up into physical mass. Define particle volume/mass and make reseeding transfer conservative intensive/extensive quantities.
- Diffusion stencils crossing leaf faces require neighbor lookup and at least one valid thermal halo. Thermal-only leaves may outlive liquid occupancy, especially for frozen regions and remote vapor.
- Pressure projection cannot represent vapor expansion until conversion provides a divergence source and the selected pressure/gas adapter consumes it.
- Cache/output schemas must distinguish authored artistic temperature, physical temperature, enthalpy, phase fraction and vapor mass; schema versioning and reopen migration are required.

## Technique/fidelity decision

1. **Tier A — artistic thermal field.** Add conservative scalar P2G/G2P and optional diffusion/buoyancy, but no latent heat or phase mass. Truthful claim: "advected temperature-like field with thermal buoyancy/art direction." Never claim energy conservation, freezing, steam, or boiling.
2. **Tier B — stationary freezing/melting.** Use apparent heat capacity or enthalpy with an implicit sparse heat solve; derive liquid fraction over a mushy interval; suppress/lock liquid velocity as fraction approaches zero. Truthful claim: "energy-accounted melting/freezing with stationary frozen regions." It is not a moving solid and viscosity is not the solid model.
3. **Tier C1 — conservative interface conversion + passive vapor.** Compute bounded interfacial mass/energy flux, transact liquid particle mass and vapor species, advect/diffuse vapor without vapor feedback on liquid pressure. Truthful claim: "mass/energy-accounted evaporation and passive vapor; local or remote condensation" only when the vapor transport/deposition test passes. Artistic rendering remains an adapter.
4. **Tier C2 — low-Mach coupled gas.** Add gas momentum/pressure and phase-change divergence/volume source, two-way interface forces and saturation closure. Truthful claim: "coupled boiling/condensation with pressure-driven bubble growth" after reference validation.
5. **Later research.** Moving frozen solids need rigid/elastic/MPM state and coupling. Nucleation needs a separate physically parameterized model. Neither belongs in the first four slices.

Implicit diffusion is preferred because thermal time scales should not force tiny explicit substeps. An apparent-heat-capacity formulation is a pragmatic first stationary-freezing adapter; conserved enthalpy remains authoritative so the latent plateau and ledger are testable. Sharp-interface boiling and compressible steam are rejected for the interactive first implementation because they expand topology, pressure, and stability scope simultaneously.

## Ordered implementation plan

1. **Clarify existing semantics.** Document/migrate current `temperature` to `artistic_temperature` or a compatibility alias. Add settings enums that make ARTISTIC versus PHYSICAL explicit. Update `properties.py`, `panels.py`, `solver_settings.py`, `pending_settings.py`, scene source adapters and tests.
2. **Thermal storage/admission.** Add `sim/thermal.py`, fields/scratch, memory estimation, allocation and cleanup. Extend `simulation_admission.py` peak-residency planning and transactional cutover. Extend particle growth/compaction for enthalpy without breaking attribute alignment.
3. **Transport slice.** Add enthalpy/weight P2G, normalization, implicit sparse diffusion with declared halo support, and enthalpy G2P. Order it after sourcing and before temperature-dependent forces/viscosity. Add Boussinesq force only from physical temperature.
4. **Stationary phase slice.** Add material properties, enthalpy-to-temperature/liquid-fraction closure, porous drag/velocity lock before pressure, and phase-aware sourcing/reseeding. Do not emit/remove mass in this slice.
5. **Schema/output slice.** Version cache/checkpoint/export for enthalpy, physical temperature and liquid fraction; teach HostFrame and Output Publication adapters; preserve old-cache error/migration behavior. Ensure `cache_bake.py` stores authoritative state and playback only presents it.
6. **Conversion transaction.** Add interface reconstruction input, bounded mass/energy flux kernels, particle mass/volume rules, emission/removal requests, and device ledger reductions. Any lifecycle deletion caused by conversion must pass through this module.
7. **Passive vapor/condensation.** Add separate vapor topology/state and transport, remote condensation transaction, cache/output channels, admission cost and artistic volume publication.
8. **Coupled gas.** Add low-Mach gas MAC velocity/pressure, equation of state/saturation closure, divergence source, two-way pressure/interface coupling, and bubble diagnostics.
9. **Separate later projects.** Moving solid/MPM coupling, then nucleation, each behind a new approved design and validation plan.

## Tests/acceptance

Small coherent vertical slices and measurable gates:

- **Thermal transport:** constant-field preservation; manufactured linear/advection case; 1-D heat diffusion convergence under voxel refinement; CPU reference on a tiny domain; equal result when the same domain is split across leaf boundaries; no host round-trips in substeps.
- **Stationary freezing:** one-phase Stefan front-position convergence; temperature stalls across the latent interval; total sensible+latent energy closes against boundary/source work; freeze-thaw cycle returns mass and enthalpy within declared tolerance; velocity tends to zero with solid fraction without using huge viscosity.
- **Sparse contract:** same solution for interior versus cross-leaf stencil; thermal halo allocation/deallocation; fixed 8x8x8/512 layout; inactive neighbor boundary cases; admission estimate matches measured allocations.
- **Conversion:** planar evaporation flux convergence; liquid mass loss equals vapor gain plus declared boundary loss; latent energy debit equals phase mass times latent heat; bounded behavior when particle capacity is insufficient; deterministic ledger tolerance despite atomics.
- **Passive vapor:** transported plume and remote condensation conserve species/energy; no claim of pressure feedback; cache/reopen resumes the same declared state.
- **Coupled gas only:** classical bubble-growth benchmark and pressure/divergence closure; two-way momentum/energy ledger. Bubble growth must not be an acceptance target for passive vapor.
- **Adapters:** property/pending transition tests; Simulation Admission failure/cutover cleanup; cache schema round-trip and old-schema handling; Blender output attribute alignment; authoritative cache bake and preview playback parity; saved demo reopens at frame start and changes through real handlers.

Focused commands as each slice lands:

```powershell
python -m pytest -q tests/test_thermal_transport.py tests/test_sparse_leaf_grid_contract.py tests/test_particle_capacity_growth.py
python -m pytest -q tests/test_phase_change.py tests/test_pressure_projection_regressions.py tests/test_memory_safety_recovery.py
python -m pytest -q tests/test_cache_io.py tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_pending_settings_registry.py tests/test_simulation_admission.py
blender --background --factory-startup --python tests/blender_state_transition_acceptance.py
python tools/run_production_gate.py
python -m pytest -q
```

For interactive performance, also run the repository user-path profile and GPU solver profile documented in `AGENTS.md`; new thermal/gas phases need profiler slots and a bounded GPU memory/time budget. GUI-inclusive production gate plus policy-drift validation is required before release claims.

## Risks/rejected approaches

- **Rejected:** rename dissipation or custom-kernel presets as thermal physics. They lack conservation and reference validation.
- **Rejected:** use whitewater as steam or bubbles. Its source is churn and its mass is disconnected from liquid.
- **Rejected:** delete hot particles and spawn visual vapor. This breaks mass, energy, capacity, cache and replay truth.
- **Rejected:** use very high viscosity as ice. It neither supplies latent heat nor a solid constitutive model.
- **Rejected:** overload ST-FLIP `phase`/`density`; those names serve projection occupancy, not material phase/species.
- **Rejected:** add Blender-handler physics. It forks preview/bake behavior and violates owner locality.
- **Risk:** atomic emission/compaction ordering is numerically nondeterministic. Validate conserved reductions within tolerances; do not promise bitwise repeatability.
- **Risk:** thermal/vapor support can grow beyond liquid support. Admission, halo policy, cache size and interactive cost must be measured before default enablement.
- **Risk:** simultaneous introduction of moving solids, gas pressure and nucleation would create multiple new solvers. Freeze scope at the last validated vertical slice.

## Exact evidence index

| Fact | Live evidence |
|---|---|
| Flat sparse 8x8x8 leaf grid | `sim/constants.py:47-52`; `sim/fields.py:1-10,74-94,469-475` |
| Particle temperature storage and compaction | `sim/particles.py:16-18,50-80,341-383` |
| Source initialization | `sim/sourcing.py:95-125,129-158,780-817,943-981,1097-1139` |
| Grid fields and per-substep clearing | `sim/solver.py:710-722,1350-1363` |
| P2G excludes temperature | `sim/solver.py:1520-1587` |
| Forces/dissipation semantics | `sim/solver.py:1595-1667`; `sim/kernels_fluid.py:1386-1407` |
| Pressure seam | `sim/pressure.py:4-12`; `sim/solver.py:1742-1758` |
| G2P and lifecycle exclude thermal conversion | `sim/solver.py:1799-1865` |
| Whitewater is churn/SDF driven | `sim/solver.py:1894-1950` |
| Shared substep caller/order | `sim/solver.py:2067-2155` |
| Cache/HostFrame/publication scaffolding | `io/cache.py:227-269,333-380`; `sim/frame_output.py:199-240`; `blender/output_publication.py:1-5,736-748` |
| UI calls temperature an attribute | `properties.py:663-665,889-908,1218-1222`; `panels.py:696-704,799-804,1065-1069` |
| Existing tests prove scaffolding only | `tests/test_mesh_source_pipeline.py:132-170`; `tests/test_frame_output.py:22-25,234-245`; `tests/blender_background_acceptance.py:288-301` |
| Research conclusions and live-truth caveat | `research/state-transitions/REPORT.md:12-29`; `research/SYNTHESIS.md:15-24,40-53` |

All paths in this index are relative to `fluidSim_warp_blender51/` unless prefixed with `tests/` or `research/`.
