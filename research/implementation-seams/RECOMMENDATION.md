# Consolidated implementation-technique recommendation

## Outcome

Implement the researched capabilities as deep solver modules behind the existing `FluidSolver.step_frame()` orchestration and existing ownership seams. Do not extend the current `color`, `density`, or `temperature` names in place without first separating their meanings.

Recommended modules:

- `sim/species.py`: `SpeciesTransport` for miscible clear-dye state, conservative mixing, receipts, and derived optics.
- `sim/material_interface.py`: `MaterialInterfaceState` for material identity, conservative phase fraction, narrow-band interface geometry, coefficients, and interface receipts.
- `sim/pressure.py`: deepen the existing `PressureSolver.project_frame()` seam for arbitrary device-resident face coefficients and pressure jumps. Do not create another pressure orchestrator.
- `sim/thermal.py`: `ThermalState` for enthalpy, particle/grid transfer, sparse heat diffusion, temperature/liquid-fraction closure, and thermal receipts.
- `sim/phase_conversion.py`: later atomic liquid/vapor mass-and-energy transactions.
- `sim/gas.py`: later passive vapor, then low-Mach gas coupling. This is not required for the first freezing/melting tier.

The three detailed analyses are [color mixing](color-mixing.md), [density/multiphase](density-multiphase.md), and [state transitions](state-transitions.md).

## Live-code correction to the research baseline

ST-FLIP already implements a variable-coefficient liquid/gas occupancy projection:

- `sim/pressure.py:81-109` derives liquid occupancy from transfer mass and computes a face coefficient from configured liquid/gas densities.
- `sim/pressure.py:1144` uses that coefficient in the ST matrix path.
- `sim/pressure.py:1697-1762` owns the production `project_frame()` interface.
- `sim/solver.py:1742-1758` is the production caller.

This is reusable numerical scaffolding. It is not material-density or water/oil support: the coefficient is derived from liquid/gas occupancy, the global `spacetime_liquid_density`/`spacetime_gas_density` pair, and transfer weights—not particle `density` or a persistent material identity. Generalize this existing pressure implementation rather than adding a parallel solver.

## Semantic split required first

Reserve distinct authoritative meanings:

| Name | Meaning | Authority |
|---|---|---|
| `liquid_occupancy` | ST-FLIP/free-surface activity derived from transfer mass | grid/transient |
| `material_density` | physical density selected by material/phase composition | material interface state |
| `phase_id` | persistent immiscible material identity | particle |
| `phase_fraction` | conservative cell volume fraction for two liquids | material interface state |
| `species_concentration[K]` | miscible dissolved dye/species state | particle |
| `specific_enthalpy` | conserved thermal state | particle |
| `temperature` | derived from enthalpy/material closure in physical mode | thermal state |
| `liquid_fraction` | solid/liquid phase fraction derived from enthalpy | thermal state |
| `vapor_mass` | transported gas species for condensation/evaporation tiers | gas state |

The present particle `color`, particle `density`, grid `density`, and particle/grid `temperature` are useful lifecycle/output scaffolding, but they do not prove these physical meanings.

## Shared foundation before feature kernels

### 1. Particle semantics

Define represented particle mass or volume, source/sink accounting, compaction/deletion receipts, and future reseeding rules. Color conservation, phase-volume conservation, and phase conversion all depend on this contract. If uniform represented volume is selected initially, encode and test it explicitly rather than leaving it implicit.

Extend `ParticleSystem` through its existing allocation, growth, compaction, checkpoint, and restore interface. Do not scatter feature-specific array copies across callers.

### 2. Sparse stencil support

Deepen the existing flat-leaf storage interface with:

- declared stencil support radius;
- active-plus-halo leaf admission;
- deterministic positive-axis global face ownership;
- cross-leaf neighbor resolution through the flat leaf table;
- explicit inactive-neighbor boundary policy;
- topology epoch/invariant checks.

This is shared by species diffusion, conservative phase reconstruction, interfacial forces, and heat diffusion. It must remain fixed 8x8x8 flat-leaf storage; do not introduce hierarchical OpenVDB/NanoVDB live state.

### 3. Admission, memory, and persistence

`blender/simulation_admission.py` must preflight exact particle, grid, halo, scratch, and cache demand before replacement cutover. Extend `sim/memory_safety.py`, capacity planning, checkpoint format, cache schema, `HostFrame`, cache bake, and output adapters as one lifecycle slice for each authoritative field.

`pending_settings.py` remains the exhaustive settings lifecycle owner. Blender properties/panels and `solver_settings.py` are adapters; they must not reconstruct numerical policy.

## Techniques to implement

### A. Miscible clear-dye mixing

Implement first because it exercises the shared particle/sparse/persistence contracts without changing pressure or adding an interface solver.

Technique:

1. Particle-authoritative nonnegative species concentrations, specialized to `K <= 4` on GPU.
2. Existing FLIP/APIC particle motion supplies advection.
3. Mass-weighted species P2G into temporary flat-leaf fields.
4. Bounded equal-and-opposite face diffusion using deterministic global face ownership and admitted one-cell halos.
5. Conservative grid-delta G2P with a derived shared-capacity/simplex limiter.
6. Beer-Lambert absorption/extinction derived from concentrations for calibrated clear dyes.
7. Existing `color` remains a derived compatibility/preview channel; authoritative cache stores concentrations and palette metadata/hash.

Initial vertical slice: two calibrated dyes, constant-density common carrier, equal represented particle mass/volume, common diffusivity, closed box, source + direct injection, Mixing Off invariant, point and reconstructed-surface derived color, checkpoint/cache resume.

Do not call this paint/pigment mixing. Turbid pigments require a separate calibrated optics adapter such as Kubelka-Munk or measured spectral absorption/scattering. Do not add Mixbox as a production dependency without licensing clearance.

### B. Variable-density mixed liquid

Implement the pressure foundation before immiscibility:

1. Add immutable two-material records and persistent particle `phase_id`/represented volume, but allow the first physical tier to derive a blended material density without promising a retained interface.
2. Rename/separate current occupancy storage from physical material fields.
3. Have `MaterialInterfaceState` produce positive bounded cell density and face `inv_rho` arrays.
4. Generalize the existing ST `PressureSolver` path so matrix diagonal/application, preconditioner, residual, and velocity correction consume the same coefficient bundle.
5. Keep one shared MAC velocity and one pressure solve.
6. Emit coefficient range, density ratio, iteration, residual, and GPU timing diagnostics.

Truthful claim: density-driven stratification/buoyancy for mixed liquids. Do not claim water/oil separation.

### C. Two immiscible liquids

Add only after the variable-density pressure tier is stable:

1. Persistent particle `phase_id` through every source, growth, compaction, deletion, checkpoint, cache, and output lane.
2. Conservative bounded cell `phase_fraction` derived from represented particle volume.
3. Narrow-band signed-distance/interface geometry coupled to the conservative fraction.
4. Phase-aware FLIP/APIC transfer with the existing single velocity field.
5. Consistent face material coefficients supplied to the deepened pressure solver.
6. Ghost-Fluid or discretely balanced interfacial pressure jump using pairwise tension.
7. Static contact angle only after wall/interface normals are validated; defer dynamic wetting and N-phase/triple-junction physics.

Reserve the oil/water claim until phase-volume convergence, interface retention under stirring, Laplace pressure jump, and spurious-current refinement all pass.

### D. Thermal transport and freezing/melting

Separate current artistic temperature from physical thermodynamics.

Technique:

1. Particle-authoritative specific enthalpy plus material thermal properties.
2. Mass-weighted enthalpy P2G and G2P.
3. Implicit sparse-grid heat diffusion using the shared halo/stencil contract.
4. Enthalpy-to-temperature/liquid-fraction closure with latent heat and a declared transition interval/hysteresis.
5. Initial frozen-state mechanics through stationary porosity/velocity lock before pressure; no particle creation/deletion in this tier.
6. Optional Boussinesq force only from physical temperature. The existing custom Boussinesq snippet is an artistic/custom surface, not the physical owner.

Truthful claim: thermal diffusion and stationary freezing/melting after Stefan, latent-plateau, reversible-cycle, and mass/energy-ledger validation. Do not claim moving ice or fracture.

### E. Evaporation, condensation, and steam

Stage separately:

1. Energy-limited interfacial mass flux and one atomic `PhaseConversion` transaction with liquid mass, vapor mass, sensible energy, latent energy, and bounded capacity receipts.
2. Artistic steam output may be offered earlier only when labeled non-conservative and non-transported.
3. Passive vapor adds transported vapor mass/temperature and enables remote condensation without pressure feedback.
4. Coupled boiling/bubble growth requires a low-Mach gas velocity/pressure/divergence solve in `GasState`.
5. Moving frozen bodies/MPM and stochastic/material-specific nucleation require separate architecture approval.

Whitewater must not be relabeled as steam; deleting hot liquid particles and spawning display vapor is rejected.

## Recommended execution order

1. Semantic split, represented mass/volume, conservation receipt types, and cache migration policy.
2. Shared flat-leaf halo/face-ownership interface plus admission/memory accounting.
3. Clear-dye `SpeciesTransport` vertical slice and its conservation/convergence gates.
4. Material identity lifecycle plus unambiguous occupancy/material fields.
5. Generalized coefficient-driven `PressureSolver` and variable-density mixed-liquid gates.
6. Conservative fraction/interface geometry and two-immiscible-liquid gates.
7. `ThermalState` transport and stationary freezing/melting.
8. `PhaseConversion`, passive vapor/condensation, then `GasState` pressure coupling.

Do not implement all lanes simultaneously. The first implementation project should stop after steps 1-3: shared contracts plus the clear-dye vertical slice. Pressure, interface, thermal, and gas work should remain separate approved projects with their own runtime gates.

## Highest-value gates

- Disabled-feature parity and no device-to-host transfer in substeps.
- Cross-leaf face/edge/corner translation invariance.
- Exact cache/checkpoint/reopen identity for all authoritative fields.
- Color: closed-box species conservation `<= 1e-5` over 240 frames; no negative/NaN/Inf; Gaussian diffusion `<= 2%` L2; resume `<= 1e-4` L2.
- Density: hydrostatic piecewise slopes; coefficient identity between assembly/correction; density-ratio conditioning and GPU timing.
- Immiscible: bounded phase fraction, phase-volume convergence, Laplace jump, spurious-current refinement, interface retention under stirring.
- Thermal: analytic diffusion, Stefan convergence, latent plateau, reversible cycle, closed mass/energy ledger.
- Vapor/gas: evaporation/condensation flux sign and balance; passive-vapor round trip; bubble growth only for the pressure-coupled gas tier.

## Rejected architecture

- RGB blur or frame-rate-dependent color interpolation as mixing.
- Reusing the viscosity `attr_weight`, ST occupancy `density`, or display `color` as authoritative new physics.
- A second pressure solver or independent per-phase projections.
- Separate full velocity grids for the first immiscible tier.
- Dense grids, hierarchical tree storage, or live VDB as authoritative state.
- Very high viscosity presented as freezing.
- Whitewater or visual particles presented as steam.
- Physics in Blender handlers, output publication, playback, or demo builders.
- Generalized N-phase, moving solids, gas pressure, and nucleation in the first vertical slice.

## Remaining blockers

- Represented particle mass/volume and reseeding semantics require an explicit decision.
- The bounded conservative delta-G2P species limiter requires derivation and independent numerical tests.
- Sparse global face ownership and halo admission must be proven under capacity pressure.
- Sharp density ratios may expose PCG conditioning/performance limits.
- Variable-viscosity stress jumps and triple-line policy remain unresolved for advanced immiscibility.
- The implicit sparse thermal solver and any gas solver require independent performance prototypes.

