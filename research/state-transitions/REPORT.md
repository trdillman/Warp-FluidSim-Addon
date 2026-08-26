# Fluid state transitions for a GPU-resident FLIP/APIC liquid solver

## 1. Research metadata

- Research date: 2026-08-22.
- Scope: evaporation and steam generation, boiling, condensation, freezing, melting, thermal transport, latent heat, phase fraction, solid/liquid/gas coupling, rendering, conservation, stability, and interactive-VFX fidelity tiers.
- Target constraints: Blender 5.1 add-on; GPU-resident simulation; interactive viewport playback; fixed 8x8x8 sparse leaf tiles addressed by a flat leaf table; solver policy remains below Blender adapters.
- Method: primary papers, first-party manuals, and pinned source code were downloaded or clipped locally. The artifact inventory and hashes are in [sources/MANIFEST.md](sources/MANIFEST.md).
- Evidence labels used below: **Confirmed** means directly supported by a cited primary artifact; **Inference** means a consequence of those sources and the supplied architecture; **Recommendation** is a project decision proposed here. Research is not implementation proof.
- Deliberate boundary: no production code was written and no existing project source, tests, documentation, history, or template was inspected.

## 2. Executive conclusions

1. **Recommendation — implement freezing/melting first and independently.** Add temperature transport plus a latent-enthalpy or phase-fraction state, then use the liquid fraction to change viscosity and suppress solid motion. This fits the current particle/grid liquid architecture and does **not** require a gas solver. A smoothed apparent-heat-capacity update is the simplest grid implementation; a per-particle latent-energy buffer is the better conservation anchor for FLIP/APIC transfers. Kopp et al. show the apparent-capacity formulation; Gao et al. and Fuchs et al. show particle/grid and particle-solid transition designs ([paper](https://doi.org/10.1007/s00366-022-01719-1), [local](sources/kopp_et_al_2022_apparent_heat_capacity.pdf); [paper](https://doi.org/10.1016/j.gmod.2017.09.001), [local](sources/gao_et_al_2017_heat_phase_transition.pdf); [paper](https://doi.org/10.1186/s40323-021-00200-w), [local](sources/fuchs_et_al_2021_sph_reversible_phase_transitions.pdf)).
2. **Confirmed — physically coupled steam and condensation require a vapor carrier.** Bures and Sato solve liquid- and gas-side temperature gradients, gas species transport, and interface mass flux. A liquid-only solver can emit a visual steam volume, but it cannot transport humidity, return condensed mass at another location, or represent vapor pressure/expansion conservatively ([paper](https://doi.org/10.1016/j.ijheatmasstransfer.2021.121233), [local](sources/bures_sato_2021_sharp_interface_phase_change.pdf)).
3. **Recommendation — separate three fidelity tiers.** Tier A maps temperature to viscosity, color, and artistic steam with no physical phase conversion. Tier B conserves energy for solid-liquid transition and optionally converts surface mass to a passive vapor field. Tier C conservatively couples liquid mass, latent heat, vapor transport, divergence/volume source, and condensation. UI and cache metadata must identify the tier.
4. **Recommendation — do not begin with full sharp-interface boiling or compressible steam.** Those require interface reconstruction, two-sided gradients, pressure/volume coupling, and a gas equation of state. They are materially more expensive and less compatible with interactive sparse FLIP than enthalpy-based freezing.
5. **Confirmed — high viscosity is not a solid model.** Blender's 5.1 manual explicitly warns that very large viscosity does not produce rigid-body behavior and may become unstable ([manual source](https://projects.blender.org/blender/blender-manual/src/commit/b8df653bf9e26ca59337b0d6a32e227407623b02/manual/physics/fluid/type/domain/liquid/diffusion.rst), [local](sources/blender_5_1_liquid_diffusion.rst)). A first freezing release may use porosity drag or velocity locking as a bounded approximation, but moving fractured ice or elastic solids need a separate rigid/MPM-style representation.
6. **Recommendation — conservation accounting is a release gate.** Track sensible energy, latent energy, liquid mass, vapor mass, emitted/absorbed boundary flux, and numerical correction. Temperature-threshold deletion, particle count changes without variable mass, or steam output with no mass ledger are artistic effects, not physical phase change.

## 3. User-visible capability definition

| Tier | User-visible behavior | Physical meaning | Required solver support |
|---|---|---|---|
| A — Thermal art direction | Hot/cold color, temperature-driven viscosity, thermal buoyancy, steam-like emission | No guaranteed phase conservation | Temperature scalar, diffusion/advection, property ramps, secondary output |
| B — Solid/liquid phase change | Melting/freezing front, latent-heat plateau, solid fraction, reheating/remelting, stationary or highly damped frozen regions | Energy-aware solid-liquid transition; solid mechanics simplified | Enthalpy/phase fraction, diffusion, phase-dependent properties, solid-motion policy |
| B+ — Passive vapor | Surface evaporation consumes liquid mass and latent heat; vapor advects and may condense on cold surfaces | Conservative mass/energy if vapor is transported and returned; pressure effect omitted | Vapor density and temperature fields or mass-carrying vapor particles, interface exchange |
| C — Coupled boiling/condensation | Boiling, bubble/steam growth, condensation away from source, buoyant plume, volume expansion | Two-way liquid-gas mass, energy, momentum, and volume coupling | Gas velocity/pressure or low-Mach gas solver, divergence source, saturation relation, interfacial flux |
| D — Unified solids | Mobile ice chunks, fracture/plasticity, solid-liquid momentum exchange | Separate solid mechanics coupled to thermal phase conversion | Rigid clusters, MPM, or another solid solver plus contact and remeshing/particle conversion |

The feature must expose its fidelity honestly. “Steam” without transported vapor mass is a rendering/output effect. “Frozen” liquid represented only by high viscosity is nearly rigid liquid, not a mechanically valid solid. “Boiling” driven only by `temperature >= boiling_point` is a rate control, not a saturation-pressure model.

## 4. Physical model and terminology

### Energy and phase state

For material temperature `T`, density `rho`, heat capacity `cp`, conductivity `k`, and heat source `Q`, the sensible-energy equation is

`rho cp DT/Dt = div(k grad T) + Q`.

An enthalpy representation folds latent heat into the state:

`h(T, f_l, f_v) = h_ref + integral(cp dT) + L_f f_l + L_v f_v`,

where `f_l` and `f_v` are liquid and vapor fractions, `L_f` is latent heat of fusion, and `L_v` is latent heat of vaporization. An apparent-heat-capacity method instead uses `c_app = dh/dT`; Kopp et al. regularize the phase step with a smooth `tanh` phase function so that latent heat becomes a finite heat-capacity peak ([paper](https://doi.org/10.1007/s00366-022-01719-1), [local](sources/kopp_et_al_2022_apparent_heat_capacity.pdf)).

**Recommendation:** keep `T` and a bounded latent-energy/phase state. Temperature alone is insufficient inside a phase plateau because many enthalpy states can share the same transition temperature.

### Stefan condition and interface mass flux

A Stefan problem is a heat equation with an unknown moving phase boundary. For evaporation/condensation, Bures and Sato calculate interface mass flux from the imbalance of liquid- and gas-side conductive heat fluxes:

`j_m = (q_l + q_g) / L = (lambda_l grad(T_l).n - lambda_g grad(T_g).n) / L`.

The same mass flux moves the interface, removes/adds liquid mass, adds/removes vapor mass, and consumes/releases latent heat. Their geometric VOF method couples this to gas species transport and validates Stefan, sucking, and bubble-growth cases ([paper](https://doi.org/10.1016/j.ijheatmasstransfer.2021.121233), [local](sources/bures_sato_2021_sharp_interface_phase_change.pdf)).

### Phase fraction, mushy zones, and solids

`f_l = 0` is solid, `f_l = 1` is liquid, and intermediate values form a numerically broadened transition or mushy interval. OpenFOAM clamps its updated liquid fraction to `[0,1]`, applies a Darcy-like momentum sink proportional to `(1-f_l)^2/(f_l^3+q)`, and adds Boussinesq buoyancy ([source](https://github.com/OpenFOAM/OpenFOAM-11/blob/9cbf94f3339dd49286633b303c4155fd10ea8200/src/fvModels/derived/solidificationMeltingSource/solidificationMeltingSource.C), [local](code/openfoam11_solidificationMeltingSource.C)). This is a useful fixed-grid model for an immobile solid matrix; it is not a fracture, elastic, or rigid-body model.

Fuchs et al. instead represent solids as mobile rigid particle clusters and move particles between fluid and solid descriptions based on temperature, demonstrating that mobile solids require explicit mechanics and contact beyond viscosity ([paper](https://doi.org/10.1186/s40323-021-00200-w), [local](sources/fuchs_et_al_2021_sph_reversible_phase_transitions.pdf)).

### Evaporation, boiling, condensation, and gas pressure

- **Evaporation** is interfacial mass transfer below or at saturation, often limited by heat and vapor diffusion.
- **Boiling** is vapor generation within or at a superheated liquid when local thermodynamic/nucleation conditions permit bubbles. It introduces strong density/volume change.
- **Condensation** reverses the mass and latent-heat flux when vapor reaches a cold or supersaturated region/interface.
- **Steam/vapor** is a gas species; visible white “steam” is usually condensed droplets, not molecular water vapor.

IAPWS supplies pressure-dependent saturation pressure and saturated liquid/vapor density, enthalpy, and entropy correlations. NIST records water phase-change reference data including a normal boiling point near 373.17 K; this must not be treated as pressure independent ([IAPWS release](https://iapws.org/documents/release/Supp-sat), [local](sources/IAPWS_SR1-86_1992_saturation_water.pdf); [NIST WebBook](https://webbook.nist.gov/cgi/cbook.cgi?Mask=4&Name=water), [local](sources/nist_water_phase_change_data.html)).

**Inference:** an incompressible liquid projection can accept a divergence source caused by mass-to-volume conversion, but pressure feedback and large vapor expansion need at least a low-Mach Eulerian gas solve. A passive vapor scalar is sufficient only when pressure coupling is intentionally omitted.

### Buoyancy, viscosity, and yield

Thermal buoyancy can use a Boussinesq force `rho_ref g beta (T-T_ref)` when density variation is small. For gas, buoyancy generally depends on temperature and composition/density. Viscosity can vary continuously with temperature or phase fraction, but fully frozen material needs velocity constraints, porosity drag, rigid clustering, or a solid constitutive model. Yield stress/plasticity belongs to the solid or viscoplastic model, not the heat equation.

### Nucleation and supercooling

Real freezing and boiling can remain metastable beyond equilibrium temperature until a nucleus forms. A production physical model would need material-specific nucleation kinetics and stochastic state. **Recommendation:** initial releases should use an explicit deterministic hysteresis (`freeze_onset < melt_onset`) and label it an artistic/macroscopic approximation. Do not add random per-cell nucleation without a resolution-independent rate model; it will change with tile count and timestep.

## 5. Implementation technique comparison

| Technique | Fidelity | Complexity | GPU cost | Memory cost | Sparse-grid/project fit | Evidence strength | Recommendation |
|---|---|---:|---:|---:|---|---|---|
| Smoothed apparent heat capacity on grid | Solid-liquid energy, diffuse interface | Low-medium | One scalar diffusion plus nonlinear property update | 1-2 scalar grids | Excellent; local tile stencils with halo | Strong paper + MOOSE implementation | **Stage 1 physical core**; pair with conservation ledger |
| Per-particle latent-energy buffer plus grid temperature | Solid-liquid energy with FLIP/APIC persistence | Medium | Transfer thermal mass/energy in P2G/G2P; grid diffusion | 2-3 particle scalars plus grid accumulators | Excellent | Strong graphics literature; Gao is closest local FLIP evidence | **Preferred particle state** over temperature-only |
| Enthalpy-porosity / Darcy drag | Fixed or mushy solidification | Medium | Local phase update plus momentum sink | Phase fraction + optional drag | Excellent | Strong mature OpenFOAM/MOOSE code | Use for stationary frozen regions; not moving ice |
| Temperature-to-viscosity ramp | Artistic thickening/cooling | Low | Existing viscosity path plus scalar | Temperature scalar | Excellent | Strong industry baseline | Tier A only; never call it solid mechanics |
| Rigid clusters or MPM solid conversion | Mobile solid chunks, contact, plasticity | High | New P2G/G2P constitutive and contact passes | Deformation/cluster state | Moderate; particles fit, but it is a second solver | Strong particle-solid paper evidence | Later separate subsystem; explicit architecture approval |
| Sharp-interface Stefan/VOF | Engineering interface and flux | Very high | Interface reconstruction, one-sided gradients, coupled iterations | Volume fraction, geometry, both phase fields | Poor-moderate with FLIP surface and sparse activation | Strong CFD paper and verification | Reference/validation tier, not interactive default |
| Passive vapor scalar with conservative surface exchange | Evaporation/condensation transport without pressure | Medium-high | Gas advection/diffusion plus surface exchange | Vapor, gas temperature, source fields | Good if gas tiles share leaf table | Strong mass/heat-flux basis; pressure omission is explicit | Stage after freezing; acceptable B+ tier |
| Artistic steam secondary particles/volume | Visual steam only | Low | Emission and advection | Secondary particles or one density field | Excellent | Industry practice; no physical conservation claim | Optional Tier A, clearly labeled |
| Low-Mach Eulerian gas with divergence source | Coupled boiling/condensation and expansion | High | Gas pressure solve, advection, species, coupling | Multiple gas fields + pressure | Moderate; same tiles possible, wider active region | Strong CFD basis | Required for Tier C |
| Fully compressible vapor | Shocks/acoustics and large pressure ratios | Very high | Acoustic CFL or implicit compressible solve | Conservative gas state | Poor for interactive target | Strong physics generally, no project-specific proof | Reject for initial product |
| Phase-field nucleation/dendrites | Microstructure/supercooling | Very high | Fine interface stencils and stochastic integration | Phase field + gradients/noise | Poor at VFX voxel scales | Strong research area, weak product fit | Reject except specialized future mode |

## 6. Primary-source evidence

- **Kopp et al. (2022):** a smoothed phase function places latent heat in an apparent heat capacity; inadequate resolution of the heat-capacity peak distorts temperatures and melt geometry. This supports a finite mushy interval and nonlinear-solver safeguards, not an infinitesimal temperature jump ([DOI](https://doi.org/10.1007/s00366-022-01719-1), [local PDF](sources/kopp_et_al_2022_apparent_heat_capacity.pdf)).
- **Gao et al. (2017):** a graphics-focused simplified heat model couples solid-liquid-gas transition to multiphase FLIP and uses SPH-FLIP sub-grid bubbles. It demonstrates feasibility and visual value, but the paper's simplified temperature model is not an engineering conservation certificate ([DOI](https://doi.org/10.1016/j.gmod.2017.09.001), [local PDF](sources/gao_et_al_2017_heat_phase_transition.pdf)).
- **Bures and Sato (2021):** a sharp geometric VOF solver computes mass flux from two-sided heat flux divided by latent heat, transports non-condensable gas species, and verifies Stefan, sucking, and bubble-growth cases with grid convergence. This is the strongest local evidence for conservative evaporation/condensation requirements ([DOI](https://doi.org/10.1016/j.ijheatmasstransfer.2021.121233), [local PDF](sources/bures_sato_2021_sharp_interface_phase_change.pdf)).
- **Fuchs et al. (2021):** SPH heat conduction, liquid/gas phases, mobile rigid solids, contact, and reversible melting/solidification show what a mechanically meaningful moving-solid tier entails. It is evidence for architecture boundaries, not code reuse ([DOI](https://doi.org/10.1186/s40323-021-00200-w), [local PDF](sources/fuchs_et_al_2021_sph_reversible_phase_transitions.pdf)).
- **IAPWS SR1-86(1992):** authoritative saturation correlations cover pressure and saturated liquid/vapor properties; water transition presets should derive from a pressure-dependent property source, not a single boiling temperature ([official release](https://iapws.org/documents/release/Supp-sat), [local PDF](sources/IAPWS_SR1-86_1992_saturation_water.pdf)).
- **NIST WebBook:** provides traceable phase-change reference data. It is suitable for preset spot checks, not a complete runtime equation of state ([official data](https://webbook.nist.gov/cgi/cbook.cgi?Mask=4&Name=water), [local snapshot](sources/nist_water_phase_change_data.html)).

## 7. Reference implementations and code findings

### OpenFOAM 11, commit `9cbf94f3339dd49286633b303c4155fd10ea8200`

- Inspected `src/fvModels/derived/solidificationMeltingSource/solidificationMeltingSource.C` functions `readCoeffs`, `Cp`, `update`, and `addSup`, plus the class declaration in `.H` ([source](https://github.com/OpenFOAM/OpenFOAM-11/tree/9cbf94f3339dd49286633b303c4155fd10ea8200/src/fvModels/derived/solidificationMeltingSource), [local C](code/openfoam11_solidificationMeltingSource.C), [local H](code/openfoam11_solidificationMeltingSource.H), [GPL notice](code/openfoam11_COPYING)).
- `update` relaxes and clamps `alpha1` to `[0,1]`; vector `addSup` applies the porosity sink and Boussinesq buoyancy. **Reusable idea:** bounded phase fraction, once-per-time-index update, and separate energy/momentum sources. **Potentially reusable code:** none recommended without a GPL compatibility decision. **Unsuitable:** direct cell-loop structure and OpenFOAM field ownership do not match GPU Warp kernels or the supplied sparse leaf table.

### MOOSE, commit `31c4ce30a68ef937ee9a4aed875f9eba45155751`

- Inspected `modules/navier_stokes/src/fvkernels/NSFVPhaseChangeSource.C`, its header, and `examples/solidification/gallium_melting.i` ([source](https://github.com/idaholab/moose/tree/31c4ce30a68ef937ee9a4aed875f9eba45155751/modules/navier_stokes), [local C](code/moose_NSFVPhaseChangeSource.C), [local H](code/moose_NSFVPhaseChangeSource.h), [local case](code/moose_gallium_melting.i), [LGPL-2.1](code/moose_LICENSE)).
- `NSFVPhaseChangeSource::computeQpResidual` computes `f_l=(T-T_solidus)/(T_liquidus-T_solidus)`, uses `max(6 f_l (1-f_l),0)` for a bounded derivative, multiplies by `L rho/(T_liquidus-T_solidus)`, and couples to `T_dot`. The gallium case wires liquid fraction, buoyancy, gravity, and a phase-change energy source. **Reusable idea:** smooth compact latent-heat distribution and automatic-differentiation-friendly bounded derivative. **Potentially reusable code:** formula only after independent implementation and license review. **Unsuitable:** finite-element/FV object framework and CPU-oriented solve plumbing.

### Mantaflow, commit `494445143507296d16e467160c1889be657f8f74`

- Inspected `source/plugin/extforces.cpp` kernels/functions `KnApplyForceField`, `addGravity`, `addBuoyancy`, and `KnDissolveSmoke`; inspected `source/plugin/flip.cpp` `mapPartsToMAC`, `mapMACToParts`, and `combineGridVel` ([source](https://github.com/thunil/mantaflow/tree/494445143507296d16e467160c1889be657f8f74/source/plugin), [local forces](code/mantaflow_extforces.cpp), [local FLIP](code/mantaflow_flip.cpp), [Apache-2.0](code/mantaflow_LICENSE)).
- **Reusable idea:** maintain thermal force application as a grid force and attach thermal/phase state to the same FLIP transfer seam. **Potentially reusable code:** Apache-2.0 permits adaptation with notices, but these CPU/C++ kernels are not directly suitable for Warp. **Unsuitable:** treating Mantaflow smoke heat dissolution as phase transition; it dissipates a visual field and has no liquid/vapor mass ledger.

No inspected repository provides the complete target combination of GPU sparse FLIP/APIC, enthalpy, gas-phase mass transfer, and moving solids. Combining these pieces is an engineering proposal.

## 8. Industry/product feature baseline

- **Blender 5.1 manual:** liquid and gas are distinct domain modes; the domain type cannot change dynamically. Liquid exposes FLIP settings, viscosity, surface tension, particles, and meshing. The viscosity manual warns that high viscosity does not create rigid behavior ([settings source](https://projects.blender.org/blender/blender-manual/src/commit/b8df653bf9e26ca59337b0d6a32e227407623b02/manual/physics/fluid/type/domain/settings.rst), [local](sources/blender_5_1_fluid_domain_settings.rst); [diffusion source](https://projects.blender.org/blender/blender-manual/src/commit/b8df653bf9e26ca59337b0d6a32e227407623b02/manual/physics/fluid/type/domain/liquid/diffusion.rst), [local](sources/blender_5_1_liquid_diffusion.rst)). **Confirmed baseline:** built-in Blender does not present dynamic liquid-gas phase conversion as a domain feature.
- **SideFX FLIP Solver:** supports FLIP/APIC transfer choice, viscosity and density by particle attribute, adaptive viscosity, substeps, and field/particle diagnostics ([manual](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html), [local](sources/sidefx_houdini21_flip_solver.html)).
- **SideFX Gas Temperature Update:** diffuses and cools a FLIP temperature field and maps normalized temperature to viscosity/density, including a “Solidify Threshold” that produces nearly rigid high viscosity ([manual](https://www.sidefx.com/docs/houdini/nodes/dop/gastemperatureupdate.html), [local](sources/sidefx_houdini21_gas_temperature_update.html)). **Confirmed baseline:** this is a strong artist workflow, but the page does not document latent heat, liquid/vapor mass exchange, or a mechanically separate solid.

The competitive minimum is therefore temperature-driven appearance and material properties. The differentiator is an honest conserved phase-state path with cacheable diagnostics, followed later by gas coupling.

## 9. Fit to FluidSim architecture

Everything in this section is a **proposal**, not an assertion about existing implementation.

### Likely persistent fields on each active 8x8x8 leaf

- Required for solid-liquid: `temperature`, `thermal_mass` or `heat_capacity`, `latent_energy` or `liquid_fraction`, `conductivity`, and a `thermal_valid/weight` accumulator.
- Optional for vapor: `vapor_mass_density`, `gas_temperature`, `phase_mass_source`, `phase_energy_source`, `gas_divergence_source`, and `condensate_mass`.
- Derived/transient: `effective_heat_capacity`, `viscosity`, `density_delta`, `porosity_drag`, interface area/normal, and solver residual.
- Tile policy: activate a one-tile thermal halo before stencil passes; preserve a deterministic inactive-cell ambient/boundary value; do not allocate vapor fields in solid-liquid-only scenes.

At float32, one scalar is 2 KiB per 512-cell leaf. Six persistent scalar fields are 12 KiB/leaf before double buffers and metadata. This makes field count and gas-domain activation range first-order product costs.

### Likely particle attributes

- `temperature`, `thermal_mass` (`m*cp`), `latent_energy` or `phase_fraction`, `material_id`, and immutable/ledgered particle mass.
- Optional: `phase_state`, `freeze_onset`, `solid_cluster_id`, `vapor_mass`, and rendering attributes.
- Transfer rule: P2G/G2P thermal quantities must be mass weighted. Clamp phase fraction only after accounting for the clamped energy difference in a correction ledger.

### Proposed solver phases/kernels

1. `thermal_p2g_accumulate`: mass-weight temperature/enthalpy and material coefficients.
2. `thermal_activate_halo_and_boundaries`: allocate neighboring leaves and apply ambient/collider heat conditions.
3. `thermal_advect_grid_or_particles`: use the existing particle motion for material heat; advect gas scalars on the gas velocity.
4. `thermal_diffuse`: explicit preview or matrix-free implicit production solve across flat leaf neighbors.
5. `phase_update_enthalpy`: invert enthalpy to `T` and phase fraction, applying hysteresis if enabled.
6. `phase_update_properties`: viscosity, density, surface tension, and porosity/solid constraint coefficients.
7. `phase_interface_flux`: estimate surface area and liquid/gas heat gradients; calculate bounded evaporation/condensation mass flux.
8. `phase_apply_mass_energy_exchange`: update particle mass/count and gas mass, latent energy, and correction ledgers in one transaction.
9. `thermal_buoyancy_force`: apply liquid or gas buoyancy before projection.
10. `phase_divergence_source`: add volume expansion/contraction for Tier C before gas pressure projection.
11. `solid_transition_coupling`: stationary drag/locking initially; cluster/MPM handoff only in its own later subsystem.
12. `thermal_g2p_and_diagnostics`: return state and reduce mass/energy/residual statistics.

The phase transaction must complete before output publication. Playback, cache bake, publication, and overlay should consume authoritative solver state without recreating transition policy.

### Cache and output proposal

- Cache metadata: model tier/version, units, material parameters, pressure assumption, solver tolerances, hysteresis policy, and conservation ledger.
- Per-frame channels: particle `temperature`, `phase_fraction`, `mass`, optional `solid_cluster_id`; leaf `temperature`, `vapor_density`, optional `gas_velocity`; global energy/mass totals.
- Surface output: mesh attributes `temperature`, `liquid_fraction`, `heat_flux`, and `condensation`; phase-aware color/roughness hooks.
- Volume output: vapor/visible-condensate density and temperature as separate channels. Do not render transparent water vapor as opaque white smoke by default.
- Diagnostic overlay only: total liquid/vapor/solid mass, sensible/latent energy, maximum thermal residual, clamped corrections, active thermal/gas leaves, and phase-conversion rate.

### UI proposal

- Master mode: `Off`, `Artistic Thermal`, `Conserved Solid-Liquid`, `Passive Vapor`, `Coupled Gas`.
- Material: preset plus `rho`, `cp`, conductivity, fusion/vaporization latent heat, solidus/liquidus, saturation model, viscosity curve, expansion ratio.
- Boundaries: initial/ambient/source temperature, heat flux/insulation, collider conductivity, pressure assumption.
- Controls: thermal substeps/tolerance, transition width, hysteresis/supercooling toggle and onset, evaporation coefficient/rate cap, condensation targets, gas resolution/tier.
- Truth labels: “Artistic steam (non-conservative)” and “Nearly rigid frozen liquid” where applicable.

## 10. Performance and numerical-risk analysis

- **Diffusion stability:** explicit heat diffusion is local and cheap but in 3D is limited approximately by `dt <= dx^2/(6 alpha)` for a uniform six-neighbor stencil. High conductivity or fine voxels can dominate the timestep. Use explicit preview only with a reported stability cap; use an implicit/matrix-free solve for production.
- **Latent stiffness:** a very narrow transition interval creates a sharp `c_app` peak, increasing iterations and grid sensitivity. Kopp et al. directly show that the peak must be resolved. Expose transition width as a numerical parameter with material-derived defaults.
- **Sparse boundary error:** missing neighbor leaves can act as unintended insulation or ambient sinks. Thermal halo activation and tile-boundary parity tests are mandatory.
- **Particle/grid conservation:** independently transferring temperature loses energy when particle masses or `cp` differ. Transfer thermal mass and total energy/enthalpy, then divide. APIC improves velocity transfer behavior but does not by itself conserve thermal energy.
- **Mass conversion:** changing only particle count creates quantized evaporation. Variable particle mass, a residual reservoir per tile, or conservative split/merge is required. Atomics make reduction order nondeterministic; acceptance should use tolerances, not bitwise equality.
- **Gas extent:** steam can occupy much more space than liquid. Sparse storage helps, but gas active leaves may dominate memory and pressure-solve bandwidth. Passive vapor should be optional and separately budgeted.
- **Pressure/volume:** water-to-steam density ratios make a no-divergence incompressible liquid update physically inconsistent. Tier B+ may intentionally omit pressure feedback; Tier C must apply volume/divergence source and solve gas pressure.
- **Property extremes:** temperature-dependent viscosity can change by orders of magnitude. Clamp UI ranges, use implicit viscosity where already supported by solver architecture, and avoid interpreting arbitrary maximum viscosity as solid.
- **Interface flux noise:** FLIP surfaces and sparse gradients are noisy. Reconstruct a filtered interface area/normal, cap per-step converted mass to available phase mass, and use temporal residual carry rather than silent clipping.
- **Coupling order:** heat diffusion, phase change, viscosity/solid forces, and pressure are tightly coupled. A single explicit pass is fast but splitting error grows near rapid boiling/freezing. Production may need Picard iterations; preview can accept one pass with a quality label.

## 11. Validation and acceptance strategy

Source-supported benchmark families come from Bures and Sato (Stefan, sucking, and bubble growth with grid convergence) and from mature OpenFOAM/MOOSE solidification cases. The numerical tolerances below are **proposed acceptance targets** unless explicitly noted.

1. **Unit inversion:** sweep enthalpy across solid, mushy, liquid, and vapor ranges. Require monotonic `h(T)`, phase fraction in `[0,1]`, and round-trip relative error below `1e-5` away from discontinuities.
2. **Latent plateau:** heat a closed cell/particle ensemble at constant power. Integrated input energy must equal sensible plus latent energy within `0.1%` production / `0.5%` preview; temperature must remain within the configured transition interval while phase fraction changes.
3. **1D Stefan melt/freeze:** compare interface position with the analytical similarity solution over at least three resolutions. Require monotonic convergence and production interface error below one finest-grid cell; report observed order rather than claiming a fixed order in advance.
4. **Solidification cavity:** reproduce qualitative melt-front and buoyant convection structure from the gallium-style benchmark; run with buoyancy off as the contrasting path.
5. **Reversible cycle:** heat through melting, cool through freezing, then compare final mass and total energy to the boundary-flux ledger. Hysteresis must change onset only, not create energy.
6. **Adiabatic conservation:** no sources/boundary flux. Require total liquid+solid+vapor mass drift below `0.1%` and total thermal energy drift below `0.1%` in production over the acceptance interval.
7. **Evaporation sign test:** imposed hotter liquid side gives positive evaporation; hotter gas side gives condensation. `j_m L` must equal the applied two-sided heat-flux difference within discretization tolerance.
8. **Passive-vapor round trip:** evaporate into a closed gas volume and condense on a cold surface. Require returned liquid plus remaining vapor plus residual reservoir to close the mass ledger below `0.1%`.
9. **Coupled bubble growth:** compare radius history against the analytical/validated bubble-growth case. If adopting the Bures-style sharp/capillary discretization, enforce its source-reported `CFL < 0.18` and capillary timestep bound `dt < 0.42 sqrt((rho_v+rho_l) dx^3/sigma)` as a benchmark configuration, not a universal solver constant.
10. **Tile-boundary parity:** run the same conduction and phase-front problems centered within a leaf and crossing leaf boundaries. Target temperature/phase differences below `1e-5` for deterministic single-step kernels and statistically equivalent integrated results for atomic paths.
11. **Timestep/refinement:** halve timestep and voxel size independently; mass/energy error must not increase and phase-front/bubble metrics must converge.
12. **Zero-feature regression:** with transitions off, no additional fields/kernels may alter existing liquid motion, cache hashes, or viewport output except explicitly versioned metadata.
13. **GPU residency/performance:** profile P2G/G2P, diffusion iterations, active leaves, memory, phase conversion, and gas projection. No host round-trip is permitted inside the per-substep thermal/phase path.
14. **Cache/reopen:** saved state must resume with identical material parameters and conservation ledger; artistic and physical tiers must not be interchangeable without cache invalidation.

## 12. Recommended staged roadmap

1. **Stage 0 — contracts and ledgers:** define units, material schema, cache versioning, conservation reductions, inactive-tile boundary policy, and disabled-feature parity.
2. **Stage 1 — artistic thermal field:** GPU temperature P2G/G2P, diffusion, sources/boundaries, buoyancy, viscosity/color output, sparse halo tests. No phase-change claim.
3. **Stage 2 — conserved freezing/melting:** add enthalpy/latent buffer and phase fraction; implement stationary porosity/velocity-lock solid policy; validate Stefan and reversible cycles. This stage is independent of gas work.
4. **Stage 3 — surface evaporation and visual output:** add energy-limited surface mass flux, residual mass reservoir, and optional artistic steam output. Label non-transported steam as non-conservative output.
5. **Stage 4 — passive vapor and condensation:** add vapor mass/temperature transport, cold-surface condensation, closed-domain mass tests, and volume output. Explicitly omit pressure feedback.
6. **Stage 5 — low-Mach gas coupling:** gas velocity/pressure, species, divergence source, buoyancy, bubble-growth validation, and coupled boiling. This is the first tier that may claim physically coupled steam/condensation.
7. **Stage 6 — moving solids/nucleation:** only after separate architecture approval, add rigid clusters or MPM handoff and material-specific/stochastic nucleation. These are independent advanced systems, not cleanup for Stage 2.

Each stage should ship only if its own fidelity label, cache schema, diagnostics, and acceptance lane are complete.

## 13. Dependencies and cross-lane coupling

| Dependency | Shared state/contract | Order |
|---|---|---|
| Material properties | Units, `rho/cp/k/L`, phase temperatures, viscosity/density curves, pressure assumption | Before any kernel |
| Sparse-grid core | Thermal halo activation, flat leaf neighbors, inactive-cell boundary rules, scalar allocation | Before Stage 1 |
| FLIP/APIC transfers | Mass-weighted enthalpy/temperature and correction ledger | Stage 1-2 |
| Viscosity/forces | Phase-dependent viscosity, porosity drag, thermal buoyancy | Stage 1-2 |
| Surface reconstruction | Interface area/normals and surface temperature sampling | Before evaporation |
| Gas/smoke solver | Vapor species, gas temperature/velocity, divergence source, pressure projection | Stage 4-5 |
| Solid/MPM or rigid bodies | Handoff state, momentum/energy transfer, contact, remelting | Stage 6 only |
| Cache and bake | Versioned thermal/phase/vapor channels, solver tier, ledgers | With each stage |
| Output/rendering | Surface attributes and separate vapor/condensate volume | Stage 1, 3, 4 |
| UI/admission | Material completeness, memory estimate, invalid-tier rejection | Before user exposure |
| Validation/profile | Analytic benchmarks, conservation, leaf parity, GPU timing | Continuous; gates each stage |

Cross-lane order is: material/units -> sparse thermal field -> FLIP thermal transfer -> solid-liquid enthalpy -> surface flux -> passive vapor -> gas pressure coupling -> advanced solids/nucleation. Gas and advanced solids should not block the independently useful freezing/melting release.

## 14. Risks, unknowns, and rejected approaches

- **Rejected:** delete liquid particles when `T > T_boiling`. It violates mass/energy conservation, has timestep-dependent rates, and cannot condense.
- **Rejected:** call a high-viscosity ramp “freezing.” Blender and SideFX demonstrate the workflow, but only Blender explicitly warns it is not rigid behavior.
- **Rejected:** reuse smoke “heat dissipation” as latent heat. Mantaflow's `KnDissolveSmoke` is a visual-field decay path, not a phase-energy balance.
- **Rejected for initial delivery:** full sharp-interface VOF/Stefan reconstruction. It is an excellent validation reference but an expensive second interface method beside FLIP.
- **Rejected for initial delivery:** fully compressible steam. Acoustic timesteps and equation-of-state coupling conflict with the interactive target.
- **Rejected for initial delivery:** whole-domain phase-field nucleation/dendrites. Its required interface resolution is disproportionate to VFX voxel scales.
- **Risk:** apparent heat capacity can hide conservation errors if only temperature is stored; preserve an energy/latent state and correction ledger.
- **Risk:** sparse deactivation can discard latent/vapor mass. A tile cannot deactivate until all conserved thermal/phase reservoirs are transferred or zero within tolerance.
- **Risk:** variable particle mass may conflict with reseeding, surface reconstruction, or output. Define whether mass, volume, and radius change together before evaporation work.
- **Risk:** water properties vary with pressure and temperature. IAPWS/NIST values support presets but not arbitrary materials; generic materials need explicit user data.
- **Risk:** licenses differ. OpenFOAM is GPL-3-or-later, MOOSE is LGPL-2.1, Mantaflow is Apache-2.0, Blender manual is CC BY-SA 4.0, and some papers permit only limited redistribution. Prefer independent implementation from equations/ideas; obtain legal review before copying code.
- **Unknown:** the cost and convergence of an implicit thermal solve on the project's exact sparse leaf neighbor structure is unmeasured.
- **Unknown:** no local primary source validates the complete proposed GPU sparse FLIP + enthalpy + vapor + solid architecture. Every integration claim remains a recommendation until runtime-tested.

## 15. Source index

| ID | Source family | Primary source | Local artifact(s) | Main use |
|---|---|---|---|---|
| P1 | Paper | [Kopp et al., 2022](https://doi.org/10.1007/s00366-022-01719-1) | [PDF](sources/kopp_et_al_2022_apparent_heat_capacity.pdf) | Apparent heat capacity, transition smoothing, numerical stiffness |
| P2 | Paper | [Gao et al., 2017](https://doi.org/10.1016/j.gmod.2017.09.001) | [PDF](sources/gao_et_al_2017_heat_phase_transition.pdf) | Graphics FLIP/SPH solid-liquid-gas model |
| P3 | Paper | [Bures and Sato, 2021](https://doi.org/10.1016/j.ijheatmasstransfer.2021.121233) | [PDF](sources/bures_sato_2021_sharp_interface_phase_change.pdf) | Conservative evaporation/condensation and validation |
| P4 | Paper | [Fuchs et al., 2021](https://doi.org/10.1186/s40323-021-00200-w) | [PDF](sources/fuchs_et_al_2021_sph_reversible_phase_transitions.pdf) | Moving solid/liquid transition architecture |
| S1 | Standard/property release | [IAPWS SR1-86(1992)](https://iapws.org/documents/release/Supp-sat) | [PDF](sources/IAPWS_SR1-86_1992_saturation_water.pdf) | Saturation and liquid/vapor properties |
| S2 | Government data | [NIST WebBook water](https://webbook.nist.gov/cgi/cbook.cgi?Mask=4&Name=water) | [HTML](sources/nist_water_phase_change_data.html) | Reference phase-change values |
| C1 | Source code | [OpenFOAM 11 pinned tree](https://github.com/OpenFOAM/OpenFOAM-11/tree/9cbf94f3339dd49286633b303c4155fd10ea8200/src/fvModels/derived/solidificationMeltingSource) | [C](code/openfoam11_solidificationMeltingSource.C), [H](code/openfoam11_solidificationMeltingSource.H), [license](code/openfoam11_COPYING) | Enthalpy-porosity, phase fraction, buoyancy |
| C2 | Source code | [MOOSE pinned tree](https://github.com/idaholab/moose/tree/31c4ce30a68ef937ee9a4aed875f9eba45155751/modules/navier_stokes) | [C](code/moose_NSFVPhaseChangeSource.C), [H](code/moose_NSFVPhaseChangeSource.h), [case](code/moose_gallium_melting.i), [license](code/moose_LICENSE) | Apparent-capacity source and benchmark wiring |
| C3 | Source code | [Mantaflow pinned tree](https://github.com/thunil/mantaflow/tree/494445143507296d16e467160c1889be657f8f74/source/plugin) | [forces](code/mantaflow_extforces.cpp), [FLIP](code/mantaflow_flip.cpp), [license](code/mantaflow_LICENSE) | FLIP transfer and buoyancy seams |
| I1 | First-party manual | [Blender 5.1 manual source](https://projects.blender.org/blender/blender-manual/src/commit/b8df653bf9e26ca59337b0d6a32e227407623b02/manual/physics/fluid/type/domain) | [settings](sources/blender_5_1_fluid_domain_settings.rst), [diffusion](sources/blender_5_1_liquid_diffusion.rst), [license](sources/blender_manual_LICENSE) | Host baseline and high-viscosity limitation |
| I2 | First-party manual | [SideFX FLIP Solver](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html), [Gas Temperature Update](https://www.sidefx.com/docs/houdini/nodes/dop/gastemperatureupdate.html) | [FLIP HTML](sources/sidefx_houdini21_flip_solver.html), [temperature HTML](sources/sidefx_houdini21_gas_temperature_update.html) | Industry thermal/viscosity workflow baseline |

Full artifact-level provenance, licenses, retrieval failures, and SHA-256 values are in [sources/MANIFEST.md](sources/MANIFEST.md).

## 16. Orchestrator ingestion block

```yaml
recommended_techniques:
  - mass-weighted particle enthalpy or latent-energy buffer
  - sparse-grid temperature diffusion with a smoothed phase interval
  - bounded liquid fraction plus stationary porosity or velocity-lock policy
  - conservative surface mass/heat flux before any physical steam claim
alternatives:
  - artistic temperature-to-viscosity and steam output
  - passive vapor scalar without pressure feedback
  - later low-Mach gas, rigid clusters, or MPM
required_shared_state:
  - temperature, thermal mass, latent energy or liquid fraction
  - immutable/ledgered particle mass and material id
  - optional vapor mass, gas temperature, divergence source
new_phases_kernels:
  - thermal P2G and G2P
  - halo activation and boundary heat
  - heat diffusion
  - enthalpy inversion and property update
  - interface mass/energy flux and transactional conversion
  - thermal buoyancy and optional gas divergence
cache_output_impact:
  - versioned material/tier metadata and conservation ledger
  - temperature/phase particle attributes, optional vapor leaf channels
  - surface temperature/phase attributes and separate vapor volume
ui_impact:
  - explicit fidelity tier, material thermal properties, boundaries
  - transition width, hysteresis, evaporation cap, gas coupling
strongest_validation_targets:
  - Stefan convergence and latent plateau
  - closed mass/energy ledger
  - evaporation/condensation flux sign and balance
  - sparse leaf-boundary parity
  - passive-vapor round trip and coupled bubble growth
blockers:
  - no existing gas-pressure or moving-solid capability may be assumed
  - variable particle-mass/reseeding contract must be defined
  - implicit sparse thermal solver cost is unmeasured
cross_lane_order:
  - materials -> sparse thermal -> FLIP transfer -> freezing/melting
  - surface flux -> passive vapor -> gas pressure -> advanced solids/nucleation
confidence:
  solid_liquid_recommendation: high
  passive_vapor_recommendation: medium_high
  coupled_boiling_scope: high
  performance_estimates: medium_low_until_profiled
```
