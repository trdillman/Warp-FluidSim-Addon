# FluidSim advanced liquid research synthesis

## Status and evidence integrity

This is a research and planning input, not proof that any capability is implemented. The three isolated lanes were given only the supplied project specification and were prohibited from reading project files. The orchestrator separately checked the current source and independently recomputed every manifest hash.

| Lane | Report | Manifest artifacts | SHA-256 result | Report structure | Local references |
|---|---|---:|---|---|---|
| Color mixing | [color-mixing/REPORT.md](color-mixing/REPORT.md) | 24 | 24/24 match | 16/16 sections | all resolve |
| Density and multiphase | [density-multiphase/REPORT.md](density-multiphase/REPORT.md) | 25 | 25/25 match | 16/16 sections | all resolve |
| State transitions | [state-transitions/REPORT.md](state-transitions/REPORT.md) | 21 | 21/21 match | 16/16 sections | all resolve |

The standardized contracts are [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) and [MANIFEST_TEMPLATE.md](MANIFEST_TEMPLATE.md).

## Current live-code truth

The repository already contains useful storage and publication seams, but it does not currently implement the researched physics:

- `sim/particles.py` defines physical particle channels for temperature, viscosity, and density plus optional color storage (`PHYSICAL_ATTR_NAMES` and `OPTIONAL_VEC4_ATTR_NAMES`, lines 16-18). This proves storage, not transport or coupling.
- `sim/solver.py` allocates temperature, viscosity, and density sparse scalar fields (lines 711-714).
- The standard particle-to-grid path transfers only particle viscosity (`_p2g`, lines 1520 and 1573-1584). The density grid is also used as the ST-FLIP phase/activity output (lines 1555-1561), so it cannot be assumed to be an unambiguous physical material-density field.
- The current pressure formulation folds constant density into `q = p*dt/rho` (`sim/pressure.py`, lines 1-6); it is not the researched variable-coefficient material-density projection.
- `sim/attributes.py` and `sim/frame_output.py` can persist/publish color and density attributes. Surface values are currently obtained by nearest-particle sampling (`frame_output.py`, `output_attributes`, lines 267-323). That is an output transfer, not realistic color diffusion, interface conservation, or thermodynamics.
- Temperature can be sourced, dissipated, customized, cached, and displayed, but there is no energy equation, latent-heat ledger, phase fraction, conservative vapor mass flux, or gas-pressure coupling in the traced solver step.

Consequently, existing `color`, `density`, and `temperature` names must not be used to claim color mixing, oil/water physics, freezing, steam, or condensation.

## Decision summary

### Color mixing

Represent the physical state as nonnegative species concentrations on particles, not display RGB. Advection rides the existing particle motion. Mixing uses mass-weighted P2G, conservative bounded diffusion over globally owned sparse-grid faces (including leaf halos), and conservative delta G2P. Derive viewport/render color from concentrations; Beer-Lambert absorption is the recommended clear-dye baseline. Keep a specialized live path for up to four species while versioning the cache schema for extension.

Do not integrate Mixbox without commercial license clearance. Treat Kubelka-Munk, spectral, or calibrated latent pigment models as later turbid-pigment tiers.

### Different-density and immiscible liquids

Use two deliberately distinct product claims:

1. **Variable-density mixed liquid:** per-material density affects buoyancy and the pressure projection, but no persistent sharp interface is promised.
2. **Two immiscible liquids:** persistent `phase_id`, bounded conservative phase fraction, interface geometry, pairwise interfacial tension, and quantified interface retention are required.

The recommended immiscible architecture is a shared MAC velocity with phase-aware FLIP/APIC transfer, a variable-coefficient sparse pressure solve using the same face `1/rho` coefficient in matrix assembly and velocity correction, and a conservative fraction plus narrow-band geometry. Use a Ghost-Fluid or discretely balanced jump for interfacial tension. Defer generalized N-phase and triple-junction behavior.

### State transitions

Freezing and melting can be staged independently of gas work. Use conserved particle enthalpy or a latent-energy buffer, sparse-grid temperature diffusion, a bounded liquid fraction, and a clearly declared stationary-solid policy such as porosity/velocity lock for the first tier.

Artistic steam may be emitted earlier only if labeled as non-conservative output. Physical evaporation/condensation requires conservative interfacial mass and energy flux plus transported vapor mass/temperature. Boiling expansion and pressure-coupled steam additionally require a gas velocity/pressure solver. Moving frozen solids, MPM handoff, and material-specific nucleation are separate future systems, not completion work for the initial freezing tier.

## Shared dependency order

1. **State semantics and ledgers**
   - Define particle represented mass/volume, immutable material/species identity, units, and conservation receipts.
   - Reserve distinct names for solver activity/phase mask, physical density, species concentration, material phase, liquid fraction, and vapor fraction.
2. **Sparse field and halo contract**
   - Add active-plus-halo leaf activation, deterministic global face ownership, cross-leaf reductions, and boundary policies on the fixed flat 8x8x8 leaf table.
3. **Lifecycle and cache schema**
   - Make sourcing, reseeding, compaction, capacity growth, checkpoints, cache bake, reopen, and publication preserve new authoritative channels.
4. **Color transport vertical slice**
   - Implement species state, conservative diffusion, output derivation, and conservation/convergence tests. This is the lowest-risk independent user-visible feature.
5. **Variable-density projection foundation**
   - Separate physical density from the current ST-FLIP density/activity field; assemble and apply consistent face coefficients; prove conditioning and hydrostatics.
6. **Two-liquid immiscible interface**
   - Add `phase_id`, phase fraction, interface geometry, surface-tension jump, phase-aware surface reconstruction, and cache/output identity.
7. **Thermal foundation and freezing/melting**
   - Add thermal P2G/G2P, diffusion, enthalpy inversion, property update, and the first stationary-solid policy.
8. **Vapor tiers**
   - Energy-limited surface flux and labeled artistic output; then passive vapor/condensation; only then gas-pressure coupling and physically coupled boiling.

This order shares foundational work but does not require color mixing to wait for immiscibility or freezing to wait for a gas solver.

## Strongest acceptance targets

### Shared

- No device-to-host transfer inside simulation substeps.
- Flat-leaf face/edge/corner translation invariance and halo parity.
- Cache/reopen/resume preserves authoritative fields, identities, settings, and conservation ledgers.
- Disabled features preserve existing solver behavior and memory budgets.

### Color

- Closed-box species conservation at or below `1e-5` over 240 frames.
- No negative concentrations, capacity violations, NaN, or infinity.
- Gaussian diffusion error at or below 2% L2 for the specified benchmark.
- Cache-resume concentration error at or below `1e-4` L2.

### Density and immiscibility

- Hydrostatic piecewise pressure slopes for two densities.
- Bounded phase fraction and convergent per-phase volume.
- Laplace pressure jump and spurious-current reduction under refinement.
- Equal-density/same-phase, unequal-density/same-phase, and unequal-density/different-phase comparison cases; reserve the oil/water claim for persistent interface retention in the third case.
- Pressure convergence and GPU cost measured across increasing density ratios.

### State transitions

- Stefan convergence, latent-temperature plateau, and reversible freeze/melt cycles.
- Closed mass and energy ledgers.
- Correct evaporation/condensation flux sign and balance.
- Passive-vapor cache round trip before gas coupling; bubble-growth validation only for the coupled gas tier.

## Stop/go decisions before implementation

1. Choose the first color target: calibrated clear dyes or physically measured turbid pigments. Clear dyes are the supported initial recommendation.
2. Approve the product boundary between variable-density mixed liquid and true two-liquid immiscibility.
3. Define particle mass/volume and reseeding semantics before any conservative scalar or phase implementation.
4. Choose the first frozen-state behavior: stationary porosity/velocity lock, rigid clusters, or MPM. Only the first is currently a bounded initial tier.
5. Choose the steam fidelity tier and its truthful UI label before designing vapor output.
6. Review dependency licenses before copying any reference code; much of the strongest multiphase source is GPL/AGPL, and Mixbox production use requires commercial clearance.

## Research limitations

- No reviewed implementation provides the complete combination of Warp/CUDA, FLIP/APIC, fixed flat 8x8x8 sparse leaves, conservative color species, sharp variable-density multiphase interfaces, and thermal phase change.
- The proposed architecture combines independently evidenced techniques; it still requires project-specific derivation, prototypes, profiling, and regression gates.
- Direct PowerShell downloads were blocked by Windows socket policy. Researchers used a materially different web-fetch/narrow-clip fallback and recorded the failure. Full open-access PDFs were stored where permitted; other evidence is narrow and license-noted.
- Research artifacts are evidence snapshots, not approved dependencies or redistribution clearance.

