# Realistic mixing of colored miscible liquids in a GPU FLIP/APIC solver

Research checked 2026-08-22. This note is source research, not evidence about the local implementation.

## Decision

Use a **passive multi-species concentration field carried authoritatively by liquid particles**, with one concentration or mass fraction per colorant and an implicit residual solvent fraction. Particle motion supplies advection without a separate scalar semi-Lagrangian pass. Add physical and optional subgrid mixing as a **conservative, bounded diffusion step** on the existing sparse liquid grid, then transfer only the diffusion delta back to particles conservatively. Derive viewport RGB, surface attributes, and render absorption/scattering from concentration; never make RGB the simulated conserved state.

This is the smallest architecture that fits FLIP/APIC: particles already resolve transport while the grid handles local operators. The APIC paper explicitly describes the hybrid split and proves linear/angular-momentum conservation for its transfer—not scalar conservation, so scalar conservation still needs its own construction ([Jiang et al. 2015, DOI 10.1145/2766996](https://doi.org/10.1145/2766996)). NVIDIA Warp v1.13.0 provides a concrete GPU APIC transfer example in `warp/examples/fem/example_apic_fluid.py`: `integrate_fraction()`, `integrate_velocity()`, `update_particles()`, and `Example.simulate()` ([immutable v1.13.0 source](https://github.com/NVIDIA/warp/blob/v1.13.0/warp/examples/fem/example_apic_fluid.py), Apache-2.0). Its NanoVDB hierarchy is only a transfer precedent; it must **not** replace this project's flat 8x8x8 leaf-table live-storage contract.

## Confirmed facts

- A passive scalar obeys advection-diffusion, with optional sources. OpenFOAM v2506 documents and implements `Foam::functionObjects::scalarTransport` in `src/functionObjects/solvers/scalarTransport/scalarTransport.{H,C}` as a conservative transport equation with flux, diffusivity, and source terms; its `bounded01` option uses MULES to keep a multiphase scalar in `[0,1]` ([v2506 API](https://api.openfoam.com/2506/classFoam_1_1functionObjects_1_1scalarTransport.html), [header/source contract](https://api.openfoam.com/2506/scalarTransport_8H_source.html), GPL-3.0-or-later). The official documentation also warns that its ordinary scalar advection is bounded only with the upwind scheme, illustrating that higher-order transport and boundedness are separate requirements ([OpenFOAM scalar transport docs](https://doc.openfoam.com/2306/tools/post-processing/function-objects/solvers/scalarTransport/)).
- Second-order semi-Lagrangian correction is not automatically bounded. MantaFlow's `source/plugin/advection.cpp` exposes `MacCormackCorrect()`, `MacCormackClamp()`, `fnAdvectSemiLagrange()`, and `advectSemiLagrange()`; the clamp is a distinct step that limits against source-area extrema ([MantaFlow 0.11 Doxygen](https://mantaflow.com/doxy/advection_8cpp.html)). MantaFlow also exposes implicit `cgSolveDiffusion()` with a grid-space diffusion coefficient ([plugin API](https://mantaflow.com/doxy/group__Plugins.html)). MantaFlow is Apache-2.0 ([license](https://github.com/tum-pbs/mantaflow/blob/master/LICENSE)). These are reference designs, not GPU-residency proof.
- Multi-component state should remain on a simplex. OpenFOAM's `multiComponentMixture<ThermoType>::correctMassFractions()` sums species fields and divides every species by the sum; it errors on a zero sum ([OpenFOAM v2406 `multiComponentMixture.C`](https://api.openfoam.com/2406/multiComponentMixture_8C_source.html), GPL-3.0-or-later). This confirms the invariant `sum(Y_k)=1`, but post-hoc normalization is **not** a substitute for a conservative transport update because it can change each species total.
- Subgrid turbulent mixing is a model, not molecular diffusion. Greif et al. discretize diffusion symmetrically over particle neighbors, use a harmonic mean coefficient, and tie unresolved diffusivity to local velocity dispersion and smoothing length. They explicitly state that their isotropic closure yields an upper limit on mixing and does not reproduce the turbulent cascade ([paper](https://academic.oup.com/mnras/article/392/4/1381/964513), [DOI 10.1111/j.1365-2966.2008.14169.x](https://doi.org/10.1111/j.1365-2966.2008.14169.x)). A simple gradient/eddy-diffusivity closure can be unphysical for differential or counter-gradient species transport, so it should be presented as an optional scale closure, not universal ground truth ([Pouransari et al. 2014, DOI 10.1016/j.ijhydene.2014.03.018](https://doi.org/10.1016/j.ijhydene.2014.03.018)).
- Molecular diffusivity is colorant-, solvent-, concentration-, and temperature-dependent. Direct-dye NMR measurements span roughly `1.7e-10` to `4.4e-8 m^2/s` across the tested dyes, solvents, and temperatures ([DOI 10.1016/S0143-7208(01)00035-3](https://doi.org/10.1016/S0143-7208(01)00035-3)); fluorescein in water is commonly of order `5e-10 m^2/s`. Therefore a per-species world-unit diffusivity is more defensible than an arbitrary per-frame blend.
- Optical mixing depends on the material class. For a clear absorbing dye, transmittance follows Beer’s law and stays in `[0,1]`; absorption coefficients are spectral, nonnegative, and have inverse-length units ([PBRT 4e, transmittance](https://pbr-book.org/4ed/Volume_Scattering/Transmittance), [absorption](https://pbr-book.org/4ed/Volume_Scattering/Volume_Scattering_Processes)). PBRT-v4 performs light transport spectrally and supports spatial RGB absorption/scattering grids ([source repository](https://github.com/mmp/pbrt-v4), Apache-2.0). For turbid pigment/paint, RGB averaging and elementary subtractive mixing are inadequate; Kubelka-Munk models absorption and scattering coefficients ([Haase and Meyer 1992, DOI 10.1145/146443.146452](https://doi.org/10.1145/146443.146452)). A calibrated five-paint study reported mean `1.49 DeltaE00` across 33 mixes, demonstrating both feasibility and the need for calibration ([Dichter 2023, DOI 10.82019/JAIC.3207](https://doi.org/10.82019/JAIC.3207), CC BY 4.0 article).
- Blender 5.1 primarily performs materials, lighting, geometry-node, and compositing operations in a scene-linear working space ([Blender 5.1 manual](https://docs.blender.org/manual/en/5.1/render/color_management/color_spaces.html)). Display-encoded sRGB values therefore should not be averaged as if they were concentrations or extinction coefficients.
- Cache formats can retain source state rather than baked appearance. OpenVDB 13.0.1 supports named multiple grids, per-grid/file metadata, and `PointDataGrid` attributes; `openvdb::points::appendAttribute()` adds typed point attributes ([overview](https://www.openvdb.org/documentation/doxygen/overview.html), [points](https://www.openvdb.org/documentation/doxygen/points.html), [`PointAttribute.h`](https://www.openvdb.org/documentation/doxygen/PointAttribute_8h.html)). This is an interchange/cache option only, not permission to use a hierarchical VDB tree for live simulation.

## Recommended transport design

### State and units

For particle `p`, store `c[p,k]` for `K` explicit colorants, each in `[0,1]`. Define solvent as `c_solvent = 1 - sum_k c[p,k]`, so the explicit vector satisfies `sum_k c[p,k] <= 1`. Also retain the particle mass or represented volume used by the solver. A structure-of-arrays layout (`c_k[p]`) is the safest variable-`K` GPU layout; a fixed `vec4` fast path is reasonable for `K<=4` but must not define the cache schema.

Use world-unit `D_m[k]` (`length^2 / second`) and convert through the simulation scale. A user-facing “mixing” control should map to a documented `D_sgs`, not directly lerp colors per frame. Use `D_eff[k] = D_m[k] + D_sgs[k]`. Default `D_sgs=0` for laminar/physical mode; expose an explicitly labeled unresolved/turbulent mixing mode.

### Advection

Keep `c[p,k]` unchanged while the particle is advected. This eliminates scalar interpolation diffusion from the ordinary FLIP/APIC loop. Do not P2G/G2P color every substep merely for advection.

If a grid-resident scalar is ever made authoritative, use conservative flux-form advection with a limiter. MacCormack/BFECC can be a preview-quality option only when followed by a local extrema clamp; simple clamping after the fact is bounded but not conservative.

### Conservative diffusion on the sparse grid

At the diffusion stage, using the same nonnegative partition-of-unity weights `w_ip` as liquid transfer:

1. P2G liquid mass and species mass: `M_i = sum_p w_ip m_p`; `Q_ik = sum_p w_ip m_p c_pk`; `C_ik = Q_ik/M_i` for occupied nodes.
2. Compute each active face once. Apply an equal-and-opposite diffusive species flux to its two liquid nodes. Use a harmonic face diffusivity for discontinuous `D`, matching the robustness choice in Greif et al. Do not diffuse through air or solid faces; sparse leaves need a one-cell neighbor halo and cross-leaf face ownership.
3. Limit outgoing fluxes before applying them so no node can send more of species `k` than `Q_ik`, and so the residual solvent stays nonnegative. Pairwise equal/opposite application preserves every global `sum_i Q_ik`; a later clamp would not.
4. Transfer only `delta Q_ik` back to the particles. With `a_ip = w_ip m_p/M_i`, distribute `delta q_pk += sum_i a_ip delta Q_ik`. Since `sum_p a_ip=1` at every occupied node, this preserves the grid diffusion delta globally. Apply a conservative local flux limiter/simplex correction if any particle would cross a bound; do not independently clamp and renormalize.
5. Keep all arrays and reductions GPU-resident. Float atomics make execution-order hashes nondeterministic; validate invariants with tolerances rather than promise bitwise repeatability.

An alternative is symmetric unordered particle-pair exchange. It naturally applies equal/opposite deltas and is useful as a small-scene reference implementation, but requires neighbor traversal per species and is likely costlier than the already-active sparse grid at production particle counts.

### Sources, sinks, reseeding, and property coupling

- Every liquid source must specify a composition vector. New particles receive that exact composition; an omitted composition means solvent-only, not “current UI color.”
- Reseeding must interpolate concentration with mass/volume weights and preserve grid-integrated species mass across the reseed operation. Particle deletion at outflow is a physical species flux and must be included in mass accounting; numerical reseeding deletion is not.
- If colors are merely dyes in the same carrier, one velocity/pressure solve and passive transport are appropriate. If components change density, viscosity, surface tension, temperature, or reaction rates, it is no longer “color only”: mixture properties and possibly buoyancy must be coupled to the solver. Ren et al. use volume fractions and a mixture model for miscible/immiscible diffusion and reactions ([DOI 10.1145/2645703](https://doi.org/10.1145/2645703)); that larger physics should be a separate capability, not silently implied by colored water.

## Color and rendering model

Maintain three distinct layers:

1. **Simulation truth:** species concentrations/mass fractions only.
2. **Viewport approximation:** a cheap lookup from concentration to scene-linear display color. For clear dyes, add per-species absorption coefficients and evaluate a fixed preview path length, `T_rgb = exp(-L * sum_k c_k sigma_a,k,rgb)`. For an opaque/turbid preset, a calibrated pigment LUT or Kubelka-Munk approximation is appropriate. Never average display-sRGB endpoints.
3. **Final render:** for clear colored liquid, retain the dielectric water surface and drive spatial volume absorption (plus optional scattering) from the concentration field. Beer-Lambert path-length dependence means a vertex base color alone cannot reproduce clear dyed water. For milky/pigmented liquid, mix absorption and scattering coefficients and render as a participating medium; Kubelka-Munk is a turbid-layer/paint model, not a generic clear-liquid volume model.

Store optical calibration as palette metadata (`species id`, name, `D_m`, spectral or scene-linear absorption, optional scattering, reference concentration, reference length, color-space identifier). This permits recoloring/recalibration without rerunning transport.

Surface/particle publication should sample concentrations first, then derive a render attribute. If only a mesh-color preview is supported, label it an approximation; it is not proof of physical volumetric color.

## Cache contract

Authoritative resumable frames should contain:

- schema/version, frame/time, world-unit scale, species count and stable species IDs;
- per-particle positions plus `c[p,k]` and the mass/volume weight needed to reconstruct species mass;
- source/sink cumulative species budgets and applied mixing settings;
- palette/calibration metadata or a stable external palette hash;
- sufficient reseeding/RNG state for resume if the solver promises trajectory-equivalent continuation.

Derived caches may additionally contain per-species sparse concentration grids, a packed preview RGB grid, and published mesh/point color. Do not cache only RGB: it loses multi-species identity, prevents a different optical model, and cannot support a conservative resume. OpenVDB can represent named grids/point attributes for interchange, but the live solver remains the mandated flat leaf table.

## Proposed executable acceptance thresholds

These are engineering gates inferred from the invariants above, not thresholds claimed by the cited papers.

| Gate | Proposed pass criterion |
|---|---|
| Constant field P2G/diffuse/G2P | Maximum concentration error `<= 2e-6` for one step; no new species |
| Closed-box conservation | For each species, relative mass drift `<= 1e-5` over 240 frames with no source/sink; `<= 1e-6` for one transfer/diffusion step |
| Bounds/simplex | `min(c_k) >= -2e-6`, `max(c_k) <= 1+2e-6`, `sum_k c_k <= 1+5e-6`; no NaN/Inf |
| Pure advection (`D=0`) | Species totals meet conservation gate; after one periodic solid-body rotation, normalized L1 scalar error `<= 5%` at the specified reference resolution |
| Analytic diffusion | Gaussian variance follows `variance(t)=variance(0)+2Dt`; relative L2 field error `<= 2%` at 64+ cells across the profile; total species mass drift `<= 1e-5` |
| Sparse-leaf seam | A diffusion pulse crossing each leaf face matches the same pulse placed away from a seam within `1e-5` max concentration and has no tile-edge discontinuity |
| Reseeding stress | Forced birth/death cycle changes closed-box species totals by `<= 1e-4`; all newborn particles satisfy bounds/simplex |
| Source/sink budget | `initial + injected - removed - final` relative residual `<= 1e-4` per species |
| Multi-species independence | With `D_k=0`, an untouched species remains unchanged within `2e-6`; permuting species storage order changes results by `<= 2e-6` |
| Cache resume | Continuous `N+M` frames versus save/reload at `N` agrees in per-species mass within `1e-5`, bounds within thresholds, and concentration-field L2 within `1e-4`; no bitwise-hash requirement on CUDA |
| Optical endpoints | Zero concentration is optically neutral; pure-species reference concentration matches its calibrated swatch under fixed light/camera with mean `DeltaE00 <= 2` (or report uncalibrated) |
| Optical mixtures | For a calibrated palette, held-out measured binary/ternary mixes have mean `DeltaE00 <= 2` and max `<= 5`; otherwise advertise qualitative, not physical, color |
| GPU residency | Runtime trace shows no device-to-host transfer inside simulation substeps; profile time and memory for `K=1,3,4` and fail on allocation/capacity overflow rather than silently dropping species |

Also run a resolution study (`h`, `h/2`) for a stirred two-color case. The mixedness curve and species totals should converge; target `<=10%` difference in mixedness integral before claiming resolution-independent behavior.

## Remaining uncertainty and risks

- The best conservative G2P delta limiter for this solver's exact particle weights and reseeding policy needs derivation and numerical testing. Partition of unity proves global delta conservation before limiting, but bounded limiting must preserve that invariant too.
- Molecular diffusion alone will usually be visually slow at animation scale. Any accelerated mixing is an SGS/artistic closure and must be labeled, unit-scaled, and resolution-tested.
- Beer-Lambert coefficient additivity can fail at high concentration through dye aggregation or chemical interaction; use measured mixture data when “physically accurate color” is a requirement.
- Kubelka-Munk is suitable for turbid pigments/layers, not clear dyed water. A single universal “realistic mixing” toggle would conflate different optics.
- If differently colored inputs also differ in density or rheology, passive dye transport is insufficient and the task expands into coupled multi-fluid physics.

