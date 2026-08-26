# Realistic color mixing for GPU-resident FLIP/APIC liquids

## 1. Research metadata

- Question: how a Blender 5.1 GPU-resident FLIP/APIC-style liquid simulator should model, transport, mix, cache, and present two or more differently colored miscible liquids.
- Research date: 2026-08-22 UTC.
- Project constraints used: GPU-resident simulation, interactive viewport playback, fixed `8x8x8` sparse leaf tiles addressed through a flat leaf table, no hierarchical live-field tree, and separation of solver policy from Blender adapters/output/playback/diagnostics.
- Evidence policy: primary papers, first-party manuals/specifications, and source code at immutable tags/commits. Local narrow clips and code excerpts are indexed in [`sources/MANIFEST.md`](sources/MANIFEST.md).
- Scope boundary: this report proposes architecture and tests. It is not evidence that any production implementation exists or passes.
- Retrieval limitation: direct shell download was blocked by the exact socket error recorded in [`sources/manuals/network-failure.md`](sources/manuals/network-failure.md). A web-open plus narrow local-clip fallback was used once instead of retrying the same blocked transport.
- Claim labels: **Confirmed** means directly supported by cited primary evidence; **Inference** is a consequence of those facts and the supplied constraints; **Recommendation** is the proposed choice for FluidSim.

## 2. Executive conclusions

**Recommendation:** simulate colorants as nonnegative, particle-authoritative concentration or mass-fraction channels, not as RGB. Particle motion provides advection. Use the sparse grid only for a conservative, bounded diffusion/mixing phase: mass-weighted particle-to-grid (P2G), equal-and-opposite face fluxes, and a conservative grid-delta-to-particle (G2P) transfer. Derive viewport and render color afterward from the concentrations.

This split is the smallest credible design for a FLIP/APIC liquid:

1. particles already carry material through large deformation;
2. a grid is the natural place for a local diffusion operator;
3. species mass can be budgeted independently of display color;
4. the optical model can change without rebaking transport;
5. a flat sparse leaf table can execute the operator with a one-cell face halo and explicit cross-leaf ownership.

The default optical model should be **clear dye absorption**: mix per-species absorption/extinction coefficients by concentration and use Beer-Lambert attenuation. This creates subtractive-looking, path-length-dependent color without pretending RGB endpoints are conserved chemicals. A calibrated Kubelka-Munk/pigment mode is a later option for turbid paint-like liquids. It is not a universal model for clear dyed water. The Mixbox paper validates a practical pigment latent space, but the available Mixbox implementation is non-commercial/CC BY-NC; therefore it is a product benchmark and rejected production dependency unless separately licensed ([paper and local clip](https://doi.org/10.1145/3478513.3480549), [`sources/papers/pigment-mixing.md`](sources/papers/pigment-mixing.md); [tag and local notice](https://github.com/scrtwpns/mixbox/tree/v1.2), [`code/mixbox-v1.2/LICENSE-NOTICE.md`](code/mixbox-v1.2/LICENSE-NOTICE.md)).

**Confirmed:** molecular diffusivity is species/solvent/temperature dependent, and high-Schmidt scalar structure can be finer than the resolved velocity structure. **Inference:** visually fast mixing at animation resolution usually requires a declared effective/subgrid diffusivity. **Recommendation:** expose molecular/effective diffusivity in world-units squared per second, label artistic acceleration honestly, and require a resolution study. Do not implement a frame-rate-dependent color lerp.

**Do not ship as the default:** direct sRGB/RGB averaging, unconditional neighbor-color absorption, an RGB-only cache, unconstrained MacCormack/BFECC, live OpenVDB/NanoVDB storage, or an unlicensed Mixbox dependency.

## 3. User-visible capability definition

The user-visible capability should mean all of the following, not merely a multicolor surface:

- Each liquid source chooses one or more stable species IDs and nonnegative source concentrations. Two source colors can enter the same connected liquid and remain distinct when mixing is disabled or diffusivity is zero.
- Particle advection transports those concentrations through the real solver path. Concentrations survive emission, reseeding, collision, outflow, frame stepping, cache save/resume, and output publication.
- `Mixing = Off` means transport without diffusion; `Molecular` uses physical/effective coefficients in world units; `Effective (artistic/SGS)` adds an explicitly named unresolved-scale coefficient.
- The simulator reports per-species initial, injected, removed, and current mass. The closed-box residual is visible in diagnostics or validation output.
- Viewport playback displays a cheap scene-linear approximation derived from concentrations. Surface and particle outputs carry a derived color attribute; authoritative caches retain concentrations.
- Clear-dye rendering drives absorption/extinction with path-length dependence. A surface base color is labeled a preview approximation.
- A later turbid-pigment preset may use calibrated absorption plus scattering or a licensed/independently implemented pigment model. Yellow-plus-blue-to-green is an acceptance target only for such a calibrated pigment palette, not for every dye pair.
- “Color only” does not imply variable density, viscosity, surface tension, reactions, heat, phase separation, or separate meshes. If source components affect those properties, that is a separate coupled-mixture capability.

A compact first release can expose at most four interactive colorant channels while keeping the cache schema count-based and ID-based rather than hard-coding RGB. Four is a performance/product recommendation, not a scientific limit. Presets should supply names, preview swatches, absorption coefficients, diffusivity, reference concentration, and reference path length.

## 4. Physical model and terminology

### 4.1 Transport state

For liquid particle `p`, define represented liquid mass `m_p` and colorant concentration/mass fraction `c[p,k]` for species `k`.

- For the initial passive-dye capability, `c[p,k] >= 0`; a configurable total loading bound `sum_k c[p,k] <= c_max` leaves the carrier implicit. Use one common diffusivity if the channels are intended as fractions of one passive palette.
- For a future chemically meaningful mixture, explicitly include the carrier as a species and transport all mass fractions `Y[p,k]` with `sum_k Y[p,k] = 1`. Differential multicomponent diffusion is a larger model and should not be approximated by independent clamp-and-renormalize steps.
- A three-component RGB vector is not a concentration vector. Its components are display primaries, not chemical species.

The continuum passive-scalar model is

`partial_t c_k + div(u c_k) = div(D_eff,k grad(c_k)) + S_k`,

with sources/sinks `S_k` and `D_eff,k = D_molecular,k + D_unresolved,k`. IUPAC defines the diffusion coefficient through flux down a concentration gradient ([IUPAC](https://doi.org/10.1351/goldbook.D01719), [`sources/standards/iupac-diffusion.md`](sources/standards/iupac-diffusion.md)). `D` has units of length squared per time.

### 4.2 Advection versus mixing

- **Advection** moves material with the resolved velocity. Particle-authoritative concentrations remain unchanged while particles move.
- **Molecular diffusion** represents unresolved molecular transport and should use a physical coefficient transformed by the scene length scale.
- **Numerical diffusion** is unintended smoothing introduced by discretization/interpolation.
- **Subgrid or effective diffusion** is a model for unresolved stirring. It may be artistically useful but is not molecular truth.
- **Color interpolation** maps already mixed state to a visible color; it is neither advection nor species diffusion.

APIC enriches particle velocity with a local affine field and improves velocity/momentum transfer. The paper's conservation result is about momentum, not arbitrary scalar mass ([Jiang et al.](https://doi.org/10.1145/2766996), [`sources/papers/apic.md`](sources/papers/apic.md)). Therefore scalar transfer needs its own partition-of-unity and mass-budget proof.

### 4.3 Conservation and boundedness

For closed boundaries and no sources, each species budget should satisfy

`M_k = sum_p m_p c[p,k] = constant`.

Boundedness means nonnegative species concentrations, no NaN/Inf, and respect for a declared total loading or simplex. Conservation and boundedness are independent: clamping can enforce a range while creating or destroying species; a conservative high-order scheme can still overshoot. OpenFOAM separately exposes bounded transport/limiting, and its first-party manual warns that ordinary scalar transport is bounded only with a suitable scheme ([official code documentation](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/functionObjects/solvers/scalarTransport/scalarTransport.H), [`code/openfoam-244109d/scalarTransport_excerpt.H`](code/openfoam-244109d/scalarTransport_excerpt.H)). MacCormack likewise needs an extrema strategy ([Selle et al.](https://faculty.cc.gatech.edu/~jarek/papers/maccormack.pdf), [`sources/papers/maccormack.md`](sources/papers/maccormack.md)).

### 4.4 Optical models

- **Linear-light RGB:** appropriate for mixing emitted/reflected light intensities, not pigments or chemical amounts. W3C explicitly distinguishes linear-light interpolation from legacy sRGB and perceptual spaces ([W3C CSS Color 4](https://www.w3.org/TR/2026/CRD-css-color-4-20260618/), [`sources/specifications/w3c-color-4.md`](sources/specifications/w3c-color-4.md)).
- **Clear dye / Beer-Lambert:** concentration-weighted nonnegative absorption or extinction coefficients produce `T(lambda) = exp(-sigma_t(lambda) L)`. Thickness `L` matters. PBRT and OpenPBR support this interpretation ([PBRT](https://pbr-book.org/4ed/Volume_Scattering/Transmittance), [`sources/specifications/pbrt-transmittance.md`](sources/specifications/pbrt-transmittance.md); [OpenPBR](https://academysoftwarefoundation.github.io/OpenPBR/), [`sources/specifications/openpbr.md`](sources/specifications/openpbr.md)).
- **Kubelka-Munk / pigment:** appropriate for turbid, scattering paint-like media and requires absorption/scattering calibration. The Mixbox paper uses a latent representation to make plausible pigment mixing practical with RGB endpoints, but this is a derived appearance model, not the conserved state ([paper](https://doi.org/10.1145/3478513.3480549), [`sources/papers/pigment-mixing.md`](sources/papers/pigment-mixing.md)).
- **Full spectral transport/rendering:** strongest color fidelity, highest palette data, channel count, calibration, cache, and render-integration cost. It is disproportionate for an interactive first release.

## 5. Implementation technique comparison

| Technique | Fidelity | Complexity | GPU cost | Memory cost | Sparse-grid/project fit | Evidence strength | Recommendation |
|---|---|---:|---:|---:|---|---|---|
| Particle-authoritative species; no diffusion | Exact resolved advection of particle labels; no physical mixing | Low | `O(Np*K)` only at source/output | `4*K` bytes/particle (f32) | Excellent; no scalar live grid | Strong APIC/FLIP architecture evidence | **Stage 1 foundation** |
| Particle species + conservative grid diffusion + delta G2P | Best balance of resolved advection, conservation, bounded physical/effective mixing | Medium-high | `O((Np+Nfaces)*K)`; atomics/reductions | particle channels plus temporary active-node species masses/deltas | Excellent if face halo/ownership is explicit | Strong transport theory; exact limiter needs local derivation | **Default target** |
| Symmetric particle-pair diffusion | Conservative if each unordered pair applies equal/opposite exchange; easy small-scene oracle | Medium | Neighbor search `O(Nneighbors*K)`; contention | few particle scratch channels | Does not exploit already active grid; expensive at scale | Moderate paper precedent | Reference/oracle, not production default |
| Grid-authoritative first-order upwind scalar | Conservative/bounded, but visibly diffusive | Medium | face kernels each substep | persistent `K` grid channels | Fits tiles, but duplicates advection state | Strong CFD evidence | Validation fallback only |
| Semi-Lagrangian scalar on grid | Stable/simple; non-conservative and diffusive | Low-medium | one backtrace/sample per cell/species | double-buffered `K` channels | Easy, but weak conservation | Strong code evidence | Preview-only if budgets are unimportant; otherwise reject |
| MacCormack/BFECC grid scalar | Sharper than first-order SL; can overshoot and remains non-conservative unless reformulated | Medium | 2 or 3 advection passes/species plus limiter | two/three buffers | Costly across sparse halos and boundaries | Strong paper/code evidence | Alternative preview; not authoritative default |
| Flux-form conservative high-order/WENO/FIA | Strong grid scalar conservation and lower diffusion | High | multiple stencils/flux passes/species | halos and scratch buffers grow | Harder 8-cell-tile seams and GPU occupancy | Strong paper evidence | Later if grid-authoritative transport becomes necessary |
| Per-frame neighbor RGB absorption | Artist-friendly but optical/chemical state conflated; conservation depends on symmetry | Low-medium | neighbor traversal | RGB per particle | Easy product feature, weak physical semantics | Strong industry baseline, weak physical proof | Reject as solver truth |
| Scene-linear RGB averaging | Valid only for additive light intensity | Low | tiny | 3 floats | Technically easy | W3C confirms limited domain | Debug/legacy appearance only |
| Beer-Lambert from species absorption | Physically plausible clear dyes; path-length dependent | Low-medium at render/output | exponential + palette dot product | 3 RGB coefficients/species or spectral table | Excellent as derived output | Strong rendering/spec evidence | **Default optical model** |
| Calibrated Kubelka-Munk/spectral pigment | Strongest for turbid pigments within model assumptions | High | LUT or multi-wavelength coefficient mix | palette spectra plus derived fields | Output-stage fit; too heavy as live solver state | Strong papers; calibration required | Later optional mode |
| Mixbox latent model | Convincing practical pigments, compact derived operation | Medium | LUT/latent operations | latent channels/LUT | Good technically | Strong paper/product evidence; restrictive code license | Benchmark only until licensed/clean implementation |

Decision logic: transport identity and appearance are orthogonal. The recommended transport continues to work whether the palette later uses simple scene-linear preview, Beer-Lambert absorption, or a calibrated pigment model.

## 6. Primary-source evidence

1. **Hybrid particle/grid ownership.** APIC describes particles for transport/topology and the grid for mechanical updates; it adds local affine velocity and proves momentum properties ([DOI](https://doi.org/10.1145/2766996), [`local`](sources/papers/apic.md)). This supports particle-authoritative material labels, but does not certify scalar transfer.
2. **Higher order is not automatically safe.** Selle et al. show a two-pass unconditionally stable MacCormack formulation, then require extrema handling and boundary fallback ([paper](https://faculty.cc.gatech.edu/~jarek/papers/maccormack.pdf), [`local`](sources/papers/maccormack.md)). This rejects unclamped MacCormack as concentration truth.
3. **Conservation requires flux accounting.** Lentine et al. and Hirasawa et al. formulate conservative advection around fluxes/canceling face contributions ([papers](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Lentine11.pdf), [`local`](sources/papers/conservative-advection.md)). This supports equal-and-opposite diffusion face updates and explicit mass budgets.
4. **Diffusivity has physical units.** IUPAC defines `D` from flux proportional to negative concentration gradient ([term](https://doi.org/10.1351/goldbook.D01719), [`local`](sources/standards/iupac-diffusion.md)). This rejects a hidden per-frame lerp factor as the physical UI.
5. **Unresolved scalar scales are real.** passive-scalar DNS shows Schmidt-number dependence and stricter scalar resolution at high Schmidt number ([APS](https://doi.org/10.1103/PhysRevFluids.4.044607), [`local`](sources/papers/turbulent-scalar.md)). This makes accelerated mixing an explicit SGS/effective model, not a claim of resolved molecular physics.
6. **Pigment and light mixing differ.** Sochorova and Jamriska identify RGB's colored-light semantics and use a Kubelka-Munk-derived latent space for pigments ([DOI](https://doi.org/10.1145/3478513.3480549), [`local`](sources/papers/pigment-mixing.md)); W3C separately specifies linear-light and perceptual interpolation roles ([spec](https://www.w3.org/TR/2026/CRD-css-color-4-20260618/), [`local`](sources/specifications/w3c-color-4.md)).
7. **Volume appearance needs coefficients and distance.** OpenPBR derives extinction from transmission color and depth; PBRT defines transmittance through media ([OpenPBR](https://academysoftwarefoundation.github.io/OpenPBR/), [`local`](sources/specifications/openpbr.md); [PBRT](https://pbr-book.org/4ed/Volume_Scattering/Transmittance), [`local`](sources/specifications/pbrt-transmittance.md)).
8. **Cache formats can retain source channels.** OpenVDB supports named grids, metadata, and point attributes ([OpenVDB 13.0.1](https://www.openvdb.org/documentation/doxygen/overview.html), [`local`](sources/specifications/openvdb-13.0.1.md)). This is cache/interchange evidence only; its tree is incompatible with the mandated live flat leaf table.

## 7. Reference implementations and code findings

### NVIDIA Warp v1.13.0

- Inspected: `warp/examples/fem/example_apic_fluid.py` at tag `v1.13.0`.
- Exact symbols: `integrate_fraction()`, `integrate_velocity()`, `update_particles()`, and orchestration in `Example.simulate()`.
- Finding: `integrate_velocity()` reconstructs APIC velocity from particle velocity plus a velocity-gradient matrix; `update_particles()` samples grid velocity/gradient and advects particle position. This is concrete GPU-resident hybrid-transfer precedent ([upstream](https://github.com/NVIDIA/warp/blob/v1.13.0/warp/examples/fem/example_apic_fluid.py), [`local excerpt`](code/warp-v1.13.0/example_apic_fluid_excerpt.py)).
- Reuse class: **reusable idea** and **potentially reusable Apache-2.0 code**, subject to adaptation and testing. Its NanoVDB grid is **unsuitable material** for live storage because the required project contract is a flat leaf table.

### Blender/MantaFlow archive commit

- Inspected: `extern/mantaflow/preprocessed/plugin/flip.cpp` at commit `082b0253a24781ea34b69cd58e8aabb05194c212`.
- Exact symbols: `mapPartsToMAC()`, `mapLinearRealHelper()`, `mapPartsToGrid()`, `knMapLinearMACGridToVec3_FLIP::op`, and `flipVelocityUpdate()`.
- Finding: P2G accumulates values and weights, then divides by weights; FLIP G2P combines old particle velocity plus sampled grid delta with PIC interpolation. This is a useful scalar-transfer analogy, not a proof of conservative scalar delta G2P ([upstream](https://projects.blender.org/archive/blender-archive/src/commit/082b0253a24781ea34b69cd58e8aabb05194c212/extern/mantaflow/preprocessed/plugin/flip.cpp), [`local excerpt`](code/mantaflow-082b025/flip_excerpt.cpp)).
- Related official API: `source/plugin/advection.cpp` exposes `MacCormackCorrect()`, `MacCormackClamp()`, `advectSemiLagrange()`, and `cgSolveDiffusion()` in MantaFlow's generated documentation.
- Reuse class: **reusable ideas**; **potentially reusable Apache-2.0 fragments** only after provenance and GPU rewrite review. CPU/preprocessed/grid-specific code is not drop-in.

### Taichi v1.7.4 stable-fluid example

- Inspected: `python/taichi/examples/simulation/stable_fluid.py` at immutable tag `v1.7.4`.
- Exact symbols: `bilerp()`, `backtrace()`, `advect()`, `step()`.
- Finding: the same semi-Lagrangian kernel advects velocity and an RGB dye buffer, with RK3 backtrace, bilinear sampling, double buffering, and an explicit multiplicative `dye_decay`. It supports CPU/CUDA selection ([upstream](https://github.com/taichi-dev/taichi/blob/v1.7.4/python/taichi/examples/simulation/stable_fluid.py), [`local excerpt`](code/taichi-v1.7.4/stable_fluid_excerpt.py)).
- Reuse class: **reusable GPU-kernel idea**; example code is **unsuitable as authoritative mixing** because it is 2D dense-grid, non-conservative, RGB-authoritative, and intentionally decays dye.

### OpenFOAM scalar transport

- Inspected: `src/functionObjects/solvers/scalarTransport/scalarTransport.H` at exact commit `244109d2f834ae99bf2311e69c27d567416e39f7`, plus current first-party class documentation.
- Exact symbols/members: `diffusivityType`, `D()`, `subCycleMULES()`, `solveMULES()`, `D_`, `alphal_`, `alphat_`, `s_`, and `MULES_`.
- Finding: production CFD treats diffusivity choice, bounded limiting/subcycling, field state, and transport solve as distinct concerns ([upstream](https://develop.openfoam.com/Development/OpenFOAM-plus/-/blob/244109d2f834ae99bf2311e69c27d567416e39f7/src/functionObjects/solvers/scalarTransport/scalarTransport.H), [`local excerpt`](code/openfoam-244109d/scalarTransport_excerpt.H)).
- Reuse class: **reusable design evidence only**. GPL-3.0-or-later and finite-volume CPU architecture make direct code reuse unsuitable.

### Mixbox v1.2

- Inspected: tag `v1.2`, commit `ace2baebae7750643e931f170fc953a75a5a6ed3`; public conversion/latent interface.
- Exact interface concepts: RGB-to-latent, concentration-weighted latent mixing, latent-to-RGB.
- Finding: multiple pigments can be mixed through linearly weighted latent vectors even though RGB endpoints do not mix linearly ([upstream tag](https://github.com/scrtwpns/mixbox/tree/v1.2), [`local interface clip`](code/mixbox-v1.2/api_excerpt.h)).
- Reuse class: **reusable idea**; implementation is **unsuitable material without a commercial license** because the repository states CC BY-NC 4.0/non-commercial use ([local notice](code/mixbox-v1.2/LICENSE-NOTICE.md)).

No inspected implementation is proof that the proposed FluidSim path exists. The most reusable part is the architecture and invariant set; the solver-specific conservative limiter and flat-leaf GPU kernels still require implementation and validation.

## 8. Industry/product feature baseline

| Product/manual | First-party documented capability | Baseline implication | Evidence |
|---|---|---|---|
| SideFX Houdini SOP FLIP | Custom scalar/vector particle attributes and paired volume names; examples use `Cd` to dye particles and transfer it to the fluid surface | Users expect source attributes to survive FLIP and reach surface output | [SideFX](https://www.sidefx.com/docs/houdini/fluid/sopconfigflip.html), [`local`](sources/manuals/sidefx-flip.md) |
| FLIP Fluids for Blender | Source colors, `Enable Mixing`, `Mix Rate`, RGB and recommended Mixbox modes, `flip_color` surface output | A Blender-facing product should expose mixing policy, source color, and mesh attribute; RGB-only mixing is acknowledged as nonphysical | [official wiki](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-Attributes-and-Data-Settings), [`local`](sources/manuals/flip-fluids-color.md) |
| FLIP Fluids limitations | Color blending and variable viscosity do not equal full multi-fluid density/surface-tension/volumetric/separate-mesh behavior | Capability wording must stay “passive colored miscible liquid” | [official limitations](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Limitations-of-the-FLIP-Fluids-addon), [`local`](sources/manuals/flip-fluids-color.md) |
| Blender 5.1 | Named typed attributes cross point/mesh domains; Principled Volume separates color/density/absorption; liquid mesh derives from particles | Publish named point/mesh attributes and optional volume coefficients; surface interpolation is derived output | [manual family](https://docs.blender.org/manual/en/5.1/index.html), [`local`](sources/manuals/blender-5.1-output.md) |

The proposed capability meets this baseline while improving truthfulness: it retains species state and budgets instead of treating `Cd`/RGB as the chemical truth. It intentionally does not promise the larger multi-fluid features that FLIP Fluids itself lists as absent.

## 9. Fit to FluidSim architecture

Everything in this section is a **proposal**, not an observed implementation.

### 9.1 Proposed state ownership

**Particle state (authoritative):**

- `species_count` and stable palette/species IDs at solver-session level;
- structure-of-arrays `species_c[k][particle]` in `float32` for coalesced per-species passes, or packed `vec4` only as an optimized `K<=4` view;
- existing particle mass or represented volume used to convert concentration to species mass;
- optional per-particle provenance/source ID for diagnostics, not required for the physical model.

**Temporary grid state (active liquid nodes only):**

- `mix_mass` (`M_i`), the deposited carrier/liquid mass;
- `species_mass[k]` (`Q_i,k`);
- `species_delta[k]` and/or outgoing-flux scale for a two-pass limiter;
- optional active-face flags/solid fractions shared with the existing solver;
- a one-cell halo around active `8x8x8` leaves, with a canonical owner for each cross-leaf face.

**Palette/calibration state:**

- stable ID, display name and scene-linear preview swatch;
- `D_molecular` and optional effective-mixing parameters in world units;
- clear-dye RGB/spectral absorption coefficients, reference concentration and path length;
- optional scattering/Kubelka-Munk data for a later turbid mode;
- color-space/version identifier and a stable palette hash.

Solver/deep modules should own these invariants and transitions. Blender properties/operators should only adapt input and present state; they must not recreate normalization, budgeting, diffusion, or cache policy.

### 9.2 Proposed solver phases and kernels

Place mixing after particle advection/solid handling has produced the current positions and before derived output publication. It does not need to alter the pressure projection for passive dyes.

1. `species_source_apply`: initialize new particle concentrations; add injected mass to budgets.
2. `species_p2g_clear`: clear only active/halo tile channels.
3. `species_p2g_accumulate`: atomic-add `w_ip m_p` to `M_i` and `w_ip m_p c_pk` to `Q_i,k`.
4. `species_face_flux_measure`: visit every active liquid-liquid face once, compute harmonic/selected `D_face`, proposed flux, and per-node outgoing demand.
5. `species_face_flux_apply_limited`: scale outgoing fluxes so no node/species becomes negative and shared loading/simplex capacity remains valid; add equal and opposite deltas.
6. `species_g2p_delta`: distribute node `delta Q_i,k` back to particles with `a_ip = w_ip m_p / M_i`, then divide particle species-mass delta by `m_p`.
7. `species_bounds_budget_reduce`: GPU reduction for minima, maxima, totals, source/sink residuals, NaN/Inf, and limiter activity.
8. `species_preview_derive` only when viewport/output needs refresh: palette dot product plus Beer-Lambert or a cheaper fixed-path approximation.

The face phase may subcycle. For an explicit three-dimensional six-neighbor diffusion stencil, use the conservative stability guard based on `D dt / dx^2` (with `1/6` as the uniform-grid forward-Euler reference), or reduce `dt_mix`; an implicit solve avoids that limit but adds iterations, sparse reductions, convergence state, and latency. **Recommendation:** explicit conservative flux with bounded subcycling for the first release; add an implicit mode only if required effective diffusivities make subcycling dominate.

### 9.3 Proposed transfer construction

Use the same nonnegative, partition-of-unity shape weights as the liquid particle/grid transfer where possible:

`M_i = sum_p w_ip m_p`

`Q_i,k = sum_p w_ip m_p c_p,k`

`C_i,k = Q_i,k / M_i` for occupied nodes.

For each owned liquid-liquid face `(i,j)`, compute one flux `F_ij,k = -F_ji,k`. Apply limited equal-and-opposite mass deltas. Do not diffuse across air or impermeable solid faces. When boundary fractions exist, multiply by open-face area and use the appropriate distance. Canonical global cell/face coordinates—not tile-local order—must decide ownership so an `8x8x8` seam is indistinguishable from an interior face.

For G2P, distribute only the grid diffusion delta:

`delta q_p,k = sum_i (w_ip m_p / M_i) delta Q_i,k`

`c'_p,k = c_p,k + delta q_p,k / m_p`.

At each occupied node, the distribution weights sum to one, so the pre-limiter grid delta is preserved globally. The unresolved technical risk is a particle-level shared-capacity/simplex limiter that remains locally bounded and globally conservative. The first implementation should use common diffusivity/palette loading constraints that minimize this conflict, implement paired redistribution rather than independent clamping, and prove the budget executable tests before broadening to differential species diffusion.

### 9.4 Proposed source, reseeding, and sink seams

- Every source supplies a complete concentration vector; omitted means carrier-only, not the current viewport swatch.
- Birth/reseed operations reconstruct concentrations by represented mass/volume and must emit before/after per-species budget receipts.
- Numerical particle deletion must redistribute species mass before removal. Physical outflow records removed mass in the sink budget.
- Capacity growth copies all species arrays; capacity failure must stop with an actionable error rather than silently drop channels.
- Restart restores species IDs, channels, palette hash, budgets, and all state needed for the simulator's existing continuation promise.

### 9.5 Proposed cache and output channels

**Authoritative production cache:** schema/version; frame/time and scene scale; stable species table; particle positions plus concentration channels and represented mass/volume; source/sink cumulative budgets; mixing settings; palette metadata or hash; resume-related reseed/random state.

**Derived cache/output:** optional per-species sparse concentration grids; packed preview RGB; point color; surface point/corner color; optional volume `absorption`/`color` grids. Derived RGB can be regenerated and must never be the only resume state. OpenVDB is acceptable for a cache/interchange file but not for the live field representation ([OpenVDB evidence](https://www.openvdb.org/documentation/doxygen/overview.html), [`local`](sources/specifications/openvdb-13.0.1.md)).

Suggested Blender-facing names:

- surface/points: `fluidsim_color` (scene-linear float color), `fluidsim_species_0...N` where practical, `fluidsim_mix_fraction` for diagnostics;
- volumes: `density`, `color` for conventional viewport compatibility, plus `absorption` or per-species grids for a material that understands them;
- metadata: species IDs/palette hash, never inferred from attribute order alone.

Output publication should sample concentrations, then derive the display channel. Playback should present cached channels without becoming authoritative solver policy. Diagnostic overlays can show dominant species, mixedness, bounds, and budget residuals but do not certify the simulation.

### 9.6 Proposed UI settings

Domain/solver policy:

- `Enable Colorant Transport`;
- `Mixing Model`: `Off`, `Molecular Diffusion`, `Effective / Artistic Diffusion`;
- species palette list and interactive limit/status;
- per-species `Diffusivity (scene_unit^2/s)`; effective diffusion separated and clearly labeled;
- `Optical Model`: `Clear Dye Absorption`, `Surface Preview`, later `Turbid Pigment (calibrated)`;
- preview path length/exposure and surface/volume publication toggles;
- read-only diagnostics: species budgets, drift, extrema, limiter percentage, active tiles, GPU bytes/time.

Source adapter:

- stable species/concentration vector or preset;
- preview swatch derived from the selected palette;
- validation error if loading exceeds supported bound or palette ID is missing.

Changing source composition affects only future emission; changing diffusivity or optical calibration has different pending/restart/cache implications and must be classified by the solver-owned settings lifecycle rather than improvised in UI callbacks.

## 10. Performance and numerical-risk analysis

Let `Np` be live particles, `Nc` active liquid/halo nodes, `Nf` owned liquid faces, and `K` colorants.

- Particle storage: `4*K*Np` bytes in float32. `K=3` costs 12 bytes/particle; `K=4` costs 16 bytes/particle, before alignment.
- A minimal temporary node set (`M`, `Q[K]`, `delta[K]`, outgoing scale/demand `[K]`) is approximately `4*(1+3K)` bytes/node: 40 bytes/node for `K=3`, 52 for `K=4`, excluding flags, padding, and shared solver fields.
- Per `8x8x8` leaf (512 cells), that is roughly 20 KiB for `K=3` or 26 KiB for `K=4`, before halo and allocator overhead. Avoid permanently allocating this for empty leaves.
- Work is linear in channels: P2G/G2P `O(Np*K)`, diffusion `O(Nf*K)`, reductions `O((Np+Nc)*K)`.
- A two-pass face limiter adds bandwidth but avoids destructive clamping. Fusing palette preview into validation/output can save a pass, but do not fuse it into authoritative state mutation.

Primary performance risks:

1. **Atomics and nondeterminism.** P2G float atomics and face updates may vary in summation order. Do not promise bitwise CUDA repeatability; use tolerance-based budgets and invariant checks.
2. **Small tile overhead.** An `8x8x8` interior has a high halo/dispatch fraction. Batch leaves into contiguous work queues and process each cross-leaf face once.
3. **Channel divergence.** Runtime-variable `K` can reduce vectorization and complicate kernels. A `K<=4` specialized fast path plus generic cache schema is reasonable.
4. **Subcycling.** Large `D_eff`, small voxels, or large frame steps can explode explicit substeps. Report subcycle count; cap with a clear admission error or switch to a separately validated implicit mode.
5. **Reseeding drift.** Repeated particle birth/death can dominate the species budget even when diffusion is conservative.
6. **Half precision.** FP16 concentrations are tempting but threaten small species amounts and long-run budgets. Start with float32; evaluate packed/quantized cache separately from live state.
7. **Surface resampling.** High-resolution surface meshes may cost more to receive/smooth species than the solver mixing itself. Derived surface smoothing must not feed back into cached simulation truth.

Numerical risks and mitigations:

- Independent post-hoc `clamp()` destroys conservation: limit flux before application and redistribute any residual conservatively.
- Per-species bounding alone may violate total loading/simplex: use a shared-capacity limiter or explicitly transport the carrier.
- MacCormack can create extrema and fail near disparate trace regions: keep it non-authoritative or add local reversion as the paper prescribes ([paper/local](https://doi.org/10.1007/s10915-007-9166-4), [`sources/papers/maccormack.md`](sources/papers/maccormack.md)).
- Molecular diffusion may be visually negligible: expose effective diffusion honestly and run resolution convergence.
- A fixed preview path length cannot match all mesh thicknesses: label it preview; final clear-liquid rendering needs spatial absorption in a volume or a thickness-aware material.
- Beer-Lambert coefficient additivity can fail at high concentrations or interacting dyes; “calibrated physical color” requires measured mixtures.

## 11. Validation and acceptance strategy

The thresholds below are **proposed engineering gates**, not values claimed by the cited papers. Sources support the invariants and analytic forms; the tolerances must be confirmed against the actual GPU precision and scale.

| Gate | Scene/method | Proposed pass criterion |
|---|---|---|
| Constant-field transfer | Uniform concentration, one P2G/diffuse-zero/G2P cycle | max error `<=2e-6`; no new species |
| Closed-box conservation | Two/three species, 240 frames, no source/sink | relative drift per species `<=1e-5`; one transfer/diffusion step `<=1e-6` |
| Bounds/loading | Adversarial sharp interfaces and maximum allowed loading | `min(c_k)>=-2e-6`; no NaN/Inf; total loading/simplex overrun `<=5e-6` |
| Zero-diffusion advection | Periodic solid-body rotation | species totals meet budget; normalized L1 scalar error `<=5%` at named reference resolution |
| Analytic diffusion | Gaussian concentration in periodic/large box | variance follows `var(t)=var(0)+2Dt`; relative L2 error `<=2%` at 64+ cells across profile; mass drift `<=1e-5` |
| Sparse-leaf seam | Pulse centered on each tile face/edge/corner and away from seams | max concentration difference `<=1e-5`; no visible/discrete discontinuity |
| Solid/air no-flux | Dye adjacent to impermeable collider/free surface | no species crosses closed solid/air face beyond `1e-6` relative mass |
| Source/sink budget | Timed injection and physical outflow | `initial + injected - removed - final` residual `<=1e-4` per species |
| Reseeding stress | Force repeated birth/death | closed-box drift `<=1e-4`; newborns satisfy bounds |
| Species permutation | Reorder species storage/palette IDs | physical concentration fields agree `<=2e-6`; stable IDs preserve output mapping |
| Cache resume | continuous `N+M` vs save/reload at `N` | mass within `1e-5`, field L2 within `1e-4`, bounds pass; no bitwise requirement |
| GPU residency | profile/trace a normal playback substep | no device-to-host transfer inside simulation phases; allocation overflow fails explicitly |
| Scaling | profile `K=1,3,4`, fixed particles/tiles | report time/bytes; no unexplained superlinear growth or CPU fallback |
| Clear-dye endpoints | fixed lighting/camera/path-length calibration | zero concentration optically neutral; reference concentration matches recorded transmittance/target |
| Calibrated pigment mixtures | held-out binary/ternary measured swatches | mean `DeltaE00<=2`, max `<=5`; otherwise label qualitative |

Additional required checks:

- resolution study at `h` and `h/2` for a stirred interface; species totals pass both and the mixedness integral differs by at most 10% before claiming resolution-insensitive behavior;
- CPU/double-precision or particle-pair reference for tiny domains to independently check flux signs, budgets, and seam ownership;
- derivative tests for `D=0`, equal concentrations (zero flux), one species absent, one occupied node, empty neighbor tile, solid boundary, source on the frame boundary, and maximum capacity;
- visual playback test showing two sources, unmixed transport, gradual mixing when enabled, cache resume, and visibly changing derived surface/point output. Visual similarity alone cannot pass the conservation gates.

## 12. Recommended staged roadmap

### Stage 0 — contract and oracle

- Freeze terminology, units, stable species IDs, cache schema, and budget equations.
- Implement a tiny independent reference (dense CPU or symmetric pair exchange) only as a test oracle.
- Define the two-source, Gaussian diffusion, leaf-seam, reseeding, and cache-resume fixtures before GPU kernels.

Exit: expected budgets/analytic results are executable and the UI vocabulary does not conflate color interpolation with mixing.

### Stage 1 — transport without diffusion

- Add particle concentration channels, source initialization, capacity growth/copy, sink accounting, cache save/resume, and derived scene-linear preview.
- No grid concentration is persistent; no color changes after emission except through source/reseed policy.

Exit: zero-diffusion advection, source/sink, reseeding, cache, and GPU-residency gates pass.

### Stage 2 — conservative bounded molecular diffusion

- Add sparse P2G mass/species deposition, cross-leaf halo/face ownership, equal/opposite explicit fluxes, conservative delta G2P, subcycling, and invariant reductions.
- Support a common diffusivity first; add per-species coefficients only after shared-capacity/simplex behavior is proven.

Exit: constant, Gaussian, closed-box, bounds, and all leaf seam/corner gates pass at two resolutions.

### Stage 3 — production cache/output and clear-dye rendering

- Publish surface/particle attributes from concentration; store authoritative species channels and budgets; add clear-dye absorption palette/calibration and optional volume output.
- Keep RGB derived and replaceable.

Exit: cache round-trip, Blender attribute/material, path-length, and saved-playback tests pass.

### Stage 4 — effective/subgrid mixing

- Add explicitly labeled `D_unresolved` or a velocity-gradient/dispersion closure, with unit scaling, cap/subcycle admission, and a resolution study.
- Compare against the particle-pair oracle and document that the closure is an artistic/upper-bound approximation where applicable.

Exit: convergence and performance budgets are met; diagnostics report the molecular versus effective contribution.

### Stage 5 — optional calibrated turbid pigment model

- Select an implementation with acceptable license/provenance; store absorption/scattering or independently derived latent data in the palette; add measured held-out swatches.
- Keep transport unchanged.

Exit: optical mixture thresholds pass or the mode is explicitly marketed as qualitative. Mixbox code is not included without license clearance.

## 13. Dependencies and cross-lane coupling

Required shared seams, in dependency order:

1. **Particle schema/capacity/reseeding:** authoritative concentration arrays, represented mass/volume, copy/growth, birth/death receipts.
2. **Source/sink lifecycle:** stable source composition, exact emission initialization, outflow accounting, no adapter-owned defaults.
3. **Sparse grid infrastructure:** active/halo leaf discovery, global neighbor lookup, canonical cross-leaf face ownership, solid/open-face data.
4. **Dispatch/kernel inventory:** new phase ordering, channel-count specialization, scratch allocation, admission checks, profiling markers.
5. **Cache schema:** stable species table, concentration channels, budgets, palette hash/metadata, resume state and migration/version failure behavior.
6. **Output publication:** point/surface/optional volume sampling and named attributes; derived RGB/palette conversion.
7. **Settings/UI:** solver-owned setting lifecycle and restart/pending semantics; source composition adapter; read-only diagnostics.
8. **Validation/playback:** analytic and seam fixtures, cache resume, interactive changing output, GPU trace, performance profiles.

Cross-lane order: `state+invariants -> sources/reseeding -> sparse transfer/diffusion -> cache -> output/color -> UI -> full playback/performance`. Output or UI work can prototype against synthetic concentrations, but it must not certify solver completion.

Potentially coupled but out of initial scope: variable viscosity/density/surface tension, buoyancy, temperature, reactions, separated phases/meshes, and spectral renderer integration. Any of these changes the solver contract and requires a new plan.

## 14. Risks, unknowns, and rejected approaches

### Material risks/unknowns

- The conservative particle-level bound/simplex limiter for this solver's exact weights and reseeding behavior remains to be derived and tested. Partition of unity proves the unlimited delta budget, not the limited result.
- The actual particle mass/volume convention, source semantics, leaf activation policy, and existing interpolation weights were outside the permitted project evidence. Proposals must be reconciled with live code before implementation.
- Molecular coefficients may be unknown for user-selected dyes. Presets can be calibrated; arbitrary color-picker inputs cannot claim chemical accuracy.
- Beer-Lambert RGB coefficients are a three-band approximation. Strong metamerism, fluorescence, aggregation, or chemical reaction needs measured spectral/mixture data.
- High effective diffusivity may make explicit subcycling incompatible with interactive targets; an implicit sparse solve may become necessary.
- Atomic accumulation prevents bitwise repeatability. Deterministic budgets can use fixed-order reductions for tests, but production should promise tolerances only.

### Rejected approaches

- **Transport RGB/sRGB as conserved state:** RGB channels are display primaries and cannot preserve species identity or support a new optical model.
- **Average source swatches in display sRGB:** W3C identifies sRGB as neither linear-light nor perceptually uniform; it is not pigment physics.
- **Use a single “subtractive RGB” formula (CMY multiply/RYB heuristic):** cheap but palette- and illumination-independent; no general physical basis for clear dyes or turbid pigments.
- **Use Mixbox implementation directly:** strong results, but upstream code is non-commercial/CC BY-NC without a commercial license.
- **Use Kubelka-Munk for all liquids:** it models scattering turbid layers, not generic clear absorbing water.
- **Use MacCormack/BFECC without limiter:** papers/source show new extrema and boundary failure modes; it is also not automatically conservative.
- **Clamp then renormalize every particle:** enforces a range while silently changing each global species total.
- **Diffuse RGB through neighbor absorption:** matches an industry UX baseline but conflates physical transport with appearance and can be expensive/non-conservative.
- **Cache only mesh color:** cannot resume species transport, recalibrate optics, or audit conservation.
- **Use OpenVDB/NanoVDB as live state:** both are useful references/cache formats, but hierarchical tree storage violates the fixed flat-leaf contract.
- **Promise full miscible multi-fluid physics:** passive colored carrier does not cover density/rheology/surface-tension/reaction coupling.

## 15. Source index

### Papers and terminology

- Jiang et al., *The Affine Particle-In-Cell Method*, ACM TOG 34(4), 2015, [DOI](https://doi.org/10.1145/2766996), [`local clip`](sources/papers/apic.md).
- Selle et al., *An Unconditionally Stable MacCormack Method*, JSC 35, 2008, [DOI](https://doi.org/10.1007/s10915-007-9166-4), [`local clip`](sources/papers/maccormack.md).
- Lentine, Grétarsson, Fedkiw, *Mass and Momentum Conservation for Fluid Simulation*, SCA 2011, [paper](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Lentine11.pdf), [`local clip`](sources/papers/conservative-advection.md).
- Hirasawa, Kanai, Ando, *A Flux-Interpolated Advection Scheme for Fluid Simulation*, 2021, [author page](https://nhirasawa.com/publications/flux-advection/), [`local combined clip`](sources/papers/conservative-advection.md).
- Sochorova, Jamriska, *Practical Pigment Mixing for Digital Painting*, ACM TOG 40(6), 2021, [DOI](https://doi.org/10.1145/3478513.3480549), [`local clip`](sources/papers/pigment-mixing.md).
- Gotzfried et al., *Comparison of Lagrangian and Eulerian frames of passive scalar turbulent mixing*, PR Fluids 4, 2019, [DOI](https://doi.org/10.1103/PhysRevFluids.4.044607), [`local clip`](sources/papers/turbulent-scalar.md).
- IUPAC, `diffusion coefficient`, Gold Book 5th ed., [DOI](https://doi.org/10.1351/goldbook.D01719), [`local clip`](sources/standards/iupac-diffusion.md).

### Specifications and manuals

- W3C, *CSS Color Module Level 4*, CRD 2026-06-18, [specification](https://www.w3.org/TR/2026/CRD-css-color-4-20260618/), [`local clip`](sources/specifications/w3c-color-4.md).
- Academy Software Foundation, OpenPBR, [specification](https://academysoftwarefoundation.github.io/OpenPBR/), [`local clip`](sources/specifications/openpbr.md).
- Pharr, Jakob, Humphreys, *PBRT 4e: Transmittance*, [book](https://pbr-book.org/4ed/Volume_Scattering/Transmittance), [`local clip`](sources/specifications/pbrt-transmittance.md).
- OpenVDB 13.0.1, [overview](https://www.openvdb.org/documentation/doxygen/overview.html), [`local clip`](sources/specifications/openvdb-13.0.1.md).
- SideFX, Houdini SOP FLIP configuration/container, [manual](https://www.sidefx.com/docs/houdini/fluid/sopconfigflip.html), [`local clip`](sources/manuals/sidefx-flip.md).
- FLIP Fluids, Domain Attributes and Data Settings, [official wiki](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-Attributes-and-Data-Settings), [`local clip`](sources/manuals/flip-fluids-color.md).
- Blender Foundation, Blender 5.1 Manual, [manual](https://docs.blender.org/manual/en/5.1/index.html), [`local clip`](sources/manuals/blender-5.1-output.md).

### Immutable code evidence

- NVIDIA Warp `v1.13.0`, `warp/examples/fem/example_apic_fluid.py`, [upstream](https://github.com/NVIDIA/warp/blob/v1.13.0/warp/examples/fem/example_apic_fluid.py), [`code`](code/warp-v1.13.0/example_apic_fluid_excerpt.py), [`license`](code/warp-v1.13.0/LICENSE-NOTICE.md).
- Taichi `v1.7.4`, `python/taichi/examples/simulation/stable_fluid.py`, [upstream](https://github.com/taichi-dev/taichi/blob/v1.7.4/python/taichi/examples/simulation/stable_fluid.py), [`code`](code/taichi-v1.7.4/stable_fluid_excerpt.py), [`license`](code/taichi-v1.7.4/LICENSE-NOTICE.md).
- Blender/MantaFlow commit `082b0253a24781ea34b69cd58e8aabb05194c212`, `extern/mantaflow/preprocessed/plugin/flip.cpp`, [upstream](https://projects.blender.org/archive/blender-archive/src/commit/082b0253a24781ea34b69cd58e8aabb05194c212/extern/mantaflow/preprocessed/plugin/flip.cpp), [`code`](code/mantaflow-082b025/flip_excerpt.cpp), [`license`](code/mantaflow-082b025/LICENSE-NOTICE.md).
- OpenFOAM-plus commit `244109d2f834ae99bf2311e69c27d567416e39f7`, `scalarTransport.H`, [upstream](https://develop.openfoam.com/Development/OpenFOAM-plus/-/blob/244109d2f834ae99bf2311e69c27d567416e39f7/src/functionObjects/solvers/scalarTransport/scalarTransport.H), [`code`](code/openfoam-244109d/scalarTransport_excerpt.H), [`license`](code/openfoam-244109d/LICENSE-NOTICE.md).
- Mixbox `v1.2` / `ace2baebae7750643e931f170fc953a75a5a6ed3`, [upstream](https://github.com/scrtwpns/mixbox/tree/v1.2), [`interface`](code/mixbox-v1.2/api_excerpt.h), [`license`](code/mixbox-v1.2/LICENSE-NOTICE.md).

The authoritative artifact-by-artifact metadata, hashes, redistribution notes, and supported claims are in [`sources/MANIFEST.md`](sources/MANIFEST.md).

## 16. Orchestrator ingestion block

```yaml
recommended_techniques:
  transport: particle-authoritative nonnegative species concentrations
  advection: ride existing FLIP/APIC particle motion; no scalar remap for advection
  mixing: mass-weighted P2G, conservative bounded sparse face diffusion, conservative delta G2P
  optics_default: concentration-derived Beer-Lambert clear-dye absorption
  live_species_target: specialized K<=4 path with count-based extensible cache schema
alternatives:
  - symmetric particle-pair diffusion as oracle/small-scene path
  - conservative flux-form grid advection if grid-authoritative scalars become necessary
  - calibrated Kubelka-Munk or licensed pigment latent model for turbid liquids
rejected:
  - RGB/sRGB as authoritative state
  - unclamped MacCormack/BFECC
  - RGB-only cache
  - unlicensed Mixbox dependency
  - OpenVDB/NanoVDB live hierarchy
required_shared_state:
  - particle represented mass or volume
  - stable species IDs and concentration arrays
  - source/sink per-species budgets
  - active plus halo flat leaf access and global face ownership
  - palette optical/diffusion metadata and hash
new_phases_kernels:
  - species_source_apply
  - species_p2g_clear
  - species_p2g_accumulate
  - species_face_flux_measure
  - species_face_flux_apply_limited
  - species_g2p_delta
  - species_bounds_budget_reduce
  - species_preview_derive
cache_output_impact:
  authoritative: particle concentrations, mass/volume, IDs, budgets, settings, palette metadata/hash, resume state
  derived: species grids, preview RGB, surface/point color, optional absorption/color volumes
ui_impact:
  domain: enable, mixing model, diffusivity/effective diffusion, palette, optical model, publication toggles, diagnostics
  source: species/concentration vector and preset
  truth_label: effective mixing and surface-only color are approximations
strongest_validation_targets:
  - closed-box species conservation <=1e-5 over 240 frames
  - bounds/loading and no NaN/Inf
  - analytic Gaussian diffusion <=2 percent L2 error
  - 8x8x8 face edge corner seam equivalence <=1e-5
  - reseed and source/sink budget receipts
  - cache resume concentration L2 <=1e-4
  - no device-host transfer inside substeps
  - h versus h/2 mixedness difference <=10 percent
blockers:
  - derive and prove conservative particle-level shared-capacity/simplex limiter
  - reconcile proposals with live particle mass, weights, reseeding, leaf activation, and dispatch contracts
  - choose calibrated optical palette; Mixbox requires commercial license for production dependency
cross_lane_order:
  - state and invariants
  - source and reseeding lifecycle
  - sparse conservative transfer and diffusion
  - cache schema and resume
  - output and optical derivation
  - UI adapters and diagnostics
  - playback, GPU residency, performance and full acceptance
confidence:
  architecture_decision: high
  conservative_flux_core: high
  exact_bounded_delta_g2p_limiter: medium_low_until_derived
  clear_dye_optics: high_with_calibration
  arbitrary_pigment_accuracy: low_without_measured_palette
```
