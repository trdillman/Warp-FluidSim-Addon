# Density and immiscible multiphase liquids research

## 1. Research metadata

- Research question: how should a Blender 5.1 GPU-resident FLIP/APIC-style simulator represent and solve interacting incompressible liquids of different density, especially immiscible water/oil?
- Scope boundary: research and architecture proposals only; no production implementation is claimed or supplied.
- Project constraints used: simulation remains GPU-resident; interactive playback remains viable; live fields use fixed 8x8x8 sparse leaf tiles in a flat leaf table; Blender adapters do not own solver policy; persistent output, preview, production cache, and diagnostics remain separate.
- Research date: 2026-08-22 UTC.
- Evidence policy: peer-reviewed papers, theses, official manuals, and source at immutable commits. Every major evidence family has a narrow local artifact indexed in [sources/MANIFEST.md](sources/MANIFEST.md).
- Confidence: high for the two-liquid numerical architecture; medium for product-specific performance until measured on the existing sparse GPU solver; low for generalized N-phase wetting/triple-junction behavior because it should not be in the first slice.
- Labels: **Documented fact** means directly supported by a cited primary source. **Inference** connects evidence to the supplied architecture constraints. **Recommendation** is a proposed product decision, not existing behavior.

## 2. Executive conclusions

The practical first deliverable should be **two immiscible, incompressible liquids sharing one velocity field and one variable-coefficient pressure projection**, not generalized N-phase flow. Give every particle a persistent `phase_id`; derive a bounded cell phase fraction from phase-labelled particle volume; maintain a signed-distance/geometry channel only near the internal interface; derive density and viscosity from the bounded fraction each substep; and enforce pressure/surface-tension jumps with a Ghost-Fluid or equivalently balanced face discretization.

The projection must change from a constant-coefficient Poisson solve to

`div((dt / rho) grad(p)) = div(u*)`, with the same interface-aware face coefficient used in pressure assembly and velocity correction. This is the defining numerical requirement for material density contrast, supported by sharp-interface work and MultiFLIP ([Kang–Fedkiw–Liu](https://physbam.stanford.edu/papers/cam1999-27.pdf), [local clip](sources/S15-kang-fedkiw-liu-2000-jumps.md); [Boyd](https://www.cs.ubc.ca/~rbridson/docs/boyd_msc.pdf), [local clip](sources/S02-boyd-2011-multiflip.md)). A density attribute without this solve is only rendering metadata.

For interface transport, a **CLSVOF-like hybrid** is the strongest end-state: conservative bounded fraction for phase volume, plus level-set or reconstructed geometry for normals, curvature, jump placement, and meshing ([Sussman–Puckett](https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/APPEARED/2000/MSS-EGP_2000.pdf), [local clip](sources/S03-sussman-puckett-2000-clsvof.md)). A particle-level-set/label-derived signed field with explicit volume correction is a lower-complexity intermediate, but level set alone should not be the final immiscibility claim because it loses phase volume.

Use one shared MAC velocity for the first two-liquid slice. MultiFLIP proves that separate per-phase velocity grids can work, but it doubles important grids and assumes an inviscid/free-slip unresolved interface; that is a poor default for viscous water/oil on a sparse interactive GPU budget ([Boyd](https://www.cs.ubc.ca/~rbridson/docs/boyd_msc.pdf), [local clip](sources/S02-boyd-2011-multiflip.md)). APIC remains useful for low-dissipation particle/grid transfer, but APIC itself does not prevent phase mixing ([Jiang et al.](https://alexey.stomakhin.com/research/siggraph2015_apic.pdf), [local clip](sources/S09-jiang-2015-apic.md)).

Surface/interfacial tension, static contact angle, droplets, and entrainment should follow only after density projection and phase-volume retention pass. Interfacial tension is a **pair property** `sigma_AB`, distinct from each liquid’s free-surface tension against gas. Contact angle is also pair-specific and needs reliable near-wall interface geometry ([Afkhami–Bussmann](https://web.njit.edu/~shahriar/download/AB_IJNMF09.pdf), [local clip](sources/S16-afkhami-2009-contact-angle.md)). Dynamic contact-line models, surfactants/Marangoni stress, phase change, compressible entrained gas, and generalized triple-junction physics are out of the first slice.

## 3. User-visible capability definition

### Required product truth

“Variable-density mixed liquid” and “immiscible multiphase liquid” are different claims. Official FLIP Fluids documentation makes this boundary explicit: its per-source density feature floats and sinks but is not suitable for insoluble oil/water separation ([official manual](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-World-Settings), [local clip](sources/S12-flip-fluids-1.8.8-baseline.md)). Houdini’s FLIP documentation likewise warns that it does not explicitly track the boundary between density variants ([SideFX](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html), [local clip](sources/S11-sidefx-houdini22-baseline.md)).

The proposed user-visible contract is:

- Two independently sourceable liquid materials, each with density and dynamic/kinematic viscosity settings.
- Persistent material identity (`phase_id`) through emission, advection, reseeding, collision, deletion, cache, playback, and output.
- A sharp, bounded internal interface that remains identifiable under rest, stratification, stirring, splashing, and resolved droplet formation.
- Density-driven buoyancy/stratification from the momentum equation and projection, not an artistic per-particle offset.
- Optional pairwise liquid–liquid interfacial tension, plus distinct liquid–gas surface-tension values.
- Optional static liquid–solid contact angle per material/solid pair after the base interface is proven.
- Separate or material-tagged surface output so the internal interface and each external liquid surface can be shaded.
- Diagnostics that expose phase volume, mixed-interface cells, pressure iterations/residual, minimum/maximum density, capillary-limited substeps, and interface mass drift.

An equal-density two-phase test must remain useful: phase identity should persist even when buoyancy gives no separation. Conversely, unequal density without persistent interface identity must be labeled mixed variable-density liquid, not immiscible.

### Deliberate exclusions from the first capability

No first-slice promise is made for arbitrary N phases, phase change, dissolution/diffusion, reactions, surfactants, Marangoni effects, compressible bubbles, dynamic contact-angle hysteresis, or physically exact triple-line behavior.

## 4. Physical model and terminology

### Governing model

For each incompressible material region, the shared-velocity model is

`rho (du/dt + u dot grad(u)) = -grad(p) + div(mu (grad(u)+grad(u)^T)) + rho g + f_sigma`,  
`div(u) = 0`.

Let bounded phase fraction `alpha=1` be liquid A and `alpha=0` liquid B. A cell-centered mixture closure is

`rho(alpha) = alpha rho_A + (1-alpha) rho_B`,  
`mu(alpha) = alpha mu_A + (1-alpha) mu_B`.

Popinet uses these interpolations with geometric VOF and balanced surface tension ([paper](https://www.ljll.fr/~frey/papers/interfaces-free_surfaces/Popinet%20S.%2C%20An%20accurate%20adaptive%20solver%20for%20surface-tension%20driven%20interfacial%20flows.pdf), [local clip](sources/S07-popinet-2009-balanced-vof.md)). **Recommendation:** compute the pressure face coefficient from an interface-aware/harmonic representation of `1/rho`; do not simply average densities across a cut face. The exact coefficient policy must be verified for symmetry, positivity, and jump consistency.

The projection is

`u^(n+1) = u* - dt (1/rho) grad(p)`,  
`div((dt/rho) grad(p)) = div(u*)`.

For an inviscid constant-interfacial-tension interface, the Laplace pressure jump is `[p] = sigma kappa`. With viscosity contrast, the full normal/tangential stress jump must be respected; the simple pressure jump is not universally sufficient ([Kang–Fedkiw–Liu](https://physbam.stanford.edu/papers/cam1999-27.pdf), [local clip](sources/S15-kang-fedkiw-liu-2000-jumps.md)).

### Terminology

- **Phase/material label:** discrete persistent particle identity. It is not display color.
- **Phase fraction/color function `alpha`:** bounded Eulerian cell fraction used for conservation and material properties.
- **Level set `phi`:** signed interface-distance-like geometry channel; useful for normals, curvature, and meshing but not intrinsically conservative.
- **VOF/PLIC:** conservative volume fraction plus geometric interface reconstruction.
- **CLSVOF:** coupled level set and VOF, using fraction for conservation and level set/reconstruction for geometry.
- **Phase field:** a thermodynamic diffuse interface, commonly Cahn–Hilliard; materially different from a numerical conservative level set ([Jacqmin](https://doi.org/10.1006/jcph.1999.6332), [local clip](sources/S06-jacqmin-1999-phase-field.md)).
- **Ghost Fluid Method (GFM):** modifies interface-crossing stencils to impose jumps sharply ([Fedkiw et al.](https://physbam.stanford.edu/~fedkiw/papers/cam1999-31.pdf), [local clip](sources/S05-fedkiw-1999-ghost-fluid.md)).
- **CSF:** Continuum Surface Force, spreading capillarity over a finite interface thickness ([Brackbill et al.](https://doi.org/10.1016/0021-9991(92)90240-Y), [local clip](sources/S04-brackbill-1992-csf.md)).
- **Miscible:** species can interpenetrate/diffuse; a scalar concentration and diffusion model may be appropriate.
- **Immiscible:** material identity and a boundary persist; numerical mixing must be bounded and measured.
- **Entrainment:** one liquid is carried into another as resolved sheets/ligaments/droplets. Subgrid foam/spray is not the same as resolved liquid–liquid droplets.

Gravity is the same acceleration `g` for both materials. Density-driven buoyancy emerges from `rho g` and the variable-density pressure response. **Recommendation:** do not add a second arbitrary “heavy particles fall faster” force.

## 5. Implementation technique comparison

| Technique | Fidelity | Complexity | GPU cost | Memory cost | Sparse-grid/project fit | Evidence strength | Recommendation |
|---|---|---:|---:|---:|---|---|---|
| Particle label only + blended density | Low for immiscibility; useful for provenance | Low | Low | ~1–4 B/particle | Excellent | Industry manuals show mixing boundary | Use only as Stage 1 variable-density diagnostic, never final immiscibility |
| Particle-derived level set / particle level set | Good geometry and droplets; volume drift risk | Medium | Medium | `phi` narrow band + particle labels | Good with halo tiles | Strong graphics literature | Acceptable intermediate with explicit phase-volume correction |
| Geometric VOF/PLIC | Strong phase-volume conservation and sharp interface | High | Medium–high; branching/reconstruction | `alpha`, normals/reconstruction scratch | Moderate; tile halos and face-flux exchange required | Strong CFD literature | Strong option if GPU reconstruction complexity is affordable |
| CLSVOF-like fraction + SDF | High conservation plus smooth geometry | High | High | `alpha`, `phi`, normals, curvature, scratch | Good if narrow-band geometry and sparse flux exchange are carefully designed | Strong peer-reviewed evidence | **Recommended end-state for two liquids** |
| Conservative level set | Better conservation than classical LS, simpler than PLIC | Medium | Medium | one diffuse scalar + reinitialization scratch | Good | Strong paper evidence, but parameter-sensitive | Alternative if PLIC cost blocks Stage 2; validate diffusion/compression bias |
| Cahn–Hilliard phase field | High for intended diffuse-interface thermodynamics/wetting | Very high | High: extra stiff PDE/iterations | order parameter, chemical potential, solver scratch | Moderate; wide halos and stiffness conflict with interactivity | Strong research evidence | Reject as default; reserve for explicitly diffuse/miscible regimes |
| GFM pressure jump | Sharp Laplace/stress jump | High | Medium: cut-face metadata and stencil branches | interface-face fraction/jump scratch | Good but requires consistent tile-neighbor access | Strong peer-reviewed evidence | **Recommended for sharp interface** |
| Balanced CSF | Good when curvature and force/pressure gradients are co-located | Medium | Medium | curvature/normals/force | Good | Strong peer-reviewed evidence | Alternative or complement; must pass static-droplet spurious-current test |
| One shared MAC velocity | Correct no-slip normal coupling; natural viscous two-liquid baseline | Medium | Baseline-plus | No duplicate velocity grid | Excellent | Classical incompressible multiphase basis | **Recommended first slice** |
| MultiFLIP two velocity grids | Energetic free-slip two-phase motion | High | High bandwidth and transfer cost | roughly another velocity grid and weights | Weak–moderate on sparse budget | Strong thesis/paper evidence with explicit limitations | Reject for first slice; reconsider only for resolved gas/free-slip requirement |
| Separate pressure solve per phase | Incorrect global incompressibility/interface coupling risk | High | High | duplicate pressure state | Poor | No supporting baseline for intended shared interface | Reject |
| One level set per material (regional LS) | Supports arbitrary N and triple points | Very high | Grows with phase count | O(N) regional fields | Poor for first sparse slice | Strong graphics evidence | Defer until two-phase acceptance is complete |
| Dense FFT two-phase pressure | Fast on rectangular dense domains | Medium | Excellent dense throughput | Dense domain arrays | Incompatible with flat sparse live field contract | Strong FluTAS evidence | Reuse sequencing ideas only; reject architecture |
| Tree AMR VOF | Excellent adaptive interface focus | Very high | Irregular | Tree metadata | Explicitly incompatible with fixed flat leaf table | Strong Popinet evidence | Reject storage approach; reuse numerical tests/ideas |

## 6. Primary-source evidence

### Sharp variable-density projection

**Documented fact:** Ghost-Fluid and boundary-condition-capturing methods impose jumps through interface-crossing discretizations; the incompressible stress conditions include continuous velocity and normal/tangential stress balance ([Fedkiw et al.](https://physbam.stanford.edu/~fedkiw/papers/cam1999-31.pdf), [local](sources/S05-fedkiw-1999-ghost-fluid.md); [Kang et al.](https://physbam.stanford.edu/papers/cam1999-27.pdf), [local](sources/S15-kang-fedkiw-liu-2000-jumps.md)). MultiFLIP gives a graphics-oriented instance combining two phase velocity representations with one variable-density GFM projection ([thesis](https://www.cs.ubc.ca/~rbridson/docs/boyd_msc.pdf), [local](sources/S02-boyd-2011-multiflip.md)).

**Inference:** the current pressure system must expose face-wise coefficients, cut-interface fractions/jumps, consistent RHS assembly, and matching velocity correction. A constant Laplacian with density applied after the solve cannot reproduce the governing projection.

### Interface conservation and reconstruction

**Documented fact:** classical level-set geometry is convenient but loses volume; VOF is conservative; CLSVOF couples their strengths ([Sussman–Puckett](https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/APPEARED/2000/MSS-EGP_2000.pdf), [local](sources/S03-sussman-puckett-2000-clsvof.md)). Regional particle level sets support many interacting liquids and topology change ([Losasso et al.](https://physbam.stanford.edu/~fedkiw/papers/stanford2006-02.pdf), [local](sources/S01-losasso-2006-multiple-liquids.md)).

**Recommendation:** transport particle identity and conservative `alpha`; rebuild/reinitialize `phi` only in an internal-interface band. This prevents independently advected density, labels, and geometry from drifting into contradictory material states.

### Surface tension and wetting

**Documented fact:** CSF converts interfacial tension to a volumetric force; balanced geometric VOF can obtain machine-accuracy static equilibrium when pressure gradient, capillary force, and curvature are discretely compatible ([Brackbill et al.](https://doi.org/10.1016/0021-9991(92)90240-Y), [local](sources/S04-brackbill-1992-csf.md); [Popinet](https://doi.org/10.1016/j.jcp.2009.04.042), [local](sources/S07-popinet-2009-balanced-vof.md)). Static contact angles can be imposed geometrically using interface normals/ghost heights, with limitations near extreme angles ([Afkhami–Bussmann](https://web.njit.edu/~shahriar/download/AB_IJNMF09.pdf), [local](sources/S16-afkhami-2009-contact-angle.md)).

**Recommendation:** implement `sigma_AB=0` first, then constant `sigma_AB` with a sharp/balanced formulation, then static contact angle. Treat machine accuracy as a literature reference, not an initial acceptance threshold.

### Particle-grid transfer

**Documented fact:** APIC carries a local affine velocity per particle and conserves linear/angular momentum under its transfer assumptions ([Jiang et al.](https://alexey.stomakhin.com/research/siggraph2015_apic.pdf), [local](sources/S09-jiang-2015-apic.md)). MultiFLIP demonstrates phase-labelled particles, reconstructed interfaces, phase transfer, and one combined projection, but also adds duplicate grids and graphics-oriented mixing/volume repairs ([Boyd](https://www.cs.ubc.ca/~rbridson/docs/boyd_msc.pdf), [local](sources/S02-boyd-2011-multiflip.md)).

**Recommendation:** keep the existing FLIP/APIC transfer family and make P2G/G2P phase-aware without duplicating full velocity grids in the first slice.

### GPU viability

**Documented fact:** FluTAS performs two-fluid VOF/property updates, force application, GPU pressure solution, and velocity correction on an accelerator ([paper](https://www.diva-portal.org/smash/get/diva2%3A1660324/FULLTEXT01.pdf), [local](sources/S10-flutas-2022-gpu-two-phase.md); [pinned source clips](code/flutas-two-phase-main.f90.clip)). **Inference:** GPU residency is feasible, but its dense arrays/FFT are not a drop-in design for fixed sparse leaves.

## 7. Reference implementations and code findings

### OpenFOAM 13 — commit `18870c24d21c6b982e2cdec27b2f59738cca5f90`

Inspected exact files/functions:

- [`src/twoPhaseModels/interfaceProperties/interfaceProperties.C`](https://github.com/OpenFOAM/OpenFOAM-13/blob/18870c24d21c6b982e2cdec27b2f59738cca5f90/src/twoPhaseModels/interfaceProperties/interfaceProperties.C): `correctContactAngle`, `calculateK`, `sigmaK`, `surfaceTensionForce`, `nearInterface`; local clip [code/openfoam-interfaceProperties.C.clip](code/openfoam-interfaceProperties.C.clip).
- [`applications/modules/incompressibleVoF/incompressibleVoF.C`](https://github.com/OpenFOAM/OpenFOAM-13/blob/18870c24d21c6b982e2cdec27b2f59738cca5f90/applications/modules/incompressibleVoF/incompressibleVoF.C): `prePredictor`, `pressureCorrector`; local clip [code/openfoam-incompressibleVoF.C.clip](code/openfoam-incompressibleVoF.C.clip).
- License notice: [code/OpenFOAM-LICENSE.md](code/OpenFOAM-LICENSE.md).

Reusable ideas: curvature from phase-fraction normals; explicit contact-angle correction before curvature; phase-fraction mass flux before pressure correction; near-interface masks. Potentially reusable code: none recommended without GPL compatibility review. Unsuitable material: OpenFOAM’s CPU object/mesh field architecture and GPL code are not a GPU Warp sparse-tile implementation.

### MantaFlow — commit `494445143507296d16e467160c1889be657f8f74`

Inspected exact files/functions:

- [`source/plugin/flip.cpp`](https://github.com/thunil/mantaflow/blob/494445143507296d16e467160c1889be657f8f74/source/plugin/flip.cpp): `knMapLinearVec3ToMACGrid`, `mapPartsToMAC`, `mapPartsToGrid`, `knMapLinearMACGridToVec3_PIC`, `knMapLinearMACGridToVec3_FLIP`, `flipVelocityUpdate`; local clip [code/mantaflow-flip.cpp.clip](code/mantaflow-flip.cpp.clip).
- [`source/plugin/pressure.cpp`](https://github.com/thunil/mantaflow/blob/494445143507296d16e467160c1889be657f8f74/source/plugin/pressure.cpp): `solvePressureSystem`, `correctVelocity`, `solvePressure`, `MakeLaplaceMatrix`, `ApplyGhostFluidDiagonal`, `knCorrectVelocityGhostFluid`; local clip [code/mantaflow-pressure.cpp.clip](code/mantaflow-pressure.cpp.clip).
- License notice: [code/MantaFlow-LICENSE.md](code/MantaFlow-LICENSE.md).

Reusable ideas: old-grid delta FLIP update, PIC blend, safe particle-grid normalization, GFM hook placement, pressure solve/correction sequence. Potentially reusable code: Apache-2.0 makes carefully adapted transfer utilities more legally practical than the GPL references, subject to notice and architecture review. Unsuitable/misleading material: this inspected pressure path builds a baseline constant-coefficient Laplacian, not the required material variable-density projection. `implicitdensityprojection.cpp` concerns particle-number-density regularization, not physical liquid density.

### FluTAS — commit `598210616bebd51f7d51f61455f196e6f3479916`

Inspected exact files/functions:

- [`src/vof.f90`](https://github.com/Multiphysics-Flow-Solvers/FluTAS/blob/598210616bebd51f7d51f61455f196e6f3479916/src/vof.f90): `advvof`, `cmpt_vof_flux`, `cmpt_nor_curv_1o`, `cmpt_nor_curv_2o`, `cmpt_d_thinc`; local clip [code/flutas-vof.f90.clip](code/flutas-vof.f90.clip).
- [`src/apps/two_phase_inc_isot/main__two_phase_inc_isot.f90`](https://github.com/Multiphysics-Flow-Solvers/FluTAS/blob/598210616bebd51f7d51f61455f196e6f3479916/src/apps/two_phase_inc_isot/main__two_phase_inc_isot.f90): calls to `advvof`, `update_property`, `grav_tw_src`, `pres_tw_src`, `surft_src`, `solver_gpu`, `correc`; local clip [code/flutas-two-phase-main.f90.clip](code/flutas-two-phase-main.f90.clip).
- License notice: [code/FluTAS-LICENSE.md](code/FluTAS-LICENSE.md).

Reusable ideas: bounded indicator update before material properties; accelerator-resident force/pressure/correction sequencing; separate curvature scratch. Potentially reusable code: MIT-licensed snippets may be legally more adaptable with notices, but Fortran/OpenACC code is not directly suitable for Warp kernels. Unsuitable material: dense domain decomposition and FFT pressure solution conflict with the flat sparse leaf-table contract; the reference-density pressure split is not evidence for a general sparse variable-coefficient solver.

## 8. Industry/product feature baseline

| Product/version | First-party documented capability | Boundary relevant to this project |
|---|---|---|
| Houdini 22 FLIP | Per-particle/spatial density; official water/oil workflow | Solver docs say it does not track the boundary and can overmix ([SideFX](https://www.sidefx.com/docs/houdini/fluid/sopdensmult.html), [local](sources/S11-sidefx-houdini22-baseline.md)) |
| Houdini 22 Vellum Fluid Phases | More than two phases; phase-specific viscosity/surface tension; density ordering | Strong separation baseline, but Vellum is not FLIP and includes artistic behavior ([SideFX](https://www.sidefx.com/docs/houdini/vellum/fluidphases.html), [local](sources/S11-sidefx-houdini22-baseline.md)) |
| Blender 5.1 Mantaflow UI/API | Domain viscosity and surface tension | No documented per-flow physical density or phase ID; absence is an API/manual inference ([Blender](https://docs.blender.org/api/5.1/bpy.types.FluidDomainSettings.html), [local](sources/S13-blender-5.1-fluid-baseline.md)) |
| FLIP Fluids 1.8.8 | Per-source density; `flip_density` output; floating/sinking | Explicitly not suitable for immiscible oil/water; pressure solve may need more substeps ([manual](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-World-Settings), [local](sources/S12-flip-fluids-1.8.8-baseline.md)) |
| Autodesk Bifrost 2024 docs | Physical density per emitter; viscosity/surface tension | No documented sharp phase/interface contract found ([Autodesk](https://help.autodesk.com/cloudhelp/2024/ENU/Maya-Bifrost/files/GUID-FC482537-1329-414D-9D36-787CD9678D5B.htm), [local](sources/S14-other-industry-baselines.md)) |
| Cinema 4D 2026 liquid particles | Target Density for ordering; Mixture ID for separation; viscosity/tension | Strong artist-facing identity-vs-density UI baseline, not a documented sharp-interface Navier–Stokes solver ([Maxon](https://help.maxon.net/c4d/2026/en-us/Content/html/Liquid_Particles_Introduction.html), [local](sources/S14-other-industry-baselines.md)) |
| RealFlow 10 Dyverso | Interacting domains permit oil/water setup | Legacy and weaker than current comparators ([Next Limit](https://nextlimitsupport.atlassian.net/wiki/spaces/rf2016docs/pages/5497519), [local](sources/S14-other-industry-baselines.md)) |

The minimum competitive truth is therefore not merely a density field. A compelling capability needs separate phase identity, interface retention, phase-specific material parameters, renderable material output, and diagnostics revealing conditioning/mass conservation.

## 9. Fit to FluidSim architecture

Everything in this section is a **proposal**, not observed implementation.

### Proposed live fields

- `phase_fraction` (`f32`, cell-centered, bounded `[0,1]`) for the two liquid materials.
- `interface_phi` (`f32`, narrow band) for geometry/jump placement, if CLSVOF-like reconstruction is selected.
- `inv_density_face_{u,v,w}` derived or cached for pressure assembly/correction.
- `interface_normal` and `curvature` as transient narrow-band scratch, not necessarily persistent.
- `phase_volume_accum` and `phase_weight_accum` as transient P2G buffers.
- `interface_face_fraction`/jump metadata for cut pressure faces.
- Optional `viscosity_face` for an active variable-viscosity solve.
- Tile-level flags: contains A, contains B, contains internal interface, halo/guard.

The fixed 8x8x8 tile has 512 cells. One `f32` scalar costs 2 KiB per active tile before allocator/alignment overhead. Persisting `alpha`, `phi`, `rho`, and `kappa` would therefore cost about 8 KiB/tile; deriving `rho` and keeping curvature transient can halve persistent overhead. **Recommendation:** allocate geometry/scratch only for interface tiles plus the stencil guard band.

### Proposed particle attributes

- `phase_id: uint8` logically, likely packed/aligned to 4 bytes in the existing structure-of-arrays policy.
- Existing mass/volume semantics must be made explicit; use per-particle volume or phase rest volume for conservative deposition.
- Existing velocity and APIC affine state remain shared.
- Optional `source_material_id` only if provenance must be distinct from physical phase; never overload color as phase.

### Proposed solver phases/kernels

1. `validate_material_table` / upload two-material constants.
2. `bin_particles_by_leaf_and_phase`.
3. `deposit_phase_volume` and `normalize_phase_fraction`.
4. `reconstruct_internal_interface` / `reinitialize_interface_phi` in interface tiles.
5. `classify_interface_faces_and_activate_halo`.
6. `update_material_coefficients` (or derive on demand).
7. Existing phase-aware P2G/APIC transfer and force phase.
8. `assemble_variable_density_pressure` with interface jump terms.
9. Existing sparse PCG/preconditioner phases extended to variable coefficients.
10. `apply_variable_density_pressure_gradient` using the identical face coefficients/jumps.
11. Optional variable-viscosity solve.
12. `advect_interface_fraction`, boundedness/volume correction, then phase-preserving G2P/reseed.
13. Later stages: `compute_interface_normals_curvature`, `apply_interfacial_tension`, `enforce_static_contact_angle`.

Custom-kernel and dispatch integration should add explicit phase contracts; no Blender adapter should decide reconstruction, density blending, jump, iteration, or substep policy.

### Proposed cache channels and outputs

- Production cache: schema-versioned material table, particle `phase_id`, per-phase particle volume/mass needed for restart, `phase_fraction` or reconstructible equivalent, and internal-interface geometry when authoritative restart/playback needs it.
- Preview playback: material-tagged particles/surfaces sufficient for fast display, clearly non-authoritative.
- Mesh/output: separate phase surfaces or one mesh with `phase_id`, `material_index`, `density`, `viscosity`, and `interface_kind` (`liquid_air`, `liquid_liquid`). A union-only outer surface cannot show an internal oil/water interface.
- Diagnostics: per-phase volume/particle count/center of mass, interface cell count, `alpha` min/max, pressure residual/iterations, density-ratio estimate, substep limiter reason, spurious-current metric.

### Proposed UI settings

- Mode: Single Liquid / Variable-Density Mixed / Two Immiscible Liquids.
- Two named material slots with density and viscosity, physical units and validated positive ranges.
- Pair setting `liquid_liquid_surface_tension`; separate free-surface tension per liquid.
- Static contact angle disabled by default until supported; pair it with a solid/material target.
- Interface method shown as advanced/diagnostic, not reconstructed in the adapter.
- Output toggles for per-phase particles, outer surfaces, and internal interface.
- Warnings for high coefficient ratio, pressure non-convergence, excessive mixed-interface fraction, unresolved capillary time step, or cache schema incompatibility.

### Proposed validation seams

- Material table -> solver settings snapshot.
- Source emission -> particle phase identity.
- P2G phase volume -> bounded/conservative `alpha`.
- `alpha/phi` -> face coefficient/jump.
- Pressure assembly -> correction coefficient identity.
- Reseed/delete -> per-phase volume accounting.
- Solver state -> preview/cache/output separation.
- Cache read/write -> exact phase ID and material-table round trip.

## 10. Performance and numerical-risk analysis

### GPU and sparse-grid costs

The main new costs are phase-volume atomics/reduction, interface reconstruction, halo activation, variable-coefficient pressure iterations, and internal-interface meshing. Sorting/binning particles by leaf and phase can improve locality and enable block reductions before global atomics. Interface kernels should launch only over interface tiles, while pressure still spans the active incompressible union of both liquids.

Interface gradients/curvature need neighbor data. **Recommendation:** activate at least the stencil-required tile guard around every internal-interface tile and exchange face/edge/corner halo values deterministically. With 8-cell tile width, activating one whole neighboring tile can be much larger than a one-cell stencil; track this amplification in profiles.

Duplicating every velocity grid per phase is a large bandwidth/storage multiplier and undermines interactive sparse scaling. One cell scalar is 2 KiB/tile; three face velocities plus weights and old-grid transfer state are already much larger. This supports the shared-velocity choice.

### Conditioning and stability

- Large density ratios make the variable-coefficient elliptic problem harder; FLIP Fluids’ official docs independently warn that variable density can stress or fail its pressure solve ([manual](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-World-Settings), [local](sources/S12-flip-fluids-1.8.8-baseline.md)). Track iterations and residual by ratio.
- Explicit capillary forces impose a severe time scale proportional to `sqrt(rho_min dx^3 / sigma)` up to method-dependent constants. Substepping, semi-implicit treatment, or both may be required.
- Curvature noise creates parasitic currents. A balanced force/jump removes discretization imbalance only if pressure and capillary terms use the same face geometry; it cannot repair bad curvature.
- Variable viscosity may require an implicit solve at high viscosity. Large viscosity contrast changes stress jumps and should not be hidden inside a scalar average.
- Reconstructing `alpha` independently from particle labels can cause source/reseed volume impulses. The volume ownership and correction policy must be single-sourced.
- Numerical entrainment depends on resolution. Droplets smaller than roughly the reconstruction stencil cannot be claimed as physically resolved; preserve them as labelled particles only if the rendering/output contract describes the limitation.
- GPU atomics, compaction, and particle ordering imply numerical nondeterminism. Acceptance should use conservation/error envelopes, not bitwise identity.

### Memory discipline

Keep `phase_id` in particle SoA, `alpha` persistent for active liquid tiles, and `phi`/normals/curvature transient or narrow-band. Avoid persisting separately advected `rho`: derive it from the authoritative bounded phase fraction/material table to prevent state divergence. Pressure preconditioner memory may dominate; measure before enabling more geometry scratch.

## 11. Validation and acceptance strategy

The following are proposed tests. Source-backed exact relationships are distinguished from initial thresholds that must be calibrated.

1. **Identity lifecycle:** two sources with equal density but different phase ID. Emit, advect, collide, reseed, cache, reload, and play back. Acceptance proposal: 100% surviving particles retain their exact phase ID and material-table binding; no fallback to display color.
2. **Bounded/conservative transport:** translating/rotating interface and a notched-body advection case. Acceptance: `0 <= alpha <= 1` after every substep; report per-phase volume drift at multiple resolutions. Do not set a product percentage until a baseline convergence study exists.
3. **Hydrostatic two-layer column:** pressure is piecewise linear with source-backed slope `dp/dz = -rho_i g`; velocity should converge toward zero and the interface remain stationary. Fit each phase slope and report normalized slope/intercept/interface error.
4. **Stable stratification:** lighter liquid above heavier liquid at rest. Track maximum velocity, interface height, and center-of-mass drift.
5. **Rayleigh–Taylor inversion:** heavy over light. Verify physically correct early center-of-mass exchange and grid convergence; no arbitrary analytic late-time target is claimed.
6. **Static droplet/Laplace law:** for radius `R`, compare pressure jump with `sigma/R` in 2-D or `2 sigma/R` in 3-D. Measure normalized pressure error and maximum spurious capillary number/current over refinement. Popinet reports machine-accuracy equilibrium with exact balanced ingredients ([paper](https://doi.org/10.1016/j.jcp.2009.04.042), [local](sources/S07-popinet-2009-balanced-vof.md)); use that as a strongest-reference target, not a first-release promise.
7. **Capillary oscillation:** compare oscillation frequency and viscous damping against the appropriate small-amplitude analytic mode; demonstrate temporal and spatial convergence.
8. **Rising bubble benchmark:** reproduce Hysing case definitions and compare circularity, center of mass, and mean rise velocity to published reference curves ([paper](https://doi.org/10.1002/fld.1934), [local](sources/S08-hysing-2009-bubble-benchmark.md)). Require convergence/overlap in the moderate case; use an envelope after breakup in the difficult case.
9. **Sessile drop/contact angle:** after the wetting stage, measure the fitted equilibrium angle against the requested angle and test the documented robust middle range first; Afkhami–Bussmann note limitations near extreme angles ([paper](https://doi.org/10.1002/fld.1974), [local](sources/S16-afkhami-2009-contact-angle.md)).
10. **Stirred immiscibility:** compare (a) equal density/same phase, (b) unequal density/same mixed regime, and (c) unequal density/different phase. Record phase center of mass, top/bottom occupancy, phase volume, and fraction of cells with `epsilon < alpha < 1-epsilon`. Only case (c) may be called immiscible.
11. **Density-ratio matrix:** ratios 1, 10, 100, and 1000 with identical geometry. Measure PCG iterations, residual, divergence after correction, substeps, and frame time. These are stress points, not claimed water/oil ratios.
12. **Coefficient/jump unit seam:** for a one-dimensional cut face, compare assembled matrix/RHS and corrected velocity with the analytic GFM flux. Verify symmetry/positive definiteness where the formulation promises it.
13. **Sparse-boundary seam:** place the interface across every face/edge/corner of adjacent 8x8x8 leaves. Compare with the identical interface translated inside one leaf; values must agree within floating-point tolerance selected by the implementation.
14. **GPU residency:** telemetry/profiling must show no full live field or particle round trip to the CPU during substeps; only bounded counters/profile readback is allowed.
15. **Interactive budget:** profile frame time, peak GPU memory, active/interface/halo tiles, and pressure iterations against the same single-liquid scene. Set the release budget only after a vertical-slice measurement; no unsupported performance threshold is invented here.
16. **Cache/output:** round-trip authoritative phase state and validate separately that preview data remains non-authoritative. Confirm internal-interface material assignment and outer-surface material identity.

## 12. Recommended staged roadmap

### Stage 0 — contract and measurement

Define immutable two-material table semantics, phase IDs, source assignment, output/cache schema, units, conservation metrics, and pressure diagnostics. Add identity/cache/sparse-boundary tests before physics changes.

### Stage 1 — variable-density mixed-liquid baseline

Make P2G mass/material-aware and implement the shared variable-coefficient pressure projection with no sharp immiscible claim. Derive `rho` from deposited bounded composition; validate hydrostatics, density-ratio conditioning, divergence, and GPU residency.

### Stage 2 — two immiscible liquids, no interfacial tension

Persist phase ID through all particle lifecycle paths. Add conservative `alpha` plus narrow-band geometry/reconstruction and interface-aware pressure faces. Validate phase-volume retention, stratification, stirring, sheets/droplets, cache, and material output. This is the first defensible immiscible vertical slice.

### Stage 3 — interfacial tension

Add pairwise `sigma_AB`, accurate normals/curvature, and GFM/balanced pressure coupling. Add Laplace droplet, capillary wave/oscillation, rising bubble, capillary substep diagnostics, and spurious-current profiling.

### Stage 4 — static wetting/contact angle

Add pair-specific static solid contact angle with near-wall reconstruction and sessile-drop validation. Limit documented angle/rate regimes to those actually validated.

### Stage 5 — robustness and production polish

Improve preconditioning, variable-viscosity stress treatment, small-droplet retention policy, interface mesh quality, restart compatibility, warnings, and interactive profiles.

### Stage 6 — consider generalized N-phase

Only after Stages 0–5 pass, evaluate regional level sets/VOF labels, pairwise surface-tension matrix, junction law, cache schema expansion, and O(N) memory/dispatch implications. Two-phase data structures should be versioned so N-phase is possible, but not prematurely generalized.

## 13. Dependencies and cross-lane coupling

- **Pressure/projection owner first:** variable face coefficients, preconditioner behavior, cut-face jump representation, convergence telemetry, and correction consistency are the critical dependency.
- **Particle lifecycle/source owner:** phase ID allocation, emission assignment, P2G deposit, G2P, reseeding, deletion, capacity growth, and source animation must preserve identity/volume.
- **Sparse-grid owner:** interface tile classification, halo activation, leaf-neighbor lookup, transient scratch allocation, and active-work compaction.
- **Viscosity lane:** agree on whether viscosity is explicit or implicit and how face/stress coefficients and jumps are formed; density work must not silently redefine it.
- **Surface/meshing lane:** consume authoritative phase fraction/geometry; output external surfaces and internal liquid–liquid interfaces with material attributes.
- **Color/material mixing lane:** display color may be advected per phase, but physical `phase_id` remains separate. A miscible color blend must not mutate material density unless explicitly running a miscible concentration model.
- **Cache/playback/output owners:** schema, restart fidelity, preview-vs-authoritative separation, production cache, and material-tagged outputs.
- **UI/settings/admission owners:** material-table validation and snapshot/cutover must be centralized outside adapters; density/viscosity positivity and unsupported mode transitions should fail before allocation.
- **Profiling/validation:** add per-phase conservation, interface/halo tile counts, pressure iteration/residual, substep limiting, and GPU memory counters.

Recommended cross-lane order: material/identity contract -> sparse field/halo contract -> variable-density projection -> lifecycle/cache preservation -> conservative interface -> rendering/output -> interfacial tension -> wetting -> N-phase evaluation.

## 14. Risks, unknowns, and rejected approaches

### Material risks and unknowns

- The current pressure preconditioner’s tolerance for discontinuous coefficients is unknown; this is the largest technical blocker.
- The cost of conservative geometric reconstruction and curvature on fixed GPU tiles is unknown until profiled.
- Particle volume ownership during adaptive reseeding may already be implicit; the multiphase design must make it explicit without duplicating conservation corrections.
- Triple lines occur when both liquids meet gas or solids. A two-liquid slice needs a documented policy for such cells even if generalized junction physics is deferred.
- Small droplets below grid resolution can survive as labelled particles but no longer have resolved pressure/surface tension. Rendering them without a resolution warning can overstate physics.
- Variable viscosity with a sharp jump may demand a coupled/implicit stress solve. Stage 2 can constrain the supported viscosity ratio until that lane is proven.
- Contact-angle results are grid/slip-model dependent, especially for moving contact lines and extreme angles.
- Numerical diffusion versus true miscibility must remain user-visible. A diffusion coefficient belongs to a separate miscible-species model.

### Rejected/deferred approaches

- **Generalized N-phase first:** rejected because regional fields, pair matrices, junction laws, cache/output/UI, and memory scale before the two-phase fundamentals are proven.
- **Density-only particles marketed as oil/water:** rejected; official product baselines show excess mixing and explicitly disclaim immiscibility.
- **Classical level set alone:** rejected as final transport due to volume loss; acceptable only as geometry with correction/conservative companion.
- **Phase field as default:** rejected for the first slice because of diffuse-interface semantics, extra stiff solve, mobility/interface-width calibration, and wider sparse halos.
- **Two complete phase velocity grids:** deferred; MultiFLIP’s energetic free-slip benefit does not justify first-slice memory/bandwidth cost for viscous liquids.
- **Separate per-phase pressure projections:** rejected because they do not directly impose one global incompressibility/interface-normal coupling.
- **Independently advected density field:** rejected because it can drift from phase labels/fraction; derive material properties from one authoritative phase representation.
- **Tree AMR:** rejected by the fixed flat 8x8x8 leaf-table contract.
- **Dense FFT pressure path:** rejected by sparse storage/interactive activation constraints.
- **Direct GPL code transplant:** rejected without explicit license compatibility; use algorithms and independently implement from papers/specifications.

## 15. Source index

The complete artifact-by-artifact source URL, immutable revision/version, owner, redistribution note, retrieval time, SHA-256, and supported claims are in [sources/MANIFEST.md](sources/MANIFEST.md).

Primary families:

- Multiphase/regional level sets: [Losasso et al.](https://doi.org/10.1145/1179352.1141960), [local](sources/S01-losasso-2006-multiple-liquids.md).
- MultiFLIP: [Boyd](https://www.cs.ubc.ca/~rbridson/docs/boyd_msc.pdf), [local](sources/S02-boyd-2011-multiflip.md).
- CLSVOF: [Sussman–Puckett](https://doi.org/10.1006/jcph.2000.6537), [local](sources/S03-sussman-puckett-2000-clsvof.md).
- CSF: [Brackbill et al.](https://doi.org/10.1016/0021-9991(92)90240-Y), [local](sources/S04-brackbill-1992-csf.md).
- Ghost Fluid: [Fedkiw et al.](https://doi.org/10.1006/jcph.1999.6236), [local](sources/S05-fedkiw-1999-ghost-fluid.md).
- Phase field: [Jacqmin](https://doi.org/10.1006/jcph.1999.6332), [local](sources/S06-jacqmin-1999-phase-field.md).
- Balanced VOF: [Popinet](https://doi.org/10.1016/j.jcp.2009.04.042), [local](sources/S07-popinet-2009-balanced-vof.md).
- Rising-bubble benchmark: [Hysing et al.](https://doi.org/10.1002/fld.1934), [local](sources/S08-hysing-2009-bubble-benchmark.md).
- APIC: [Jiang et al.](https://doi.org/10.1145/2766996), [local](sources/S09-jiang-2015-apic.md).
- GPU two-phase: [FluTAS](https://doi.org/10.1016/j.cpc.2022.108602), [local](sources/S10-flutas-2022-gpu-two-phase.md).
- Sharp incompressible jumps: [Kang et al.](https://doi.org/10.1023/A:1011178417620), [local](sources/S15-kang-fedkiw-liu-2000-jumps.md).
- Contact angle: [Afkhami–Bussmann](https://doi.org/10.1002/fld.1974), [local](sources/S16-afkhami-2009-contact-angle.md).
- First-party product baselines: [SideFX local clip](sources/S11-sidefx-houdini22-baseline.md), [FLIP Fluids local clip](sources/S12-flip-fluids-1.8.8-baseline.md), [Blender local clip](sources/S13-blender-5.1-fluid-baseline.md), [other products local clip](sources/S14-other-industry-baselines.md).
- Pinned source evidence: [OpenFOAM clips](code/openfoam-interfaceProperties.C.clip), [MantaFlow clips](code/mantaflow-pressure.cpp.clip), [FluTAS clips](code/flutas-vof.f90.clip).

## 16. Orchestrator ingestion block

```yaml
recommended_techniques:
  vertical_slice: two immiscible incompressible liquids
  transport: persistent particle phase_id plus bounded conservative cell phase_fraction
  geometry: narrow-band signed distance or reconstructed geometry coupled to fraction
  velocity: one shared MAC velocity with phase-aware FLIP/APIC transfer
  projection: sparse variable-coefficient pressure solve with identical face coefficient in assembly and correction
  interface_force: Ghost-Fluid or discretely balanced jump using pairwise sigma_AB
alternatives:
  - particle level set plus explicit volume correction as an intermediate
  - conservative level set if geometric VOF cost blocks Stage 2
  - balanced CSF if sharp GFM integration is impractical and droplet tests pass
required_shared_state:
  - immutable two-material table: density, viscosity, liquid-gas tension
  - particle phase_id and explicit particle volume semantics
  - cell phase_fraction; narrow-band geometry; interface/cut-face metadata
  - per-phase conservation and pressure convergence telemetry
new_phases_kernels:
  - bin_particles_by_leaf_and_phase
  - deposit_phase_volume
  - normalize_phase_fraction
  - reconstruct_internal_interface
  - classify_interface_faces_and_activate_halo
  - update_material_coefficients
  - assemble_variable_density_pressure
  - apply_variable_density_pressure_gradient
  - compute_interface_normals_curvature
  - apply_interfacial_tension
  - enforce_static_contact_angle
cache_output_impact: schema-version material table and phase_id; authoritative phase fraction/restart state; per-phase/internal-interface mesh attributes
ui_impact: Single, Variable-Density Mixed, Two Immiscible modes; two material slots; sigma_AB; later static contact angle; phase/output diagnostics
strongest_validation_targets:
  - phase identity exact lifecycle/cache round trip
  - bounded alpha and phase-volume convergence
  - hydrostatic piecewise pressure slope
  - Laplace pressure jump and vanishing spurious currents under refinement
  - Hysing circularity, center-of-mass, and rise-velocity reference curves
  - sparse leaf-boundary translation invariance
  - no full live-state CPU round trip
blockers:
  - variable-coefficient sparse pressure preconditioner conditioning/performance
  - conservative GPU interface reconstruction and halo cost
  - consistent variable-viscosity stress jump
  - triple-line policy where two liquids meet gas/solid
cross_lane_order: material identity -> sparse field/halo -> projection -> lifecycle/cache -> conservative interface -> output -> tension -> wetting -> evaluate N-phase
confidence:
  two_liquid_architecture: high
  interactive_gpu_budget: medium pending profiles
  generalized_n_phase: low and deferred
```
