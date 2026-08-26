# Density and multiphase implementation seams

## Executive recommendation

Implement this capability as two truthful tiers, in order:

1. **Variable-density mixed liquid**: two material densities feed one shared MAC velocity and a variable-coefficient pressure projection. This tier promises density-driven stratification/buoyancy, but no persistent oil/water boundary.
2. **Two immiscible liquids**: add persistent particle `phase_id`, bounded conservative cell `phase_fraction`, narrow-band interface geometry, phase-aware transfer, and only then a Ghost-Fluid or discretely balanced interfacial jump. Reserve every oil/water, separation, and interfacial-tension claim until interface retention and jump tests pass.

Keep the existing shared velocity. Do not add per-phase velocity grids, separate pressure solves, dense fields, or tree storage. The external seam should remain `PressureSolver.project_frame(...)`; deepen that module so it owns coefficient construction/validation, matrix assembly, solve, and the matching velocity correction. A new in-process `MaterialInterfaceState` module should own authoritative material state (`phase_id` + fraction + narrow-band geometry) and expose a small interface to Solver orchestration and `PressureSolver`; it must not be a shallow pass-through or a second orchestrator.

This is a **shared-path** change. `FluidSolver.step_frame()` drives sourcing, P2G, forces, viscosity, pressure, G2P, lifecycle, and surface work for interactive preview and production stepping (`sim/solver.py:2067-2155`). `scene_bridge.Session.step()` calls it and then publishes/cache-writes (`blender/scene_bridge.py:263-346`); cache playback is the contrasting non-solver sibling (`blender/playback.py:237-260`), while Production Cache Bake is a separate authoritative caller of the same Solver Session policy (`blender/cache_bake.py:6`). No current demo or one-off caller owns the missing physics.

## Current truth

### Meanings currently sharing the word density

| Meaning | Live behavior | Consequence |
|---|---|---|
| Particle `density` | Always-allocated float attribute in `PHYSICAL_ATTR_NAMES` (`sim/particles.py:16`); sources write it (`sim/sourcing.py:162-186`, `1893-1968`); growth copies it (`sim/particles.py:236-258`); compaction preserves it (`347-404`); checkpoint restore requires it (`271-342`). | Persistent metadata exists, but is not material mass or a phase label. |
| Grid `density` | `FluidSolver` allocates `self.density` (`sim/solver.py:710-715`). Under ST-FLIP, `_p2g()` writes liquid occupancy/activity derived from MAC weight accumulators into it (`1520-1568`). Source field operations may also address it (`sim/sourcing.py:2813-2814`, `2964-2965`, `3006-3007`). | The name is collided: it is activity/phase-mask storage, not an authoritative physical material-density field. |
| ST-FLIP “phase” | `st_phase_from_mass()` maps face transfer weight to a bounded liquid occupancy (`sim/pressure.py:81-86`). `st_face_beta()` interpolates between one global liquid density and gas density (`90-109`). | This is liquid-versus-gas occupancy. It does not distinguish water from oil. |
| Particle sampling/rest density | `particles_per_voxel`, resolved quality/rest count, and ST reference mass control source occupancy, anti-clumping, and calibration (`sim/solver.py:463-477`, `1452-1518`; `sim/pressure.py:265-298`). | Sampling count/mass must remain separate from kg/m3 material density. |
| Particle viscosity | Sources write it; P2G averages it only when `viscosity_from_particles` is enabled (`sim/solver.py:1569-1587`); `ViscositySolver` consumes the grid field (`1696-1716`). | This is a real variable material-property path, but it is not phase-consistent and currently has no sharp stress-jump contract. |
| Output density | Product schemas expose `fluidsim_density` (`sim/attributes.py:60-63`); surfaces sample the nearest particle (`sim/frame_output.py:267-305`); cache stores `particle_density` (`io/cache.py:310-380`). | Display/cache transport proves metadata persistence only, not physical coupling or interface conservation. |

The standard P2G path transfers velocity and occupancy, not particle density (`sim/solver.py:1520-1587`); G2P transfers velocity only (`1799-1813`). Gravity/force application has no particle-density input. The standard pressure path solves the folded constant-density system documented as `q = p*dt/rho` (`sim/pressure.py:1-7`). The ST path is genuinely variable-coefficient, and consistently uses its face beta in diagonal/matrix/correction (`sim/pressure.py:428-518`, `1074-1153`), but beta is derived from occupancy and two global liquid/gas constants—not particle density or material identity. Therefore the current scope is **shared-path absence of material-density coupling**, not “no variable coefficient code at all.”

There is no live `phase_id`, `phase_fraction`, interfacial-tension, contact-angle, or reseeding implementation in production source/tests. Lifecycle today is source emission, advection/collision death, compaction, and optional capacity growth; a future reseeder must be added deliberately rather than assumed.

The research recommendation agrees with this boundary: `REPORT.md:15-23` recommends persistent identity, conservative fraction + geometry, shared MAC velocity, and variable projection; `REPORT.md:31-44` separates mixed density from immiscibility; `SYNTHESIS.md:15-26` warns that existing density storage is not the physics; `SYNTHESIS.md:36-43` gives the two-tier decision. The research is directionally sound, but its claim that density is merely constant-coefficient needs the live qualification above: the ST liquid/gas occupancy path already supplies a useful numerical prototype, not a water/oil implementation.

## Caller/ownership map

```text
Blender properties/panels
  -> solver_settings.build_settings (adapter)
  -> simulation_admission capture/preflight/replacement cutover
  -> scene_bridge Session.step / production bake
  -> FluidSolver.step_frame orchestration
       source -> P2G -> forces -> viscosity -> PressureSolver -> G2P
                     ^                ^             |
                     |                |             v
              MaterialInterfaceState +------ face coefficient/jump state
       lifecycle -> surface -> HostFrame
  -> output_publication OR cache_bake/io.cache
  -> playback consumes cache without stepping solver
```

- **Admission owner:** `blender/simulation_admission.py:184-299`, `403-515`, `565-676` captures coherent settings/sources, plans memory/capacity, and performs replacement cutover. Material-table validation and new VRAM estimates belong here.
- **Settings adapter:** `blender/solver_settings.py:22-54`, `186` copies Blender properties into `SolverSettings`; it must not reconstruct phase policy.
- **Solver orchestrator:** `sim/solver.py:659-740`, `2067-2155` allocates deep modules and orders substeps. It should call material-state methods at explicit phases, not implement interface numerics inline.
- **Pressure owner:** `sim/pressure.py:1358-1442`, `1697-1762` owns pressure buffers and the production projection interface. Production caller is only `FluidSolver._pressure()` (`sim/solver.py:1742-1758`); constant and ST paths are contrasting internal adapters. This is the deepest shared seam for coefficient/jump consistency.
- **Particle lifecycle owner:** `sim/particles.py:139-217`, `236-404`; sourcing callers are analytic, mesh, compiled point, height, and exact injection paths in `sim/sourcing.py`. All must preserve `phase_id` and represented volume.
- **Sparse storage owner:** `sim/fields.py:1-10`, `35-101`, `427-475` fixes the flat 8x8x8 leaf table; topology dilation exists at leaf granularity (`357-383`, `529-537`). Interface halos must reuse/extend this contract.
- **Persistence/output siblings:** checkpoints in `sim/solver.py:932-1182`; frame cache in `io/cache.py:310-380`, `562`, `635`, `704`; live publication in `blender/output_publication.py:1204`; cached publication at `1230`; preview policy in `blender/playback.py:237-260`; production cache owner in `blender/cache_bake.py:6`.

## Claim matrix

| Claim | Current status | Minimum proof before claim |
|---|---|---|
| Variable-density mixed liquid | **Not implemented.** ST liquid/gas occupancy projection is a reusable prototype; particle density is uncoupled. | Particle/material density drives the same face `1/rho` in assembly and correction; hydrostatic slopes, density-ratio convergence, disabled-mode parity. |
| Density-driven buoyancy/stratification | **Not implemented.** All particles receive the same acceleration; no material-density pressure response. | Stable light-over-heavy equilibrium, heavy-over-light overturning, correct piecewise hydrostatic slope; no arbitrary per-particle fall force. |
| Two-liquid immiscible separation | **Absent.** No persistent phase identity/fraction/interface. | Exact `phase_id` lifecycle/cache round trip; bounded fraction; convergent per-phase volume; equal-density identity retention; stirred water/oil retention. |
| Liquid-liquid interfacial tension | **Absent.** Existing surface meshing is liquid-air output, not a stress jump. | Laplace jump convergence and decreasing static-droplet spurious currents; same cut-face geometry in jump and projection. |
| Wetting/contact angle | **Absent.** Collider material response is friction/restitution, not contact-line physics. | Static angle convergence across walls, leaf translations, and refinement after reliable curvature/tension. |
| Generalized N-phase | **Absent and deferred.** | Two-phase tiers complete, then regional representation, pair matrix, triple-junction law, O(N) memory/timing, schema/UI migration. |

## Deep seam design

### `MaterialInterfaceState` deep module

Place a new in-process module at `sim/material_interface.py`. Its small external interface should be approximately:

- `prepare_substep(particles, bins, voxel_type, dt) -> MaterialFaceState`
- `after_advection(particles, bins, dt) -> ConservationStats`
- `checkpoint_state()` / `restore_checkpoint_state(state)`

Behind this interface it owns the material table, deposition/normalization, fraction bounds, phase-volume ledger, narrow-band `phi`, cut-face classification, interface halos, normals/curvature, and transient material coefficients. `MaterialFaceState` should expose device arrays required by `PressureSolver` and viscosity/surface consumers, with invariants (same topology epoch, positive bounded density, fraction in [0,1], device residency). Tests use this same interface; internal kernels remain internal seams.

Deletion test: without this module, phase derivation, bounds, halo activation, conservation, and coefficient consistency would spread across sourcing, Solver, pressure, viscosity, surface, cache, and output, so the module earns depth and locality.

### Deepen `PressureSolver`; do not create another pressure orchestrator

Keep `PressureSolver.project_frame()` as the production seam. Replace ST-specific scalar arguments with an optional device-resident coefficient/jump bundle. `PressureSolver` must own:

- validation that assembly and correction use the identical face coefficient;
- diagonal/matrix/Jacobi/PCG application for variable coefficients;
- free-surface/internal-interface cut-face terms and pressure jump RHS;
- residual/status/conditioning diagnostics;
- constant-coefficient behavior when the bundle is absent.

Do not put pressure kernels in `MaterialInterfaceState`: that would split one numerical invariant across modules. Solver orchestration chooses mode and ordering; `PressureSolver` owns projection mathematics; material state supplies authoritative coefficients/geometry.

### Semantic migration

- Rename grid `density` to `liquid_activity` (schema-compatible alias only during migration).
- Rename particle `density` to `density_attribute` or migrate it into material assignment; never silently reinterpret old caches as kg/m3.
- Keep `particles_per_voxel`/ST `reference_mass` as sampling/occupancy concepts.
- Add `phase_id` as an always-on integer only for two-material mode; add explicit represented particle volume/mass semantics before conservative deposition or reseeding.
- Derive `material_density` and `material_viscosity` from the material table + authoritative fraction. Do not independently advect either field.
- Keep external liquid-air surface fraction/SDF distinct from internal liquid-liquid fraction/geometry and output attributes.

## Technique decision

**A — variable-density mixed:** first derive a cell/face material mixture from source/material provenance, retain one MAC velocity, and use `div(dt/rho grad p)=div u*`. Reuse the ST operator shape, but replace occupancy-derived global liquid/gas beta with material-density coefficients. Use a symmetric positive face policy (harmonic/interfacial `1/rho`, verified rather than guessed) identically in matrix and correction. This is the smallest vertical proof of physical density coupling.

**B — two immiscible:** add persistent `phase_id`, conservative particle-volume deposition to bounded `phase_fraction`, and narrow-band geometry coupled to that fraction. Keep FLIP/APIC velocity shared; make transfer phase-aware for material-volume accounting, not by duplicating velocity grids. First pass zero `sigma_AB`; prove interface retention. Then add Ghost-Fluid or discretely balanced `sigma_AB*kappa` using the same cut-face geometry as pressure. Add static wetting later.

Recommended end state is CLSVOF-like fraction + geometry (`REPORT.md:21`, `89-106`, `188-230`). A particle-label-derived SDF with explicit volume correction is acceptable as an intermediate prototype, but cannot close the immiscible tier. Reject label-only oil/water, level set alone, independently advected density, separate phase solves/grids, phase field, dense FFT, and tree AMR.

## Ordered implementation plan

1. **Names and material contract.** Add mode enum (`SINGLE`, `VARIABLE_DENSITY_MIXED`, `TWO_IMMISCIBLE`), immutable two-material records (density, viscosity; tension later), units, represented particle volume, and migration rules. Update `sim/solver.py`, `blender/properties.py`, `solver_settings.py`, `pending_settings.py`, panels, and admission validation.
2. **Particle identity slice.** Add `phase_id` integer SoA and preserve it through every sourcing kernel/descriptor, exact injection, growth, compaction, checkpoint, cache, and HostFrame. No physics claim yet. Extend capacity estimates for particle and scratch bytes.
3. **Unambiguous grid state.** Rename activity storage; add material accumulation/weight, bounded `phase_fraction`, cell density/viscosity, face `inv_rho`, conservation counters. Add kernels `deposit_phase_volume`, `normalize_phase_fraction`, `update_material_coefficients`; activate required neighbor leaves through the existing flat-table dilation transaction.
4. **Mixed-density projection slice.** Generalize `PressureSolver` ST kernels into coefficient-driven diagonal, matrix, Jacobi/PCG, residual, and correction paths; thread one `MaterialFaceState` through `FluidSolver._pressure()`. Add min/max coefficient, ratio, iterations/residual telemetry. This closes only mixed-density claims.
5. **Lifecycle/cache completion.** Define source inheritance, deletion ledger, and future reseed split/merge rules. Version solver checkpoint and cache schemas; store material table, `phase_id`, represented volume, and authoritative fraction or sufficient restart state. Keep cache playback non-authoritative and Production Cache Bake sequential/authoritative.
6. **Immiscible zero-tension slice.** Add conservative face flux, fraction advection/correction, narrow-band geometry/reinitialization, phase-aware transfer accounting, and internal-interface halo classification. Prove interface retention before naming water/oil.
7. **Surface/output slice.** Extend `SurfacePipeline`/`frame_output.py` and `output_publication.py` to publish separate/material-tagged external and internal interfaces (`phase_id`, `material_index`, `interface_kind`); nearest-particle density remains display-only legacy output.
8. **Interfacial tension slice.** Add cut-face fraction/jump metadata, normals/curvature, Ghost-Fluid/balanced jump, capillary timestep telemetry. Add `sigma_AB` UI only when validated.
9. **Wetting then N-phase evaluation.** Static contact angle per material/solid pair after tension; N-phase only after two-phase performance, junction, schema, and memory review.

Admission must account for persistent `phase_id`, fraction, density/viscosity, and narrow-band geometry plus peak double-buffer/reconstruction/PCG scratch. A topology/particle allocation failure remains pre-cutover and recoverable under `simulation_admission.py`; no partial phase state may enter the live session.

## Tests/acceptance

Smallest coherent vertical slices and metrics:

- **Identity-only:** two sources emit distinct labels; source, growth, compaction, collision deletion, checkpoint, cache/reopen/resume, live publication, and production bake preserve exact IDs and phase volume ledger.
- **Mixed density:** same-phase equal-density negative control matches baseline; two material densities produce piecewise hydrostatic pressure slopes; density ratios 1, 2, 10, 100 report finite SPD coefficients, iterations/residual, no NaN, and GPU time.
- **Immiscible zero tension:** fraction always within `[0,1]` tolerance; per-phase volume error converges with refinement; equal-density interface survives stirring; translate the same interface across face/edge/corner and 8-voxel leaf boundaries with invariant results.
- **Tension:** static sphere gives `Delta p = sigma*kappa`; maximum spurious velocity decreases under refinement; capillary timestep limiter is reported.
- **Persistence/performance:** cache/reopen/resume agrees within stated fraction/velocity tolerances; no full state CPU transfer inside substeps; profile active/interface/halo leaves, projection time, reconstruction time, VRAM, and first-frame/playback cadence.

Focused commands during implementation:

```powershell
rtk proxy python -m pytest -q tests/test_sparse_leaf_grid_contract.py tests/test_pressure_projection_regressions.py tests/test_viscosity_active_work_regressions.py
rtk proxy python -m pytest -q tests/test_particle_capacity_growth.py tests/test_source_emission_lifecycle.py tests/test_cache_io.py tests/test_frame_output.py tests/test_simulation_admission.py
rtk proxy python -m pytest -q tests/test_scene_bridge_cache_playback.py tests/test_production_gate_truthfulness.py
```

Add focused suites such as `test_material_interface_state.py`, `test_variable_density_projection.py`, and `test_immiscible_interface.py`, plus Blender background acceptance for settings, cache/reopen, and tagged surfaces. Then run:

```powershell
rtk proxy python -m pytest -q
rtk proxy python tools/run_production_gate.py
"C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" tools/profile_gpu_solver.py --device cuda:0 --frames 8 --warmup 2 --json-out artifacts/gpu_solver_profile.json
```

Use `python tools/run_production_gate.py --include-gui` before user-visible playback handoff and validate policy-drift reports after refreshing GUI evidence.

## Risks/rejected approaches

- Highest risk is PCG conditioning/performance at sharp density ratios; current Jacobi diagonal preconditioning and ST PCG host checkpoints (`sim/pressure.py:1833-1952`) need measured evidence.
- Conservative reconstruction and cross-leaf flux ownership may dominate GPU cost; preserve flat-leaf locality and activate only interface halos.
- Existing viscosity interpolation is averaged from particles and does not enforce sharp variable-viscosity stress jumps. Initially bound supported viscosity ratios and state the limitation.
- Two liquids meeting air/solid creates triple-line policy even before N-phase; document fallback cells and do not claim dynamic contact-line fidelity.
- Subgrid labelled droplets may be renderable but have unresolved interface pressure/tension; expose a resolution diagnostic.
- Reject “density attribute makes oil/water,” heavy-particle force hacks, independently advected rho, level set alone as final conservation, separate per-phase projections, duplicate full velocity grids, generalized N-phase first, dense FFT, tree AMR, and unreviewed code transplant from research references.

## Exact evidence index

Live source (line numbers are from the inspected working tree at base `0d395d78582af1f8876539f0f927110c3eff872d`; unrelated dirty files were not modified):

- Sparse contract: `fluidSim_warp_blender51/sim/constants.py:49-52`; `sim/fields.py:1-10`, `35-101`, `357-408`, `427-475`, `529-573`.
- Particle storage/lifecycle: `sim/particles.py:16-18`, `50-88`, `139-217`, `236-342`, `347-404`.
- Source density writers and callers: `sim/sourcing.py:162-186`, `810`, `974`, `1137`, `1214-1216`, `1893-1968`, `2813-2889`, `3085-3143`, `3192-3206`; Blender adapter `blender/scene_inputs.py:435-499`, `721-755`.
- Solver allocations/orchestration: `sim/solver.py:427-555`, `659-740`, `818-930`, `1273-1378`, `1520-1587`, `1696-1758`, `1799-1864`, `2067-2155`.
- Pressure constant and ST paths: `sim/pressure.py:1-22`, `81-142`, `178-188`, `211-298`, `302-380`, `428-518`, `1074-1153`, `1358-1442`, `1697-1762`, `1793-1952`, `2243-2268`.
- Viscosity: `sim/solver.py:1569-1587`, `1696-1716`; `sim/viscosity.py` (module owner).
- Output/persistence: `sim/attributes.py:17`, `60-63`, `86-120`, `160`; `sim/frame_output.py:238`, `267-305`; `io/cache.py:220-255`, `310-380`, `562`, `635`, `704`; `blender/output_publication.py:732-737`, `1204`, `1230`.
- Blender truth labels/settings: `properties.py:666-671`, `926-960`, `995-1017`, `1256-1295`; `panels.py:537-557`, `700-701`; `blender/solver_settings.py:22-54`, `186`; `blender/pending_settings.py:86-137`, `215-235`.
- Architecture callers: `blender/simulation_admission.py:184-299`, `403-515`, `565-676`; `blender/scene_bridge.py:263-346`; `blender/playback.py:237-260`; `blender/cache_bake.py:6`.
- Existing tests prove storage/output only: `tests/test_particle_capacity_growth.py:64-75`; `tests/test_frame_output.py:24`, `236-243`. No production/test matches existed for `phase_id`, `phase_fraction`, interfacial tension, contact angle, or reseeding.

Research comparison:

- `research/density-multiphase/REPORT.md:13-25`, `29-48`, `89-106`, `184-243`, `278-327`, `329-366`, `390-436`.
- `research/SYNTHESIS.md:15-26`, `36-43`, `51-70`, `73-95`, `104-115`.

Scope classification: **shared-path** for material density and immiscible-interface absence; affected lanes are interactive live stepping, Production Cache Bake, checkpoint/resume, cache playback/publication, and surface/particle output when the new mode is enabled. Existing single-liquid and disabled-feature behavior is the contrasting path and must remain unchanged. Performance and conditioning are **unproven** until the listed GPU tests run.
