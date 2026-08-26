# Color mixing implementation seams

## Executive recommendation

Implement clear-dye mixing as a deep solver module, `sim/species.py`, whose interface owns nonnegative particle-authoritative species concentrations (`K <= 4`), temporary sparse-grid transfer/diffusion state, conservation receipts, and derived optical coefficients. Keep the existing `color` channel as a derived compatibility/output surface; it is not authoritative chemistry.

The technique recommended by [the color-mixing research](../color-mixing/REPORT.md) and [the synthesis](../SYNTHESIS.md) fits this repository with three prerequisites:

1. define represented particle mass/volume and source/outflow accounting before promising conservation;
2. extend the flat sparse-grid interface with one-cell neighbor activation and deterministic global face ownership;
3. version authoritative cache/checkpoint state for species IDs, concentrations, palette metadata, and budgets.

The smallest coherent product target is **two calibrated clear dyes in one constant-density miscible carrier**. Advection rides existing particle motion. Mixing uses mass-weighted P2G, bounded equal-and-opposite face diffusion, and conservative grid-delta G2P. Beer-Lambert absorption is derived after transport. Turbid pigments are a later optical module requiring measured absorption/scattering data; do not describe clear-dye completion as paint mixing.

## Current truth

Truth classification below is based on live source, not names, UI, demos, or reports.

- **Reusable core scaffolding:** [`ParticleSystem`](../../fluidSim_warp_blender51/sim/particles.py#L131) always allocates `age`, `life`, `temperature`, `viscosity`, and `density`, while `color` is the sole optional `vec4` channel ([lines 16-18](../../fluidSim_warp_blender51/sim/particles.py#L16), [143-183](../../fluidSim_warp_blender51/sim/particles.py#L143)). Generic capacity growth copies all enabled channels ([242-258](../../fluidSim_warp_blender51/sim/particles.py#L242)); checkpoint capture/restore includes them ([260-340](../../fluidSim_warp_blender51/sim/particles.py#L260)); compaction scatters and swaps them ([347-400](../../fluidSim_warp_blender51/sim/particles.py#L347)). No reseeding implementation or test exists in live Python source; reseeding semantics are therefore absent, not merely unverified.
- **Real source assignment, not mixing:** [`SourceShapeDesc.to_struct`](../../fluidSim_warp_blender51/sim/sourcing.py#L309) normalizes a source swatch into `source.color` ([323-330](../../fluidSim_warp_blender51/sim/sourcing.py#L323)); the liquid-source kernel copies constant `temperature`, `viscosity`, `density`, and optional `color` into newly emitted particles ([970-979](../../fluidSim_warp_blender51/sim/sourcing.py#L970)). Direct injection reaches the same `SourcePipeline` through [`FluidSolver.inject_particles_exact`](../../fluidSim_warp_blender51/sim/solver.py#L2192). These paths establish persistent labels only.
- **Absent transport physics:** [`FluidSolver._p2g`](../../fluidSim_warp_blender51/sim/solver.py#L1520) transfers velocity and, conditionally, viscosity. Its `attr_weight` is cleared and used only to average particle viscosity ([1569-1585](../../fluidSim_warp_blender51/sim/solver.py#L1569)). [`_g2p_advect`](../../fluidSim_warp_blender51/sim/solver.py#L1799) updates particle velocity/position; it does not update color or any species. Thus differently colored particles advect as unchanged swatches and do not physically diffuse.
- **Name collision:** `solver.density` is a sparse field included in checkpoints ([641-646](../../fluidSim_warp_blender51/sim/solver.py#L641)) and is populated by the ST-FLIP phase/activity path in `_p2g` ([1524-1561](../../fluidSim_warp_blender51/sim/solver.py#L1524)). Particle `density` is a separately sourced/displayed scalar. Neither can safely be reused as species mass or material density. `attr_weight` is transient viscosity accumulation, not a general mixture weight.
- **Reusable sparse storage, missing diffusion seam:** [`fields.py`](../../fluidSim_warp_blender51/sim/fields.py#L1) explicitly stores fixed `8x8x8` leaves as a dense leaf-pointer table, reverse coordinates, and flat slot-major arrays ([1-10](../../fluidSim_warp_blender51/sim/fields.py#L1)); `voxel_flat_index` resolves cross-leaf lookup ([69-84](../../fluidSim_warp_blender51/sim/fields.py#L69)). Leaf marking and expansion exist ([299-385](../../fluidSim_warp_blender51/sim/fields.py#L299)), but there is no species halo policy, globally unique face owner, or conservative face-flux interface.
- **Proxy/output-only:** [`ATTRIBUTE_SPECS`](../../fluidSim_warp_blender51/sim/attributes.py#L36) defines `fluidsim_color` as `FLOAT_COLOR`, and [`HostFrame.output_attributes`](../../fluidSim_warp_blender51/sim/frame_output.py#L267) publishes particle attributes directly or transfers surface values with `nearest_liquid_particle_v1` ([attributes.py lines 16-17](../../fluidSim_warp_blender51/sim/attributes.py#L16)). Nearest-particle surface color is presentation, not diffusion or optics.
- **Real persistence scaffolding:** cache-v1 frame payload construction is at [`io/cache.py:303`](../../fluidSim_warp_blender51/io/cache.py#L303); strict cache-v2 manifests use format 2 ([cache_schema.py:19](../../fluidSim_warp_blender51/io/cache_schema.py#L19)); checkpoint payloads use format 1 ([checkpoint.py:17](../../fluidSim_warp_blender51/io/checkpoint.py#L17)). Solver checkpoint capture includes every physical particle scalar, enabled optional channel, listed sparse fields, and source state ([solver.py:932-999](../../fluidSim_warp_blender51/sim/solver.py#L932)). This can carry new state only after explicit schema/signature migration.
- **Real adapters, not solver proof:** live publication calls [`output_publication.update_outputs`](../../fluidSim_warp_blender51/blender/output_publication.py#L1204) from [`Session.step`](../../fluidSim_warp_blender51/blender/scene_bridge.py#L263); cache playback instead calls [`update_outputs_from_cache_frame`](../../fluidSim_warp_blender51/blender/playback.py#L237) at [line 260](../../fluidSim_warp_blender51/blender/playback.py#L260). Export creates/reuses a `HostFrame` ([io/export.py:11-13](../../fluidSim_warp_blender51/io/export.py#L11). These sibling lanes must agree on derived color, but none proves mixing.

## Caller/ownership map

```text
Blender properties/panels (adapters)
  -> pending_settings (setting lifecycle authority)
  -> simulation_admission (snapshot/configure/cutover authority)
  -> scene_bridge.Session.step
       -> FluidSolver.step_frame
          -> SourcePipeline.apply            [initialize injected species]
          -> FluidSolver._p2g                [velocity/ST/viscosity today]
          -> SpeciesTransport.advance        [proposed deep seam]
          -> forces/viscosity/pressure
          -> FluidSolver._g2p_advect          [velocity/position]
       -> HostFrame                           [single readback/output seam]
          -> output_publication               [live Blender objects]
          -> io.cache / checkpoint            [authoritative persistence]
          -> io.export                        [offline files]

cache_bake -> Session.step(publish disabled) -> authoritative cache
playback   -> cached frame -> output_publication (no solver authority)
```

Production caller evidence: `scene_bridge.Session.step` invokes `solver.step_frame` ([263-270](../../fluidSim_warp_blender51/blender/scene_bridge.py#L263)); solver substeps execute indexing, sourcing, P2G, forces, viscosity, pressure, and G2P/advection in [`step_frame`](../../fluidSim_warp_blender51/sim/solver.py#L2067). The contrasting direct-injection lane delegates to `SourcePipeline` ([solver.py:2188-2214](../../fluidSim_warp_blender51/sim/solver.py#L2188). Persistent bake is owned by [`cache_bake.bake`](../../fluidSim_warp_blender51/blender/cache_bake.py#L6), whereas preview policy is owned by [`playback.handle_frame_change`](../../fluidSim_warp_blender51/blender/playback.py#L305). Therefore the defect scope is **shared-path**: source, live playback, bake/resume, direct injection, point/surface publication, and export all need the same authoritative species seam; presentation adapters must not reconstruct it.

## Claim matrix

| User-visible claim | Required real core behavior | Current evidence | Status | Proof needed |
|---|---|---|---|---|
| Two liquid colors travel independently | Per-particle species initialized by every source and preserved by motion/lifecycle | Optional `color` is sourced and preserved through grow/compact/checkpoint | scaffolding | species identity/concentration fixtures across all source lanes |
| Colors realistically mix | Conservative diffusion changes species while advection alone does not | No color/species P2G, face flux, or G2P update | absent | analytic diffusion, conservation, bounds, seam tests |
| “Mixing Off” preserves unmixed dyes | Zero-diffusivity invariant | Existing swatches remain unchanged incidentally | proxy | runtime two-source zero-D test through real frame stepping |
| Surface visibly changes as liquids mix | Derived optics from authoritative species on real particle/surface outputs | nearest-particle `fluidsim_color` publication exists | proxy | derived-species color on points and reconstructed surface after reopen |
| Clear dye looks subtractive and thickness dependent | Concentration-to-absorption mapping plus Beer-Lambert path length | No absorption/species palette state | absent | calibrated palette and two-thickness render/attribute fixture |
| Cache/bake resumes mixing | Authoritative species, IDs, palette hash, budgets, and solver scratch contract survive resume | Generic optional channels/checkpoints and cache manifests exist | scaffolding | interrupted-vs-continuous concentration and budget comparison |
| GPU-resident interactive playback | No host transfer in substeps; bounded K<=4 cost | Solver/grid are Warp-resident; output readback is isolated in `HostFrame` | scaffolding | transfer counters and user-path GPU profile with feature on/off |
| Existing color controls prove mixing | Physical state changes in solver | UI/output tests prove only attributes/materials | proxy | must not be used as acceptance evidence |
| Paint/pigment mixing | Turbid absorption/scattering or calibrated licensed latent model | None | absent | separate palette calibration and optical validation |

## Deep seam design

Create a deep `SpeciesTransport` module in `fluidSim_warp_blender51/sim/species.py`. Its small interface should hide kernels, scratch allocation, leaf-face traversal, limiters, reductions, and optics:

```python
class SpeciesTransport:
    def initialize_slots(self, start, count, source_mix, represented_mass): ...
    def advance(self, particles, grid, dt) -> SpeciesStepReceipt: ...
    def lifecycle_arrays(self) -> Mapping[str, WarpArray]: ...
    def derive_output(self, concentrations, optical_palette) -> SpeciesOutput: ...
```

This seam earns depth: one implementation enforces concentration bounds, mass accounting, sparse-face locality, and palette semantics for live stepping, direct injection, bake, checkpoint, and output. `ParticleSystem` remains the storage/lifecycle implementation and should gain an optional `species` `wp.vec4` channel plus represented mass (or a proved uniform-mass contract). Its existing generic resize, compact, checkpoint, and restore loops then provide leverage without scattering species-specific copies.

`SparseGrid` should expose the minimum shared interface needed by the module: activate active-plus-one-cell-neighbor leaves, resolve neighbor voxels through the flat table, and compute a deterministic global face owner from voxel coordinate plus positive axis. Do not add a hierarchical tree or store authoritative species in OpenVDB/NanoVDB.

Required new state and kernels:

- types: `SpeciesSettings`, `SpeciesPaletteEntry`, `SpeciesSourceMix`, `SpeciesBudget`, `SpeciesStepReceipt`, `SpeciesOutput`;
- particle fields: `species: wp.vec4`, represented mass/volume contract, optional stable palette/species IDs in metadata;
- temporary grid fields: `species_mass[K]`, `carrier_mass`, `species_delta[K]`, limiter capacity/scratch; never reuse `density` or `attr_weight`;
- kernels: `k_species_p2g_clear`, `k_species_p2g_accumulate`, `k_species_face_flux_measure`, `k_species_face_flux_apply_limited`, `k_species_g2p_delta`, `k_species_budget_reduce`, `k_species_preview_derive`;
- invariants: finite `c_k >= 0`, declared `sum(c_k) <= c_max`, equal-and-opposite owned-face exchange, no clamp-and-renormalize mass loss, and explicit injected/removed/current ledgers.

## Technique decision

**Fit as recommended:** particle authority matches the existing FLIP/APIC motion and generic particle lifecycle; `K<=4` maps efficiently to Warp `vec4`; mass-weighted P2G can reuse flat-leaf lookup patterns; face diffusion is local and GPU-resident; delta G2P avoids making the grid authoritative for advection; derived Beer-Lambert optics preserves transport independently of appearance.

**Must change:** source descriptors/plans need species mixtures rather than only display `color`; represented particle mass must be explicit; topology needs a one-cell halo and face ownership; the bounded particle-level delta limiter needs a project-specific derivation; memory admission must charge particle channels plus temporary leaf buffers; cache/checkpoint versions and signatures must change; `HostFrame` must expose authoritative concentrations for cache and derived output separately.

**Reject:** RGB/sRGB as conserved state, copying existing `color` through a blur, reusing ST `density`, reusing viscosity `attr_weight`, frame-rate-dependent lerp, unconditional neighbor absorption, clamp-then-renormalize, RGB-only caches, unconstrained MacCormack/BFECC, live hierarchical VDB storage, an unlicensed Mixbox dependency, or claims of bitwise repeatability. Atomic accumulation and face scheduling require tolerance-based reproducibility.

Clear dyes use concentration-weighted nonnegative absorption/extinction and `T = exp(-sigma_t * L)`. A surface base color is only a viewport approximation because a closed surface mesh has no unique liquid path length. Turbid pigments require a separate `PigmentOptics` adapter (calibrated Kubelka-Munk/spectral or separately licensed latent model); they must not alter conserved species transport.

## Ordered implementation plan

1. **State contract:** add `sim/species.py`; define units, represented mass/volume, `K<=4`, palette IDs/hash, bounds, budgets, and the disabled-feature identity contract. Add reference CPU tests before GPU kernels.
2. **Particle lifecycle:** extend [`particles.py`](../../fluidSim_warp_blender51/sim/particles.py) with optional species storage and mass semantics; prove initialize, growth, compaction, outflow removal, checkpoint restore, and currently absent reseeding policy. Update [`memory_safety.py`](../../fluidSim_warp_blender51/sim/memory_safety.py) and capacity planning for exact peak bytes.
3. **Sources:** extend [`source_regions.py`](../../fluidSim_warp_blender51/sim/source_regions.py) plan types and [`sourcing.py`](../../fluidSim_warp_blender51/sim/sourcing.py) legacy/native/direct transactions with `SpeciesSourceMix`; keep emission all-or-zero and record injected species mass.
4. **Sparse diffusion:** add active-plus-halo and global positive-axis face ownership to [`fields.py`](../../fluidSim_warp_blender51/sim/fields.py); implement species P2G, bounded face flux, delta G2P, and budget kernels behind `SpeciesTransport.advance`.
5. **Solver orchestration:** add settings/resources in [`solver.py`](../../fluidSim_warp_blender51/sim/solver.py), allocate the module during admission, and call it after source emission/P2G and before G2P/advection. Never route policy through Blender adapters.
6. **Persistence/migration:** bump checkpoint format 1 and cache manifest/frame schema as required in [`checkpoint.py`](../../fluidSim_warp_blender51/io/checkpoint.py), [`cache_schema.py`](../../fluidSim_warp_blender51/io/cache_schema.py), and [`cache.py`](../../fluidSim_warp_blender51/io/cache.py). Persist concentrations, represented mass, species IDs/count, palette/hash, diffusivities, budgets, and required resume state. Old caches either load as explicit legacy display-color-only playback or fail with an actionable incompatibility; never synthesize authoritative species silently.
7. **Output:** add species and derived absorption/preview declarations in [`attributes.py`](../../fluidSim_warp_blender51/sim/attributes.py) and lazy accessors in [`frame_output.py`](../../fluidSim_warp_blender51/sim/frame_output.py). Update [`output_publication.py`](../../fluidSim_warp_blender51/blender/output_publication.py) for particle and surface derived color. Cache bake stores authority; preview playback republishes cached derivations without inventing physics; export labels derived channels.
8. **Settings/adapters:** add domain mixing mode (`OFF`, `MOLECULAR`, `EFFECTIVE`), diffusivity in world-unit squared/second, palette and diagnostics; add per-source species concentrations. Register every field exhaustively in [`pending_settings.py`](../../fluidSim_warp_blender51/blender/pending_settings.py), with allocation/palette identity as Reset/Reinitialize and safe coefficients/publication toggles as Live only where proven. Wire [`properties.py`](../../fluidSim_warp_blender51/properties.py) and [`panels.py`](../../fluidSim_warp_blender51/panels.py) as adapters.
9. **Demo and gates:** add a tracked two-dye closed-box demo only after source/runtime/cache tests pass; synchronize `build_demo_scenes.py`, `check_demo_blends.py`, and `analyze_demos.py` inventory/expectations.

Migration note: `ATTRIBUTE_SCHEMA_VERSION = 1` ([attributes.py:16](../../fluidSim_warp_blender51/sim/attributes.py#L16)), cache-v1 frames, cache-v2 manifests, and checkpoint-v1 are distinct version surfaces. Change each deliberately; a manifest version bump alone does not protect checkpoint or attribute consumers.

## Tests/acceptance

### Smallest coherent vertical slice

Scope: two species, equal represented particle mass, one common constant diffusivity, closed box, source initialization through both a compiled source and direct injection, GPU conservative diffusion, point-output preview color, checkpoint/cache resume, and “Mixing Off.” Surface output may use nearest-particle derived preview but must be labeled approximate. No reseeding, variable density, reactions, viscosity coupling, surface-tension coupling, spectral rendering, or pigment claim.

Acceptance metrics:

- closed-box per-species relative mass residual `<= 1e-5` for 240 frames;
- concentration minimum `>= -1e-7`, loading maximum `<= c_max + 1e-7`, and no NaN/Inf;
- zero diffusivity changes concentrations by at most `1e-7` L-infinity;
- analytic Gaussian diffusion L2 relative error `<= 2%` at stated grid/time and improves or remains stable under refinement;
- identical face/edge/corner case shifted across an `8x8x8` leaf seam agrees within `1e-5` and has the same mass residual;
- continuous versus checkpoint-resumed concentration L2 difference `<= 1e-4`, with matching palette hash and budgets;
- source injection and removal ledger closes within `1e-5`;
- no device-to-host transfer inside substeps; feature-off allocations and numerical results preserve the current solver contract;
- `h` versus `h/2` mixedness differs by `<= 10%` for the declared effective-diffusion fixture;
- focused user-path profile reports frame time, particle/grid bytes, active/halo leaves, and species kernel timings for `K=2` and `K=4`.

Stop/go blockers before implementation: (1) approve represented mass/volume semantics; (2) derive and independently test the conservative bounded delta-G2P limiter; (3) prove global face ownership and halo capacity behavior at leaf seams; (4) choose calibrated clear-dye coefficients/reference path length; (5) define old-cache behavior. If any fails, stop at particle-authoritative transport with `Mixing Off`; do not substitute RGB smoothing.

Focused proposed tests:

```powershell
rtk proxy python -m pytest -q tests/test_species_state.py tests/test_species_sources.py tests/test_species_sparse_diffusion.py tests/test_species_checkpoint.py
rtk proxy python -m pytest -q tests/test_particle_capacity_growth.py tests/test_sparse_leaf_grid_contract.py tests/test_frame_output.py tests/test_cache_io.py tests/test_scene_bridge_cache_playback.py
```

Blender/runtime acceptance:

```powershell
blender --background --factory-startup --python tests/blender_species_mixing_acceptance.py -- --json-out artifacts/species_mixing_acceptance.json
blender --background --python tools/build_demo_scenes.py -- color_mixing_clear_dye
blender --background --python tools/check_demo_blends.py
rtk proxy python tools/analyze_demos.py color_mixing_clear_dye
```

Broader gates after the focused lane passes:

```powershell
rtk proxy python -m pytest -q tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_snippet_library.py tests/test_production_gate_truthfulness.py tests/test_sparse_leaf_grid_contract.py tests/test_viscosity_active_work_regressions.py
rtk proxy python tools/run_production_gate.py
```

Use `--include-gui` and then `tools/validate_policy_drift_reports.py` for final interactive/release evidence, per project instructions.

## Risks/rejected approaches

- **Limiter risk (highest):** conservation and boundedness are independent; naive clamping passes bounds while corrupting species totals.
- **Mass semantics:** equal particle count is not automatically equal represented liquid mass after future reseeding or variable-mass sources.
- **Sparse topology:** halo expansion can exceed leaf capacity; admission must preflight it transactionally rather than partially activating diffusion.
- **Atomic nondeterminism:** promise tolerance-based conservation and resume parity, not bitwise CUDA repeatability.
- **Cost:** `K=4` particle storage plus multiple leaf scratch arrays may materially change admission and interactive performance.
- **Optical truth:** Beer-Lambert needs calibrated absorption and path length. Surface swatches are review proxies. Pigment appearance needs scattering calibration and separate licensing review.
- **Schema fragmentation:** cache frames, v2 manifests, checkpoints, Blender attributes, and export formats must migrate together.
- **Existing tests:** attribute/material/cache presence tests are valuable regression coverage but cannot certify diffusion, conservation, or optics.

## Exact evidence index

| Seam | Live evidence | Classification |
|---|---|---|
| Particle inventory/allocation | [`particles.py:16-18`](../../fluidSim_warp_blender51/sim/particles.py#L16), [`143-183`](../../fluidSim_warp_blender51/sim/particles.py#L143) | reusable scaffolding |
| Growth/compaction/checkpoint lifecycle | [`particles.py:242-340`](../../fluidSim_warp_blender51/sim/particles.py#L242), [`347-400`](../../fluidSim_warp_blender51/sim/particles.py#L347) | real generic lifecycle; reseed absent |
| Source color/density initialization | [`sourcing.py:323-330`](../../fluidSim_warp_blender51/sim/sourcing.py#L323), [`970-979`](../../fluidSim_warp_blender51/sim/sourcing.py#L970) | real label assignment |
| Solver settings and optional outputs | [`solver.py:427-555`](../../fluidSim_warp_blender51/sim/solver.py#L427) | reusable configuration surface |
| Sparse field allocation/collisions | [`solver.py:692-722`](../../fluidSim_warp_blender51/sim/solver.py#L692), [`641-646`](../../fluidSim_warp_blender51/sim/solver.py#L641) | real storage; names unavailable for species |
| P2G/ST/viscosity weight | [`solver.py:1520-1587`](../../fluidSim_warp_blender51/sim/solver.py#L1520) | real velocity/ST/viscosity; species absent |
| G2P/advection | [`solver.py:1799`](../../fluidSim_warp_blender51/sim/solver.py#L1799) | real motion; species update absent |
| Substep orchestration | [`solver.py:2067`](../../fluidSim_warp_blender51/sim/solver.py#L2067) | deepest production insertion caller |
| Flat sparse leaf contract | [`fields.py:1-10`](../../fluidSim_warp_blender51/sim/fields.py#L1), [`69-84`](../../fluidSim_warp_blender51/sim/fields.py#L69), [`299-385`](../../fluidSim_warp_blender51/sim/fields.py#L299) | reusable lookup/activation; owned-face halo absent |
| Attribute schema and nearest transfer | [`attributes.py:16-17`](../../fluidSim_warp_blender51/sim/attributes.py#L16), [`54-63`](../../fluidSim_warp_blender51/sim/attributes.py#L54), [`frame_output.py:267`](../../fluidSim_warp_blender51/sim/frame_output.py#L267) | proxy/output-only for mixing |
| Live publication | [`scene_bridge.py:263-346`](../../fluidSim_warp_blender51/blender/scene_bridge.py#L263), [`output_publication.py:1204`](../../fluidSim_warp_blender51/blender/output_publication.py#L1204) | real adapter consumer |
| Cache playback sibling | [`playback.py:237-260`](../../fluidSim_warp_blender51/blender/playback.py#L237), [`output_publication.py:1230`](../../fluidSim_warp_blender51/blender/output_publication.py#L1230) | real non-authoritative presentation lane |
| Cache/checkpoint versions | [`cache.py:34`](../../fluidSim_warp_blender51/io/cache.py#L34), [`cache_schema.py:19`](../../fluidSim_warp_blender51/io/cache_schema.py#L19), [`checkpoint.py:17`](../../fluidSim_warp_blender51/io/checkpoint.py#L17) | real persistence contracts requiring migration |
| Research recommendation | [`color-mixing/REPORT.md`](../color-mixing/REPORT.md), [`SYNTHESIS.md`](../SYNTHESIS.md) | planning evidence, not implementation proof |
