# Different-Density and Immiscible Multiphase Implementation Plan

> **For agentic workers:** execute this plan task-by-task with TDD and a review gate after every commit. Do not merge archived experiments. Re-read `AGENTS.md`, `research/implementation-seams/RECOMMENDATION.md`, `research/implementation-seams/density-multiphase.md`, and `research/density-multiphase/REPORT.md` before implementation.

**Goal:** Add truthful different-density liquid behavior first, then a conservative two-liquid immiscible mode, while keeping the simulation GPU-resident, the shared MAC velocity, `PressureSolver.project_frame()` as the only production projection seam, and the flat 8x8x8 sparse leaf table.

**Architecture:** Add one deep `MaterialInterfaceState` module that owns material identity, volume/fraction state, sparse interface geometry, coefficient production, conservation, and checkpoint state. Deepen the existing `PressureSolver` so one coefficient/jump representation is used by diagonal construction, matrix application, PCG/Jacobi, and velocity correction. Stage 1 is explicitly a numerically mixed variable-density liquid; Stage 2 adds conservative VOF/PLIC-style interface transport and is the first tier allowed to claim immiscibility.

**Tech stack:** Python 3, NVIDIA Warp CUDA kernels, Blender 5.1 Python API, NumPy cache/checkpoint payloads, pytest, Blender background/GUI acceptance.

**Specs:** `research/implementation-seams/RECOMMENDATION.md`; `research/implementation-seams/density-multiphase.md`; `research/density-multiphase/REPORT.md`.

## Truth contract and scope

- Existing particle `density` is display/source metadata, existing grid `FluidSolver.density` is ST-FLIP liquid activity, and ST “phase” is liquid-versus-gas occupancy. None is water/oil material physics. Migrate names; never reinterpret old cache values as kg/m3.
- `VARIABLE_DENSITY_MIXED` promises mass-weighted transfer, density-derived body response, and variable-coefficient projection. It does **not** promise a retained oil/water boundary, interfacial tension, droplets, or wetting.
- `TWO_IMMISCIBLE` may be exposed only after exact phase identity, bounded conservative `alpha`, interface reconstruction, sparse-boundary invariance, cache/restart fidelity, and tagged internal-interface output pass.
- `PressureSolver.project_frame()` remains the production seam. Do not create a second pressure solver, per-phase projection, per-phase velocity grid, dense pressure grid, tree storage, or CPU substep path.
- Shared-path callers are `FluidSolver.step_frame()` -> `blender.scene_bridge.Session.step()`, live publication, sequential Production Cache Bake, checkpoint/resume, cache playback, and cached publication. `playback.play_cache_frame()` is the non-solver sibling. Existing `SINGLE` mode is the contrasting lane and must remain behaviorally compatible.
- Bitwise repeatability is not required; GPU atomics, compaction, and reconstruction ordering require tolerance/convergence envelopes.

## Fixed interfaces and layouts

Create `fluidSim_warp_blender51/sim/material_interface.py` with these public types; supporting Warp kernels may remain private in that module unless file size requires `sim/kernels_material.py`.

```python
class MaterialMode(IntEnum):
    SINGLE = 0
    VARIABLE_DENSITY_MIXED = 1
    TWO_IMMISCIBLE = 2

@dataclass(frozen=True)
class MaterialSpec:
    name: str
    density_kg_m3: float
    dynamic_viscosity_pa_s: float
    free_surface_tension_n_m: float = 0.0

@dataclass(frozen=True)
class TwoMaterialTable:
    material_a: MaterialSpec
    material_b: MaterialSpec
    liquid_liquid_surface_tension_n_m: float = 0.0

@dataclass(frozen=True)
class PressureCoefficientBundle:
    topology_epoch: int
    coefficient_epoch: int
    inv_rho_u: object
    inv_rho_v: object
    inv_rho_w: object
    jump_q_u: object
    jump_q_v: object
    jump_q_w: object
    density_min: float
    density_max: float

@dataclass(frozen=True)
class MaterialFaceState:
    pressure: PressureCoefficientBundle
    phase_fraction: object
    interface_phi: object | None
    interface_flags: object | None

@dataclass(frozen=True)
class ConservationStats:
    volume_a: float
    volume_b: float
    ledger_error_a: float
    ledger_error_b: float
    alpha_min: float
    alpha_max: float

class MaterialInterfaceState:
    def __init__(self, grid, device, settings): ...
    def prepare_substep(self, particles, bins, voxel_type, dt: float) -> MaterialFaceState: ...
    def after_advection(self, particles, bins, dt: float) -> ConservationStats: ...
    def export_checkpoint_state(self) -> dict: ...
    def import_checkpoint_state(self, state: dict) -> None: ...
```

Persistent device layout for non-`SINGLE` modes:

- Particle SoA: `phase_id: int32[particle_capacity]` (`0=A`, `1=B`) and `represented_volume: float32[particle_capacity]` in both compaction buffers. Invalid IDs and non-positive/non-finite volume fail admission, injection, restore, and source compilation.
- Active-cell SoA over `leaf_capacity * 512`: `phase_fraction: float32` (`alpha`, A volume fraction), `liquid_activity: float32` (renamed from grid `density`), and a two-component phase-volume ledger or two transient `float32` deposit arrays. Do not persist an independently advected density field; derive `rho(alpha)`.
- Face SoA over the same lower-face indexing used by `MacVelocityField`: `inv_rho_u/v/w: float32`. Stage 2 adds transient `jump_q_u/v/w`, `cut_fraction_u/v/w`, interface flags, narrow-band `interface_phi`, normals, curvature, and reconstruction/flux scratch only on interface plus guard leaves.
- Tile flags: `CONTAINS_A`, `CONTAINS_B`, `INTERNAL_INTERFACE`, `INTERFACE_GUARD`. Address through the existing dense `leaf_table` and slot-major `leaf_coords`; fixed `LEAF_DIM=8`, `LEAF_VOLUME=512` remain unchanged.

Coefficient convention:

- The generalized path solves `q = dt * p` in `A q = -div(u*) h^2`; constant legacy mode keeps its current folded `q = dt*p/rho` path when no bundle is supplied.
- Cell density is `rho_c = alpha*rho_A + (1-alpha)*rho_B`.
- For an uncut liquid face with two active cells, use the symmetric harmonic average of `beta=1/rho`: `beta_f = 2*beta_L*beta_R/(beta_L+beta_R) = 2/(rho_L+rho_R)`. At a free surface use the liquid-side coefficient; at a collision face use the existing Neumann/collider treatment.
- For a reconstructed A/B cut face, let `theta` be the fraction of center-to-center distance in the left material, clamped only to `[1e-4, 1-1e-4]`; use `beta_f = 1/(theta*rho_L + (1-theta)*rho_R)`. Every coefficient must be finite and positive.
- Store oriented `jump_q = dt*(p_R-p_L)` on positive-axis faces. Define `interface_phi < 0` in A, normal A->B, and `[p]=p_A-p_B=sigma_AB*kappa`; therefore an A-left/B-right face stores `jump_q=-dt*sigma_AB*kappa`.
- Assembly and correction must call one shared Warp helper equivalent to `beta_f * (q_R - q_L - jump_q)`. The linear matrix uses the same `beta_f`; the constant jump term is moved to RHS with the sign obtained from that helper. A one-face unit test must prove the signs, symmetry, positive diagonal, and corrected flux.

## Task 1: Contracts, settings, and admission (identity only)

**Files:**

- Modify `fluidSim_warp_blender51/sim/enums.py` (`MaterialMode`).
- Create `fluidSim_warp_blender51/sim/material_interface.py` (types and validation only).
- Modify `fluidSim_warp_blender51/sim/solver.py` (`SolverSettings`, `FrameStats`, `FluidSolver.__init__`).
- Modify `fluidSim_warp_blender51/properties.py` (`FLUIDSIM_PG_settings`, `FLUIDSIM_PG_object`).
- Modify `fluidSim_warp_blender51/blender/solver_settings.py` (`DIRECT_SETTING_FIELDS`, `enum_setting_fields`, `build_settings`).
- Modify `fluidSim_warp_blender51/blender/pending_settings.py` (`_RESET_REGISTRY_FIELDS`, `_REINITIALIZE_REGISTRY_FIELDS`, `_OUTPUT_REGISTRY_FIELDS`, `_PRESENTATION_FIELDS`).
- Modify `fluidSim_warp_blender51/blender/simulation_admission.py` (`AdmissionSnapshot`, `capture_snapshot`, `SimulationAdmission._preflight`).
- Modify `fluidSim_warp_blender51/panels.py` (`FLUIDSIM_PT_physics`, `FLUIDSIM_PT_liquid_emitters`, diagnostics/output panels).
- Test `tests/test_material_interface_state.py`, `tests/test_simulation_admission.py`, `tests/test_pending_settings_registry.py`, `tests/test_operator_panel_behavior.py`.

Add settings named exactly `material_mode`, `material_a_name`, `material_a_density`, `material_a_viscosity`, `material_b_name`, `material_b_density`, `material_b_viscosity`, `liquid_liquid_surface_tension`, `publish_internal_interface`, and source property `source_material_slot` (`A`/`B`). Mode/material changes are `REINITIALIZE`; source assignment is `RESET`; output toggle is `OUTPUT`; diagnostics are presentation-only. Admission constructs one immutable `TwoMaterialTable`, validates finite density `>0`, viscosity `>=0`, tension `>=0`, exactly two unique slot names, and rejects tension in mixed mode. Capture the table inside `AdmissionSnapshot`; adapters must not rebuild it during allocation.

RED first:

```powershell
rtk proxy python -m pytest -q tests/test_material_interface_state.py tests/test_simulation_admission.py tests/test_pending_settings_registry.py tests/test_operator_panel_behavior.py
```

Expected RED: missing `MaterialMode`/`MaterialInterfaceState`, undeclared properties absent from the exhaustive registry, and invalid tables not rejected pre-allocation. GREEN requires `SINGLE` defaults to reproduce the current settings record and no material buffers are allocated.

Commit boundary: `feat(materials): define two-material admission contract`.

## Task 2: Particle identity, volume, sources, growth, and checkpoint schema

**Files:**

- Modify `fluidSim_warp_blender51/sim/particles.py` (`OPTIONAL_FLOAT_ATTR_NAMES` Create, `OPTIONAL_INT_ATTR_NAMES`, `ParticleSystem.__init__`, `checkpoint_arrays`, `restore_checkpoint_prefix`, `compact`).
- Modify `fluidSim_warp_blender51/sim/particle_capacity.py` (copy every enabled float/int channel during growth).
- Modify `fluidSim_warp_blender51/sim/attributes.py` (`persistent_particle_channels`; keep legacy `density` semantics unchanged).
- Modify `fluidSim_warp_blender51/sim/source_regions.py` (`PayloadPlan`, `SourcePlan`, `CompiledPoints`, `CompiledRegion`, `CompiledSourceFootprint`).
- Modify `fluidSim_warp_blender51/sim/sourcing.py` (`SourceShapeDesc`, `MeshSourceDesc`, `DirectInjectionPacket`, all particle-emission kernels, `_upload_direct_packets`, `inject_particle_packets_exact`).
- Modify `fluidSim_warp_blender51/sim/solver.py` (`export_checkpoint_state`, `_validate_checkpoint_state`, `import_checkpoint_state`).
- Modify `fluidSim_warp_blender51/io/checkpoint.py` (`CHECKPOINT_FORMAT_VERSION=2`; exact v1 single-mode migration).
- Modify `fluidSim_warp_blender51/sim/memory_safety.py` (`estimate_solver_bytes`).
- Test `tests/test_source_emission_lifecycle.py`, `tests/test_particle_capacity_growth.py`, `tests/test_point_source_pipeline.py`, `tests/test_source_region_compilation.py`, `tests/test_cache_io.py`.

Extend optional channel inventory to accept `represented_volume` (float32) and `phase_id` (int32), enabled iff mode is not `SINGLE`. All analytic, mesh, point, volume, height, compiled, and direct-injection paths write the two channels in the same transaction as position/velocity. Default volume is `voxel_size**3 / resolved_quality.rest_particle_count`; exact injection must either supply a positive volume array or opt into that explicit default. Split/reseed copies phase and partitions volume; merge is same-phase only and sums volume and momentum.

Checkpoint v2 adds `schema_version`, `material_mode`, `material_table_signature`, `particle_attr_phase_id`, `particle_attr_represented_volume`, and `material_state`. Read checkpoint v1 only into `SINGLE`; never infer phase from `particle_density`, color, or source ID. A v1 resume requested in either new mode fails with “legacy checkpoint has no authoritative material state.”

Memory estimate adds two double-buffered 4-byte particle channels, interface-state persistent bytes by mode, and peak deposit/reconstruction scratch. The estimate must be conservative and participate in `SimulationAdmission._preflight` before Replacement Cutover.

RED first:

```powershell
rtk proxy python -m pytest -q tests/test_source_emission_lifecycle.py tests/test_particle_capacity_growth.py tests/test_point_source_pipeline.py tests/test_source_region_compilation.py tests/test_cache_io.py
```

Expected RED: missing channels, emitted/grown/compacted IDs misaligned, checkpoint v1 incorrectly accepted for multiphase, and memory estimate unchanged. GREEN requires 100% exact ID preservation and sum of represented volume unchanged by growth/compaction within float32 summation tolerance.

Commit boundary: `feat(materials): preserve phase identity and represented volume`.

## Task 3: Sparse material deposition and mixed-density transfer

**Files:**

- Modify `fluidSim_warp_blender51/sim/fields.py` (`SparseGrid` topology epoch/guard activation; no storage-model change).
- Modify `fluidSim_warp_blender51/sim/material_interface.py` (`deposit_phase_volume`, `normalize_phase_fraction`, `update_material_coefficients`, `prepare_substep`).
- Modify `fluidSim_warp_blender51/sim/kernels_fluid.py` (`k_particle_set_grid_data`, `k_particle_set_grid_data_spacetime`, `k_normalize_mac`).
- Modify `fluidSim_warp_blender51/sim/solver.py` (`density` -> `liquid_activity` with temporary internal alias, `_clear_grid_fields`, `_p2g`, `step_frame`).
- Modify `fluidSim_warp_blender51/sim/custom.py` so custom field name `density` remains the documented display/source field and cannot alias physical `rho`.
- Test `tests/test_material_interface_state.py`, `tests/test_sparse_leaf_grid_contract.py`, `tests/test_pressure_projection_regressions.py`, `tests/test_snippet_library.py`.

In `VARIABLE_DENSITY_MIXED`, `prepare_substep` deposits `V_A`, `V_B`, computes bounded `alpha=V_A/(V_A+V_B)`, cell density, and face coefficients every substep. This alpha is reconstructed from particles and is deliberately allowed to numerically mix; it is not an immiscible interface state.

Make P2G mass-aware without duplicating velocity grids: per particle `m_i=rho[phase_id]*represented_volume`; scatter momentum `w*m_i*v` and mass `w*m_i`, then normalize. Preserve separate volume/validity sums so ST liquid activity is not derived from mass and changing density cannot change the fluid mask. G2P still samples the one shared velocity and never changes phase identity.

RED first:

```powershell
rtk proxy python -m pytest -q tests/test_material_interface_state.py tests/test_sparse_leaf_grid_contract.py tests/test_pressure_projection_regressions.py tests/test_snippet_library.py
```

Expected RED: alpha buffers absent, equal velocity with unequal particle masses is count-weighted, ST occupancy changes when density changes, or a phase cell on a face/edge/corner leaf boundary differs from the translated in-leaf case. GREEN requires `0<=alpha<=1`, finite positive coefficients, exact single-mode dispatch compatibility, and no `.numpy()`/full-array CPU readback in substeps.

Commit boundary: `feat(materials): add GPU mixed-density transfer state`.

## Task 4: Deepen the existing pressure solver and close mixed-density scope

**Files:**

- Modify `fluidSim_warp_blender51/sim/pressure.py` (`PressureSolver.project_frame`, ST coefficient construction, inverse diagonal, matrix/Jacobi/PCG application, correction, status).
- Modify `fluidSim_warp_blender51/sim/solver.py` (`FluidSolver._pressure`, `FrameStats`).
- Test `tests/test_variable_density_projection.py` Create, `tests/test_pressure_projection_regressions.py`, `tests/test_performance_hotpath_contracts.py`.

Change the production signature compatibly:

```python
def project_frame(self, velocity, voxel_type, collision_velocity,
                  voxel_counts, settings, *, max_gradient,
                  coefficients: PressureCoefficientBundle | None = None,
                  reference_mass=None, pressure_source=None):
```

When `coefficients is None`, keep current constant behavior; when ST variable projection is enabled, populate the same internal bundle from ST occupancy and use the generalized kernels; when material state supplies a bundle, validate device, array lengths, topology epoch, coefficient epoch, min/max density, and identical object identity across assembly/correction. Remove duplicate ST mathematics only after parity tests pass; do not add another solver class.

Status adds `coefficient_min`, `coefficient_max`, `density_ratio`, `post_projection_divergence_linf`, and `condition_warning`. Pressure-source callbacks remain after solve and before threshold/correction, but jump RHS must be assembled before solve. PCG failure continues to prevent unsafe projection.

RED first:

```powershell
rtk proxy python -m pytest -q tests/test_variable_density_projection.py tests/test_pressure_projection_regressions.py tests/test_performance_hotpath_contracts.py
```

Expected RED: bundle argument missing; a one-face coefficient differs between matrix and correction; equal-density coefficient path differs from constant baseline; hydrostatic slope ignores density; ratios 1/2/10/100 do not report convergence. GREEN acceptance:

- Equal-density/mixed negative control: corrected velocity and pressure residual match constant path within `5e-6` relative/absolute float32 tolerance.
- Manufactured divergence: post-projection `L_inf <= max(10*pressure_tolerance, 1e-5)` and decreases under grid refinement.
- Hydrostatic two-layer column: fitted `dp/dz=-rho_i*g` in each layer; normalized slope error decreases for two successive refinements and finest-grid error is `<=5%` before product exposure.
- Ratios 1, 2, 10, 100: finite SPD coefficients, no NaN, no unsafe projection, residual reaches configured tolerance. Ratio 1000 is a reported stress test, not a release promise.

Commit boundary: `feat(pressure): project mixed variable-density liquid`.

**Rollback boundary:** Tasks 1-4 are a complete shippable tier behind `VARIABLE_DENSITY_MIXED`. If later interface work fails, retain this tier and keep `TWO_IMMISCIBLE` hidden/unsupported.

## Task 5: Cache v3, production bake, playback, and mixed-density UI/output

**Files:**

- Modify `fluidSim_warp_blender51/io/cache_schema.py` (`FORMAT_VERSION=3`, `CacheManifestV3`; v2 read adapter).
- Modify `fluidSim_warp_blender51/io/cache.py` (`CACHE_FORMAT_VERSION=3`, `_frame_payload`, `AUTHORITATIVE_PAYLOAD_FIELDS`, v1/v2 migration, v3 read/write).
- Modify `fluidSim_warp_blender51/io/checkpoint.py`, `fluidSim_warp_blender51/blender/production_bake.py`, `fluidSim_warp_blender51/blender/cache_bake.py`.
- Modify `fluidSim_warp_blender51/sim/frame_output.py` (`HostFrame` phase/material channels).
- Modify `fluidSim_warp_blender51/blender/output_publication.py` (`PublicationPayload`, `prepare_live_payload`, `prepare_cache_payload`, `publish_payload`).
- Modify `fluidSim_warp_blender51/blender/playback.py` (`_check_cache_scene_key`, `play_cache_frame`).
- Test `tests/test_cache_io.py`, `tests/test_scene_bridge_cache_playback.py`, `tests/test_frame_output.py`, `tests/test_production_gate_truthfulness.py`.

Cache v3 stores `material_mode`, canonical JSON material table + SHA-256 signature, `particle_phase_id`, `particle_represented_volume`, active `phase_fraction` with its `grid_leaf_coords`, and per-phase conservation counters. Production manifest binds the material signature. V2/v1 remain readable only as `SINGLE`; a legacy cache may be displayed but cannot resume a multiphase solver. Preview remains non-authoritative. Production Cache Bake writes sequential authoritative v3 frames and checkpoint v2 state, then atomically commits the manifest as today.

Mixed-mode output publishes `fluidsim_phase_id`, `fluidsim_material_index`, and `fluidsim_material_density`; legacy `fluidsim_density` remains display metadata. Do not publish or label an internal-interface mesh in mixed mode.

RED first:

```powershell
rtk proxy python -m pytest -q tests/test_cache_io.py tests/test_scene_bridge_cache_playback.py tests/test_frame_output.py tests/test_production_gate_truthfulness.py
```

Expected RED: v3 schema unknown, material signature not scene-bound, phase arrays omitted/misaligned, or playback invents multiphase state from v1/v2. GREEN requires exact int32 ID round trip, represented-volume and alpha round trip within float32 serialization tolerance, verified checksums, and unchanged `SINGLE` legacy playback.

Commit boundary: `feat(cache): version material state and publication`.

## Task 6: Conservative two-liquid transport and sparse interface reconstruction

**Files:**

- Modify `fluidSim_warp_blender51/sim/material_interface.py` (authoritative immiscible state, PLIC reconstruction, VOF flux, reconciliation, `after_advection`).
- Create `fluidSim_warp_blender51/sim/kernels_material.py` only if keeping Warp kernels in the owner module would make review unsafe; it is an implementation detail, not another orchestrator.
- Modify `fluidSim_warp_blender51/sim/fields.py` (transactional interface-guard leaf marking).
- Modify `fluidSim_warp_blender51/sim/solver.py` (`step_frame`: prepare after P2G; conservative alpha update after projected advection; lifecycle ledger before compaction).
- Modify `fluidSim_warp_blender51/sim/sourcing.py` (grid/particle source volume transaction and sink/open-boundary ledger).
- Test `tests/test_immiscible_interface.py` Create, `tests/test_material_interface_state.py`, `tests/test_sparse_leaf_grid_contract.py`, `tests/test_source_emission_lifecycle.py`.

In `TWO_IMMISCIBLE`, `phase_fraction` becomes authoritative and is not overwritten by particle deposition. Reconstruct each `0<alpha<1` cell with a mixed-Youngs centered normal; solve the PLIC plane constant by fixed 12-iteration monotone bisection to match cell volume. Advect with directionally split geometric face fluxes under the existing CFL substeps; the lower-face owner writes each flux once, including across leaf boundaries. Apply equal-and-opposite cell updates so global phase volume changes only through sources, sinks, or open boundaries. Clamp only roundoff excursions `<=8*float32_eps`; larger bound violations set a failing diagnostic and stop the frame.

Activate one whole neighboring guard leaf in all 26 directions around interface leaves, transactionally through existing topology admission. If guard demand exceeds capacity, fail before swapping topology and preserve the last live state. Rebuild narrow-band `interface_phi` from PLIC planes; use alpha for conservation and phi only for geometry. Particle/grid reconciliation may split particles with exact volume partition or merge same-phase particles with summed volume/momentum; it must never cross phases or relabel from display attributes.

RED first:

```powershell
rtk proxy python -m pytest -q tests/test_immiscible_interface.py tests/test_material_interface_state.py tests/test_sparse_leaf_grid_contract.py tests/test_source_emission_lifecycle.py
```

Expected RED: alpha overwritten by particle deposit, bounds violated, leaf-crossing flux double-counted, or translated interface changes result. GREEN acceptance:

- `alpha` is bounded after every sweep/substep.
- Closed-domain phase-volume ledger equals conservative grid volume within `1e-6` relative plus one-cell roundoff allowance; error decreases under two refinements for rotating and notched-body advection.
- Face/edge/corner and adjacent-leaf translations agree within `5e-6` on alpha/flux.
- Equal-density stirring retains a coherent internal interface and performs materially better under refinement; no fixed product drift percentage is claimed until the convergence report exists.
- No full live state transfers to CPU.

Commit boundary: `feat(multiphase): advect a conservative two-liquid interface`.

**Rollback boundary:** if this task misses conservation or GPU budget, do not expose `TWO_IMMISCIBLE`; revert/hide Tasks 6+ without altering the validated mixed-density tier.

## Task 7: Sharp interface coefficients and zero-tension immiscible closure

**Files:**

- Modify `fluidSim_warp_blender51/sim/material_interface.py` (cut-face classification and theta).
- Modify `fluidSim_warp_blender51/sim/pressure.py` (cut coefficient and zero-jump RHS/correction through the existing bundle).
- Modify `fluidSim_warp_blender51/sim/viscosity.py` only to enforce the documented initial limitation; do not claim a sharp viscous stress jump.
- Test `tests/test_variable_density_projection.py`, `tests/test_immiscible_interface.py`, `tests/test_viscosity_active_work_regressions.py`.

Use reconstructed theta and the exact cut coefficient above. `jump_q` is zero in this task. Initially allow equal material viscosity or a measured bounded ratio `<=10`; admission rejects higher sharp-viscosity contrast until a consistent stress-jump solve exists. Triple-line cells use zero-tension interface continuation and existing solid no-penetration; dynamic contact-line behavior is explicitly unsupported.

RED/GREEN:

```powershell
rtk proxy python -m pytest -q tests/test_variable_density_projection.py tests/test_immiscible_interface.py tests/test_viscosity_active_work_regressions.py
```

Expected RED: cut theta ignored or inconsistent with correction. GREEN requires coefficient/jump one-face analytic tests, hydrostatic piecewise slopes, stable heavy-below-light rest, physically correct early Rayleigh-Taylor center-of-mass exchange, and convergence rather than a hard-coded late-time target.

Commit boundary: `feat(multiphase): couple sharp interface to pressure`.

## Task 8: Internal-interface output and truthful Blender acceptance

**Files:**

- Modify `fluidSim_warp_blender51/sim/surface.py`/the live `SurfacePipeline` owner, `sim/frame_output.py`, `sim/attributes.py`.
- Modify `fluidSim_warp_blender51/blender/output_publication.py`, `properties.py`, `panels.py`.
- Create `tests/blender_multiphase_output_acceptance.py` and `tests/blender_variable_density_acceptance.py`.
- Test `tests/test_frame_output.py`, `tests/test_cache_io.py`, `tests/test_operator_panel_behavior.py`.

Extract the liquid-liquid PLIC/phi interface separately from the union liquid-air surface. Publish `fluidsim_phase_id`, `fluidsim_material_index`, and `fluidsim_interface_kind` (`0=liquid_air`, `1=liquid_liquid`) on meshes/particles; bind Blender material slots A/B without using display color as identity. UI labels mixed mode “Variable-Density Mixed (no retained boundary)” and only labels Stage 2 “Two Immiscible Liquids.” Warn on unresolved subgrid droplets.

```powershell
rtk proxy python -m pytest -q tests/test_frame_output.py tests/test_cache_io.py tests/test_operator_panel_behavior.py
blender --background --factory-startup --python tests/blender_variable_density_acceptance.py
blender --background --factory-startup --python tests/blender_multiphase_output_acceptance.py
```

Expected RED: internal surface absent/material tags lost or UI overclaims mixed mode. GREEN requires live output, Production Cache Bake, reopen, and cache playback to show identical material assignment while playback remains non-authoritative.

Commit boundary: `feat(output): publish tagged liquid interfaces`.

## Task 9: Interfacial tension (separate opt-in closure)

**Files:** modify `sim/material_interface.py`, `sim/pressure.py`, `sim/solver.py` (`_compute_substeps`, `FrameStats`), admission/UI files; create `tests/test_interfacial_tension.py` and `tests/blender_interfacial_tension_acceptance.py`.

Compute normals/curvature from the narrow-band geometry using the same guard data and cut faces used by projection. Populate oriented `jump_q` with `sigma_AB*kappa` as defined above; do not add a separate force-only pressure bypass. Add capillary limit `dt_cap = C_cap*sqrt(rho_min*h**3/max(sigma_AB,eps))`, report the chosen `C_cap` from a convergence calibration, and include `CAPILLARY` as a substep-limiter reason.

```powershell
rtk proxy python -m pytest -q tests/test_interfacial_tension.py tests/test_variable_density_projection.py tests/test_immiscible_interface.py
blender --background --factory-startup --python tests/blender_interfacial_tension_acceptance.py
```

Expected RED: Laplace jump zero/wrong sign, matrix/correction geometry differs, or capillary limit unreported. GREEN requires 3D droplet `Delta p=2*sigma/R` error and maximum spurious velocity both decrease for two refinements, plus capillary oscillation frequency/damping convergence. Machine-accuracy equilibrium is a reference target, not an initial release promise.

Commit boundary: `feat(multiphase): add balanced interfacial pressure jump`.

## Task 10: Performance, full gates, and release decision

**Files:** modify `diagnostics/profiler.py`, `tools/profile_gpu_solver.py`, production gate registration, and focused acceptance tests; update user docs/changelog only after executable gates pass.

Telemetry: phase volumes/counts/centers of mass; alpha min/max; interface/guard leaves and guard amplification; density/coefficient ratio; pressure iterations/residual/divergence; material P2G, reconstruction, pressure, internal meshing time; peak estimated/observed VRAM; capillary limiter; unresolved-interface count. Bounded scalar readback is allowed; full arrays are not.

Performance budgets are baseline-relative because research has no measured product baseline. Capture the same scene/device/driver and commit before setting release numbers:

- `SINGLE`: median frame time and peak VRAM must regress `<=5%` from its pre-feature baseline; no material allocations or dispatches.
- `VARIABLE_DENSITY_MIXED`: provisional gate `<=25%` median frame-time and `<=20%` peak-VRAM overhead versus `SINGLE`, with no playback-cadence miss. If the first measured vertical slice cannot meet it, keep the mode experimental and record the measured budget; do not tune tests to an invented threshold.
- `TWO_IMMISCIBLE` without tension: provisional gate `<=60%` median frame-time and `<=35%` peak-VRAM overhead; guard amplification `guard_leaves/max(interface_leaves,1) <=27` by construction.
- Tension: measure separately; release only when capillary substeps still satisfy the documented interactive scene cadence. No unsupported universal ratio is claimed.

Focused then broad verification:

```powershell
rtk proxy python -m pytest -q tests/test_material_interface_state.py tests/test_variable_density_projection.py tests/test_immiscible_interface.py tests/test_interfacial_tension.py
rtk proxy python -m pytest -q tests/test_particle_capacity_growth.py tests/test_source_emission_lifecycle.py tests/test_cache_io.py tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py tests/test_simulation_admission.py
rtk proxy python -m pytest -q tests/test_sparse_leaf_grid_contract.py tests/test_pressure_projection_regressions.py tests/test_viscosity_active_work_regressions.py tests/test_production_gate_truthfulness.py
rtk proxy python -m pytest -q
rtk proxy python tools/run_production_gate.py
"C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" tools/profile_gpu_solver.py --device cuda:0 --frames 8 --warmup 2 --json-out artifacts/gpu_solver_profile.json
python tools/run_production_gate.py --include-gui
python tools/validate_policy_drift_reports.py
```

Run hydrostatic refinement, rotating/notched advection, stable stratification, Rayleigh-Taylor, Laplace droplet, capillary oscillation, and Hysing moderate rising-bubble reference curves on fixed scene definitions. Release requires error/conservation convergence, not merely non-crash. Hysing difficult breakup remains an envelope comparison.

Commit boundary: `test(multiphase): gate conservation convergence and GPU budget`.

## Explicit deferrals and stop conditions

- Static contact angle/wetting is a later plan after tension passes; dynamic contact lines and extreme angles are unsupported.
- General N-phase, pairwise N-by-N tension, triple-junction laws, MultiFLIP/free-slip dual velocities, phase field/Cahn-Hilliard, dense FFT, tree AMR, and sharp variable-viscosity stress jumps are deferred.
- A particle-label SDF without conservative alpha may be used only as a quarantined experiment and cannot close Task 6.
- Stop and retain the last rollback boundary if pressure fails at ratio 100, alpha is not bounded/convergent, phase volume changes without an owned source/sink/open boundary, topology guard allocation is non-transactional, cache migration invents material state, full live arrays cross to CPU, or `SINGLE` exceeds its regression budget.

## Final review checklist

- Every new setting is in the exhaustive pending-settings registry and the coherent Admission Snapshot.
- Every source/injection/growth/compaction/delete/reseed/checkpoint/cache path preserves phase and represented volume.
- `PressureSolver.project_frame()` is still the only production projection entry and assembly/correction consume the same bundle.
- Mixed and immiscible product labels match their executable physics.
- Cache v1/v2 migration maps only to `SINGLE`; v3 and checkpoint v2 bind material signatures.
- Preview/cache playback is not described as authoritative restart state; Production Cache Bake remains sequential and bypasses preview publication handlers.
- Sparse storage remains fixed 8x8x8 flat leaves; interface guards are bounded and transactional.
- Tests state their expected RED cause before implementation and acceptance uses direct runtime/convergence evidence.
