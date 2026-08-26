# Color Mixing / Species Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development` for each task and `superpowers:verification-before-completion` before each commit. Execute tasks in order; do not parallelize writes to `particles.py`, `solver.py`, cache code, or Blender settings adapters.

**Goal:** Add a truthful, GPU-resident clear-dye species-transport vertical slice: two miscible species advect with liquid particles, diffuse conservatively across the existing flat 8x8x8 leaf grid, survive all particle/source/cache lifecycles, and produce derived viewport/output color.

**Architecture:** `ParticleSystem` stores authoritative per-particle represented volume and concentrations; a new deep `SpeciesTransport` module owns sparse P2G, deterministic positive-face diffusion, bounded conservative delta-G2P, receipts, and derived optics. `FluidSolver.step_frame()` remains the sole orchestrator, `simulation_admission.py` remains the preflight/cutover owner, `pending_settings.py` remains settings-policy owner, and `output_publication.py` receives derived payloads only.

**Tech stack:** Python 3.11, NVIDIA Warp kernels/CUDA, NumPy host serialization, Blender 5.1 RNA and background/GUI acceptance, pytest.

**Specs:** `research/implementation-seams/RECOMMENDATION.md`, `research/implementation-seams/color-mixing.md`, `research/color-mixing/REPORT.md`, and root `AGENTS.md`.

## Scope and truth contract

The deliverable is **miscible clear-dye species transport in a constant-density carrier**. It is not RGB interpolation, paint/pigment mixing, material-density coupling, immiscible water/oil separation, thermal transport, or phase change. Authoritative state is concentration, not preview color.

The smallest truthful slice is deliberately narrower than the generic research envelope:

- exactly two active species (`species_count == 2`) stored in lanes x/y of a `wp.vec4`; z/w are zero and validated;
- equal represented particle volume, initialized at every birth to `voxel_size**3 / particles_per_voxel`;
- one common scene-unit diffusivity for both species, including `0.0` for Mixing Off;
- closed-domain transport, analytic/mesh/point sources, and direct injection;
- particle and reconstructed-surface derived clear-dye color;
- production-frame cache and deterministic checkpoint/reopen support;
- feature flag defaults off and allocates/launches/reads back no species resources when off.

Concentrations satisfy `c.x >= 0`, `c.y >= 0`, `c.x + c.y <= 1`; the remainder is clear carrier. Preview color is derived with Beer-Lambert absorption in scene-linear RGB. Because Warp float atomics are order-dependent, acceptance is tolerance-based and must not claim bitwise CUDA repeatability.

Deferred to separate approved plans: `K=3..4`, differential diffusivity, subgrid/turbulent mixing, calibrated turbid pigment/Kubelka-Munk/Mixbox, material/phase density, immiscible interfaces/surface tension, thermal/enthalpy, vapor/gas, and OpenVDB as live storage. Do not add their settings, cache promises, or UI labels in this project.

## Live seam manifest

All non-Create paths and symbols below exist in the inspected dirty worktree at `0d395d78582af1f8876539f0f927110c3eff872d`; preserve its unrelated modifications.

- `fluidSim_warp_blender51/sim/particles.py`: `ParticleSystem.__init__`, `resized_copy`, `checkpoint_arrays`, `restore_checkpoint_prefix`, and `compact`; existing optional channels are `color`, `source_id`, and `uid`.
- `fluidSim_warp_blender51/sim/fields.py`: `SparseGrid.begin_topology`, `mark_points`, `mark_world_box`, `expand`, `finalize_topology`, `restore_topology`; storage is `leaf_table` plus contiguous 512-voxel leaves.
- `fluidSim_warp_blender51/sim/solver.py`: `SolverSettings`, `FluidSolver.__init__`, `ensure_particle_capacity`, `reset`, `export_checkpoint_state`, `_validate_checkpoint_state`, `import_checkpoint_state`, `_rebuild_topology`, `_source`, `step_frame`, `inject_particle_packets_exact`, `inject_particles`, and `host_frame`.
- `fluidSim_warp_blender51/sim/sourcing.py`: `DirectInjectionPacket`, sourcing `apply`, `inject_particle_packets_exact`, `inject_particles_exact`, and `inject_particles`.
- `fluidSim_warp_blender51/sim/source_regions.py`: `AnalyticRegionPlan`, `MeshRegionPlan`, `PointRegionPlan`, and `VolumeRegionPlan`.
- `fluidSim_warp_blender51/sim/registry.py`: `register`, `register_fn`, `lookup`, and `registered_names`.
- `fluidSim_warp_blender51/sim/frame_output.py`: `HostFrame` and `particle_attribute`.
- `fluidSim_warp_blender51/io/cache.py`: `_frame_payload`, `write_frame_v2_atomic`, `_read_v2_frame`, and `read_frame`.
- `fluidSim_warp_blender51/io/checkpoint.py`: `CheckpointPayload`, `capture_solver`, and `restore_solver`.
- `fluidSim_warp_blender51/io/cache_schema.py`: manifest `FORMAT_VERSION == 2`.
- `fluidSim_warp_blender51/blender/solver_settings.py`: `build_settings` and `apply_scene_settings`.
- `fluidSim_warp_blender51/blender/simulation_admission.py`: `AdmissionSnapshot`, `_preflight`, `_allocate_solver`, `_configure_candidate`, `_prepare_solver_replacement`, and `_commit_solver_replacement`.
- `fluidSim_warp_blender51/blender/pending_settings.py`: `validate_transition_registry`, `_reset_signature`, `_reinit_signature`, `refresh_pending_state`, and `capture_applied_state`.
- `fluidSim_warp_blender51/blender/output_publication.py`: `publish_attribute`, `_values_to_colors`, `_write_mesh_colors`, and output update functions in that module.
- `fluidSim_warp_blender51/blender/production_bake.py`: production cache/checkpoint controller.
- Create: `fluidSim_warp_blender51/sim/species.py`, the species-specific tests and acceptance/profile scripts named below.

## Fixed interfaces and data layout

Implement these names and signatures once; later tasks consume them unchanged.

```python
# Create in sim/species.py
@dataclass(frozen=True)
class SpeciesPalette:
    ids: tuple[str, str]
    absorption_rgb: tuple[tuple[float, float, float], tuple[float, float, float]]
    reference_path_length: float
    palette_hash: str

@dataclass(frozen=True)
class SpeciesReceipt:
    before: tuple[float, float]
    sources: tuple[float, float]
    sinks: tuple[float, float]
    after: tuple[float, float]
    residual: tuple[float, float]
    min_concentration: float
    max_loading: float
    limiter_alpha: float
    diffusion_substeps: int

class SpeciesTransport:
    def __init__(self, particles, grid, device, *, enabled: bool,
                 diffusivity: float, max_diffusion_substeps: int,
                 palette: SpeciesPalette): ...
    @staticmethod
    def estimate_bytes(particle_capacity: int, voxel_capacity: int,
                       enabled: bool) -> int: ...
    def reset(self) -> None: ...
    def diffuse(self, dt: float) -> None: ...       # device-only, no receipt readback
    def record_source(self, amount: tuple[float, float]) -> None: ...
    def record_sink(self, amount: tuple[float, float]) -> None: ...
    def finalize_frame_receipt(self) -> SpeciesReceipt: ...  # one frame-boundary sync
    def derive_preview(self, count: int): ...       # returns device wp.vec4 array
    def checkpoint_arrays(self, count: int) -> dict[str, np.ndarray]: ...
    def checkpoint_scalars(self) -> dict[str, object]: ...
    def restore_checkpoint(self, arrays, scalars, count: int) -> None: ...
```

`ParticleSystem` gains optional float channel `represented_volume` and optional vec4 channel `species_concentration`, each with current and compaction double-buffer arrays. They are allocated together only when `species_enabled`. `species_concentration[p] = (c0,c1,0,0)`; `represented_volume[p] > 0`. Growth and compaction copy both exactly.

`SpeciesTransport` scratch layout when enabled:

- per particle capacity: `particle_delta: wp.vec4`, `particle_alpha_candidate: float`;
- per voxel capacity: `mix_volume: float`, `species_mass: wp.vec4`, `species_delta: wp.vec4`, `outgoing_conductance: float`, `node_scale: float`;
- per voxel positive-face owner: `conductance_x/y/z: float` and `face_delta_x/y/z: wp.vec4`;
- scalars: `global_alpha: float`, two-species before/source/sink/after reductions, min/max diagnostics.

The exact admission charge is therefore `60 * particle_capacity + 104 * voxel_capacity + scalar/reduction workspace bytes` while enabled: 40 B/particle persistent double buffers, 20 B/particle scratch, and 104 B/voxel scratch. `estimate_bytes` must derive from dtype item sizes and include the actual reduction workspace, not copy these constants internally. Disabled charge is zero.

Each active voxel owns only its positive x/y/z faces. Cross-leaf neighbors resolve through `SparseGrid.leaf_table`; inactive or out-of-domain neighbors are impermeable. No dense or hierarchical grid is introduced.

For face `(i,j)`, use the same scalar conductance for both species:

`raw_lambda = D * dt * open_area / distance`, `delta_Q = lambda * (C_j - C_i)`.

First accumulate `raw_lambda` at both nodes, then set `node_scale_i=min(1, mix_volume_i/sum_lambda_i)` and `lambda=raw_lambda*min(node_scale_i,node_scale_j)`. This makes each grid update a convex combination, keeps both channels and their sum bounded, and applies each face exchange once as equal/opposite mass. Delta-G2P distributes node `delta_Q` by the same normalized P2G weights. Compute one device-wide `global_alpha` from every particle/channel lower/upper bound and the `c0+c1<=1` bound, then multiply **all** particle deltas by that one alpha. A uniform alpha preserves the zero global delta; independent clamping is forbidden.

Diffusion substeps are `ceil(6 * D * dt / voxel_size**2)` with minimum 1 when `D>0`; admission rejects a configuration needing more than `species_max_diffusion_substeps` (default 16). `D==0` performs no diffusion launches and leaves concentrations invariant.

## Task 1: Particle schema and disabled-feature contract

**Files:**

- Modify: `fluidSim_warp_blender51/sim/particles.py`
- Modify: `fluidSim_warp_blender51/sim/solver.py` (`SolverSettings`, `FluidSolver.__init__`, `ensure_particle_capacity`)
- Create: `tests/test_species_state_contract.py`

- [ ] Write RED tests proving: disabled construction has neither new channel and matches the old optional inventory; enabled construction allocates both; invalid half-selection raises `ValueError`; `resized_copy` and `compact` preserve alignment; new births with nonpositive volume or invalid simplex values are rejected before count mutation.
- [ ] Run `rtk proxy python -m pytest -q tests/test_species_state_contract.py tests/test_particle_capacity_growth.py`; expect RED because the optional float inventory and species channels do not exist.
- [ ] Extend `ParticleSystem` with `OPTIONAL_FLOAT_ATTR_NAMES`, typed dummy float storage, current/double optional-float dictionaries, and copy/compact/checkpoint handling. Add the vec4 concentration channel. Preserve every existing kernel argument and behavior when disabled.
- [ ] Add default-off `SolverSettings` fields: `species_enabled=False`, `species_count=2`, `species_diffusivity=0.0`, `species_max_diffusion_substeps=16`, two stable IDs, two scene-linear preview swatches, two nonnegative RGB absorption triples, and positive reference path length. Fail closed unless count is exactly 2 in this slice.
- [ ] Run the same command; expect GREEN with all existing particle-growth tests unchanged.
- [ ] Commit only these files as `feat(species): add lifecycle-safe particle state`.

**Rollback boundary:** no solver phase uses the fields yet; reverting this commit returns exact prior behavior and cache shape.

## Task 2: Flat-leaf stencil and deterministic face ownership

**Files:**

- Modify: `fluidSim_warp_blender51/sim/fields.py`
- Modify: `fluidSim_warp_blender51/sim/capacity_planner.py`
- Create: `tests/test_sparse_species_stencil.py`

**Produces:** `SparseGrid.expand(iterations=1)` remains the admission mechanism; Create helpers `neighbor_flat_index(flat_index: wp.int32, axis: int, direction: int, ...) -> wp.int32` and `positive_face_neighbor(flat_index: wp.int32, axis: int, ...) -> wp.int32` for kernels.

- [ ] Write RED CPU-oracle and Warp tests for interior, face, edge, and corner neighbors; translations across every local index 7→0 leaf seam; missing-neighbor impermeability; exactly one positive-axis owner per admitted face; and topology-capacity failure when the one-cell halo cannot be admitted.
- [ ] Run `rtk proxy python -m pytest -q tests/test_sparse_species_stencil.py tests/test_sparse_leaf_grid_contract.py`; expect RED on missing neighbor/owner helpers.
- [ ] Add device functions using the existing flat `leaf_table` and 512-entry leaf layout. Extend capacity demand with an explicit one-leaf halo when species is enabled; do not alter live topology when disabled.
- [ ] Run the same command; expect GREEN, including the existing sparse-grid contract.
- [ ] Commit as `feat(grid): add flat-leaf species stencil support`.

**Rollback boundary:** generic helpers and admission only; no physics launches. Reject rather than truncate a halo at capacity.

## Task 3: Conservative bounded GPU diffusion and solver phase

**Files:**

- Create: `fluidSim_warp_blender51/sim/species.py`
- Modify: `fluidSim_warp_blender51/sim/solver.py` (`FluidSolver.__init__`, `reset`, `step_frame`, stats/checkpoint hooks)
- Modify: `fluidSim_warp_blender51/sim/registry.py`
- Modify: `fluidSim_warp_blender51/diagnostics/profiler.py`
- Create: `tests/test_species_diffusion.py`
- Modify: `tests/test_profiler_phase_contract.py`

**Kernel ownership:** all Create kernels live in `sim/species.py` and register these canonical names: `species_p2g_clear`, `species_p2g_accumulate`, `species_face_conductance_measure`, `species_node_scale`, `species_face_flux_apply`, `species_g2p_delta`, `species_particle_alpha`, `species_delta_apply`, `species_bounds_budget_reduce`, `species_preview_derive`.

- [ ] Write RED tests for Mixing Off identity; two-particle and two-cell equal/opposite exchange; nonnegative/simplex bounds; conservation after global-alpha limiting; positive-face single ownership; cross-leaf translation equivalence; no NaN/Inf; and no host conversion/synchronization inside `diffuse`.
- [ ] Add a RED phase-order test that spies `FluidSolver.step_frame()` and requires `sourcing -> p2g -> species -> forces -> viscosity -> pressure -> g2p`; expect missing `species` profiler slot/kernel registrations.
- [ ] Run `rtk proxy python -m pytest -q tests/test_species_diffusion.py tests/test_profiler_phase_contract.py`; expect RED on missing module and phase.
- [ ] Implement the fixed interfaces/data layout and kernels above. Reuse the liquid transfer shape weights; do not use display `color`, grid `density`, viscosity `attr_weight`, CPU arrays, or a second grid hierarchy.
- [ ] Instantiate `self.species` only when enabled. After `_p2g(dt)` in every substep call `self.species.diffuse(dt)` under `_phase("species")`; finalize the receipt once at frame end into `FrameStats`. When disabled, preserve the previous launch sequence and profiler values.
- [ ] Run focused GREEN: `rtk proxy python -m pytest -q tests/test_species_diffusion.py tests/test_profiler_phase_contract.py tests/test_particle_grid_transfer.py tests/test_sparse_leaf_grid_contract.py`.
- [ ] Commit as `feat(species): add conservative sparse diffusion phase`.

**Rollback boundary:** feature remains default off. Any conservation, seam, or hidden-sync failure blocks later adapters.

## Task 4: Source, direct-injection, sink, and capacity transactions

**Files:**

- Modify: `fluidSim_warp_blender51/sim/source_regions.py` plan records
- Modify: `fluidSim_warp_blender51/sim/sourcing.py` (`DirectInjectionPacket`, `apply`, exact injection methods)
- Modify: `fluidSim_warp_blender51/blender/sources.py` (`PreparedParticleSeed`, gather/commit paths)
- Modify: `fluidSim_warp_blender51/sim/solver.py` public injection adapters
- Create: `tests/test_species_source_lifecycle.py`

**Interface additions:** every liquid source plan/packet carries `species_concentration: tuple[float,float,float,float]`; direct APIs add keyword-only `species_concentration=(0.0,0.0,0.0,0.0)`. Birth writes both concentration and the fixed represented volume in the same all-or-zero transaction. Sink deletion records `represented_volume*c` before compaction.

- [ ] Write RED tests for analytic, mesh, point, prepared-seed, and direct packets; exact-fit growth; refused batch leaves counts/budgets unchanged; initial-fill fires once; continuous source receipts are substep-independent; reset rearms; and sink/source conservation closes `after = before + sources - sinks` within `1e-6` for a one-frame fixture.
- [ ] Run `rtk proxy python -m pytest -q tests/test_species_source_lifecycle.py tests/test_source_emission_lifecycle.py tests/test_particle_capacity_growth.py`; expect RED on missing composition fields.
- [ ] Thread composition through existing owned plans and transactions. Validate finite, nonnegative, z/w zero, x+y<=1 in the deep sourcing seam, not Blender callbacks. Preserve the entire batch for retry on capacity failure.
- [ ] Run the same command; expect GREEN with existing source lifecycle tests.
- [ ] Commit as `feat(species): preserve composition through source lifecycle`.

**Rollback boundary:** source schema is additive and the species-disabled default produces the existing birth payload exactly.

## Task 5: Additive cache/checkpoint schema and migration

**Files:**

- Modify: `fluidSim_warp_blender51/sim/solver.py` checkpoint export/validate/import
- Modify: `fluidSim_warp_blender51/io/checkpoint.py`
- Modify: `fluidSim_warp_blender51/io/cache.py`
- Modify: `fluidSim_warp_blender51/io/cache_schema.py` only if manifest validation needs the additive species record
- Modify: `fluidSim_warp_blender51/blender/production_bake.py`
- Create: `tests/test_species_cache_io.py`

**Migration policy:** keep global cache-v2 manifest `FORMAT_VERSION == 2` and frame format 2. Add `species_schema_version=1`, `particle_species_concentration` (`float32[N,4]`), `particle_represented_volume` (`float32[N]`), stable IDs, palette JSON/hash, settings, cumulative budgets, and receipt state. Old v2 frames/checkpoints with no marker load only when the current solver has species disabled. A species-enabled restore with absent/unknown marker, wrong palette hash, wrong shapes, nonfinite values, or violated simplex fails before solver mutation with “species resume state is incompatible; regenerate the cache.” Derived RGB is never sufficient resume state.

- [ ] Write RED tests for enabled round trip, checkpoint/reopen, palette hash identity, exact authoritative counts, corrupt shapes/nonfinite/simplex rejection, unknown species schema rejection, old-cache disabled migration, and old-cache enabled refusal with candidate solver unchanged.
- [ ] Run `rtk proxy python -m pytest -q tests/test_species_cache_io.py tests/test_cache_io.py tests/test_scene_bridge_cache_playback.py`; expect RED on absent species payload and validation.
- [ ] Extend `ParticleSystem.checkpoint_arrays`, `FluidSolver.export_checkpoint_state/import_checkpoint_state`, `_frame_payload`, v2 reader, checkpoint capture/restore, and production bake transaction. Validate all arrays before assigning topology or particle prefixes.
- [ ] Run the same command; expect GREEN.
- [ ] Commit as `feat(cache): persist authoritative species state`.

**Rollback boundary:** additive format preserves species-disabled cache playback. Never silently synthesize concentrations from legacy RGB.

## Task 6: Derived optics, HostFrame, publication, and Blender settings policy

**Files:**

- Modify: `fluidSim_warp_blender51/sim/frame_output.py` (`HostFrame`)
- Modify: `fluidSim_warp_blender51/sim/attributes.py`
- Modify: `fluidSim_warp_blender51/blender/output_publication.py`
- Modify: `fluidSim_warp_blender51/blender/solver_settings.py`
- Modify: `fluidSim_warp_blender51/blender/pending_settings.py`
- Modify: `fluidSim_warp_blender51/properties.py`
- Modify: `fluidSim_warp_blender51/panels.py`
- Modify: `fluidSim_warp_blender51/operators.py` only where source composition needs an existing operator adapter
- Create: `tests/test_species_output_ui.py`

**Output contract:** Create `HostFrame.species_concentrations(stride=1)` and `HostFrame.species_preview_colors(stride=1)`. Publish authoritative diagnostic attribute `fluidsim_species` as `FLOAT_COLOR` only when explicitly selected, and derived scene-linear `fluidsim_color` for particles/surface. Surface color samples concentrations first, then applies Beer-Lambert; it must not interpolate already-converted sRGB.

- [ ] Write RED tests proving lazy one-readback-per-request, no species readback when unselected/disabled, known Beer-Lambert endpoint/midpoint values, particle/surface derivation from the same concentrations, and cache playback presentation without solver-policy calls.
- [ ] Add RED schema tests requiring domain properties `species_enabled`, `species_diffusivity`, `species_max_diffusion_substeps`, palette IDs/swatches/absorption/reference length, publication toggles, and source `species_concentration`; expect `validate_transition_registry` to report unclassified fields.
- [ ] Run `rtk proxy python -m pytest -q tests/test_species_output_ui.py tests/test_frame_output.py tests/test_pending_settings_registry.py tests/test_operator_panel_behavior.py`; expect RED on missing accessors/properties/classification.
- [ ] Implement adapters. Classify enable/count/palette as reinitialize, diffusivity/substep cap as reset (not live mutation), publication toggles as live/output-only, and source composition under the existing source-reset policy. `build_settings` copies values; it does not normalize physics. Label UI “Clear Dye Species” and “Effective Diffusivity”; never “Paint,” “Pigment,” or “Water/Oil.”
- [ ] Run the same command; expect GREEN.
- [ ] Commit as `feat(blender): expose truthful clear-dye outputs`.

**Rollback boundary:** output/color is derived and removable without touching authoritative cache state; settings remain default off.

## Task 7: Admission, memory, analytic acceptance, playback, and performance

**Files:**

- Modify: `fluidSim_warp_blender51/sim/memory_safety.py`
- Modify: `fluidSim_warp_blender51/sim/capacity_planner.py`
- Modify: `fluidSim_warp_blender51/blender/simulation_admission.py`
- Create: `tests/blender_species_transport_acceptance.py`
- Create: `tools/profile_species_transport.py`
- Create: `tests/test_species_acceptance_report.py`
- Modify: `tools/run_production_gate.py` to include the focused lane after it is stable

- [ ] Write RED admission tests: exact species bytes and halo leaves appear in the single `AdmissionSnapshot`; excessive diffusion substeps and insufficient leaf/VRAM capacity reject before `_commit_solver_replacement`; candidate failure preserves the old solver; disabled estimates remain byte-for-byte unchanged.
- [ ] Run `rtk proxy python -m pytest -q tests/test_simulation_admission.py tests/test_memory_safety_recovery.py tests/test_species_state_contract.py`; expect RED on absent species demand.
- [ ] Add `SpeciesTransport.estimate_bytes` and halo/cache staging demand to capacity/memory assessment. Keep all demand in preflight before Replacement Cutover; do not allocate candidate species buffers to discover their size.
- [ ] Implement the background acceptance script with deterministic fixtures and JSON receipt. It must check: Mixing Off invariant; 240-frame closed-box relative conservation `<=1e-5`; min concentration `>=-1e-6`; max loading `<=1+1e-6`; no NaN/Inf; Gaussian diffusion L2 `<=2%`; face/edge/corner translated seam L2 `<=1e-5`; checkpoint resume concentration L2 `<=1e-4`; changing particle and surface color after saved-scene reopen; and zero device-to-host transfers inside substeps.
- [ ] Run `blender --background --factory-startup --python tests/blender_species_transport_acceptance.py -- --device cuda:0 --json-out artifacts/species_transport_acceptance.json`; expect `SPECIES-TRANSPORT-ACCEPTANCE-PASS`.
- [ ] Run `rtk proxy python -m pytest -q tests/test_species_acceptance_report.py` against that JSON; expect GREEN only when every threshold and runtime identity is present.
- [ ] Profile disabled, transport-only (`D=0`), and diffusion-on (`K=2`) with `blender --background --factory-startup --python tools/profile_species_transport.py -- --frames 30 --particles 200000 --json-out artifacts/species_transport_profile.json`. Require: disabled adds zero kernels/readbacks/allocated bytes; transport-only median frame overhead `<=12%`; diffusion-on overhead `<=35%`; measured peak allocation within `5%` of admission estimate; no per-substep host synchronization. A missed budget blocks gate integration; do not raise the budget in the same commit.
- [ ] Run focused full lane: `rtk proxy python -m pytest -q tests/test_species_state_contract.py tests/test_sparse_species_stencil.py tests/test_species_diffusion.py tests/test_species_source_lifecycle.py tests/test_species_cache_io.py tests/test_species_output_ui.py tests/test_species_acceptance_report.py tests/test_particle_capacity_growth.py tests/test_sparse_leaf_grid_contract.py tests/test_source_emission_lifecycle.py tests/test_cache_io.py tests/test_frame_output.py tests/test_simulation_admission.py tests/test_pending_settings_registry.py tests/test_profiler_phase_contract.py`.
- [ ] Run project gates: `rtk proxy python -m pytest -q` then `rtk proxy python tools/run_production_gate.py`. For final GUI evidence run `rtk proxy python tools/run_production_gate.py --include-gui`, then `rtk proxy python tools/validate_policy_drift_reports.py` only after report refresh is intentionally in scope.
- [ ] Commit as `test(species): gate clear-dye transport end to end`.

**Rollback boundary:** do not add the lane to the default production gate until focused runtime, cache resume, performance, and saved playback all pass. Generated JSON and demo artifacts are evidence, not implementation truth, and are not committed by this plan unless a separate evidence-refresh request authorizes them.

## Final review checklist

- [ ] Diff review shows no production policy in handlers, playback, publication, panels, demo builders, or reports.
- [ ] Every authoritative field is covered by allocation, growth, compaction, reset, source/sink, checkpoint, cache, admission, and teardown.
- [ ] Kernel inventory and profiler have one canonical species phase; disabled mode has no launches or reads.
- [ ] Global face ownership is based on canonical grid coordinates, not tile-local iteration order.
- [ ] All conservation claims use represented-volume-weighted species amounts and explicit source/sink receipts.
- [ ] Preview RGB is derived; cache resume never reconstructs chemistry from RGB.
- [ ] Failure paths preserve the old solver and whole source batch.
- [ ] No claim exceeds “two-species miscible clear-dye transport in a constant-density carrier.”
- [ ] Each commit passes its focused GREEN command before the next task begins; no tests are weakened.

