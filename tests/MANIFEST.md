# tests/ manifest

Root-level repository tests. Two suites exist: this folder (`tests/`) and the add-on's own
`fluidSim_warp_blender51/tests/` (see the end). `pytest.ini` runs both:

`gate/test_storage_hygiene_integration.py` proves the normal gate's pre/post
reports, real vendored storage suite, substituted full orchestration, and
failure/exit preservation without launching DCC/GPU work. The checker itself has
26 stdlib regressions under `tools/storage_hygiene/tests/`.

```
testpaths = tests fluidSim_warp_blender51/tests
pythonpath = .
addopts = --strict-markers      # markers: cuda, blender, slow
```

Run everything with `python -m pytest -q -p no:cacheprovider`. Deselect long tests with
`-m "not slow"`. Without a CUDA device the `cuda` tests skip; `FLUIDSIM_TEST_DEVICE=cpu`
(or `cuda:1`) overrides the device. The `blender` marker is not enforced anywhere: tests that
need the real `bpy` use `pytest.importorskip("bpy")` and skip under a plain interpreter.

## Layout


| Path | Files | What is in it |
|---|---|---|
| `conftest.py` | 1 | `cuda_device` fixture, auto-`cuda` marker, private Warp kernel cache, adds `support/` to `sys.path` |
| `test_ownership_map.py` | 1 | Pins `docs/ownership-map.md`. Stays at the root because `tools/ownership_map.py` writes that path into the generated doc |
| `support/` | 2 | Shared helpers (not tests) |
| `validation/` | 7 | Sparse-grid / transfer / pressure / viscosity numerical validation |
| `solver/` | 19 | Solver kernels and contracts (CUDA kernels, pressure, viscosity, whitewater) |
| `runtime/` | 30 | Add-on runtime logic that runs without Blender: settings, capacity, scene bridge, checkpoints, Warp discovery |
| `sources/` | 16 | Fluid sources, sinks and force regions |
| `surface/` | 7 | Render-surface (SDF) pipeline and viewport point-cloud output |
| `cache_io/` | 5 | Cache IO and output publication |
| `ui/` | 13 | Operators, panels, guided setup and artist workflows tested against a stub `bpy` |
| `gate/` | 7 | Production-gate and performance-budget machinery (the gate itself is `tools/run_production_gate.py`) |
| `perf/` | 11 | Benchmark, profiler and playback-performance tooling contracts |
| `demos/` | 2 | Demo scene geometry and demo-analysis acceptance criteria |
| `repo/` | 5 | Repository, packaging and release-surface contracts |
| `acceptance/` | 43 | Blender scripts (NOT pytest): see below |
| `scripts/` | 3 | Headless/smoke scripts (NOT pytest): see below |

## Rules for this folder

- No `__init__.py`: pytest uses prepend import mode, so every `test_*.py` basename must be unique across the whole repo's test folders, and a test file's own folder is on `sys.path`. Files that import each other (the `ui/` group) must stay together.
- Helpers are imported by bare name (`from settings_doubles import ...`, `from _test_device import ...`) from `support/`; `conftest.py` adds it for pytest and each Blender script inserts it itself, because Blender's `--python` does not add the script's folder to `sys.path`.
- Scripts under `acceptance/` and `scripts/` find the repo root with `Path(__file__).resolve().parents[2]`. Keep them exactly two levels below the root.
- The gate (`tools/run_production_gate.py`) names scripts by path. Moving one means updating the gate; `gate/test_production_gate_coverage.py` fails if an acceptance script is neither run nor excused there.
- Nothing is generated; `__pycache__` is disposable.


## `support/`

Shared helpers (not tests). `conftest.py` puts this folder on `sys.path`; acceptance scripts insert it themselves.

| File | Covers | Needs |
|---|---|---|
| `_test_device.py` | Warp device choice shared by pytest and the Blender/headless scripts. | CUDA |
| `settings_doubles.py` | Scene doubles shaped like the split FluidSim settings groups. | - |

## `validation/`

Sparse-grid / transfer / pressure / viscosity numerical validation. `test_validation.py`, `test_solver_validation.py` are also run by the gate as plain scripts.

| File | Covers | Needs |
|---|---|---|
| `_subset_runner.py` | Shared runner for focused validation-subset test modules. | CUDA |
| `test_field_indexing.py` | Sparse tiled leaf-grid indexing/addressing validation (tests 1-3). | CUDA |
| `test_particle_grid_transfer.py` | P2G/G2P transfer validation (tests 4-5). | CUDA |
| `test_pressure_projection.py` | Pressure projection + solver convergence validation (tests 8-9). | CUDA |
| `test_solver_validation.py` | End-to-end solver validation: sourcing, collision, surface, whitewater. | CUDA |
| `test_validation.py` | Headless CUDA validation suite runner (no Blender required). | CUDA, slow |
| `test_viscosity.py` | Viscosity solve stability validation (test 11). | - |

## `solver/`

Solver kernels and contracts (CUDA kernels, pressure, viscosity, whitewater).

| File | Covers | Needs |
|---|---|---|
| `test_cfl_clamp_reporting.py` | Regression tests for surfacing the CFL substep clamp. | - |
| `test_collision_sdf_gradient.py` | REVIEW-017: collider normals at the edge of the sparse topology. | CUDA |
| `test_color_mixing.py` | Color-mixing kernels and per-particle color/attribute transfer. | CUDA |
| `test_cuda_reduction_abi.py` | Argument/ABI contract of the CUDA reduction kernels. | - |
| `test_cuda_reduction_active_count.py` | Device-counted dot product: a graph-capturable reduction whose length is read from ``num_leaves`` at launch instead of being baked into the launch. | CUDA |
| `test_force_subsystem_contracts.py` | Regression contracts for the force settings and dispatch seam. | - |
| `test_particle_capacity_growth.py` | Particle buffer auto-growth policy and limits. | - |
| `test_particles_per_voxel_cap.py` | REVIEW-005: one liquid-occupancy ceiling, not six copies of the literal 64. | - |
| `test_pressure_projection_regressions.py` | Focused pressure projection safety regressions. | CUDA |
| `test_pressure_st_and_graph_cache.py` | PCG initialization and CUDA-graph cache regressions. | CUDA |
| `test_solver_checkpoint_roundtrip.py` | A real solver's exported checkpoint restores into a fresh solver, and the checkpoint scalars carry no removed-feature keys. | CUDA |
| `test_solver_collaborator_contract.py` | Sub-solvers read only the pinned private solver attributes; the surface-transfer slot is public. | - |
| `test_sparse_leaf_grid_contract.py` | Regression guard for the live sparse tiled leaf-grid wording contract. | - |
| `test_validation_releases_native_leases.py` | Every GPU self-validation must return its native CUDA kernel leases. | CUDA |
| `test_viscosity_active_work_regressions.py` | Focused viscosity active-work regressions. | CUDA |
| `test_viscosity_conditional_pcg_cuda.py` | Device-evaluated viscosity PCG convergence check (CUDA conditional graph). | CUDA |
| `test_voxel_type_marking_contract.py` | Voxel-type marking must be additive, matching paradigmParticleToGridActive. | - |
| `test_whitewater_kernel_outputs.py` | Whitewater kernels must write every ``out_*`` array they declare. | - |
| `test_whitewater_truth_contract.py` | Truth contracts for whitewater SDF dependency and capacity admission. | - |

## `runtime/`

Add-on runtime logic that runs without Blender: settings, capacity, scene bridge, checkpoints, Warp discovery. Scene-bridge tests install a stub `bpy` with `monkeypatch`.

| File | Covers | Needs |
|---|---|---|
| `test_addon_core_regressions.py` | Focused regressions for bpy/Warp-independent add-on authorities. | - |
| `test_bake_run.py` | The one BakeRun lifecycle, bake registry and restart-entry list. | - |
| `test_capacity_approval.py` | Capacity Approval transaction (settings write, clamp read-back, domain resize, rollback) with stand-ins. | - |
| `test_capacity_planner.py` | Capacity planner estimates and recommendations. | - |
| `test_capacity_service.py` | Contracts for the single active capacity assessment/recommendation service. | - |
| `test_checkpoint_state.py` | Pure checkpoint schema validation tests. | - |
| `test_cross_extension_warp.py` | The active Warp module is process-wide even when extensions bundle wheels. | - |
| `test_device_preference_routing.py` | Regression tests for the single device-preference lookup. | CUDA |
| `test_frame_change_catchup.py` | Regression tests for the live frame-change catch-up gate and handler teardown. | - |
| `test_frame_features.py` | Frame feature sequencing contracts. | - |
| `test_memory_safety_recovery.py` | Regression contracts for pre-allocation GPU memory recovery. | - |
| `test_pending_settings_registry.py` | Exhaustive/disjoint Pending Settings transition registry contracts. | - |
| `test_phase2_review_repairs.py` | Regressions for the Phase 2 review repairs. | CUDA |
| `test_preset_schema_matches_properties.py` | The preset schema matches the Blender properties it assigns. | - |
| `test_quality_contract.py` | Pure and caller-parity tests for the authoritative quality resolver. | - |
| `test_registration.py` | One registration table; operators discovered from their modules. | - |
| `test_readback_stride_and_discovery.py` | Device-side display decimation and lazy Warp discovery contracts. | - |
| `test_scene_access.py` | Active Domain accessors and scene identity have one owner. | - |
| `test_scene_bridge_cache_playback.py` | Regression tests for cache playback frame-change routing. | - |
| `test_scene_bridge_capacity_initialization.py` | Capacity initialization when the scene bridge starts a session. | - |
| `test_scene_bridge_collider_cache.py` | Regression tests for collider Warp mesh cache invalidation. | - |
| `test_scene_bridge_settings.py` | Regression tests for Blender scene-to-solver settings mapping. | - |
| `test_session_runtime.py` | The Solver Session owns its runtime fields; Simulation Admission only asks. | - |
| `test_session_capacity_transactions.py` | Trace regressions for active capacity and scene-owned plan transactions. | - |
| `test_settings_default_agreement.py` | Settings declarations agree with each other and with their meaning. | - |
| `test_settings_getattr_names.py` | Defaulted settings reads must name a field that exists. | - |
| `test_slowdown_telemetry.py` | Regression tests for slowdown diagnostics and reset cache behavior. | - |
| `test_solver_settings_projection.py` | Focused contracts for owned Blender-to-solver value projection. | - |
| `test_topology_transaction.py` | Behavioral contracts for transactional sparse topology and leaf-demand quality. | - |
| `test_warp_discovery.py` | Regression coverage for the explicit external-Warp developer override. | - |

## `sources/`

Fluid sources, sinks and force regions.

| File | Covers | Needs |
|---|---|---|
| `test_compiled_source_operations.py` | Compiled source operations (emit/sink) run on the GPU. | CUDA |
| `test_height_source_pipeline.py` | Height-field source pipeline. | - |
| `test_mesh_source_pipeline.py` | GPU behavior checks for evaluated-triangle sources and sinks. | - |
| `test_point_source_pipeline.py` | Point-cloud source pipeline. | CUDA |
| `test_source_apply_result.py` | Source apply-result records and reporting. | - |
| `test_source_compatibility.py` | Source type/mode compatibility rules. | - |
| `test_source_emission_lifecycle.py` | Milestone 1: Initial Fill / Continuous Inflow emission lifecycle. | - |
| `test_source_encoding.py` | Add-on source carries no UTF-8-read-as-cp1252 mojibake. | - |
| `test_source_field_force_contract.py` | Primary-source contracts for FIELD mode and analytic FORCE regions. | - |
| `test_source_identity.py` | Scene-input session generation and object identity tracking. | - |
| `test_source_plan_contract.py` | Source plan data contract. | - |
| `test_source_region_compilation.py` | Compilation of source regions into solver-ready form. | - |
| `test_source_sampling_migration.py` | Migration and effective-value contracts for source liquid sampling. | - |
| `test_sources_object_points_release.py` | object_points_world always releases the evaluated mesh (REVIEW-041). | - |
| `test_sourcing_pipeline.py` | Interface checks for the stateful fluid sourcing Module. | - |
| `test_volume_source_pipeline.py` | Volume (fog/SDF grid) source pipeline. | - |

## `surface/`

Render-surface (SDF) pipeline and viewport point-cloud output.

| File | Covers | Needs |
|---|---|---|
| `test_surface_attribute_transfer.py` | Attribute transfer onto the render surface. | - |
| `test_surface_pipeline_ownership.py` | Which module owns each surface-pipeline stage. | - |
| `test_surface_resolution_feature_contracts.py` | Contracts for decoupled render-surface resolution. | - |
| `test_surface_sdf_band_propagate_cuda.py` | Error bounds for the closest-point band extension of the particle SDF. | CUDA |
| `test_surface_sdf_splat_cull_cuda.py` | Bit-identity of the culled particle-SDF splat against the full-cube splat. | CUDA |
| `test_surface_stack.py` | Regression tests for the realtime liquid surfacing filter stack. | - |
| `test_viewport_pointcloud_prewarm.py` | Regression test for B6's point-cloud half: the first live write after ``ensure_outputs`` must not force ``bpy.data.pointclouds.new``/``.remove()`` from inside ``frame_change_post`` (see output_publication._new_prewarmed_pointcloud). | - |

## `cache_io/`

Cache IO and output publication. Named `cache_io`, not `io`, so it cannot shadow the stdlib.

| File | Covers | Needs |
|---|---|---|
| `test_cache_io.py` | Cache write/read round trip and sequence layout. | - |
| `test_frame_output.py` | Host-frame readback and live/cache payload assembly (some tests import `bpy`, skipped without). | CUDA, some tests need `bpy` |
| `test_io_dependency_contract.py` | Dependency-direction contracts for low-level cache IO. | - |
| `test_io_path_policy.py` | Cache path policy (relative/absolute, safe locations). | - |
| `test_output_publication_buffers.py` | Exercise publication against Blender RNA's bulk-buffer contract. | - |

## `ui/`

Operators, panels, guided setup and artist workflows tested against a stub `bpy`. Kept in one folder because these files import each other.

| File | Covers | Needs |
|---|---|---|
| `test_artist_apply.py` | Restart decisions follow the registry and preserve progress on cancellation. | - |
| `test_artist_bake_safety.py` | Regression coverage for artist-facing production bake safety. | - |
| `test_artist_effect_admission.py` | Draft effects enter admission without prematurely publishing applied evidence. | - |
| `test_artist_mode_callback_ownership.py` | Scene-ownership regressions for Blender property mode callbacks. | - |
| `test_artist_preview.py` | User actions must drive playback without resetting compatible progress. | - |
| `test_artist_setup.py` | Focused contracts for transactional Guided Setup composition. | - |
| `test_file_compat.py` | 2.0 recognizes .blend files saved before FluidSim 1.1 and says so. | - |
| `test_operator_panel_behavior.py` | Operator and runtime regressions retained after the Properties UI migration. | CUDA |
| `test_operator_poll_and_guards.py` | Guard regressions: operator polls, panel draw safety, released solvers, and out-of-domain topology boxes. | - |
| `test_phase3_source_ui_schema.py` | Phase 3 source-UI schema contract. | - |
| `test_phase4_pending_interaction.py` | Behavioral contracts for Phase-4 pending/reinitialize interaction. | - |
| `test_simulation_admission.py` | Characterization and ownership contracts for Simulation Admission. | - |
| `test_ui_tooltip_contract.py` | Static contract for user-facing Blender tooltip metadata. | - |

## `gate/`

Production-gate and performance-budget machinery (the gate itself is `tools/run_production_gate.py`).

| File | Covers | Needs |
|---|---|---|
| `test_demo_budget_runner.py` | Fresh-child failure propagation and the release lane's fixed contract. | - |
| `test_gate_marker_and_playback_validation.py` | Production-gate positive-marker and playback-report truth contracts. | - |
| `test_performance_budget.py` | Focused tests for the production performance-budget evidence reducer. | - |
| `test_production_gate_coverage.py` | Every Blender acceptance script is run by the production gate or excused. | - |
| `test_production_gate_semantics.py` | Meaning of each production-gate check (pass/fail rules). | - |
| `test_production_gate_truthfulness.py` | The gate cannot report PASS on missing or stale evidence. | - |
| `test_source_acceptance_report.py` | Source acceptance JSON report shape and validation. | - |

## `perf/`

Benchmark, profiler and playback-performance tooling contracts.

| File | Covers | Needs |
|---|---|---|
| `test_benchmark_validation.py` | Fail-closed benchmark comparison and baseline operator regressions. | - |
| `test_benchmark_versions.py` | tools/benchmark_versions.py: ratios pair each trial with the same round. | - |
| `test_blender_performance_playback_benchmark_contract.py` | Contract of the Blender playback benchmark script's report. | - |
| `test_compare_playback_performance_trials.py` | Comparison of playback performance trials. | - |
| `test_liquid_compaction_benchmark.py` | Liquid compaction benchmark harness. | - |
| `test_performance_hotpath_contracts.py` | Small contracts preventing the measured live-playback hot paths returning. | CUDA |
| `test_pressure_performance_probe.py` | Pressure-segment attribution probe (plan 2026-08-26-pressure-follow-on-performance, Task 1). | CUDA |
| `test_profile_safe_addon_review.py` | Focused contracts for the profile-safe Blender review launcher. | - |
| `test_profile_user_path_contract.py` | Contract of the user-path profiler. | - |
| `test_profiler_phase_contract.py` | Contract tests for the canonical solver profiler phase vocabulary. | - |
| `test_whitewater_finalization_benchmark.py` | Whitewater finalization benchmark harness. | - |

## `demos/`

Demo scene geometry and demo-analysis acceptance criteria.

| File | Covers | Needs |
|---|---|---|
| `test_dam_break_cove.py` | Contracts for the dam_break rocky-cove collider geometry. | - |
| `test_demo_analysis_contract.py` | Regression contracts for demo-analysis acceptance criteria. | - |

## `repo/`

Repository, packaging and release-surface contracts.

| File | Covers | Needs |
|---|---|---|
| `test_contract_wording.py` | Wording contracts in shipped docs/source. | - |
| `test_cuda_device_fixture.py` | Regression gate for the shared ``cuda_device`` fixture (no GPU required). | CUDA |
| `test_extension_manifest.py` | Blender extension manifest contents. | - |
| `test_install_local.py` | Host-side checks for tools/release/install_local.py (no Blender needed). | - |
| `test_release_1_0_253_integration.py` | Integration contract for the current release surface. | - |

## `acceptance/` (non-pytest Blender scripts)

Not collected by pytest (no `test_` prefix). Each is run as `blender --background --factory-startup --python <script>` and prints a `...-PASS` marker that the gate checks. They need Blender 5.1+, the add-on's bundled Warp, and for most a CUDA GPU. `_phase2_source_acceptance_common.py` is a shared helper (not a script to run).

### Run by the production gate (32)

| Script | Proves |
|---|---|
| `blender_animated_cache_playback_acceptance.py` | Blender/CUDA acceptance: a bake with an animated collider plays and resumes. |
| `blender_animated_source_acceptance.py` | Phase 2 source acceptance, a thin wrapper over `_phase2_source_acceptance_common.py`. |
| `blender_attribute_viz_acceptance.py` | Background acceptance for viewport attribute visualization modes. |
| `blender_background_acceptance.py` | Blender --background acceptance test for the FluidSim Warp add-on. |
| `blender_cache_export_pipeline.py` | Blender --background gate for cache sequence export and playback. |
| `blender_capacity_planner_acceptance.py` | Real CUDA acceptance for fail-closed source-capacity transactions. |
| `blender_chromatic_vortex_containment_acceptance.py` | Prove the chromatic-vortex demo starts inside a real vessel collider. |
| `blender_closed_domain_stability_acceptance.py` | CUDA regression for closed-domain drift and artificial energy growth. |
| `blender_color_mixing_runtime_acceptance.py` | Focused Blender/CUDA proof for the saved chromatic-vortex workflow. |
| `blender_color_mixing_ui_acceptance.py` | Focused Blender 5.1 acceptance for color-mixing authoring controls. |
| `blender_custom_effect_example_scenes_acceptance.py` | Blender/CUDA playback of the 12 Custom Effect example scenes: ship untrusted, apply once trusted, particles advance. Manual (not in the gate's background list). |
| `blender_custom_effects_acceptance.py` | Blender 5.1 acceptance for built-in Custom Effects and package import. |
| `blender_force_region_acceptance.py` | Real Blender/CUDA acceptance for object-level FluidSim FORCE regions. |
| `blender_gui_acceptance.py` | Interactive Blender GUI acceptance test for the FluidSim Warp add-on. |
| `blender_gui_playback_probe.py` | Drive real Blender timeline playback and report solver advancement. |
| `blender_handler_persistence_gate.py` | Regression gate for timeline playback after opening a .blend file. |
| `blender_memory_safety_gate.py` | Real Blender gate for pre-allocation stop and confirmed recovery. |
| `blender_mesh_source_acceptance.py` | Blender/Warp acceptance for modifier-evaluated mesh surface sourcing. |
| `blender_mixed_source_batch_acceptance.py` | CUDA acceptance for unified source ownership and atomic direct injection. |
| `blender_phase3_source_ui_acceptance.py` | Blender 5.1 acceptance for the Phase 3 role-first source UI translators. |
| `blender_phase4_live_interaction_acceptance.py` | Blender/CUDA acceptance for Phase-4 live and pending interaction. |
| `blender_point_source_acceptance.py` | Phase 2 source acceptance, a thin wrapper over `_phase2_source_acceptance_common.py`. |
| `blender_pre_bridge_file_acceptance.py` | Headless .blend round trip: 2.0 flags files saved before FluidSim 1.1. |
| `blender_quick_setup_idempotence.py` | Acceptance: Quick Setup is transactional, composable, and idempotent. |
| `blender_simulation_admission_acceptance.py` | Authoritative Blender/CUDA acceptance for Simulation Admission. |
| `blender_source_lifecycle_acceptance.py` | Phase 2 source acceptance, a thin wrapper over `_phase2_source_acceptance_common.py`. |
| `blender_source_region_acceptance.py` | Phase 2 source acceptance, a thin wrapper over `_phase2_source_acceptance_common.py`. |
| `blender_source_sampling_acceptance.py` | Blender RNA acceptance for source sampling defaults and migration. |
| `blender_tooltip_acceptance.py` | Blender RNA acceptance for user-facing FluidSim tooltip metadata. |
| `blender_topology_transaction_acceptance.py` | Blender/CUDA acceptance for transactional sparse-topology refusal. |
| `blender_viewport_2frame_gate.py` | Production gate: prove live viewport simulation advances for two frames. |
| `blender_volume_source_acceptance.py` | Phase 2 source acceptance, a thin wrapper over `_phase2_source_acceptance_common.py`. |
| `blender_whitewater_truth_acceptance.py` | Blender/CUDA acceptance for render-independent SDF and honest overflow. |

### Manual probes, deliberately not run by the gate (9)

| Script | Why the gate does not run it |
|---|---|
| `blender_artist_bake_deleted_scene_acceptance.py` | needs a real GUI modal bake; exits silently in --background |
| `blender_artist_bake_gui_acceptance.py` | timer-driven GUI probe: call start(report_path) in a GUI session |
| `blender_artist_context_consent_acceptance.py` | needs real editor contexts and simulated ESC/RET events in a GUI session |
| `blender_artist_workflow_acceptance.py` | GUI timeline probe: call start(report_path) in a GUI session |
| `blender_effect_add_invoke_acceptance.py` | needs the interactive Add Effect search popup |
| `blender_particle_autogrowth_live_review.py` | profile-safe live review via tools/gate/launch_profile_safe_addon_review.py; leaves the GUI open |
| `blender_performance_playback_benchmark.py` | separate foreground handler/visibility evidence; the mandatory saved-demo budget runs via tools/benchmark_versions.py |
| `blender_phase1_source_ui_acceptance.py` | profile-safe interactive review via tools/gate/launch_profile_safe_addon_review.py |
| `blender_phase3_profile_safe_review.py` | profile-safe visible review via tools/gate/launch_profile_safe_addon_review.py |

### Helper

`_phase2_source_acceptance_common.py`: shared quantitative implementation used by the five source acceptance scripts above.

## `scripts/` (non-pytest)

Plain Python scripts run inside Blender (headless) or directly. `smoke_core.py` and `smoke_solver.py` are run by the production gate as plain scripts; the `headless_*` script is a manual surface acceptance run invoked with Blender.


| Script | Does | Needs |
|---|---|---|
| `headless_surface_resolution_pipeline.py` | Headless surfacing-resolution acceptance for Blender + Warp. | Blender/Warp; CUDA |
| `smoke_core.py` | Headless smoke test of core (non-solver) modules. | Blender/Warp; CUDA |
| `smoke_solver.py` | Headless smoke test: full solver pipeline on CUDA (dam break in a box). | Blender/Warp; - |


## Add-on's own tests: `fluidSim_warp_blender51/tests/` (not restructured)

58 test modules plus their own `conftest.py` ship beside the add-on source and run standalone (`cd fluidSim_warp_blender51; python -m pytest`, using the add-on's `pyproject.toml`, importlib mode) as well as from the root via `pytest.ini`. They cover the extension packaging and the add-on's internal layers: build/manifest/wheel handling and release identity (`test_build_extension`, `test_manifest_wheels`, `test_wheel_fetch`, `test_release_*`), the effects system (`test_effect_*`), cache schema/manager and attribute caches, presets, properties/UI source contracts, registration, Warp runtime and validator API, diagnostics, whitewater/surface-tension source, and capacity planning. They are mostly source-level and pure-Python and none imports `bpy`; `test_warp_runtime.py`, `test_diagnostics_report.py` and `test_runtime_integration_source.py` use the `cuda_device` fixture and skip without CUDA. Their `conftest.py` mirrors the root `cuda_device` fixture (`tests/repo/test_cuda_device_fixture.py` asserts the two agree). Keep their basenames distinct from every file in `tests/`.
