# tools/ manifest

Developer tooling; nothing here ships in the extension. Everything is
authored (no generated files) except where noted. `tools` is an importable
package (`from tools.demos import dam_break_cove`); scripts also run directly
(`python tools/<sub>/<name>.py`) and put the repo root on `sys.path` themselves.

## Layout

Kept at `tools/` root on purpose: these form the hash-bound performance-budget
and gate core. `benchmark_versions._inventory` hashes these exact paths and
`_snapshot_candidate` copies them by name, and `run_production_gate.py` is
launched by path from outside the repo (`wt_gate.ps1`, README/AGENTS/CI notes).

| File | Purpose | Called / read by |
|---|---|---|
| `run_production_gate.py` | Production verification gate (`--include-gui` for the full GUI gate) | CLI, `wt_gate.ps1`, `tests/gate/test_production_gate_*.py`, `tests/gate/test_demo_budget_runner.py`, ci.yml (comment) |
| `benchmark_versions.py` | Cross-version dam_break benchmark and saved-demo budget runner (`--demo-budget`) | gate, `tests/perf/test_benchmark_versions.py`, `tests/gate/test_demo_budget_runner.py` |
| `benchmark_demo_trial.py` | Fresh-Blender child process for one saved-demo budget trial | `benchmark_versions.py`, gate |
| `performance_budget.py` | Pure budget evaluation / ABBA schedule | `benchmark_versions.py`, gate, `tests/gate/test_performance_budget.py` |
| `repo_addon_loader.py` | Enable this checkout's add-on inside Blender | `demos/build_demo_scenes.py`, `demos/check_demo_blends.py`, `benchmark_demo_trial.py`, ~20 `tests/acceptance/blender_*` scripts |
| `ownership_map.py` | Derives `docs/ownership-map.md` (`--check` verifies it is current). Left here because it embeds the path `tools/ownership_map.py` in the generated doc, which `tests/test_ownership_map.py` compares byte for byte | `tests/test_ownership_map.py` |

### `demos/` (demo scenes: build, analyze, verify)
| File | Purpose | Called by |
|---|---|---|
| `build_demo_scenes.py` | Builds `demos/*.blend`, `demos/<scene>/meta.json`, `artifacts/demo_cache` (Blender) | gate, `tests/acceptance/blender_capacity_planner_acceptance.py`, `tests/acceptance/blender_simulation_admission_acceptance.py` |
| `analyze_demos.py` | Offline physics checks on the demo caches -> `artifacts/demo_analysis/` (plain Python) | gate, `tests/demos/test_demo_analysis_contract.py` |
| `check_demo_blends.py` | Executable all-demo gate: every tracked `demos/*.blend` reopens, resets, simulates (Blender) | gate |
| `dam_break_cove.py` | Shared rocky-cove geometry for the dam_break builder/analyzer/checker | the three above, `tests/demos/test_dam_break_cove.py` |

### `profiling/` (measurement tools, run by hand unless noted)
| File | Purpose | Called by |
|---|---|---|
| `profile_gpu_solver.py` | Headless solver profiler (GPU smoke in the gate) | gate, `tests/perf/test_profiler_phase_contract.py`, `profile_kernel_breakdown.py`, `profile_host_breakdown.py`, `check_launch_identity.py` |
| `profile_user_path.py` | Blender user-path (viewport playback) profiler | gate, `tests/perf/test_profile_user_path_contract.py`, `tests/perf/test_performance_hotpath_contracts.py` |
| `profile_kernel_breakdown.py` | Per-kernel GPU vs wall time | manual |
| `profile_host_breakdown.py` | cProfile of host side of `step_frame` | manual |
| `check_launch_identity.py` | Proves a solver change leaves a bake bitwise identical (plain vs cached-launch SHA-256) | manual |
| `profile_source_pipeline.py` | Source-compiler profile (`SOURCE-PIPELINE-PROFILE-PASS`), cited in AGENTS.md | manual |

### `benchmarks/` (paired CUDA campaigns; each has its own test)
| File | Test |
|---|---|
| `benchmark_liquid_compaction.py` | `tests/perf/test_liquid_compaction_benchmark.py` |
| `benchmark_whitewater_finalization.py` | `tests/perf/test_whitewater_finalization_benchmark.py` |
| `compare_playback_performance_trials.py` | `tests/perf/test_compare_playback_performance_trials.py` |

### `gate/` (gate validators and review launcher)
| File | Purpose | Called by |
|---|---|---|
| `validate_gui_playback_report.py` | Validates the GUI playback report JSON | gate, `tests/gate/test_gate_marker_and_playback_validation.py`, `tests/gate/test_production_gate_truthfulness.py` |
| `validate_source_acceptance_report.py` | Validates source-acceptance reports | `tests/acceptance/_phase2_source_acceptance_common.py`, `tests/gate/test_source_acceptance_report.py` |
| `launch_profile_safe_addon_review.py` | Owned, profile-safe interactive Blender add-on review session | `tests/perf/test_profile_safe_addon_review.py`, gate (messages), `tests/acceptance/blender_phase1_source_ui_acceptance.py` |

### `release/` (packaging and release validation; unchanged location)
Tightly coupled to `fluidSim_warp_blender51/tests/test_release_*`, ci.yml and
`blender_manifest.toml`, so left in place.

| File | Purpose | Called by |
|---|---|---|
| `build_extension.py` | Verify / build / smoke the Blender extension zip | ci.yml, many tests, gate |
| `fetch_warp_wheels.py`, `wheels.lock.json` | Fetch / checksum-lock bundled Warp wheels (lock is JSON data, authored) | `build_extension.py`, ci.yml cache key, `blender_manifest.toml` comments |
| `install_local.py` | Install a built extension into local Blender and prove what loads | `tests/repo/test_install_local.py`, manual |
| `smoke_test_blender.py` | Installed-package lifecycle / Custom Effects / Warp smoke | `build_extension.py` |
| `gpu_regression_blender.py` | Full NVIDIA GPU validation + 10-frame preview | `build_extension.py` |
| `generate_examples.py`, `generate_geometry_nodes.py`, `generate_custom_effect_demos.py`, `custom_effect_demo_runtime.py`, `tool_loading.py` | Generators for bundled example/asset `.blend` files and their shared helpers | add-on asset tests, smoke test |
| `validate_{phase3,phase4,phase5,custom_effects}_{warp,blender}.py`, `validate_examples_blender.py`, `validate_properties_ui_blender.py` | Release-only validators run on CPU Warp in ci.yml and on CUDA/Blender in the gate | ci.yml, gate (`RELEASE_WARP_VALIDATORS`, `RELEASE_BLENDER_VALIDATORS`), smoke test |
| `candidate_collision_guard_1_0_253.json`, `candidate_integration_1_0_255.json` | Historical release-candidate records (authored data) | both: `tests/repo/test_release_1_0_253_integration.py`; `collision_guard` also `build_extension.py` and `fluidSim_warp_blender51/tests/test_build_extension.py` |
