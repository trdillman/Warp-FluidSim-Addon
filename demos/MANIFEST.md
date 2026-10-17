# demos/ manifest

Shipped demo scenes. Paths here are hard-coded by tools, tests and the
performance-budget hash inventory, so the layout is intentionally flat; do not
move or rename anything without patching every consumer listed below.

Eight demos exist: `dam_break`, `crown_splash`, `paddle_stir`,
`chromatic_vortex`, the two rigid-coupling demos `buoyancy_ladder` and
`water_wheel`, and the feature demos `viscosity_race` and `deep_plunge`.
`.blend` bytes feed the production gate and the
performance inventory hash (`tools/benchmark_versions.py::_inventory`), so any
change to them is a real, gate-visible change.

The eight `.blend` files also ship in the extension:
`tools/release/build_extension.py` (`stage_demo_assets`) copies them into the
git-ignored `fluidSim_warp_blender51/resources/demos/` before each build, and
the in-add-on Demo Gallery (`blender/demo_gallery.py`) loads them from there in
installed builds. A new or renamed demo must also be added to `DEMO_IDS` in
`build_extension.py` and `tools/release/smoke_test_blender.py`.

| Path | Kind | Purpose | Read by | Written by |
|---|---|---|---|---|
| `dam_break.blend` | generated, committed | Rocky-cove dam break; baseline for gate, playback and performance budget | `tools/demos/check_demo_blends.py`, `tools/run_production_gate.py` (`FLUIDSIM_PLAYBACK_BLEND`), `tools/benchmark_versions.py`, `tools/benchmark_demo_trial.py`, `tools/profiling/profile_gpu_solver.py` (mirrors its domain), `tests/acceptance/blender_performance_playback_benchmark.py` | `tools/demos/build_demo_scenes.py` |
| `crown_splash.blend` | generated, committed | Drop-into-pool crown splash with whitewater | `check_demo_blends.py`, `benchmark_versions.py`, `tests/acceptance/blender_closed_domain_stability_acceptance.py` | `build_demo_scenes.py` |
| `paddle_stir.blend` | generated, committed | Keyframed paddle in a cylindrical tank (opt-in build: `build_demo_scenes.py -- paddle_stir`) | `check_demo_blends.py`, `benchmark_versions.py` | `build_demo_scenes.py` |
| `buoyancy_ladder.blend` | generated, committed | Four Dynamic collider blocks (density 250/500/750/1500) dropped into a still tank; they float in density order and the heaviest sinks (opt-in build: `build_demo_scenes.py -- buoyancy_ladder`) | `check_demo_blends.py` (`check_buoyancy_ladder`) | `build_demo_scenes.py` |
| `water_wheel.blend` | generated, committed | Six-paddle Dynamic collider wheel, locked to spin about its axle, turned by an inflow stream (opt-in build: `build_demo_scenes.py -- water_wheel`) | `check_demo_blends.py` (`check_water_wheel`) | `build_demo_scenes.py` |
| `viscosity_race.blend` | generated, committed | Three columns (water 0, syrup 0.004, tar 0.3 m^2/s per-source viscosity, From Particles on) collapse in walled lanes over a Friction 1 floor; the fronts run out in viscosity order (opt-in build: `build_demo_scenes.py -- viscosity_race`) | `check_demo_blends.py` (`check_viscosity_race`) | `build_demo_scenes.py` |
| `deep_plunge.blend` | generated, committed | A block dropped into a 70 cm tank with Narrow Band on; the builder also runs it full-band and requires 3x fewer particles at the same water level (measured 9.3x) (opt-in build: `build_demo_scenes.py -- deep_plunge`) | `check_demo_blends.py` (`check_deep_plunge`) | `build_demo_scenes.py` |
| `chromatic_vortex.blend` | authored/committed (no builder script) | Colour-mixing vortex demo | `check_demo_blends.py`, `benchmark_versions.py`, `tests/acceptance/blender_chromatic_vortex_containment_acceptance.py`, `tests/acceptance/blender_color_mixing_*_acceptance.py` | by hand in Blender |
| `dam_break/meta.json`, `crown_splash/meta.json`, `paddle_stir/meta.json`, `buoyancy_ladder/meta.json`, `water_wheel/meta.json`, `viscosity_race/meta.json`, `deep_plunge/meta.json` | generated, committed | Scene geometry/settings sidecar (domain, voxel size, particle counts) | `tools/demos/analyze_demos.py` (`load_meta`) | `build_demo_scenes.py` |

Not stored here: per-frame demo caches and fresh analysis output go to
`artifacts/demo_cache/` and `artifacts/demo_analysis/` (git-ignored, regenerated
by `tools/demos/build_demo_scenes.py` and `tools/demos/analyze_demos.py`; the
gate reads `artifacts/demo_analysis/analysis_report.json`). Preview PNGs that
`build_demo_scenes.py` renders to `demos/<name>_preview.png` are generated
leftovers and are no longer tracked.
