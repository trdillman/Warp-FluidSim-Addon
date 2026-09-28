# FluidSim 2.0 release: progress record

The working record for executing [NEXT_MAJOR_RELEASE_PLAN.md](NEXT_MAJOR_RELEASE_PLAN.md).
It keeps the current state and the next action so work can resume after a handoff.

**Outcome:** ship 1.1.0 (the bridge release), then prepare 2.0.0 through Phases 1–4.
Stop for **one owner review before tagging or releasing 2.0.0**, with a 2.1 handoff plan.

**Decisions taken** (the recommendations in plan §7):

- D1: Blender 5.2 LTS only.
- D2: Warp 1.17.x cu12 with a range pin.
- D3: bundle the hash-locked wheels, with a guided-installer fallback if size forbids it.
- D4: Linux is certified on WSL2 and labelled that way; otherwise it ships as "packaged, not certified".
- D5: 2.0 opens files saved by 1.1 or later.
- D6: ST-FLIP is Experimental, with Variable Projection off by default.
- D7: performance work goes to 2.1, with a ≤5% no-regression budget for 2.0.
- D8: no folder rename.
- D9: a required manual GPU gate with an attached manifest.

## Phase 0: bridge release 1.1.0

| Item | State | Evidence |
|---|---|---|
| REVIEW-006 (P0): private per-user effect module directory plus digest-verified exec | done | `tests/test_effect_source.py`, `tests/test_effect_compiler.py`; POSIX cases pass on WSL2 |
| REVIEW-021: Restart refuses to replace folders outside a FluidSim cache | done | `tests/test_artist_bake_safety.py`, `tests/test_cache_manager.py` |
| REVIEW-019: cubin integrity | won't-fix (same trust boundary as Warp's own cache) | card |
| Defect #5 regression: 35 fields read from the wrong settings group | done: detection, capture and revert read each field from its declaring group; the per-attribute export toggles (hidden since the split) draw again | `tests/test_pending_settings_registry.py`, `tests/test_operator_panel_behavior.py`; Blender 5.1 `PHASE4-LIVE-INTERACTION-PASS` |
| Root CI workflow (CPU, PRs + master) | done (`e6983d1`): `checks` job (ruff, pytest, Warp-CPU validators) and `package` job (build, install and smoke in Blender 5.1.0 Linux) | local WSL2 emulation of both jobs green; first hosted run is the 1.1.0 PR |
| Version 1.1.0, CHANGELOG catch-up | done (`cc68193`); determinism wording narrowed to what `d2a6b6d` proves | `test_release_identity.py` (changelog leads with the version; user guide agrees) |
| Close the 14 fixed + 7 obsolete cards; PRUNE-001 | done: 10 completed, 5 obsolete, 2 won't-fix (032, 050), PRUNE-001 closed. Kept open after re-verification: 005 (Fill Multiplier still silently clamped at 64, Phase 3) and 034 (Warp not declared in manifest, Phase 1 D3) | each card's `## Resolution` |
| Full pytest, GUI gate, demo analysis | done: hosted CI green on PR #10, on `master` and on the tag | CI runs 36341927576, 36342631940, 36343372072 |
| PR, merge, tag v1.1.0, GitHub release | done: PR #10 merged (`c5919d6`); release "FluidSim Warp 1.1.0 — bridge release" carries both platform zips, `SHA256SUMS.txt` and the gate manifest | GitHub release `v1.1.0` |

**Phase 0 is complete.**

## Phase 1: platform (branch `claude/2.0-phase1-platform`)

| Item | State | Evidence |
|---|---|---|
| D1: Blender 5.2 LTS only (manifest `blender_version_min` 5.2.0, `version.py`, user guide, notices) | done (`13929b3`) | `test_release_identity.py`, `tests/test_extension_manifest.py` |
| D2: Warp 1.17.x range pin: refuse below 1.17.0 or 2.x, warn on newer untested 1.x | done (`13929b3`) | `tests/test_warp_runtime.py`, `tests/test_operator_panel_behavior.py` |
| D3: bundle the checksum-locked Warp 1.17.0 wheels (REVIEW-034 closed) | done in source (`13929b3`, `39083a8`); real package build pending | `tests/test_manifest_wheels.py`, `tests/test_build_extension.py` |
| Geometry Nodes input setter fails loudly; `cuda_device` fixture; strict markers | done (merged `b8b0507`) | root 1170, add-on 459, `-m cuda` 115 passed on GPU |
| Release validators read the certified versions from `version.py` | done (`13929b3`) | `validate_*_warp.py` |
| Gate runs every lane on the certified Warp under Blender 5.2 (`PYTHONNOUSERSITE=1`, `--python-use-system-env`, refuses an uncertified Warp) | done (`0e01cdf`) | `tests/test_production_gate_semantics.py` |
| Full GUI gate, Blender 5.2.0 × Warp 1.17.0 × Windows 11, RTX 4080 SUPER | **FULL-PRODUCTION-GATE-PASS** at `0e01cdf`; demo analysis 18/18, blend checks, GUI acceptance 53/53, saved-demo playback | demo artifact sha256 `a32872b8…0926`, identical to the 1.x reference: the platform move changed no simulation output. Log `C:\Users\Tyler\fluidsim-gate-logs\gate-52-20260927-160102.log` |
| Ruff target py313 (Blender 5.2 bundles Python 3.13.13) | done | add-on ruff clean |
| Build the real split packages with the bundled Warp 1.17.0 wheels (D3) | done: `release_ready: true`; `windows_x64` 146.6 MB, `linux_x64` 159.1 MB | `C:\Users\Tyler\fluidsim-gate-logs\build-52-20260927-161112\` (`windows-build-report.json`, `SHA256SUMS.txt`) |
| Clean-profile install on Windows: extension smoke + installed GPU regression (21/21 validations, 10 preview frames) on Blender 5.2.0 with no PYTHONPATH | done | `smoke-test-report.json`, `gpu-regression-report.json` in the build folder |
| GPU self-validations leaked 3 native CUDA kernel leases, so disabling the add-on after Run Validations logged `unregister step 'native CUDA cache' failed` | fixed: 5 validations now release their solvers | `tests/test_validation_releases_native_leases.py` |
| Clean-profile install on WSL2 (D4) | done at release (see "Release" below) | |
| GUI gate reproducible 3/3 on the new stack | 3/3 on `a32872b8…0926`: passes at `0e01cdf`, `2a6eec0` and `bd67291`. Superseded by the Phase 3 rebind below | `gate-52-20260927-164658.log`, `gate-52-20260927-200134.log` |

## Phase 2: WS5 breaking cleanup

| Item | Status | Evidence |
|---|---|---|
| Retired snippet runtime (`CompiledSnippet`, `SnippetManager`, kernel templates), `blender/snippets.py` except the migration reader, the `_snippets()` shims, 13 unused enums | done (`d8f9d6e`, −583 lines) | presets compiled through the Custom Effects compiler (`tests/test_snippet_library.py`, deleted with the rest of the snippet code in slice D, `956c468`) |
| `k_ww_compact_scatter` unwritten `out_count` (REVIEW-036 closed) | done (`2a6eec0`) | `tests/test_whitewater_kernel_outputs.py`; gate demo hash unchanged |
| Gate truthfulness tests missed the `0e01cdf` certified-Warp stub | fixed (`55a4c6b`) | root 1191 passed |
| `write_sequence_metadata` | removed with cache v1 (slice C below) | |
| Dead snippet library UI (6 scene fields, enum builders, `user_snippet_dir`); `sim/custom.py` cut to the migration records + `legacy_snippet_to_effect` | done (`de038e7`, −1448 lines) | ownership map 233 → 227 fields; Blender 5.2 background acceptance pass |
| D6: ST-FLIP labelled Experimental; Variable Projection default off (property + `SolverSettings`) | done (`3ba90d5`) | `tests/test_stflip_experimental_defaults.py`; no demo enables ST-FLIP |
| Version 2.0.0 (manifest, `version.py`, docs); 2.0.0 CHANGELOG section opened | done (`3ba90d5`); `BUILD_CHANNEL` stays `beta` until owner review | `test_release_identity.py` |
| `setting_owner`/`_authoring_owner` divergence | already resolved by the split (`b6fe26c`): both route through `_scene_owned`; no change needed | `blender/pending_settings.py:259-286, 402-415` |
| 139 defaulted settings `getattr`s | kept (minimal test doubles depend on them); guarded instead: every one must name a declared field, so a removed/renamed field fails the suite rather than silently defaulting. Owner-review note: this deviates from the plan's "remove" | `tests/test_settings_getattr_names.py` (mutation-checked) |
| Settings drift (REVIEW-028/029/037/045/046/047 closed): one default per setting across RNA, `SolverSettings` and literal fallbacks; DISTANCE/VELOCITY units; FACTOR bounded; smooth-type saved numbers pinned (the card's renumbering would have flipped saved files) | done (`427a42f`) | `tests/test_settings_default_agreement.py`. 2.1 handoff: `viscosity_constant`, `ww_bubble_drag`, `ww_spray_drag` differ between RNA and `SolverSettings`; aligning them moves headless baselines |
| Can-only-fail UI: height-field `INFINITE` removed from the dropdown (FLAT/PROFILE keep stored 1/2; a saved Infinite reads back as "" and gets an actionable error) | done (`8b5c067`) | `test_height_back_mode_offers_only_runnable_modes_with_stable_numbers`, `test_height_source_rejects_removed_infinite_back_mode` |
| FIELD source mode | not a can-only-fail choice: hidden, and runs as a compiled field plan (pressure by default). Stale "not exposed" text corrected | `8b5c067` |
| Compiled FORCE plans | internal only: no UI path reaches them, and FORCE-role objects work through the analytic path. Left unchanged | `sim/source_regions.py:669` |
| D5 slice A: pre-1.1 settings migration removed (legacy union PropertyGroup, both `schema_version` stamps). A file that skipped 1.1 is recognized and flagged (load message + Setup/Physics alert: re-save in 1.1 first). `chromatic_vortex.blend` was pre-1.1 and was re-saved through 1.1's own migration code | done (`625b937`) | `tests/test_file_compat.py`, `tests/blender_pre_bridge_file_acceptance.py`; `check_demo_blends` refuses a flagged demo |
| D5 slice B: source schema < 4 compatibility (`source_compat.py`, `LEGACY_ABSOLUTE`) and legacy output adoption (`output_publication.py`) | **kept**. 1.1 itself still saves sources below schema 4 and adopts legacy outputs, so removing them would break files 1.1 saved. Owner-review deviation; 2.1 handoff: stamp the source schema on save | |
| D5 slice C: cache format 1 (`fluidsim_*.npz` + `fluidsim_sequence.json`) removed. 1.1 never writes it (tag `c5919d6`). A 1.0 cache is still detected, refused with "written by FluidSim 1.0 … Free the cache and bake again", and can be freed | done (`af42f80`) | `tests/test_cache_io.py`, `fluidSim_warp_blender51/tests/test_cache_manager.py`; root 1158 + add-on 458 passed |
| D5 slice D: Custom Warp Snippet RNA (`FLUIDSIM_PG_snippet*`, the `snippets` store), `fluidsim.snippet_apply`, `migrate_legacy_snippets` and its load handler, `blender/snippets.py`, `sim/custom.py` removed. 1.1 cannot create snippets and converts 1.0 snippets on load (marking the effect stack `migration_complete`, originals kept), so every 1.1-saved file already holds the Custom Effects copies. A stored snippet list without that mark is flagged as pre-1.1 and refused at admission; the raw list survives a 2.0 save so 1.1 can still recover it. Also fixed stale docs: user guide still called cache v1 readable (slice C), and CHANGELOG now carries the 1.0.x file/cache upgrade notes | done | `tests/test_file_compat.py`, `tests/blender_pre_bridge_file_acceptance.py` (1.0/1.1 snippet fixtures), admission acceptance and `check_demo_blends` pass on Blender 5.2; ownership map 226 → 225 fields; root 1155 + add-on 455 passed |
| REVIEW-009 (closed): Blender scripts that nothing ran. 13 `tests/blender_*.py` acceptance scripts were outside the gate (5 broken since 1.0.255); none of the 6 `tools/release/validate_*_blender.py` validators was run, and 5 of the 6 had broken against the 1.1 split (including `validate_phase3_blender.py`'s `missing Blender attribute fluidsim_velocity`). All fixed and gated (`RELEASE_BLENDER_VALIDATORS` lane); GUI-only probes are excused with a reason | done | `tests/test_production_gate_coverage.py`; all 6 release validators pass on Blender 5.2 through the gate's own `_run`; background acceptance batch green |
| Root cause of the silent breakage: `scene.fluidsim.voxel_size = x` set a plain Python attribute and never reached the solver. Both settings groups now raise `AttributeError` naming the right group. **Owner review: behaviour change for user scripts** | done | `tests/test_file_compat.py::test_writing_a_setting_to_the_wrong_group_raises` |
| Found by the validators: the Live Tuning shelf lost its Particle Display Fraction knob at the split (drawn from the domain, which lacks it); examples 01–04 shipped with pre-1.1 settings, so 2.0 flagged its own examples. **Owner review: both are user-visible fixes** | done: knob bound to its declaring group; examples regenerated with `generate_examples.py` | `test_viewport_shelf_is_lean_live_tuning_not_a_full_mirror`; `validate_examples_blender.py` now refuses a pre-1.1 example |
| REVIEW-044 (closed): one device switch, `FLUIDSIM_TEST_DEVICE` (default `cuda:0`), for pytest and every Blender/headless script | done | `tests/test_cuda_device_fixture.py` |

## Phase 3: harden

| Item | State | Evidence |
|---|---|---|
| REVIEW-004: ST Jacobi reported residual `-1.0` | done (`45c1cec`): measured against the variable-coefficient operator the solve used | card; `tests/` residual tests |
| REVIEW-005: six copies of the 64 particles-per-voxel cap | done (`45c1cec`): one shared constant, cap kept at 64 (RNA already limited authoring to 64). **Owner review:** cap not raised | card |
| REVIEW-008: module-import assertions | already fixed in `16e5090`; verified | card |
| REVIEW-011: capacity apply had no undo step | done (`45c1cec`): `{"REGISTER", "UNDO"}`; differs from the 032/050 won't-fix because it only writes RNA and the transform, which undo restores. **Owner review** | card |
| REVIEW-014: NaN/Inf MAC weights | done (`97eb132`): one `_sanitize_mac_face`; a non-finite weight zeroes weight and velocity | card; no demo output change |
| REVIEW-016: CUDA-graph fallback was silent | done (`97eb132`): new `stat_native_solver_message` readout (ownership map 225 → 226). **Owner review:** new user-visible field | card |
| REVIEW-017: collision SDF gradient read the 1e9 fallback on sparse edges | done (`97eb132`): central differences where both samples exist, one-sided otherwise | card; see the gate rebind below |
| REVIEW-022 and the **P1 it exposed**: a bake with an animated source could not play itself back (paddle_stir refused frames 2–5) or resume | done (`97eb132`): playback compares only the frame's `settings_signature`; bake signatures pinned to the start frame. **Owner review:** a moved source no longer invalidates a cache per frame; a per-frame source stamp is a 2.1 item | `tests/blender_animated_cache_playback_acceptance.py` (gated) |
| GUI gate on `97eb132` | **FULL-PRODUCTION-GATE-PASS**, new demo artifact sha256 `e49d23ef…e1ae` (1/3 on the new hash). **Owner review: hash rebind.** Cause isolated by rebuilding dam_break with only the REVIEW-017 files reverted to `45c1cec`: the old metrics came back. The change is dam_break whitewater ±2 particles of ~10k; no liquid metric moved, crown_splash is identical, REVIEW-014 has no demo effect | `gate-52-20260927-210531.log`; isolation logs in the session scratchpad (`build_revert017.log`, `analyze_revert017.log`) |
| REVIEW-041: an evaluated mesh leaked if reading its vertices raised | done: `to_mesh_clear()` in a `finally` | `tests/test_sources_object_points_release.py` |
| REVIEW-031: Live toggle has no undo | won't-fix, same policy as 032/050: Live can initialize the GPU solver, which undo cannot reverse; the operator already has a `poll()` | card |
| REVIEW-042: `read_sequence_metadata` swallows corrupt metadata | obsolete: no production caller since REVIEW-022 (playback reads the frame's `settings_signature`) | card |
| Linux relabelled uncertified (D4): `VALIDATED_RUNTIME_PLATFORMS` is now `("windows-x64",)`; the user guide, migration guide and CHANGELOG say Linux is packaged but not certified. **Owner review** | done | `test_release_identity.py`, `tests/test_release_1_0_253_integration.py` |
| Open cards | no P0/P1 open. 6 medium/low remain (013, 015, 024, 025, 027, 038), each with a suggested fix in the 2.1 handoff | `backlog/tasks/`, `docs/RELEASE_2_1_HANDOFF.md` |
| Performance budget (≤5% vs 1.0.255, D7) | measured. crown_splash −2.7%, dam_break +0.4%, chromatic_vortex −2.2% (vs 1.1.0; 1.0.255 cannot open the file). paddle_stir **+4.0%** on absolute medians, paired median +5.5% (n=8), so it sits on the line. The stage profiler puts the gap in P2G: 1.78 ms in 1.0.255, 3.42 ms in 1.1.0 (the deterministic gather, `d2a6b6d`), 3.50 ms in 2.0. 2.0 adds no measurable solver cost over 1.1.0, and Warp 1.17 is neutral or faster. **Owner review:** accept the 1.1.0 determinism cost against the 1.0.255 budget; a faster deterministic P2G is the first 2.1 perf item | `C:\Users\Tyler\fluidsim-gate-logs\perf-2.0-20260927\` (ABBA pairs, 4-way Warp/code split, stage profiles) |
| Defect #2 (compounding slowdown with viscosity + surface tension) | not reproduced on 2.0: 10 UI resets × 24 frames on dam_break with both enabled; median 106 → 96 ms, GPU memory fixed at 408 MB, datablocks and particles constant. Carried to 2.1 pending a GUI repro | `defect2.json` in the perf evidence folder |

## Phase 4: release

| Item | State | Evidence |
|---|---|---|
| Migration guide `fluidSim_warp_blender51/docs/migrating_to_2_0.md` (install, 1.0.x files, caches, snippets, scripts, ST-FLIP, results); linked from the user guide and CHANGELOG | done | |
| CHANGELOG 2.0.0: upgrade notes, Platform, Fixed, Changed, Known limitations (including perf and defect #2), Removed | done | `test_release_identity.py` |
| README, AGENTS.md, user guide say Blender 5.2 LTS / bundled Warp 1.17.0; support matrix shows Linux uncertified (certified at release, below) | done | |
| 2.1 handoff `docs/RELEASE_2_1_HANDOFF.md` | done | |
| Full pytest on Blender 5.2's Python + Warp 1.17.0 | root 1176 passed, 1 skipped; add-on 455 passed, 2 skipped (POSIX-only) | |
| GUI gate 3/3 on `e49d23ef…e1ae` | **3/3**: FULL-PRODUCTION-GATE-PASS at `97eb132` and twice at `1bbe907`; the logs differ only in timestamps and timings | `gate-52-20260927-225858.log`, `gate-52-20260927-230604.log` |
| Clean install on Linux (WSL2) | not run: needs owner approval (D4). Linux ships labelled uncertified | superseded by the release rows below |

## Release (owner approved 2026-09-28)

| Item | State | Evidence |
|---|---|---|
| Release build refused the retired `sim/custom.py` (collision guard wanted every stale owner present) | fixed (`5a70656`): the guard lists `retired_owners`; a retired owner that reappears fails the build | `tests/test_release_1_0_253_integration.py`, `test_build_extension.py` |
| `BUILD_CHANNEL` → `stable` | done (`7a27aab`); no consumer feeds cache identity, so the gate hash cannot move | `test_release_identity.py` |
| Linux package on WSL2 Ubuntu 24.04 + Blender 5.2.0 Linux: build, clean-profile install, smoke, installed GPU regression | pass (`release_ready: true`, `FLUIDSIM_GPU_REGRESSION_OK`). Needs `LD_LIBRARY_PATH=/usr/lib/wsl/lib`: Ubuntu's `libnvidia-compute-580` (from `nvidia-cuda-toolkit`) otherwise shadows the WSL driver and Warp finds no CUDA device | `~/fluidsim-2.0-dist` in WSL |
| Full production gate on WSL2 (`--include-gui`) | run 1: every headless lane passed (CUDA regressions, acceptance lanes, demo analysis 18/18, blend checks); GUI lane crashed in WSLg's Wayland EGL (Mesa zink, `EGL_BAD_MATCH`). Run 2 under X11 with `GALLIUM_DRIVER=d3d12` (renderer `D3D12 (NVIDIA GeForce RTX 4080 SUPER)`): every headless lane passed again, GUI acceptance **52/53**, `PRODUCTION-GATE-FAIL`. The failure: `viewport screenshot 2090x1201, 0/4000 visible samples`. The saved-demo GUI playback lane after it did not run | `C:\Users\Tyler\fluidsim-gate-logs\linux-wsl2-20260928\` |
| Screenshot failure isolated to Blender's window readback on Linux Mesa, not the add-on: a bare `screen.screenshot_area` probe (factory startup, no add-on) is all black on WSLg + d3d12, WSLg + llvmpipe, Xvfb + llvmpipe and Xvfb + d3d12 (backend OpenGL in each; the PNG RGB is truly zero, not a zero alpha). The same probe on Windows Blender 5.2 sees 3997/4000 visible samples. The gate check was not weakened | Linux not certified | probe `shot_probe.py`, `pngstat.py` in the evidence folder |
| D4 rule applied: Linux ships "packaged, not certified"; `VALIDATED_RUNTIME_PLATFORMS` stays `("windows-x64",)`. User guide and CHANGELOG state the WSL2 result and carry the WSL2 driver/display notes | done | `test_release_identity.py` |
| Observation: on Linux, `crown_splash.blend` and `dam_break.blend` bytes differed between the two gate runs (`chromatic_vortex` and `paddle_stir` matched). Not investigated; no Linux manifest exists to compare the demo report hash. 2.1 item | open | sha256s in the evidence folder |

## Owner review (before tagging 2.0.0)

**Approved by the owner on 2026-09-28** ("approved, release 2.0 and run the WSL2 Linux test"). Item 6 was resolved (`BUILD_CHANNEL` flipped to `stable`). Item 5: the WSL2 test ran but the gate did not pass there (see "Release"), so Linux stays uncertified.

Decisions this work made that the owner should confirm, most consequential first:

1. **Performance budget:** paddle_stir is +4% vs 1.0.255, from 1.1.0's deterministic P2G (Phase 3 row). Accept, or hold 2.0 for a faster gather.
2. **Demo hash rebind** to `e49d23ef…e1ae` (REVIEW-017 whitewater normal; dam_break whitewater ±2 of ~10k).
3. **Script behaviour change:** writing a setting to the wrong group raises `AttributeError`.
4. **Cache policy:** playback compares only the settings signature; bake signatures are pinned to the start frame; a moved source no longer invalidates a cache (per-frame source stamp is a 2.1 item).
5. **Linux uncertified:** `VALIDATED_RUNTIME_PLATFORMS = ("windows-x64",)`; WSL2 install and gate need approval to run.
6. **`BUILD_CHANNEL`** is still `beta`; flip to `stable` when releasing.
7. User-visible fixes: Live Tuning knob rebound; examples 01–04 regenerated; new `stat_native_solver_message`; garbled UI text corrected.
8. Plan deviations: 139 settings `getattr`s kept (guarded, not removed); D5 slice B (source schema < 4, legacy output adoption) kept because 1.1 still writes it; sink schema-0 `geometry_mode` kept.
9. Kept as-is: ppv cap stays 64 (REVIEW-005); RNA/solver default divergences for `viscosity_constant`, `ww_bubble_drag`, `ww_spray_drag` (2.1 item).
10. Capacity apply is now undoable (REVIEW-011); REVIEW-031 Live toggle stays non-undoable (won't-fix).
11. chromatic_vortex's perf baseline is 1.1.0 on the 1.1-re-saved file.

## Environment notes

- Blender 5.2.0 LTS (portable, Python 3.13) is at `C:\tmp\blender-cu-vm\staging\blender-5.2\blender-5.2.0-windows-x64`. Its Python shares the user site-packages where Warp 1.16.0 is installed, so every run sets `PYTHONNOUSERSITE=1`.
- Certified Warp 1.17.0 plus pytest live outside the user site: `PYTHONPATH=C:\Users\Tyler\fluidsim-warp\site-1.17.0;C:\Users\Tyler\fluidsim-warp\devsite`. The gate needs `BLENDER_EXE`/`BLENDER_PY` pointed at the portable 5.2.0 build.
- WSL2 Ubuntu 24.04 has Python 3.12 and pytest, and the RTX 4080 SUPER is visible through `nvidia-smi` (driver 610.47). Blender 5.2.0 Linux is at `~/fluidsim-ci-tools/blender-5.2.0-linux-x64`, a clean clone at `~/fluidsim-2.0`, Warp 1.17.0 in `~/fluidsim-2.0-site`, pytest in `~/fluidsim-2.0-devsite`. Every GPU run needs `LD_LIBRARY_PATH=/usr/lib/wsl/lib`; GUI runs also need `WAYLAND_DISPLAY=` and `GALLIUM_DRIVER=d3d12`. The Ubuntu CUDA 12.0 packages were left installed.
- Disk is tight: about 22 GB free.
- 1.x needs Warp 1.16 (user site, `PYTHONNOUSERSITE` unset, empty `PYTHONPATH`); 2.0 needs 1.17. Blender-loaded worktrees must sit on short paths (`C:\Users\Tyler\...`) because of the 260-character limit.
- Whole-frame timing on this workstation varies 5–30% run to run; use ABBA pairs and the stage profiler (`profiling_enabled`/`profiling_sync` on the domain) for perf questions.

## Next action

Rebuild the release packages at the final commit, push the branch, open the PR to master, merge once CI is green, then tag `v2.0.0` and publish the GitHub release (both zips, `SHA256SUMS.txt`, `artifacts/production_gate_manifest.json`). 2.1 starts from `docs/RELEASE_2_1_HANDOFF.md`.
