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
| Build and install the real split packages; clean-profile install on Windows and WSL2 (D4) | pending | |
| GUI gate reproducible 3/3 on the new stack | pending (1/3) | |

## Phases 2–4

Not started. Phase 2 (WS5 breaking cleanup) starts with an inventory of the removal targets.

## Environment notes

- Blender 5.2.0 LTS (portable, Python 3.13) is at `C:\tmp\blender-cu-vm\staging\blender-5.2\blender-5.2.0-windows-x64`. Its Python shares the user site-packages where Warp 1.16.0 is installed, so every run sets `PYTHONNOUSERSITE=1`.
- Certified Warp 1.17.0 plus pytest live outside the user site: `PYTHONPATH=C:\Users\Tyler\fluidsim-warp\site-1.17.0;C:\Users\Tyler\fluidsim-warp\devsite`. The gate needs `BLENDER_EXE`/`BLENDER_PY` pointed at the portable 5.2.0 build.
- WSL2 Ubuntu 24.04 has Python 3.12 and pytest, and the RTX 4080 SUPER is visible through `nvidia-smi` (driver 610.47).
- Disk is tight: about 22 GB free.

## Next action

Build the real split packages with `tools/release/build_extension.py --blender <5.2.0>` and smoke-install them in a clean profile; then start Phase 2.
