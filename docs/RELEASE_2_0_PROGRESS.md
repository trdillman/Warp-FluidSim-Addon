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
| Close the 14 fixed + 7 obsolete cards; PRUNE-001 | PRUNE-001 done (archive re-hashed 25/25); cards in progress | `backlog/tasks/` |
| Full pytest, GUI gate, demo analysis | pending | |
| PR, merge, tag v1.1.0, GitHub release | pending | |

## Phases 1–4

Not started.

## Environment notes

- Blender 5.2 is not installed yet (needed for Phase 1).
- WSL2 Ubuntu 24.04 has Python 3.12 and pytest, and the RTX 4080 SUPER is visible through `nvidia-smi` (driver 610.47).
- Disk is tight: about 22 GB free.

## Next action

Commit the closed cards, push, open the 1.1.0 PR and get the first hosted CI run green.
