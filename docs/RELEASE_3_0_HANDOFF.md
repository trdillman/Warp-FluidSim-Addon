# FluidSim Warp 3.0: handoff

This is the starting point for whoever finishes and releases 3.0. It records what
the merged lanes shipped, the evidence known so far, and what is still open. Items
marked `TBD (orchestrator)` are not known to the documentation lane and must be
filled in from the gate and benchmark runs. The per-lane record is
`docs/RELEASE_3_0_STATUS.md`; the plan is `docs/RELEASE_3_0_PLAN.md`.

## Baseline 3.0 hands over

| Item | State at handoff |
|---|---|
| Platform | Blender 5.2 LTS, bundled Warp 1.17.0 (cu12), as in 2.x |
| Demo reference hash | master and every merge through lane F before lane G: `2cb4f781…3a73`. After lane G (grid snapping): `8baa886c…baf3`, changed by design. Lane F's merge has the same hash as G. |
| Tests | 2413 passed, 3 skipped, 2 xfailed at lane F's merge `cfff382` (before the documentation lane) |
| GUI gate references | TBD (orchestrator) |
| Performance (ABBA) | TBD (orchestrator) |
| Open P0/P1 | TBD (orchestrator) |

## Scope shipped, by lane

| Lane | What it shipped | Merge commit | Tests at merge | Demo hash |
|---|---|---|---|---|
| D | ST-FLIP removed (solver, 14 settings, panel, Variable Projection) | `6ce6b33` | 2053 passed, 3 skipped | unchanged |
| E | Hygiene | `dfb2d91` | | |
| B | Cache format 3, checkpoint format 2, per-frame source stamp | `57c6540` | 2271 passed, 3 skipped (E and B) | |
| B2 | Cache v3 consumers (playback, resume, Alembic, headless exit 14) | `fcc6357` | 2307 passed, 3 skipped (B2 and C) | |
| C | Multigrid V-cycle in PCG (opt-in) | `881b0ab` | 2307 passed, 3 skipped (B2 and C) | unchanged |
| A | Domain/scene settings split, defaults unified | `cc21cc7`, fixups `582ccbc` | 2313 passed, 3 skipped | settings dump identical |
| A2 | Preconditioner setting wired (RNA, `SolverSettings`, migration) | `6b0abbd` | 2357 passed, 3 skipped | unchanged |
| I | VDB writer spike (internal, not wired) | `aae0607` | 2364 passed, 3 skipped | |
| H | Alembic surface export, Playback-only mode | `2eb067e`, gate entry `b54f1e2` | 2374 passed, 3 skipped, 1 xfailed | |
| G | Resolution Divisions, float32 ceil fix | `b4c9dc5` | 2388 passed, 3 skipped, 1 xfailed | `8baa886c…baf3` |
| F | APIC transfer (own kernels) | `cfff382` | 2413 passed, 3 skipped, 2 xfailed | equal to G |
| K | Documentation (this lane): changelog, migration guide, user guide, technical report | TBD (orchestrator) | | |

Facts that explain results:

- Lane G changed the demo hash on purpose: crown_splash's grid went from
  88 × 88 × 56 to 80 × 80 × 56, and dam_break's voxel size from 0.03 to 0.02995
  (187 divisions).
- Lane F is PARTIAL. The first head changed FLIP bits, so APIC got its own kernels.
  The 0.98 angular-momentum gate is a strict xfail (APIC 0.939, FLIP 0.940, PIC
  0.596); the enforced gate is APIC > 0.90 and > PIC + 0.25.
- Lane H is PARTIAL. The strict xfail `test_playback_signature_check_does_not_import_warp`
  marks the open Warp-free signature work (see section 2).
- Lane I's VDB writer (`io/vdb_writer.py`) is not connected to any operator.

## 1. Carry-over work (must do)

1. **GUI gate.** One full-production-gate pass on the release commit whose demo
   hash matches the recorded reference (scoped validation, `docs/RELEASING.md`).
   TBD (orchestrator)
2. **ABBA performance run** against 2.1.x. TBD (orchestrator)
3. **Beta re-baseline (lane M, `3.0/lane-m-defaults`).** APIC is now the default
   Transfer for new domains; domains made before this change are pinned to FLIP and
   JACOBI on load, File > Append and the Demo Gallery (`fluidsim_defaults_epoch`
   object stamp, `file_compat.migrate_domain_defaults`). **Multigrid stays opt-in:**
   on dam_break (frames 2-48, tol 1e-5, APIC) it needs 5 against 50 median PCG
   iterations (10x fewer) but ~4.8 ms against ~1.9-2.4 ms per solve (about 2.5x
   slower), missing the spec's 1.5x-faster gate. **Narrow band stays off in
   dam_break:** it was 1.4x slower per frame and grew to the 600k particle cap
   (256k full-band). Both are owner decisions. New demo hash
   `c7542ea19d7d7df6db97051f9a05760c9f5514bd01f7ee3bfabea88678f796a4` (2 runs, 21/21
   demo checks). The committed `demos/*.blend` were not re-saved: they read as
   pre-3.0 (FLIP/JACOBI) until the owner re-saves them from the builder.
4. **Lane J (narrow band).** Merged as `2952a95` (opt-in, off by default; results
   with it off are bitwise unchanged and the demo hash stays `8baa886c…baf3`).
   2471 passed, 3 skipped, 1 xfailed at the merge. The settings signature is
   re-pinned to `f812a312…f606` (two new settings); the ownership map has 217
   rows. With narrow band on, frames carry `liquid_sdf_*` data; the VDB writer
   is still not wired to it. The Liquid voxels readout shows only in
   narrow-band mode (`0d39abc`). CHANGELOG lines under Added, Upgrade notes and
   Known limitations.
5. **Lane L (Warp-free cache signature).** Merged as `90d13ba`. `SolverSettings`
   lives in `sim/solver_settings_model.py` (re-exported from `sim/solver.py`); the
   settings signature is pinned to `e2a87bf9…8d38` and unchanged; the strict xfail is
   gone. CHANGELOG line under Fixed.
6. **Release steps.** Version bump to 3.0.0 (alpha channel) done in `800b0ca`
   (`version.py` and the manifest). Archive, tag and publication: TBD (orchestrator)

## 2. Open items from the lanes

- POINTCLOUD Alembic export writes only an empty transform on Blender 5.2, so
  particle export is out of scope.
- Restoring a scene that started with live on during an Alembic export is
  untested: it needs CUDA. Cover it in the GUI gate run.
- Unguarded unregister loops and the TIME_OFFSET question remain from the status
  record (see `docs/RELEASE_3_0_STATUS.md`).
- Blender runs owed at the alpha.1 gate: see the status record.

## 3. Process notes

- The documentation lane changed no code. Its claims were checked against the
  code on the merged base, not by running Blender.
- The "49 authored surfaces, 51 View3D mirrors" count in the technical report is
  1.0.255 history. The validator no longer pins it, and the 3.0 panel set differs
  (lane A moved the domain panels to Physics mirrors). The current count can only
  be read from a Blender run of `tools/release/validate_properties_ui_blender.py`.
- Time Offset colour mode reads a channel only ST-FLIP filled; it now always shows
  zero. Decide whether to hide it.
