# Release 3.0 status

The task record for executing `docs/RELEASE_3_0_PLAN.md`. The orchestrator updates it at
each dispatch, merge and gate. Newest state wins; history lives in git.

## Bases and baselines

| Ref | Commit | Suite | Demo artifact sha256 |
|---|---|---|---|
| master | `cc36403` | 2079 passed, 3 skipped | `2cb4f781…3a73` (GUI gate reference) |
| release/3.0 base: master + merge of release/2.1.0 | `94c79eb` | 2104 passed, 3 skipped | `2cb4f781…3a73` (headless, equals master) |
| `archive/st-flip-final` tag (local) | `94c79eb` | | |
| release/3.0 after lane D merge | `6ce6b33` | 2053 passed, 3 skipped (on D head `34528aa`; merge adds docs only) | `2cb4f781…3a73` (headless, equals master) |
| release/3.0 after lanes E and B merge | `57c6540` | 2271 passed, 3 skipped (= 2053 + E 132 + B 86) | `2cb4f781…3a73` (headless, equals master) |
| release/3.0 after lanes B2 and C merge | `881b0ab` | 2307 passed, 3 skipped | `2cb4f781…3a73` (headless, equals master) |
| release/3.0 after lane A merge + fixups | `582ccbc` | 2313 passed, 3 skipped | `2cb4f781…3a73` (headless, equals master) |
| release/3.0 after lane A2 merge | `6b0abbd` | 2357 passed, 3 skipped | `2cb4f781…3a73` (headless, equals master) |
| release/3.0 after lane I merge | `aae0607` | 2364 passed, 3 skipped | not recomputed (no `sim/` or `properties.py` change) |
| release/3.0 after lane H merge + gate entry | `b54f1e2` | 2374 passed, 3 skipped, 1 xfailed | not recomputed (no `sim/` or `properties.py` change) |
| release/3.0 after lane G merge | `b4c9dc5` | 2388 passed, 3 skipped, 1 xfailed | **`8baa886c…baf3`** (2/2 headless). Changed by G by design; all 21 demo checks pass |
| release/3.0 after lane F merge | `cfff382` | 2413 passed, 3 skipped, 2 xfailed | `8baa886c…baf3`, equal to G, so APIC leaves FLIP byte-identical |
| release/3.0 after lane L merge | `90d13ba` | 2416 passed, 3 skipped, 1 xfailed (H's xfail now passes, +2 pin tests) | `8baa886c…baf3` (unchanged, as a pure move must be) |
| release/3.0 after lane K merge + 3.0.0 alpha bump | `21bc7c4` | 2416 passed, 3 skipped, 1 xfailed | `8baa886c…baf3` on `800b0ca` (the build label is not in the artifact) |
| release/3.0 after lane J merge | `2952a95` | 2471 passed, 3 skipped, 1 xfailed | `8baa886c…baf3` on J's head `5bb8d6f` (narrow band off is bitwise unchanged); merged tree: see the next row |
| release/3.0 after the zero-size-domain fix | `6cf4b24` | 2472 passed, 3 skipped, 1 xfailed | `8baa886c…baf3` (merged tree, headless) |

- **Gate run 1 (on `6206978`) failed** in `validate_phase3_blender.py`. The
  `phase3_points_f12.png` render was uniform (std 0.0016). Bisect: pass at `b54f1e2`, fail
  from G's merge `b4c9dc5`. Cause: G derives `voxel_size` from the domain's bounding box,
  and the probe's domain is an empty mesh, so the derived voxel size, and with it the point
  radius, was 0. Fixed in `6cf4b24` (a zero-size domain falls back to the default box);
  the probe passes and the demo hash is unchanged. The gate restarts from run 1.
- **Gate retry on `b873f10` failed** in `validate_properties_ui_blender.py`, which still
  expected the Voxel Size control G replaced. `d04adf5` expects Resolution Divisions.
- **Gate retry on `d04adf5` failed** in `blender_memory_safety_gate.py`. The capacity
  planner proposed off-lattice voxel sizes; the divisions setter rounded them (sometimes
  finer than costed) and the strict read-back refused the option. Lane G2
  (`3.0/lane-g2-capacity`, `C:\Users\Tyler\wt\r30-g2`) fixed it as `1c57ffe`, merged by
  fast-forward. Suggestions snap coarser onto whole divisions; the apply path resizes the
  domain before writing settings and restores divisions exactly on rollback.
  `MEMORY-SAFETY-GATE-PASS` and `CAPACITY-PLANNER-ACCEPTANCE-PASS` on 5.2.
  Suite 2479 passed, 3 skipped, 1 xfailed; demo hash `8baa886c…baf3` unchanged.
- **Gate retry on `1f90bfd`:** Blender exited -1 at the start of the phase3 validator with
  no output (the validator then passed alone, so treated as a one-off crash). The next
  retry failed in `blender_demo_gallery_acceptance.py`: appended demos came in at 64
  divisions (dam_break 0.0875 instead of 0.03), because G's migration runs on
  `load_post` only. The gallery now migrates the appended scene;
  `DEMO-GALLERY-ACCEPTANCE-PASS`, suite 2480 passed, 3 skipped, 1 xfailed.
  Appending a pre-3.0 domain by hand (File > Append) still skips the migration and is
  an open item.
- **Gate retry on `cc8d23d`:** every Blender check passed; the performance budget came back
  `invalid` because one 1.0.255 *baseline* trial (chromatic_vortex b1 s2) exited -1 1.6 s
  after reading its blend. The run on `fb530b6` then died the same way in the demo gallery
  acceptance. **Cause (not this repo):** the astrophysical-warp project's
  `tools/aw-window-walk/launch-blender.ps1` / `run-probe.ps1` run
  `Get-Process blender | Stop-Process -Force`, killing every Blender on the machine
  (exit -1 = .NET `Process.Kill`; each kill left an orphan `blender_<pid>` temp dir and no
  crash file; its session ran the kill at 11:36–11:37, one minute before the first crash).
  Flagged as a separate task. `wt_gate.ps1 -ExeName blender_fsgate.exe` now runs a
  hardlink of the same Blender under another process name. That run (`fcfb91a`, run2c)
  passed every Blender check, but the budget was `invalid`: a candidate trial could not
  allocate 3 MB of host memory while an unrelated `graft build` (node) held 30+ GB of
  commit (5.4 of 119 GB free). **Gate blocked until the machine is quiet.**
  Rebuilt blend bytes differ on every gate run (compressed; already seen on Linux in
  2.1), so blends are restored to HEAD before each run and the qualifying run's blends
  get committed (`docs/QUALIFIED_DEMO_INPUTS.md` note, `f851bfd`). The gate's rebuilt `crown_splash`/`dam_break` blends were committed as
  `fb530b6` (crown_splash build whitewater peak 5346 → 5529); demo hash unchanged.
- **Gate run2d (`ebaf9ef`, after graft closed):** the whole gate process tree vanished at
  14:25:50 during the paddle_stir budget trials (no exit line, summary `incomplete`, no
  kill found in other sessions). Suspected: the launching tool shell's job object. Gates
  now launch through WMI (`orch\scratch\launch_gate_wmi.ps1`), outside any tool shell.
- **Budget once per commit (user decision 2026-10-09).** The 3 consecutive passes now
  measure the performance budget on the first run only; runs 2 and 3 pass
  `--reuse-budget`, which reuses that summary only for the same HEAD with unmodified
  add-on/benchmark sources and records the reuse in the manifest. Saves ~10 min per run.
- **Gate run2e (`ebaf9ef`):** every Blender check passed; the budget, reached cleanly for
  the first time, was `invalid` on comparison contract, not speed: 3.0's new settings
  (narrow band, preconditioner) and removed ST-FLIP settings were unknown to the
  comparator, and Resolution Divisions snaps crown_splash/dam_break grids (crown_splash
  88x88x56 -> 80x80x56 padded, dam_break 188 -> 187 voxels on x). Owner decision
  2026-10-09: compare per frame, allowing only a one-voxel lattice snap and recording it
  (`latticeSnap`); the new/removed settings are neutral only while their switches are off.
  Replaying run2e's 64 trials: pass, host ratio crown 0.65 (cell ratio 0.83), dam 0.60,
  paddle 0.81, chromatic 0.91.
- **Gate speed-ups:** parallel Python checks (64 s wall for 10 steps), one batched Blender
  session for the 33 background acceptance scripts with standalone rerun of any non-pass
  (alembic export reruns: it expects Warp not yet loaded), per-step timing and a
  `--quick` tier (QUICK-GATE-PASS in 351 s on the gate branch).
- **3x GUI gate on `5a2ce80`: 3/3 FULL-PRODUCTION-GATE-PASS (2026-10-09).** Run3a measured
  the budget (PERFORMANCE-BUDGET-PASS; host ratio crown 0.66, dam 0.71, paddle 0.71,
  chromatic 0.95; 1690 s total, budget 1127 s); run3b/run3c reused it (553 s, 543 s).
  Demo analysis report sha256 `8baa886c018245f3ea9187933b7d4108891a79998de861da24578e113efbbaf3`
  reproduced by all three: the 3.0 reference. Run3c's rebuilt crown_splash/dam_break
  blends are committed with this note (compressed-byte drift only).
- **Scoped validation (owner decision 2026-10-09).** The three-consecutive-passes rule
  is retired: one `--include-gui` pass whose demo hash matches the reference is release
  evidence. The gate now reuses a passing budget by default whenever no budget source
  (add-on minus docs/Markdown, loader, benchmark tools, budget evaluator) changed since the
  measured commit, so docs/blends/test commits skip the ~19 min budget; `--measure-budget`
  forces it. Localized changes get focused tests or `--quick`. The budget measured on
  `5a2ce80` covers `fad2e6e` and this commit.

- Demo artifact = `artifacts/demo_analysis/analysis_report.json`, the file the production
  gate hashes. The orchestrator recomputes it headless after every merge that touches
  `sim/` or `properties.py`.
- Refactor merges must keep it identical.
- The headless recompute is `C:\Users\Tyler\wt\demo_hash.ps1` in the throwaway worktree
  `C:\Users\Tyler\wt\r30-hash`. On `94c79eb` it reproduced the GUI gate reference exactly,
  so the headless hash stands in for the gate between milestones.

## Deviations from the plan

1. **The branch base is not a shipped 2.1.1.**
   - The plan branches `release/3.0` off master once 2.1.1 ships.
   - Shipping is an external action, and the owner's 2.1.1 version bump is still
     uncommitted in the main checkout.
   - So `release/3.0` is master plus a merge of `release/2.1.0` at `cd39e78`, which
     carries the plan, the Demo Gallery and the kernel prewarm.
   - The 2.1.1 bump will merge in when it is committed.
2. **Lane worktrees are created by hand** from `release/3.0` under `C:\Users\Tyler\wt\`,
   not with the Agent tool's worktree isolation.
   - That isolation would base each lane on the main checkout's branch, `release/2.1.0`.
3. **Wave 0 runs in two steps.**
   - D (ST-FLIP removal) shares `properties.py`, `pending_settings.py`, `ui_scene.py`,
     `solver_settings.py` and `pressure.py` with A and C.
   - So D, B and E go first. A and C start once D is merged.
4. **Lane E's Linux offscreen check** goes into
   `tests/acceptance/blender_gui_acceptance.py` and a new helper module, not
   `tools/run_production_gate.py`.
   - That script is what the gate runs for the viewport check.
5. **Lane E takes REVIEW-024 and REVIEW-025.** The plan's lane table gives them to E;
   one bullet suggests doing them with the split instead. E lands before A starts, so
   the split rebases on top.
6. **`SPEC-settings-split-completion` is partly stale.** Checked on `94c79eb`:
   - `FLUIDSIM_PG_scene` (76 fields) and `FLUIDSIM_PG_domain` (150) already share no
     field. 2.0 (`625b937`) narrowed the scene group and dropped the pre-1.1 migration.
   - So Phase 4 step 2 is done, and step 1 (a 3.0 migration operator) has nothing to
     migrate. Lane A does not build it.
   - Physics-tab domain panels are generated from the Scene-tab panel classes, so
     "delete the Scene panels" becomes "stop registering them on the Scene tab".
   - Phase 5 has one remaining wrong-group read, in an acceptance script.
7. **`SPEC-multigrid-pressure`'s hook point is stale.** Production PCG fuses Jacobi into
   three update kernels; `k_apply_preconditioner` was used only by the ST-FLIP solver.
   Lane C's brief says so. The preconditioner setting (RNA, `SolverSettings`, migration)
   is wired by the orchestrator after A and C merge, not by C.

## Lanes

| Lane | Branch | Worktree | Head | Tests | Blocker | Next action |
|---|---|---|---|---|---|---|
| D st-flip-removal | `3.0/lane-d-stflip-removal` | `C:\Users\Tyler\wt\r30-d` | `34528aa` | 2053 passed, 3 skipped | | **merged** as `6ce6b33`; demo hash unchanged |
| E hygiene | `3.0/lane-e-hygiene` | `C:\Users\Tyler\wt\r30-e` | `cd57eb1` | 2236 passed, 3 skipped | | **merged** as `dfb2d91` |
| B cache-v3 | `3.0/lane-b-cache-v3` | `C:\Users\Tyler\wt\r30-b` | `c391234` | 2190 passed, 3 skipped | | **merged** as `57c6540` |
| B2 cache-v3 consumers | `3.0/lane-b2-cache-consumers` | `C:\Users\Tyler\wt\r30-b2` | `d3cbe3e` | 2289 passed, 3 skipped | | **merged** as `fcc6357`; cache-v3 Blender acceptance passed on `881b0ab` |
| C multigrid | `3.0/lane-c-multigrid` | `C:\Users\Tyler\wt\r30-c` | `37b2315` | 2071 passed, 3 skipped | | **merged** as `881b0ab`; JACOBI stays the default, so the demo hash is unchanged |
| A split + defaults | `3.0/lane-a-split-defaults` | `C:\Users\Tyler\wt\r30-a` | `5cc031d` | | | **merged** as `cc21cc7`, fixups `582ccbc`; shipped-scene settings dump identical to base |
| A2 preconditioner wiring | `3.0/lane-a2-preconditioner` | `C:\Users\Tyler\wt\r30-a2` | `697f342` | 2357 passed, 3 skipped | | **merged** as `6b0abbd`; JACOBI default, demo hash unchanged |
| F APIC | `3.0/lane-f-apic` | `C:\Users\Tyler\wt\r30-f` | `5307eb9` | 2338 passed, 3 skipped, 1 xfailed (lane) | | **PARTIAL**, **merged** as `cfff382`. The first head `b14b3ad` changed FLIP bits (hash `6a579cd4…`), so it was pulled back out; the fix gives APIC its own kernels. The 0.98 angular-momentum gate is a strict xfail; APIC keeps 0.939 vs FLIP 0.940 because the loss is in the shared free-surface path. Enforced instead: APIC > 0.90 and > PIC + 0.25 |
| H Alembic + playback-only | `3.0/lane-h-export` | `C:\Users\Tyler\wt\r30-h` | `b322950` | 1 strict xfail (Warp-free signature) | settings signature imports Warp | **PARTIAL**, **merged** as `2eb067e` + gate entry `b54f1e2`; `ALEMBIC_EXPORT_OK` on Blender 5.2 with CUDA hidden (21 checks, bbox error 0) |
| I VDB spike | `3.0/lane-i-vdb-spike` | `C:\Users\Tyler\wt\r30-i` | `8ebb281` | 2319 passed + gate fix | | **KEEP**, **merged** as `aae0607` (2 of 3 attempts; Blender 5.2 meshes a sphere within 0.05 voxel). Not wired to export; wiring waits for lane J's `liquid_sdf_*` data (check cell-centre `origin` then) |
| G resolution divisions | `3.0/lane-g-divisions` | `C:\Users\Tyler\wt\r30-g` | `2bc4a91` | 2371 passed, 3 skipped (lane) | | **PARTIAL→accepted**, **merged** as `b4c9dc5`. `RESOLUTION_DIVISIONS_OK` on 5.2 (merged tree). About 19 acceptance scripts now snap `voxel_size` to whole divisions (table in lane report); the GUI gate covers them |
| L Warp-free signature | `3.0/lane-l-warpfree-signature` | `C:\Users\Tyler\wt\r30-l` | `b769a65` | 2416 passed + 3 skipped + 1 xfailed | | **merged** as `90d13ba`. Signature pinned `e2a87bf9…8d38` (unchanged on base and head). L stopped PARTIAL on 3 tests that ast-parse `sim/solver.py` for `SolverSettings`; the orchestrator repointed them (`b769a65`) |
| J narrow-band | `3.0/lane-j-narrow-band` | `C:\Users\Tyler\wt\r30-j` | base `90d13ba` | | | Stop 1 `b38d017` (2455 passed, 3 skipped, 6 xfailed; ownership map 214 → 217) **accepted**. Stop 2 `5bb8d6f` **merged** as `2952a95` (2471 passed, 3 skipped, 1 xfailed). Orchestrator follow-up `0d39abc`: the Liquid voxels readout shows only in narrow-band mode. Stop 2 measured 8.0× fewer particles under Multigrid (7.6× under Jacobi), column height 0.74217 full band against 0.74218 narrow band, hydrostatic pressure within 1%, bitwise determinism. Decisions: the settings-signature re-pin is accepted (3.0 unreleased); stop 1 is not merged alone (its toggle would do nothing); the overflow counter is solver-side only; the column test runs under Multigrid. **Finding:** the full-band 16×16×32 column at rest sinks from 0.75 to 0.558 in 30 frames (about 34% compression) under the default Jacobi pressure solve. Stop 2 confirmed it is the Jacobi solve, not narrow band (Jacobi full band 0.558, narrow 0.550; Multigrid holds 0.742); this supports making Multigrid the default at the beta |
| K docs | `3.0/lane-k-docs` | `C:\Users\Tyler\wt\r30-k` | `feb8ba1` | 2412 + 1 expected red | | **merged** as `9e8912c`. The red test (CHANGELOG must lead with `version.py`) was closed by the 3.0.0 alpha bump `800b0ca` + `21bc7c4`. Technical-report "49/51 surfaces" count is undeterminable headless; prose generalised, needs the Blender validator at the gate. L's CHANGELOG line added by the orchestrator |
| M defaults + re-baseline | `3.0/lane-m-defaults` | `C:\Users\Tyler\wt\r30-m` | `0ad5b2b` | 2510 passed, 3 skipped, 1 xfailed, 1 failed (fixed in `0ad5b2b`, file 9/9) | targets missed | **held, not merged.** APIC default for new domains; a `fluidsim_defaults_epoch` stamp pins unstamped (pre-change) domains to FLIP + Jacobi on load, Demo Gallery and File > Append (`blend_import_post`; closes the Append gap); `DEFAULTS_EPOCH_OK` on 5.2; hash with APIC `c7542ea1…96a4` (2/2, 21/21 checks). Single-run dam_break probe (per-solve sync, so indicative only): APIC ≈1.5× per frame vs FLIP and ~35% more solves; Multigrid 10× fewer iterations but ≈2.5× slower per solve; narrow band ≈1.4× slower per frame and **particles run away to the 600k cap** by f48 (full band 256k). Owner decision 2026-10-09: optimize before 3.0 (lanes O, P, Q) and fix narrow band; defaults decided on paired re-measurement |
| O APIC perf | `3.0/lane-o-apic-perf` | `C:\Users\Tyler\wt\r30-o` | base `c10c32b` | no commits | — | done, no code change. Paired ABBA (dam_break f2–48, Jacobi, band off): step ms per substep FLIP 7.20 vs APIC 6.62 (−8 %), so per-substep cost meets the target; but APIC runs 548 substeps vs FLIP 406 (+35 %) because it dissipates less energy (peak speed ~14 vs ~9; energy lost by f44 PIC 1.11M, FLIP 0.93M, APIC 0.80M). CFL reads plain `p.velocity`, no affine inflation, no defect. Median frame 118.8 vs 94.2 ms (1.26×). A percentile CFL would cut substeps in every mode but changes FLIP bits: owner decision. Evidence `C:\Users\Tyler\wt\orch\scratch\lane-o\` |
| P Multigrid perf | `3.0/lane-p-mg-perf` | `C:\Users\Tyler\wt\r30-p` | `9ae88bf` | merged `7ed73ee` | target not met. Cause: launch-bound V-cycle (eager start V-cycle ~1.7 ms, host hierarchy rebuild ~0.2 ms, ~40 launches). Fix: start V-cycle and hierarchy replay from cached graphs, gather restriction, prolongation folded into first post sweep, compact fine level; V-cycle output bitwise, demo hash `8baa886c…` unchanged. Paired (APIC, f2–48, 1e-5): Jacobi 1.54 ms/solve (50 it), MG 3.84 → 2.07 ms/solve (5 it), MG now 1.34× slower than Jacobi. At 2× Resolution Divisions: Jacobi 2.92 (75 it) vs MG 5.62 (10 it), no crossover. Suite 2508 passed / 3 skipped / 1 xfailed. Evidence `C:\Users\Tyler\wt\orch\scratch\lane-p\` |
| Q narrow-band fix | `3.0/lane-q-nb-fix` | `C:\Users\Tyler\wt\r30-q` | `9aa4c2f` | merged `49d7988` | runaway fixed, speed target not met. Cause: reseed topped every sparse band cell (<4) back to 8 with nothing removing surplus, a ratchet as the liquid spread. Fix: reseed only cells with <2 particles, fill to ppv/2, capped by `owed` (particles killed, not yet replaced; checkpointed, old checkpoints restore 0); new `test_narrow_band_collapse.py`; Lane J overflow test now uses an 8-layer lattice (assertions unchanged). Paired dam_break: particles nb 46k/144k/256k at f2/f20/f48 (never above full 256k); wall FLIP+Jacobi 1.58×, APIC+MG 1.36× of full band — dam_break is mostly thin sheet after f20 and nb keeps more liquid voxels (32.9k→54k) plus ~22 % overhead. Band off bitwise (`8baa886c…`). Suite 2508 passed; merged-tree `tests/solver` 233 passed / 1 xfailed. Evidence `C:\Users\Tyler\wt\orch\scratch\lane-q\` |
| N hygiene follow-ups | `3.0/lane-n-hygiene2` | `C:\Users\Tyler\wt\r30-n` | `ea1961d` | 2506 passed, 3 skipped, 1 xfailed (ownership map regenerated after) | | **merged**. 11 of 12 listed loops were already guarded; the Add-menu remove (`menus.py`) and `interchange.py` now keep walking. Time Offset removed: Blender stores the enum index, so a stale index read back as `''`; a load migration sets it to Material (probe on 5.2). The stale test name was accurate; unchanged |

### Wave 2 deviations

- **Lane J spec is wrong at its core (verified on `cfff382`).**
  - Topology is rebuilt every substep from particle positions.
  - `VOXEL_FLUID` comes from per-cell particle counts.
  - So the spec's "phi from `VOXEL_FLUID`" is circular. Once the interior particles are
    killed, the interior leaves its leaves and FLUID flags, and the liquid collapses to
    the band.
  - The lane must carry a persistent, advected liquid level set and interior grid
    velocity across the rebuild, as `churn` is carried today.
  - The particle-splat SDF also feeds surface tension and whitewater typing, not just
    the render surface.
  - Top-up sources would refill the interior.
  - The "Manual rebuild reads phi from cache" item has no consumer.
  - So J runs test-first in two stops. The orchestrator reviews the design note and
    tests before any kernel lands. Demos are enabled by the orchestrator in a separate
    commit.
- **Disk incident, 2026-10-09.** Stray background `python -` heredocs left by lanes F,
  H and I fell into a REPL error loop and wrote about 10 GB of tracebacks to the session
  TEMP task directory. Free space fell to 30 GB.
  - The processes were stopped and the junk logs truncated.
  - The stray commands also left post-merge edits in the lane worktrees: a duplicate
    `Step` in r30-h `registration.py` (reverted), and a test docstring in r30-f
    (reverted by F).
  - The merged tree was unaffected.
  - `_env.md` now forbids stdin interpreters.

### Wave 1 deviations

- **Wave 1 started before the alpha.1 gate.** The plan only needs A and B merged; the
  alpha.1 gate (one GUI pass, ABBA if needed) runs on the merged tree once Wave 1 merges settle.
- **Defaults held for the hash.** APIC ships with FLIP still the default; multigrid ships
  with JACOBI. Both flip at the beta re-baseline, so each hash change is attributable.
- **Lane F (APIC).** `sample_type` gains `APIC = 2` (no rename, no ASFLIP). LIVE tier
  with a lazily zeroed affine channel. Stop-1 review changed two things: a cross-mode
  checkpoint import drops or zero-fills the affine instead of raising, and the memory
  estimate must match real allocations. It added tests that the affine follows its
  particle through reorder and compaction.
- **Lane H scope.** It exports the SURFACE mesh only. Cache playback needs live mode plus
  `cache_playback_enabled`, not live off. `register()` was already Warp-free.
- **Lane I.** It writes the surface level set only. There is no velocity grid, because
  the cache stores no MAC field.
- **Lane G design.** `voxel_size` stays as a derived get/set RNA property over the new
  `resolution_divisions`; it is not deleted. The tier is REINIT, not RESET. Domain
  resize is already in the REINIT signature. Migration runs on load. The demos are
  measured, not re-saved. Fixed a ceil rounding trap that could pad 64 divisions to 72.

- **Lane G demo grids (decided by the orchestrator, 2026-10-09).**
  - The spec wants each migrated demo to keep its padded dims. crown_splash cannot: the
    shipped 88,88,56 came from a float32 ceil artifact (0.8000000119 / 0.00999999978 =
    80.0000030 → 81 cells), which the required rounding tolerance fixes. It is now 80,80,56.
  - dam_break's 5.6 m axis is not a whole number of 0.03 m cells, so 187 divisions give
    0.02995 m. Its dims are unchanged.
  - Both changes are accepted. G merges on its own, after F's FLIP hash is proven
    identical, so any demo-hash change is attributable to G. It becomes part of the
    alpha.2 re-baseline the plan already schedules.

### Open items from lane H

- **Warp-free settings signature (owner: orchestrator, after F and G merge).**
  - Opening a cache with no importable Warp still fails. The chain is
    `playback._check_cache_scene_key` → `cache_identity.settings_signature` →
    `solver_settings.build_settings` → `sim.solver.SolverSettings`, and `sim/solver.py`
    imports Warp.
  - With Warp importable but no CUDA device (the common case), playback-only works.
    The Alembic acceptance proves that.
  - Fix: move the `SolverSettings` dataclass into a Warp-free module and re-export it
    from `sim/solver.py`. Then remove the strict xfail
    `test_playback_signature_check_does_not_import_warp`.
  - Held until F and G merge, because F edits `sim/solver.py` and both may touch
    `SolverSettings`.
- POINTCLOUD Alembic export writes only an empty transform on 5.2, so particles are out of
  scope. Blender 5.2's `alembic_export` has no `visible_objects_only`.
- Restoring a scene that started with live on during export is untested: it needs CUDA.
  Cover it in the GUI gate run.

### Open items from lane D

- Decide what the `TIME_OFFSET` color mode shows now that ST-FLIP is gone
  (`output_publication.py:840`).
- Run the Blender scripts D edited but could not run headless.
- Verify the count quoted at `technical_report.md:481`.
- Regenerate the ownership map after all Wave 0 merges and re-check
  `EXPECTED_FIELD_COUNT = 212`.
- CHANGELOG entry for the ST-FLIP removal (written by the orchestrator at alpha.1).

### Open items from lanes E and B

- **Blender runs owed at the alpha.1 gate.**
  - The GUI acceptance's new `gpu.offscreen` viewport check has never drawn on a real
    GPU.
  - The five Phase 2 source acceptance lanes have not run under the new `setup()` flow.
  - B's acceptance scripts ran at `fc99915`, not on the merged tree.
- **Unregister loops lane E did not own.**
  - `blender/menus.py:107-111` has no guard at all.
  - These are guarded only for `(RuntimeError, ValueError)`:
    - `blender/effects.py:1130`, `blender/demo_gallery.py:386`, `preferences.py:194`,
      `blender/effect_ui.py:962`, `blender/effect_workspace.py:930`;
    - `properties.py:1657`, `ui_header.py:187`, `ui_physics.py:446`, `ui_stages.py:406`,
      `ui_scene.py:1361`, `ui_object.py:336`.
  - Candidate for a hygiene follow-up after lane A merges.
- **REVIEW-027.** E read "sourcing count mismatch" as the direct-injection transaction
  count checks plus the packet velocity/position check. The orchestrator accepts that
  reading; no code has a check by that exact name.
- **Order-dependent test, pre-existing.**
  `tests/solver/test_pressure_st_and_graph_cache.py::test_native_pcg_module_unloads_on_pressure_solver_release`
  fails in some hand-picked file lists. It passes in `tests/solver` and in the full suite.
- **Cache v3 caveats from B.**
  - Resume scans stamps forward with `frame_set` per frame, so its cost grows with the
    number of baked frames.
  - The stamp covers `matrix_world` and geometry. No test covers rigid-body-driven
    colliders.
- **Stale "cache-v2" text for lane K (docs).** `user_guide.md` 9, 23, 97, 363, 392, 484,
  579, 583, 638, 757; `technical_report.md` 5, 48, 50, 255, 321, 506, 517;
  `custom_effects.md` 91, 98; `resources/examples/README.md` 20. The owner's uncommitted
  2.1.1 bump also touches the first lines of `user_guide.md` and `technical_report.md`.
- ~~Lane A strings~~ done in `582ccbc` (enum relabelled "Resumable Cache", identifier
  `V2` kept).

### Open items from lane A

- **Lane A2 (merged as `6b0abbd`).**
  - Wire the preconditioner setting: RNA enum (JACOBI default, MULTIGRID),
    `SolverSettings` field, `build_settings` projection, pressure panel, pending-settings
    classification.
  - "Open Full Settings" buttons (`fluidsim.open_scene_properties`) should open the
    Physics tab when a domain is active, now that domain panels live only there.
  - Guard the unregister loops listed above.
  - Rename the stale test name at `tests/.../test_effect_cache_contract.py:130`.
- Three non-Blender `SolverSettings` users now get the unified defaults; check them at
  alpha.1.
- `setting_owner` docs still describe the pre-split ownership.
- Delete the `lane-a-base` export (about 12 MiB) once alpha.1 passes.

### Lane C stop-1 review (accepted)

- **Spec corrections.**
  - `k_apply_preconditioner` is not dead code: `sim/viscosity.py` launches it. Keep it.
  - Restriction is a gathered sum × 0.25 with piecewise-constant prolongation masked to
    fluid. The spec's fluid-count average with injection is not SPD.
  - Air neighbours count on the coarse diagonal; solids and out-of-domain neighbours do
    not. A coarse cell is fluid if any child is fluid, and solid only if all children are.
- **Test 6 agreement tolerance.** Velocities are compared at tolerance 1e-6 with the 1e-4
  bound; convergence is still asserted at 1e-5. Two correct solvers stopped at 1e-5
  differ by about 7e-4, so the brief's 1e-4 at 1e-5 was infeasible.
- **Check interval.** MULTIGRID chooses its own odd interval that divides
  `max_iterations` (5 for the default 200). JACOBI keeps the caller's 25. At 25 every
  MULTIGRID count would round up to 25 and understate the iteration ratio.
- **Lane C's own estimate.** On the 32³ test, the float64 reference needs 13 MULTIGRID
  iterations against 123 for JACOBI at 1e-5. Lane C expects dam_break to keep ≥3× fewer
  iterations, but ≥1.5× faster per solve is at risk: one V-cycle costs about 5–8
  fine-grid passes.
- **Added for stop 2:**
  - a test through the production `project` path;
  - a test that the graph key separates the preconditioners;
  - a scanner that also covers recorded-launch builders.

Lane briefs are under `C:\Users\Tyler\wt\r30_briefs\`, outside the repository. Each is
`docs/specs/3.0/WORKER_BRIEF_TEMPLATE.md` filled in, with a shared environment section.

## Milestones

| Milestone | Contents | Gate | State |
|---|---|---|---|
| alpha.1 | Wave 0 (A–E) | 1 GUI gate (hash match), ABBA perf if budget sources changed | **passed** on `5a2ce80` (gate includes Waves 1–2) |
| alpha.2 | Wave 1 (F–I) | | lanes merged; gate passed on `5a2ce80` with FLIP/Jacobi defaults |
| beta | Wave 2 (J, K), defaults switch + re-baseline | 1 GUI gate, `--measure-budget` | lanes J, K, N merged; perf lanes O (APIC, no change), P (Multigrid 1.85×, merged) and Q (narrow-band runaway, merged) missed their targets, so the owner kept FLIP, Jacobi and full band as defaults (2026-10-09). Lane M (defaults switch) dropped, not merged |
| 3.0.0 | channel `alpha` → stable, tag, publish | owner authorization | channel stable in `448b286`; 1 GUI gate FULL-PRODUCTION-GATE-PASS on `448b286` with re-measured budget (PERFORMANCE-BUDGET-PASS, 8 trials) and demo hash `8baa886c…baf3` (log `C:\Users\Tyler\wt\logs\gate_r30_448b286.log`). Lane R (narrow-band ghost liquid from `team-debug-20261009-findings.md`) refuted the hypothesis: `update_phi` runs `band+2` inward Eikonal sweeps right after `k_nb_union_band`, so splat-born thin liquid gets depth ≤ h, stays above `pin_depth` and can return to air; on dam_break band-only ghost cells (liquid phi, no particle within 2 cells) drain 18.6k → 0 by f40 and are the particle-free interior by design, and band liquid voxels track the full band's occupied voxels. Only the regression test `test_narrow_band_ghost.py` merged (`a7f60de`); the remaining ~1.44× is per-substep level-set and reseed cost, not profiled. Tag and publish await owner authorization |
