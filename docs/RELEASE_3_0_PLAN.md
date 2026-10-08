# FluidSim Warp 3.0 — release plan

Status: proposal, 2026-10-07. Baseline is the 2.1.1 working tree on `release/2.1.0`
(Blender 5.2 LTS, Warp 1.17.0, CUDA only, Windows certified, Linux packaged).
Backlog source of record is `docs/RELEASE_2_1_HANDOFF.md`; this plan consumes it.

Detailed specs for the items most likely to be implemented wrongly live in
`docs/specs/3.0/`. **Implementers read the spec before touching the code.** Each spec
states the trap it exists to prevent, names the real symbols, and defines acceptance.

## Theme: bake once, render anywhere

3.0 finishes the data-model and cache work 2.x could not finish without breaking
files, and spends that one break on the three things artists hit first: noisy
transfer, slow pressure, and no interchange export.

## Why a major version

Three backlog items invalidate saved scenes or caches. They land together, once:

| Break | What changes for the user | Spec |
|---|---|---|
| Settings split Phases 3–5 | 232 fields leave `scene.fluidsim`; the domain object is the only owner; Scene-tab mirror panels go | `SPEC-settings-split-completion.md` |
| Cache v3 | per-frame source/collider stamps, source schema, transfer mode, grid spec; v2 caches refused | `SPEC-cache-v3.md` |
| Defaults and resolution | one default per field; `resolution_divisions` replaces `voxel_size` as the authored control | `SPEC-default-unification.md`, `SPEC-resolution-divisions.md` |

Policy: **no 2.x `.blend` or cache compatibility shims beyond the one-shot migration**
described in the split spec. Old caches are refused with the same message pattern cache
v1 gets today (`io/cache.py: detect_cache_format`).

## Pillars

### 1. Solver quality — APIC transfer and narrow-band particles
Today: PIC/FLIP blend, `flip_ratio` 0.97, no APIC (`sim/enums.py: ParticleSampleType`).
Add `APIC` as a transfer mode and make it the default for new domains. Narrow-band
particles then cut particle counts on deep water. Both are handoff §4 items.
Specs: `SPEC-transfer-modes-apic.md`, `SPEC-narrow-band-particles.md`.

### 2. Pressure — multigrid-preconditioned CG
Today: PCG with inverse-diagonal preconditioner and captured conditional graphs
(`sim/pressure.py: PressureSolver`). A two-level "electrostatic" leaf guess already
exists (`coarse_initial_guess`) but only seeds x0. Build a real symmetric V-cycle
preconditioner inside the existing graph capture. Spec: `SPEC-multigrid-pressure.md`.

### 3. Interchange and solver-free playback
Today: OBJ surface and PLY points only (`io/export.py`). Add Alembic mesh sequences and
OpenVDB grids, and make a baked cache playable and renderable on a machine with no
CUDA. Spec: `SPEC-interchange-export.md`.

### 4. Platform and trust
- Certify Linux (handoff §1.1). Blocker is the black Mesa viewport screenshot; use a
  `gpu.offscreen` draw in the GUI gate instead of a window grab.
- Close REVIEW-013 (unregister stops at first failure), REVIEW-027 (no solver error-path
  tests), REVIEW-038 (`CudaDotProduct.launch()` length checks). REVIEW-024/025 are
  test hygiene; do them alongside the split since those tests change anyway.
- **Remove ST-FLIP.** Variable Projection has been off by default since 2.0 and has no
  gated demo. Delete `*_spacetime` kernels in `sim/kernels_fluid.py`, the `*_st`
  pressure path in `sim/pressure.py`, `spacetime_enabled` and its settings, and the
  user-guide section. Record the removal in the changelog and keep the archive tag.

## Explicitly out of 3.0

- Two-way rigid-body coupling and a Species v2 whitewater rewrite: no file break
  needed, so they go in 3.1/3.2 without forcing re-bakes.
- Pressure-field sourcing and the tracking latch (`docs/source_shape_spec.md`): still
  deferred; no demo needs them.
- Defect #2 (compounding reset slowdown): parked until it reproduces with viscosity and
  surface tension enabled.
- CPU or macOS solver support. Solver-free *playback* on those platforms is in scope;
  simulation is not.
- ASFLIP: optional stretch inside the transfer spec, not a release requirement.

## Sequence and acceptance

Every milestone must pass the existing reproducible GUI gate three times with identical
hashes (`tools/run_production_gate.py --include-gui`, see
`memory: full-gui-gate-run-procedure`) and a paired ABBA step-time comparison against
2.1.1 on paddle_stir and crown_splash (median of frames 2–48, n≥6 pairs).

| Milestone | Contents | Hash change attributable to |
|---|---|---|
| 3.0.0-alpha.1 | split Phases 3–5, cache v3, default unification, ST-FLIP removal, Linux gate green, REVIEW cards | defaults only (verify by running alpha.1 with the 2.1.1 defaults forced: hash must equal 2.1.1) |
| alpha.2 | APIC default, narrow band, `resolution_divisions` | transfer; re-baseline the four demos, rebind `max_crown_height` evidence (see `docs/DEMO_SCENE_CONTRACT.md`) |
| beta | MGPCG, VDB writer spike result (ship or defer to 3.1) | none expected (preconditioner must not change the converged solution beyond tolerance; see spec) |
| rc | Alembic export, solver-free playback, user guide and technical report rewritten for the new ownership model | none |

Perf budget: alpha.2 may be slower per particle (APIC adds a 3×3 per particle) but must
be faster per frame on dam_break Final because of narrow band. beta must cut pressure
iterations ≥3× at tolerance 1e-5 on dam_break Final. Anything else over +5% is a stop.

## Orchestration: parallel subagent lanes

The work is run by one orchestrator agent that owns integration and never writes
solver code itself, dispatching bounded lanes to subagents. Two orchestrator profiles
are supported; the lane plan is identical, only dispatch mechanics differ:

| Profile | Orchestrator | Subagents | Dispatch |
|---|---|---|---|
| Anthropic | Claude Code (this repo's default) | Sonnet 5.5 | `Agent` tool with `isolation: "worktree"`, one agent per lane, `run_in_background: true`; continue a lane with `SendMessage` rather than respawning |
| OpenAI | Codex orchestrator | GPT-6.1-Sol | Codex subagent/worktree lanes; `omp` dispatch surfaces (`memory: omp-orchestration-surfaces`, note the headless footguns and that its `tools:` allowlist is not a sandbox) |

Rules that hold for both profiles:

- **One lane, one worktree, one owner-module set.** Lane ownership is the "Owner
  module" line of each spec. A subagent that needs a file outside its lane stops and
  reports; the orchestrator either re-slices or serializes. Never two lanes in
  `properties.py` at once.
- **Subagents never run the GUI gate or GPU benchmarks.** There is one GPU and the gate
  must not be disturbed (`memory: full-gui-gate-run-procedure`). Subagents run the
  pytest slice for their lane (`tests/<area>/`) and the headless Blender probe only.
  The orchestrator queues gate and ABBA runs serially after merges.
- **Shared index.** Subagents commit only inside their worktree. The orchestrator
  integrates with a private `GIT_INDEX_FILE` (`memory: shared-index-concurrent-commit`).
- **Brief = spec + template.** Every dispatch uses `docs/specs/3.0/WORKER_BRIEF_TEMPLATE.md`
  filled in; the subagent's first action is to read its spec's "Trap" section and
  restate it in its own words in the first progress message. A restatement that
  misses the trap is a re-dispatch, not a correction.
- **Evidence, not claims.** A lane is done when the orchestrator has re-run the lane's
  tests on the merged tree, not when the subagent says it passed. Subagent reports are
  model output; verify before acting on anything surprising.

### Lane map and dependency waves

Waves run in parallel inside; a wave starts when its dependencies are merged.

**Wave 0 (alpha.1, 5 lanes in parallel)**

| Lane | Spec | Owner modules | Parallel-safe because |
|---|---|---|---|
| A split | `SPEC-settings-split-completion` | `properties.py`, `ui_scene.py`, `pending_settings.py`, `solver_settings.py`, `setup_wizard.py`, `presets/`, `tools/release/validate_*` | sole owner of `properties.py` this wave |
| B cache-v3 | `SPEC-cache-v3` | `io/cache*.py`, `io/checkpoint.py`, `blender/cache_identity.py`, `production_bake.py` | io layer, no RNA |
| C multigrid | `SPEC-multigrid-pressure` | `sim/pressure.py` only (+ `tests/solver/`) | single module; preconditioner enum RNA is added by lane A on request via a one-line message, not by C |
| D st-flip-removal | plan §4 | `sim/kernels_fluid.py` `*_spacetime`, `sim/pressure.py` `*_st`, `spacetime_enabled` settings | **conflicts with C in `pressure.py`**: D lands first, C rebases (D is a day; C is weeks) |
| E hygiene | REVIEW-013/024/025/027/038, Linux `gpu.offscreen` gate draw | `blender/__init__.py` unregister path, `tests/`, `tools/run_production_gate.py`, `sim/cuda_reduction.py` (`CudaDotProduct`) | touches no solver math |

Lane A also absorbs `SPEC-default-unification` after Phase 4 (same files, same agent,
same worktree). Lane D is dispatched first and merged before C starts editing.

**Wave 1 (alpha.2, after A and B merge)**

| Lane | Spec | Owner modules | Notes |
|---|---|---|---|
| F apic | `SPEC-transfer-modes-apic` | `sim/kernels_fluid.py` transfer kernels, `sim/solver.py` particle buffers and `_g2p_advect`, `sim/enums.py`, `particle_capacity.py`, `capacity_planner.py` | F owns `solver.py` this wave |
| G divisions | `SPEC-resolution-divisions` | `sim/quality_contract.py`, `properties.py` (A is finished), `presets/`, `ui_stages.py` Setup | no overlap with F |
| H export-alembic | `SPEC-interchange-export` §A and §C | new `blender/interchange.py`, `io/export.py`, `playback.py`, `__init__.py` import graph | needs B's manifest helpers only |
| I vdb-spike | `SPEC-interchange-export` §B | new `io/vdb_writer.py` only | time-boxed 3 days; kill or keep decided by orchestrator |

**Wave 2 (after F merges)**

| Lane | Spec | Owner modules |
|---|---|---|
| J narrow-band | `SPEC-narrow-band-particles` | new `sim/narrow_band.py`, `sim/solver.py` step order, `sim/surface.py` SDF union, `simulation_admission.py`, demos re-save |
| K docs | user guide, technical report, CHANGELOG 3.0.0, `docs/RELEASE_3_0_HANDOFF.md` skeleton | `fluidSim_warp_blender51/docs/`, `CHANGELOG.md` (watch `tests/repo/test_contract_wording.py`) |

Multigrid (C) can merge in any wave; it is independent. The orchestrator slots its
gate re-baseline into beta per the milestone table.

### Orchestrator-only serial duties
1. Merge order within a wave: smallest diff first, then re-run the next lane's tests on
   the merged tree before merging it.
2. After each merge touching `sim/` or `properties.py`: one headless demo bake of all
   four demos, hash compared to the expected attribution (refactor waves must match the
   previous hash exactly).
3. Milestone close: 3× GUI gate, ABBA perf pairs, re-baseline commit with before/after
   hashes in the message, `max_crown_height` evidence rebind.
4. Decide I (VDB) at day 3. Decide ASFLIP at F's close based on remaining budget.
5. Keep one task record (`docs/RELEASE_3_0_PLAN.md` status block or a sibling
   `STATUS.md`): lane, worktree, head commit, tests run, blocker, next action.

### Sizing guidance
Sonnet 5.5 / GPT-6.1-Sol subagents handle the mechanical lanes (A Phase 3 audit, B,
D, E, G, H, K) end to end. Lanes with a numerical argument in them (C's symmetry proof
and restriction gather, F's per-component `D_p`, J's reseeding determinism) get the same
subagent but with an explicit instruction to produce the unit test named in the spec's
acceptance section **before** the kernel change, and the orchestrator reviews the test,
not the kernel, before letting the lane continue.

## Process notes

- Branch `release/3.0` off master once 2.1.1 ships. Each pillar on its own branch,
  stacked where the specs say so (cache v3 before narrow band; split before anything
  touching `properties.py`).
- Sessions share one git index (`memory: shared-index-concurrent-commit`); commit via a
  private `GIT_INDEX_FILE` when another session is active.
- Headless `--factory-startup` runs must set a private `BLENDER_USER_EXTENSIONS`
  (`memory: factory-startup-wipes-user-warp`).
- Re-baseline evidence is a deliberate commit with the before/after hashes in the
  message, never a silent update.
