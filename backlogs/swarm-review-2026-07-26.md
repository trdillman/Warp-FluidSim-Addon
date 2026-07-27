# Swarm Codebase Review — 2026-07-26

**Scope:** whole repo (`fluidSim_warp_blender51/`, `fluidSim_synth/`, `tests/`, `tools/`, `dispatch_extraction_from_houdini_archive/`). ~25.6K LoC Python.
**Lanes:** 8 parallel (security skipped per `--skip`). Read-only.
**Tally:** 85 findings — **2 BLOCKER · 19 HIGH · 30 MEDIUM · 24 LOW · 10 NIT**

| Lane | B | H | M | L | N | Tot |
|---|---|---|---|---|---|---|
| 1 Correctness              | 0 | 0 | 0 | 3 | 0 | 3 |
| 2 Performance/scalability  | 1 | 3 | 4 | 2 | 1 | 11 |
| 3 Architecture/design      | 0 | 3 | 6 | 3 | 1 | 13 |
| 4 Numerical stability      | 0 | 4 | 6 | 3 | 2 | 15 |
| 5 Testing/verification     | 1 | 5 | 5 | 3 | 0 | 14 |
| 6 API/Blender contracts    | 0 | 1 | 3 | 5 | 3 | 12 |
| 7 Error handling           | 0 | 3 | 6 | 2 | 0 | 11 |
| 8 GPU concurrency          | 0 | 0 | 0 | 3 | 3 | 6 |

---

## BLOCKER (2)

**B1 · Performance — `diagnostics/profiler.py:144-151`** `sync=True` default. Every `with self._phase(name)` drains the CUDA stream at entry + exit. 14 phases/substep × N substeps = **112 host↔device barriers/frame at 4 substeps**. Defeats all async streaming; "GPU-resident" claim false whenever profiler attached. **Fix:** default `sync=False`; use CUDA events / `wp.ScopedCapture` for substep timing. *(Same issue surfaces as NIT in lane 8 — consolidate here.)*

**B2 · Testing — `tools/run_production_gate.py:170-189`** gate `py_compile`s + runs 17 test scripts but **14 do not exist** in this worktree (incl. `test_pressure_projection_regressions.py`, the only file that would exercise `sim/pressure.py` projection correctness). `py_compile` aborts on first missing file → gate never runs. Acknowledged in `backlog/tasks/TASK-POLICY-DRIFT-010-NESTED-FULL-LANE-MISSING-SCRIPTS.md` but unresolved. **Fix:** restore scripts or skip missing lanes with explicit `MISSING-LANE` marker.

---

## HIGH (19)

### Numerical stability — scale-dependent convergence (CLAUDE.md smoking gun)
*"If scaling up materially changes shape/semantics, that is a design defect."* Three of four iterative solvers violate this.

**N1 · `sim/pressure.py:1788,1818-1831` (also 1582,1636 in `solve_pcg_st`)** PCG stop test `res_sq < tol²·max(1,full_n)` normalizes by `full_n`=active voxels, but residual summed only over pressure unknowns (`pressure_count`, padded to `work_n`). For sparse splash (1% fluid) effective per-unknown tolerance ~10× nominal; 0.1% fluid ~30×. Sparse scenes converge looser. **Fix:** `tol_sq = tolerance²·max(1,pressure_count)`; better, relative `‖r‖/‖b‖ < tol`.

**N2 · `sim/viscosity.py:414`** same scale-dependent criterion (`tolerance²·max(1,n)`, n=full active grid) **+ no `p_ap≈0`/`rz≈0`/non-finite safeguards**. Singular/NaN system iterates to `max_iterations`, writes garbage velocity via `k_viscosity_apply_collision`. **Fix:** divide by viscous-unknown count; add INVALID/SINGULAR classification like pressure driver.

**N3 · `sim/solver.py:1251-1274`** CFL substep count hard-clamped to `max_substeps` (default 4). When requested `n>4`, steps with `dt>dt_cfl` — CFL violation, advection undersample, collision tunneling — **no user signal**. **Fix:** surface non-fatal diagnostic; clamp `dt≤dt_cfl`; or runtime-raise `max_substeps` to a safety budget.

**N4 · `sim/pressure.py:1669-1670,1851-1852` (also 1473-1481)** non-converged (STALLED/MAX_ITER) pressure solve **zeroes entire field** → divergent velocity untouched → visible velocity pop on stall frame. `solve_jacobi:1721` sets `pressure_valid=True` unconditionally (Jacobi always projects; PCG stalled projects nothing) — inconsistent contract. Stall threshold `1−1e-6` (~1 ppm) below float32 eps, rarely fires. **Fix:** STALLED keep best-iteration pressure (track it); MAX_ITER zero only if residual non-finite; tighten stall threshold to ~1e-2 relative.

### Performance
**P1 · `sim/viscosity.py:387-415`** viscosity PCG: ~1320 launches + 12 host syncs/substep (3 axes × 40 iter × ~11 launches). Pressure PCG (`pressure.py:1325-1357`) already CUDA-graph-captured; viscosity not migrated. **Fix:** apply `_pcg_apply_cmd`/`wp.ScopedCapture` pattern. Matrix fixed per substep → same graph keying works.

**P2 · `sim/kernels_core.py:232,244`** max-speed reduction = single-address `wp.atomic_max` across all threads. At 2M particles: fully serialized tail. Warp ships tiled `wp.utils.array_inner` (used in pressure). **Fix:** tiled reduction or `array_inner` then `sqrt`.

**P3 · `sim/sourcing.py:1544-1556,1660-1679`** one `wp.launch` per mesh source per substep over full active voxel set → `2·M·S` full-voxel scans/frame. Analytic-source path one screen above already batches all shapes into one launch. **Fix:** pack `MeshSourceDesc` into `wp.array(dtype=struct)`, loop sources inside kernel.

### Architecture
**A1 · `sim/registry.py:24-57`** `KERNEL_REGISTRY` populated by ~150 `register_fn()` calls; solver calls kernels **directly by Python name**. `lookup()` used once (validation.py:464 alias-identity assertion). `registered_names()` has zero callers. Speculative dispatch layer, inert inventory. **Fix:** route launches through `lookup(name).fn` (commit to contract) or delete registry + register_fn callsites.

**A2 · `sim/custom.py:824,863`** snippet templates hardcode `from fluidSim_warp_blender51.sim import ...` (rest of code uses relative imports). Generated module written to temp dir, loaded via `spec_from_file_location` → absolute path only resolves if package on `sys.path` under that exact name. Rename/vendor/zip → every user snippet compile fails. **Fix:** resolve parent package at runtime; emit `from {parent_pkg}.sim...`.

**A3 · `sim/custom.py:724-750`** `FIELD_ACCESSORS` lambdas reach into solver privates (`s.pressure_solver.pressure`, `s.vel.u`, `s.particles.attr("age")`, `s.surface_sdf.data`). Rename of any solver attr silently breaks snippet binding at runtime (deferred failure). `properties.py:22` imports `sim.custom` at registration so `KNOWN_FIELDS` must mirror accessor keys. **Fix:** expose `solver.field_handle(name)`/`particle_attribute_handle(name)`; replace lambdas with static name→dtype table.

### Testing
**T1 · `sim/viscosity.py` 0% direct coverage** — 10 kernels + `ViscositySolver` never imported by any test. Regression undetectable.
**T2 · `sim/custom.py` 0% direct coverage** — `validate_snippet`/`autofix_snippet` (security-relevant AST gate) only exercised end-to-end via CUDA+Blender acceptance lane. **Fix:** pytest-unit against fixed input strings.
**T3 · `sim/kernels_fluid.py` 1/~30 kernels directly tested** (only `k_particle_in_sink_check`). 29 kernels incl. `k_advect_particles`, `k_forces_speed_limit`, `k_forces_gravity` untested.
**T4 · `sim/pressure.py` PCG/Jacobi/MG convergence untested** — `solve_pcg`/`_launch_pcg_batch` checked only via `inspect.getsource()` substring. No test runs PCG to convergence. Would have lived in missing `test_pressure_projection_regressions.py`.
**T5 · `sim/surface.py` 1/12 kernels directly tested** (only `k_sdf_to_axis_collision`). Viewport surface only smoke-checked (`len(verts)>100`).

### API contracts
**AP1 · `operators.py:*`** **zero `poll()` classmethods across all 33 operators.** Every guard runtime-only in `execute()`. UI buttons clickable in invalid contexts. **Fix:** add `@classmethod poll()` for solver-dependent ops (`snippet_apply/test`, `reset`, `step`, `inject_particles`, `export_*`).

### Error handling
**E1 · `blender/scene_bridge.py:193-196`** `Session.step` swallows `cache.write_frame` exceptions into `print()`; `fs.status` overwritten next frame, `sess.error` never set. User discovers cache gaps only on playback `FileNotFoundError`. **Fix:** set `fs.status` + `set_error` with distinct prefix.
**E2 · `operators.py:202-210`** `FLUIDSIM_OT_step` doesn't release broken solver on step failure — inconsistent with `handle_frame_change` (scene_bridge.py:309-314) which calls `release(preserve_error=True)` + `fs.live=False`. Subsequent ops hit broken solver. **Fix:** call `_bridge().release_session(scene)` on except path.
**E3 · `sim/memory_safety.py:35-42`** `release_cuda_memory` can raise from `wp.synchronize_device`/`set_mempool_release_threshold` (most common when device in CUDA error state) → masks original init/step error. Called from init's except via `_clear_runtime`. **Fix:** wrap body in broad try/except, log secondary cleanup errors.

---

## MEDIUM (30) — abbreviated

**Numerical:** `pressure.py:821` leaf-Jacobi rhs `64.0` literal unjustified (right value, obscured form; breaks if `LEAF_DIM`≠8). `surface.py:510-511` mean-curvature SDF step uses linear-diffusion CFL `h²/6`, unstable where `|κ|·dt>1` → SDF sign flip → floating fragments. `viscosity.py:60` explicit viscosity clamp saturates diffusion CFL silently → resolution-dependent dissipation. `pressure.py:1718-1722` `solve_jacobi` reports success, `last_residual=-1.0` sentinel, no convergence test. `kernels_fluid.py:800-801` ST-FLIP per-particle dt_clamp can stall particles with negative residuals (frame-rate-dependent). `pressure.py:73-78` phase weight `sqrt` can pass inf from unsanitized P2G accumulators.

**Performance:** `surface.py:219-220` low-res SDF rebuilt every substep when whitewater enabled regardless of churn. `frame_output.py:67,77` particle/ww attr readback pulls full capacity across PCIe then host-slices (sibling `positions` correctly device-slices). `profiler.py:183` `history.pop(0)` O(N) on 600-deep list every frame. `solver.py:960-962,1345-1347; viscosity.py:311-313` three separate `wp.copy` launches where one strided kernel suffices (recurring pattern).

**Architecture:** `solver.py`+`sourcing.py` descriptor dataclasses scattered; `solver.py` re-exports sourcing descriptors it doesn't own. `solver.py:484-1448` `FluidSolver` god-object (963 lines, ~70 methods) mixes allocation + orchestration + host I/O. `scene_bridge.py:26-314` `Session` mixes lifecycle/error-persistence/settings/handler registration; singleton global state. `scene_bridge.py:598-616` dead `gather_*` compat wrappers (zero callers). `__init__.py:23` `ensure_warp_discoverable()` runs subprocess probe of every PATH `python.exe` at addon import. `properties.py:227-254` `_snippet_library_items` does `os.listdir`+`json.load` on every UI redraw. `properties.py:4-5,22` docstring "registration never touches Warp/CUDA" misleading — imports `sim.custom` which builds 9 presets, 22 accessor lambdas, 2 templates at registration.

**API:** `operators.py:703-706` `snippet_validate` returns FINISHED on ERROR-level issues (sibling `snippet_test` returns CANCELLED). `operators.py:558-576` `snippet_apply` applies valid snippets then returns CANCELLED (violates "CANCELLED = no state change"). `operators.py:94-166` `apply_safe_settings` mutates RNA props without `UNDO` in `bl_options`.

**Error handling:** `scene_bridge.py:247-275` `export_cache_sequence` mid-bake failure leaves orphan frames + no sequence metadata. `io/export.py:21-65` + `diagnostics/report.py:53-59` direct `open()` not temp-rename pattern (cache.py does it right). `sim/surface.py:774-783` mesh verts not sanitized for NaN/Inf before publish/export. `operators.py:147-166` `apply_safe_settings` mutates props before init; failure leaves mutated settings, no rollback. `io/cache.py:201-211` `available_frames` lists filename matches without verifying readability (partial .npz reported as available). `sim/sourcing.py:1684-1719` `inject_particles` no shape validation → confusing deep Warp error.

---

## LOW / NIT (34) — see lane detail
Highlights: `constants.py:55` `DEFAULT_LEAF_CAPACITY=8192` + max particles/whitewater hard caps with no growth policy or near-capacity warning. `pressure.py:1875-1876` `measure_residual` pulls two full arrays to host (numpy norm). Whitewater free-list publication order (count-before-slot) fragile under future concurrency. `whitewater.py:65-72` churn uses lower-face MAC not cell-centered velocity. `surface.py:778-782` MC sample-index convention unconfirmed (possible +0.5 voxel shift). `enums.py:66` `SdfFilterType.DIALATE` typo in UI. `enums.py:297-304` `enum_items` emits int in icon slot. `__init__.py:13` vs `constants.py:10` version drift `(1,0,241)` vs `"1.0.240-beta"`. `sim/custom.py:933-948` shared temp-dir snippet write (skip-if-exists) plantable on multi-user host. `solver.py:96-122` ST-FLIP host mirror kernels unverified against device. `fluidSim_synth/` contracts unenforced (no `test_kernel_coverage.py`).

---

## Cross-cutting themes (synthesize before fixing)

1. **Scale-dependent numerics (N1+N2+N3+N4)** — the single most important cluster. Pressure + viscosity convergence, CFL clamp, pressure-zeroing all change physics as scene grows. This is the exact defect class CLAUDE.md flags. Fix convergence normalization first (one-line changes, high leverage).
2. **Silent failure masking (N3+N4+P-surface + E1+E2)** — every clamp/stall/gap hides from the user. Add `stats.*` counters and `set_error` plumbing before tuning.
3. **Profiler is the performance multiplier (B1)** — flipping `sync` default unlocks every other perf win (P1/P2/P3 measurable only after).
4. **Test gate is broken (B2+T1-T5)** — coverage holes exactly where numerics bugs live (pressure PCG convergence, viscosity, surface). Restore `test_pressure_projection_regressions.py` first.
5. **Dead speculative layer (A1)** — `registry.py` is 150 import-time inserts buying nothing. Decide commit-or-delete; current state is worst trade-off.

## Suggested fix order
1. B2 restore missing gate scripts (or skip-missing) — unblocks verification of everything else.
2. B1 flip profiler `sync` default — single line, unlocks perf measurement.
3. N1+N2 convergence normalization — one line each, fixes core physics-at-scale.
4. N3+N4 + E1+E2 add diagnostics/stats for silent failures.
5. P1 viscosity CUDA-graph capture — pattern already proven in PressureSolver.
6. A1 registry commit-or-delete — removes 150 inserts of inert indirection.
7. T1-T5 fill coverage for viscosity + custom + pressure convergence.

## Verified clean (no action)
Pressure projection sign convention (`A·q = -div·h²`, `u -= (q_hi−q_lo)/h`). PCG buffer parity swap. ST-FLIP temporal kernel host/device consistency. MAC neighbor leaf-stride (1,8,64). Whitewater regime + gravity signs. Viscosity matrix/stability forms. Enum consumption vs dispatch_contract. Atomic scatter (P2G), atomic_min (SDF union), counting-sort cursor, atomic_max reductions — all race-free. `bl_info`, register/unregister order, bl_idname uniqueness. Cache versioning + atomic write + `allow_pickle=False`. Kernel AST sandbox coherent with `@wp.kernel` template. Single-stream serialization enforces launch ordering across substep pipeline. **No confirmed GPU race condition.**
