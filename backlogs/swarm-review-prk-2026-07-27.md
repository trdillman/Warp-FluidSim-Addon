# Swarm Review — PRK (Particle Rendering Kernel) API — 2026-07-27

**Scope:** simulation → rendered-output pipeline only.
- `sim/frame_output.py` (HostFrame readback)
- `sim/surface.py` (SurfacePipeline/Extractor, MarchingCubes)
- `blender/viewport.py` (output collection publish)
- `blender/render.py` (render visibility/handlers)
- `io/export.py` + `io/cache.py` (OBJ/PLY/npz writers)
- `operators.py` export ops + `properties.py` show_*/surface props
- callers in `blender/scene_bridge.py`

**Lanes:** 6 parallel (no security lane — scoped review). Read-only.
**Tally:** 51 findings — **0 BLOCKER · 8 HIGH · 18 MEDIUM · 18 LOW · 7 NIT** (+1 unconfirmed VERIFY)

| Lane | B | H | M | L | N | Tot |
|---|---|---|---|---|---|---|
| 1 Correctness         | 0 | 1 | 3 | 2 | 1 | 7  |
| 2 Performance         | 0 | 3 | 3 | 2 | 0 | 8  |
| 3 API contracts       | 0 | 0 | 5 | 6 | 2 | 13 |
| 4 Numerical/surface   | 0 | 1 | 1 | 2 | 2 | 6  |
| 5 Error handling      | 0 | 3 | 5 | 3 | 1 | 12 |
| 6 GPU sync            | 0 | 0 | 1 | 3 | 1 | 5  |

---

## HIGH (8)

**H1 · Correctness — `sim/surface.py:109-111` `SurfacePipeline.build_key()` ignores particle state.** Key = `(frame, sim_time, settings_signature)`. Any particle mutation without a frame advance (e.g. `solver.inject_particles` → `sourcing.inject_particles:1718`, which clears only `_host_frame_cache`) leaves `built_key == build_key()` → `mesh()` (line 245) skips rebuild, serves stale `mesh_cache`. Inject path's followup `viewport.update_outputs` (operators.py:363) publishes a surface that doesn't enclose new particles; gap persists until next `step_frame`. At `live_surface_publish_cadence>1` wrong for multiple frames. **Fix:** fold particle topology/count hash into `build_key`, OR have particle-mutation entry points (`inject_particles`, custom snippets writing particle fields, `particles.compact`) set `surface.built_key=None` + `invalidate()`.

**H2 · Numerical — `sim/surface.py:510-511` mean-curvature step has no CFL guard on local curvature.** `dt = clamp(strength,0,1)·h²/6` is the *linear-diffusion* bound; nonlinear curvature flow CFL is `dt ≤ h²/(2d·max|κ|)`. Sharp feature (radius ~h) → `max|κ|≈1/h` → safe bound `h³/6`, code uses `h²/6` (20× over at h=0.05). Update unclamped → sign flip → floating fragments detached from surface. Band guard `if abs(c)>3h` (line 460) bounds spatial extent not magnitude. Worst at small h (MULTIPLIER mode). **Fix:** `step = clamp(dt·kappa_grad, -0.4h, 0.4h)` or adaptive substep.

**H3 · Performance — `sim/frame_output.py:62-70` `particle_attr` pulls full capacity across PCIe then host-slices.** `self.solver.particles.attr(name).numpy()[:n]`. At `max_particles=2M` = 8MB per scalar attr discarded tail. Comment blames "Warp 1.12.1 sliced views" but `positions`/`velocities` (incl. vec3 `velocity`) use device slice fine — self-contradictory. **Fix:** `attr(name)[:n].numpy()`; gate by dtype if a real failure exists.

**H4 · Performance — `sim/frame_output.py:72-80` `whitewater_attr` same, worse (vec3=12B).** At `max_whitewater=1M`, `whitewater_attr("velocity")` = 12MB. Combined with H3, `cache._frame_payload` (cache.py:66-74) always reads 6 particle scalars + 1 ww vec3 + 2 ww scalars = **~68MB PCIe/frame wasted**. **Fix:** as H3; also gate cache payload attr reads on what viewport displays.

**H5 · Performance — `sim/surface.py:219-220` low-res SDF rebuilt every substep when `whitewater_enabled`.** Bypasses `surface_update_cadence` (gates only `after_frame` path lines 226-230). Default `max_substeps=4` → 4× per-frame surface-build cost, no user signal. Each `build_sdf` = reset + splat + up to 3 smoothing-filter iteration loops. **Fix:** gate substep lowres rebuild on explicit flag / rebuild once per frame from prev SDF for ww advection.

**H6 · Error handling — `blender/scene_bridge.py:193-196` cache write failures swallowed by `print()` only.** Every other failure path in file calls `set_error()` + updates `fs.status`. This one: `except Exception: print(...)`. `fs.status` stays at "Frame N: ...ms". Disk-full/permission during long bake → every subsequent frame fails silently → hours later playback fails "missing/corrupt" with no hint live bake was failing. **Fix:** `set_error(scene, ...)` + `fs.status = f"Frame {frame}: cache write failed - {exc}"`; live sim continues (cache supplemental).

**H7 · Error handling — `operators.py:994-1005` `FLUIDSIM_OT_export_frame.execute` no try/except.** Every sibling operator wraps primary action in `try/except→report ERROR`. This one doesn't. `export_frame` can raise `FileNotFoundError` (`_require_existing_parent`), `RuntimeError` (`SurfaceExtractor.extract` on oversized dense span), `OSError`/`PermissionError` (`open()`). → Python traceback to console, not clean operator error. Inconsistent with `export_cache_sequence` immediately below. **Fix:** wrap in try/except → `report({"ERROR"})` + `CANCELLED`.

**H8 · Error handling — `io/export.py:25,50` OBJ/PLY direct `open()`, not temp-rename.** Sibling `io/cache.py` (write_frame:159-172, write_sequence_metadata:116-130) does `tempfile.mkstemp`→write→`os.replace` with except-cleanup. Export doesn't. Kill/crash/disk-full mid-write → truncated `.obj`/`.ply` that *parses valid* (header first) → downstream consumer silently drops geometry. **Fix:** factor atomic-write helper from cache.py, reuse in both export writers.

---

## MEDIUM (18) — abbreviated, deduped

**Readback / HostFrame (consolidates L1+L3+L6):**
- `frame_output.py:43,65,75` `velocities`/`particle_attr`/`whitewater_attr` use cached `host_count` while `positions`/`whitewater` refresh — stale slice latent bug, masked only by call order in `_solver_frame`. **Fix:** refresh once in `HostFrame.__init__` or in every loader.
- `frame_output.py:21-22` `invalidate_surface()` pops only `surface_mesh`; attr caches never invalidated — misleading name. **Fix:** rename `invalidate_surface_only` or `_values.clear()`.
- `frame_output.py:63-79` `particle_attr`/`whitewater_attr` catch only `AttributeError`, return silent `np.zeros(0)` — misconfigured attr (typo, disabled) shows blank colormap, no diagnostic. Comment also wrong about what's caught. **Fix:** widen except + log once per key.
- `frame_output.py:62-80` fallback shape `(0,)` for attr vs `(0,3)` for vectors — vec3 attr via `particle_attr` breaks downstream indexing.

**Surface:**
- `surface.py:774-783` no NaN/Inf sanitize on `verts`/`tris` before publish/export (consolidates L4+L5). Solver sanitizes particles/velocity but not SDF. NaN → Blender mesh NaN coords → OBJ `v nan nan nan`. Entry via BEFORE_PROJECT snippets (run pre-sanitize). **Fix:** `if not np.isfinite(verts).all(): return self._empty()` or sanitize in place.
- `surface.py:728-783` `SurfaceExtractor.extract` no docstring; return dtype of `tris` follows Warp MC (not guaranteed int32 across versions). **Fix:** docstring + `np.ascontiguousarray(tris, dtype=np.int32)`.
- `surface.py:362-366,393-397` Dilate/Erode 6-face neighborhood → ~1.7× axis bias (X feature erodes faster than diagonal). **Fix:** document as legacy FluidSim or extend to 26-neighborhood.

**Viewport publish:**
- `viewport.py:261-285` `_swap_points` allocates fresh pointcloud every call, no reuse branch like `_swap_mesh`; error path half-init risk. **Fix:** reuse on `pc_old.users==1` via `resize`; try/finally material transfer.
- `viewport.py:53-91,763-784` `_output_collection` rebuilds name-set + rechecks link every `output_object` call; `update_outputs` calls 3+/frame. **Fix:** resolve once per update pass / memoize on session.
- `viewport.py:60-63,75-79` collection/object link failures silently swallowed (`except RuntimeError: pass`) — output object created but invisible, no error. **Fix:** log once or fall back to `scene.collection`.
- `viewport.py:669-681` `remove_outputs` deletes objects/mesh/pc but never removes the output collection → orphaned on every `unregister()`. **Fix:** unlink + `bpy.data.collections.remove` when empty.

**Cache / export:**
- `cache.py:23-30` `_savez_fast_compressed` hardcodes `compresslevel=0` with `ZIP_DEFLATED`; `cache_compress` BoolProperty never plumbed → toggle is dead (silent false advertising). **Fix:** thread `fs.cache_compress`; `ZIP_STORED` when off.
- `cache.py:19,142-148,185-194` `CACHE_FORMAT_VERSION=1` hard-checked, no migration, undocumented schema (21 keys). First schema bump → every user cache unreadable. **Fix:** document v1 schema; `_migrate(old_data, old_version)` hook; forward-compat additive reads.
- `cache.py:201-211,214-224` `available_frames` name-only; `validate_contiguous_frames` reuses it → corrupt `.npz` reported present, fails at playback. **Fix:** `verify_integrity=True` kwarg via `zipfile.testzip`.
- `scene_bridge.py:247-275` mid-bake export failure leaves `[start,N)` frames + no sequence metadata; operator reports generic error, no last-ok count. **Fix:** write partial metadata in `finally`; surface `last_ok` in error.

**API:**
- `operators.py:982,1008` `export_frame`/`export_cache_sequence` no `poll()` — menu enabled with no solver alive. **Fix:** `poll` checks `session().active`.
- `operators.py:703-706` `snippet_validate` returns FINISHED on ERROR-level issues (sibling `snippet_test` returns CANCELLED).
- `operators.py:558-576` `snippet_apply` applies valid snippets then returns CANCELLED (violates "CANCELLED = no state change").
- `frame_output.py:8-80` `HostFrame` class contract undocumented (dtype/shape/fallback/thread-safety). 3 modules consume it.

---

## LOW / NIT (25) — highlights
`frame_output.py:26` `should_publish_surface` keys on absolute frame not elapsed (scrubbing irregular); boundary `frame<=1` breaks if `frame_start≠1`. `viewport.py:539` `round(2.5)→2` banker's rounding drifts display fraction up to half-stride. `viewport.py:159-164` `_nearest_values` silently decimates >120k samples — colormap loses detail at 2M particles, no signal. `surface.py:777-782` vertex map fully float32 → sub-mm jitter at large world offsets (origin 1000, h=0.001). `viewport.py:544-545` viewport point radius `0.25h` = half physical radius, undocumented. `viewport.py:244` `shade_smooth()` unconditional, no sharp-edge preservation. `cache.py:175-198` `read_frame` inner except misses `zlib.error`/`BadZipFile` on member-data corruption. `scene_bridge.py:284-292` cache playback error doesn't disable `cache_playback_enabled` → spam on persistent misconfig. `frame_output.py:33,53` redundant `refresh_host_count()` re-reads count post-step_frame (already refreshed). `operators.py:985,1011` export `bl_options` missing `REGISTER` (no Redo panel). `cache.py:133-148` `read_sequence_metadata` inconsistent (missing→None, corrupt→None, version→raise). `properties.py:729-734` `show_surface`/`show_particles`/`show_whitewater` no descriptions.

---

## UNCONFIRMED — needs visual check
**`sim/surface.py:773,315` MC winding vs negative-inside SDF.** `k_build_surface_sdf` builds `d = length(cw−p) − radius` (negative inside); `extract` passes `iso=0.0` to `wp.MarchingCubes.surface`. Warp MC tables produce fixed winding for `(field−threshold)`; if convention assumes positive-inside density field, mesh renders inside-out under backface culling + flips computed normals. **Vertex→world transform verified CORRECT** (no ±0.5 voxel shift — confirmed against Warp 1.12.0 source `marching_cubes.py:732`, node-indexed; extract offset `origin+(lo+0.5)·vs` matches cell-center convention). Only winding unconfirmed. **Action:** visual check under backface culling; if inverted, negate SDF into `mc.surface` or reverse winding in `extract`.

## Verified clean
- `solver.py:1377` `wp.synchronize_device` before all end-of-frame readback — single-threaded, post-step sync correct (L6).
- Cache write atomicity (`write_frame`/`write_sequence_metadata` temp-rename) — correct.
- `extract` empty/zero-leaf path via `_active_voxel_box`→`_empty()` — no crash.
- `handle_frame_change` playback wrap + live-sim release-on-failure — correct.
- Device flows unchanged `solver.device`→all arrays→`SurfaceExtractor` — no mid-stream device swap.
- `render.py` handler idempotent register/unregister, `__module__` filter survives reload.
- SDF union `atomic_min(point_radius)` = true signed distance to union-of-spheres.
- MULTIPLIER resolution mode preserves SDF sign convention.
- Publish contract: live path (cadence-gated) + cache path (hardcoded True) agree.

## Cross-cutting themes
1. **Stale-surface after particle mutation (H1)** — only PRK-specific logic bug; inject + custom snippets bypass `build_key`. Highest leverage single fix.
2. **Readback waste cluster (H3+H4+M-readback)** — ~68MB/frame PCIe for nothing; one-line device-slice fix, self-contradictory comment.
3. **Silent-failure cluster (H6+H7+H8+M-swallow)** — cache write, export raise, OBJ/PLY truncate, link fail, attr fallback all hide from user. Add `set_error`/`report`/log before tuning.
4. **NaN surface path (H2+M-sanitize)** — curvature CFL *causes* NaN; sanitize is defense. Fix CFL first, sanitize second.
5. **HostFrame contract fragility (M×4)** — refresh asymmetry, invalidate name, fallback shape, attr catch. One focused pass on `frame_output.py` clears all.

## Suggested fix order
1. H1 surface `build_key` particle-state — prevents wrong renders after inject.
2. H3+H4 device-slice readback — one line each, ~68MB/frame win.
3. H6+H7+H8 error surfacing — cache write / export / atomic write.
4. H2 curvature CFL clamp — prevents floating fragments.
5. M HostFrame contract pass (refresh + invalidate + fallback).
6. M cache schema doc + migration hook (before first schema bump).
7. VERIFY MC winding (visual check).
