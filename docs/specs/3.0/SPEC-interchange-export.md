# SPEC — Interchange export and solver-free playback

Owner module: `io/export.py`, new `io/vdb_writer.py`, new `blender/interchange.py`,
`blender/ui_stages.py` (Bake & Export stage), `blender/playback.py`,
`blender/output_publication.py`, `__init__.py` (registration), `blender_manifest.toml`.
Depends on: `SPEC-cache-v3.md` (sparse liquid SDF payload for VDB).

## Trap this spec prevents

1. **Blender does not ship `pyopenvdb`.** Any plan that says "use openvdb's Python
   module" fails at import. Warp can write NanoVDB (`wp.Volume.save_to_nvdb`), but
   Blender's Volume object reads only `.vdb`. The only self-contained route is a
   minimal pure-Python OpenVDB file writer. Treat it as a **spike with a kill
   criterion** (below), not a committed feature.
2. **Alembic export must go through Blender's exporter, not a hand-written writer.**
   `bpy.ops.wm.alembic_export` evaluates the scene per frame, which fires
   `frame_change_post`, which is exactly how cache playback publishes a frame today
   (`playback.handle_frame_change → play_cache_frame → update_outputs_from_cache_frame`).
   Export = cache playback enabled + exporter over the range with the output objects
   selected. Varying topology is fine: the exporter writes a non-homogeneous mesh.
   What breaks it: live mode on (the solver would step during export) and the
   `suppress_frame_handlers` guard. Force cache mode for the duration and restore after.
3. **Solver-free playback must not import Warp.** `sim/solver.py` imports `warp` at
   module top (line 16). The 2.1.1 kernel prewarm is already deferred (a `bpy.app.timers`
   callback in `__init__.py` with a lazy import of `runtime/warp_runtime.py`), so
   register itself is probably safe; the risk is transitive: any `blender/*` module that
   does `from ..sim import ...` at top level drags Warp in. Audit with the masked-import
   test below. Playback (`io/cache.py`, `output_publication.py`) is numpy only. The
   import graph must be split so a machine without CUDA can still register the
   extension, read a cache, publish meshes and render.

## A. Alembic mesh sequence (and points)
- New operator `fluidsim.export_alembic` in `blender/interchange.py`:
  1. Verify a verified-contiguous cache exists (`cache_schema.verified_contiguous_frames`).
  2. Set cache playback on, live off (`playback.set_cache_playback_enabled`,
     `set_live_enabled`); remember prior state.
  3. Select the surface output object (and whitewater/particle point-cloud objects when
     the user ticks them); call `bpy.ops.wm.alembic_export(filepath=…, start=…, end=…,
     selected=True, visible_objects_only=False, flatten=True, export_hair=False,
     export_particles=False, uvs=False, normals=True, vcolors=False, evaluation_mode='RENDER')`.
  4. Restore state in a `finally`.
- Point clouds: check on 5.2 whether the exporter writes `PointCloud` objects as Alembic
  points. If not, export them as a mesh of vertices (the output objects can already be
  switched to a mesh carrier; see `output_publication`). Record the finding in the
  spec's acceptance log.
- Attributes: Alembic carries `velocity` on the mesh when `export_attributes` is on;
  nothing else is promised in 3.0.

## B. OpenVDB (spike, kill criterion)
- `io/vdb_writer.py`: write one file with two grids, `surface` (FloatGrid, SDF from
  `liquid_sdf_sparse`, or from the render SDF when the pipeline can expose it) and
  `velocity` (Vec3SGrid, from the MAC field averaged to cell centres). File format
  version 224, 5-4-3 tree, uncompressed leaf buffers, no blosc/zlib, one grid per
  descriptor. Metadata: `file_voxel_size`, `class = level set` for `surface`.
- Kill criterion: if a bundled-tests import of the written file via
  `bpy.data.volumes.new` + `volume.filepath` does not show the right grid names, bounds
  and values on Blender 5.2 within **three working days**, the writer is cut from 3.0
  and the handoff records what failed. Do not spend the release on it.
- No read path. No NanoVDB.

## C. Solver-free playback
- Split `sim/__init__.py` so `from .sim import solver` is lazy; `blender/__init__.py`
  registers UI and playback first and only then tries `warp` inside a guarded
  `runtime_probe()`. On failure: status strip shows "Playback only — CUDA unavailable",
  Simulate stage is disabled with that reason, Bake stage shows Load/Export only.
- Kernel prewarm (2.1.1) runs only when the probe succeeds.
- The `wheels` entry in `blender_manifest.toml` stays (Warp still installs); only the
  import becomes optional. Check `tools/release/build_extension.py` smoke still passes
  with `--output C:/fsb` (`memory: smoke-test-needs-short-output-path`).
- `play_cache_frame` already avoids the solver; add a headless test that imports the
  package with `warp` masked out of `sys.modules` and plays one frame of a tiny cache
  committed under `tests/cache_io/fixtures/`.

## Acceptance
- Alembic: export dam_break frames 1–24, re-import with `bpy.ops.wm.alembic_import` into
  a fresh scene; per frame, vertex count equals the cache's `surface_verts` count and
  the bounding box matches to 1e-4. Add to `tests/acceptance/`.
- VDB (if kept): import the written file as a Volume; grid names, voxel size and the
  active voxel count match the cache frame; render one Cycles frame in the gate.
- Playback-only: `CUDA_VISIBLE_DEVICES=""` headless run registers the extension, plays
  a cache frame, renders it. Also run once on Linux (ties to the Linux certification).
- Gate hash unchanged (export is read-only over the cache).
