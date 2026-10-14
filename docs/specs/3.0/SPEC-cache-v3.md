# SPEC — Cache v3

Owner module: `fluidSim_warp_blender51/io/cache_schema.py`, `io/cache.py`,
`io/checkpoint.py`, `blender/cache_identity.py`, `blender/production_bake.py`.
Depends on: nothing. Must land before narrow band and transfer modes (they add payload).

## Trap this spec prevents

1. `FrameEntry.from_dict` and `CacheManifestV2.from_dict` **reject unknown keys** and
   **require every dataclass field**. Adding a field without bumping
   `FORMAT_VERSION` makes every old manifest unreadable with a confusing
   "invalid frame entry keys" error, and adding it as an optional-with-default silently
   lets unstamped frames pass as valid. Do neither. Bump to 3, add the fields as
   required, refuse 2.
2. `source_signature_from_records` hashes the evaluated geometry **once, at bake
   start**. A keyframed collider that moves on frame 40 produces the same signature as
   one that does not. That is REVIEW-022 and CHANGELOG "Known limitations".
3. Frame `.npz` payloads are read with `frame_data.get(key, default)` in
   `blender/playback.py` and `output_publication.py`. New *required* keys must be read
   with `[]` and validated in `_read_v2_frame` (rename `_read_v3_frame`), not defaulted.

## Changes

### Manifest (`cache_schema.py`)
- `FORMAT_VERSION = 3`. `validate()` keeps the exact-match check.
- `CacheManifestV3` fields added (all required, all non-empty):
  - `source_schema_version: int` — from a new constant `SOURCE_SCHEMA_VERSION = 1` in
    `blender/sources.py`. Bump whenever a role/shape property is added or renamed.
  - `transfer_mode: str` — `"FLIP"` or `"APIC"` (`sim/enums.py: ParticleSampleType.name`).
  - `grid_spec: dict` — `{"resolution_divisions": int, "voxel_size": float,
    "domain_size": [x,y,z], "leaf_dim": 8}` as resolved by `quality_contract`.
  - `narrow_band: dict` — `{"enabled": bool, "band_voxels": float}`.
  - `platform: dict` — `{"os": ..., "gpu_name": ..., "driver": ..., "warp": ...}`.
    Informational; **never** part of any signature (bitwise identity is per GPU/driver,
    see user guide §12, and a cache must stay playable on another machine).
- `FrameEntry` fields added:
  - `source_frame_stamp: str` — sha256 over, for every source/collider record at
    *this* frame: `identity | matrix_world | geometry bytes` (same digest rules as
    `source_signature_from_records`, excluding `properties`, which the static signature
    already covers).
  - `whitewater_count: int`, `particle_count: int` — so resume and the UI can report
    counts without opening the `.npz`.

### Per-frame stamping (`blender/cache_identity.py`, `production_bake.py`)
- Add `source_frame_stamp(scene, depsgraph, frame) -> str`. It must evaluate the
  depsgraph **at the frame being written** (the bake loop already sets `scene.frame_set`;
  compute the stamp after `frame_set`, before `step`).
- Resume (`cache_schema.py: ResumeDecision`) gains a check: recompute the stamp for the
  last verified frame; mismatch → `ResumeDecision(reason="SOURCE_ANIMATION_CHANGED")`
  and the UI offers "Re-bake from frame N" where N is the first mismatching frame
  (walk backwards from the last frame, stop at the first match).
- Cost: one hash per source per frame. Mesh vertex bytes for a 1M-vertex collider are
  12 MB/frame; acceptable, but **cache the stamp per (object, frame) inside one bake
  session** so the pre-flight scan and the write don't hash twice.

### Frame payload (`cache.py: _frame_payload`)
- Add `liquid_sdf_sparse` (float32, active voxels), `liquid_sdf_leaf_coords` (int32 N×3),
  `liquid_sdf_voxel_size` — required when `narrow_band.enabled`, else empty arrays.
  Narrow band deletes interior particles, so a particle-only payload can no longer
  rebuild the surface (see `SPEC-narrow-band-particles.md`).
- Add `transfer_mode` (string scalar) and `particle_affine` (float32 N×9) **only when**
  transfer_mode is APIC and `export_attributes` is on; it is a checkpoint field, not a
  render attribute, so default off.
- `authoritative_payload_counts` must include the new per-particle array in the
  `"particles"` tuple so count validation catches a truncated write.

### Checkpoint (`checkpoint.py`)
- `CHECKPOINT_FORMAT_VERSION = 2`. Payload adds `particle_affine` (APIC) and
  `narrow_band_seed_state` (reseed RNG state, so resume is bitwise identical).
- Exact-match version check stays. A v1 checkpoint is refused, not migrated.

### Refusal path
- `detect_cache_format` returns 2 for old caches. `read_sequence_metadata` and the Bake
  stage show: "This cache was baked by FluidSim 2.x and cannot be played by 3.0. Re-bake,
  or open it in 2.1.1." Same wording pattern as the v1 refusal.
- `scene_key` and `_check_cache_scene_key` in `playback.py` stay as the cheap playback
  guard; recompute `scene_key_from_records` with `SCENE_KEY_VERSION` bumped so a 2.x key
  can never match.

## Acceptance
- `tests/cache_io/`: round-trip a v3 manifest; a manifest missing any new field fails
  `from_dict`; a v2 manifest directory yields the refusal message.
- Headless bake of `paddle_stir` (keyframed paddle): stamps differ between frames 1
  and 30; re-opening with the paddle keyframe moved by 5 frames yields
  `SOURCE_ANIMATION_CHANGED` at the right frame.
- Static-scene bake: all `source_frame_stamp` values identical; resume still succeeds.
- GUI gate hash unchanged when cache v3 lands alone (the payload adds arrays; the demo
  artifact hash covers the simulation outputs, not the manifest; confirm which files the
  hash covers in `tools/run_production_gate.py` before claiming this).

## Corrections found in implementation
Added by lane B after implementing this spec. The text above is left as written.

- Resume does a forward scan, not a backward walk: `cache_schema.first_changed_source_frame`
  stamps each committed frame in order and returns the first that differs.
- `ResumeDecision` has no `reason=` field. It is `ResumeDecision(allowed, resume_frame, code,
  message, first_changed_frame=None)`. The refusal is `code == "SOURCE_ANIMATION_CHANGED"`.
- Sparse SDF arrays: `liquid_sdf_sparse` is float32 `(N, 512)`, `liquid_sdf_leaf_coords` is
  int32 `(N, 3)`, and `liquid_sdf_voxel_size` is a float32 scalar (0.0 when disabled).
- `transfer_mode` and `particle_affine` are not written by lane B (they belong to lane G).
  The manifest records `transfer_mode="FLIP"` until then. `narrow_band` is
  `{"enabled": False, "band_voxels": 0.0}` until lane H fills it.
- The frame writer is renamed `write_frame_v3_atomic`; the v2 name is gone. Callers outside
  lane B that still use the old name or its keyword arguments must be updated.
- `source_frame_stamp` is stored in every frame NPZ as well as the manifest, and the reader
  checks the two agree.
- There is no per-(object, frame) stamp memo. The resume scan stamps each frame once and the
  bake stamps each frame once, so a memo had nothing to save.
- The stamp covers `matrix_world` and geometry only. It excludes `properties`, `name_full`,
  `library` and `role`, and does not depend on object order.
- Resume signatures (scene key, settings, static source signature) are compared against
  `manifest.frame_start`, not the frame the playhead is on. A change to a source at
  `frame_start` itself surfaces as a signature mismatch (restart), not
  `SOURCE_ANIMATION_CHANGED`.
- `CacheManifestV2` stays as an alias of `CacheManifestV3` so older importers keep working.
- Gate hash claim, checked: `analysis_report.json` is built by `tools/demos/analyze_demos.py`
  from `artifacts/demo_cache/<scene>/frame_NNNN.npz` and `series.json`, which
  `tools/demos/build_demo_scenes.py: run_scene` writes directly with `np.savez_compressed`
  after `bpy.ops.fluidsim.step`. That path does not go through `io.cache`, the cache
  manifest, stamps or checkpoints, so cache-v3 alone does not change it. The gate manifest
  also hashes `artifacts/performance_budget/summary.json`. This was established by reading
  the code; the GUI gate was not run.
