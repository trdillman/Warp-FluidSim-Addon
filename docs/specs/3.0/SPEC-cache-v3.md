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
