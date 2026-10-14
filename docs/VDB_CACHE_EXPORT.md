# Unreleased baked-cache VDB export

Based on 04d732a; this candidate is not part of the published 3.0.0 release.

## Artist workflow

Bake with Narrow-Band Particles enabled. In Bake & Export (or Playback-only), choose **Export Liquid Level Set as VDB**. Select an existing parent directory and a new sequence folder name. Output is `<parent>/<sequence>/frames/surface_000001.vdb` and subsequent cache frame numbers; `export.json` records source manifest identity, file hashes, and per-frame sample origins.

This exports one float liquid surface level-set grid named `surface`. It does not export smoke/density, velocity, particles, or an Alembic replacement. Full-band caches lack this SDF and are refused with re-bake guidance. No narrow-band setting is enabled automatically and no simulation default changes.

Only committed, checksum-verified, contiguous cache frames are accepted. The exporter checks frame data and manifest identity before publication and rejects changed caches, invalid/missing grids and out-of-range requests. It refuses every existing destination, including empty folders; there is no overwrite mode. A private sibling directory is staged first and published under an exclusively created destination. A failed writer never publishes a partial sequence or recursively deletes the selected destination.

## Coordinates and runtime independence

New narrow-band frames additionally store `liquid_sdf_origin`, a finite float32 three-vector containing the solver grid's **corner** origin. The original required SDF triple and cache format number 3 stay unchanged. This optional field is covered by the existing frame checksum; older 3.0 readers ignore/retain extra arrays without requiring a format upgrade. SDF samples are cell-centred: the VDB sample transform uses the cached corner plus half a cached voxel. Leaf coordinates remain domain-relative 8-voxel leaf indices.

These self-contained caches export with no domain object, no Warp installation, and no CUDA device. Moving or editing the current scene does not move a baked cache. Older v3 frames without this optional field require an unchanged saved scene and an exact canonical baked settings signature to authenticate their unstored origin. That legacy signature includes the device, so legacy export can be refused on a different/no-runtime machine; restore the original scene/runtime or re-bake. Never guess a legacy origin from the current domain or assume world zero.

## Acceptance and limits

- `tests/cache_io/test_cache_vdb_export.py`: transaction protection, corrupt cache, range, origin schema, legacy fallback and a fresh process that rejects every Warp import.
- `tests/acceptance/blender_cache_vdb_export_acceptance.py`: actual public operator on a synthetic committed moving-sphere cache, independent Blender OpenVDB read-back and world-space mesh bounds.
- `tests/acceptance/blender_vdb_production_bake_acceptance.py`: actual two-frame CUDA ProductionBakeBackend output, origin capture, public export and independent Blender reader. Both Blender scripts are wired into the production gate and remain distinct from full release qualification.

The writer is the existing narrow float-level-set implementation, not a general OpenVDB library. Export is synchronous and checks full cache-frame integrity, so large sequences may take time. Foreground file-browser interaction, package installation, broader DCC interchange, and a full GUI release gate require their own evidence; they are not established by these focused tests.
