# SPEC — Warp 1.18 certification, sparse marching cubes, VDB narrow-band wiring

Owner module: `runtime/warp_runtime.py`, `version.py`, `preferences.py` (Install Warp
download target and checksum), `tools/release/install_local.py`, `sim/surface.py`
(`SurfacePipeline` extractor around line 1271–1350, currently `wp.MarchingCubes` on a
dense sub-grid), `io/vdb*` writer. Depends on: nothing; lands first.

## Trap this spec prevents

1. **Driver floor surprise.** Warp 1.18 PyPI wheels are built with CUDA 13.4 and need a
   correspondingly new NVIDIA driver (the exact floor is in the 1.18 release notes;
   record it in `version.py` as `CERTIFIED_DRIVER_MIN` and in the Install Warp panel).
   The runtime probe must report a too-old driver as a named reason *before* download,
   and 1.17.0 must stay loadable for 4.0 through the existing "older than certified"
   path so users on older drivers are not stranded.
2. **Deprecated API noise.** `wp.MarchingCubes` is deprecated in 1.18; on 1.17 the new
   `warp.geometry` module does not exist. The extractor gets one adapter with two
   backends chosen at import: `warp.geometry.sparse_marching_cubes()` when available,
   the legacy dense path otherwise, both producing the same mesh for the same SDF
   (pinned by a test on a sphere SDF: vertex count and bounding box equal within 1e-5).
3. **Topology arrival-order nondeterminism.** `CONTEXT.md` already notes derived
   marching-cubes surfaces are unverified for repeatability. The sparse extractor must
   visit active leaves in slot order and emit triangles in a fixed per-cell order so
   two bakes give byte-identical surface meshes; add that to the determinism test.
4. **Losing the span cap semantics silently.** The dense path has a per-axis span cap
   (`surface_max_extraction_span`). With the sparse path the cap is unnecessary; keep
   the property, mark it "no effect with sparse extraction" in its description, and
   remove it in 4.1 with a migration.
5. **VDB writer still ignoring narrow band.** 3.0 left `liquid_sdf_*` in frames but not
   in the VDB export. Wire it: when the cache carries a narrow-band SDF, export it as
   the `surface` grid instead of re-splatting from particles; otherwise keep the
   current path. Pin with a cache-to-VDB round-trip test.

## Design

- Certification: `CERTIFIED_WARP_VERSION = "1.18.0"`, `MIN_WARP_VERSION = "1.17.0"`,
  distribution `1.18.0+cu13`; Install Warp downloads the 1.18 wheel with its SHA-256;
  Repair handles a 1.17 → 1.18 upgrade.
- Extractor: `SurfaceExtractor` protocol with `extract(sdf, grid) -> (points, indices)`;
  `SparseLeafExtractor` (1.18) seeds `sparse_marching_cubes` with the active-leaf cell
  list from `SparseGrid`, no dense allocation; `DenseExtractor` (legacy).
- Capacity planner: surface vertex capacity estimated from active leaves × a per-leaf
  bound, replacing the dense sub-grid estimate.

## Acceptance

- `test_warp_runtime_118.py`: probe reports driver floor; 1.17 loads with the warning;
  1.18 is certified.
- `test_surface_extractor_parity.py`: sphere and crown_splash frame-20 SDF produce
  equal meshes (within 1e-5) on both backends.
- `test_surface_determinism.py`: two bakes → identical surface mesh bytes.
- `test_vdb_narrow_band_export.py`: narrow-band cache exports the stored SDF.
- ABBA: surface phase time on crown_splash Final ≤ 0.7× of 3.2.1 (no dense pass).
- Full gate on Warp 1.18 with the new reference hash (shared with `SPEC-defaults-4.md`).
