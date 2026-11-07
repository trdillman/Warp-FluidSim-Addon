# Warp 1.19 upgrade and meshing options

Date: 3 November 2026; installer follow-up 4 November. Status: **Warp 1.19.0
has not passed the full GPU gate**. The PR47 author reports a Windows Blender
5.1 attempt failing whitewater truth on both 1.19 and 1.18; this is not Blender
5.2 qualification or proof of a dependency regression. Historical "certified"
identifiers and status tokens in code name the locked gate target, not a
finished qualification. User-facing installer and runtime labels say "locked".

## What changed in the repo

- `CERTIFIED_WARP_VERSION` is now `1.19.0` (`1.19.0+cu13`). The wheel lock
  (both copies), `requirements-dev.txt`, `tools/release/build_extension.py`,
  notices, README, user guide and technical report follow it.
- Wheel hashes and sizes come from PyPI; the Linux wheel was downloaded and
  its SHA-256 checked against PyPI's value. The Windows wheel is pinned from
  PyPI metadata only.
- 1.19.0 is the same CUDA 13.4 build with the same R580 driver floor.
- 1.17.0 and 1.18.0 still load (1.18.0 now shows the "older than locked"
  warning). `MIN_WARP_VERSION` stays at 1.17.0.
- Warp 1.19 removed arguments that 1.18 had marked deprecated on the dense
  marching cubes class (`max_verts`, `max_tris`, `device`). The add-on already
  calls the new class with only `nx, ny, nz`, so no solver code changed. The
  old `wp.MarchingCubes(max_verts=...)` call remains only in the 1.17 fallback.

## What was run

On Warp 1.19.0 (CPU only, no CUDA device in this environment): the full
non-Blender suite passes except one test that needs Blender's `bpy`
(`test_frame_output.py::test_output_cost_counters_separate_all_four_phases`,
which fails the same way on the old code). The 39 CUDA-only surface tests were
skipped. The GPU production gate still has to be run on the Windows host.

## The 1.18 and 1.19 geometry features, as they apply here

| Feature | Since | Fit for this add-on |
|---|---|---|
| `sparse_marching_cubes_from_cells` | 1.18 | **Already used** by the `sparse` surface backend. Cost follows the surface area, not the domain volume. |
| `sparse_marching_cubes`, `lipschitz_octree` | 1.18 | Meant for a formula-style field. The liquid surface is built from particles on a tiled grid, so it already has its own cell list. Could help only for analytic shapes (source shapes, collider SDF previews). |
| `IsoSurfaceMarchingCubes`, `IsoSurfaceBase` | 1.18 | Already used by the `dense` fallback. |
| `IsoSurfaceNets` (triangle or quad output) | **1.19** | Promising for a cleaner liquid mesh. See below. |
| `swept_volume_mesh` / `_field` / `_sdf` | 1.18 | Possible later use: a collider that moves far in one frame could be swept into a solid region so fast paddles cannot slip past liquid. Not tried. |
| `tri_tri_adjacency`, `delaunay_edge_flip` | 1.18 | Mesh clean-up helpers. The edge flip works on 2D points, so it does not fit the 3D liquid surface. |
| `wp.mesh_query_sphere`, BVH sphere queries | 1.17 | Collider contact lookups. Already available to the collider code. |

## Surface Nets vs. marching cubes (quick trial)

Setup: a 64³ wobbly sphere field, run on CPU, same grid, same surface level.

| | Marching cubes | Surface Nets |
|---|---|---|
| Vertices | 6,390 | 6,392 |
| Triangles | 12,776 | 12,780 |
| Closed, no holes | yes | yes |
| Average triangle quality (1 = perfect) | 0.68 | 0.80 |
| Sliver triangles (quality < 0.1) | 4.8 % | 0 % |

So for the same triangle count Surface Nets gives much more even triangles,
which should render and shade the liquid more smoothly and give the later
smoothing and cache steps cleaner input. It can also output quads.

Catches:

- It is dense only. It needs the same padded sub-box the `dense` fallback
  builds, so it loses the sparse backend's "pay only for the surface" speed
  and is limited by Surface Max Dense Dim.
- The trial is a synthetic shape on CPU. Speed and look on a real dam break
  need a GPU run.

## Suggested next steps (not done)

1. Add `nets` as a third surface backend that reuses the existing dense
   sub-box path, then compare it with `sparse` on `dam_break` and
   `crown_splash`: frame time, triangle quality, and whether the mesh bytes
   stay repeatable run to run (the marching-cubes repeatability claim in
   `AGENTS.md` is not known for Surface Nets).
2. If the dense cost is too high, keep `sparse` and add a cheap vertex
   relaxation pass on its output instead.
3. Separately, the sparse path copies vertices back to the CPU every frame
   (`.numpy()` in `sim/surface.py`). Keeping them on the GPU is the bigger win
   and belongs with the native GPU interop work the Owner reactivated.
