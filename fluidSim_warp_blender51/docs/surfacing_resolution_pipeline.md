# Surfacing Resolution Pipeline

Historical feature note: version 1.0.233 separated simulation resolution from render/viewport surfacing resolution.

## Why this exists

The solver grid controls physics: pressure projection, particle/grid transfers, collisions, and whitewater support fields.  The render surface is a visualization/export product and can be reconstructed at a different resolution from the particle set.  The sharp render surface is reconstructed from the particle set on a grid finer than the simulation grid, followed by smoothing/denoising.

## New controls

Surface / Whitewater > Surface Mesh now exposes:

- **Resolution Mode**
  - `SIMULATION`: legacy behavior; render surface uses the simulation grid.
  - `MULTIPLIER`: render surface voxel size is `simulation_voxel_size / multiplier`.
  - `CUSTOM`: explicit surface voxel size.
- **Resolution Multiplier**: default `2.0`.
- **Surface Voxel Size**: used in Custom mode.
- **Leaf Capacity Scale**: allocation multiplier for the high-resolution sparse surface grid.  A 2x grid may need up to 8x as many leaves over the same active volume.
- **Max Dense Dim**: marching-cubes extraction safety limit.  If the active dense extraction span is larger than this, the extractor raises a clear error instead of silently clipping the surface.
- **Update Cadence**
  - `SUBSTEP`: legacy behavior, build every solver substep.
  - `FRAME`: default/recommended, build once per display frame.
  - `MANUAL`: lazy rebuild when a mesh is requested.
- **Keep Sim Surface Field**: also builds the simulation-grid `surface_sdf` for whitewater, snippets, diagnostics, and backwards compatibility.
- **Splat Radius**: minimum particle SDF splat support in surface-grid voxels. The solver also guarantees enough support for the configured particle radius.
- **Filter** controls are now active.  In prior builds `surface_filter_type`, `surface_filter_iterations`, and `surface_filter_strength` existed but were not consumed by the solver path.

## Internal architecture

The legacy simulation-grid surface remains:

```text
surface_sdf
surface_tmp
surface_extractor
```

The render/viewport surface can now use a separate sparse grid:

```text
render_surface_grid
render_surface_sdf
render_surface_tmp
render_surface_extractor
```

`surface_mesh()` extracts from the selected render surface.  OBJ export and cache writes therefore use the same mesh as the viewport.

## Recommended defaults

```text
Resolution Mode: Simulation for interactive performance; Multiplier for detail
Resolution Multiplier: 2.0
Update Cadence: Frame
Keep Sim Surface Field: On
Surface Max Dense Dim: 512 or 768 for larger scenes
Smooth Type: Box or Mean Curvature Flow
Smooth Iterations: 1 for viewport, higher for export tests
```

## Release validation

P0 consolidates surfacing acceptance into the installed-package NVIDIA GPU gate:

```bash
python ../tools/release/build_extension.py \
  --blender /absolute/path/to/blender \
  --output dist \
  --gpu-regression-platform linux-x64 \
  --json-report dist/gpu-build-report.json
```

The gate builds and inspects both split-platform extension archives, installs the selected archive into an isolated Blender profile, runs every test registered by `sim.validation` (including surface SDF parity), and advances a ten-frame preview simulation. The JSON reports under `dist/` are the release evidence. Historical benchmark JSON under `demo_outputs/` remains source documentation and is excluded from shipping archives.
