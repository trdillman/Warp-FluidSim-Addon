# Migrating from FluidSim 2.x to 3.0

3.0 changes how the simulation grid is authored and how caches are stored.
Settings migrate when a file opens. Caches do not: they must be baked again.

## Summary

| You have | What to do |
|---|---|
| A `.blend` saved by FluidSim 2.x | Open it in 3.0. Voxel Size converts to Resolution Divisions on load (section 1). |
| A cache baked by 2.x | Bake again. 3.0 refuses to play, resume or export it (section 2). |
| A solver checkpoint from 2.x | It is part of the 2.x cache, so it goes with it. A loose checkpoint file is refused. |
| A scene with ST-FLIP enabled | It opens without ST-FLIP. Re-bake it (section 3). |
| A machine without a CUDA GPU | Caches now load, play and export there (section 4). |
| Python scripts that set `voxel_size` | They still work. Prefer `resolution_divisions` (section 1). |

3.0 needs the same platform as 2.x: Blender 5.2 LTS with the bundled Warp 1.17.0.

## 1. Files saved by 2.x

### Resolution Divisions replaces Voxel Size

Before 3.0 the grid was authored as an absolute **Voxel Size**. It is now authored
as **Resolution Divisions**: the number of cells along the longest domain axis
(default 64). The cell size is derived:

    cell size = longest domain axis / Resolution Divisions

When a 2.x file opens, each Domain object that stores a voxel size and no
divisions gets `round(longest axis / voxel size)` divisions, with a minimum of 8.
The migration is idempotent, and a domain that already has divisions is left
alone.

What that means for your scene:

- The cell size moves to the nearest whole division. A 0.05 m cell on a 4 m domain
  is exactly 80 divisions. A 0.03 m cell on the 5.6 m dam_break domain is 187
  divisions, which gives 0.02995 m.
- **Resizing the domain now changes the cell size, not the cell count.** A scene
  that relied on a fixed cell size while its domain was scaled needs its
  divisions adjusted.
- The Setup stage and the Physics tab show the effective cell size and the padded
  grid (the solver pads each axis to whole 8 × 8 × 8 leaves) as read-only labels.
- Changing divisions needs a Reinitialize, as changing the voxel size did.
- `voxel_size` still exists as a derived property. Reading it gives the cell size,
  and writing it sets the nearest whole divisions. Presets that store a voxel
  size still apply. Quality tiers still scale the cell size, not the divisions.

### The grid can be smaller than before

Cell counts are rounded up to whole cells. In 2.x a cell size that had gone
through float32 sat a few parts in 10⁸ below the exact value, rounded up to one
cell too many, and padded an extra leaf on that axis (64 divisions could pad to
72). 3.0 ignores overshoot below one part per million of a cell. Scenes that hit
this get a smaller grid for the same settings. crown_splash's grid changes from
88 × 88 × 56 to 80 × 80 × 56 cells. Re-bake any scene whose grid changes.

### The settings split

Domain settings live on the Domain object and scene-wide settings live on the
scene. 3.0 finishes that split without changing any stored value: a dump of the
domain settings of the 24 shipped `.blend` files is identical before and after.
You will notice:

- The **Scene tab shows only scene settings** (lifecycle, live timeline, control,
  presets, outputs, display, materials, diagnostics, runtime statistics,
  benchmark and report). Every domain setting is on the **Physics** tab, which
  mirrors the former Scene panels. Height Fill, Persistent Attributes and
  Profiling are mirrored there too.
- **Open Full Settings** buttons open the Physics tab when a domain is active,
  and the Scene tab otherwise.
- Setting a value on the wrong group still raises an error that names the right
  group (see the 2.0 guide, section 5).
- Code that builds `SolverSettings()` directly sees new defaults for five fields,
  which now equal the Blender property defaults: `viscosity_constant` 0.05,
  `ww_bubble_drag` 4.0, `ww_spray_drag` 0.2, `speed_limit_percent` 0.5 and
  `churn_decay` 2.0. Blender scenes already passed the property value, so their
  results do not change.

### Where controls moved

| Control | Where it is now |
|---|---|
| Voxel Size | Resolution Divisions, in the Setup stage and the Domain & Particles panel of the Physics tab |
| Spatiotemporal Processing panel | Removed |
| Transfer (PIC, FLIP, APIC) | Simulation Physics > Time & Transfer, on the Physics tab |
| Pressure > Preconditioner | Simulation Physics > Pressure, on the Physics tab; shown for the PCG solver |
| Export Surface as Alembic | Bake & Export stage |

## 2. Caches and checkpoints

3.0 reads only **cache format 3** and **checkpoint format 2**. Nothing is
migrated or defaulted.

- **A 2.x cache is recognized and refused** with:

  > This cache was baked by FluidSim 2.x and cannot be played by 3.0. Re-bake, or
  > open it in 2.1.1.

  The Bake stage shows that message. Its Restart button replaces the old cache.
  The headless bake command (`--resume` in any mode) exits with code 14 and the
  same message before it does anything else. Alembic export refuses a 2.x cache
  with "The cache manifest is unreadable:" followed by the same message.
- A FluidSim 1.0 cache is still refused, as in 2.0.
- **A 2.x checkpoint is refused** ("unsupported checkpoint format 1; expected 2").
  A 3.0 checkpoint always stores the reserved `particle_affine` and
  `narrow_band_seed_state` slots, and a file without them is rejected.
- The scene key and the settings signature changed, so a 2.x cache could not match
  a 3.0 scene even if it were read.

### Animated sources are now checked

Every frame of a 3.0 bake records a stamp of every source and collider as it was
at that frame: its world matrix and evaluated geometry, not its panel settings or
names. In 2.x a cache frame recorded the settings only, and moving a keyframed
source or collider after a bake did not invalidate it.

In 3.0, **Resume** compares the stamp of every baked frame and refuses at the
first frame that no longer matches. The Bake stage names that frame and offers
**Re-bake from Frame N**, which keeps the frames before N and bakes the rest.
Checking costs one `frame_set` per baked frame, so Resume starts more slowly on a
long cache.

## 3. Removed features

**ST-FLIP is removed.** That covers the Spatiotemporal FLIP solver, its 14
settings, the Spatiotemporal Processing panel, **Variable Projection**, the
validation test for it and the checkpoint values it wrote. 2.0 had labelled it
Experimental.

- A saved scene that had ST-FLIP enabled opens without it. The scene simulates with
  the ordinary pressure projection, so the result differs from the ST-FLIP
  result. Re-bake it.
- The Time Offset particle colour mode is removed. It read a channel that only
  ST-FLIP filled, and that channel is now always zero. A scene that used it opens
  with the particle colour set to Material.
- To keep working with ST-FLIP, stay on 2.1.x.

## 4. Playback only mode

On a machine where Warp loads but no CUDA device is available, 3.0 enables the
extension instead of leaving it unusable:

- The status strip says **Playback only — CUDA unavailable**, with the probe's
  reason beneath it.
- You can load a baked cache, play it, and **Export Surface as Alembic**.
- The Simulate stage is drawn but disabled, and baking is hidden from the Bake
  stage.
- Kernel prewarm is skipped.

Baking and live simulation still need an NVIDIA GPU with CUDA 12 support. Bake on
a machine that has one, then copy the cache folder across.

## 5. New in 3.0, off by default

These do not change an existing scene.

- **APIC** (Transfer). FLIP stays the default. Switching between FLIP and APIC
  in a running simulation is allowed.
- **Multigrid** (Pressure > Preconditioner). Jacobi stays the default.
- **Export Surface as Alembic**. A mesh sequence of the liquid surface only.

See the user guide for each.

## 6. Results

Results change in three ways. Each is why a 2.x scene should be baked again:

- Scenes whose cell size migrates to the nearest whole division, and scenes that
  lose a padded leaf (section 1), simulate on a slightly different grid.
- ST-FLIP scenes lose ST-FLIP.
- The demo reference hash changed because of the grid changes above. Choosing
  APIC or Multigrid changes results further. With FLIP and Jacobi, adding APIC
  and Multigrid changed nothing: the demo bakes are bitwise identical to the
  build that changed the grid.
