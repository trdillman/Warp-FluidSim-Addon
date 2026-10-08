# SPEC — Narrow-band particles

Owner module: `sim/solver.py` (step order, particle lifecycle), new `sim/narrow_band.py`,
`sim/surface.py` (reuse `k_propagate_surface_sdf_band`, `k_seed_surface_sdf_closest`),
`sim/particle_capacity.py`, `io/cache.py` (sparse liquid SDF payload), `properties.py`.
Depends on: `SPEC-cache-v3.md` (SDF payload, checkpoint v2), `SPEC-transfer-modes-apic.md`
(particle state layout).

## Trap this spec prevents

1. **Deleting interior particles destroys every per-particle attribute that lives
   there.** Color mixing (`sim/color_mixing.py`), `viscosity_from_particles`,
   particle temperature, `particle_time_offset` and the 33 exportable attributes are
   stored on particles. In 3.0 narrow band is **mutually exclusive** with those features;
   the UI disables the toggle with a diagnostic listing which feature blocks it. Do not
   attempt grid advection of attributes in 3.0.
2. **Surfacing from particles alone is no longer possible.** `SurfacePipeline` splats
   particles into an SDF. With narrow band the interior has no particles, so the render
   surface must be `min(particle_splat_sdf, grid_liquid_sdf)`. The grid liquid SDF must
   be written to the cache (`liquid_sdf_sparse`), and the on-demand render-grid rebuild
   in "Manual" mode must read it back.
3. **Whitewater seeding samples particle neighbourhoods** (`sim/whitewater.py`). Interior
   cells have no particles, so curvature/churn statistics computed from particle counts
   would read "empty" there. Compute them from the grid liquid SDF when narrow band is
   on, or restrict seeding to the band (acceptable: whitewater is a surface effect).
4. **The Safe Settings / capacity planner assumes `particles_per_voxel × fluid voxels`.**
   With narrow band the bound is `ppv × band voxels`; interior reseeding must never
   exceed the planned capacity. Capacity is computed from the band, with a 1.5× margin
   for splash.

## Algorithm (Ferstl et al. 2016, simplified)

Per substep, after P2G and before pressure:
1. Build grid liquid SDF `phi` on the sparse grid from `voxel_type & VOXEL_FLUID`
   (0.5h inside fluid cells, propagate outwards with the existing surface propagation
   kernels, reusing `band_propagation_passes`). This is a sim-side field, separate from
   the render pipeline's SDF.
2. Classify cells: `band` if `|phi| < band_voxels·h`, `interior` if `phi < −band_voxels·h`.
3. Grid velocity in interior cells is advected on the grid (semi-Lagrangian RK2 using the
   existing `TraceMode` helpers) instead of from particles; band cells keep the
   particle transfer.
4. After G2P, particles whose cell is interior are killed (`alive = 0`); band cells
   whose particle count < `ppv/2` are reseeded with jittered positions and grid
   velocity (and `affine = 0`). Reseed RNG state is checkpointed (cache v3).
5. Pressure, viscosity, surface tension operate on the full fluid region as today.

`band_voxels` default 3.0, UI range 2–6. Below 2 the surface pops; above 6 there is no
saving.

## Settings
- Domain group: `narrow_band_enabled: Bool` (default off in 3.0; demos decide per scene),
  `narrow_band_voxels: Float`. Tier RESET.
- Blocked when `color_mixing_enabled`, `viscosity_from_particles`, any Scalar/Temperature
  field source exists, or `export_attributes` is on. The admission check lives in
  `blender/simulation_admission.py` and the Setup stage shows the reason.

## Cache and playback
- `liquid_sdf_sparse` + leaf coords per frame (cache v3). Playback uses the stored
  surface mesh; only the render-grid rebuild needs the SDF.
- `stat_particles` becomes "band particles"; add `stat_liquid_voxels` so the status strip
  still conveys fluid volume.

## Demos
- dam_break: enable (deep water, the showcase). Expect ≥3× fewer particles after frame 20.
- crown_splash: off (thin sheets; band would cover everything anyway).
- paddle_stir: enable only if the swirl metric in `docs/paddle_spinup_calibration.md`
  stays within its tolerance.
- chromatic_vortex: off (color mixing).

## Acceptance
- Column of water at rest: free-surface height identical to full-band to 0.1h over 200
  frames; interior pressure hydrostatic to 1 %.
- dam_break Final: wall-clock per frame ≤ 0.6× of full-band at equal `resolution_divisions`.
- Particle count never exceeds planned capacity across the four demos (assert in the gate
  via `stat_particles ≤ planned_capacity`).
- Bitwise determinism 3/3 with reseeding on (RNG is counter-based, keyed by frame and
  cell index, never by thread arrival).
