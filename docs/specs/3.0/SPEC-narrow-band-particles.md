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

## Implementation notes (verified, lane J)

Written at stop 1 against base `90d13ba` (release/3.0 after lane L). Every claim
about existing code cites `file:line` in `fluidSim_warp_blender51/`. Where this
section and the text above disagree, this section wins. Stop 1 ships only the
settings, the pure blocker rules, the Setup stage UI, the Initialize refusal, the
manifest block, `stat_liquid_voxels` and the strict-xfail acceptance tests in
`tests/solver/test_narrow_band_*.py`. The solver ignores the setting until stop 2.

### 1. Why the algorithm above is circular, and the fix: a persistent grid `phi`

The topology is rebuilt from particle positions only
(`sim/solver.py:1109-1115`, `g.mark_points(self.particles.position, ...)`).
`VOXEL_FLUID` is set in P2G only where `voxel_counts > 0`
(`sim/kernels_fluid.py:589-605`, launched at `sim/solver.py:1323-1328`). Both
`voxel_counts` and `voxel_type` are zeroed every substep
(`_clear_grid_fields`, `sim/solver.py:1181-1196`). So once interior particles are
killed, the interior leaves drop out of the topology and the FLUID flag vanishes
there; pressure then sees only the band and the liquid collapses.

The fix is a grid liquid level set `phi` that is the **only** persistent record of
the interior. It lives on the sparse grid beside `churn`, which is today's only
persistent grid field and the precedent for carrying a field across a rebuild
(`ScalarField` at `sim/solver.py:574`, remap at `sim/solver.py:1166-1173` via
`k_remap_leaf_field`, `sim/solver.py:203-224`). Convention: `phi < 0` is liquid,
in metres, clamped to `[-L, +L]` with `L = (band_voxels + 2) h` so semi-Lagrangian
smearing cannot make large values.

Per substep, `phi` is maintained in three moves:

1. **Carry.** On a topology rebuild, remap `phi` to the new leaf table. The existing
   `k_remap_leaf_field` writes `0.0` into leaves that did not exist before
   (`sim/solver.py:221-222`), which would read as "on the surface". A new
   `k_nb_remap_leaf_field_default` writes `+L` instead.
2. **Advect.** Semi-Lagrangian: `phi_new(x) = phi_old(x - u dt)`. No grid
   self-advection kernel exists, so reuse the `trace_position` `@wp.func`
   (`sim/kernels_fluid.py:873-895`) with a negated `dt` and
   `TraceMode.MID_POINT` (`trace_mode == 2`, `sim/kernels_fluid.py:888`), which is
   the RK2 member; there is no RK2 enum. The velocity read is the previous
   substep's post-pressure grid velocity (section 4).
3. **Re-union with the particle splat near the band.** Build the particle splat
   with the existing `Surface.build_sdf` (`sim/surface.py:346`). Wherever the splat
   is within `(band_voxels + 1) h` of the surface it **overwrites** `phi`
   (`phi = splat`). Deeper than that `phi` keeps its advected value. The union
   only ever moves the zero crossing where particles exist to witness it.

How it stays mass-stable (Ferstl et al. 2016, "Narrow band FLIP for liquid
simulations"): the zero crossing of `phi`, the actual free surface, is owned by
particles in the band and is re-derived from them every substep, so it cannot
drift. In the interior, semi-Lagrangian advection of a signed distance only
smears magnitudes, and the interior keeps a sign test (`phi < 0`) that never
approaches zero because the band is at least `band_voxels` cells thick. Pressure
is divergence-free, so interior volume does not change; a flux across the inner
edge of the band is exactly a particle crossing the band edge, which the kill and
reseed step (section 5) keeps balanced. Clamping to `[-L, +L]` bounds the error
growth. Sheets thinner than `2 * band_voxels` have no interior, so they stay
fully particle-owned.

### 2. Per-substep insertion points (by method)

`step_frame` is at `sim/solver.py:1908-2080`; the substep loop begins at `:1939`.

| Where | Change when `narrow_band_enabled` |
|---|---|
| `_rebuild_topology` (`:1109`), after the `old_leaf_table` copy (`:1112`) | also snapshot `old_leaf_coords` and `old_num_leaves` (a new device copy; the grid overwrites both in `begin_topology`) |
| `_rebuild_topology`, after the particle mark (`:1114-1115`) | `k_nb_mark_leaves_from_phi` flags every old leaf holding a voxel with `phi < 0`, so interior leaves survive the rebuild |
| `_rebuild_topology`, next to the churn remap (`:1166-1173`) | remap `phi`, and the persistent interior velocity (section 4), with the default-value remap |
| `_clear_grid_fields` (`:1181`) | unchanged; persistent state lives outside the cleared arrays |
| `_p2g` (`:1285`, called `:1971`) then `_save_velocities` (`:1348`, called `:1972`) | new `_narrow_band_grid_stage()` between them (section 3), so the FLIP delta (`vel - vel_saved`) is taken against a velocity that already includes the interior |
| `_pressure` (`:1511`, called `:1991`) then `_sanitize` (`:1993`) | new `_narrow_band_store_velocity()`: persist the post-projection velocity for the next substep |
| `_lifecycle` (`:1649`, called `:1998`) | kill and reseed, before `p.compact` (section 5) |
| `Surface.ensure_solver_sdf_current` (`sim/surface.py:526`) | union `min(splat, phi)` (section 6) |

Step order, for the record: surface tension is inside `_forces` (`:1356`, called
`:1976`) and viscosity is `_viscosity` (`:1460`, called `:1978`), both before
`_pressure`. So the grid stage runs after P2G and before `_forces`, and
`_save_velocities` follows it.

### 3. Topology and `VOXEL_FLUID` from `phi`

`_narrow_band_grid_stage()`, in order:

1. advect `phi` (move 2 above);
2. build the splat SDF and apply move 3;
3. `k_nb_set_fluid_from_phi`: `voxel_type |= VOXEL_FLUID | VOXEL_ACTIVE` where
   `phi < 0` and the voxel is not `VOXEL_COLLISION`. This is additive, exactly the
   rule `k_particle_to_grid_active` already follows (`sim/kernels_fluid.py:598-605`),
   so the existing particle-count rule keeps working in the band;
4. `k_nb_classify`: write a per-voxel class (0 outside, 1 band: `|phi| < band h`,
   2 interior: `phi < -band h`) and atomically count liquid voxels with an integer
   `atomic_add` (order independent), which feeds `stat_liquid_voxels`.

### 4. Interior grid velocity

`_clear_grid_fields` zeroes `vel.u/v/w` every substep (`:1181-1196`), and
`k_gather_grid_data` (`sim/kernels_fluid.py:268`) fills a face only from particles.
A face with no particles in its stencil would read 0. So:

- two persistent MAC triples, `nb_vel` (u, v, w) and `nb_vel_tmp`, are carried
  across the rebuild with the default-0 `k_remap_leaf_field` (a new face has no
  history), like `churn`;
- `_narrow_band_grid_stage()` advects `nb_vel` semi-Lagrangian (the same negated-`dt`
  `trace_position`) into `nb_vel_tmp`, then `k_nb_fill_interior_velocity` writes it
  into `vel.u/v/w` on every face whose two adjacent cells are interior (class 2) or
  whose MAC weight (`vel.wu/wv/ww`) is zero and that borders a `phi < 0` cell. Band
  faces keep the particle transfer;
- `_narrow_band_store_velocity()` copies `vel.u/v/w` into `nb_vel` after `_pressure`
  and `_sanitize`, so the persisted field is divergence-free.

### 5. Kill and reseed (`_lifecycle`)

Sequence, all inside `_lifecycle` (`:1649-1665`) and gated:

1. `_index_particles()` first (`:1260`). Particles moved in G2P, so the
   start-of-substep `voxel_counts` are stale.
2. `k_nb_kill_interior`: `alive = 0` for particles whose cell has `phi < -band h`.
   Killing happens before `p.compact` (`:1664`).
3. `k_nb_reseed_demand`, an exclusive scan, then `k_nb_reseed_commit`: for each band
   cell (`class == 1`, `phi < 0`) with `voxel_counts < ppv / 2`, emit
   `ppv - have` jittered particles, velocity sampled from the grid, `affine = 0`.
   The demand/scan/commit shape is the one the top-up sources already use for
   determinism (`sim/source_kernels.py:467-519`), so slots follow voxel order and
   not thread arrival.
4. `p.compact`, then `_index_particles()` again so `voxel_counts` is current for
   `_relax` and the next substep.

**Capacity guard.** There is no `planned_capacity`. The capacity is the authored
`settings.max_particles`, checked at planning in `sim/capacity_planner.py:401-420`
(`PARTICLE_CAPACITY`). The commit kernel carries its own limit, as in
`sim/source_kernels.py:585-587` and `:1837-1841`: `slot_limit = max_particles - count`;
past it the kernel stops emitting and sets a device flag `nb_overflow[0] = 1`. The
host reads it in the end-of-frame stats block that already syncs
(`sim/solver.py:2053`) into new `FrameStats.narrow_band_reseed_overflow`, and the
status string reports it. Reseed is under-filled and loud, never silent and never
over capacity. Gate assertion: `stat_particles <= max_particles` and the overflow
flag is 0.

**RNG.** `wp.rand_init(seed, key)` is counter-based. The sourcing seed starts at the
constant `12345` and advances once per transaction (`sim/source_execution.py:607`),
so its jitter repeats; do not reuse it. The reseed seed is
`seed = (frame * 1000003 + substep_index * 7919 + 0x9E3779B1) & 0x7FFFFFFF` and the
key is `flat_cell_index * 31 + k`, with `k` the per-cell sample index. No
thread-arrival dependence. Checkpoint: write a one-element `uint32` array
`[reseed_epoch]` (count of reseed launches) into the reserved `narrow_band_seed_state`
slot (`io/checkpoint.py:28-31`), written by `export_checkpoint_state`
(`sim/solver.py:828`) and restored by `import_checkpoint_state` (`:909`); the seed
itself is recomputed from the checkpointed `frame` and substep. The persistent `phi`
and `nb_vel*` are added to the generic `field_*` checkpoint arrays
(`_CHECKPOINT_FIELD_NAMES`, `sim/solver.py:495-500`) only when the mode is on.

### 6. Union-SDF consumers

`Surface.sdf` (exposed as `solver.surface_sdf`, `sim/solver.py:1084`) feeds more
than the spec lists. With an empty interior each sees a false free surface at the
band's inner edge:

- surface tension, `sim/surface_tension.py:480`, `:501`, `:524`;
- whitewater candidate marking, `sim/whitewater.py:225-263`;
- whitewater foam, bubble and spray typing, `sim/whitewater.py:412-462`;
- whitewater advection, `sim/solver.py:1759` (`self.surface_sdf.data`) and
  `sim/whitewater.py:807-819`;
- the render surface when the render grid is the simulation grid
  (`sim/surface.py:561-565`).

One change covers all of them: at the end of `ensure_solver_sdf_current`
(`sim/surface.py:526-548`) launch `k_nb_union_into_solver_sdf`, `sdf = min(sdf, phi)`.

**One consumer the brief did not list.** When the render grid differs from the
simulation grid (`surface_resolution_mode` multiplier or custom,
`sim/surface.py:561-566`), `render_sdf` is built directly from particle positions
with no solver SDF involved, so its interior is empty too. It needs
`k_nb_union_into_render_sdf`, which samples `phi` trilinearly at each render voxel
centre. See open question 2.

### 7. Source top-up guard

The four Top-up paths compute `want - have` from `voxel_counts`:
`sim/source_kernels.py:484` (demand), `:578-582` (commit), `:741-746` (mesh surface)
and `:1831` (volume grid). Interior cells inside a source region would re-emit every
substep. Each gets two new arguments, `interior_phi: wp.array(float)` and
`credit_below: float`; a cell with `interior_phi < credit_below` counts as
satisfied. With the mode off the solver passes a dummy array and
`credit_below = -1.0e30`, so the branch is never taken and the arithmetic is
untouched.

### 8. Checkpoint and cache (producer side only)

Already reserved and validated: `LIQUID_SDF_KEYS`, `validate_liquid_sdf`
(`io/cache.py:347`), the `liquid_sdf=` parameter on `write_frame_v3_atomic`
(`io/cache.py:600`), the manifest `narrow_band` block (`io/cache.py:548`) and the
reserved checkpoint slot. Producer changes in stop 2:

- `production_bake.py` passes `liquid_sdf=` from the solver: the sparse `phi` leaves
  (float32 `(N, 512)`), their `int32 (N, 3)` coordinates and the voxel size.
  Stop 1 already reads `enabled` and `band_voxels` into the manifest
  (`_narrow_band_record`);
- the "Manual rebuild reads phi from the cache" bullet is **dropped**: playback uses
  stored meshes, so nothing consumes it.

Hazard in stop 1: the manifest now carries `enabled = True` when the setting is on,
and `write_frame_v3_atomic` then requires a populated `liquid_sdf`
(`io/cache.py:616`, `:354`). With the solver still ignoring the setting a bake fails
loudly at the first frame. The mode is off by default, but see open question 1.

### 9. New kernels (bounded list)

All in a new `sim/kernels_narrow_band.py`, used by a `NarrowBand` class in
`sim/narrow_band.py` that imports it lazily (the pure rules stay importable without
Warp). The signatures are the contract the tests rely on.

- `k_nb_remap_leaf_field_default(old, old_leaf_table, leaf_coords, num_leaves, leaves_dim, default_value, new)`: remap with an outside default.
- `k_nb_mark_leaves_from_phi(old_phi, old_leaf_coords, old_num_leaves, leaves_dim, leaf_flags)`: keep interior leaves in the topology.
- `k_nb_advect_scalar(phi, u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, origin, voxel_size, dt, limit, phi_out)`: semi-Lagrangian `phi`.
- `k_nb_advect_velocity(nb_u, nb_v, nb_w, <same grid args>, dt, out_u, out_v, out_w)`: semi-Lagrangian MAC velocity.
- `k_nb_union_band(phi, splat_sdf, num_leaves, band_outer, limit)`: `phi = splat` near the surface.
- `k_nb_set_fluid_from_phi(phi, num_leaves, voxel_type)`.
- `k_nb_classify(phi, num_leaves, band, cls, liquid_count)`.
- `k_nb_fill_interior_velocity(nb_u, nb_v, nb_w, cls, wu, wv, ww, num_leaves, u, v, w)`.
- `k_nb_store_velocity(u, v, w, num_leaves, nb_u, nb_v, nb_w)`.
- `k_nb_kill_interior(position, count, phi, leaf_table, voxels_dim, leaves_dim, origin, voxel_size, kill_depth, alive)`.
- `k_nb_reseed_demand(cls, phi, voxel_counts, num_leaves, ppv, demand)`, an exclusive scan, and `k_nb_reseed_commit(demand_scan, cls, leaf_coords, num_leaves, origin, voxel_size, seed, slot_base, slot_limit, <velocity sample args>, position, velocity, alive, count, nb_overflow)`.
- `k_nb_union_into_solver_sdf(sdf, phi, num_leaves)` and `k_nb_union_into_render_sdf(render_sdf, <render grid args>, phi, <sim grid args>)`.

Existing kernels edited: the four top-up kernels (extra arguments, section 7), and
nothing else.

### 10. How the off path stays bitwise identical

- Every new launch, allocation, buffer, checkpoint field and `FrameStats` write sits
  behind `settings.narrow_band_enabled`. The `NarrowBand` object is `None` when the
  mode is off, so no code path reads it.
- Nothing is allocated before the existing arrays, and the memory-safety estimate
  (`sim/memory_safety.py`) adds the narrow-band bytes only when on.
- The four top-up kernels gain arguments. Warp recompiles them, but with the
  disabled dummy the new branch is not taken, so the floating-point operations are
  the same ones in the same order. The demo hash is the proof; the orchestrator
  checks it.
- `_CHECKPOINT_FIELD_NAMES` is a module tuple; the extra names are appended in an
  instance copy only when on, so the off checkpoint is byte-identical.
- Stop 1 changes no solver code. The only `sim/` edits are the Warp-free
  `solver_settings_model.py` (two fields), `frame_runtime.py` (`liquid_voxels`,
  reported as `liquidVoxels` in `FrameStats.as_dict`, always 0) and the new pure
  `sim/narrow_band.py`.

### 11. Metric note (paddle)

`tools/demos/analyze_demos.py:480-511` sums `x*vy - y*vx` over particles. With the
mode on, interior liquid holds no particles, so that sum shrinks and its meaning
becomes "band angular momentum". Do not enable paddle_stir until the metric is
re-based on grid velocity or the swirl tolerance is re-derived.

### 12. Open questions for the orchestrator

1. A bake with the mode on fails at the first frame in stop 1 (section 8). Accept
   until stop 2, or force `enabled: False` in the manifest until the producer lands?
2. Render grid != sim grid needs `k_nb_union_into_render_sdf` (section 6). In scope
   for stop 2, or restrict narrow band to `surface_resolution_mode == SIMULATION` and
   add that to the blockers?
3. Persisting `phi` and `nb_vel*` adds `field_*` arrays to the checkpoint (four extra
   float fields on active voxels). These are generic arrays, not a schema change.
   Acceptable?
4. `FrameStats.narrow_band_reseed_overflow` is new stats surface. Wanted in the gate
   output?

### 13. As built (stop 2) and deviations from the notes above

Stop-1 questions 1-4 were answered: the producer lands, the render-grid union kernel
is in scope, the checkpoint gains the four `field_narrow_band_*` arrays (only when the
mode is on), and the overflow stat is a solver-side `FrameStats` field.

Deviations from sections 1-10:

- The stateful class is `NarrowBand` in `sim/narrow_band_state.py` (the pure helpers
  in `sim/narrow_band.py` may not import bpy or warp).
- Union rule: `phi` is used only where `phi < -h`; otherwise the particle splat wins.
  Where `phi_adv < -pin_depth` it is kept as is; `pin_depth = max(1, band - 1) * h`
  (not `(band + 1) * h`, which left air holes between the kill depth `band * h` and the
  pin depth). A cell inside the splat but not yet pinned gets `-limit` and is refined by
  `ceil(band) + 2` inward Eikonal sweeps. Solid cells get `+limit`.
- The Eikonal sweep ignores solid and unallocated neighbours (a wall is not a free
  surface).
- Reseed RNG key is `wp.rand_init(seed, flat * 64 + k)`; target density is capped at 64
  per voxel; demand only for band cells with `phi <= -h` and `count * 2 < ppv`.
- Source top-up guard (section 7) is a reversible count credit
  (`k_nb_adjust_interior_counts`) around `_source`, not an edit of the source kernels.
- Interior velocity is advected at the start of the substep (`prepare_grid`) and `phi`
  is updated at the end (`update_phi`, in `_lifecycle` before `compact`).
- The render grid, when different from the solver grid, gets
  `k_nb_mark_render_leaves` in `rebuild_topology` and `k_nb_union_into_render_sdf`
  after `build_sdf`.
- Checkpoint: the reserved `narrow_band_seed_state` slot holds `[reseed_epoch]`
  (uint32); written only when the mode is on.
- Cache: `production_bake._liquid_sdf_arrays` writes `NarrowBand.leaf_phi()` as the
  `liquid_sdf_*` arrays and logs a reseed overflow.
- Memory estimate adds 56 bytes per allocated voxel when the mode is on.
- Known limit: under the default Jacobi preconditioner the full-band column itself
  compresses (0.75 to about 0.56 in 30 frames); the hydrostatic acceptance runs under
  Multigrid.
