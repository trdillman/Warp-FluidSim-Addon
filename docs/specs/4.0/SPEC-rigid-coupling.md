# SPEC — Two-way rigid-body coupling (Dynamic colliders)

Owner module: new `sim/rigid_coupling.py` (state, integrator, publication record) and
`sim/kernels_rigid.py` (boundary-sample force/torque gather). Hooks:
`sim/solver.py: step_frame` (one call after pressure projection, before G2P),
`blender/colliders.py` (Dynamic flag, mass/density/inertia, initial velocity),
`blender/output_publication.py` (pose publication as keyframes, owned), `io/cache*`
(cache v3.1 frame record), `blender/interchange.py` (poses ride along in Alembic because
they are object animation). Depends on: Multigrid default (`SPEC-defaults-4.md`) for the
buoyancy test; nothing else.

## Trap this spec prevents

1. **Driving Bullet.** Python has no API to apply per-frame forces to Blender's rigid
   body world, and the depsgraph evaluates it on its own clock. FluidSim owns the 6-DoF
   state of a Dynamic collider; on admission the object's `rigid_body` (if any) is set
   to `KINEMATIC` with a reason string, and the pose is published as keyframes on the
   object by Output Publication so it renders, exports and plays back with no solver.
2. **Atomics on the bake path.** Force/torque accumulation over boundary samples must
   be a segmented gather per body (fixed sample order), not `wp.atomic_add`, or bakes
   stop being bitwise repeatable. Samples are generated once per collider at a fixed
   spacing (≈ 0.5 voxel) in the body frame and transformed per substep.
3. **Double counting the pressure.** The fluid already sees the body as a solid in the
   projection (Neumann faces). The force on the body is the pressure impulse on the
   solid faces of its voxels: `F = Σ_faces p·A·n` over faces with a fluid neighbour,
   read from the projected `q` field (`q = p·dt/ρ`), plus a drag term from the relative
   velocity at the samples. Do not add a particle-sphere penalty force on top.
4. **Stability with light bodies.** Explicit coupling with a body lighter than its
   displaced fluid oscillates. Clamp the per-substep impulse to `m·Δv_max` and use
   the substep count already chosen by the CFL logic; document that the demo cork has
   density ratio 0.3 and show it is stable at the default substeps.
5. **Cache identity.** Dynamic bodies make colliders *outputs*. The frame stamp still
   covers the *authored* collider (geometry, flags, initial pose); the *solved* pose is
   a cache v3.1 record per frame (`rigid_bodies: [{object_uid, pos, quat, lin_vel,
   ang_vel}]`). Resume restores body state from the last baked frame. v3 caches load
   with no rigid records (nothing to restore).
6. **Dynamic collider that is also a source or sink.** Refuse at admission with a
   reason, as the narrow-band toggle already does for incompatible combinations.

## Design

- Per Dynamic collider: mass (from density × mesh volume, winding-number mesh), inertia
  tensor (sampled), pose (`wp.transform`), linear/angular velocity. Up to 64 bodies in
  4.0 (fixed arrays; device-side count).
- Per substep, after projection and before G2P: (1) gather face impulses and sample
  drag into per-body `F, τ` (segmented gather); (2) semi-implicit Euler on the body;
  (3) update the collider's SDF pose for the next substep through the existing moving-
  collider path (`_update_moving_colliders`), which already handles velocity history.
- Settings per collider object: `collider_dynamic` (bool), `collider_density`,
  `collider_drag`, `collider_initial_velocity`, `collider_lock_axes` (6 bools).
- Publication: Output Publication writes `matrix_world` keyframes per published frame
  (live and cache playback), tagged so they can be cleared on Reset.

## Acceptance

- `tests/solver/test_rigid_buoyancy.py`: density-ratio-0.5 cube in a still tank settles
  at 50% ± 5% submersion within 48 frames (Multigrid); ratio 1.5 sinks; ratio 1.0 is
  neutral within 10%.
- `tests/solver/test_rigid_momentum.py`: dam break into a free crate; total linear
  momentum (fluid + body) conserved to 2% between frames 10 and 40.
- `tests/solver/test_rigid_determinism.py`: two bakes with one Dynamic body are bitwise
  identical.
- `tests/cache/test_cache_v31_rigid.py`: v3 cache loads with empty body records; v3.1
  resume restores body state.
- Demo: crown_splash gains a floating cork; `check_demo_blends` asserts its pose track
  moves and the Alembic export carries it. Budget ≤ 5% per frame on crown_splash.
