# SPEC — Velocity transfer modes (APIC, optional ASFLIP)

Owner module: `sim/kernels_fluid.py` (`k_particle_set_grid_data`, `k_gather_grid_data`,
`k_staggered_field_to_particle`), `sim/solver.py` (`_g2p_advect`, particle buffers,
`SolverSettings.sample_type`), `sim/enums.py`, `sim/particle_capacity.py`,
`sim/capacity_planner.py`, `io/checkpoint.py`, `properties.py`.
Depends on: `SPEC-cache-v3.md`.

## Trap this spec prevents

1. **The textbook APIC constant `D_p^{-1} = 4/h²` is only valid for quadratic
   B-spline weights.** This solver uses trilinear (hat) weights on a MAC grid. With hat
   weights `D_p = Σ_i w_i (x_i − x_p)(x_i − x_p)ᵀ` is position dependent and must be
   computed per particle, per MAC component. Using `4/h²` with hat weights adds angular
   momentum that is not there and shows up as spurious swirl. Either compute `D_p`
   explicitly (chosen) or switch the whole transfer to quadratic B-splines (rejected:
   it widens the gather stencil from 3×3×3 bins to 5×5×5 and changes every demo).
2. **P2G and G2P expressions must stay identical between the scatter kernel and the
   deterministic gather.** `k_gather_grid_data`'s docstring is the contract: same weight
   expressions, only the summation order differs. The APIC momentum term must be added
   to both `k_particle_set_grid_data` and `k_gather_grid_data`, and the gather stays the
   path that ships.
3. **Existing scenes must not silently change look.** Migration keeps `FLIP` on every
   2.x domain; only *new* domains default to APIC. Demos are re-saved to APIC in alpha.2
   as a deliberate, hashed re-baseline.

## Enum and settings

- `ParticleSampleType`: keep `PIC = 0`, `FLIP = 1`; add `APIC = 2`, `ASFLIP = 3`
  (ASFLIP optional, see below). Integer values are kernel arguments and cache stamps;
  never renumber.
- RNA: `transfer_mode: EnumProperty` on the domain group, items FLIP / APIC (ASFLIP only
  if implemented), default APIC. `flip_ratio` is shown only when FLIP. Tier: RESET in
  `blender/pending_settings.py` (changes particle state layout).
- Cache v3 stamps `transfer_mode` (manifest) per `SPEC-cache-v3.md`.

## Particle state

- Add `affine: wp.array(dtype=wp.mat33)` to the particle set, allocated only when
  `sample_type == APIC/ASFLIP`. 36 bytes/particle; add to `particle_capacity` and
  `capacity_planner` bytes-per-particle so Safe Settings and VRAM planning stay honest.
- Emission writes `affine = 0`. Checkpoint v2 persists it. Sanitize (`k_sanitize_particles`)
  zeroes it for killed particles.

## Kernels

### G2P (`k_staggered_field_to_particle`)
Per MAC component `a ∈ {u,v,w}` with that component's face offset:
```
v_a   = Σ_i w_i u_i                         (PIC value, as today)
B_a   = Σ_i w_i u_i (x_i − x_p)             (vec3, row a of B)
D_a   = Σ_i w_i (x_i − x_p)(x_i − x_p)ᵀ     (mat33)
C_row_a = B_a · inverse(D_a)                 (row a of C)
```
`x_i` is the face centre in world space. Guard: if `det(D_a) < 1e-12·h⁶` (particle sits
exactly on a face plane so one direction has zero spread) fall back to `C_row_a = 0`
for that component; it is a measure-zero case but it does occur with analytic emitters
on grid-aligned planes. For APIC the particle velocity is the PIC value (no FLIP delta).
Keep FLIP/PIC branches byte-for-byte as today when `sample_type != APIC`.

### P2G (`k_gather_grid_data` and `k_particle_set_grid_data`)
Momentum contribution to face `i` of component `a` becomes
`w_i · (v_p[a] + dot(C_row_a, x_i − x_p))`. Weight sums unchanged. Add a
`affine` input and a `sample_type` input; when not APIC the extra term is multiplied by
zero via `wp.where`, not branched, so the generated code is identical in structure and
the FLIP gate hash is unchanged. **Verify** that claim by running the gate with the
kernel change and `transfer_mode = FLIP` before adding the enum to the UI.

### Advection
Unchanged. APIC does not need the saved `vel_saved_*` arrays; keep them allocated for
FLIP and skip the copy when APIC to recover a little bandwidth.

## Optional: ASFLIP
Only if APIC lands with budget to spare. ASFLIP adds a position adjustment
`x_p += β·dt·(v_p^{new} − v_p^{PIC})` (Fei et al. 2021) that lets particles separate at
free surfaces. Needs a boundary-proximity test; use `voxel_type & VOXEL_COLLISION`
within one voxel to disable the adjustment near solids. If not implemented, do not add
the enum item.

## Interactions to check
- Deterministic gather: the `C` term adds products in the fixed gather order; bitwise
  reproducibility must hold (3/3 gate runs).
- Whitewater emission reads particle velocity only; unaffected.
- Color mixing and `viscosity_from_particles`: per-particle attributes, unaffected.
- Custom effects API (`docs/custom_effects_api.md`): expose `affine` read-only if a
  template needs it; otherwise do not widen the API in 3.0.

## Acceptance
- Unit (`tests/solver/`): single particle in a rigid-rotation grid velocity field
  recovers `C = [[0,−ω,0],[ω,0,0],[0,0,0]]` to 1e-5 after one G2P with explicit `D`;
  with the `4/h²` constant the same test fails (keep that as a negative test to pin the
  decision).
- Angular momentum of a spinning fluid disc (no gravity, no walls) decays < 2 % over 100
  steps under APIC vs > 20 % under PIC; FLIP 0.97 result is recorded, not gated.
- dam_break and crown_splash re-baselined under APIC; paired ABBA step time ≤ +8 % per
  particle vs 2.1.1 FLIP before narrow band lands.
- Gate hash with `transfer_mode = FLIP` equals 2.1.1's reference hash (proves the
  kernel edits are behaviour-preserving for FLIP).
