# Issue: velocity impulse at every pi-crossing of collider rotation (`paddle_stir`)

Status: **RESOLVED.** Cause was quaternion double-cover in
`SceneInputs.object_angular_velocity`; fixed by shortest-arc canonicalization
(`blender/scene_inputs.py:311`). `paddle_stir` now passes **10/10**.

Scope: moving-collider path. **Pre-existing** - not introduced by any fix in this session.
Does not affect the shipped demos with static colliders (`dam_break`, `crown_splash` remain 18/18).

Date: 2026-08-09 (opened and resolved)

> **Reading note.** Sections 2-3 are the original, still-accurate observations.
> Section 4.1 recorded a *false* elimination and section 5 ranked a candidate that
> turned out to be wrong; both are retained, corrected in place, because the way they
> misled the investigation is the most transferable lesson here. **Section 10 is the
> actual root cause.**

---

## 1. Summary

A large, single-frame velocity impulse is injected into every fluid particle near a
rotating collider, on exactly the frame where the collider's accumulated Z rotation
crosses **pi (mod 2*pi)**. Confirmed at two different rotation rates. The trigger is the
rotation angle itself, not blade geometry, not collision timing, and not a fixed frame
cadence.

pi is precisely where the `w` component of the orientation quaternion changes sign
(`w = cos(z/2)`). That coincidence was the strongest available lead, and it was the
right one: `object_angular_velocity` reported **-140.32 rad/s instead of +10.47** on the
crossing frame - 13x too large and reversed. Section 4.1 originally cleared that code,
incorrectly; see section 10.

## 2. Symptom

`tools/analyze_demos.py paddle_stir` fails one check:

```
[FAIL] paddle_stir: velocity bounded (p99.9) - p99.9 speed 36.33 m/s (limit 11.94)
```

The limit is `3.0 * peak_tip_speed`; peak tip speed is 3.98 m/s (`peak_rad_per_s` 10.472 x
`blade_outer_radius` 0.38). Observed p99.9 is ~9x the blade's own tip speed. Observed max
pins the solver's hard safety clamp, `max(s.cfl * g.voxel_size / dt * 2.0, 50.0)`
(`sim/solver.py:1486`), so the true unclamped magnitude is unknown and larger.

## 3. Verified evidence

All numbers read directly from `artifacts/demo_cache/paddle_stir/frame_*.npz`.

### 3.1 Baseline (100 RPM): periodic, once per full revolution

Frames where any particle exceeds 20 m/s (96 frames total):

```
frame nfast maxspeed   rmin   rmax   zmin   zmax
   19  1070    50.00  0.169  0.454  0.066  0.274
   20     2    24.56  0.308  0.347  0.265  0.275
   34   645    49.97  0.174  0.455  0.066  0.263
   49   816    50.03  0.178  0.449  0.066  0.311
   63   681    50.00  0.180  0.458  0.056  0.393
   78   677    50.03  0.168  0.447  0.052  0.238
```

Spacing 15, 15, 14, 15. One revolution at `peak_rad_per_s` 10.472 / 24 fps = **14.4
frames**. So the event is once per full 2*pi revolution.

The paddle has `blade_count = 4` and is 4-fold symmetric, so any genuine blade/fluid
interaction must repeat every **quarter** revolution (~3.6 frames). It does not. Something
tracks absolute accumulated rotation, not blade geometry.

### 3.2 It is an injected impulse, not flow

Tracking the exact 1070 particle indices that are fast at frame 19 across neighbouring
frames (particle count is constant at 158,220, so indices are stable):

```
frame   mean speed, same index set   max
   17          1.31 m/s             3.42
   18          1.64 m/s             3.95
   19         27.19 m/s            50.00   <-- event
   20          4.73 m/s            24.56
   21          2.90 m/s            10.88
   22          2.43 m/s             6.46
   25          1.68 m/s             4.50
```

Mean speed for a **fixed** particle set jumps 16x in one frame, then decays to baseline
over ~5 frames. Energy is added in a single step and dissipated. Signature of a bad
impulse, not of physical acceleration.

### 3.3 Spatial extent rules out a localized pinch

An early hypothesis was a pinch between blade tip (r = 0.38) and tank wall (r = 0.45). The
data rejects it. Affected particles span **r = 0.168 to 0.458**, **z = 0.052 to 0.393**:

- inner edge 0.168 ~= `blade_inner_radius` 0.18
- outer edge 0.458 is *past* `tank_inner_radius` 0.45 - inside the wall band
- lower edge 0.052 is *below* `blade_z_min` 0.10 - under the blades entirely

The impulse covers the entire blade sweep volume plus margin. Consistent with a collider
field being wrong over its whole neighbourhood for one step; inconsistent with contact at
any single radius.

### 3.4 Decisive experiment: period scales with RPM

Rebaked at **50 RPM** (single change: `omega_peak` in `build_paddle_stir`,
`tools/build_demo_scenes.py:1331`). Particles over 10 m/s:

```
frame nfast maxspeed   rmin   rmax
   27  7799    50.01  0.104  0.458
   28   100    16.29  0.239  0.454
   56  5316    50.02  0.130  0.459
   57    42    15.83  0.246  0.456
```

Spacing **29 frames**; one revolution at 50 RPM is 28.8 frames.

| RPM | revolution | observed spacing |
|-----|-----------|------------------|
| 100 | 14.4 frames | 15, 15, 14, 15 |
|  50 | 28.8 frames | 29 |

**The period scales with rotation rate.** This eliminates any fixed-cadence infrastructure
cause - graph-cache eviction, SDF rebuild interval, reseed cadence, cache flush. The
trigger is rotation-tied.

Note also: at **half** the speed the event is **worse** (7,799 affected particles vs 1,070)
and still pins the 50 m/s clamp. Impact energy would scale the other way. Further evidence
this is a field error, not a collision.

### 3.5 The trigger is the pi-crossing, confirmed at both rates

Accumulated rotation `z mod 2*pi` at each event, and at the frame immediately before:

```
50 RPM:
  f26  z= 3.0543  mod2pi= 3.0543      (just under pi)
  f27  z= 3.2725  mod2pi= 3.2725   <-- EVENT, first frame past pi
  f55  z= 9.3811  mod2pi= 3.0980      (just under pi)
  f56  z= 9.5993  mod2pi= 3.3161   <-- EVENT, first frame past pi

100 RPM events, mod2pi:  3.244  3.316  3.578  3.403  3.283   (all just past pi)

pi = 3.14159
```

At both rates, every event is the **first frame in which accumulated rotation crosses pi
(mod 2*pi)**, and the preceding frame sits just below pi. This is the tightest
characterization available and holds across a 2x change in rotation rate.

For a Z-axis rotation, `q = (cos(z/2), 0, 0, sin(z/2))`. At `z = pi`, `w = cos(pi/2) = 0`
and changes sign immediately after. The trigger condition and the quaternion hemisphere
boundary coincide exactly.

### 3.6 The staleness fix is exonerated

Control run: per-substep collider re-rasterization forced off
(`moving_colliders = False and ...`, `sim/solver.py:1781`), rebaked at 50 RPM:

```
NO per-substep re-rasterization, 50 RPM:
frame nfast maxspeed  mod2pi
   27  6913    50.01  3.2725
   28   289    16.86  3.4907
   56  4637    42.26  3.3161
   57   261    17.97  3.5343
```

Identical frames, identical pi-crossing trigger, same clamp. **The defect pre-dates the
per-substep re-rasterization fix added in this session.** Both edits above have been
reverted; the working tree is back to 100 RPM with the fix enabled.

## 4. Hypotheses eliminated (with evidence)

### 4.1 Quaternion double-cover in `object_angular_velocity` - ~~ELIMINATED~~ **THIS WAS THE CAUSE**

> **This elimination was wrong, and it cost the investigation most of its time.** The
> table below was produced by a *reimplementation* of the angular-velocity math written
> for the experiment, not by calling the shipped function. The reimplementation assumed
> `rotation_difference` returns the shortest arc. It does not. The claim
> "`rotation_difference` returns the shortest-arc quaternion" in the last paragraph is
> **false** - see section 10. The original text follows unaltered.


The natural reading of section 3.5 is that a naive finite difference of `to_quaternion()`
across the hemisphere boundary yields one enormous spurious angular velocity per
revolution. Tested against the real code path (`SceneInputs.object_angular_velocity`,
`blender/scene_inputs.py:301`, which does `previous.rotation_difference(current)` then
`.angle / dt` scaled by `.axis`), driven by the exact `rotation_keyframes` ladder from
`demos/paddle_stir/meta.json`:

```
frame   z_rad   z_mod2pi   delta.angle   omega_z
   18  2.9026    2.9026     0.32251         7.740
   19  3.2440    3.2440     0.34148         8.195   <-- event frame
   20  3.6045    3.6045     0.36045         8.651
   33  9.1630    2.8798     0.43633        10.472
   34  9.5993    3.3161     0.43633        10.472   <-- event frame
   35 10.0356    3.7525     0.43633        10.472
```

`omega_z` is smooth and correct through both event frames. `rotation_difference` returns
the shortest-arc quaternion, so the hemisphere flip is already handled here.
**Angular velocity is not the defect.**

This is the central open puzzle: the trigger sits exactly on the quaternion sign boundary,
but the one place that obviously converts orientation to a quaternion is clean. Some
*other* stage must be orientation-sensitive at pi.

> **Correction.** There was no other stage. That place was not clean; the experiment
> could not see it. Section 10 has the measured numbers from the live path.

### 4.2 Fixed-cadence infrastructure (graph cache, rebuild interval) - ELIMINATED

Section 3.4. Period scales with RPM; a fixed cadence cannot.

### 4.3 Blade-tip / tank-wall pinch - ELIMINATED

Section 3.3. Affected volume is far larger than any contact region and extends below and
outside the blades.

### 4.4 Unbounded SDF push-out in `resolve_collision` - ELIMINATED as this cause

Push-out was scaled by `voxel_size / |grad|` with no bound; near a medial axis or thin
feature the gradient estimate collapses and the factor diverges. Genuine defect,
**fixed** this session (clamped to `[0.25, 4.0]`, `sim/kernels_fluid.py`). Did not change
this symptom.

### 4.5 CFL blind to collider motion - ELIMINATED as this cause

`_compute_substeps` derived `vmax` from fluid velocities plus gravity only, so a fast
collider meeting still fluid could force too-coarse substeps. Genuine defect, **fixed**
this session via a new `_collider_max_speed()` (`sim/solver.py`). Did not change this
symptom.

### 4.6 Collider staleness - ELIMINATED as this cause

Colliders were rasterized only on substep 0. Fixed this session. Section 3.6 shows the
spike is present with that fix both enabled and disabled.

4.4, 4.5 and 4.6 are real robustness improvements, verified to cause no regression on the
demos with static colliders. None is the mechanism here.

## 5. Remaining candidates (all three DISPROVEN - see section 10)

Must explain: fires on the pi-crossing of accumulated rotation, at any rotation rate,
affecting the collider's whole neighbourhood, for exactly one frame, without corrupting
`angular_velocity`.

> **All three were wrong, and the last clause of that sentence is why.** It smuggles in
> section 4.1's false result as a constraint, which excluded the true cause by
> construction.
>
> Candidates 1 and 2 are both refuted by one argument that needs no measurement: **the
> paddle is exactly 4-fold symmetric** (a 24-gon hub plus four identical blades at
> `k * pi/2`, centered on the world origin and spinning about its own axis). Its
> world-space geometry is therefore **pi/2-periodic**, so *any* function of that geometry -
> SDF sign, mesh winding, evaluated normals, transform handedness - must repeat every
> quarter revolution. The defect is **2*pi-periodic**. No geometric quantity can produce
> it. This also generalizes section 3.1's blade-count observation from "must repeat every
> quarter revolution" to "cannot be geometric at all", which is what redirected the
> search to orientation carried as a *number* rather than as a shape.
>
> Candidate 3 was half right: `collision_vel` really is the corrupted field, and the
> whole-neighbourhood extent really does come from the `cross(angular, p - center)` term.
> But `center` was fine; `angular` was the input that was wrong.

1. **Orientation-dependent collider SDF sign test.** Colliders use the orientation-
   *dependent* sign query with `strict_degenerate=False` (`blender/scene_inputs.py:848`),
   unlike sources, which use orientation-independent `mesh_query_point_sign_parity`
   (`sim/sourcing.py:805`). A sign inversion over part of the paddle for one frame flips
   `phi` for every particle near the collider and fires the push-out across exactly the
   observed volume, in exactly one step. **Best fit.**
   *Test:* dump `collision_sdf` at frames 26/27/28 (50 RPM) and diff the sign field near
   the paddle.

2. **Evaluated-mesh winding / normal flip at the pi boundary.** Colliders read the
   evaluated mesh with `MeshEvaluationMode.EVERY_SIM_FRAME`. If the SOLIDIFY result or the
   evaluated transform flips handedness as the object passes pi, effective normals invert
   for one frame - which feeds directly into candidate 1.
   *Test:* check `matrix_world.determinant()` and evaluated normal direction across the
   crossing.

3. **`center` and `angular` sampled from different frames.** `collision_vel = linear +
   cross(angular, p - center)`; `center` is `matrix_world.translation`
   (`blender/scene_inputs.py:860`). Incoherent sampling injects a spurious orbital term
   over the whole neighbourhood. Weaker fit - does not obviously key on pi.
   *Test:* log `center` and `angular` per substep and confirm same-frame sourcing.

## 6. Impact (as assessed while open)

- **Shipped demos unaffected.** `dam_break` and `crown_splash` use static colliders and
  remain `DEMO-ANALYSIS-PASS (18/18)`.
- **`paddle_stir` is 9/10**, standalone, not wired into the production gate.
- **The moving-collider path is what is at risk.** It had no demo, no check and no test
  before this work; `paddle_stir` exists to exercise it. This is the first real defect it
  surfaced.
- **Any rotating collider is affected**, at any speed, once per revolution - not specific
  to this scene.
- The 50 m/s safety clamp is currently the only thing preventing a visible blowup. It masks
  the true magnitude; anything depending on that clamp for stability is fragile.

## 7. Reproduction

```bash
blender --background --factory-startup --python tools/build_demo_scenes.py -- paddle_stir
```

```bash
python tools/analyze_demos.py paddle_stir
```

Before the fix: `DEMO-ANALYSIS-FAIL (9/10)`, fast frames at the shipped 100 RPM 19, 34,
49, 63, 78. After the fix: `DEMO-ANALYSIS-PASS (10/10)`, no frame above 20 m/s.
To vary the rate, change `omega_peak` at `tools/build_demo_scenes.py:1331`.

The cheap check is now the unit test, which needs no GPU and no Blender:

```bash
python -m pytest tests/test_scene_bridge_collider_cache.py -k hemisphere -q
```

## 8. Related fixes already landed

Committed as `e13945b` (`sim/`) and with the `paddle_stir` demo in `d5ce823`
(`tools/analyze_demos.py`). None is the cause of this issue, but the analyzer change is
load-bearing for the 10/10 result reported in section 10.3.

- `sim/solver.py` - per-substep collider re-rasterization (staleness fix); collider tip
  speed folded into CFL `vmax` via new `_collider_max_speed()`.
- `sim/kernels_fluid.py` - `resolve_collision` push-out gradient correction clamped to
  `[0.25, 4.0]`.
- `tools/analyze_demos.py` - radial containment now tracks per-particle "ever cleared the
  rim" state instead of an instantaneous z cutoff. The tank is open-topped by design, so a
  particle that legitimately splashes over the rim stays exempt for the rest of its
  trajectory rather than being re-flagged once it falls back below rim height while still
  laterally outside. 848.7 mm -> 461.2 mm (limit 468 mm).

## 9. Secondary observation

Cached velocities are stored as float16. Summing squared speeds in that dtype overflows to
`inf` during analysis. Harmless for stored values (50 m/s is well inside float16 range), but
analysis scripts must cast to float32 or float64 before accumulating.

## 10. Root cause and fix

### 10.1 Mechanism

Three `mathutils` behaviours compose into the defect. Each is individually documented and
individually reasonable; the combination is not obvious from any one call site.

1. **`Matrix.to_quaternion()` returns the canonical `w >= 0` representative.** Since
   `w = cos(z/2)`, the continuous branch goes negative just past `z = pi`, so mathutils
   hands back the *antipodal* quaternion instead. `q` and `-q` are the same rotation, so
   this is correct - but it means consecutive samples straddling a half-turn land in
   **opposite hemispheres**.
2. **`Quaternion.rotation_difference` is `conj(q1) * q2` with no hemisphere
   normalization.** Given such a pair, the delta comes back negated.
3. **`Quaternion.angle` is `2 * acos(w)` over `[0, 2*pi]`, not the shortest arc.** A
   negated delta therefore reads as the long way round, with `.axis` flipped.

Measured on the live path in Blender 5.1 at the crossing frame:

```
q_prev (z=2.8416)   w=+0.149435  z=+0.988772
q_cur  (z=3.2779)   w=+0.068101  z=-0.997678   <-- hemisphere flip
delta               w=-0.976300  angle=5.846886  axis=(0, 0, -1)
omega_z = -140.3244 rad/s        (true +10.4712)
```

The error is exact: `2*pi/dt - omega_true` = `2*pi*24 - 10.472` = **140.3244**. This is
why the event is worse at 50 RPM (section 3.4) - the error *grows* as the true rate
shrinks.

That `angular_velocity` feeds `collision_vel = linear + cross(angular, p - center)`
(`k_rasterize_collider_material`, `sim/solver.py:240`), so a single bad scalar corrupts
the collision velocity field across the collider's **entire neighbourhood** for exactly
one frame - reproducing section 3.2's impulse signature and section 3.3's spatial extent
without any contact event. Measured `max|collision_vel| = 127.40` on the grid.

Every observation in sections 2-3 follows: 2*pi period keyed at pi (that is where the
`w` sign changes; `3*pi` fires, `2*pi` does not), one frame only (the next frame's two
samples share a hemisphere again), and indifference to the section 3.6 re-rasterization
control (the bad `omega` feeds `collision_vel` either way).

### 10.2 Fix

`blender/scene_inputs.py:311` - canonicalize the delta to the shortest arc before reading
`.angle` / `.axis`:

```python
if delta.w < 0.0:
    delta.negate()
```

### 10.3 Verification

| Check | Before | After |
|---|---|---|
| `analyze_demos.py paddle_stir` velocity gate | 36.33 m/s | **4.64 m/s** (limit 11.94) |
| `analyze_demos.py paddle_stir` | FAIL 9/10 | **PASS 10/10** |
| `dam_break` + `crown_splash` | PASS 18/18 | **PASS 18/18** |
| `tests/test_scene_bridge_collider_cache.py` | 27 passed | **29 passed** |

4.64 m/s is not merely under the limit - it is the physically right magnitude for a
3.98 m/s blade tip. Four unrelated failures in
`tests/test_pressure_projection_regressions.py` (PCG `cuda_conditional` vs
`host_batched` classification) were confirmed pre-existing by re-running with this change
stashed.

### 10.4 Why the test suite did not catch it

`tests/test_scene_bridge_collider_cache.py` already covered `object_angular_velocity`.
Its `_FakeQuaternion.rotation_difference` returned `abs(other.angle - self.angle)` with a
hand-picked axis - it computed the answer the production code was supposed to derive, so
it asserted its own arithmetic and was **structurally incapable** of expressing this bug
class. Combined with the section 4.1 reimplementation, which made the same assumption,
the codebase held two independent copies of the wrong mental model and no copy of
mathutils' actual semantics.

The fake is now faithful (real quaternion algebra, `w >= 0` canonicalization on
`to_quaternion()`, no hemisphere fix in `rotation_difference`, `.angle` as
`2 * acos(w)`). It reproduces the live Blender number to 5 significant figures
(-140.3252 vs -140.3244; the residual is `Matrix.Rotation`'s matrix round-trip) in
**0.65 s with no GPU**. The regression test
`test_angular_velocity_survives_quaternion_hemisphere_flip` covers crossings at `pi` and
`3*pi` with controls at `2*pi` and a no-crossing case.

**Transferable rule: a fake that reimplements the logic under test asserts nothing.** The
value of a library fake is in reproducing the library's *surprising* behaviour, not its
expected behaviour.
