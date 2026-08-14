# Durable Correction Plan — FluidSim Warp Blender 5.1 Add-on

Companion to `reviews/ADDON_REVIEW.md` (findings + live evidence) and
`artifacts/live_review/` (probe scripts, JSON, logs).

**Status: PLAN ONLY. No code has been changed.**

---

## 0. Landed so far

| Item | Status | Evidence |
|---|---|---|
| **B5** — `_watertight_triangles` deferred + vectorized | **DONE** | Live: watertight calls over 4 playback frames **8 → 0**; 4-frame playback **2171.7 ms → 458.2 ms** (`artifacts/live_review/probe3_after.json`). Sink consumer still enforced: `MESH-SOURCE-ACCEPTANCE-PASS` with "Solid mesh sink requires a watertight mesh". |
| **`render_init` `@persistent`** + widened handler gate | **DONE** | `HANDLER-PERSISTENCE-GATE-PASS`. Negative test: removing the decorator yields `HANDLER-PERSISTENCE-GATE-FAIL: [after register] add-on app handlers missing @persistent: ['render_init._on_render_init']`. |

Regression lanes after both changes: `261 passed` host pytest (was 256; +5 new),
`BACKGROUND-ACCEPTANCE-PASS`, `VIEWPORT-2FRAME-GATE-PASS`, `CACHE-EXPORT-PIPELINE-PASS`,
`MESH-SOURCE-ACCEPTANCE-PASS`, `HANDLER-PERSISTENCE-GATE-PASS`.

Durability layers achieved: L1 fix, L2 red-first test (both were demonstrated failing first),
L3 gate wiring (the handler gate now checks **all** add-on handler lists, not just
`frame_change_post`, which is what let the `render_init` bug through). L4/L5 for these two items are
still outstanding: no `AGENTS.md` note and no backlog task files yet.

**A test-design correction worth recording:** my first version of
`test_watertight_triangles_semantics` asserted that two coincident triangles are *not* watertight.
That was my opinion, not the code's contract — every edge has count 2, so the existing criterion
reports `True`. I changed the test to pin existing behavior rather than silently change semantics
while claiming a perf fix.

---

## 1. What "durable" means here

A correction is not done when the symptom disappears. Each item below lands **five
layers**, and an item is only closed when all five are present:

| Layer | Requirement |
|---|---|
| **L4b Parity/ledger rule** (adopted from §2.7) | No claim of `fluidSim.cl` equivalence without a direct parity test; any feature that is not a faithful port is recorded in `docs/feature_ledger.md` as deferred or port-extension. |
| **L1 Fix** | The code change itself. |
| **L2 Red-first test** | A test that **fails against today's tree** and passes after. Written and demonstrated red *before* the fix. |
| **L3 Gate wiring** | The test runs inside `tools/run_production_gate.py`, so it cannot rot silently. New Blender lanes get a `*-PASS`/`*-FAIL` marker pair. |
| **L4 Doc truth** | `AGENTS.md` workflow line, and correction of any doc/docstring the fix contradicts (`docs/technical_report.md`, `POLICY_DRIFT_REPORT.md`, module docstrings). |
| **L5 Backlog record** | A `backlog/tasks/TASK-*.md` following the existing 5-heading convention (State / What to build / Acceptance criteria / Evidence / Blocked by), with the live-probe evidence path. |

**Anti-pattern to avoid, explicitly:** this repo already contains tests that assert
*source text* rather than behavior (`assert "tests/blender_memory_safety_gate.py" in gate`).
No L2 test in this plan may be a grep. Every L2 test asserts an observable value.

### The single highest-leverage durability move

`tests/blender_handler_persistence_gate.py` already does a real save→reopen cycle, but
`addon_frame_handlers()` filters to `_on_frame_change` **only** — which is exactly why the
missing `@persistent` on `render_init` survived. Widening that one gate to assert **every**
handler the add-on registers survives a reopen closes the whole class of bug, not just the
one instance. Same shape for the enum gate: assert every dynamic-enum property on a fresh
scene resolves to a valid item, not just the snippet one.

---

## 2. Decisions I need from you

These change behavior or numbers, so I do not want to pick unilaterally.

**D1 — mostly already decided by this repo; see §2.8.** `docs/SOURCE_EMISSION_PRODUCT_PLAN.md:76`
already diagnoses the exact B1 mechanism and prescribes the resolution: remove `Mode = Field` from
the normal Sources UI, retain the enum/kernel as compatibility/advanced internals. I should not
re-litigate that. The **remaining** open question is narrower and real:
- **D1' — the FORCE role depends on the broken FIELD path.** `scene_inputs.py:341-343` forces
  `mode = FIELD, emit_rate = 0` for `role == "FORCE"`, and the FIELD branch is where
  `shape_force_velocity()` is applied to the grid (`sourcing.py:578`). So hiding FIELD from the
  Sources UI does **not** fix object-level force regions. Decide: (i) make the FIELD grid write
  correct (post-normalize, weighted face) so FORCE regions work — recommended; or (ii) reroute the
  FORCE role through the existing forces stage (`k_forces_directional` /
  `k_forces_omnidirectional`), which is what `fluidSim.cl` does (§2.6C); or (iii) declare the FORCE
  role unsupported in the ledger.
- ~~(a) prescribed-without-ledger, (b) weighted, (c) staggered-composite, (d) keep-as-extension,
  (e) remove~~ — superseded.

**D2 — `domain_sdf_walls` default.** Currently `False`, which the kernel's own docstring says
causes boundary compression. The repo's own handler-persistence gate manually sets it `True`.
Flipping the default is correct but **changes the output of every existing scene**, including
the two calibrated demos. Options: flip the default (recommended), or always run
`k_mark_domain_walls` for closed domains and keep the property as an escape hatch only.

**D3 — Demo recalibration authority.** B3 (fail-open projection), D1 and D2 all change solver
output. `tools/analyze_demos.py` thresholds are physics-calibrated and currently pass 18/18
(e.g. Martin–Moses `Z(T=2) ∈ [2.5, 4.0]`). If a fix moves a demo outside a band, I need to
know whether I may **re-calibrate the band** (and record why) or must **treat it as a
regression** and change the fix. Recommended: treat drift as a regression by default; only
re-calibrate with an explicit physics justification written into the backlog task.

**D4 — Execution granularity.** Land all seven waves in one pass and report at the end, or
stop after each wave for your review? Recommended: stop after Wave 2 (the sim blockers,
highest risk), then continue unattended through Wave 6.

**D5 — Dead-kernel policy (new, from §2.5).** 37 of 157 kernels are defined, registered, and
**never dispatched**. Options: (i) leave them, but mark each `register_fn` with an explicit
`notes="not dispatched"` and make the coverage test report the count (recommended — cheap,
honest, no behavior change); (ii) wire up the ones that represent real missing pipeline stages
(only `k_core_source_staggered_vector` is clearly in that category, via D1c); (iii) delete them.
I recommend (i) + (ii)-for-D1c only, because several are legitimately API-surface for custom
snippets.

---

## 2.5 Evidence from `docs/references/houdini_dispatch/`

You pointed me at the archive; it materially changed three things. **First, a correction to
myself:** my initial grep suggested `paradigmResetSourceShapeTrackedVoxelType` was unimplemented
in the port. That was wrong — I searched for the `paradigm*` name while the port registers the
canonical `fluidSim*` name. It **is** registered (`kernels_fluid.py:1865`).

**Finding 1 — B1 is a restoration, not a judgement call.** The original engine has a kernel whose
entire job is clearing the source-tracked bit: `paradigmResetSourceShapeTrackedVoxelType`
(fluidSim.cl:17855). A dedicated reset kernel only makes sense if the tracked bit is *meant to
persist* through the intervening pipeline stages — including active/fluid marking — and be cleared
at one defined point. Our port sets the bit in sourcing, **never launches the reset kernel**, and
lets `k_particle_to_grid_active` clobber `VOXEL_FLUID` in tracked voxels. That is now three
independent lines of evidence for the B1 fix: the ST-FLIP twin kernel honors the bit, the original
architecture requires the bit to survive, and the live measurement shows `total_fluid = 0`.

**Finding 2 — WITHDRAWN.** I claimed `coreSourceStaggeredVector` was the original's seam for
compositing shape-source velocity into the MAC field. Its signature takes
`CoreParametersSourceVolume` plus a mask field — it is the **VDB/volume** sourcing path, not the
shape-source path. My D1(c) recommendation rested on that mistake and is withdrawn. See §2.6.

**Finding 3 (NEW defect) — kernel coverage is name-deep, not dispatch-deep.** Measured by AST:
**157 `@wp.kernel` definitions, 116 launched, 37 never launched or referenced anywhere** outside
their own definition/registration (plus a few more that are imported but never launched, e.g.
`k_clear_just_initialized_tracked`, `k_source_resolve_type`, `k_source_height`). `tests/test_kernel_coverage.py`
passes because it only checks that every manifest name resolves to *something* in the registry.
Also note `k_clear_just_initialized_tracked` is registered under **two** distinct original kernel
names (`fluidSimClearJustInitializedVoxelTypeTracked` **and** `fluidSimResetSourceShapeTrackedVoxelType`,
two separate kernels at separate source lines in the original) — collapsing two behaviors into one
unlaunched kernel while both manifest entries report as covered.

**Limit of the archive alone:** its substep pseudocode (§3.5) is explicitly generic — *"Concrete
ordering mirrors fluidSim.cl solver phases; shown generically"* — so it gives no authoritative
dispatch order. §2.6 replaces this with the real kernel source.

---

## 2.6 Ground truth from `fluidSim.cl` (20,640 lines, read directly)

Source: `C:\Users\Tyler\Downloads\paradigm1.0.226\...\compute\openCL\fluidSim.cl` (772 KB).
The surrounding directory is a **separate planning workspace, not the Paradigm distribution**, and
is not treated as authoritative — only `fluidSim.cl` itself was read.

**A. `k_particle_to_grid_active`'s `else` branch is a port invention.** The original
`paradigmParticleToGridActive` (:18987) does exactly one thing:
`a_voxelType[l_voxelIndex] |= Bit_Core_VoxelType_Active`. Purely additive; it never touches the
Fluid bit. Across all 20,640 lines, `Bit_Core_VoxelType_Fluid` (= 4) appears 13 times and is
**never cleared by any kernel** — it is set by `paradigmParticleSetGridData` (:18563), propagated by
`paradigmExpandFluidMask` (:18098), and reset only by the whole-buffer `coreClearScalarVoxelType`.

Our port added `else: voxel_type[tid] = vt & (~VOXEL_FLUID)` (`kernels_fluid.py:154-155`). No
counterpart in the original; redundant (the port already zeroes `voxel_type` in
`_clear_grid_fields` each substep); and the direct cause of B1.

> **Corrected B1 fix: delete the `else` branch** so the kernel is purely additive. My earlier
> proposal — add `or (vt & VOXEL_SOURCE_REGION_TRACKING)` to the predicate — was wrong in
> mechanism: it patches over the invented branch and would still wrongly clear FLUID in every
> non-source voxel. Right effect for source voxels, wrong everywhere else.

**B. `VOXEL_SOURCE_REGION_TRACKING` is a one-shot emission latch, not a fluid marker.**
`paradigmSourceShapes` (:11655-11660):
```c
voxelType_t l_voxelTypeTracked = a_voxelTypeTracked[l_index];
a_voxelType[l_index]        |= Bit_Core_VoxelType_SourceRegionTracking;
a_voxelTypeTracked[l_index] |= Bit_Core_VoxelType_SourceRegionTracking;
if((l_voxelTypeTracked & Bit_Core_VoxelType_SourceRegionTracking) != 0) return;  // already sourced
```
The original keeps **two** voxel-type arrays — live `a_voxelType` and persistent
`a_voxelTypeTracked` — and the latch prevents re-emitting into an already-sourced voxel;
`paradigmResetSourceShapeTrackedVoxelType` releases it only where the live bit is absent. Our port
collapsed both arrays into one `voxel_type`, so it cannot implement the latch — which is why the
reset kernel is unlaunched and aliased to two manifest names — and then repurposed the bit to mean
"fluid because a source says so" in the ST path.

**C. The original never writes grid MAC velocity for *fluid* sources.** Of 19 `coreWriteFieldValue`
calls in `paradigmSourceShapes`, the fluid-source path writes to **particles**
(`a_particleVelocity_*` at `l_newParticleIndex`, :11753-11755); the only `a_velocity_*` grid writes
(:11810-11819) are on the **collision** path, scaled by `collisionVelocity.scale` — prescribing
solid velocity. Force-shape influence on the grid is applied by the dedicated
`paradigmForcesDirectionalForce` / `paradigmForcesOmnidirectionalForce` kernels in the forces stage,
not by sourcing.

> So the port's FIELD mode for fluid sources (`role == "FORCE"` → `SourceShapeSourceMode.FIELD`,
> writing MAC faces inside `k_source_shapes`) is a **port invention with no original counterpart**.
> B2 is not "a ported feature was mis-ordered" — it is "an invented feature shipped without a
> coherent grid-write contract". Hence D1 is now (d) vs (e).

**D. Related gap:** `paradigmExpandFluidMask` is how the original propagates the Fluid bit, and our
`k_expand_fluid_mask` is in the never-launched set. Decide whether the port needs it once the
`else`-branch clear is removed.

**Fidelity note on P2G:** the original averages by integer particle *count*
(`coreParticleToGridAverage`: `gridCollect/gridCount`, fixed-point `/1000`, then `gridValue += value`),
whereas the port uses float trilinear weights (`u/wu`). The port's choice is a legitimate, arguably
better FLIP formulation and I am **not** proposing to change it. The structural lesson that does
carry over: the original *adds* normalized particle contributions onto whatever the field already
holds, and never divides a source-written value by a particle weight.

---

## 2.7 Reuse assessment of the sibling port workspace

Per your "only if it is a verified improvement" constraint, I assessed it rather than assuming.

**Code: REJECTED — not an improvement.**
- Its `particle_to_grid_active_kernel` is documented as a *"Dense Warp port"*
  (`paradigm_fluidsim/core/kernels.py:719,734`). Dense contradicts this project's mandated sparse
  tiled 8×8×8 leaf-grid contract.
- The only implementation lives in `.claude/worktrees/orbit-fluidsim-warp-blender-port/` and
  `.drift_quarantine/20260624-171337/` — a worktree and an explicitly quarantined drift snapshot.
- Its own `AGENTS.md` forbids implementation until `plans/000-port-roadmap.md` reaches
  `APPROVED_FOR_IMPLEMENTATION`; git log shows it at "APPROVED-READY", i.e. pre-implementation.
- `VOXEL_FLUID` / tracked-array greps in its shipped package dirs return nothing, so it has not
  solved either problem we are fixing.

**Process: ADOPT two rules — verifiable improvement over what this repo has.**
1. **Unsupported-feature / extension ledger.** Its rule: *"Do not reduce the original OpenCL feature
   surface silently. If a feature is not ported, record it in the unsupported-feature ledger."*
   This repo has no such ledger. It is exactly the mechanism that would have caught FIELD mode
   (invented, undocumented) and the 37 never-dispatched kernels. **Adopt as
   `docs/feature_ledger.md`**, with three states per original kernel/feature: ported / deferred /
   port-extension. Verification that it is an improvement: the ledger, applied retroactively, flags
   both defects §2.6C and §2.5-Finding-3 that our current gates pass silently.
2. **Parity-test rule.** Its rule: *"Do not claim equivalence to `fluidSim.cl` without a direct
   parity test or a documented unsupported/deferred feature."* Directly targets the
   overclaim class found throughout this review (multigrid docstring, "reads back only small
   statistics", name-deep kernel coverage). **Adopt as an L4 requirement in §1.**

Both are documentation/process, carry no runtime risk, and need no code from the sibling repo.

---

## 2.9 FIELD mode — DEFINITIVE, from `fluidSim.cl` + `paradigm.hda`

**This supersedes §2.6C and §2.8, and overturns my B1/B2 framing. I was wrong twice.**

**It is not a port invention.** `fluidSim.cl:10841` defines
`Enum_Core_SourceShape_SourceMode_Field = 0`, `_Particles = 1`, and `paradigm.hda` exposes it as parm
`sourceMode`, label **"Mode"**, `hidewhen "{ type != 0 }"` (shown only for `type == Source`). The
companion `Enum_Core_SourceShape_Type` is `Source=0, Collision=1, Sink=2, Activation=3,
CameraFrustum=4` — which our port's `stype` matches exactly.

**What Field mode actually does** (`paradigmSourceShapes`, line 11644 onward):
```c
if (type == Source) {
    if (sourceMode == Field) {
        if (voxelType & Collision) return;
        if (voxelType & Sink) return;
        if (sourcePressure.enable) {
            l_fieldValue = coreReadFieldValue(l_index, a_pressure);
            l_fieldValue = paradigmSourceShapeSourceField(sourcePressure, l_fieldValue,
                                                         l_sourcePressureValue);
            coreWriteFieldValue(l_fieldValue, l_index, a_pressure);
        }
        /* sibling blocks for the other scalar fields */
    }
    else if (sourceMode == Particles) {
        if (regionAwareFill) {
            voxelType_t l_voxelTypeTracked = a_voxelTypeTracked[l_index];
            a_voxelType[l_index]        |= Bit_Core_VoxelType_SourceRegionTracking;
            a_voxelTypeTracked[l_index] |= Bit_Core_VoxelType_SourceRegionTracking;
            ...
```
Field mode **blends source values into existing scalar grid fields** (pressure, temperature, …) and
**never sets `Bit_Core_VoxelType_Fluid`, never sets the tracking bit, and never writes MAC velocity.**
The HDA help text agrees: *"Influence fields allow you to add extra forces without sourcing any
density."* Liquid is created **only** by Particles mode.

**Corrections this forces:**

1. **§2.6C was wrong.** I concluded "FIELD mode is a port invention with no original counterpart"
   after scanning `paradigmSourceShapes` for MAC-velocity writes and finding only collision-path
   ones. The mode exists; I looked for the wrong signature (velocity writes) instead of scalar-field
   blends.
2. **The tracking bit belongs to Particles + `regionAwareFill`, not to sourcing generally.** §2.6B
   described the latch mechanism correctly but placed it in the wrong branch. It is a
   "don't re-fill this voxel" guard for region-aware particle fill.
3. **B1's observed `total_fluid = 0` is arguably *correct* behavior for a Field-mode source.** The
   defect is not "Field mode fails to create liquid" — it is that our port made Field mode set
   `VOXEL_FLUID | VOXEL_ACTIVE | VOXEL_SOURCE_REGION_TRACKING` and write MAC velocity (all three
   unlike the original), and then separately clobbers FLUID in `k_particle_to_grid_active` (an
   invented `else` branch, §2.6A — that finding stands). Two inventions cancelling into an inert
   feature.
4. **The repo's own product plan is also mis-premised.** `SOURCE_EMISSION_PRODUCT_PLAN.md:76` frames
   the claim as "`Mode = Field` creates liquid". The original never claimed that either. The honest
   restatement: *Field mode is an influence mode that blends values into grid fields; it does not
   create liquid.*

**Revised fix direction (replaces D1/D1'):**
- Make FIELD mode blend **scalar field values only** (pressure/temperature/viscosity/color), and stop
  setting `VOXEL_FLUID`, `VOXEL_ACTIVE`, and the tracking bit in the field branch.
- Move the tracking bit to the **particle** path, gated on a `regionAwareFill` equivalent.
- Route the FORCE role's grid influence through the **forces stage**
  (`k_forces_directional` / `k_forces_omnidirectional`), not through source MAC writes — i.e. former
  option (ii), now evidence-backed rather than a guess.
- Keep §2.6A (delete the invented `else` branch in `k_particle_to_grid_active`) — independent and
  still correct.
- Relabel the UI "Mode" to match the original's meaning and fix the missing `description=`.

---

## 2.8 What FIELD mode actually is — first pass (superseded by §2.9)

Four sources, in increasing order of authority for *this* port:

1. **In `fluidSim.cl`: no such concept for fluid sources.** Shape sources emit particles; the only
   grid-velocity writes are collision-path (§2.6C). So there is no original semantic to restore.
2. **In the synthesized enum:** `SourceShapeSourceMode = {FIELD, PARTICLES}`. FIELD = "mark grid
   fields directly (voxel type, velocity, temperature, viscosity) instead of emitting particles."
   Note `source_mode` at `properties.py:397-399` has **no `description=`** — it is an undocumented UI
   control, which is part of why its intent is unclear from the code alone.
3. **In practice it is the implementation of the FORCE role.** `scene_inputs.py:341-343` sets
   `mode = FIELD, emit_rate = 0` whenever `role == "FORCE"`. FIELD's real shipped job is "this shape
   is a force region, not an emitter", and the grid velocity write is how that force is applied
   (`shape_force_velocity` at `sourcing.py:578`; the same helper also sets emitted-particle
   velocities at `:669`, which is the working path).
4. **This repo already diagnosed and decided it.** `docs/SOURCE_EMISSION_PRODUCT_PLAN.md:76`:
   *"`Mode = Field` creates liquid | Field mode marks grid fields, but the normal particle-to-grid
   path clears fluid cells without particles | **Nonconformant product surface** | Remove it from the
   normal Sources UI; retain the synthesized enum/kernel only as compatibility/advanced internals"* —
   plus `:81` on Blend/Feather being field-mode-only and compatibility-only.

**Consequence I initially missed:** the product plan's remedy is a *UI* remedy. Because the FORCE
role routes through FIELD, hiding FIELD from the Sources UI leaves **object-level force regions
non-functional**. That, not "an inert enum value", is B1/B2's real user-facing impact.

**Not yet verified:** my live probe exercised `role = SOURCE` + `source_mode = FIELD`
(→ `total_fluid = 0`). I have **not** measured a `role = FORCE` object. The demo suite's force
coverage uses Blender native effectors (`bpy.ops.object.effector_add`), which go through the forces
stage, not the FIELD path — so `DEMO-ANALYSIS-PASS (18/18)` does not exonerate shape force regions.
A FORCE-role probe is the next measurement, and it belongs in Wave 0.

---

## 3. Waves

Ordered so that each wave's verification is cheap and independent, and the risky numerical
work happens while the test scaffolding is fresh.

### Wave 0 — Red-first scaffolding (no production code touched)

Build the tests that prove the bugs, and watch them fail. This is what makes the rest durable.

| New/edited test | Asserts | Lane |
|---|---|---|
| `tests/blender_field_source_gate.py` **(new)** | A FIELD-mode source produces `total_fluid > 0` and tracked faces have non-zero weight. Fails today: measured `total_fluid = 0`, `zero_weight = 8448/8448`. | Blender + CUDA, new gate lane |
| `tests/blender_pressure_divergence_gate.py` **(new)** | `‖div u‖` after projection ≤ 1e-3 × before, on a fluid/air/solid mix; and that a *stalled* solve still reduces divergence. Fails today: stall → `338.95 → 338.95`. | Blender + CUDA, new gate lane |
| `tests/test_cache_io.py` **(extend)** | Frame written during live playback is addressed by **timeline** frame. Fails today: timeline 2 → `fluidsim_00001.npz`. | host pytest |
| `tests/blender_handler_persistence_gate.py` **(widen)** | **Every** add-on app handler survives save→reopen. Fails today: `render_init` `[[...]] → []`. | existing gate lane |
| `tests/blender_background_acceptance.py` **(extend)** | Every dynamic-enum property resolves to a valid item on a fresh scene; `snippet_load_preset()` succeeds without manual assignment. Fails today: `RuntimeError: No library snippet selected`. | existing lane |
| `tests/test_performance_hotpath_contracts.py` **(extend)** | `_watertight_triangles` is **not** called for a non-sink collider. Fails today: 8 calls / 4 frames. | host pytest |
| `tests/test_viewport_output_identity.py` **(new)** | A pre-existing wrong-typed object named `FluidSim Particles` does not kill the session. Fails today: `TypeError`, live silently off. | host pytest (stub bpy) |

**Verify:** `python -m pytest -q` (new host tests red), plus each new Blender lane run
directly and shown failing. Record the red output under `artifacts/corrections/wave0/`.

### Wave 1 — Sim blockers: field-mode sources (B1 + B2)

- **B1** `sim/kernels_fluid.py:152-155`: **delete the `else` branch** so
  `k_particle_to_grid_active` is purely additive (`|= VOXEL_FLUID | VOXEL_ACTIVE` when occupied),
  matching `paradigmParticleToGridActive`, which never clears the Fluid bit (§2.6A). Per-substep
  `voxel_type` zeroing in `_clear_grid_fields` already provides the clear, exactly as
  `coreClearScalarVoxelType` does in the original. **Do not** add a tracking-bit term to the
  predicate — that was my earlier, wrong proposal.
- **B1 follow-up (scope decision, not a blocker):** the tracked-bit latch (§2.6B) needs the
  original's **two** voxel-type arrays (live + persistent). Options: implement the second array and
  the real latch, or keep one array and record the divergence in the feature ledger. Recommend the
  ledger for now — adding a persistent per-voxel array is a memory/architecture change that should
  not ride along with a blocker fix. Either way, split the double alias so
  `fluidSimResetSourceShapeTrackedVoxelType` and `fluidSimClearJustInitializedVoxelTypeTracked` stop
  reporting as covered by one unlaunched kernel.
- **B2** per **D1(d)/(e)**. If (d): apply field-source velocity *after* `k_normalize_mac` with a
  non-zero face weight, in both `k_source_shapes` (`sourcing.py:594-596`) and `k_mesh_source_field`
  (`:748-750`) — fix both or neither — and add the ledger entry marking it a port extension.
- **L4**: `docs/technical_report.md`, `docs/feature_ledger.md` (new), and any UI tooltip that
  describes field-source velocity.

**Verify:** `blender --background --factory-startup --python tests/blender_field_source_gate.py`;
then `python -m pytest -q tests/test_sourcing_pipeline.py tests/test_mesh_source_pipeline.py
tests/test_source_emission_lifecycle.py`; then full demo analysis (D3 applies).

### Wave 2 — Sim blockers: projection integrity (B3) + HIGH sim defects

- **B3** `sim/pressure.py:1669-1670`: on `stalled`/`max_iterations`, keep best-so-far
  pressure and project anyway (it strictly reduces divergence); zero-and-skip **only** for
  `PRESSURE_STATUS_INVALID` (non-finite). `should_project_pressure()` (`:1433-1435`) must
  return True for those statuses. Surface the status to the profiler/UI instead of degrading
  silently.
- **B3b** `solve_jacobi` (`:1712-1723`) currently reports `pressure_valid=True` with
  `last_residual = -1.0` — no convergence check at all. Give it a real residual.
- **HIGH-5** `kernels_fluid.py:1275-1297`: double-buffer `k_forces_speed_limit` (read/write
  race on neighbor faces), following the pattern `_forces` already uses for diffusion.
- **D2** closed-domain walls.
- **MEDIUM-6** count CFL-clamped faces into `FrameStats` so the clamp stops being silent.

**Verify:** `blender --background --factory-startup --python tests/blender_pressure_divergence_gate.py`;
`python -m pytest -q tests/test_pressure_projection_regressions.py`; determinism spot-check
(the race fix should *reduce* run-to-run variance); full demo analysis expecting
`DEMO-ANALYSIS-PASS (18/18)`.

### Wave 3 — Integration correctness (B4, B5, H1)

- **B4** `io/cache.py:150-158`: add an explicit `frame=` parameter to `write_frame`
  (defaulting to `solver.frame` so `test_cache_io.py` and the export path stay green), and
  have the live path in `scene_bridge.py:193-197` pass `self.start_frame + self.solver.frame`.
  Emit/refresh `write_sequence_metadata` as the live range grows.
  *Blast radius is small* — verified that only `test_cache_io.py` pins the naming, via a
  stubbed solver frame.
- **B5** `blender/scene_inputs.py:38-44,196`: make `watertight` **lazy** (store a callable or
  `None`, evaluate only in the SOLID-sink branch at `:333-338`, memoize into the entry) **and**
  vectorize it (`np.sort` the three edge pairs → `np.unique(..., return_counts=True)`).
  Both, not either: lazy removes it from the common path, vectorized makes the sink path sane.
- **H1** `scene_inputs.py:165-169,498-502`: filter degenerate triangles (`tris = tris[valid]`,
  fail only if none survive) and surface `_error` onto `obj.fluidsim` + `fs.status` as the
  source path already does. Note this finding is **[NOT REPRODUCED]** — see §5.
- **M4** cache key compares original matrix but stores evaluated (`:130-143`) → compare both
  against the same source. **M5** store a hash, not raw vertex bytes.

**Verify:** `python -m pytest -q tests/test_frame_output.py tests/test_scene_bridge_cache_playback.py
tests/test_performance_hotpath_contracts.py tests/test_scene_bridge_collider_cache.py`;
`blender --background --factory-startup --python tests/blender_cache_export_pipeline.py`;
re-run the B5 probe and expect watertight calls = 0 for a plain collider.

### Wave 4 — UI / lifecycle correctness

- **`@persistent` on `_on_render_init`** (`blender/render.py:43-49`) — one decorator, locked
  by the widened Wave 0 gate.
- **Snippet enum default** (`properties.py:228-255`): emit a real leading item
  (`("NONE", "Select a snippet…", "")`) that the load operator skips, so index 0 is never a
  separator.
- **Prefs probe cache** (`preferences.py:10-18,30-37`): module-level cache + an explicit
  "Re-detect Device" operator. Live cost today: **426.6 ms per redraw**.
- **Coalesce heavy `update=` callbacks** (`properties.py:142-151` → `scene_bridge.py:696-735`):
  set a dirty flag, do the rebuild once from a one-shot timer / next frame-change. Live cost
  today: **174–191 ms per drag tick**.
- **Output object identity** (`viewport.py:64-77`): resolve by a custom ID-property tag scoped
  to the output collection, and recreate the object when `obj.type` mismatches the target data.
- **Status/live ordering** (`scene_bridge.py:310-317`): set `fs.live = False` *before* writing
  `fs.status`, so the real error survives.
- Cheap hygiene from the review: `__package__`-derived root instead of the hard-coded module
  name (`scene_bridge.py:759-766`, `render.py:52-56`, `operators.py:18-26`); `poll()` on
  session-dependent operators; `UNDO` on `apply_safe_settings`; `snippet_apply` returning
  `FINISHED`; unused `import json`; `FLUIDSIM_PT_surface`/"Whitewater" id-label mismatch.

**Verify:** `python -m pytest -q tests/test_operator_panel_behavior.py tests/test_scene_bridge_settings.py`;
`blender --background --factory-startup --python tests/blender_handler_persistence_gate.py`;
`blender --factory-startup --python tests/blender_gui_acceptance.py` expecting 35/35; re-run
the drag probe expecting per-tick cost to collapse.

### Wave 5 — Performance on the interactive path

Measured, not speculative: `_values_to_colors` 60–82 ms/frame, `_nearest_values` 199 ms @5k /
427 ms @50k, plus a full particle readback before display decimation.

- Sub-sample or `np.partition` the 5/95 percentile range and cache it across frames
  (`viewport.py:91-92`).
- Replace the Python KD-tree with a Warp GPU nearest lookup or a vectorized grid hash
  (`viewport.py:120-133`) — this also restores the GPU-resident claim.
- Push the display stride into the host snapshot so decimation happens in a Warp kernel
  *before* `.numpy()` (`viewport.py:417-418` vs `:537-542`).
- `wp.copy(..., count=self._nvox)` on the collision-impact copies (`solver.py:1514-1516`) —
  ~50 MB/substep of pure waste today.
- Reuse the pointcloud in `_swap_points` when `pc_old.users == 1` (`viewport.py:261-285`),
  removing per-frame ID create/free inside a handler.
- Pack `num_leaves`+`overflow` into one readback (`fields.py:504-514`).
- Move `status`/`stat_*` mirrors to `WindowManager` so playback stops dirtying the .blend.

**Verify:** `FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py` against the
existing `--max-mean-view-ms 120` budget, **and** promote that lane out of opt-in (see Wave 6),
otherwise these wins are unprotected.

### Wave 6 — Gate truthfulness + doc corrections

- **Require success markers.** Add `expect_marker=` to `_run` (`tools/run_production_gate.py:66-83`)
  and pass it for every `reject_output=True` lane. Absence-of-failure is not proof.
- **Fix the dead/missing markers:** `VIEWPORT-GATE-FAIL` → `VIEWPORT-2FRAME-GATE-FAIL`; add
  `HANDLER-PERSISTENCE-GATE-FAIL`; give `blender_memory_safety_gate.py` a real FAIL marker
  instead of relying on a bare `assert` traceback.
- **Harden the playback validator** (`tools/validate_gui_playback_report.py`): delete the report
  before the lane; re-derive the numeric invariants instead of trusting one `passed` boolean;
  check `report["blend"]` against the expected path; record `timed_out` in the probe.
  Then **rewrite its docstring**, which today overstates what it proves.
- **Promote the user-path profile lane** out of `FLUIDSIM_USER_PATH_PROFILE=1` opt-in.
- **Doc corrections:** `docs/technical_report.md:305-308` ("reads back only small statistics" —
  false for viewport/cache paths); `sim/pressure.py:12-16` (claims a V-cycle that has no call
  sites); mark the unreachable multigrid `register_fn` entries with explicit notes; note the
  gate/pytest split in `AGENTS.md` (default gate runs 3 of 48 test modules).
- **Convert the grep-tests** to import-and-inspect where cheap.
- **Create `docs/feature_ledger.md`** (§2.7): every original kernel/feature marked ported /
  deferred / port-extension, seeded with FIELD mode, the tracked-bit latch divergence, the
  count-vs-weight P2G difference, the unreachable multigrid path, and the 37 never-dispatched
  kernels. Add a test that the ledger covers every manifest entry.
- **Dispatch-reachability test (per D5):** extend `tests/test_kernel_coverage.py` to assert each
  registry entry is a real `wp.Kernel` **and** to report the launched/never-launched split (AST or
  import-time instrumentation), failing if the never-dispatched count rises above a recorded
  baseline. Today's 37 becomes a ratchet, not a surprise.

**Verify:** `python tools/run_production_gate.py`, then `--include-gui`, reporting each lane's
command, exit code and exact marker separately. Deliberate negative test: break one lane's
marker and prove the gate now fails.

---

## 4. Correction ledger (all findings)

| ID | Verdict | Wave | Durability lock |
|---|---|---|---|
| B1 field sources inert | CONFIRMED | 1 | new `blender_field_source_gate.py` |
| B2 field source velocity | CONFIRMED | 1 (D1) | same gate asserts face weights |
| B3 projection fails open | CONFIRMED | 2 (D3) | new `blender_pressure_divergence_gate.py` |
| B4 cache frame offset | CONFIRMED | 3 | `test_cache_io.py` timeline-indexing case |
| B5 watertight on hot path | **FIXED** (§0) | — | `test_watertight_check_is_deferred_to_solid_sink_consumers` + semantics guard |
| HIGH closed-domain walls | UNTESTED | 2 (D2) | divergence gate + demo analysis |
| HIGH speed-limit race | UNTESTED | 2 | determinism assertion |
| HIGH `render_init` persistence | **FIXED** (§0) | — | widened handler-persistence gate, negative-tested |
| HIGH snippet enum default | CONFIRMED | 4 | enum-validity check, all dynamic enums |
| HIGH collider silent drop | NOT REPRODUCED | 3 | fix + test **only if** a reproducer lands (§5) |
| HIGH handler ID churn | UNTESTED | 5 | pointcloud reuse assertion |
| HIGH viewport CPU stalls | CONFIRMED | 5 | user-path profile budget, promoted to default |
| HIGH pre-decimation readback | CONFIRMED | 5 | readback-size assertion vs stride |
| HIGH prefs `draw()` probe | CONFIRMED | 4 | probe-call-count assertion |
| HIGH drag rebuild cost | CONFIRMED | 4 | per-tick cost assertion |
| MED CFL clamp silent | UNTESTED | 2 | clamped-face stat in `FrameStats` |
| MED full-capacity `wp.copy` | UNTESTED | 5 | active-work contract test |
| MED CUDA-only PCG | UNTESTED | 5 | device validation at session start |
| MED multigrid overclaim | CONFIRMED (dead code) | 6 | docstring + registry notes |
| MED output name collision | CONFIRMED | 4 | `test_viewport_output_identity.py` |
| MED status overwritten | CONFIRMED | 4 | asserts real error reaches `fs.status` |
| MED cache key matrix mismatch | UNTESTED | 3 | collider cache hit-rate assertion |
| MED vertex-bytes retention | UNTESTED | 3 | hash-based signature |
| MED ID writes in handler | CONFIRMED | 5 | `WindowManager` mirrors |
| MED blocking cache write | CONFIRMED | 5 | covered by profile budget |
| MED per-substep syncs | CONFIRMED | 5 | packed readback |
| Gate: no success markers | CONFIRMED | 6 | negative test on a broken marker |
| Gate: self-graded playback report | CONFIRMED | 6 | re-derived invariants + identity + freshness |
| Gate: dead markers | CONFIRMED | 6 | marker/emitter cross-check test |
| Gate: grep-as-contract | CONFIRMED | 6 | converted to import-and-inspect |
| **NEW: 37/157 kernels never dispatched** | CONFIRMED (AST-measured) | 6 (D5) | coverage test reports dispatch reachability, not just registration |
| **NEW: two original kernels aliased to one unlaunched kernel** | CONFIRMED | 6 (D5) | alias map asserts one canonical name per distinct behavior |
| Panel draw cost | **RETRACTED** | — | §5: record the retraction, change nothing |
| Stuck `hide_viewport` | **NOT REPRODUCED** | — | §5: downgraded to Low, no fix |

---

## 5. Handling the findings I got wrong

Durable also means not leaving false claims lying around, and not "fixing" things that
aren't broken.

- **Panel draw cost — RETRACTED.** Measured 1.356 ms for all 16 panels expanded; 14 are
  `DEFAULT_CLOSED`. No code change. The retraction is already recorded in
  `reviews/ADDON_REVIEW.md` with the measurement.
- **Stuck `hide_viewport` — NOT REPRODUCED.** The code asymmetry is real but produces no
  user-visible symptom I could trigger. Downgraded to Low. I will **not** change behavior
  here; I will add a comment at `viewport.py:596-621` documenting that hide flags are
  intentionally only written while inactive, so the next reader doesn't re-file it.
- **Collider silent drop (H1) — NOT REPRODUCED.** Blender's tessellation dropped my
  degenerate face before it reached the check. The `_error`-discarding path is real by
  inspection. Plan: land the *diagnostic* half (surface `_error` — safe, strictly more
  information) in Wave 3, but **defer the triangle-filtering half** until there is a real
  reproducer, because filtering silently changes collider geometry on a path I cannot
  currently exercise.

---

## 6. Risk register

| Risk | Mitigation |
|---|---|
| Wave 2 changes solver output → demo bands drift | D3 governs. Run full demo analysis before/after each numerical change, one change at a time so attribution is unambiguous. |
| B4 changes on-disk cache layout | Existing caches become mis-indexed relative to new ones. Bump/verify `CACHE_FORMAT_VERSION` handling, or write sequence metadata that records the indexing basis. Needs an explicit call in the backlog task. |
| D1 option (a) adds a kernel launch per substep | Measure with `tools/profile_gpu_solver.py` before/after; reject if the regression is material. |
| Wave 5 touches the hottest code | Land behind the profile budget; each change measured individually, not as a batch. |
| Gate hardening could red-light the build for pre-existing reasons | Expected and desirable — but land Wave 6 **last** so a newly-strict gate reports the fixed tree, not the broken one. |
| My own probe scripts are throwaway | Promote the valuable ones into `tests/` as real lanes (Wave 0), rather than leaving evidence in `artifacts/`. |

---

## 7. Out of scope unless you say otherwise

- Adding `blender_manifest.toml` / extension packaging (forward-compat debt, not a defect).
- Wiring up the unreachable multigrid V-cycle (Wave 6 corrects the *claim*; implementing the
  algorithm is a feature).
- Making long operators modal with progress/cancel (real UX gap, but a feature-sized change).
- Restoring `userpref.blend` to its pre-review bytes — not possible, disclosed in
  `reviews/ADDON_REVIEW.md`.
