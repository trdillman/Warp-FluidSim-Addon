# Source-shape spec, extracted from `fluidSim.cl` + `paradigm.hda`

Authoritative reference for porting `paradigmSourceShapes`. Extracted directly from
`fluidSim.cl` (20,640 lines / 772,693 bytes) and the Houdini asset `paradigm.hda`.
Line numbers are `fluidSim.cl` lines.

Purpose: this repo's FIELD-mode source behaviour was implemented without a coherent
contract, and three successive attempts to specify a fix from secondary sources
(kernel-name inventory, dispatch design doc, product plan) each rested on a wrong
premise. This file records the primary-source semantics so that stops happening.

---

## 1. Enums

`Enum_Core_SourceShape_Type` (`:10520-10800`) — the shape's role:

| Value | Name |
|---:|---|
| 0 | `Source` |
| 1 | `Collision` |
| 2 | `Sink` |
| 3 | `Activation` |
| 4 | `CameraFrustum` |

This matches the port's `stype` exactly, including `3`/`4` mapping to activation-style
`VOXEL_ACTIVE` marking.

`Enum_Core_SourceShape_SourceMode` (`:10841-10890`):

| Value | Name |
|---:|---|
| 0 | `Field` |
| 1 | `Particles` |

Exposed in `paradigm.hda` as parm `sourceMode`, label **"Mode"**, `type ordinal`,
`default 0`, `hidewhen "{ type != 0 }"` — i.e. only meaningful for `type == Source`.

`Enum_Core_ParametersBlendMode` (`:9765`): `Add = 0`, `Replace = 1`, then `Maximum`,
`Pull`, `Blend` (names confirmed from the blend function; their numeric values were not
individually confirmed).

**`CoreSourceShape` has 100 fields and no `mode`/`sourceMode` member.** The mode lives in
the per-dispatch parameter struct (`a_parametersSourceShape[0].sourceMode`), not per shape.

---

## 2. `sourceMode == Field` — the complete branch (`:11636-11650`)

```c
if (a_parametersSourceShape[0].sourceMode == Enum_Core_SourceShape_SourceMode_Field) {
    if (a_voxelType[l_index] & Bit_Core_VoxelType_Collision) return;
    if (a_voxelType[l_index] & Bit_Core_VoxelType_Sink)      return;
    if (a_parametersSourceShape[0].sourcePressure.enable) {
        float l_fieldValue = coreReadFieldValue(l_index, a_pressure);
        l_fieldValue = paradigmSourceShapeSourceField(
            a_parametersSourceShape[0].sourcePressure, l_fieldValue, l_sourcePressureValue);
        coreWriteFieldValue(l_fieldValue, l_index, a_pressure);
    }
}
```

That is the entire branch. Verified by enumerating every `coreWriteFieldValue` inside it:
**one write, to `a_pressure`, gated on `sourcePressure.enable`.**

Field mode therefore:

- **does NOT** set `Bit_Core_VoxelType_Fluid`
- **does NOT** set `Bit_Core_VoxelType_Active`
- **does NOT** set `Bit_Core_VoxelType_SourceRegionTracking`
- **does NOT** write MAC velocity (`a_velocity_*`)
- **does NOT** write temperature, density, colour, or viscosity
- **does** blend a pressure value into the existing pressure field, skipping
  collision and sink voxels

Consistent with the HDA help text: *"Influence fields allow you to add extra forces
without sourcing any density."* Field mode is a **pressure-influence mode**. Liquid is
created only by `Particles` mode.

### Blend semantics — `paradigmSourceShapeSourceField`

```c
float l_sourceValue = a_sourceValue * a_settings.scale;
Add:      output = fieldValue + (sourceValue * settings.timeStep);
Replace:  output = sourceValue;
Maximum:  output = max(sourceValue, fieldValue);
Pull:     if ((field < source && source > 0) || (field > source && source < 0))
              output = lerp(sourceValue, fieldValue, settings.blendAmount);
Blend:    output = lerp(sourceValue, fieldValue, settings.blendAmount);
```

Note `Add` is time-scaled (`* timeStep`); `Replace`/`Maximum` are not. `Pull` only moves
the field toward the source when the field lags the source in a sign-consistent direction.

---

## 3. `sourceMode == Particles` — emission gate (`:11651` onward)

```c
else if (sourceMode == Enum_Core_SourceShape_SourceMode_Particles) {
    if (a_parametersSourceShape[0].regionAwareFill) {
        voxelType_t l_voxelTypeTracked = a_voxelTypeTracked[l_index];
        a_voxelType[l_index]        |= Bit_Core_VoxelType_SourceRegionTracking;
        a_voxelTypeTracked[l_index] |= Bit_Core_VoxelType_SourceRegionTracking;
        if ((l_voxelTypeTracked & Bit_Core_VoxelType_SourceRegionTracking) != 0) return;
    }
    if (a_voxelType[l_index] & Bit_Core_VoxelType_Collision) return;
    if (a_voxelType[l_index] & Bit_Core_VoxelType_Sink)      return;

    int l_particleToSource = a_parametersSourceShapeExtra[0].particleDensity;
    if (a_parametersSourceShape[0].cullParticlesWithNoise)
        l_particleToSource = (int)((float)l_particleToSource * l_sourceValueNoise);
    int l_voxelParticleCount = a_particleCount[l_index];
    if (!a_parametersSourceShape[0].overlapParticles) {
        l_particleToSource -= l_voxelParticleCount;   // top up to target
        if (l_particleToSource <= 0) return;
    }
    l_voxelParticleCount += l_particleToSource;
    a_particleCount[l_index] = ...
```

Key semantics:

- **`VOXEL_SOURCE_REGION_TRACKING` is a one-shot per-voxel fill latch, owned by
  `Particles` mode and gated on `regionAwareFill`.** It is not a fluid marker and has
  nothing to do with Field mode. Two arrays are required: live `a_voxelType` and
  persistent `a_voxelTypeTracked`. `paradigmResetSourceShapeTrackedVoxelType` (`:17855`)
  releases the latch where the live bit is absent.
- **`particleDensity` is target particles per voxel**, and `overlapParticles == false`
  means "top up to the target" rather than "add this many". This confirms the repo's own
  note that `Emit Rate` is really target-particles-per-voxel — the label was wrong, but
  the semantics match the original.
- Emitted particles receive the source velocity (`a_particleVelocity_*` at
  `l_newParticleIndex`, `:11753-11755`) — this is the working velocity path.

---

## 4. Port status after the correction

| Aspect | Original | Corrected port |
|---|---|---|
| Field mode voxel bits | none | none; FIELD compatibility path is non-liquid |
| Field mode writes | `a_pressure` only | pressure sourcing is deferred/not exposed; no incorrect MAC/attribute writes |
| Tracking bit owner | `Particles` + `regionAwareFill` | not implemented; recorded as a deferred two-array latch |
| Tracked storage | two arrays (live + persistent) | one `voxel_type` array; persistent tracked array remains deferred |
| `ParticleToGridActive` | additive; never clears Fluid | additive; occupied cells are marked Fluid/Active, no bits are cleared |
| FORCE role | force influence runs in the forces stage | analytic shape acceleration runs after P2G on valid weighted MAC faces (`k_source_shape_forces`) |

`Bit_Core_VoxelType_Fluid` (= 4) appears 13 times in `fluidSim.cl` and is **never cleared
by any kernel** — set by `paradigmParticleSetGridData` (`:18563`), propagated by
`paradigmExpandFluidMask` (`:18098`), reset only by whole-buffer `coreClearScalarVoxelType`.

---

## 5. Consequences for the port's own claims

- An earlier source-emission product plan (since removed) previously framed the defect as "`Mode = Field`
  creates liquid ... nonconformant". That premise is corrected: Field is pressure influence,
  creates no liquid, and is hidden from the normal Sources UI until pressure sourcing exists.
- A regression test asserting "a FIELD-mode source produces fluid voxels" would encode the
  **wrong** expectation. An earlier draft of this plan proposed exactly that test; it was
  never built.
