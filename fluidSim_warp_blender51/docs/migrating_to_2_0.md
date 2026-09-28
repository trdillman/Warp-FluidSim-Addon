# Migrating from FluidSim 1.x to 2.0

2.0 is a breaking release. It drops the compatibility code that 1.x carried
for older files, caches and runtimes. Most of the work happens once, in
FluidSim **1.1.0**, the bridge release.

## Summary

| You have | What to do |
|---|---|
| A `.blend` saved by FluidSim 1.1.0 or later | Nothing. Open it in 2.0. |
| A `.blend` saved by FluidSim 1.0.x | Open it in Blender 5.1 with FluidSim 1.1.0, save it, then open it in 2.0. |
| A cache baked by 1.1.0 or later (cache v2) | Nothing. It plays back and resumes in 2.0. |
| A cache baked by 1.0.x (`fluidsim_*.npz` + `fluidsim_sequence.json`) | Free it and bake again. 2.0 refuses to play it. |
| Custom Warp Snippets from 1.0.x | Open and save the file in 1.1.0; it converts them to Custom Effects. |
| Python scripts that set FluidSim settings | Check which group each setting lives on (see below). |
| An ST-FLIP scene | Check **Variable Projection** (see below). |

## 1. Install

- 2.0 requires **Blender 5.2 LTS**. Blender 5.1 is no longer supported.
- Warp is bundled. Blender installs the checksum-locked Warp 1.17.0 wheel with
  the extension. Nothing needs to be installed by hand, and the Warp you
  installed into Blender 5.1's Python for 1.x is neither used nor needed.
- If FluidSim reports that it "requires NVIDIA Warp 1.17.0 or newer" but loaded
  an older version, a Warp left in Blender's Python user site-packages is
  loading ahead of the bundled one. Remove it and restart Blender.
- You need an NVIDIA driver that supports CUDA 12 (R525 or newer).
- Windows x64 is certified by a recorded GPU gate run. Linux x64 is packaged
  from the same source but is not certified yet.

Keep Blender 5.1 with FluidSim 1.1.0 installed until every file you care about
has been re-saved (section 2).

## 2. Files saved by 1.0.x

1.0.x kept domain settings on the scene. 1.1.0 moved them to the Domain object
and migrates a 1.0.x file when it opens. 2.0 no longer contains that migration.

When 2.0 opens a 1.0.x file it does not guess. It:

- prints a load message and shows an alert on the Setup and Physics tabs;
- refuses to start a simulation if the file still holds Custom Warp Snippets
  that 1.1.0 never converted;
- keeps the stored data, and keeps it through a save in 2.0, so the file can
  still be recovered in 1.1.0.

To migrate: open the file in Blender 5.1 with FluidSim 1.1.0, check the
simulation, save, then open it in Blender 5.2 with FluidSim 2.0.

## 3. Caches

2.0 plays and resumes only cache v2, the format 1.1.0 already writes. A 1.0.x
cache is recognized and refused with a message saying it was written by
FluidSim 1.0. Free it from the Cache panel and bake again.

Bakes with an animated source or collider now play back and resume. In 1.x a
production bake of such a scene played frame 1 and then refused the rest as
"baked from different scene settings". In 2.0 a cache frame is invalidated
by a change to the simulation settings. Moving a source or collider after a
bake does **not** invalidate it; bake again yourself if you change the
animation.

## 4. Custom Warp Snippets

The 1.x snippet runtime, its library, the snippet validator and the
`fluidsim.snippet_apply` operator are gone. Custom Effects replace them.
1.1.0 converts saved snippets to Custom Effects when it opens a 1.0.x file, so
a file re-saved in 1.1.0 already holds the converted effects.

## 5. Settings and scripts

- **Setting a value on the wrong settings group now raises an error.** Since
  1.1.0 scene-wide settings live on `scene.fluidsim` and per-domain settings
  on the domain object's `fluidsim_domain`. In 1.x, a script that wrote
  `scene.fluidsim.voxel_size = 0.05` silently created a plain Python attribute
  that the solver never read. 2.0 raises `AttributeError` and names the right
  group. Fix the script to write the group the error names.
- **Removed settings:** the snippet library and shortcut settings, the
  `schema_version` stamp and the legacy `snippets` store.
- **New read-only status:** `stat_native_solver_message` reports when the
  pressure or viscosity solve fell back from CUDA graphs to a slower path.
- **Height-field back mode:** *Infinite* was never runnable and has been
  removed. A file that saved it reads back as empty and reports an actionable
  error. Choose Flat or Profile.
- **One default per setting.** Scene defaults, solver defaults and fallbacks now
  agree, and length and speed settings carry Blender units. Values you set
  explicitly are unchanged.

## 6. ST-FLIP

ST-FLIP is labelled **Experimental** in 2.0 because its reference gates have
not run. **Variable Projection** is now off by default. A saved ST-FLIP scene
that never changed Variable Projection runs with it off. To keep the 1.x
result, turn it back on in **Spatiotemporal Processing**.

## 7. Results

For the same scene, 2.0 gives the same liquid simulation results as 1.1.0 on the
shipped demos, with one intended difference: whitewater particles near a
collider edge follow a corrected collision normal, so whitewater counts can
differ by a few particles. Demo bakes are bitwise reproducible on the same
GPU, driver and Warp build.
