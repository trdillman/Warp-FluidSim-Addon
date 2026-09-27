# FluidSim artist workflows

FluidSim's everyday controls describe the result you want. Solver admission,
effect preparation, GPU setup, and normal output preparation happen inside
those actions.

## Set up and preview

Select one or more mesh objects, open **Scene Properties > FluidSim > Setup**,
and choose **Set Up & Preview**. Confirm the setup mode in the dialog. FluidSim
creates the owned scene setup and starts real Blender timeline playback.

Use **Set Up Only** when you want to inspect or adjust the generated domain,
source, and outputs before playback. Setup remains committed if the scene was
created successfully but playback cannot start; the status message explains
the playback problem.

## Add and adjust an effect

Open **Custom Effects**, press **Add Effect**, choose an effect, and adjust its
artist controls. Press **Preview**. Eligible draft effects are prepared as part
of simulation admission, so the ordinary workflow does not require a separate
Test Draft or Compile & Apply step.

Built-in effects are trusted. Imported executable effect packages keep their
existing review and trust checks; use **Review & Trust** only after inspecting
the exact package. Import, update comparison, export, and local trust controls
remain available in the Effect Packages panel. The old one-frame test and
manual compile/revert tools are retained under **Advanced > Effect
Engineering** for troubleshooting.

## Preview, pause, resume, apply, and restart

- **Preview** prepares the current scene and starts Blender timeline playback.
- **Pause** stops the timeline but keeps the owned solver and current frame.
- **Resume Preview** continues that paused solver when the current changes are
  compatible. If changes require a restart, FluidSim asks before discarding
  progress.
- **Apply Changes** uses the shared pending-settings policy to make the least
  destructive valid transition. A safe live edit does not ask for confirmation;
  a structural edit confirms the actual preview progress that will be lost.
- **Restart Simulation** explicitly discards preview progress and starts again
  at the scene start frame. Because the action already states its consequence,
  it does not add a second confirmation.

Examples of live-compatible adjustments include FLIP Ratio and many force,
pressure, viscosity, and timing controls. Voxel Size, Quality, particle capacity,
and Liquid Sampling change simulation structure or initial state. The Pending
Settings panel is authoritative: it shows the action and consequence for the
current combination of edits.

## Cached preview and production bake

**Preview Cache** validates the selected cache frame and starts timeline
playback without allocating a live solver. Pause and resume act on that cached
timeline; an invalid cache stays paused and reports the error. Cached preview is
non-authoritative review, not a production bake.

**Production Bake** is a separate sequential, authoritative cache-writing
workflow. Its range and storage controls live under **Cache & Production Bake**.
Starting or restarting a bake may require confirmation when an existing managed
cache would be replaced. FluidSim does not silently delete a cache. Baking the
`dam_break` or `crown_splash` demo twice on the same GPU gives bit-identical
frames. Other scenes may still differ slightly between bakes: mesh emitters and
color mixing accumulate on the GPU in arrival order, and results change across
GPUs and Warp versions.

## Basic, Advanced, and diagnostics

Basic mode keeps setup, quality, artist parameters, outputs, preview, and bake
actions visible. Switch the Interface control to **Advanced** to reveal manual
solver initialization, release, single stepping, forced render-node repair,
detailed output-attribute inventories, effect engineering, profiling,
benchmarks, reports, and the headless bake command.

Normal output preparation preserves user-authored node and material changes.
When FluidSim reports an output-specific error, **Repair Outputs** remains next
to that error. Essential runtime and ownership errors stay visible in Basic
mode.

## Context and ownership limits

Preview needs a Blender window showing the active scene and a 3D View or
animation editor that can start the timeline. A different scene cannot take over
an owned live solver, cache playback, or production bake. Switch to the owning
scene or stop its activity first. If no suitable editor context exists, FluidSim
keeps a successfully prepared scene paused and reports why playback could not
start.
