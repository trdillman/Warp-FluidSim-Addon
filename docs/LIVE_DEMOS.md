# Show any demo live

One command opens a demo the way an artist would see it: the latest add-on from
this checkout, installed into a brand new temporary Blender profile, Live
Simulation on, cache playback off, the Liquid Deck on, and Play pressed in the
3D Viewport. The solver steps as the timeline plays. Nothing is baked or
replayed. Your normal Blender profile is never touched, and the temporary
profile is deleted when Blender closes.

```powershell
python tools/demos/live_demo.py --list
python tools/demos/live_demo.py --blender "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --demo dam_break
```

Demos: `dam_break`, `crown_splash`, `paddle_stir`, `chromatic_vortex`,
`buoyancy_ladder`, `water_wheel`, `viscosity_race`, `deep_plunge`,
`capillary_droplet` (all tracked in `demos/`), and the opt-in `tsunami_parthenon`,
`upres_cove` and `nets_showcase`.

## Opt-in demos

Those three are not tracked `.blend` files. Add `--build` to author the scene
first (it lands in `artifacts/review_demos/`), or leave it off when it is already
there.

```powershell
python tools/demos/live_demo.py --blender ... --demo nets_showcase --preset draft --build
```

`nets_showcase` also needs the Surface Nets backend. The launcher sets
`FLUIDSIM_SURFACE_BACKEND=nets` for you and the in-Blender script refuses to
start the demo without it. **Speed:** the 1 mm hero takes about 0.75 s per step on
a 4080 SUPER, so live it plays at roughly one frame a second. The 2 mm `draft`
is the one to show at a comfortable pace.

## What it does, in order

1. Builds the extension zip from this checkout (`tools/release/build_extension.py`,
   Python checks skipped).
2. Creates a temporary profile and installs the zip into it, then fetches the
   certified Warp (checksum locked) into that profile.
3. Starts Blender on the demo. A small script (`live_demo_boot.py`) checks the
   add-on that loaded is the one in the temporary profile, turns on Live, turns off
   cache writing and playback, shows the Liquid Deck, switches the viewport to
   Material Preview through the camera, and presses Play.
4. Every five seconds it writes `live_status.json` (frame, particles, step time,
   Live and Deck state, add-on path) into the profile, or to `--report`, so a run can
   be checked without watching the window.

Options: `--shading SOLID` for a lighter viewport, `--no-camera-view`,
`--keep-profile`, `--prepare-only` (build and install, do not open Blender).
It needs a CUDA GPU, like every FluidSim run.
