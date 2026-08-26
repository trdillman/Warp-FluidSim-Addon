# AGENTS.md

Build the Blender 5.1 FluidSim Warp add-on from the synthesized requirements only. Keep simulation GPU-resident, support interactive viewport playback, preserve the listed FluidSim features, and use the synthesized Python modules for enums, kernel inventory, dispatch/custom-kernel contracts, UI behavior, profiling, and validation targets. The live field-storage contract is a sparse tiled leaf grid: fixed 8x8x8 leaf tiles addressed through a flat leaf table, without hierarchical tree storage.

## Active project: native GPU interop

- Product authority remains the Blender source tree's `GOALS.md`; sequence and completion gates live in
  `GPU_BUFFER_INTEROP_NATIVE_FEATURE_PLAN.md` and `GPU_BUFFER_INTEROP_ACCEPTANCE.md` there.
- `docs/NATIVE_GPU_INTEROP.md` records the current consumer baseline and reproduction commands; it is not
  a separate plan or completion authority.
- Native presentation is bounded to Windows Vulkan + NVIDIA CUDA/Warp for this release. Preserve the CPU
  point-cloud path and every existing surface, whitewater, cache, export, render, and display feature.
- Reuse the Blender-owned external buffers and directional timelines. Route product drawing through the
  existing PointCloud/Draw Manager path before adding a generic provider ABI or new geometry type.
- A passing import, screenshot, draw handler, or benchmark is not feature completion. Require every gate
  in the Blender acceptance contract, including parity, lifecycle, stress, packaging, and clean-install
  evidence.

## Verified workflows

- From the repo root, run `python tools/run_production_gate.py` for the default headless gate; add `--include-gui` for the full production gate.
- For interactive playback performance changes, run `FLUIDSIM_USER_PATH_PROFILE=1 python tools/run_production_gate.py` or the focused `blender --background --factory-startup --python tools/profile_user_path.py -- --frames 6 --voxel-size 0.08 --max-particles 200000 --collider-subdiv 2 --json-out artifacts/user_path_profile.json` profile.
- Run the full Python regression suite with `python -m pytest -q`; use the production gate for Blender/CUDA/runtime changes because it invokes Blender 5.1's bundled Python and acceptance lanes.
- For GPU solver timing changes, run `"C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe" tools/profile_gpu_solver.py --device cuda:0 --frames 8 --warmup 2 --json-out artifacts/gpu_solver_profile.json`.
- Build demos with `blender --background --python tools/build_demo_scenes.py`; use `-- --list-scenes`, `-- --standard-cache-scenes`, or `-- dam_break slosh_tank` to select targets.
- Verify saved demo files with `blender --background --python tools/check_demo_blends.py`, then analyze outputs with `python tools/analyze_demos.py`.
- Native interop runtime gates require the interop-enabled Blender build with `--gpu-backend vulkan`;
  see `docs/NATIVE_GPU_INTEROP.md` for exact commands and evidence paths.
