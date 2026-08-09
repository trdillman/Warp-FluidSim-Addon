# Lane: Architecture & Style
- **Dimension name:** Architecture
- **ID prefix:** ARCH
- **Output path:** C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port/reviews/swarm-2026-08-08/findings/arch.json

Follow `_common.md` in this same directory.

## Scope (review only these — Glob for `__init__.py` files)
- fluidSim_warp_blender51/__init__.py  (addon register/unregister entry; has uncommitted edits)
- fluidSim_warp_blender51/blender/blender_manifest.toml
- every __init__.py under fluidSim_warp_blender51/**
- fluidSim_synth/*.py  (distilled blueprint/CONTRACT reference: dispatch_contract, custom_kernel_contract, enums, fluid_meta, blender_addon_blueprint)
- Cross-cutting pass over fluidSim_warp_blender51/**: import hygiene, dead/unused code, circular imports, coupling/cohesion, naming, magic constants.

## Dimension Focus
- **Registration correctness:** register/unregister symmetry, bl_idname collisions, register ORDER, classes left registered after unregister (leaks), handlers registered without removal.
- **Manifest correctness:** blender version compat, permissions, wheels, type extension.
- **Conformance:** does fluidSim_warp_blender51 honor the fluidSim_synth contracts, or has it drifted?
- **Module structure:** circular deps, god-objects, unclear ownership.
