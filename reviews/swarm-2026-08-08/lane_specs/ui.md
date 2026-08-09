# Lane: UI/UX & Operators
- **Dimension name:** UI/UX & Operators
- **ID prefix:** UIUX
- **Output path:** C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port/reviews/swarm-2026-08-08/findings/ui.json

Follow `_common.md` in this same directory.

## Scope (review only these files)
- fluidSim_warp_blender51/panels.py  (~1311)
- fluidSim_warp_blender51/operators.py  (~1573)
- fluidSim_warp_blender51/blender/viewport.py
- fluidSim_warp_blender51/blender/render.py
- fluidSim_warp_blender51/blender/materials.py
- fluidSim_warp_blender51/blender/colliders.py
- fluidSim_warp_blender51/diagnostics/profiler.py  (~215)

## Dimension Focus
- Operator poll() guards: must fail-closed (return False) when context is invalid, not raise. Operators running without validation are defects.
- Error feedback: do operators report errors to the user (self.report('ERROR', ...)) or fail silently?
- UI truthfulness: NOTE — this project has a heavy backlog of "POLICY_DRIFT" tasks about UI making false claims. Flag any panel control / enum / display that claims behavior the code does not implement (greyed-out-but-functional, functional-but-no-UI, misleading labels).
- Undo/redo: operators should declare undo support where state changes.
- State management: missing depsgraph tag/refresh after edits, panels reading stale RNA.
- Registration: operator/panel bl_idname correctness and register order.
