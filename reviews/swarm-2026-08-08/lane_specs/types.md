# Lane: Type System
- **Dimension name:** Type System
- **ID prefix:** TYPE
- **Output path:** C:/Users/Tyler/Downloads/liquid_warp_blender51_port/liquid_warp_blender51_port/reviews/swarm-2026-08-08/findings/types.json

Follow `_common.md` in this same directory.

## Scope (review only these files)
- fluidSim_warp_blender51/properties.py  (~1318 — RNA property definitions)
- fluidSim_warp_blender51/sim/enums.py  (~512)
- fluidSim_warp_blender51/sim/quality_contract.py  (~226)
- fluidSim_warp_blender51/blender/scene_inputs.py  (~890 — RNA scene input binding)

## Dimension Focus
- RNA property correctness: FloatProperty/IntProperty subtypes, min/max bounds, units (VELOCITY/LENGTH/TIME), step, precision, out-of-range or unit-mismatched defaults.
- Enum consistency: do properties.py EnumProperty items match sim/enums.py definitions (keys, order, int mapping)? Mismatched enum string<->int is a common defect.
- quality_contract.py: are declared contracts actually enforced, and are types consistent with properties?
- Type coercion at Python <-> warp <-> Blender boundaries: numpy dtypes vs warp types (wp.float32 vs float64), implicit conversions that lose precision or silently widen.
- update callbacks: correct types on read/write; any callback that can raise on uninitialized RNA?
- Serialization safety of property types (PointerProperty/CollectionProperty) for preset save/load.
