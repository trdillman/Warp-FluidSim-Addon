# AGENTS.md

Build the Blender 5.1 FluidSim Warp add-on from the synthesized requirements only. Keep simulation GPU-resident, support interactive viewport playback, preserve the listed FluidSim features, and use the synthesized Python modules for enums, kernel inventory, dispatch/custom-kernel contracts, UI behavior, profiling, and validation targets. The live field-storage contract is a sparse tiled leaf grid: fixed 8x8x8 leaf tiles addressed through a flat leaf table, without hierarchical tree storage.
