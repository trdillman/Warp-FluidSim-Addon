# Source Synthesis

This package was synthesized from the uploaded FluidSim implementation references. It does not include the original files. The source material established that the solver is FluidSim 1.0.226, that `liquid.cl` and `fluidSim.cl` identify the same solver, and that the add-on target is Blender 5.1.

The OpenCL solver inventory contains 188 kernels: 80 core kernels and 108 FluidSim/paradigm liquid kernels. The original uploaded kernel namespace uses `paradigm` for liquid kernels; this package exposes `fluidSim` as the canonical target namespace while preserving source-name aliases. The OpenCL source uses local memory, barriers, atomics, and optional half/float field precision, so the Warp port needs both normal flat kernels and tiled/cooperative kernels.

The Houdini references were distilled into Blender-portable semantics only: enum values, UI labels, custom-kernel/snippet data model, profiling fields, benchmark fields, dispatch concepts, and behavior notes. Houdini APIs, HScript wrappers, license UI, PySide widgets, and original source bodies are excluded.

Known audit items:

- Verify the staggered vector magnitude conversion logic while porting.
- Decide whether to correct the original custom-snippet loop behavior so multiple snippets are supported.
- Replace Houdini profile panels with Blender UI/overlay and JSON diagnostics.
