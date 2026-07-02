# Porting Brief

Use the synthesized modules as a compact clean-room map for the Blender 5.1 Warp implementation. The central requirements are:

1. Implement the sparse-octree field and particle storage model on the GPU.
2. Preserve PIC/FLIP sampling, pressure projection, viscosity, surface SDF, collision, source, and whitewater behavior by feature group.
3. Use Warp flat kernels for simple per-element work and Warp tiled/cooperative strategies for OpenCL kernels that originally used local memory, barriers, reductions, and workgroup assumptions.
4. Keep simulation buffers GPU-resident during viewport playback.
5. Expose Blender-native sources, colliders, materials, render output, diagnostics, profiling, and optional cache/export.
6. Port UI semantics from the synthesized enum and UI contracts, not from Houdini implementation mechanics.
7. Use the validation plan to test indexing, particle/grid transfer, pressure, viscosity, surface, and whitewater behavior.

The original code included an advanced custom OpenCL snippet surface. In Blender, recreate this as a controlled custom Warp extension surface only after the core solver is stable.
