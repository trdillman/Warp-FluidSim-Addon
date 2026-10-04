# Porting Brief

This brief is a compact clean-room map for the Blender 5.1 Warp implementation (the original synthesized porting contracts live in git history). The central requirements are:

1. Implement the sparse tiled leaf-grid field and particle storage model on the GPU.
2. Preserve PIC/FLIP sampling, pressure projection, viscosity, surface SDF, collision, source, and whitewater behavior by feature group.
3. Use Warp flat kernels for simple per-element work and Warp tiled/cooperative strategies for OpenCL kernels that originally used local memory, barriers, reductions, and workgroup assumptions.
4. Keep simulation buffers GPU-resident during viewport playback.
5. Expose Blender-native sources, colliders, materials, render output, diagnostics, profiling, and optional cache/export.
6. Port UI semantics from the enum and UI contracts in `fluidSim_warp_blender51/`, not from Houdini implementation mechanics.
7. Use `fluidSim_warp_blender51/sim/validation.py` to test indexing, particle/grid transfer, pressure, viscosity, surface, and whitewater behavior.

The original code included an advanced custom OpenCL snippet surface. In Blender, recreate this as a controlled custom Warp extension surface only after the core solver is stable.

## GPU sync discipline (performance invariants)

Host/device synchronization is the dominant per-substep cost outside the kernels
themselves. The solver maintains these invariants; keep them when patching:

1. PCG (`sim/pressure.py`): iterations are fully device-resident. `alpha`/`beta`
   are computed by `k_compute_alpha`/`k_compute_beta` (which clamp to zero on
   tiny denominators, freezing `x`/`r` on breakdown). Convergence, divergence,
   singularity, and stall detection happen only at checkpoints (every
   `check_every` iterations) via `k_pack_pcg_stats`: one 3-float readback per
   checkpoint, not per iteration.
2. Whitewater compaction (`sim/whitewater.py`): `compact()` scans the full
   capacity with no host readback. Invariant: `alive` is zero beyond the live
   range (`clear()` and the trailing `k_ww_mark_alive` maintain this).
3. Particle emission (`sim/solver.py::_source`): the emission shape array is
   cached (`_emit_shape_array`) and invalidated by `set_source_batch` /
   `set_height_source` / domain changes; it is not rebuilt per substep. The
   post-emission `_index_particles` re-run happens only when an emitter fired.
4. `host_count` mirrors (`particles`, `whitewater`) are refreshed once per
   frame in `step_frame` stats (after `wp.synchronize_device`), by
   `inject_particles`, and by `clear()`. Do not call `refresh_host_count()`
   inside the substep loop; gate optional launches on settings, not on the
   mirror.
5. Topology rebuilds (`SparseGrid.finalize_topology(sync=False)`) do not read
   back leaf counts inside the substep loop. `FluidSolver.step_frame()` calls
   `refresh_topology_counts()` after the frame's existing `wp.synchronize_device`
   and still raises the leaf-capacity overflow error before stats are reported.
