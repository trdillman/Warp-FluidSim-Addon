# Artist workflow design

Status: implementation design derived from the user's workflow complaints and the subsequent recommendation. No runtime implementation is included in this document.

## Outcome

An artist can configure liquid, add Radial Pulse, adjust it, and press Preview to see a real simulation. Compilation, solver initialization, and output preparation are internal steps. The user sees separate actions only when their consequences differ.

## Requirements

1. Start Preview prepares eligible effects and outputs, admits the simulation, and starts actual Blender timeline playback. It does not merely arm a mode.
2. Pause stops playback without releasing the solver or discarding progress. Resume continues the same owned simulation when settings are compatible. Restart Simulation explicitly starts again at the scene's start frame.
3. Runtime-compatible effect parameters update directly without replacing the solver or recompiling kernels. Structural effect edits are prepared at Start/Resume/Apply; they must not silently execute the previous stack as if the edit had applied.
4. One Apply Changes action delegates reset/rebuild selection to pending_settings. Confirmation describes actual loss of preview progress. Fresh initialization and safe live edits require no confirmation. An explicit restart action needs no second redundant confirmation when it already makes the loss clear.
5. Effect preparation preserves content trust, migration policy, validation, sandbox checks, and cache identity. Publish applied authoring evidence only for a successfully installed candidate. A rejected candidate preserves the previous authoritative solver and applied effect evidence before cutover.
6. Set Up & Preview produces a real owned domain, liquid source, outputs, and running simulation. Set Up Only remains available. A setup failure restores its own partial edits; a subsequent playback failure leaves a valid configured scene paused with a useful error.
7. Live simulation, cached playback, and production bake retain distinct semantics. Invalid cached playback never falls back to live simulation. Bake restart cannot destroy valid existing cache before replacement admission succeeds. Modal bake work remains bound to its initiating scene.
8. Normal controls use user intent. Keep settings beside their subject: effects, quality, materials, sources. Set Up, Preview, Bake, and Adjust guide grouping but do not mandate four tabs or a miscellaneous Adjust panel.
9. Preserve existing operator IDs and low-level Python workflows. Move manual initialization/release, diagnostic stepping, effect authoring tools, forced output repair, profiling, and headless commands into existing Advanced/Diagnostics disclosure. Keep essential error recovery discoverable.
10. Remove Voxel Size and Liquid Sampling from Live Tuning; retain them with quality controls and lifecycle consequences. Use consistent Basic/Advanced presentation in Scene, Physics, and View3D surfaces.
11. Reported transactional hazards are hypotheses until tested. Trace the shared owner and a sibling caller for each confirmed defect. Bake-specific work does not block the independently verified Preview milestone.
12. Completion requires source-bound Blender GUI proof: actual timeline advancement, nonzero liquid, changing owned output, production effect execution, Pause/Resume preservation, restart behavior, and saved-demo reopen behavior. Tests must distinguish old installed code from this worktree.

## Global constraints

- Target Blender 5.1 on Windows with the existing Warp/CUDA runtime; add no dependency.
- Keep simulation GPU-resident and retain the fixed 8x8x8 sparse leaf-grid contract.
- Preserve authoritative object-owned domain settings and the existing scene ownership rules.
- Keep simulation_admission, scene_bridge, playback, pending_settings, output_publication, and cache_bake authoritative in their existing responsibilities.
- Do not change catch-up, desynchronization, frame stepping, cache formats, numerical algorithms, or bake reproducibility claims.
- Preserve unrelated work and archived experiments; do not merge or extract experiments.
- Use temporary source-loaded Blender sessions for development; do not change the normal installed add-on/profile.
- No external publication, automatic cache deletion, or runtime rewrite is part of this work.

## Interaction decisions

| Situation | Action and result |
|---|---|
| Fresh valid scene, built-in draft effect | Preview prepares the stack and starts playback |
| Paused compatible simulation | Resume preserves solver, frame, and effect cadence |
| Safe parameter edit | Apply to running/paused owned solver directly |
| Structural edit | Pause advancement; show changes pending; Start/Resume/Apply prepares current draft |
| Edit requires losing progress | Explain restart from the scene's start frame; cancel preserves progress and draft |
| Rejected effect | Show the specific error; do not start playback or publish applied evidence |
| No usable GUI playback context | Explain that playback could not start; keep valid prepared state paused |
| Another scene owns the solver | Keep existing ownership protection and actionable message |
| Invalid cache selection | Remain paused; never initialize a live solver as fallback |
| Output warning | Distinguish a working simulation with a display warning from admission failure |
| Production bake active | Disable conflicting authoring/start/restart actions through existing eligibility policy |

Structural effects may be installed in place only if the existing runtime explicitly supports their structural compatibility and initialization cadence. Otherwise use a confirmed restart. Do not build a new hot-swap subsystem to avoid this consequence.

## Out of scope

Installer changes, new effect features, new solvers, a general transaction framework, performance tuning unrelated to the changed path, cache schema migration, and broad role/GPU/preset UI redesign. The earlier installer investigation remains separate from this workflow plan.
