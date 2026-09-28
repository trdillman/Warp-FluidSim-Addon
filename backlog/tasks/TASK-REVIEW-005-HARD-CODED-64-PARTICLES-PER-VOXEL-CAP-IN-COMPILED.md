# TASK-REVIEW-005: Hard-coded 64 particles-per-voxel cap in compiled emission silently truncates user density settings

## State

completed

## Severity

high

## What to build

The compiled emission kernel k_emit_compiled_source_particles uses a compile-time constant loop `for sample in range(64)` (line 1900) that limits emission to at most 64 particles per voxel per pass. This cap is reinforced by normalized_liquid_target() in source_regions.py (line 465: `return max(0, min(64, int(round(float(plan.payload.target_particles_per_voxel)))))`), which clamps the target before passing it to the kernel. If a user sets particles_per_voxel > 64 in the Blender UI, the compiled source pipeline will silently emit only 64 particles per voxel. No warning or error is raised. The legacy shape/mesh emission path (k_source_shapes_particles line 763: `want = int(s.emit_rate)`) does NOT have this cap, creating an inconsistency between compiled and legacy source paths for the same authored density setting.

**Location:** `fluidSim_warp_blender51/sim/sourcing.py:1900`
**Dimension:** Particle Pipeline

## Impact

The simulation cannot reach fluid densities above 64 particles per voxel through the compiled (primary) source path, regardless of available GPU memory or particle capacity. Users setting higher density values get silently fewer particles than authored, degrading simulation fidelity without any diagnostic signal. This directly conflicts with the project mandate that behavior must be robust as particle counts and resolution scale up.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/sim/sourcing.py:1900` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Make the compile-time loop bound a module-level constant (e.g., MAX_PPV = 128 or 256) and have normalized_liquid_target clamp to that constant rather than a literal 64. Alternatively, restructure the emission kernel to use a dynamic loop bound (wp.while_loop) so the cap can be configured at runtime. At minimum, emit a diagnostic warning when the authored particles_per_voxel exceeds the effective cap.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `PIPE-001` in `reviews/swarm-2026-08-08/findings/pipeline.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (High Findings section).
- Identified by the Particle Pipeline lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed for 2.0 (2026-09-27) as a single shared constant. The cap was not raised.

Finding re-checked: a user could not author more than 64. Both authored density settings (`emit_rate` and `particles_per_voxel` in `properties.py`) already had an RNA hard `max=64`, so nothing the UI accepts was silently truncated. The real defect was six independent copies of the literal 64: the kernel loop, three host clamps (`source_regions.normalized_liquid_target`, `source_execution._legacy_liquid_target`, `simulation_admission._legacy_footprint`) and the two RNA maxima. Any one of them could drift from the others. All six now read `constants.MAX_PARTICLES_PER_VOXEL = 64`.

The cap was kept at 64 on purpose: raising it is a capacity and memory decision (64 ppv is already 8x the default), so it is left for the owner/2.1. Because the RNA max is the same constant as the kernel bound, no warning is needed: a value above the kernel limit cannot be authored.

Test: `tests/test_particles_per_voxel_cap.py`. It covers:
- the host clamps at and above the ceiling;
- that both RNA maxima reference the constant (AST);
- that no literal `min(64` / `range(64)` / `max=64` remains in the add-on.

## Blocked by

None - can start immediately
