# TASK-REVIEW-018: normalized_liquid_target and _legacy_liquid_target clamp to 64 but SourceShapeDesc.to_struct bakes raw emit_rate into the GPU struct

## State

ready-for-agent

## Severity

medium

## What to build

normalized_liquid_target() (line 465) returns `max(0, min(64, ...))` and _legacy_liquid_target() (sourcing.py:2329) returns `max(0, min(64, ...))`. Both are used to compute demand and decide whether emission should proceed. However, SourceShapeDesc.to_struct() (sourcing.py:227) copies the raw, unclamped emit_rate into the SourceShape struct: `shape.emit_rate = float(self.emit_rate)`. The legacy emission kernel k_source_shapes_particles (sourcing.py:763) reads `want = int(s.emit_rate)` directly from the struct, bypassing the 64 clamp entirely. MeshSourceDesc.to_struct() (sourcing.py:312) has the same pattern. This means legacy shape/mesh sources can emit more than 64 particles per voxel when emit_rate > 64, while compiled sources are capped at 64. The demand counting kernels (k_count_source_shapes_particles:2000, k_count_mesh_source_particles:2039) also read raw emit_rate, so demand and emission are internally consistent within the legacy path, but the two paths have different effective density limits for the same authored setting.

**Location:** `fluidSim_warp_blender51/sim/source_regions.py:465`
**Dimension:** Particle Pipeline

## Impact

If the same particles_per_voxel setting drives both compiled and legacy sources, the legacy sources will achieve higher density than compiled sources, producing visually inconsistent fluid density in mixed-source scenes. The 64 cap in _legacy_liquid_target is documented as preserving historical truncation behavior, but the actual struct baking does not truncate, contradicting the stated intent.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/sim/source_regions.py:465` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Either clamp emit_rate in to_struct() to match normalized_liquid_target (both paths capped at 64), or document that legacy sources intentionally have no cap and remove the misleading min(64) from _legacy_liquid_target, making it a simple max(0, int(...)) zero-check.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `PIPE-005` in `reviews/swarm-2026-08-08/findings/pipeline.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Particle Pipeline lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
