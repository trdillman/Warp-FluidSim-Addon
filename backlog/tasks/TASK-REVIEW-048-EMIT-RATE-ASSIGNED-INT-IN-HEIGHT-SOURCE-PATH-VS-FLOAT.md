# TASK-REVIEW-048: emit_rate assigned int in height source path vs float in all other assignment paths

## State

ready-for-agent

## Severity

low

## What to build

In gather_height_source() (scene_inputs.py line 918), desc.emit_rate is assigned fs.particles_per_voxel directly, which is an IntProperty yielding a Python int. In gather_sources() (line 384), desc.emit_rate = float(_effective_source_target(scene, pg)) explicitly converts to float. In the FORCE override path (line 467), desc.emit_rate = 0.0 is also float. The height source is the only path that assigns a raw int, creating a type inconsistency on the same descriptor field.

**Location:** `fluidSim_warp_blender51/blender/scene_inputs.py:918`
**Dimension:** Type System

## Impact

Python int and float are interchangeable in most contexts, so no immediate runtime error. However, if HeightSourceDesc.emit_rate has a typed annotation (e.g., float) or if downstream warp/CUDA code expects a float32 and performs strict type checking, the int could cause a subtle mismatch.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/blender/scene_inputs.py:918` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Change line 918 to desc.emit_rate = float(fs.particles_per_voxel) for consistency with all other assignment paths.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TYPE-006` in `reviews/swarm-2026-08-08/findings/types.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Type System lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
