# TASK-REVIEW-033: Absolute intra-package imports in sim/custom.py instead of relative imports

## State

completed

## Severity

low

## What to build

sim/custom.py uses absolute imports at lines 824-825 and 863: 'from fluidSim_warp_blender51.sim import constants as C' and 'from fluidSim_warp_blender51.sim.fields import voxel_coord_of_flat, voxel_center_world'. These are the only absolute intra-package imports in the entire addon; every other module uses relative imports (e.g., 'from ..sim.memory_safety import ...' in scene_bridge.py, 'from . import scene_inputs' in scene_bridge.py). Blender extensions can be installed under modified package names, and the absolute form breaks if the top-level package directory is renamed or the module is loaded outside the expected package context.

**Location:** `fluidSim_warp_blender51/sim/custom.py:824`
**Dimension:** Architecture

## Impact

If the package is renamed (e.g., during Blender extension installation or test harness loading), these imports fail with ModuleNotFoundError while all other modules continue working. Inconsistent with the rest of the codebase.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/sim/custom.py:824` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Replace with relative imports: 'from . import constants as C' and 'from .fields import voxel_coord_of_flat, voxel_center_world'.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `ARCH-003` in `reviews/swarm-2026-08-08/findings/arch.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Architecture lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed before 1.1.0. `fluidSim_warp_blender51/sim/custom.py` imports `from . import constants as C`. The generated kernel-source preamble (`_VOXEL_IMPORT_PREAMBLE`, the former absolute imports) now uses `_ADDON_ROOT = addon_root_package(__package__)`, so it works under `bl_ext.*` extension package names. No hard-coded `fluidSim_warp_blender51` import remains in add-on runtime code.

Tests: `fluidSim_warp_blender51/tests/test_namespace_safety.py::test_no_runtime_file_hardcodes_legacy_package_name`, `::test_custom_package_root_handles_extension_namespace`, `::test_custom_package_root_rejects_non_sim_package`.

Fixing commit: `bb6d3fc`.

## Blocked by

None - can start immediately
