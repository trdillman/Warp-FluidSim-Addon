# TASK-REVIEW-026: No conftest.py anywhere -- cuda_device fixture is defined locally in a single file, no shared fixtures or test isolation configuration

## State

completed

## Severity

medium

## What to build

No conftest.py exists anywhere in the tests/ directory tree (confirmed: find returned 'no conftest found'). The cuda_device fixture is defined only in tests/test_pressure_projection_regressions.py line 112 as a module-scoped fixture. Every other CUDA test file hardcodes 'cuda:0' and calls wp.init() independently with no shared fixture. There is no shared setUp/teardown for Warp device initialization, no test isolation markers, and no pytest configuration file (no pytest.ini, no pyproject.toml [tool.pytest.ini_options]).

**Location:** `tests/:1`
**Dimension:** Test Coverage

## Impact

Device initialization is duplicated and inconsistent across files (some use wp.init() at module scope, some inside functions, some via pytest.importorskip). Test isolation is unmanaged. Adding a second CUDA device or testing multi-GPU behavior requires touching every file.

## Acceptance criteria

- [x] Root cause at `tests/:1` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Create a tests/conftest.py with a session-scoped cuda_device fixture, standardize Warp initialization, and add pytest markers for CUDA-required vs CPU-only tests.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Resolution

Fixed in `16e5090` on `claude/2.0-phase1-testinfra`.

- `tests/conftest.py` defines a session-scoped `cuda_device` fixture
  (`select_test_device`): `FLUIDSIM_TEST_DEVICE` overrides the device and fails
  loudly if unusable; otherwise `cuda:0`, skipping cleanly without CUDA. The
  same fixture is duplicated in `fluidSim_warp_blender51/tests/conftest.py`
  because the add-on suite also runs standalone.
- `cuda`, `blender` and `slow` markers are registered in `pytest.ini` and the
  add-on `pyproject.toml` with `--strict-markers`; tests requesting
  `cuda_device` are auto-marked `cuda` by `pytest_collection_modifyitems`.
- Local fixtures and inline `is_cuda_available` skips migrated in 12 test
  files. Left as-is: `test_source_emission_lifecycle.py` (preferred-device
  helper with ~20 callers) and `test_surface_attribute_transfer.py`
  (deliberate cpu + optional cuda parametrization).

Regression gate: `tests/test_cuda_device_fixture.py` (default, no-CUDA skip,
override, invalid override, marker hook, parity of both conftest copies, and
no local `cuda_device` fixture that bypasses the shared one).

Verification: `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider`
1477 passed, 136 skipped (baseline 1456 passed, 121 skipped); ruff findings
unchanged from baseline (407).

## Evidence

- Source finding: `TEST-008` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
