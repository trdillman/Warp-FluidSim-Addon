# TASK-REVIEW-023: Production gate tests mock out _run (the core subprocess executor) and all guard functions -- only verifies command assembly, never actual execution

## State

ready-for-agent

## Severity

medium

## What to build

Both test_production_gate_semantics.py (_stub_gate at line 18) and test_production_gate_truthfulness.py (fake_run at line 21) monkeypatch _run -- the core subprocess executor of run_production_gate.py -- with a no-op lambda that just records labels. They also stub _check_no_tracked_npz, _check_no_tracked_ignored_files, _clear_gate_manifest, _write_gate_manifest, and _print_gate_metadata. The tests verify that main() assembles the right command labels and output strings, but never verify that any subprocess actually executes, that markers are enforced, or that reject_output scanning works. The production gate is a safety gate -- mocking its execution layer means the tests prove 'the code calls things in the right order' but not 'the things work.'

**Location:** `tests/test_production_gate_semantics.py:18`
**Dimension:** Test Coverage

## Impact

A regression in _run (subprocess invocation, marker checking, output rejection) would pass all production gate tests while failing in real CI. The gate could silently pass a broken build.

## Acceptance criteria

- [ ] Root cause at `tests/test_production_gate_semantics.py:18` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: This is partially intentional (you cannot run Blender in unit tests). The gap is that there is no integration test layer that exercises _run with a real subprocess. Add at least one integration test that runs a trivial 'echo MARKER' command through _run and verifies marker enforcement and output rejection end-to-end.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TEST-005` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
