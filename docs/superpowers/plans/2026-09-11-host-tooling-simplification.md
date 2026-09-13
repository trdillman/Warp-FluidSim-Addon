# Host and Tooling Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the cognitive load of three deterministic host-side routines without altering validation results, ownership-map output, or release-package verdicts.

**Architecture:** Keep every public orchestrator and its current ordering. Extract only cohesive private helpers from the existing routines: AST-node validation, ownership-map expression classification, and release-archive checks. Each task is independently committable and has no runtime dependency on the others.

**Tech Stack:** Python 3, `ast`, `pathlib`, `zipfile`, pytest, Blender 5.1 headless acceptance.

**Spec:** User-authorized simplification direction in this Codex task; repository behavior and ownership constraints in `AGENTS.md`.

## Global Constraints

- Make no feature, schema, cache, package-format, or user-facing behavior change.
- Preserve issue order, issue codes, line/column locations, exception classes, error text, and generated-document bytes unless an existing test intentionally changes them.
- Keep the add-on GPU-resident; do not touch Warp-decorated kernels or solver dispatch ordering in this plan.
- Do not alter the archived experiment worktrees named in `AGENTS.md`.
- Before execution, create a dedicated implementation worktree using `superpowers:using-git-worktrees`; do not implement on an unrelated dirty checkout.
- Run `git diff --check` before each commit and stage only the paths listed for that task.

---

### Task 1: Split effect-kernel AST validation into ordered private checks

**Files:**
- Modify: `fluidSim_warp_blender51/sim/effects/validation.py:94-246`
- Modify: `fluidSim_warp_blender51/tests/test_effect_validation.py:1-109`

**Interfaces:**
- Consumes: `KernelDefinitionV2`, `EffectInstanceV2`, `EffectIssue`, `_FORBIDDEN_NODES`, `_ALLOWED_GLOBALS`, and `validate_field_binding`.
- Produces: `ValidationResult` exactly as today; private helpers accept an `ast.AST` node and append `EffectIssue` values to the caller-owned list.
- Preserves: one `ast.walk(kernel_tree)` traversal in `_validate_kernel`, the existing traversal order, and `validate_effect()` as the sole production caller.

- [ ] **Step 1: Add a characterization test for ordered mixed AST issues**

Add this test beneath `test_validation_issues_identify_the_exact_effect_instance` in `fluidSim_warp_blender51/tests/test_effect_validation.py`:

```python
def test_validation_keeps_mixed_ast_issue_order_and_locations():
    instance = _instance(
        "while strength > 0.0:\n"
        "    strength = strength - 1.0\n"
        "print(strength)"
    )

    result = validate_effect(instance)

    assert [(issue.code, issue.line, issue.column) for issue in result.issues] == [
        ("UNBOUNDED_LOOP", 1, 0),
        ("PRINT_FORBIDDEN", 3, 0),
    ]
```

- [ ] **Step 2: Run the characterization test before extraction**

Run:

```powershell
python -m pytest -q fluidSim_warp_blender51/tests/test_effect_validation.py::test_validation_keeps_mixed_ast_issue_order_and_locations
```

Expected: PASS on the unmodified implementation. If it fails, record the exact current tuple and update only the expected tuple before continuing; the characterization must describe current behavior, not a desired replacement.

- [ ] **Step 3: Extract local node-check helpers without creating a validator framework**

In `validation.py`, introduce the private helpers `_append_forbidden_node_issue`, `_append_loop_issue`, `_append_attribute_issues`, `_append_name_issues`, and `_append_field_access_issues`. Move the corresponding current branches from the single `for node in ast.walk(tree)` loop into those helpers verbatim. Invoke them from that same loop in current source order: forbidden-node check, bounded-loop check, print check, attribute check, loaded-name check, and field-subscript check. Pass the existing `kernel`, `safe_mode`, `allowed`, `assigned`, `known`, and `binding_map` inputs only to the helper that currently reads each value. Keep `_validate_kernel` responsible for parsing and final list construction. Do not replace the helpers with a visitor class or make a second AST traversal.

- [ ] **Step 4: Run the focused effect-validation suite**

Run:

```powershell
python -m pytest -q fluidSim_warp_blender51/tests/test_effect_validation.py fluidSim_warp_blender51/tests/test_effect_builtins.py fluidSim_warp_blender51/tests/test_effect_fields.py
```

Expected: all tests pass, including the ordered characterization test.

- [ ] **Step 5: Run the runtime effect acceptance gate**

Run:

```powershell
blender --background --factory-startup --python tests/blender_custom_effects_acceptance.py
```

Expected: `CUSTOM-EFFECTS-ACCEPTANCE-PASS` (or the repository's current matching pass marker) and exit code 0.

- [ ] **Step 6: Measure, inspect, and commit the isolated change**

Confirm `_validate_kernel` now orchestrates the original checks in fewer lines, then run:

```powershell
git diff --check
git diff -- fluidSim_warp_blender51/sim/effects/validation.py fluidSim_warp_blender51/tests/test_effect_validation.py
git add -- fluidSim_warp_blender51/sim/effects/validation.py fluidSim_warp_blender51/tests/test_effect_validation.py
git commit -m "refactor: split effect AST validation checks"
```

Expected: one commit containing only the implementation and its characterization test.

### Task 2: Name the ownership-map expression-classification phases

**Files:**
- Modify: `tools/ownership_map.py:375-499`
- Modify: `tests/test_ownership_map.py:107-118`

**Interfaces:**
- Consumes: the existing function-record object passed to `analyze_function(func, fields, module=None)`, `loop_name_bindings`, `resolve_expr`, and the analyzer's field/call collections.
- Produces: unchanged `func.writes`, `func.reads`, `func.calls`, and `func.call_names`; the checked-in `docs/ownership-map.md` remains byte-identical.
- Preserves: the intentional two-pass analysis, the mutually exclusive property/draw/readout/`getattr`/`setattr` precedence, and unconditional outbound-call recording.

- [ ] **Step 1: Add a byte-level output stability assertion to the existing subprocess gate**

Extend `test_map_is_current` in `tests/test_ownership_map.py` so it captures the pre-run document bytes and asserts them after the `--check` subprocess exits successfully:

```python
def test_map_is_current():
    before = MAP_PATH.read_bytes()
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "docs/ownership-map.md is stale; regenerate with "
        f"`python tools/ownership_map.py`.\n{result.stdout}{result.stderr}"
    )
    assert MAP_PATH.read_bytes() == before
```

- [ ] **Step 2: Run the map gate before extraction**

Run:

```powershell
python -m pytest -q tests/test_ownership_map.py::test_map_is_current
python tools/ownership_map.py --check
```

Expected: both commands pass and `docs/ownership-map.md` is not modified.

- [ ] **Step 3: Extract only the two cohesive phases from `analyze_function`**

Create the private helpers `_classify_accesses` and `_record_outbound_call` in `tools/ownership_map.py`. `_classify_accesses` receives the current `ast.Call`, the existing `env`, `fields`, `loop_fields`, and `table_names` closure, then performs the current mutually exclusive `prop`, draw-helper, readout-helper, `getattr`/`hasattr`, and `setattr` branches against the caller-owned `func`. `_record_outbound_call` receives the same call plus `env`, `fields`, and `func`, then performs the current unconditional resolved-kind/name binding append. Keep `analyze_function` responsible for environment initialization, both existing walks, the nested `table_names` closure, and fixpoint-visible mutation of `func`.

- [ ] **Step 4: Run all ownership-map assertions and the stale-file gate**

Run:

```powershell
python -m pytest -q tests/test_ownership_map.py
python tools/ownership_map.py --check
```

Expected: all tests pass; `--check` reports the map is current without rewriting it.

- [ ] **Step 5: Inspect the generated-document boundary and commit**

Run:

```powershell
git diff --check
git diff -- docs/ownership-map.md tools/ownership_map.py tests/test_ownership_map.py
git diff --exit-code -- docs/ownership-map.md
git add -- tools/ownership_map.py tests/test_ownership_map.py
git commit -m "refactor: name ownership map analysis phases"
```

Expected: the generated documentation has no diff; the commit contains only the analyzer and test.

### Task 3: Decompose release-package inspection while preserving release identity

**Files:**
- Modify: `tools/release/build_extension.py:425-574`
- Modify: `fluidSim_warp_blender51/tests/test_build_extension.py:249-323`

**Interfaces:**
- Consumes: `Path`, package ZIP paths, `SourceVerification`, `_collision_guard`, and the existing `ArtifactRecord` schema.
- Produces: `list[ArtifactRecord]` with existing sort/order semantics; `inspect_packages(root, packages)` remains the public inspection API.
- Preserves: split-platform package-name validation, forbidden Blender/Warp runtime detection, required add-on integration source, release-only-content rejection, payload fingerprints, and original `RuntimeError` messages.

- [ ] **Step 1: Add a success-path record characterization**

Add this test beside `test_inspect_packages_accepts_binary_free_platform_archive` in `fluidSim_warp_blender51/tests/test_build_extension.py`:

```python
def test_inspect_packages_keeps_record_shape_for_valid_archive(tmp_path: Path):
    _write_project(tmp_path)
    package = tmp_path / f"fluidsim_warp-{VERSION}-linux_x64.zip"
    _write_package(package)

    records = build_extension.inspect_packages(tmp_path, [package])

    assert len(records) == 1
    record = records[0]
    assert record.platform == "linux_x64"
    assert record.filename == package.name
    assert record.sha256
    assert record.size_bytes == package.stat().st_size
```

- [ ] **Step 2: Run the new characterization and existing rejection tests**

Run:

```powershell
python -m pytest -q fluidSim_warp_blender51/tests/test_build_extension.py -k "inspect_packages"
```

Expected: the new valid-record test and all current rejection tests pass before code movement.

- [ ] **Step 3: Extract archive-member, required-content, and manifest checks**

Add private helpers named `_validate_archive_members`, `_validate_required_package_content`, and `_validate_embedded_manifest_and_payload`. Move the current archive-member loop that rejects runtime, bytecode, release-only, and backup payloads into the first helper. Move the existing `required`, collision-guard, built-in-effect inventory, and Geometry Nodes asset checks into the second. Move embedded-manifest validation and cross-platform normalized payload comparison into the third. Keep `inspect_packages` responsible for normalization, `verify_source_tree`, package-set validation, opening each archive, computing the existing `ArtifactRecord(path, platform, "external", size, sha256)`, and final platform sort. Copy error messages and exception types exactly.

- [ ] **Step 4: Run package inspection and full extension tests**

Run:

```powershell
python -m pytest -q fluidSim_warp_blender51/tests/test_build_extension.py
python tools/release/build_extension.py --verify-only
```

Expected: tests pass. The verification command succeeds only against the repository's valid isolated release inputs; if required package artifacts are absent, preserve that environmental blocker and do not weaken release validation.

- [ ] **Step 5: Measure, inspect, and commit**

Run:

```powershell
git diff --check
git diff -- tools/release/build_extension.py fluidSim_warp_blender51/tests/test_build_extension.py
git add -- tools/release/build_extension.py fluidSim_warp_blender51/tests/test_build_extension.py
git commit -m "refactor: separate release package inspection checks"
```

Expected: one focused commit; no package content or release metadata changes are staged.

## Plan Self-Review

- Spec coverage: Tasks 1-3 cover the three deterministic host/tooling candidates, retain public interfaces, and include focused regression evidence plus their required runtime or command-line gate.
- Scope: the runtime profiler and frame-output code are intentionally absent; they belong to `2026-09-11-profile-frame-output-simplification.md` because their proof requires Blender/CUDA output comparison.
- Type consistency: all introduced helpers are private and receive only existing AST, path, verification, issue, and collection types.
- Completeness: every task names exact paths, a behavior characterization, implementation seam, verification commands, diff inspection, and a single-purpose commit.
