# Repository Structure Cleanup Design

Date: 2026-08-13  
Status: Approved design; implementation pending review of this written specification

## Goal

Make the repository root understandable without deleting maintained evidence, changing the FluidSim runtime, touching archived experiment worktrees, or rewriting Git history.

The approved scope is cleanup Sets A+B from the repository audit:

- low-risk documentation and ignore-rule improvements;
- non-destructive relocation of tracked historical/reference material;
- selective organization of review records while preserving active ticket links.

Generated local data, unreachable Git objects, and repository history are outside this change.

## Options considered

### 1. Ignore confusing folders

Rejected. `fluidSim_synth/`, the Houdini extraction, review records, and both backlog roots are already tracked. Ignore rules would not remove them from Git and would hide their ownership problem rather than solve it.

### 2. Delete historical/reference material

Rejected. Active tests consume `fluidSim_synth/`; `reviews/CORRECTION_PLAN.md` consumes the Houdini extraction; and 48 active review tickets cite the swarm evidence. Deletion would break validation or provenance.

### 3. Clarify maintained content and relocate historical material

Approved. Keep authoritative material, add orientation documentation, move reference evidence beneath `docs/`, consolidate the backlog namespace, and archive completed phase/worktree review records without disturbing active swarm evidence.

## Resulting repository structure

```text
README.md
backlog/
  reports/
    daily-bug-scan-1907.md
  tasks/
docs/
  references/
    houdini_dispatch/
      README.md
      paradigm_dispatch_enqueue_design.md
      paradigm_enqueue_dispatch.py
      paradigm_kernel_names.txt
      paradigm_mapping_verify.txt
fluidSim_synth/
  README.md
reviews/
  ADDON_REVIEW.md
  CORRECTION_PLAN.md
  archive/
    phases/
      PHASE_*.md
    worktrees/
      WORKTREE_*.md
  swarm-2026-08-08/
```

## Detailed changes

### Root orientation

Add `README.md` describing:

- the production add-on location;
- the distinction between runtime code, synthesized contracts, validation, curated demos, evidence, and generated local outputs;
- the authoritative commands from `AGENTS.md` without duplicating its full workflow catalogue;
- the four archived experiment worktrees as non-merge candidates.

The README is an orientation surface. `AGENTS.md`, `PROMPT.md`, `CONTEXT.md`, and `CONTRIBUTING.md` remain the detailed authorities.

### Synthesized contracts

Retain `fluidSim_synth/` at its current path because tests and plans already consume it. Add `fluidSim_synth/README.md` stating that it is:

- the distilled FluidSim 1.0.226 fidelity contract;
- validation/reference input, not Blender add-on runtime code;
- consumed directly by kernel-coverage and wording tests;
- subordinate to live runtime evidence when historical contracts and implementation disagree.

Renaming the package is deliberately deferred because it would create needless import and documentation churn.

### Houdini dispatch evidence

Move the four tracked files from `dispatch_extraction_from_houdini_archive/` to `docs/references/houdini_dispatch/` using Git-aware moves. Add a README recording:

- source archive and version provenance;
- commit `a6965e6` as the repository handoff;
- the evidence types represented by each file;
- the known limitation that generic dispatch pseudocode does not prove concrete solver phase order;
- that the folder is not runtime source.

Update `reviews/CORRECTION_PLAN.md` and any remaining tracked references to the new path.

### Backlog namespace

Move `backlogs/daily-bug-scan-1907.md` to `backlog/reports/daily-bug-scan-1907.md`. The canonical live tracker remains `backlog/tasks/`; historical scan reports belong under the same singular root but outside the task queue.

### Review records

Keep these at the `reviews/` root because they are current evidence surfaces:

- `ADDON_REVIEW.md`;
- `CORRECTION_PLAN.md`;
- `swarm-2026-08-08/`, including findings and lane specifications, because 48 active `TASK-REVIEW-*` cards link to it.

Move completed `PHASE_*.md` records to `reviews/archive/phases/` and the two `WORKTREE_*.md` records to `reviews/archive/worktrees/`. Update all tracked cross-references, including `TASK-WORKTREE-PRUNE-001`.

This resolves the policy mismatch without treating still-referenced swarm evidence as disposable generated output. A future change may distill or remove swarm internals only after active tickets no longer depend on them.

### Claude agent definitions

Retain the three tracked `.claude/agents/liquid-warp-blender51-port/*.md` files unchanged. They are tiny, intentionally committed role definitions and do not create Git dirt. Nested `.claude/worktrees/` remains ignored and excluded from this cleanup.

### Ignore policy

- Add `/bench/` so future benchmark output cannot become untracked accidentally.
- Simplify `.codex` ignoring to `/.codex/`; remove the absent `liquid-review-fixes-loop.md` exception.
- Do not add ignore rules for tracked reference, backlog, or review content.
- Do not delete current ignored artifacts, caches, logs, session files, or worktrees.

## Safety and compatibility

- No production Python behavior changes.
- No Blender files, caches, artifacts, branches, experiment worktrees, Git objects, or history are deleted.
- Moves preserve file contents and Git history.
- All tracked path references are updated atomically in the same change.
- The branch remains isolated as `codex/repo-structure-cleanup` until verified.

## Verification

The implementation is complete only when:

1. every intended old tracked path is absent and every destination is tracked;
2. repository-wide searches find no stale old paths;
3. `tests/test_kernel_coverage.py` and `tests/test_contract_wording.py` pass;
4. `tools/validate_policy_drift_reports.py` is run, with any expected HEAD-freshness failure separated from structural failures;
5. documentation links and backlog references resolve;
6. `git diff --check` passes;
7. the diff contains no product-runtime or experiment-worktree changes;
8. an independent reviewer reports no broken authority or provenance link;
9. the branch is committed cleanly, while ignored local artifacts remain untouched.

## Explicit exclusions

- Deleting ignored local data.
- Removing or changing any of the four archived experiment worktrees.
- Pruning unreachable Git objects or temporary object files.
- Git LFS adoption or history rewriting.
- Resolving the 49 active review tickets.
- Renaming `fluidSim_synth/`.
