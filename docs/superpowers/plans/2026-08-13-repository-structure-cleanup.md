# Repository Structure Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the tracked repository structure self-explanatory while preserving runtime behavior, active evidence, archived experiments, and local ignored data.

**Architecture:** Treat the cleanup as a documentation and path-ownership migration. Add orientation at the root and inside maintained reference packages, relocate historical evidence into explicit archive/reference namespaces, update every tracked cross-reference in the same change, and finish with focused contract tests plus repository-wide structural checks.

**Tech Stack:** Markdown, Git, PowerShell, Python/pytest.

## Global Constraints

- No production Python behavior changes.
- No Blender files, caches, artifacts, branches, experiment worktrees, Git objects, or history are deleted.
- Keep `fluidSim_synth/` at its current path.
- Keep `reviews/swarm-2026-08-08/` and `.claude/agents/liquid-warp-blender51-port/` unchanged.
- Do not resolve or rewrite active backlog tickets.
- Do not push the branch without a separate user request.

---

### Task 1: Add repository orientation and local-output ignore rules

**Files:**
- Create: `README.md`
- Create: `fluidSim_synth/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: repository authority from `AGENTS.md`, `PROMPT.md`, `CONTEXT.md`, and `CONTRIBUTING.md`.
- Produces: human-facing repository orientation and ignore rules for `/bench/` and all of `/.codex/`.

- [ ] **Step 1: Add the root orientation document**

Create `README.md` with a concise map of production runtime, synthesized contracts, tests/tools, demos, tracked evidence, and ignored generated output. Point readers to the existing authority documents and label all four experiment worktrees from `AGENTS.md` as archived and not merge candidates.

- [ ] **Step 2: Document the synthesized contract package**

Create `fluidSim_synth/README.md` stating that the package distills the FluidSim 1.0.226 fidelity contract, is validation/reference input rather than Blender runtime code, is consumed by `tests/test_kernel_coverage.py` and `tests/test_contract_wording.py`, and is subordinate to current runtime evidence.

- [ ] **Step 3: Tighten generated-output ignore rules**

Replace the four `.codex` exception rules with `/.codex/` and add `/bench/`. Leave tracked reference, backlog, review, and experiment-worktree rules unchanged.

- [ ] **Step 4: Check the documentation and ignore diff**

Run: `git diff --check -- .gitignore README.md fluidSim_synth/README.md`

Expected: exit code 0 with no output.

- [ ] **Step 5: Commit the orientation change**

Run:

```powershell
git add -- .gitignore README.md fluidSim_synth/README.md
git commit -m "docs: explain repository structure"
```

Expected: one commit containing only the three listed paths.

### Task 2: Relocate and document Houdini dispatch evidence

**Files:**
- Move: `dispatch_extraction_from_houdini_archive/paradigm_dispatch_enqueue_design.md` to `docs/references/houdini_dispatch/paradigm_dispatch_enqueue_design.md`
- Move: `dispatch_extraction_from_houdini_archive/paradigm_enqueue_dispatch.py` to `docs/references/houdini_dispatch/paradigm_enqueue_dispatch.py`
- Move: `dispatch_extraction_from_houdini_archive/paradigm_kernel_names.txt` to `docs/references/houdini_dispatch/paradigm_kernel_names.txt`
- Move: `dispatch_extraction_from_houdini_archive/paradigm_mapping_verify.txt` to `docs/references/houdini_dispatch/paradigm_mapping_verify.txt`
- Create: `docs/references/houdini_dispatch/README.md`
- Modify: `reviews/CORRECTION_PLAN.md`

**Interfaces:**
- Consumes: tracked Paradigm/FluidSim 1.0.226 evidence introduced by commit `a6965e6`.
- Produces: a clearly owned reference package and a corrected path from the active correction plan.

- [ ] **Step 1: Create a focused recovery archive**

Create `artifacts/repo_structure_cleanup_backup_20260813.zip` from the four source files and verify that the archive contains exactly four entries. The archive remains ignored local recovery data and is not staged.

- [ ] **Step 2: Move the four evidence files**

Create `docs/references/houdini_dispatch/`, then move the four files to their exact destinations. Verify that all destination paths resolve inside the repository before moving.

- [ ] **Step 3: Add provenance and limitation documentation**

Create `docs/references/houdini_dispatch/README.md` naming the Paradigm/FluidSim 1.0.226 source, commit `a6965e6`, the meaning of each file, the fact that the package is not runtime source, and the limitation that generic dispatch pseudocode does not prove concrete solver phase order.

- [ ] **Step 4: Update the active correction-plan reference**

Replace `dispatch_extraction_from_houdini_archive/` in `reviews/CORRECTION_PLAN.md` with `docs/references/houdini_dispatch/`.

- [ ] **Step 5: Prove the migration is complete**

Run:

```powershell
git ls-files --error-unmatch docs/references/houdini_dispatch/README.md docs/references/houdini_dispatch/paradigm_dispatch_enqueue_design.md docs/references/houdini_dispatch/paradigm_enqueue_dispatch.py docs/references/houdini_dispatch/paradigm_kernel_names.txt docs/references/houdini_dispatch/paradigm_mapping_verify.txt
rg -n --hidden --glob '!.git/**' --glob '!docs/superpowers/specs/2026-08-13-repository-structure-cleanup-design.md' --glob '!docs/superpowers/plans/2026-08-13-repository-structure-cleanup.md' --fixed-strings 'dispatch_extraction_from_houdini_archive' .
```

Expected: all five destination files are listed; the search returns no matches. Before the commit, use `git status --short` to verify the old files appear as deletions and the destination files appear as additions or renames.

- [ ] **Step 6: Commit the evidence relocation**

Run:

```powershell
git add -- dispatch_extraction_from_houdini_archive docs/references/houdini_dispatch reviews/CORRECTION_PLAN.md
git commit -m "docs: relocate Houdini dispatch evidence"
```

Expected: one commit containing only the evidence moves, the new README, and the corrected active reference.

### Task 3: Consolidate historical backlog and review records

**Files:**
- Move: `backlogs/daily-bug-scan-1907.md` to `backlog/reports/daily-bug-scan-1907.md`
- Move: `reviews/PHASE_*.md` to `reviews/archive/phases/`
- Move: `reviews/WORKTREE_*.md` to `reviews/archive/worktrees/`
- Modify: moved phase/worktree documents containing root-qualified review paths
- Modify: `backlog/tasks/TASK-WORKTREE-PRUNE-001-RETIRE-REDUNDANT-FLUIDSIM-CHECKOUTS.md`

**Interfaces:**
- Consumes: completed historical reports currently stored beside active review surfaces.
- Produces: one singular backlog namespace and archive namespaces whose internal and active-ticket links resolve.

- [ ] **Step 1: Move the historical scan report**

Create `backlog/reports/` and move `backlogs/daily-bug-scan-1907.md` to `backlog/reports/daily-bug-scan-1907.md`.

- [ ] **Step 2: Move completed phase records**

Create `reviews/archive/phases/` and move every tracked root-level `reviews/PHASE_*.md` file into it without changing filenames.

- [ ] **Step 3: Move completed worktree records**

Create `reviews/archive/worktrees/` and move both tracked root-level `reviews/WORKTREE_*.md` files into it without changing filenames.

- [ ] **Step 4: Rewrite root-qualified review links**

Within moved phase documents, replace each `reviews/PHASE_` link prefix with `reviews/archive/phases/PHASE_`. Within moved worktree documents and the active worktree-prune ticket, replace each `reviews/WORKTREE_` link prefix with `reviews/archive/worktrees/WORKTREE_`.

- [ ] **Step 5: Prove active surfaces and references remain valid**

Run:

```powershell
git ls-files --error-unmatch reviews/ADDON_REVIEW.md reviews/CORRECTION_PLAN.md
git ls-files -- reviews/swarm-2026-08-08 backlog/tasks
rg -n --hidden --glob '!.git/**' --glob '!docs/superpowers/specs/2026-08-13-repository-structure-cleanup-design.md' --glob '!docs/superpowers/plans/2026-08-13-repository-structure-cleanup.md' 'reviews/(PHASE_|WORKTREE_)|backlogs/daily-bug-scan-1907' .
```

Expected: the active review surfaces and task directory are still tracked; the search returns no stale path matches. Verify every Markdown path beginning with `reviews/archive/` or `backlog/reports/` exists.

- [ ] **Step 6: Commit the historical-record migration**

Run:

```powershell
git add -- backlog backlogs reviews/archive reviews/PHASE_*.md reviews/WORKTREE_*.md backlog/tasks/TASK-WORKTREE-PRUNE-001-RETIRE-REDUNDANT-FLUIDSIM-CHECKOUTS.md
git commit -m "docs: archive completed repository records"
```

Expected: one commit containing only the backlog/report move, review-record moves, and their path-reference updates.

### Task 4: Run the final repository gate

**Files:**
- Verify: all files changed since `master`
- Update: `.codex/context/repo-structure-audit-loop.md` (ignored local checkpoint only)

**Interfaces:**
- Consumes: Tasks 1-3 and the approved design.
- Produces: executable evidence that the cleanup is complete and a clean branch handoff.

- [ ] **Step 1: Run focused contract tests**

Run: `python -m pytest -q tests/test_kernel_coverage.py tests/test_contract_wording.py`

Expected: both test modules pass.

- [ ] **Step 2: Run the policy-drift validator**

Run: `python tools/validate_policy_drift_reports.py`

Expected: record the exact result. If it fails only because the current branch HEAD differs from checked-in report HEAD, classify that separately from structural failures; any missing path, malformed report, or policy failure blocks completion.

- [ ] **Step 3: Validate tracked paths and references**

Run repository-wide checks confirming that every new path exists, all intended old paths are absent except historical mentions in the approved spec and plan, active swarm/ticket evidence remains tracked, `.claude/agents/liquid-warp-blender51-port/` is unchanged, and no path under `.claude/worktrees/` changed.

- [ ] **Step 4: Validate the complete diff**

Run:

```powershell
git diff --check master...HEAD
git diff --name-status master...HEAD
git status --short --branch
```

Expected: no whitespace errors, no product-runtime or experiment-worktree paths, and a clean working tree on `codex/repo-structure-cleanup`.

- [ ] **Step 5: Obtain independent review**

Ask an independent reviewer to compare `master...HEAD` with the approved design and report only concrete path, authority, provenance, or scope violations. Resolve any valid finding before the final handoff.

- [ ] **Step 6: Finalize the autonomous checkpoint**

Update `.codex/context/repo-structure-audit-loop.md` with the branch, commits, changed artifacts, exact verification outcomes, reviewer verdict, and final state. Keep this ignored checkpoint out of Git.
