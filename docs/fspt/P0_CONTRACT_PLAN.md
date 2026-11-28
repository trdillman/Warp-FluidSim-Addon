# FSPT P0 contract implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Root owns native synchronization, commits and final adjudication.

**Goal:** Freeze the joint FSPT compatibility, quality, freshness and solver-overhead contract without promoting an unmeasured result.

**Architecture:** A versioned JSON contract records the adopted engineering thresholds and preserved product scope. A pure evaluator beside the existing gate code rejects inconsistent or insufficient trial records. Existing production qualification remains the authority for native runtime, packages and release.

**Tech Stack:** Python standard library, pytest, existing FluidSim gate and Blender/CUDA evidence.

**Spec:** `P2_LIVE_BRIDGE_SPEC.md`, the architecture roadmap sequence 11, and the 27 November implementation/qualification Page.

## Global constraints

- Keep `tools/performance_budget.py` and its release references, allowances and accuracy guards unchanged.
- No solver, renderer, native worktree or existing acceptance script edits in this lane.
- Preserve full requested Principled, attributes, lights, liquid-interior, whitewater and export coverage as requirements.
- P0 freezes targets; it does not certify that the current renderer satisfies them.
- Native GPU jobs and mutations remain root-owned. Tests here use synthetic report records only.
- Lead approved authored-24-Hz Live: 24 fresh solved states/s; p95 age 100 ms; worst age 250 ms; D/A wall and p95 interval at most 1.05; combined VRAM at most 80%; simulated-time/wall at least 1.0 for the realtime claim.
- Lead approved native 1080p independent 1/2/4-spp Live profiles, 12 total scattering events including at most 12 transmission events, no implicit quality adaptation.

## Review focus

- Moving timeline labels with frozen geometry or pixels must not count as fresh work.
- Cheap rendering obtained by changing solver, output obligations, resolution or quality must fail.
- Fast median or enqueue timing must not hide slow completed steps or old displayed images.
- Missing, nonfinite, reversed or duplicate state records must fail closed.
- Passing this pure evaluator must never be labeled native or installed qualification.

## Task 1 — freeze the contract and guard false performance success

**Files:**

- Create `tests/render/fixtures/fspt_acceptance_contract_v1.json`.
- Create `tools/gate/fspt_contract.py`.
- Create `tests/render/test_fspt_acceptance_contract.py`.
- Create `docs/fspt/P0_ACCEPTANCE_CONTRACT.md`.

**Interfaces:**

- `load_contract(path=None) -> dict`: load the repository's frozen contract.
- `contract_sha256(contract) -> str`: canonical JSON identity, independent of line endings.
- `evaluate_live_trial(report, contract=None) -> dict`: return `PASS` or `FAIL`, numeric metrics and reasons. This is a pure evidence evaluator, not a runtime launcher or qualification receipt.

- [x] Write a valid 600-state paired synthetic report and negative controls for no images, repeated source/pixel content, altered quality, mismatched solver settings, excessive overhead/age/VRAM, incomplete warmup and missing quality evidence.
- [x] Run the tests and retain the expected missing-evaluator failure.
- [x] Implement the smallest evaluator that recomputes completed-step timing, actual fresh-state counts and joint threshold decisions from the records.
- [x] Run the focused tests and independent negative-control mutation checks.
- [x] Document fixture bindings, masks, error definitions, quality presets, required compatibility/pass semantics and exact existing qualification commands.
- [x] Deliver the immutable file hashes to root for source review and native commit.


## Verification record

The initial missing-evaluator control failed as expected. A later generation-reuse control reproduced a false fresh-state pass; the evaluator now rejects reused publication generations. Four malformed-type controls then reproduced three false passes and one exception; strict profile and state/count/date validation closes them. The focused host suite and Ruff results are reported with the final frozen file hashes. These controls use synthetic records and make no native/runtime/installed claim.
