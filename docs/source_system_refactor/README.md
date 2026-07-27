# Source System Refactor — Program Roadmap

## Purpose

This directory is the execution plan for making FluidSim Warp's source system
truthful, complete, and understandable without presenting planned capabilities
as implemented ones.

The program is intentionally ordered as:

1. **Phase 1 — UI Truthfulness and Workflow Repair**
2. **Phase 2 — Source Backend Normalization and Capability Completion**
3. **Phase 3 — Full Source UI Information Architecture**

This order differs from a UI-first redesign. Phase 1 corrects current lies and
missing controls, Phase 2 makes the requested source capabilities real, and only
then does Phase 3 expose the complete product model.

## Authoritative contracts

These plans supplement rather than replace:

- `AGENTS.md` — verified test and production-gate workflows.
- `docs/SOURCE_EMISSION_PRODUCT_PLAN.md` — target source lifecycle and
  mesh-region semantics. Its historical implementation-status/gap matrix must
  be reconciled against the current branch at G0; stale status text is not
  evidence that behavior is still missing or complete.
- `docs/source_shape_spec.md` — primary-source Houdini semantics for Source
  Shapes, including the non-liquid FIELD representation.
- `reviews/ADDON_REVIEW.md` — review findings and completed corrective work.

If target semantics conflict, agents must stop and resolve the conflict in the
plan or with the user. If only implementation status conflicts, G0 must inspect
current source, behavioral tests, and Blender evidence, then record an explicit
status-resolution note before work proceeds. Agents must not silently choose the
easier interpretation or treat either stale prose or passing unit tests alone as
conclusive.

## Program goal

Deliver one coherent source system in which:

- geometry/input representation is independent of operation and lifecycle;
- liquid emitters support Initial Fill and Continuous Inflow;
- analytic regions, solid meshes, surface shells, points, volume grids, and
  height regions are represented truthfully;
- sinks, colliders, force regions, and field sources are not mislabeled as
  liquid emitters;
- source data is normalized into GPU-resident structures compatible with the
  flat sparse 8×8×8 leaf table;
- UI labels, saved demos, tests, documentation, and runtime behavior agree;
- Blender-owned RNA/mathutils values are copied into owned descriptors before
  they cross the Blender-to-solver seam;
- no review artifact, label, enum, or mock is accepted as proof of capability.

## Program non-goals

- Do not replace the live flat sparse leaf table with OpenVDB/NanoVDB hierarchy.
  VDB may be an import/staging format only.
- Do not expose arbitrary Volume/Second controls without a separate approved
  physical flux contract.
- Do not make FIELD mode emit liquid. Original FIELD behavior is pressure/field
  influence and remains advanced or hidden until implemented truthfully.
- Do not preserve misleading UI merely because an RNA compatibility property
  exists.
- Do not rewrite unrelated solver, surface, cache, or rendering systems.

## Program architecture

The intended deep-module seam is:

```text
Blender adapters
  -> immutable SourcePlan descriptors
  -> SourceCompiler / SourcePipeline
  -> compiled GPU source regions and payloads
  -> source operations
```

The small interface must represent these independent axes:

- **Operation:** liquid emit, sink, collision/activation mask, force, or advanced
  field source.
- **Region/input:** analytic SDF, mesh solid, mesh shell, point set, imported
  volume grid, or height field.
- **Lifecycle:** Initial Fill, Continuous Inflow, or explicitly operation-specific.
- **Payload:** velocity, pressure, temperature, viscosity, density attribute,
  lifespan, and optional scalar/vector fields.
- **Blend:** only where the runtime operation actually honors it.

Adapters may vary; callers must not learn adapter-specific implementation
complexity. Tests should exercise the same SourcePlan/compiled-region seam used
by Blender.

## Phase dependency map

```text
Phase 1: truthful current UI
    |
    v
Phase 2A: source contract + failing tests
    -> 2B runtime identity/lifecycle
    -> 2C sparse compiler + mesh solid/shell
    -> 2D point/particle adapter
    -> 2E height/profile completion
    -> 2F operation-specific liquid/sink/force/field gates
    -> 2G GPU/performance hardening
    -> 2V separate Volume-grid adapter milestone
    |
    v
Phase 3: final role-specific UI + docs + demos + full acceptance
```

Phase 2 work packages may be implemented sequentially and reviewed separately.
The final Phase 2 gate requires all capabilities that Phase 3 will expose.

## Common agent workflow policy

These policies apply to every phase and must be copied into worker handoffs.

### Repository and scope safety

1. Read `AGENTS.md`, the relevant phase plan, and the authoritative contracts
   before editing.
2. Capture `git status --short` before work. The worktree may already contain
   unrelated modifications; never reset, clean, stage, or rewrite them.
3. Keep exactly one writer in an active worktree. Planning and review agents are
   read-only. Parallel writers require isolated clean worktrees and explicit
   approval.
4. Make surgical edits. No opportunistic formatting or unrelated cleanup.
5. Preserve existing line endings and saved-file compatibility unless a
   migration is explicitly approved.
6. Never modify the user's persistent Blender profile. Visible review sessions
   must use the reusable profile-safe launcher created in Phase 1. The required
   command shape is:

   ```bash
   python tools/launch_profile_safe_addon_review.py \
     --blend demos/dam_break.blend \
     --report artifacts/profile_safe_source_review.json
   ```

   The launcher must snapshot enabled add-ons plus `userpref.blend` and
   `startup.blend` existence/hashes, copy the worktree add-on to a temporary
   add-on root, disable conflicting package identities only in the owned
   process, enable the temp copy non-persistently, record PID/module/version and
   active blend, wait for the owned process to close, remove the temp copy, and
   emit before/after integrity results. On mismatch it fails and retains backup
   evidence; it never overwrites the user's profile or kills unrelated Blender
   processes.

### Truth policy

1. Build or update a claim matrix before adding a user-facing label.
2. A capability is **real** only when source behavior, focused runtime evidence,
   and appropriate integration evidence agree.
3. A symbol, enum, panel, mock, demo label, or non-crash test is not behavioral
   proof.
4. Missing or partial behavior must be hidden, labeled Experimental, or called
   out as a non-goal. It must not be presented as complete.
5. Never silently fall back from Solid Volume to Surface Shell, from volume
   sampling to vertices, or from GPU behavior to a cosmetic preview.
6. FIELD compatibility must remain non-liquid unless pressure/field sourcing is
   implemented and independently validated.

### Test-first policy

1. Reproduce or encode the behavior at the deepest correct seam before fixing it.
2. Prefer behavioral assertions over source-string assertions.
3. Blender-runtime defects require a Blender-runtime regression gate when a
   pure Python seam cannot represent the failure.
4. Tests must cover invalid input and failure reporting, not only the happy path.
5. Any skipped acceptance lane must be reported as unverified; headless proof
   must not be called full GUI acceptance.

### Delegation policy

- **Phase lead:** owns scope, decisions, and final synthesis.
- **Context/planning agents:** read-only; return file/line evidence, dependencies,
  tests, and unresolved decisions.
- **Writer:** sole source-code writer for the phase or work package.
- **Standards reviewer:** fresh context; checks repository rules, module depth,
  maintainability, and regression risk.
- **Spec/truth reviewer:** fresh context; checks implementation against this
  plan and authoritative source contracts.
- **Validator:** read-only; runs or inspects the exact required gates and artifacts.

A reviewer must not approve its own implementation. Reviewer findings are
triaged by the phase lead; optional polish does not trigger endless loops.

### Stop and escalation rules

Stop before implementation or handoff when:

- a UI decision would expose behavior not yet implemented;
- a saved-file migration or default change is not backward compatible;
- Solid Volume validity, VDB access, or source-field semantics are ambiguous;
- an external dependency or hierarchical live storage would be introduced;
- the required GPU/Blender acceptance lane cannot run;
- a performance threshold has not been baselined and approved;
- a failing test indicates a pre-existing defect that invalidates the phase
  assumptions.

## Common reviewer gate model

Each phase uses the following gates. A phase-specific file adds exact evidence.

### G0 — Baseline and contract gate

- Baseline status and focused tests recorded.
- Claim matrix identifies real, partial, missing, compatibility-only, and
  explicitly deferred behavior.
- Scope and non-goals reviewed before writing.

### G1 — Focused behavioral gate

- New tests fail for the intended reason before implementation where practical.
- Focused Python tests pass after implementation.
- No shallow symbol-presence test substitutes for behavior.

### G2 — Independent code review gate

Two fresh-context reviews:

1. Standards/module review.
2. Spec/truth review.

The handoff records reviewer run/session identifiers, model, fresh-context
attestation, findings, and disposition. No blocker or high-severity finding may
remain. Deferred findings require owner, rationale, and target phase.

### G3 — Blender runtime gate

- Required background acceptance markers pass.
- Visible GUI review is performed when UI or interaction changes.
- Playback advances through multiple frames and remains responsive.
- Temporary add-on identity and profile integrity are proven.

### G4 — Integration/production gate

Run the phase's required production subset. The final program gate is:

```bash
python tools/run_production_gate.py --include-gui
```

A headless-only result must be reported separately.

### G5 — Handoff gate

The phase handoff is complete and reproducible. The parent/orchestrating phase
lead—not an implementation worker or reviewer—issues the final verdict after
checking all mandatory evidence and reviewer dispositions. The next phase lead
must acknowledge the handoff before work begins. No phase proceeds on verbal
“looks good” alone.

## Common handoff requirements

Every phase/work-package handoff must include:

1. Goal and delivered scope.
2. Explicit non-goals and deferred work.
3. Changed files grouped by ownership/module.
4. Interface and saved-data changes.
5. Claim-matrix delta: what moved from missing/partial to real.
6. Exact commands, exit codes, pass markers, and artifact paths.
7. Manual GUI actions and observations.
8. Failed attempts and known environmental limitations.
9. Reviewer findings and their disposition.
10. Performance/memory before-and-after evidence when relevant.
11. `git status --short`, identifying pre-existing versus authored changes.
12. Residual risks and rollback instructions.
13. Next-phase prerequisites and unresolved decisions.
14. A concise implementation-ready prompt for the next sole writer.

Use the following completion verdicts only:

- **Accepted:** all mandatory gates passed.
- **Accepted with documented deferral:** no blockers/high findings; named work is
  intentionally assigned to a later phase.
- **Blocked:** a required behavior, decision, or acceptance lane is missing.
- **Failed:** the implementation does not satisfy the phase truth contract.

## Plan files

- `PHASE_1_UI_TRUTHFULNESS.md`
- `PHASE_2_SOURCE_BACKEND.md`
- `PHASE_3_UI_REFACTOR.md`
