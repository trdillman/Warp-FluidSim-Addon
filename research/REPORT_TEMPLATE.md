# <Lane title>

## 1. Research metadata

- Lane:
- Researcher:
- Date (UTC):
- Project constraint summary:
- Scope exclusions:
- Evidence root:

## 2. Executive conclusions

State the strongest source-backed conclusions, the recommended implementation families, and the largest unresolved risks. Distinguish facts, engineering inferences, and proposals.

## 3. User-visible capability definition

Define what realistic success would look like in Blender playback, caching, rendering, controls, and failure presentation. Separate minimum viable capability from later fidelity tiers.

## 4. Physical model and terminology

Define the governing quantities, equations, assumptions, stability constraints, and relevant nondimensional numbers. State what the model can and cannot represent.

## 5. Implementation technique comparison

For each credible technique, document:

- Algorithm and representation.
- Accuracy and known artifacts.
- GPU suitability and synchronization/readback implications.
- Memory and compute scaling.
- Sparse 8x8x8 leaf-grid compatibility.
- FLIP/APIC/MPM coupling implications.
- Surface reconstruction, cache, and viewport/render implications.
- Maturity, licensing, and source evidence.

Include a decision table with: technique, fidelity, complexity, GPU cost, memory cost, project fit, evidence strength, and recommendation.

## 6. Primary-source evidence

Every material claim must cite both the original URL/publication identifier and a local artifact path under this lane. Prefer papers, specifications, official documentation, and source code at immutable revisions. Note inaccessible or incomplete sources explicitly.

## 7. Reference implementations and code findings

For each GitHub/GitLab/official codebase:

- Repository and immutable commit/tag.
- License and integration implications.
- Exact files/functions/kernels inspected.
- Technique demonstrated.
- What is reusable as an idea, what might be reusable as code, and what is unsuitable.
- Local clipped/sample path and hash manifest entry.

Do not infer production behavior from names, demos, documentation, or tests alone.

## 8. Industry/product feature baseline

Compare documented behavior in established DCC, simulation, VFX, or scientific tools. Use first-party manuals/specifications where possible. Separate documented feature claims from observed implementation details and industry convention.

## 9. Fit to FluidSim architecture

Map the research to this supplied project contract only:

- GPU-resident simulation and interactive viewport playback.
- Sparse tiled field storage using fixed 8x8x8 leaf tiles through a flat leaf table.
- Existing synthesized enums, kernel inventory, dispatch/custom-kernel contracts, UI behavior, profiling, and validation targets remain authoritative.
- Blender adapters must not own solver policy.
- Persistent output, preview playback, production cache, and diagnostics have separate owners.

Name likely new fields, particle attributes, kernels/phases, boundary conditions, cache channels, output attributes, UI settings, and validation seams. This is a proposal, not proof of current implementation.

## 10. Performance and numerical-risk analysis

Cover timestep restrictions, conservation, diffusion, interface thickness, conditioning, atomic/nondeterministic operations, precision, sparse activation, memory budgets, and worst-case workloads. Identify where host synchronization could accidentally enter the hot path.

## 11. Validation and acceptance strategy

Propose analytical tests, conservation/error metrics, convergence studies, visual benchmarks, cross-solver comparisons, GPU profiling, cache/reopen checks, and Blender interactive playback acceptance. Give measurable pass/fail criteria where sources justify them.

## 12. Recommended staged roadmap

Give ordered research-to-prototype stages with prerequisites, stop/go criteria, and the smallest meaningful vertical slice. Do not write production code or claim schedule certainty.

## 13. Dependencies and cross-lane coupling

Identify required shared fields and solvers, conflicts with other lanes, and ordering constraints. Call out dependencies on color transport, variable density/multiphase interfaces, temperature/energy, or phase fraction.

## 14. Risks, unknowns, and rejected approaches

List evidence gaps, licensing blockers, algorithms rejected for this project, and questions requiring prototypes or domain-expert review.

## 15. Source index

| ID | Primary source | Immutable version | Local evidence | License | Claim(s) supported |
|---|---|---|---|---|---|

## 16. Orchestrator ingestion block

Keep this section concise and machine-scannable.

- Recommended technique(s):
- Alternative technique(s):
- Required shared state:
- New solver phases/kernels:
- Cache/output schema impact:
- UI impact:
- Strongest validation targets:
- Major blocker(s):
- Cross-lane dependency order:
- Confidence: high / medium / low, with reason

