# Species v2 Reconciliation

## Preserved candidate

- Source branch: `codex/species-v2`
- Preserved tip: `9f44cd44cf5f0030922d68f686862c840cf9eb81`
- Classification: valuable prototype, not historical debris and not merge-ready
- Candidate-focused audit: 222 tests passed on the preserved tip

Do not delete the source branch or wholesale merge/cherry-pick its divergent history. Reuse requires a fresh port onto current `master` ownership seams.

## Already present on master

Current `master` already owns the user-facing implementation that prompted this reconciliation:

- per-object settings through `Object.fluidsim`;
- the Object Properties > Physics FluidSim root and action panels;
- per-source RGBA liquid color;
- GPU-resident practical color mixing;
- the independently corrected swarm-review contracts.

The current object/Physics UI is factored through `blender/ui_object.py` and `blender/ui_physics.py`. The candidate's monolithic panel is older and must not replace or duplicate it.

## Valuable candidate work not on master

The branch contains a prototype concentration/species pipeline: particle concentration state, GPU exchange, cache/checkpoint hooks, Beer-Lambert-derived surface color, acceptance scripts, and profiling tools.

It is not a bounded merge candidate because it changes the particle ABI, all source families, solver ordering, cache schema, output publication, UI, and performance evidence together. It also does not yet satisfy its own dynamic-species design:

- species count is hard-coded to two rather than runtime-dynamic;
- storage remains fixed-width rather than active four-species chunks;
- it exposes a separate species diffusion-substep control;
- disabling species requests solver rebuild instead of preserving solver identity and concentrations;
- public authoring, arbitrary stable IDs, cache/resume acceptance, and release proof remain incomplete.

## Fresh-port sequence

1. Preserve current no-species and RGBA color-mixing behavior.
2. Add a pure stable-ID species registry and active four-species chunk schema.
3. Compile registry and mixtures through current Simulation Admission and every source family.
4. Add concentration transport and hot pause/resume without replacing solver identity.
5. Add capacity, cache/checkpoint, output-publication, and GPU surface-color support incrementally.
6. Prove N=1 through N=5, source lifecycle, off/on identity, cache reopen, zero-work no-species behavior, Blender/CUDA runtime, and performance thresholds before integration.

This is a separate feature project requiring explicit authorization for the material architecture and behavior expansion. Until then, the branch and this record preserve the work without weakening `master`.
