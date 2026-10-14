# SPEC — 4.0 defaults: Multigrid preconditioner, re-baseline, 3.0 leftovers

Owner module: `sim/solver_settings_model.py` (`pressure_preconditioner` default),
`properties.py` (the Blender-authored default and the stamped `domain_schema_version`),
`blender/file_compat.py` (migration), demo rebuild (`tools/demos/build_demo_scenes.py`),
`docs/DEMO_SCENE_CONTRACT.md` evidence rebind. Depends on: Warp 1.18 lane landing first
(so the re-baseline happens once).

## Trap this spec prevents

1. **Silently changing old files.** Files saved before 4.0 keep Jacobi via the
   `domain_schema_version` stamp (the same mechanism B62 used for the 64-division
   default). A one-time notice in the Bake stage says "4.0 default is Multigrid; this
   file keeps Jacobi" with a button to switch. New files get Multigrid.
2. **Two re-baselines.** The demo hash changes once at alpha.1 for Multigrid *and* the
   sparse marching-cubes extractor together. Lane W lands first; lane D rebuilds the
   demos once and rebinds `max_crown_height` (crown phase, frames 2–5) per
   `docs/DEMO_SCENE_CONTRACT.md`.
3. **Mistaking preconditioner for solution change.** Multigrid changes iterates, not
   the converged solution beyond tolerance. Prove it: with tolerance tightened 100×,
   Jacobi and Multigrid final pressure fields agree to 1e-4 relative on dam_break frame
   10. The hydrostatic-column test (`tests/solver/test_hydrostatic_column.py`, new)
   asserts the 3.0 known limitation is gone: a 2 m column settles within 2% of its
   analytic height by frame 30.
4. **Narrow band / APIC drift.** Both stay opt-in; their defaults are untouched in this
   lane. Do not bundle.

## Also in this lane (3.0 handoff leftovers)

- Remove `TIME_OFFSET` colour-mode remnants (the enum item was removed in 3.0; confirm
  no migration path references it).
- Guard every unregister loop (one helper, used by all registration sites, continue on
  failure, report at the end) — shared with the Deck registry.
- CHANGELOG: Upgrade notes ("Simulation results change. Bake caches again."), the
  schema stamp, the notice.

## Acceptance

- `test_hydrostatic_column.py` red on Jacobi, green on Multigrid.
- `test_preset_schema_matches_properties.py` and the ownership map regenerate cleanly;
  settings signature re-pinned with a changelog line.
- Full gate pass; new reference hash recorded; evidence rebound.
- ABBA on all four demos vs 3.2.1: ≤ +10% per frame on dam_break Final, iterations
  ≥ 3× fewer at 1e-5.
