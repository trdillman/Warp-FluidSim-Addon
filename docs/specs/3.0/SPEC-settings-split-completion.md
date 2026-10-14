# SPEC — Settings split, Phases 3–5 (completion)

Owner module: `properties.py` (`FLUIDSIM_PG_scene`, `FLUIDSIM_PG_domain`),
`blender/ui_scene.py`, `blender/ui_stages.py`, `blender/pending_settings.py`,
`blender/solver_settings.py`, `blender/setup_wizard.py`, `presets/library.py`,
`tools/release/validate_*_blender.py`, `docs/ownership-map.md`.
Background: `docs/PROPOSAL-split-settings-propertygroup.md` (Phases 0–2 DONE),
`memory: phase2-split-done`, `memory: phase2-hasattr-guard-blocked`.
Depends on: nothing. **Lands first** in 3.0.0-alpha.1; every other spec that touches
`properties.py` stacks on it.

## Trap this spec prevents

1. **`getattr(group, "field", default)` and `hasattr` guards turn a wrong-group read
   into a silent no-op.** The proposal measured 131 defaulted reads (94 % of reads).
   This is how defects #1/#3/#5 ("settings don't apply") happened. Phase 3 is the
   point of the exercise; without it the split enforces nothing.
2. **Both groups still carry all 232 fields.** `FLUIDSIM_PG_scene` is wide so authored
   values are not lost (`memory: split-does-not-invalidate-blends`). Deleting the
   scene-side fields without first *reading them into the domain object* loses user
   data in every 2.x `.blend`.
3. **Test doubles.** Many tests pass a `SimpleNamespace` as the settings group. A
   class-identity check inside the apply path (the Phase 2 guard) must accept the
   doubles the suite uses, or the suite will be weakened to make it pass. Use a
   marker attribute (`_fluidsim_group_kind = "domain"`) that both real groups and
   test doubles set, not `isinstance`.

## Phase 3 — kill defaulted reads
- Enumerate with `rg -n "getattr\((scene|fs|settings|domain|group)[^,]*,\s*\"" fluidSim_warp_blender51`.
  Every hit is one of:
  - a field that must exist on that group → direct attribute access;
  - a field legitimately optional (only `schema_version`, `active_snippet`, `snippets`
    per `memory: description-audit-exemptions`, plus the `stat_*` fields read during
    cache playback) → keep `getattr`, add the name to an explicit allowlist in
    `blender/solver_settings.py` and a test that fails on any `getattr` outside it.
- `apply_scene_settings` and `pending_settings.capture_applied_state` read **only** from
  `active_domain_settings(scene)` for domain-owned fields. `docs/ownership-map.md` is the
  authority for which fields are domain-owned; `tests/test_ownership_map.py` must pass
  on the exact tree (`memory: ownership-map-concurrent-session-drift`).

## Phase 4 — delete scaffolding
In this order, one commit each, suite green after each:
1. One-shot migration operator `fluidsim.migrate_scene_settings_3_0`: for every scene,
   for every domain object, copy any scene-group value that differs from its RNA default
   into the domain group **if the domain value equals its default** (domain wins when
   both were authored). Run automatically on file load via `blender/file_compat.py`,
   once, recording `scene.fluidsim.schema_version = 3`.
2. Narrow `FLUIDSIM_PG_scene` to the scene-owned set from the ownership map (cache path
   and range, quality tier, playback and live toggles, status/stat fields, UI mode,
   effects library). Everything else is deleted from the class.
3. Delete `setting_owner`, `_authoring_owner`, `SCENE_OWNED_SETTINGS`, the
   `owner_redirect` mirror machinery, and the five ownership-pinning tests in
   `tests/runtime/test_pending_settings_registry.py`.
4. Delete the Scene-tab panels in `ui_scene.py` that mirror domain fields. Keep only the
   scene-owned panels. The Physics tab and the staged sidebar are the domain UI.
5. `schema_version` stays (it is how step 1 knows it already ran).

## Phase 5 — non-shipped callers
- `tools/release/validate_*_blender.py` and `tests/acceptance/*.py`: migrate to
  `active_domain_settings`. They gate releases.
- `artifacts/live_review/probe*.py`, `artifacts/diagnose_*.py`: delete. They are
  one-off repro scripts and the archive tag keeps them.

## Multi-domain note
`active_domain_settings(scene)` keys off the active domain (`memory: task-4-4-scope-decision`).
Nothing in this spec changes that; do not widen to per-solver settings.

## Acceptance
- `rg -c "getattr\(" fluidSim_warp_blender51/blender/solver_settings.py fluidSim_warp_blender51/blender/pending_settings.py`
  ≤ the allowlist size, enforced by test.
- Each of the four demo `.blend` files and the 7 example `.blend` files opens, migrates,
  and `active_domain_settings(scene)` equals the pre-migration effective settings
  field-for-field (write the comparison as a headless acceptance script run on 5.2).
- A 2.x file with an authored scene-side `voxel_size` and default domain-side value
  ends up with the authored value on the domain after load.
- GUI gate hash equals 2.1.1's reference (the split is a refactor; any hash change is a
  bug until proven to be a defaults change under `SPEC-default-unification.md`).
- `description-audit` tooltip coverage test still passes on the narrowed class.
