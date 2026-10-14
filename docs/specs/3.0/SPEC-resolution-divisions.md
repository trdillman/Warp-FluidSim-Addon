# SPEC — `resolution_divisions` replaces authored `voxel_size`

Owner module: `sim/quality_contract.py` (`AuthoredQualitySettings`, `resolve_quality`),
`properties.py`, `presets/`, `blender/pending_settings.py`, `blender/setup_wizard.py`,
`blender/ui_stages.py` (Setup stage), `sim/fields.py: SparseGrid.__init__`.
Depends on: `SPEC-settings-split-completion.md`, `SPEC-cache-v3.md` (grid_spec stamp).

## Trap this spec prevents

1. **Quality tiers already rescale `voxel_size`** (`_QUALITY_*_FIELDS` in
   `quality_contract.py`; `memory: quality-tier-wiring-done`). If divisions are added
   beside `voxel_size` instead of replacing it, two controls fight and the tier
   multiplier applies twice. `voxel_size` becomes **derived and read-only** in the UI.
2. **Leaf alignment.** The grid is 8³ leaves. Divisions must round so the padded
   dimension is a multiple of 8 per axis (`padded_voxel_dimensions` already does this);
   show the user the *effective* cell size, not the requested one.
3. **Domain resize now changes cell size.** That is the Mantaflow-style semantics
   artists expect, but it means scaling the domain object is a RESET-tier edit. Add
   `domain_size` to the RESET signature in `pending_settings.py` or the live preview
   will keep running on a stale grid.
4. `MIN_VOXEL_SIZE` clamp still applies after derivation; when it clamps, surface a
   diagnostic ("resolution capped at N divisions by minimum cell size").

## Definition
`voxel_size = longest_domain_axis / resolution_divisions`, computed in
`resolve_quality` from `AuthoredQualitySettings.resolution_divisions` and the domain
bounds that `SparseGrid.__init__` already passes. Quality tier multiplies divisions
(Draft 0.5, Preview 1.0, Final 2.0) then rounds to the nearest integer ≥ 8.

## Settings
- Domain group: `resolution_divisions: Int`, default 64, soft range 16–512, hard min 8.
  Tier RESET.
- `voxel_size` RNA property is deleted; a read-only UI label shows the effective size
  and the padded grid dimensions.
- Presets: replace `voxel_size` entries with `resolution_divisions`. Migration maps
  `divisions = round(longest_axis / voxel_size)` per domain on file load (part of the
  3.0 migration operator in the split spec). Demos are re-saved with explicit divisions.
- `surface_resolution_multiplier` and `surface_voxel_size` are unchanged (they derive
  from the effective `voxel_size`).

## Acceptance
- `resolve_quality` unit tests: cube 2 m, 64 divisions → `voxel_size` 0.03125;
  non-cube 2×1×0.5 m, 64 → 0.03125 on x, padded dims (64, 32, 16).
- Each demo migrated from its 2.1.1 `voxel_size` reproduces the same padded grid
  dimensions (so alpha.2's hash change is attributable to transfer, not to the grid).
- Scaling the domain object in the viewport raises a RESET pending state.
