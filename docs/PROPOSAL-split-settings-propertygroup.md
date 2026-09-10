# Proposal: split `FLUIDSIM_PG_settings` into scene-scoped and domain-scoped groups

Status: proposed, not started. Written 2026-09-10.

## Problem

`properties.py:1728` and `properties.py:1731` bind **the same PropertyGroup class** to
two different scopes:

```python
bpy.types.Scene.fluidsim         = PointerProperty(type=FLUIDSIM_PG_settings)
bpy.types.Object.fluidsim_domain = PointerProperty(type=FLUIDSIM_PG_settings)
```

All 232 annotated fields therefore physically exist on both datablocks. Nothing but
convention says which copy is authoritative for a given field, so reading the wrong one
returns a plausible-looking stale value instead of failing.

Every one of the following exists only to re-derive, at runtime, the ownership
information that this binding discards:

| mechanism | file |
|---|---|
| `active_domain_settings()` | `blender/scene_bridge.py` |
| `schema_version` + one-way copy | `blender/migration.py` |
| `setting_owner()` / `_SCENE_OWNED_TIERS` / `_SCENE_OWNED_FIELDS` | `blender/pending_settings.py` |
| `_authoring_owner()` / `_DOMAIN_AUTHORED_SIGNATURE_FIELDS` | `blender/pending_settings.py` |
| `SCENE_OWNED_SETTINGS` | `presets/schema.py` |
| `_domain_owner()` / `owner_redirect` mirrors | `blender/ui_scene.py`, `blender/ui_physics.py` |

Defect #5 and the ~53-field "UI writes scene, solver reads domain" bug are both symptoms
of it, not independent bugs.

## Proposed end state

Two PropertyGroups, disjoint:

- `FLUIDSIM_PG_scene` on `Scene.fluidsim` — **85 fields**
- `FLUIDSIM_PG_domain` on `Object.fluidsim_domain` — **147 fields**

No field on both. A read against the wrong group raises `AttributeError` at the read
site rather than returning a stale value. `migration.py` and `schema_version` are
deleted outright; `setting_owner`, `_authoring_owner` and `SCENE_OWNED_SETTINGS` are
deleted; `active_domain_settings()` shrinks to a plain accessor with no fallback branch
and no version check.

This is a `.blend` file-format break. Acceptable — the project makes no backward
compatibility guarantee.

## Measured blast radius

All figures verified by static analysis of the current tree (`feat/ui-redesign-phases-0-3`).

| quantity | value |
|---|---|
| annotated fields on the shared class | 232 |
| already classified by the tier registry | 232 (100%, no orphans) |
| candidate scene-scoped | 85 |
| candidate domain-scoped | 147 |
| fields whose scope is **contested** (UI writes scene, consumer reads domain) | 35 within the reset/reinit signature; ~53 including SURFACE-tier and whitewater fields outside it |
| `scene.fluidsim` references in the shipped add-on (non-test) | 206 across 31 files |
| files outside the add-on referencing either datablock | 72 (`tools/`, `artifacts/`, root review scripts, repo tests) |
| `getattr()` calls against a settings group | 139 |
| ...of those, passing a default | **131 (94%)** |

The tier registry already partitions all 232 fields with no unclassified remainder. That
is the single biggest de-risker: the partition does not have to be invented, only
audited and then enforced by the type system.

## The 94% problem

131 of the 139 `getattr()` reads against a settings group pass a default:

```python
profiling_enabled=bool(getattr(fs, "profiling_enabled", False))
```

After the split, that line **still returns `False`** when `fs` is the wrong group. It
does not raise. Without Phase 3 below, the split buys naming clarity and almost none of
the enforcement value that motivates it. Worst-affected files: `blender/playback.py`
(18), `blender/output_publication.py` (16), `blender/pending_settings.py` (12),
`blender/ui_scene.py` (11), `blender/geometry_nodes.py` (10).

**This is the main reason the split could fail to pay for itself.** Treat Phase 3 as
mandatory, not optional cleanup.

## Phases

Each phase is independently shippable and independently revertable.

### Phase 0 — Produce the authoritative ownership map (no code changes)

Deliverable: a checked-in table, one row per field, three columns — where the UI writes
it, where its consumer reads it, declared scope. Build it by AST-resolving every panel
`draw()` (including the `owner_redirect=True` mirrors in `ui_physics.py`, which rewrite
`context.scene.fluidsim` to the domain group) and every consumer read site.

**This is the gate.** No later phase starts until every field has exactly one declared
scope. The 35–53 contested fields are resolved here, on the evidence, and the resolution
is domain for essentially all of them (their consumers are `build_settings`,
`gather_height_source`, `simulation_admission`).

Naive scope estimate is misleading here: the map is the hard part, not the refactor.

### Phase 1 — Repoint the contested panels

Point the ~53 contested panels at `_domain_owner` so the UI writes where the solver
reads. This is the standing live bug fix.

**This phase changes simulation results.** Settings that were silently ignored on
migrated scenes start applying — height fields, per-attribute output toggles, surface
tension and whitewater parameters. Expect demo `.blend` scenes, any golden outputs, and
the perf benchmark baselines to shift. Ship it alone so the change is bisectable and the
result diffs are attributable.

### Phase 2 — Split the class

Mechanical but wide: 206 references across 31 non-test files in the shipped add-on.
Behavior-neutral by construction if Phase 0 and 1 are correct; any behavior change here
is a bug.

### Phase 3 — Audit the 131 defaulted `getattr()` reads

Convert to direct attribute access wherever the field must exist on that group. Keep the
default only where absence is genuinely legitimate. Without this the split does not
enforce anything.

### Phase 4 — Delete the scaffolding

`migration.py`, `schema_version`, `setting_owner`, `_authoring_owner`,
`SCENE_OWNED_SETTINGS`, the `owner_redirect` mirror machinery, and the five ownership
tests in `tests/test_pending_settings_registry.py` that exist to pin a partition the
type system now enforces.

### Phase 5 — Non-shipped callers

72 files. Triage rather than migrate wholesale: `tools/release/validate_*_blender.py`
gate releases and must be migrated; `artifacts/live_review/probe*.py` and
`artifacts/diagnose_*.py` are historical one-off repro scripts and should be deleted or
left to rot deliberately, not carried forward.

## Risks

1. **The 94% defaulted-`getattr` problem** — mitigated only by Phase 3. Highest risk to
   the value of the whole exercise.
2. **Phase 1 changes simulation output.** Not a regression, but it will read like one in
   any comparison against existing baselines. Re-baseline deliberately.
3. **41 test files** reference the datablocks and will need updating alongside Phase 2.
4. **Blender RNA/registration** — renaming the registered PropertyGroup invalidates every
   existing `.blend`. Accepted by the no-backcompat position, but it means no gradual
   rollout: the split lands atomically for users.

## Recommended sequencing

Phase 0 first and alone — it is cheap, it is the gate, and its output is independently
valuable even if the split is never done (it is the map that would have prevented
defect #5). Decide whether to continue after reading it.
