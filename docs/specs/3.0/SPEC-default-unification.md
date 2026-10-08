# SPEC — Default unification (RNA vs `SolverSettings`)

Owner module: `properties.py`, `sim/solver.py: SolverSettings`, `sim/constants.py`,
`presets/schema.py`, `blender/solver_settings.py`.
Depends on: `SPEC-settings-split-completion.md` (same files; stack on it).

## Trap this spec prevents

The RNA default is what every shipped scene, demo, preset and the GUI gate baked with.
The dataclass default is only reached by headless harnesses and tests. **Resolving the
divergence toward the dataclass changes the gate hash and every user's result; resolving
toward RNA changes nothing visible.** Pick RNA.

## Known divergences (handoff §1.4)

| Field | RNA (`properties.py`) | `SolverSettings` | 3.0 value |
|---|---|---|---|
| `viscosity_constant` | 0.05 | 0.0 | 0.05 |
| `ww_bubble_drag` | 4.0 | 8.0 | 4.0 |
| `ww_spray_drag` | 0.2 | 0.5 | 0.2 |

There may be more. Do not trust this table; generate it.

## Procedure
1. Write `tests/runtime/test_default_parity.py`: for every field in
   `blender/solver_settings.py`'s mapping that exists on both the domain RNA group and
   `SolverSettings`, assert `rna_default == dataclass_default` (with enum ints mapped
   through the same converter `apply_scene_settings` uses). Run it; the failures are the
   true divergence list. Commit the test red.
2. Change the dataclass defaults (and any `C.DEFAULT_*` constant feeding them) to the
   RNA values. Commit; test green.
3. Where a `C.DEFAULT_*` constant is also used in a kernel or a clamp (grep each), check
   the use is a default and not a physical limit before changing it.
4. Presets (`presets/schema.py`) only carry ranges, not defaults; nothing to do unless
   step 1 finds a preset file that pins a divergent value.

## Acceptance
- `test_default_parity.py` green and kept.
- GUI gate hash unchanged (nothing GUI-authored changed).
- Headless harness `tests/perf/` baselines that relied on the old dataclass defaults are
  re-recorded in the same commit with the reason in the message.
