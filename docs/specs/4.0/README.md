# 4.0 specs

Each spec exists to prevent one or more concrete traps, names the real symbols it
touches, and defines acceptance. Plan: `docs/RELEASE_4_0_PLAN.md`. Interactive UI reference:
`docs/design/liquid_deck_mockup.html` (https://claude.ai/artifact/5jjwB1o9BEDnYjseufEe1V).

| Spec | Plan item | Lane |
|---|---|---|
| `SPEC-liquid-preview.md` | §1 Liquid Preview (screen-space fluid rendering) | V |
| `SPEC-liquid-deck.md` | §2 Liquid Deck (in-viewport interaction layer) | K |
| `SPEC-rigid-coupling.md` | §3 Two-way rigid-body coupling, cache v3.1 | R |
| `SPEC-defaults-4.md` | §4 Multigrid default and re-baseline | D |
| `SPEC-learned-pressure-guess.md` | §5 Learned pressure guess | N |
| `SPEC-warp-118-surface.md` | §6 Warp 1.18 certification, sparse marching cubes, VDB narrow band | W |

Shared rules (from `AGENTS.md` and `CONTEXT.md`): a behaviour change lands behind a
failing regression test; `Output Publication` owns every Blender ID mutation;
`viewport_overlay.py` and the new draw layers only draw; deterministic bakes must
stay bitwise repeatable (no atomics on the bake path); no host sync inside captured
graphs.
