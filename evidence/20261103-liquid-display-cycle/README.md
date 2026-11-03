# Liquid viewport display cycle — 2026-11-03

Base: 665bd48aaf02522db5e69f014ea0b66f79b3a886, verified origin/master by parent.

Intent: cycle Preview → Surface Mesh → Off in the Deck, showing at most one liquid representation while preserving separately controlled whitewater and render/export configuration. Existing preview Boolean remains an API adapter; a scene-owned saved enum is the display authority. An unset enum in an older file inherits the legacy preview Boolean on load. The viewport policy uses per-view-layer hide state and retains render flags. Preview point capture is independent of particle render selection; cache records retain already-read points for paused mode transitions. On shader failure, the mesh is revealed when available.

Alternatives considered: changing show_surface/show_particles would alter publication and rendering; an independent viewport mode preserves those policies. A transient mode would lose the user's selection on reopen.

Tasks: implement mode and legacy adapter; synchronize viewport objects and draw handler; preserve preview points for live/cache; test cycles and visibility, migration and reopen in Blender; run production gate when GPU is available; review and prepare PR. Runtime and GPU acceptance remain pending parent verification. No merge or release authorized.

Focused CPU verification: `python -m pytest -q tests/ui/test_deck_state_follow_through.py tests/ui/test_liquid_preview.py tests/runtime/test_pending_settings_registry.py --basetemp C:/workspace/liquid-cycle-suite11-temp` → 61 passed. Cycle test was observed failing before implementation and passing after. This is not Blender runtime acceptance.
