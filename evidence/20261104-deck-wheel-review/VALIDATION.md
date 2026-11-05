# Liquid Deck wheel review corrections

Base: master 80bf4aaae0a108602ef1ef73cdf6c74e30dc8354 (PRs #44, #45 and #46 merged).
Scope: shared wheel path for Deck-button and Alt-RMB entry. No solver, dependency,
installation, display-cycle, or package changes.

- Tooltip fallback clears FOOTER_EXTENT plus a gap; if neither above nor below
  fits without crossing the Deck bar, omit the card rather than obscure controls.
- Successful paint publishes the resolved controls and scene/geometry identity.
  Picks use those controls, checking current availability. Unpainted, failed,
  resized, rescaled, relocated and foreign-scene layouts cannot execute slots.
- MOUSEPAN/NOTHING accumulates vertical motion to 40 UI-scaled pixels; reversal
  resets accumulation. Horizontal, zero, modified and out-of-region pans do not
  turn pages. At most one page per event, without residual overshoot.

Validation (system Python):
- New navigation regressions against unchanged base: 15 failed, 30 passed.
  Failures reproduce tooltip overlap, queued-navigation misrouting, missing
  paint invalidation and ignored vertical trackpad events.
- All tests/ui/test_deck*.py plus tests/ui/test_artist_preview.py: 256 passed.
  Includes existing Preview -> Surface Mesh -> Off and both wheel entry paths.
- Ruff on all three changed Python files: passed. git diff --check: passed.
- Broader Deck/artist-preview/liquid-preview/cache-playback/reference selection:
  356 passed, 6 failed. All six failures reproduce with unchanged base wheel:
  ModuleNotFoundError for Warp during output-publication/simulation imports.
  They are not counted as passing or repaired. The broader selection inadvertently
  included the native offscreen reference test; no install/profile changes were made.
- Storage startup check completed; free-space and bounded-scan warnings reported.
  No storage cleanup performed. Evidence budget < 100 KiB; owner Codex Liquid Deck
  repair; retain with PR; active task lease ends at handoff. Source/tests protected.

Limits: synthetic bpy/drawing/event doubles establish routing and card bounds,
not framebuffer appearance, real event delivery, or hardware trackpad sensitivity.
No physical trackpad, interactive Blender UI, simulation GPU, performance, package,
or release qualification is claimed. Existing unresolved #46 threads are historical;
review this follow-up correction before merging it.
