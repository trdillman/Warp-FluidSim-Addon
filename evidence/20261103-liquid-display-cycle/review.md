# Initial review snapshot — 2026-11-03

The implementation was changing during this review. These observations refer to an intermediate uncommitted diff and must be rechecked against the final candidate.

- Legacy saved scenes with `liquid_preview_enabled=True` appear to acquire the new `SURFACE` default without explicit load reconciliation, leaving the mode and draw toggle inconsistent.
- Deck button label/state remains tied to a Boolean, so Surface and Off have indistinguishable presentation.
- When paused cache playback has no owned live session, switching to Preview calls the metadata-only output refresh. If the earlier cache publication recorded no particles because preview and show-particles were off, the preview remains empty until another cache frame is published.
- `hide_set` without an explicit view layer changes the context view layer, which may not be the scene being updated and does not cover other view layers. The code also unconditionally hides particle output and must preserve render/export and scene ownership.

No tests were run by this reviewer. Runtime, save/reopen, cache, multiple-scene/view-layer, GPU and full production acceptance remain unverified. The implementer is still changing the diff; this is not a final review or acceptance.

## Scoped re-review of subsequent working tree

**Verdict: changes requested.** Renderer failure remains a blocking correctness issue. `liquid_preview/handler.py:_draw_callback` latches `_FAILURE` globally and calls `show_surface_fallback` only for the scene that encountered the error. A subsequent `publish_payload` or `refresh_existing_output_settings` calls `liquid_display.apply_viewport`, which hides that surface again while mode remains PREVIEW; future callbacks return immediately because `_FAILURE` is set. Other scenes in PREVIEW are not restored at all. A user can therefore lose all liquid viewport output after a renderer error. Verify fallback remains effective across later publication and all affected scenes.

The later implementation retains cache particle data for paused cache transitions, loops through the target scene's view layers, migrates legacy mode on load, and presents a dynamic expanded Deck label. These address the earlier source-level concerns, but actual Blender behavior and multi-scene visibility remain unverified. The renderer fallback and preservation of render/export likewise require runtime verification.

No tests were run by this reviewer. The implementer reported 59 focused tests passing; this reviewer did not inspect their results. Full production gate, GUI playback, saved-file reopen, GPU renderer failure, cache transitions, multiple view layers/scenes, and render/export acceptance are unverified.

## Follow-up review of registration and fallback fixes

The earlier blocking fallback finding is addressed in source: renderer failure restores surfaces across preview-enabled scenes, and subsequent viewport application preserves the surface fallback while the global failure is latched. Restricted-data registration now defers restoration through a timer and unregister removes that timer. No further blocking findings in this limited follow-up. This is a source review, not runtime acceptance of the failure or registration paths.

The parent reports a real Blender run on port 9880 passed OFF save/reopen and paused mesh/preview point-record visibility. A prior run on port 9879 exited unexplained after six cycles; its cause is not established. I did not rerun tests or independently inspect those runtime receipts. Multi-scene renderer failure, restricted registration, render/export and full production acceptance remain unverified by this review.
