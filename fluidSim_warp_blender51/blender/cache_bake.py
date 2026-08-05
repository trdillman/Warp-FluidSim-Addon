"""Deterministic sequential Production Cache Bake.

The runner advances the Solver Session directly and writes authoritative cache
payloads.  It does not depend on Blender frame-change delivery or Output
Publication, and it never consumes preview-decimated geometry.
"""

from __future__ import annotations

import bpy


def bake(scene, sess, start: int, end: int) -> str:
    """Bake a range, restoring only Blender frame/live and suppression state.

    The solver intentionally remains at the last state reached, including when
    a step fails.  This preserves the pre-extraction bake contract; this runner
    is not a solver transaction and does not promise rollback.
    """
    from . import playback, scene_bridge
    from ..io import cache

    start = int(start)
    end = int(end)
    if end < start:
        raise ValueError(f"Cache end frame {end} is before start frame {start}")

    fs = scene.fluidsim
    base = bpy.path.abspath(fs.cache_path)
    original_frame = scene.frame_current
    original_live = fs.live
    frame_set = getattr(scene, "frame_set", None)
    settings_records = {}

    # live=False prevents preview authority while the explicit suppression
    # context makes the runner independent even if frame_set delivers nested
    # frame-change callbacks synchronously.
    fs.live = False
    try:
        sess.initialize(scene)
        sess.start_frame = start
        with playback.suppress_frame_handlers():
            for timeline_frame in range(start, end + 1):
                if frame_set is not None:
                    frame_set(timeline_frame)
                sess.solver.frame = timeline_frame - 1
                sess.sync_scene(scene)
                scene_bridge.step_frame(
                    scene, sess, write_cache=False, publish_outputs=False)
                sess.solver.frame = timeline_frame
                cache.write_frame(scene, sess, timeline_frame)
                record_settings = getattr(cache, "settings_record", None)
                if callable(record_settings):
                    settings_records[timeline_frame] = record_settings(
                        sess.solver.settings)
        sess.cache_settings_records = dict(settings_records)
        cache.write_sequence_metadata(
            base, start, end, source="simulation",
            settings_records=settings_records,
            scene_key_value=sess.scene_key)
    finally:
        if frame_set is not None:
            try:
                with playback.suppress_frame_handlers():
                    frame_set(original_frame)
            except Exception:  # noqa: BLE001 - preserve original failure
                pass
        fs.live = original_live
    return base
