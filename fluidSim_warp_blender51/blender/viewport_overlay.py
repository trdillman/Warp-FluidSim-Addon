"""Narrow Blender adapter for the diagnostic viewport overlay only.

This module owns draw-handler registration and presentation rendering. It must
not create, resize, remove, or publish persistent simulation output datablocks.
"""

from __future__ import annotations

import sys

from ..sim.constants import TIMER_PHASE_SLOTS

_OVERLAY_HANDLE = None
bpy = sys.modules.get("bpy")


def _require_bpy():
    global bpy
    if bpy is None:
        try:
            import bpy as blender_api
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Viewport overlay requires Blender's Python runtime"
            ) from exc
        bpy = blender_api
    return bpy


def _draw_overlay():
    _require_bpy()
    import blf

    context = bpy.context
    scene = context.scene
    fs = getattr(scene, "fluidsim", None)
    if fs is None or not fs.show_overlay:
        return
    from .scene_bridge import pending_presentation, session

    sess = session()
    pending = pending_presentation(scene)
    owns_scene = getattr(sess, "owns_scene", lambda _scene: sess.active)(scene)
    if not owns_scene and pending["action"] == "NONE":
        return

    font_id = 0
    dpi_scale = context.preferences.system.ui_scale
    size = int(12 * dpi_scale)
    blf.size(font_id, size)
    blf.color(font_id, 0.9, 0.95, 1.0, 1.0)
    x = 20 * dpi_scale
    y = 60 * dpi_scale
    lh = size * 1.4

    lines = [f"FluidSim Warp  |  {fs.status}"]
    if owns_scene:
        lines.append(
            f"particles {fs.stat_particles:,}   whitewater {fs.stat_whitewater:,}   "
            f"voxels {fs.stat_voxels:,}   substeps {fs.stat_substeps}   "
            f"{fs.stat_step_ms:.1f} ms/frame")
    if pending["action"] != "NONE":
        label = ("REINITIALIZE REQUIRED" if pending["action"] == "REINIT_PENDING"
                 else "RESTART REQUIRED")
        lines.append(f"{label}: {pending['count']} pending change(s)")
        if pending["reasons"]:
            lines.append("pending: " + "; ".join(pending["reasons"][:2]))
        if pending["blocked_reason"]:
            lines.append("blocked: " + pending["blocked_reason"][:120])
    if owns_scene and sess.profiler is not None:
        summary = sess.profiler.summary(last_n=30)
        phase_names = {
            f"compute_{slot}": phase
            for phase, slot in TIMER_PHASE_SLOTS.items()
        }
        parts = []
        for slot, label in phase_names.items():
            value = summary.get(slot, 0.0)
            if value > 0.05:
                parts.append(f"{label} {value:.1f}")
        if parts:
            lines.append("timers(ms): " + "  ".join(parts))
    if owns_scene and sess.error:
        lines.append(f"error: {sess.error}")

    for index, line in enumerate(reversed(lines)):
        blf.position(font_id, x, y + index * lh, 0)
        blf.draw(font_id, line)


def register_overlay():
    global _OVERLAY_HANDLE
    _require_bpy()
    unregister_overlay()
    _OVERLAY_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        _draw_overlay, (), "WINDOW", "POST_PIXEL")


def unregister_overlay():
    global _OVERLAY_HANDLE
    if _OVERLAY_HANDLE is None:
        return
    _require_bpy()
    try:
        bpy.types.SpaceView3D.draw_handler_remove(_OVERLAY_HANDLE, "WINDOW")
    except Exception:
        pass
    _OVERLAY_HANDLE = None
