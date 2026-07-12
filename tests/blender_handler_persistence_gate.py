"""Regression gate for timeline playback after opening a .blend file.

Run: blender --background --factory-startup --python tests/blender_handler_persistence_gate.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon


def addon_frame_handlers():
    return [
        fn for fn in bpy.app.handlers.frame_change_post
        if getattr(fn, "__module__", "").startswith("fluidSim_warp_blender51")
        and getattr(fn, "__name__", "") == "_on_frame_change"
    ]


try:
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    before = addon_frame_handlers()
    if not before:
        raise RuntimeError("add-on frame handler was not registered")

    with tempfile.TemporaryDirectory(prefix="fluidsim-handler-gate-") as temp_dir:
        blend_path = os.path.join(temp_dir, "handler-persistence.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=False)
        bpy.ops.wm.open_mainfile(filepath=blend_path)

    after = addon_frame_handlers()
    if not after:
        raise RuntimeError("add-on frame handler was cleared by opening a .blend file")
except Exception as exc:
    print(f"HANDLER-PERSISTENCE-GATE-FAIL: {exc}")
    sys.exit(1)

print("HANDLER-PERSISTENCE-GATE-PASS")
