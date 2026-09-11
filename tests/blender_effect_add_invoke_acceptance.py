"""Interactive Blender acceptance for the Custom Effects search-popup operator."""

from __future__ import annotations

import os
import sys
import traceback

import addon_utils
import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon


def _quit() -> None:
    addon_utils.disable("fluidSim_warp_blender51", default_set=False)
    bpy.ops.wm.quit_blender()


def _cancel_popup() -> None:
    bpy.context.window.event_simulate(type="ESC", value="PRESS")
    bpy.app.timers.register(_quit, first_interval=0.25)


def _probe() -> None:
    try:
        enable_repo_addon(REPO_ROOT, default_set=False, persistent=False)
        result = bpy.ops.fluidsim.effect_add("INVOKE_DEFAULT")
        if result != {"RUNNING_MODAL"}:
            raise AssertionError(f"unexpected Add Effect invoke result: {result}")
        print("CUSTOM-EFFECT-ADD-INVOKE-ACCEPTANCE-PASS")
    except Exception:  # noqa: BLE001 - emit the complete Blender-side failure
        traceback.print_exc()
        print("CUSTOM-EFFECT-ADD-INVOKE-ACCEPTANCE-FAIL")
    finally:
        bpy.app.timers.register(_cancel_popup, first_interval=0.25)


bpy.app.timers.register(_probe, first_interval=0.25)
