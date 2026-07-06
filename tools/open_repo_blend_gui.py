"""Visible Blender launcher probe for opening a saved blend with repo add-on.

Run with Blender as:
    blender --factory-startup <file.blend> --python tools/open_repo_blend_gui.py
"""

from __future__ import annotations

import os
from pathlib import Path

import addon_utils
import bpy


REPO = Path(__file__).resolve().parents[1]
ADDON_NAME = "fluidSim_warp_blender51"


def main():
    import sys

    sys.path.insert(0, str(REPO))
    from tools.repo_addon_loader import enable_repo_addon

    module = enable_repo_addon(REPO, default_set=True, persistent=False)
    expected = os.environ.get("FLUIDSIM_EXPECT_BLEND")
    if expected:
        expected_path = Path(expected).resolve()
        current = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
        if current != expected_path:
            bpy.ops.wm.open_mainfile(filepath=str(expected_path))
            module = enable_repo_addon(REPO, default_set=True, persistent=False)
    blend = Path(bpy.data.filepath).resolve()
    if expected and blend != Path(expected).resolve():
        raise RuntimeError(f"opened {blend}, expected {expected}")
    if not addon_utils.check(ADDON_NAME)[1]:
        raise RuntimeError(f"{ADDON_NAME} is not enabled")
    if ADDON_NAME not in bpy.context.preferences.addons:
        raise RuntimeError(f"{ADDON_NAME} missing from Blender preferences")

    scene = bpy.context.scene
    if hasattr(scene, "fluidsim"):
        scene.fluidsim.live = True
    scene.frame_set(max(scene.frame_start, 1))
    marker = os.environ.get("FLUIDSIM_OPEN_MARKER")
    message = (
        "FLUIDSIM-REPO-BLEND-OPEN "
        f"blend={blend} addon={Path(module.__file__).resolve()} "
        f"enabled={addon_utils.check(ADDON_NAME)[1]}"
    )
    if marker:
        Path(marker).write_text(message + "\n", encoding="utf-8")
    print(
        message,
        flush=True,
    )


main()
