"""Open the maelstrom basin demo and prepare it for live playback."""

from pathlib import Path

import addon_utils
import bpy


REPO = Path(__file__).resolve().parents[1]
BLEND = REPO / "demos" / "maelstrom_basin.blend"


def main():
    addon_utils.enable("fluidSim_warp_blender51", default_set=False)
    bpy.ops.wm.open_mainfile(filepath=str(BLEND))
    addon_utils.enable("fluidSim_warp_blender51", default_set=False)
    scene = bpy.context.scene
    if hasattr(scene, "fluidsim"):
        scene.fluidsim.live = True
    scene.frame_set(max(scene.frame_start, 1))
    print("MAELSTROM-BASIN-PLAYABLE-READY")


main()
