"""Acceptance: Quick Setup preserves an existing complete setup."""

from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fluidSim_warp_blender51 as addon  # noqa: E402


def snapshot(scene):
    return tuple(sorted((obj.name, obj.type) for obj in scene.objects))


def main():
    addon.register()
    scene = bpy.context.scene
    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    first = snapshot(scene)
    assert scene.fluidsim.domain_object is not None
    assert any(obj.fluidsim.role == "SOURCE" for obj in scene.objects)
    assert any(obj.fluidsim.role == "COLLIDER" for obj in scene.objects)

    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    assert snapshot(scene) == first

    addon.unregister()
    print("QUICK-SETUP-IDEMPOTENCE-PASS")


if __name__ == "__main__":
    main()
