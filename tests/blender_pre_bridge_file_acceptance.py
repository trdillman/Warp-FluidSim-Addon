"""Headless .blend round trip: 2.0 flags files saved before FluidSim 1.1.

Run:  blender --background --factory-startup --python tests/blender_pre_bridge_file_acceptance.py

Before 1.1 the domain settings lived on the scene group; 1.1 moved them onto
the domain object and stamped ``schema_version = 1``. 2.0 reads only the
domain object. Three fixtures are saved and reopened:

* pre-1.1: domain settings stored on the scene group, no stamp -> flagged;
* 1.1: the same stale scene copy but stamped 1 -> not flagged;
* 2.0: a clean scene -> not flagged.

In every case the solver must read the domain object's value, never the stale
scene copy.
"""

import os
import sys
import tempfile

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon

FAILURES = []

STALE_VOXEL_SIZE = 0.037
DOMAIN_VOXEL_SIZE = 0.052


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def author(stamp):
    """Rebuild the scene with a designated domain; ``stamp`` shapes the file.

    ``None`` writes a clean 2.0 scene. Otherwise the scene group also stores a
    raw ``voxel_size`` as the older add-on did, with ``schema_version = stamp``.
    """
    from fluidSim_warp_blender51.blender.menus import create_domain_object

    scene = bpy.context.scene
    scene.fluidsim.domain_object = None
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    domain = create_domain_object(scene)
    domain.fluidsim_domain.voxel_size = DOMAIN_VOXEL_SIZE
    stored = scene.bl_system_properties_get(do_create=True)["fluidsim"]
    for key in ("voxel_size", "schema_version"):
        if key in stored:
            del stored[key]
    if stamp is not None:
        stored["voxel_size"] = STALE_VOXEL_SIZE
        stored["schema_version"] = stamp
    return scene


def run_fixture(label, stamp, expect_flagged, tmpdir):
    path = os.path.join(tmpdir, f"{label}.blend")
    author(stamp)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)

    from fluidSim_warp_blender51.blender import file_compat, scene_bridge

    scene = bpy.context.scene
    flagged = file_compat.is_pre_bridge_scene(scene)
    check(f"{label}: flagged == {expect_flagged}", flagged == expect_flagged, flagged)
    voxel = scene_bridge.active_domain_settings(scene).voxel_size
    check(f"{label}: solver reads the domain object's voxel_size",
          abs(voxel - DOMAIN_VOXEL_SIZE) < 1e-6, voxel)
    check(f"{label}: scene group exposes no domain setting",
          not hasattr(scene.fluidsim, "voxel_size"))


def main():
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    from fluidSim_warp_blender51.blender import file_compat

    check("load_post check registered",
          file_compat._on_load_post in bpy.app.handlers.load_post)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_fixture("pre-1.1", 0, True, tmpdir)
        run_fixture("1.1", 1, False, tmpdir)
        run_fixture("2.0", None, False, tmpdir)
        # Blender holds the last file open; leave a fresh one before cleanup.
        bpy.ops.wm.read_factory_settings(use_empty=True)

    if FAILURES:
        print(f"PRE-BRIDGE-FILE-ACCEPTANCE-FAIL {FAILURES}")
        sys.exit(1)
    print("PRE-BRIDGE-FILE-ACCEPTANCE-PASS")


main()
