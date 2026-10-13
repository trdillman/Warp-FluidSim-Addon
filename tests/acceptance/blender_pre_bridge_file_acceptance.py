"""Headless .blend round trip: 2.0 flags files saved before FluidSim 1.1.

Run:  blender --background --factory-startup --python tests/acceptance/blender_pre_bridge_file_acceptance.py

Before 1.1 the domain settings lived on the scene group; 1.1 moved them onto
the domain object and stamped ``schema_version = 1``. 2.0 reads only the
domain object. Fixtures are saved and reopened:

* pre-1.1: domain settings stored on the scene group, no stamp -> flagged;
* 1.1: the same stale scene copy but stamped 1 -> not flagged;
* 2.0: a clean scene -> not flagged;
* 1.0 snippets: a stored snippet list 1.1 never copied -> flagged, and
  admission refuses the scene;
* 1.1 snippets: the same list after 1.1 marked the effect stack migrated ->
  not flagged.

In every case the solver must read the domain object's value, never the stale
scene copy. The stored snippet list survives a 2.0 save untouched, so 1.1 can
still recover it.
"""

import os
import sys
import tempfile

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

FAILURES = []

STALE_VOXEL_SIZE = 0.037
DOMAIN_VOXEL_SIZE = 0.052
SNIPPET_KERNEL = "legacy_probe"


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def author(stamp, snippets=None):
    """Rebuild the scene with a designated domain; ``stamp`` shapes the file.

    ``None`` writes a clean 2.0 scene. Otherwise the scene group also stores a
    raw ``voxel_size`` as the older add-on did, with ``schema_version = stamp``.
    ``snippets`` stores a raw 1.0 snippet list: ``"1.0"`` unmigrated, ``"1.1"``
    with the effect stack marked migrated as 1.1 left it.
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
    if "snippets" in stored:
        del stored["snippets"]
    effects = scene.bl_system_properties_get(do_create=True).get("fluidsim_effects")
    if effects is not None and "migration_complete" in effects:
        del effects["migration_complete"]
    if snippets is not None:
        stored["snippets"] = [{"kernel_name": SNIPPET_KERNEL, "enabled": 1}]
        if snippets == "1.1":
            scene.fluidsim_effects.artist_mode = True  # materialize the stored group
            scene.bl_system_properties_get()["fluidsim_effects"]["migration_complete"] = 1
    return scene


def run_fixture(label, stamp, expect_flagged, tmpdir, snippets=None):
    path = os.path.join(tmpdir, f"{label}.blend")
    author(stamp, snippets)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)

    from fluidSim_warp_blender51.blender import file_compat, scene_bridge

    scene = bpy.context.scene
    flagged = file_compat.is_pre_bridge_scene(scene)
    check(f"{label}: flagged == {expect_flagged}", flagged == expect_flagged, flagged)
    voxel = scene_bridge.active_domain_settings(scene).voxel_size
    check(f"{label}: solver reads the domain object's voxel_size",
          abs(voxel - DOMAIN_VOXEL_SIZE) < 1e-3, voxel)  # snapped to whole divisions
    check(f"{label}: scene group exposes no domain setting",
          not hasattr(scene.fluidsim, "voxel_size"))
    if snippets is None:
        return
    check(f"{label}: scene group exposes no snippet list",
          not hasattr(scene.fluidsim, "snippets"))
    from fluidSim_warp_blender51.blender.effects import build_effect_stack_for_admission

    try:
        build_effect_stack_for_admission(scene)
        refused = ""
    except RuntimeError as exc:
        refused = str(exc)
    check(f"{label}: admission refuses unmigrated snippets == {expect_flagged}",
          bool(refused) == expect_flagged, refused)
    # Re-save in 2.0: the stored list must survive for recovery in 1.1.
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    stored = bpy.context.scene.bl_system_properties_get()["fluidsim"]
    check(f"{label}: stored snippets survive a 2.0 save",
          stored["snippets"][0]["kernel_name"] == SNIPPET_KERNEL)


def main():
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    from fluidSim_warp_blender51.blender import file_compat

    check("load_post check registered",
          file_compat._on_load_post in bpy.app.handlers.load_post)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_fixture("pre-1.1", 0, True, tmpdir)
        run_fixture("1.1", 1, False, tmpdir)
        run_fixture("2.0", None, False, tmpdir)
        run_fixture("1.0 snippets", 1, True, tmpdir, snippets="1.0")
        run_fixture("1.1 snippets", 1, False, tmpdir, snippets="1.1")
        # Blender holds the last file open; leave a fresh one before cleanup.
        bpy.ops.wm.read_factory_settings(use_empty=True)

    if FAILURES:
        print(f"PRE-BRIDGE-FILE-ACCEPTANCE-FAIL {FAILURES}")
        sys.exit(1)
    print("PRE-BRIDGE-FILE-ACCEPTANCE-PASS")


main()
