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


ADDON_PREFIX = "fluidSim_warp_blender51"

# Every app-handler list the add-on registers into. Checked as a whole, because
# filtering to a single handler is what let a missing @persistent on
# render._on_render_init survive: it was silently dropped on every file load.
HANDLER_LISTS = (
    "frame_change_post",
    "load_pre",
    "render_init",
    "render_pre",
    "render_post",
    "depsgraph_update_post",
    "undo_post",
    "save_pre",
    "load_post",
    "animation_playback_pre",
)


def addon_frame_handlers():
    return [
        fn for fn in bpy.app.handlers.frame_change_post
        if getattr(fn, "__module__", "").startswith(ADDON_PREFIX)
        and getattr(fn, "__name__", "") == "_on_frame_change"
    ]


def addon_handlers():
    """Map of handler-list name -> add-on callbacks currently registered."""
    found = {}
    for list_name in HANDLER_LISTS:
        handlers = getattr(bpy.app.handlers, list_name, None)
        if handlers is None:
            continue
        owned = [
            fn for fn in handlers
            if getattr(fn, "__module__", "").startswith(ADDON_PREFIX)
        ]
        if owned:
            found[list_name] = owned
    return found


def assert_all_addon_handlers_persistent(stage):
    """Every add-on app handler must carry @persistent."""
    offenders = [
        f"{list_name}.{getattr(fn, '__name__', fn)}"
        for list_name, fns in addon_handlers().items()
        for fn in fns
        if not hasattr(fn, "_bpy_persistent")
    ]
    if offenders:
        raise RuntimeError(
            f"[{stage}] add-on app handlers missing @persistent: {offenders}. "
            "Blender frees non-persistent handlers on file load."
        )


try:
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    before = addon_frame_handlers()
    if not before:
        raise RuntimeError("add-on frame handler was not registered")
    registered_before = {k: [getattr(f, "__name__", "?") for f in v]
                         for k, v in addon_handlers().items()}
    if "render_init" not in registered_before:
        raise RuntimeError(
            "render_init handler was not registered; "
            f"registered lists: {sorted(registered_before)}")
    assert_all_addon_handlers_persistent("after register")

    mesh = bpy.data.meshes.new("Persistence Seed")
    mesh.from_pydata(
        [(0.1, 0.1, 0.1), (0.2, 0.1, 0.1),
         (0.1, 0.2, 0.1), (0.1, 0.1, 0.2)],
        [],
        [],
    )
    seed = bpy.data.objects.new("Persistence Seed", mesh)
    bpy.context.scene.collection.objects.link(seed)
    seed.fluidsim.auto_inject_seed = True
    bpy.context.scene.fluidsim.domain_sdf_walls = True
    bpy.context.scene.fluidsim.live = True

    with tempfile.TemporaryDirectory(prefix="fluidsim-handler-gate-") as temp_dir:
        blend_path = os.path.join(temp_dir, "handler-persistence.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=False)
        bpy.ops.wm.open_mainfile(filepath=blend_path)

        scene = bpy.context.scene
        scene.frame_set(scene.frame_start + 1)

        from fluidSim_warp_blender51.blender.scene_bridge import session

        sess = session()
        if not sess.active:
            raise RuntimeError("solver did not initialize after reopening")
        if not str(sess.solver.device).startswith("cuda"):
            raise RuntimeError(f"solver is not CUDA-backed: {sess.solver.device}")
        if not sess.solver.settings.domain_sdf_walls:
            raise RuntimeError("closed domain collision walls were not restored")
        count = sess.solver.particles.refresh_host_count()
        if count != 4:
            raise RuntimeError(f"reopened seed count is {count}, expected 4")

        for _ in range(2):
            scene.frame_set(scene.frame_start)
            count = sess.solver.particles.refresh_host_count()
            if count != 4:
                raise RuntimeError(
                    f"reset seed count is {count}, expected exactly 4"
                )
            scene.frame_set(scene.frame_start + 1)

    after = addon_frame_handlers()
    if not after:
        raise RuntimeError("add-on frame handler was cleared by opening a .blend file")

    # Every handler list populated before the reopen must still be populated.
    registered_after = {k: [getattr(f, "__name__", "?") for f in v]
                        for k, v in addon_handlers().items()}
    lost = {
        name: names for name, names in registered_before.items()
        if not registered_after.get(name)
    }
    if lost:
        raise RuntimeError(
            f"add-on handlers cleared by opening a .blend file: {lost}; "
            f"surviving: {registered_after}")
    assert_all_addon_handlers_persistent("after reopen")
except Exception as exc:
    print(f"HANDLER-PERSISTENCE-GATE-FAIL: {exc}")
    sys.exit(1)

print("HANDLER-PERSISTENCE-GATE-PASS")
