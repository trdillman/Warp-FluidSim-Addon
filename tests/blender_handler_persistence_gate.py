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
    "render_complete",
    "render_cancel",
)

# Every handler the add-on is supposed to register, by list. Checking only
# "whatever is registered is @persistent" passes vacuously when a handler is
# missing entirely -- which is exactly how a silently-stripped render_init
# handler got through. Absence is the failure this catches.
EXPECTED_HANDLERS = {
    "frame_change_post": {"_on_frame_change"},
    "load_pre": {"_on_load_pre"},
    "animation_playback_pre": {"_on_playback_pre"},
    "depsgraph_update_post": {"_on_interaction_state_update"},
    "undo_post": {"_on_interaction_state_update"},
    "render_init": {"_on_render_init"},
    "render_complete": {"_on_render_end"},
    "render_cancel": {"_on_render_end"},
}
EXPECTED_HANDLER_MODULES = {
    "frame_change_post": {
        "_on_frame_change": "fluidSim_warp_blender51.blender.playback",
    },
    "animation_playback_pre": {
        "_on_playback_pre": "fluidSim_warp_blender51.blender.playback",
    },
    "depsgraph_update_post": {
        "_on_interaction_state_update":
            "fluidSim_warp_blender51.blender.pending_settings",
    },
    "undo_post": {
        "_on_interaction_state_update":
            "fluidSim_warp_blender51.blender.pending_settings",
    },
    "load_pre": {
        "_on_load_pre": "fluidSim_warp_blender51.blender.pending_settings",
    },
    "render_init": {
        "_on_render_init": "fluidSim_warp_blender51.blender.render",
    },
}


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


def assert_expected_handlers_registered(stage):
    """Every handler in EXPECTED_HANDLERS must actually be on its list."""
    found = {name: {getattr(fn, "__name__", "?") for fn in fns}
             for name, fns in addon_handlers().items()}
    missing = {
        list_name: sorted(names - found.get(list_name, set()))
        for list_name, names in EXPECTED_HANDLERS.items()
        if names - found.get(list_name, set())
    }
    if missing:
        raise RuntimeError(
            f"[{stage}] add-on handlers missing: {missing}; "
            f"registered: { {k: sorted(v) for k, v in found.items()} }")
    wrong_owner = []
    for list_name, callbacks in EXPECTED_HANDLER_MODULES.items():
        for callback_name, expected_module in callbacks.items():
            matches = [
                fn for fn in addon_handlers().get(list_name, ())
                if getattr(fn, "__name__", "") == callback_name
            ]
            if any(getattr(fn, "__module__", "") != expected_module
                   for fn in matches):
                wrong_owner.append(
                    f"{list_name}.{callback_name} expected {expected_module}")
    if wrong_owner:
        raise RuntimeError(
            f"[{stage}] add-on handler ownership mismatch: {wrong_owner}")


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
    assert_expected_handlers_registered("after register")
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
    assert_expected_handlers_registered("after reopen")
    assert_all_addon_handlers_persistent("after reopen")
except Exception as exc:
    print(f"HANDLER-PERSISTENCE-GATE-FAIL: {exc}")
    sys.exit(1)

print("HANDLER-PERSISTENCE-GATE-PASS")
