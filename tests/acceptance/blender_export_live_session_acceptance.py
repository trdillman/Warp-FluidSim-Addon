"""Real Blender/CUDA refusal check; use the isolated production-gate environment."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

import bpy
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests/cache_io/fixtures")]


def progress(stage):
    print(stage, flush=True)
    path = ROOT / "artifacts" / "export-live-progress.txt"
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(stage + "\n")


def main():
    from tools.repo_addon_loader import enable_repo_addon
    enable_repo_addon(str(ROOT), default_set=True, persistent=False)
    from tiny_cache import build_tiny_cache
    from fluidSim_warp_blender51.blender import (
        cache_identity, interchange, playback, scene_bridge)
    from fluidSim_warp_blender51.version import CERTIFIED_WARP_VERSION
    assert wp.__version__ == CERTIFIED_WARP_VERSION
    scene = bpy.context.scene
    bpy.ops.fluidsim.setup_scene()
    fs = scene.fluidsim
    fs.live = False
    progress("setup complete")
    settings = scene_bridge.active_domain_settings(scene)
    settings.voxel_size = 0.15
    settings.leaf_capacity = 256
    settings.max_particles = 80000
    settings.max_whitewater = 1000
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    fs.live = False
    fs.show_surface = False
    fs.show_particles = True
    progress("settings complete")
    bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0.4, 0.0, 0.6))
    cork = bpy.context.object
    cork.fluidsim.role = "COLLIDER"
    cork.fluidsim.enabled = True
    cork.fluidsim.collider_dynamic = True
    cork.fluidsim.collider_density = 300.0

    progress("collider complete")
    with tempfile.TemporaryDirectory(prefix="export-live-") as work:
        base = build_tiny_cache(Path(work) / "cache", frame_count=2,
                               settings_signature=cache_identity.settings_signature(scene))
        fs.cache_path = str(base)
        progress("cache created")
        playback.play_cache_frame(scene_bridge.session(), scene, 1)
        progress("cache published")
        plans = (interchange.plan_export(scene), interchange.plan_usd_export(scene))
        progress("plans created; initializing")
        assert bpy.ops.fluidsim.initialize() == {"FINISHED"}
        progress("solver initialized")
        sess = scene_bridge.session()
        solver = sess.solver
        assert sess.owns_scene(scene) and str(solver.device).startswith("cuda")
        scene_bridge.step_frame(scene, sess, write_cache=False)
        progress("solver stepped")
        count = solver.particles.refresh_host_count()
        assert count > 0 and solver.rigid.bodies

        def digest_cache():
            return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in base.rglob("*") if p.is_file()}

        def snapshot():
            wp.synchronize_device(solver.device)
            return (solver.particles.position.numpy().copy(),
                    solver.rigid.export_state(),
                    (sess.scene_identity, sess.scene_key, sess.last_frame),
                    digest_cache())

        before = snapshot()
        for live in (False, True, False):
            fs.live = live
            state = (scene.frame_current, scene.frame_start, scene.frame_end,
                     bool(fs.live), bool(fs.cache_playback_enabled),
                     tuple((o.name, o.select_get(), o.hide_get(), o.hide_viewport)
                           for o in scene.objects))
            for fmt, plan in zip(("alembic", "usd"), plans):
                for _ in range(2):
                    try:
                        getattr(interchange, "export_" + fmt)(scene, str(Path(work) / fmt), plan)
                    except interchange.ExportRefused as exc:
                        assert "live solver" in str(exc)
                    else:
                        raise AssertionError("active export was not refused")
                    assert sess.solver is solver and sess.owns_scene(scene)
                    after = snapshot()
                    np.testing.assert_array_equal(before[0], after[0])
                    assert before[1][0] == after[1][0]
                    np.testing.assert_array_equal(before[1][1], after[1][1])
                    assert before[2:] == after[2:]
                    assert state == (
                        scene.frame_current, scene.frame_start, scene.frame_end,
                        bool(fs.live), bool(fs.cache_playback_enabled),
                        tuple((o.name, o.select_get(), o.hide_get(), o.hide_viewport)
                              for o in scene.objects))
        fs.live = True
        frame = solver.frame
        scene_bridge.step_frame(scene, sess, write_cache=False)
        assert sess.solver is solver and solver.frame > frame
        assert digest_cache() == before[3]
        print(json.dumps({"blender": bpy.app.version_string, "warp": wp.__version__,
                          "device": str(solver.device), "particles": count,
                          "rigid_bodies": len(solver.rigid.bodies),
                          "background": bpy.app.background, "source": interchange.__file__}))
        print("EXPORT_LIVE_SESSION_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if not bpy.app.background:
            def quit_after_startup():
                bpy.ops.wm.quit_blender()
                return None

            bpy.app.timers.register(quit_after_startup, first_interval=1.0)
