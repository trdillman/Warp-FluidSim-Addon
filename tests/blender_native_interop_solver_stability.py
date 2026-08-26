"""One-thousand-frame real FluidSim/Warp native presentation gate."""

import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools.repo_addon_loader import enable_repo_addon
from blender_native_interop_stress import process_sample


FRAME_COUNT = 1_000
WARMUP_FRAMES = 100


def main():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    scene = bpy.context.scene
    settings = scene.fluidsim
    settings.voxel_size = 0.2
    settings.max_particles = 50_000
    settings.max_whitewater = 5_000
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    settings.liquid_view_mode = "PARTICLES"
    settings.particle_display_fraction = 1.0
    settings.particle_attribute_mode = "NONE"
    settings.native_gpu_interop = True
    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, settings.status
    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    samples = []
    baseline = None
    started = perf_counter()
    for frame in range(1, FRAME_COUNT + 1):
        runtime.solver.step_frame()
        viewport.update_outputs(scene, runtime)
        assert runtime.native_presenter is not None, runtime.native_interop_error
        if frame <= WARMUP_FRAMES or frame % 100 == 0:
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        if frame == WARMUP_FRAMES or frame % 100 == 0:
            sample = {
                "frame": frame,
                "particles": runtime.solver.particles.host_count,
                **process_sample(),
            }
            samples.append(sample)
            if baseline is None:
                baseline = sample
    elapsed = perf_counter() - started
    stats = dict(runtime.native_presenter.bridge.stats)
    bpy.ops.fluidsim.release()
    particle_object = bpy.data.objects.get("FluidSim Particles")
    if particle_object is not None:
        bpy.data.objects.remove(particle_object, do_unlink=True)
    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=3)
    gc.collect()
    final = process_sample()
    assert stats["imports"] == 6
    assert stats["maps"] == 2
    assert stats["published_frames"] >= FRAME_COUNT
    assert stats["bulk_host_position_bytes"] == 0
    assert final["handles"] <= baseline["handles"]
    assert final["private_bytes"] <= baseline["private_bytes"] * 1.05
    report = {
        "frames": FRAME_COUNT,
        "elapsed_seconds": elapsed,
        "baseline": baseline,
        "final": final,
        "samples": samples,
        "bridge_stats": stats,
    }
    artifact_dir = REPO_ROOT / "docs" / "evidence" / "native_interop"
    (artifact_dir / "solver-stability.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_SOLVER_STABILITY", json.dumps(report, sort_keys=True))

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


if __name__ == "__main__":
    main()
