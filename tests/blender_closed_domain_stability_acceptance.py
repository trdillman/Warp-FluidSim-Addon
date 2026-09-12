"""CUDA regression for closed-domain drift and artificial energy growth."""

from __future__ import annotations

import addon_utils
import json
from pathlib import Path
import sys

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402


def check(label, condition, detail=""):
    if not condition:
        raise RuntimeError(f"{label}: {detail}")


def particle_metrics(solver):
    count = solver.particles.refresh_host_count()
    position = solver.particles.position[:count].numpy()
    velocity = solver.particles.velocity[:count].numpy()
    return {
        "count": int(count),
        "com_xy": np.mean(position[:, :2], axis=0),
        "rms_speed": float(np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))),
    }


def main():
    enable_repo_addon(str(ROOT), default_set=True, persistent=False)
    blend = ROOT / "demos" / "crown_splash.blend"
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    enable_repo_addon(str(ROOT), default_set=True, persistent=False)
    from fluidSim_warp_blender51.blender.scene_bridge import (
        active_domain_settings,
        session,
    )

    scene = bpy.context.scene
    domain = active_domain_settings(scene)
    fs = scene.fluidsim

    module = sys.modules.get("fluidSim_warp_blender51")
    loaded_path = Path(getattr(module, "__file__", "")).resolve()
    expected_root = (ROOT / "fluidSim_warp_blender51").resolve()
    check("add-on enabled", addon_utils.check("fluidSim_warp_blender51")[1])
    check(
        "add-on preferences enabled",
        "fluidSim_warp_blender51" in bpy.context.preferences.addons,
    )
    check("intended module root", expected_root in loaded_path.parents, loaded_path)

    fs.live = False
    scene.frame_set(scene.frame_start)

    sess = session()
    if sess.active:
        sess.release(scene)
    check("initialize", bpy.ops.fluidsim.initialize() == {"FINISHED"}, fs.status)
    check("closed scene", not domain.open_boundaries)
    check("saved wall policy", domain.domain_sdf_walls)
    check("runtime wall policy", sess.solver.settings.domain_sdf_walls)

    initial = particle_metrics(sess.solver)
    check("seeded particles", initial["count"] > 0, initial["count"])
    check("advance 48 frames", bpy.ops.fluidsim.step(frames=48) == {"FINISHED"})
    final = particle_metrics(sess.solver)

    drift = final["com_xy"] - initial["com_xy"]
    drift_norm = float(np.linalg.norm(drift))
    report = {
        "particle_count_initial": initial["count"],
        "particle_count_final": final["count"],
        "com_drift_xy": [float(value) for value in drift],
        "com_drift_norm": drift_norm,
        "rms_speed_initial": initial["rms_speed"],
        "rms_speed_final": final["rms_speed"],
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    check("particle count stable", final["count"] == initial["count"], report)
    check("horizontal center-of-mass stable", drift_norm < 0.005, report)
    check(
        "no artificial energy growth",
        final["rms_speed"] <= initial["rms_speed"] * 1.05,
        report,
    )
    sess.release(scene)
    print("CLOSED-DOMAIN-STABILITY-ACCEPTANCE-PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CLOSED-DOMAIN-STABILITY-ACCEPTANCE-FAIL: {exc!r}", flush=True)
        raise
