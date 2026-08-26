"""Verify the species toggle round-trip, then leave the GUI review scene open."""

from __future__ import annotations

import json
import os
from pathlib import Path
import traceback

import bpy


RESULT_PATH = Path(os.environ["FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT"]).resolve()
RESULT = {
    "schema": "fluidsim-species-toggle-visible-review-v1",
    "ok": False,
    "checks": [],
    "errors": [],
}


def check(name, condition, detail=""):
    passed = bool(condition)
    RESULT["checks"].append({
        "name": name, "passed": passed, "detail": str(detail)})
    if not passed:
        RESULT["errors"].append(f"{name}: {detail or 'check failed'}")
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)


try:
    import addon_utils
    import fluidSim_warp_blender51 as addon
    from fluidSim_warp_blender51.blender import scene_bridge

    scene = bpy.context.scene
    fs = scene.fluidsim
    check("temporary addon active", addon_utils.check(addon.__name__)[1])
    check("chromatic demo loaded", Path(bpy.data.filepath).name == "chromatic_vortex.blend")
    check("authored Clear Dye modes", fs.particle_attribute_mode == "CLEAR_DYE"
          and fs.surface_attribute_mode == "CLEAR_DYE")

    sess = scene_bridge.initialize_session(scene)
    scene_bridge.step_frame(scene, sess, write_cache=False)
    enabled_count = int(sess.solver.particles.host_count)
    check("species run populated", enabled_count > 0
          and sess.solver.settings.species_enabled, enabled_count)

    fs.species_enabled = False
    scene_bridge.reset_simulation(scene)
    scene_bridge.step_frame(scene, sess, write_cache=False)
    disabled_count = int(sess.solver.particles.host_count)
    check("disabled run resumed", disabled_count > 0
          and not sess.solver.settings.species_enabled
          and not scene_bridge.current_error(scene),
          f"particles={disabled_count}; status={fs.status}")

    fs.species_enabled = True
    scene_bridge.reset_simulation(scene)
    scene_bridge.step_frame(scene, sess, write_cache=False)
    reenabled_count = int(sess.solver.particles.host_count)
    surface = bpy.data.objects.get("FluidSim Surface")
    check("re-enabled run restored color", reenabled_count > 0
          and sess.solver.settings.species_enabled
          and surface is not None
          and surface.data.attributes.get("fluidsim_color") is not None
          and not scene_bridge.current_error(scene), reenabled_count)

    fs.live = False
    fs.status = (
        "Species toggle verified: disable/resume/re-enable passed — review ready")
    scene["fluidsim_species_toggle_verified"] = True
    RESULT.update({
        "enabled_particles": enabled_count,
        "disabled_particles": disabled_count,
        "reenabled_particles": reenabled_count,
    })
except Exception as exc:
    RESULT["errors"].append(f"{type(exc).__name__}: {exc}")
    RESULT["traceback"] = traceback.format_exc()

RESULT["ok"] = not RESULT["errors"]
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(RESULT, indent=2), encoding="utf-8")
print(
    "SPECIES-TOGGLE-VISIBLE-REVIEW-PASS"
    if RESULT["ok"] else "SPECIES-TOGGLE-VISIBLE-REVIEW-FAIL",
    flush=True,
)
if not RESULT["ok"]:
    raise RuntimeError("species toggle visible review failed")
