"""Fresh Blender child for the saved-demo performance budget (not GUI proof)."""
import hashlib
import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from pathlib import Path


def translate_reference_settings(owner, payload, images, *, apply_values):
    """Translate ownership and export schema while preserving effective work."""
    if apply_values:
        for name, value in payload.items():
            if name not in owner.bl_rna.properties:
                if name == "species_enabled" and value:
                    raise RuntimeError("1.0.255 cannot compare a species-enabled workload")
                continue
            if isinstance(value, dict) and "image" in value:
                value = images[value["image"]]
            setattr(owner, name, value)
    if "export_attributes" not in owner.bl_rna.properties and not payload.get("export_attributes", True):
        for name in payload:
            if "_attribute_" in name and name in owner.bl_rna.properties:
                setattr(owner, name, name == "whitewater_attribute_type")


def comparable_profiler(profiler, phase_slots):
    """Keep phase timers; sum outer spans identically for legacy/current code."""
    original_track = profiler.track
    state = {"depth": 0, "outerMs": 0.0}
    @contextmanager
    def track(phase):
        slot = f"compute_{phase_slots[phase]}"
        before = profiler.frame.timers.get(slot, 0.0)
        outer = state["depth"] == 0
        state["depth"] += 1
        try:
            with original_track(phase):
                yield
        finally:
            state["depth"] -= 1
            if outer:
                state["outerMs"] += profiler.frame.timers.get(slot, 0.0) - before
    profiler.track = track
    return state


def main():
    import bpy
    import warp as wp

    argv = sys.argv[sys.argv.index("--") + 1:]
    extract = argv[0] == "--extract"
    if extract:
        _, root, blend, out = argv
    else:
        root, blend, out, frames, reference, payload_path = argv
    root, blend = Path(root).resolve(), Path(blend).resolve()
    inventory_spec = importlib.util.spec_from_file_location('budget_inventory', Path(__file__).with_name('benchmark_versions.py'))
    inventory_module = importlib.util.module_from_spec(inventory_spec)
    inventory_spec.loader.exec_module(inventory_module)
    root_inventory = inventory_module._inventory(root)
    sys.path.insert(0, str(root))
    # Use the selected checkout's loader; never an installed add-on.
    from tools.repo_addon_loader import enable_repo_addon
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    addon = enable_repo_addon(str(root), default_set=False, persistent=False)
    from fluidSim_warp_blender51 import version
    from fluidSim_warp_blender51.blender import scene_bridge

    scene = bpy.context.scene
    scene.fluidsim.live = False
    getter = getattr(scene_bridge, "active_domain_settings", None)
    owner = getter(scene) if getter else scene.fluidsim
    if extract:
        values = {}
        for prop in owner.bl_rna.properties:
            name = prop.identifier
            if name == "rna_type" or prop.is_readonly:
                continue
            value = getattr(owner, name)
            if prop.type == "POINTER":
                values[name] = {"image": value.name} if value is not None else None
            elif getattr(prop, "is_array", False):
                values[name] = list(value)
            elif prop.type != "COLLECTION":
                values[name] = value
        Path(out).write_text(json.dumps(values, sort_keys=True, indent=2), encoding="utf-8")
        print("DEMO-SETTINGS-EXTRACT-PASS", flush=True)
        return
    payload_bytes = Path(payload_path).read_bytes()
    payload = json.loads(payload_bytes)
    translate_reference_settings(owner, payload, bpy.data.images,
                                 apply_values=version.VERSION_STRING in ("1.0.255", "1.1.0"))
    owner.profiling_enabled = True
    owner.profiling_sync = True
    scene.frame_set(scene.frame_start)
    scene_bridge.release_session(scene)
    sess = scene_bridge.initialize_session(scene)
    if not sess or not sess.solver or sess.error:
        raise RuntimeError(f"demo initialization failed: {getattr(sess, 'error', None)}")
    profiler = sess.solver.profiler
    if profiler is None or not profiler.enabled or not profiler.sync:
        raise RuntimeError("synchronized stage profiler is required")
    from fluidSim_warp_blender51.sim.constants import TIMER_PHASE_SLOTS
    stage_state = comparable_profiler(profiler, TIMER_PHASE_SLOTS)
    device = str(sess.solver.settings.device)
    if not device.startswith("cuda"):
        raise RuntimeError(f"GPU comparison requires CUDA, got {device}")
    rows = []
    settings = sess.solver.settings
    protected = {name: getattr(settings, name) for name in (
        "voxel_size", "min_substeps", "max_substeps", "domain_min", "domain_max",
        "surface_enabled", "whitewater_enabled", "viscosity_enabled",
        "viscosity_constant", "surface_tension_enabled", "surface_tension_coefficient",
        "max_particles", "max_whitewater")}
    effective = json.loads(json.dumps(asdict(settings), default=lambda value: value.name
                                      if isinstance(value, Enum) else str(value)))
    from fluidSim_warp_blender51.sim.attributes import (
        OUTPUT_TOGGLE_NAMES,
        enabled_attribute_keys,
    )
    attributes = {kind: list(enabled_attribute_keys(settings, kind))
                  for kind in OUTPUT_TOGGLE_NAMES}
    # Compare resolved output contracts, not inactive authored toggles.
    effective.pop("export_attributes", None)
    for kind, toggles in OUTPUT_TOGGLE_NAMES.items():
        for key, name in toggles.items():
            if name in effective:
                effective[name] = key in attributes[kind]
    try:
        for frame in range(1, int(frames) + 1):
            scene.frame_set(scene.frame_start + frame - 1)
            wp.synchronize_device(device)
            start = time.perf_counter()
            stage_state["outerMs"] = 0.0
            sim_time_before = float(sess.solver.sim_time)
            scene_bridge.step_frame(scene, sess, write_cache=False,
                                    refresh_pending=False,
                                    timeline_frame=scene.frame_current,
                                    publish_outputs=True)
            wp.synchronize_device(device)
            if sess.error:
                raise RuntimeError(sess.error)
            stats = sess.solver.stats
            timers = dict(stats.timers)
            reported_solver_ms = timers["solver"]
            timers["solver"] = stage_state["outerMs"]
            rows.append({"frame": frame, "hostMs": (time.perf_counter() - start) * 1000,
                         "timers": timers, "reportedSolverMs": reported_solver_ms,
                         "frameDt": float(settings.time_scale / settings.frame_rate),
                         "simTimeBefore": sim_time_before,
                         "simTimeAfter": float(sess.solver.sim_time),
                         "substeps": int(stats.substeps),
                         "activeParticles": int(stats.active_particles)})
        if not rows[-1]["activeParticles"]:
            raise RuntimeError("demo produced no liquid particles")
        if inventory_module._inventory(root) != root_inventory:
            raise RuntimeError('selected source changed during the trial')
        report = {"root": str(root), "addon": addon.__file__,
                  "processId": os.getpid(), "rootInventorySha256": root_inventory,
                  "version": version.VERSION_STRING, "referenceVersion": reference,
                  "blend": str(blend), "blendSha256": hashlib.sha256(blend.read_bytes()).hexdigest(),
                  "device": device, "warp": wp.config.version, "profilerSync": True,
                  "protectedSettings": protected,
                  "effectiveSettings": effective,
                  "attributeInventory": attributes,
                  "settingsSha256": hashlib.sha256(payload_bytes).hexdigest(),
                  "computeProfile": profiler.compute_profile(),
                  "solverTimerContract": "sum_of_outermost_synchronized_phase_spans",
                  "records": rows, "measurementKind": "saved_demo_scene_bridge_with_stage_sync"}
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        scene_bridge.release_session(scene)
    print("DEMO-PERFORMANCE-TRIAL-PASS", flush=True)


if __name__ == "__main__":
    main()
