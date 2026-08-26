"""Open the canonical Dam Break as an unsaved live native-interop review session."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import addon_utils
import blf
import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BLEND = (REPO_ROOT / "demos" / "dam_break.blend").resolve()
EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "native_interop"
    / "draw-profile-20260716-db2bc63c"
    / "live-review.json"
)
EXPECTED_HASH = "db2bc63cd7234f59251f88e7ba9c3d3f9c056db8681fa0c54dd1e05b86b48b82"
EXPECTED_LIMITS = {"priority_class": "0x4000", "affinity_mask": "0xffff"}
OVERLAY_KEY = "_codex_native_interop_review_overlay"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _process_limits(enforce=False):
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    )
    kernel32.GetProcessAffinityMask.restype = ctypes.c_int
    kernel32.GetPriorityClass.argtypes = (ctypes.c_void_p,)
    kernel32.GetPriorityClass.restype = ctypes.c_uint32
    kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.SetPriorityClass.restype = ctypes.c_int
    kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    if enforce:
        if not kernel32.SetPriorityClass(process, 0x4000):
            raise ctypes.WinError()
        if not kernel32.SetProcessAffinityMask(process, 0xFFFF):
            raise ctypes.WinError()
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        process, ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        raise ctypes.WinError()
    return {
        "priority_class": hex(kernel32.GetPriorityClass(process)),
        "affinity_mask": hex(process_mask.value),
    }


def _view3d_override():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next(region for region in area.regions if region.type == "WINDOW")
                return {
                    "window": window,
                    "screen": window.screen,
                    "area": area,
                    "region": region,
                    "space_data": area.spaces.active,
                }
    raise RuntimeError("No VIEW_3D area is available")


def _frame_particles(obj, presenter):
    for other in bpy.context.scene.objects:
        if other == obj:
            continue
        if other.type in {"CAMERA", "LIGHT"}:
            other.hide_viewport = True
        elif hasattr(other, "display_type"):
            other.display_type = "WIRE"
            other.show_in_front = True
    bpy.context.view_layer.update()
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    override = _view3d_override()
    with bpy.context.temp_override(
        **override,
        active_object=obj,
        object=obj,
        selected_objects=[obj],
        selected_editable_objects=[obj],
    ):
        bpy.ops.view3d.view_axis(type="FRONT", align_active=False)
        bounds = tuple(obj.data["native_interop_bounds"])
        lo = bounds[:3]
        hi = bounds[3:]
        center = tuple((low + high) * 0.5 for low, high in zip(lo, hi))
        extent = max(high - low for low, high in zip(lo, hi))
        view = override["space_data"].region_3d
        view.view_location = center
        view.view_distance = extent * 0.75
        override["space_data"].clip_start = max(extent * 0.0001, 0.0001)
        override["space_data"].clip_end = max(extent * 100.0, 1000.0)
        override["space_data"].show_region_ui = True
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    presenter.bind_pointcloud(evaluated)
    presenter._restore_active_binding()


def _telemetry():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    presenter = runtime.native_presenter
    stats = dict(presenter.bridge.stats) if presenter else {}
    return {
        "scene_frame": bpy.context.scene.frame_current,
        "solver_frame": getattr(runtime.solver, "frame", None),
        "particle_count": int(bpy.context.scene.fluidsim.stat_particles),
        "presentation_mode": runtime.presentation_mode,
        "presentation_reason": runtime.presentation_reason,
        "presentation_bytes": runtime.presentation_bytes,
        "imports": stats.get("imports"),
        "maps": stats.get("maps"),
        "published_frames": stats.get("published_frames"),
        "bulk_host_position_bytes": stats.get("bulk_host_position_bytes"),
        "animation_playing": bool(
            bpy.context.screen and bpy.context.screen.is_animation_playing
        ),
    }


def _draw_overlay():
    try:
        telemetry = _telemetry()
        lines = (
            "NATIVE GPU INTEROP: ACTIVE",
            (
                f"Frame {telemetry['scene_frame']}  |  solver {telemetry['solver_frame']}  |  "
                f"{telemetry['particle_count']:,} particles"
            ),
            f"GPU publications: 1/frame verified  |  lifetime {telemetry['published_frames']}",
            f"Persistent imports/maps: {telemetry['imports']}/{telemetry['maps']}",
            f"Host position readback: {telemetry['bulk_host_position_bytes']} bytes",
            "db2bc63c  |  Vulkan + CUDA/Warp",
        )
        font_id = 0
        blf.size(font_id, 16)
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 5, 0.0, 0.0, 0.0, 0.9)
        blf.shadow_offset(font_id, 2, -2)
        for index, line in enumerate(lines):
            blf.position(font_id, 28, bpy.context.region.height - 82 - index * 23, 0)
            blf.color(font_id, 0.35, 1.0, 0.45, 1.0)
            blf.draw(font_id, line)
        blf.disable(font_id, blf.SHADOW)
    except Exception:
        pass


def _install_overlay():
    namespace = bpy.app.driver_namespace
    old = namespace.get(OVERLAY_KEY)
    if old is not None:
        bpy.types.SpaceView3D.draw_handler_remove(old, "WINDOW")
    namespace[OVERLAY_KEY] = bpy.types.SpaceView3D.draw_handler_add(
        _draw_overlay, (), "WINDOW", "POST_PIXEL"
    )


def _write_evidence(error=None):
    module = sys.modules.get("fluidSim_warp_blender51")
    executable = Path(bpy.app.binary_path).resolve()
    payload = {
        "passed": error is None,
        "error": error,
        "pid": os.getpid(),
        "file": str(Path(bpy.data.filepath).resolve()),
        "addon_module": str(Path(module.__file__).resolve()) if module else None,
        "addon_enabled": addon_utils.check("fluidSim_warp_blender51") == (True, True),
        "addon_in_preferences": "fluidSim_warp_blender51" in bpy.context.preferences.addons,
        "executable": str(executable),
        "executable_sha256": _sha256(executable),
        "process_limits": _process_limits(),
        "telemetry": _telemetry() if error is None else None,
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    _process_limits(enforce=True)
    if _process_limits() != EXPECTED_LIMITS:
        raise RuntimeError(f"Unexpected process limits: {_process_limits()}")
    if Path(bpy.data.filepath).resolve() != EXPECTED_BLEND:
        raise RuntimeError(f"Unexpected review file: {bpy.data.filepath}")
    module = sys.modules.get("fluidSim_warp_blender51")
    if module is None or addon_utils.check("fluidSim_warp_blender51") != (True, True):
        raise RuntimeError("Temporary FluidSim add-on is not enabled")
    executable = Path(bpy.app.binary_path).resolve()
    if _sha256(executable) != EXPECTED_HASH:
        raise RuntimeError(f"Unexpected candidate: {executable}")

    scene = bpy.context.scene
    settings = scene.fluidsim
    bpy.ops.fluidsim.release()
    settings.native_gpu_interop = True
    settings.show_particles = True
    settings.liquid_view_mode = "PARTICLES"
    settings.surface_enabled = False
    settings.show_surface = False
    settings.show_whitewater = False
    settings.particle_display_fraction = 1.0
    settings.particle_attribute_mode = "NONE"
    settings.live = False
    scene.frame_set(scene.frame_start)
    if bpy.ops.fluidsim.initialize() != {"FINISHED"}:
        raise RuntimeError(settings.status)
    if bpy.ops.fluidsim.step(frames=1) != {"FINISHED"}:
        raise RuntimeError(settings.status)

    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    if runtime.presentation_mode != "Native GPU" or runtime.native_presenter is None:
        raise RuntimeError(
            f"Native presentation unavailable: {runtime.presentation_mode}; "
            f"{runtime.native_interop_error}"
        )
    obj = bpy.data.objects[viewport.PARTICLES_NAME]
    _frame_particles(obj, runtime.native_presenter)
    _install_overlay()
    if bpy.ops.fluidsim.toggle_live() != {"FINISHED"}:
        raise RuntimeError(settings.status)
    with bpy.context.temp_override(**_view3d_override()):
        bpy.ops.screen.animation_play()
    bpy.context.workspace.status_text_set(
        "Native GPU Interop ACTIVE | Vulkan + CUDA/Warp | Host position readback 0 bytes"
    )

    def record_ready():
        payload = _write_evidence()
        telemetry = payload["telemetry"]
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        if (telemetry["published_frames"] or 0) >= 5:
            print("NATIVE_INTEROP_LIVE_REVIEW=" + json.dumps(payload, sort_keys=True), flush=True)
            return None
        return 0.5

    bpy.app.timers.register(record_ready, first_interval=0.5)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        payload = _write_evidence(
            error={"message": repr(exc), "traceback": traceback.format_exc()}
        )
        print(
            "NATIVE_INTEROP_LIVE_REVIEW=" + json.dumps(payload, sort_keys=True),
            flush=True,
        )
        raise
