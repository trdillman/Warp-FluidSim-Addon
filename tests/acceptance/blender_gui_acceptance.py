"""Interactive Blender GUI acceptance test for the FluidSim Warp add-on.

Run with a real (non-background) Blender:
    blender --python tests/acceptance/blender_gui_acceptance.py

Uses a timer-driven state machine so the UI event loop keeps running:
enables the add-on, builds the demo scene, initializes the CUDA solver,
starts viewport animation playback with live simulation, verifies the solver
advances during real playback, checks rewind-reset behavior, forces UI
redraws of the sidebar panels, captures a viewport screenshot, renders a
frame, and writes a JSON verdict before quitting Blender.

Set FLUIDSIM_PANEL_DRAW_INJECT_FAILURE=1 to prove the panel draw smoke gate
fails on real Blender panel draw exceptions.

For an isolated source-loaded GUI, set FLUIDSIM_EXPECTED_ADDON_ROOT to the
expected package directory and FLUIDSIM_GUI_RESULT to the receipt path. This
mode verifies the loaded/enabled add-on and leaves Blender open.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import addon_utils
import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

ARTIFACTS = os.path.join(REPO_ROOT, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)
RESULT_PATH = os.environ.get(
    "FLUIDSIM_GUI_RESULT", os.path.join(ARTIFACTS, "phase3_gui_result.json"))
SCREENSHOT_PATH = os.path.join(ARTIFACTS, "phase3_gui_viewport.png")
RENDER_PATH = os.path.join(ARTIFACTS, "phase3_gui_render.png")
PANEL_DRAW_FAILURE_ENV = "FLUIDSIM_PANEL_DRAW_INJECT_FAILURE"
PANEL_DRAW_FAILURE_PANEL = "FLUIDSIM_PT_scene_control"
EXPECTED_PRELOADED_ADDON_ROOT = os.environ.get("FLUIDSIM_EXPECTED_ADDON_ROOT")
PANEL_CALL_SUCCESS = ({"FINISHED"}, {"RUNNING_MODAL"}, {"INTERFACE"})
PLAYBACK_MIN_FRAMES = 3
PLAYBACK_PROGRESS_TIMEOUT_SECONDS = 45.0

PANEL_IDS = [
    "FLUIDSIM_PT_viewport_status", "FLUIDSIM_PT_header_popover",
    "FLUIDSIM_STAGE_PT_setup", "FLUIDSIM_STAGE_PT_setup_advanced",
    "FLUIDSIM_STAGE_PT_simulate",
    "FLUIDSIM_STAGE_PT_look", "FLUIDSIM_STAGE_PT_look_materials",
    "FLUIDSIM_STAGE_PT_bake", "FLUIDSIM_STAGE_PT_bake_storage",
    "FLUIDSIM_STAGE_PT_bake_manage",
    "FLUIDSIM_PT_physics_root",
    "FLUIDSIM_PT_physics_surface_output",
    "FLUIDSIM_PT_scene_lifecycle",
    "FLUIDSIM_PT_scene_control", "FLUIDSIM_PT_scene_pending",
    "FLUIDSIM_PT_scene_setup",
    "FLUIDSIM_PT_scene_runtime_stats",
    # Domain, surface and whitewater controls are registered only as Physics
    # tab mirrors (ui_physics._make_panel_mirrors) since the 3.0 split.
    "FLUIDSIM_PT_physics_domain_scene_domain",
    "FLUIDSIM_PT_physics_domain_scene_simulation",
    "FLUIDSIM_PT_physics_domain_sim_time",
    "FLUIDSIM_PT_physics_domain_sim_gravity",
    "FLUIDSIM_PT_physics_domain_sim_pressure",
    "FLUIDSIM_PT_physics_domain_sim_viscosity",
    "FLUIDSIM_PT_physics_domain_sim_surface_tension",
    "FLUIDSIM_PT_physics_domain_sim_color_mixing",
    "FLUIDSIM_PT_physics_domain_sim_relax",
    "FLUIDSIM_PT_physics_domain_sim_limits",
    "FLUIDSIM_PT_physics_domain_sim_forces",
    "FLUIDSIM_PT_physics_domain_sim_dissipation",
    "FLUIDSIM_PT_physics_domain_sim_attraction",
    "FLUIDSIM_PT_physics_surface_scene_surface",
    "FLUIDSIM_PT_physics_whitewater_scene_whitewater",
    "FLUIDSIM_PT_scene_outputs",
    "FLUIDSIM_PT_scene_diagnostics",
    "FLUIDSIM_PT_object_role", "FLUIDSIM_PT_object_source",
    "FLUIDSIM_PT_object_collider", "FLUIDSIM_PT_object_force",
    "FLUIDSIM_PT_custom_effects", "FLUIDSIM_PT_effect_engineering",
    "FLUIDSIM_PT_effect_packages",
    "FLUIDSIM_PT_effect_warp_code",
]

STATE = {
    "phase": "init",
    "checks": [],
    "t_play_start": 0.0,
    "frame_at_play": 0,
    "solver_frame_at_play": 0,
    "panel_draw_state": "",
    "panel_draw_hits": [],
    "panel_draw_errors": [],
    "panel_draw_probes_installed": False,
    "panel_failure_injected": False,
}


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    STATE["checks"].append({"name": name, "passed": bool(cond),
                            "detail": str(detail)})


def _image_has_visible_content(path, min_width, min_height,
                               sample_limit=4000, threshold=0.02):
    if not os.path.exists(path):
        return False, "missing"
    try:
        img = bpy.data.images.load(path)
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {exc!r}"
    try:
        width, height = img.size
        if width < min_width or height < min_height:
            return False, f"{width}x{height}"
        pixels = img.pixels
        pixel_count = width * height
        if pixel_count == 0:
            return False, f"{width}x{height}, empty pixels"
        stride = max(pixel_count // sample_limit, 1)
        visible = 0
        sampled = 0
        for pixel_idx in range(0, pixel_count, stride):
            rgba_idx = pixel_idx * 4
            luminance = (
                pixels[rgba_idx] + pixels[rgba_idx + 1] + pixels[rgba_idx + 2]
            ) / 3.0
            sampled += 1
            if luminance > threshold:
                visible += 1
            if sampled >= sample_limit:
                break
        minimum_visible = max(20, sampled // 40)
        if visible <= minimum_visible:
            return False, (
                f"{width}x{height}, {visible}/{sampled} visible samples"
            )
        return True, f"{width}x{height}, {visible}/{sampled} visible samples"
    finally:
        bpy.data.images.remove(img)


def _view3d_override():
    wm = bpy.context.window_manager
    scene = getattr(bpy.context, "scene", None)
    windows = sorted(wm.windows, key=lambda window: window.scene != scene)
    for window in windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next(r for r in area.regions if r.type == "WINDOW")
                return {"window": window, "area": area, "region": region,
                        "screen": window.screen}
    return None


def _panel_override(panel_id):
    panel_cls = getattr(bpy.types, panel_id, None)
    if panel_cls is None or getattr(panel_cls, "bl_space_type", "") != "PROPERTIES":
        return _view3d_override()
    wm = bpy.context.window_manager
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type != "PROPERTIES":
                continue
            context_name = str(getattr(panel_cls, "bl_context", "scene") or "scene").upper()
            try:
                area.spaces.active.context = context_name
            except TypeError:
                pass
            region = next(r for r in area.regions if r.type == "WINDOW")
            return {"window": window, "area": area, "region": region,
                    "screen": window.screen}
    return None


def _redraw_all():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def _select_object(name):
    obj = bpy.data.objects.get(name)
    for candidate in bpy.context.scene.objects:
        candidate.select_set(False)
    if obj is not None:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    return obj


def _install_panel_draw_probes():
    if STATE["panel_draw_probes_installed"]:
        return

    inject_failure = os.environ.get(PANEL_DRAW_FAILURE_ENV) == "1"
    for panel_id in PANEL_IDS:
        cls = getattr(bpy.types, panel_id, None)
        if cls is None or not hasattr(cls, "draw"):
            continue
        original_draw = cls.draw

        def make_wrapped(panel_cls, draw_fn):
            def wrapped(self, context):
                try:
                    if (
                        inject_failure
                        and panel_cls.bl_idname == PANEL_DRAW_FAILURE_PANEL
                        and not STATE["panel_failure_injected"]
                    ):
                        STATE["panel_failure_injected"] = True
                        raise RuntimeError(
                            "intentional panel draw smoke failure")
                    result = draw_fn(self, context)
                    STATE["panel_draw_hits"].append({
                        "state": STATE["panel_draw_state"],
                        "panel": panel_cls.bl_idname,
                    })
                    return result
                except Exception as exc:  # noqa: BLE001
                    STATE["panel_draw_errors"].append({
                        "state": STATE["panel_draw_state"],
                        "panel": panel_cls.bl_idname,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=8),
                    })
                    raise

            return wrapped

        cls.draw = make_wrapped(cls, original_draw)

    STATE["panel_draw_probes_installed"] = True


def _call_panel(panel_id):
    override = _panel_override(panel_id)
    kwargs = {"name": panel_id, "keep_open": False}
    if override:
        with bpy.context.temp_override(**override):
            return bpy.ops.wm.call_panel(**kwargs)
    return bpy.ops.wm.call_panel(**kwargs)


def _panel_effectively_visible(panel_id, context):
    """Match Blender's parent-aware panel disclosure for smoke selection."""
    seen = set()
    current_id = panel_id
    while current_id:
        if current_id in seen:
            raise RuntimeError(f"Panel parent cycle at {current_id}")
        seen.add(current_id)
        panel_cls = getattr(bpy.types, current_id, None)
        if panel_cls is None:
            return False
        poll = getattr(panel_cls, "poll", None)
        if callable(poll) and not poll(context):
            return False
        current_id = getattr(panel_cls, "bl_parent_id", "") or ""
    return True


def _draw_all_panels_smoke(state_name, active_object_name=None):
    _install_panel_draw_probes()
    STATE["panel_draw_state"] = state_name
    if active_object_name is not None:
        _select_object(active_object_name)

    missing = [p for p in PANEL_IDS if not hasattr(bpy.types, p)]
    call_failures = []
    expected_visible = []
    before_hits = len(STATE["panel_draw_hits"])
    before_errors = len(STATE["panel_draw_errors"])

    if not missing:
        for panel_id in PANEL_IDS:
            override = _panel_override(panel_id)
            try:
                if override:
                    with bpy.context.temp_override(**override):
                        visible = _panel_effectively_visible(panel_id, bpy.context)
                else:
                    visible = _panel_effectively_visible(panel_id, bpy.context)
            except Exception as exc:  # noqa: BLE001
                call_failures.append(f"{panel_id}.poll: {exc!r}")
                continue
            if not visible:
                continue
            expected_visible.append(panel_id)
            try:
                result = _call_panel(panel_id)
                if result not in PANEL_CALL_SUCCESS:
                    call_failures.append(f"{panel_id}: {result}")
            except Exception as exc:  # noqa: BLE001
                call_failures.append(f"{panel_id}: {exc!r}")
            _redraw_all()
            override = _view3d_override()
            if override:
                with bpy.context.temp_override(**override):
                    bpy.ops.wm.redraw_timer(
                        type="DRAW_WIN_SWAP", iterations=1)

    hits = STATE["panel_draw_hits"][before_hits:]
    errors = STATE["panel_draw_errors"][before_errors:]
    seen = {hit["panel"] for hit in hits}
    undrawn = [p for p in expected_visible if p not in seen]
    details = []
    if missing:
        details.append(f"missing={missing}")
    if undrawn:
        details.append(f"undrawn={undrawn}")
    if call_failures:
        details.append(f"call_failures={call_failures}")
    if errors:
        compact_errors = [
            f"{err['panel']}: {err['error']}" for err in errors]
        details.append(f"errors={compact_errors}")

    check(
        f"panel draw smoke ({state_name})",
        not missing and not undrawn and not call_failures and not errors,
        "; ".join(details) if details else f"{len(seen)} panels drew",
    )


class _PanelCaptureLayout:
    def __init__(self):
        self.labels = []
        self.operators = []
        self.properties = []
        self.enabled = True

    def column(self, align=False):
        return self

    def row(self, align=False):
        return self

    def box(self):
        return self

    def grid_flow(self, **_kwargs):
        return self

    def template_list(self, *_args, **_kwargs):
        return None

    def separator(self):
        return None

    def prop(self, _data, property_name, **_kwargs):
        self.properties.append(property_name)
        return None

    def label(self, text="", icon="NONE"):
        self.labels.append({"text": text, "icon": icon})

    def operator(self, operator_id, **kwargs):
        op = type("PanelOperatorCapture", (), {})()
        self.operators.append({"id": operator_id, **kwargs})
        return op


def _assert_panel_state(panel_id, state_name, *, label_contains=None,
                        operator_id=None, property_name=None, icon=None):
    cls = getattr(bpy.types, panel_id, None)
    if cls is None:
        check(f"panel state {state_name}", False, f"{panel_id} missing")
        return
    layout = _PanelCaptureLayout()
    panel = type("PanelDrawProbe", (), {"layout": layout})()
    try:
        cls.draw(panel, bpy.context)
    except Exception as exc:  # noqa: BLE001
        check(f"panel state {state_name}", False, repr(exc))
        return

    labels = [item["text"] for item in layout.labels]
    operators = [item["id"] for item in layout.operators]
    properties = layout.properties
    ok = True
    details = []
    if label_contains is not None:
        label_ok = any(label_contains in text for text in labels)
        ok = ok and label_ok
        if not label_ok:
            details.append(f"missing label {label_contains!r}")
    if operator_id is not None:
        operator_ok = operator_id in operators
        ok = ok and operator_ok
        if not operator_ok:
            details.append(f"missing operator {operator_id}")
    if property_name is not None:
        property_ok = property_name in properties
        ok = ok and property_ok
        if not property_ok:
            details.append(f"missing property {property_name}")
    if icon is not None:
        icon_ok = any(item["icon"] == icon for item in layout.labels)
        ok = ok and icon_ok
        if not icon_ok:
            details.append(f"missing icon {icon}")
    check(
        f"panel state {state_name}",
        ok,
        "; ".join(details) if details else
        f"{len(labels)} labels, {len(operators)} operators, "
        f"{len(properties)} properties",
    )


def _assert_panel_disclosure(panel_id, state_name, expected_visible):
    cls = getattr(bpy.types, panel_id, None)
    if cls is None:
        check(f"panel disclosure {state_name}", False, f"{panel_id} missing")
        return
    try:
        visible = _panel_effectively_visible(panel_id, bpy.context)
    except Exception as exc:  # noqa: BLE001
        check(f"panel disclosure {state_name}", False, repr(exc))
        return
    check(
        f"panel disclosure {state_name}",
        visible == expected_visible,
        f"visible={visible}, expected={expected_visible}",
    )


def _verify_preloaded_addon():
    expected_root = Path(EXPECTED_PRELOADED_ADDON_ROOT).resolve()
    module = sys.modules.get("fluidSim_warp_blender51")
    loaded_path = Path(getattr(module, "__file__", "")).resolve()
    path_matches = loaded_path == expected_root or expected_root in loaded_path.parents
    runtime_enabled = bool(addon_utils.check("fluidSim_warp_blender51")[1])
    preferences_enabled = "fluidSim_warp_blender51" in bpy.context.preferences.addons
    check("preloaded add-on source", path_matches,
          f"expected={expected_root}, loaded={loaded_path}")
    check("preloaded add-on enabled", runtime_enabled)
    check("preloaded add-on preferences enabled", preferences_enabled)
    return path_matches and runtime_enabled and preferences_enabled


def finish():
    passed = [c for c in STATE["checks"] if c["passed"]]
    verdict = {
        "result": "pass" if len(passed) == len(STATE["checks"]) else "fail",
        "passed": len(passed),
        "total": len(STATE["checks"]),
        "checks": STATE["checks"],
    }
    if STATE["panel_draw_errors"]:
        verdict["panel_draw_errors"] = STATE["panel_draw_errors"]
    with open(RESULT_PATH, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"GUI-ACCEPTANCE-{verdict['result'].upper()} "
          f"({verdict['passed']}/{verdict['total']})")
    if not EXPECTED_PRELOADED_ADDON_ROOT:
        bpy.ops.wm.quit_blender()


def state_machine():
    scene = bpy.context.scene
    phase = STATE["phase"]
    try:
        return _run_phase(phase, scene)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        check(f"phase {phase} raised", False, repr(exc))
        finish()
        return None


def _run_phase(phase, scene):
    if phase == "init":
        if EXPECTED_PRELOADED_ADDON_ROOT:
            if not _verify_preloaded_addon():
                finish()
                return None
        else:
            try:
                enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
                check("addon enable", True)
            except Exception as exc:  # noqa: BLE001
                check("addon enable", False, repr(exc))
                finish()
                return None
        check("gui mode", not bpy.app.background)
        scene.fluidsim.ui_mode = "BASIC"
        _draw_all_panels_smoke("addon enabled, solver inactive")
        _assert_panel_state(
            "FLUIDSIM_PT_viewport_status",
            "Basic Play action",
            operator_id="screen.animation_play",
        )
        _assert_panel_state(
            "FLUIDSIM_PT_header_popover",
            "Header popover guided setup without a domain",
            operator_id="fluidsim.setup_scene",
        )
        _assert_panel_state(
            "FLUIDSIM_PT_scene_setup",
            "Basic guided setup action",
            operator_id="fluidsim.setup_scene",
        )
        _assert_panel_disclosure(
            "FLUIDSIM_PT_scene_lifecycle", "Basic hides Lifecycle", False)
        _assert_panel_disclosure(
            "FLUIDSIM_PT_scene_diagnostics", "Basic hides Diagnostics", False)
        _assert_panel_disclosure(
            "FLUIDSIM_PT_effect_engineering", "Basic hides Effect Engineering", False)

        # Legacy manual controls remain available only through the explicit
        # Advanced disclosure tier.
        scene.fluidsim.ui_mode = "ADVANCED"
        _assert_panel_disclosure(
            "FLUIDSIM_PT_scene_lifecycle", "Advanced shows Lifecycle", True)
        _assert_panel_state(
            "FLUIDSIM_PT_scene_lifecycle",
            "Advanced manual Initialize",
            operator_id="fluidsim.initialize",
        )
        _assert_panel_state(
            "FLUIDSIM_PT_scene_lifecycle",
            "Advanced manual Step",
            operator_id="fluidsim.step",
        )
        check(
            "Lifecycle defaults closed",
            "DEFAULT_CLOSED" in bpy.types.FLUIDSIM_PT_scene_lifecycle.bl_options,
        )
        scene.fluidsim.ui_mode = "BASIC"

        # Factory-startup can legitimately provide an empty scene. Create the
        # setup input only after proving the no-Domain Basic presentation.
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 1.0))
        setup_input = bpy.context.object
        setup_input.name = "FluidSim GUI Setup Input"
        STATE["setup_input_name"] = setup_input.name
        check(
            "guided setup input selected",
            setup_input.select_get() and bpy.context.view_layer.objects.active == setup_input,
            setup_input.name,
        )
        STATE["phase"] = "setup"
        return 0.1

    from fluidSim_warp_blender51.blender.scene_bridge import (
        active_domain_settings,
        session,
    )

    if phase == "setup":
        fs = scene.fluidsim
        fs.show_overlay = True
        bpy.ops.fluidsim.setup_scene(start_preview=False)
        check("setup_scene", fs.domain_object is not None)
        domain_settings = active_domain_settings(scene)
        check(
            "authoritative domain settings",
            domain_settings is not None,
            getattr(fs.domain_object, "name", "no designated domain"),
        )
        if domain_settings is None:
            raise RuntimeError(
                "setup_scene did not designate an authoritative domain settings owner")
        fs.ui_mode = "ADVANCED"
        domain_settings.voxel_size = 0.08
        domain_settings.max_particles = 500_000
        domain_settings.whitewater_enabled = True
        domain_settings.profiling_enabled = True
        domain_settings.height_enabled = True
        domain_settings.viscosity_enabled = True
        domain_settings.max_speed = 2.5
        domain_settings.limit_behavior = "PERCENT"
        src = next(
            (
                obj for obj in scene.objects
                if getattr(getattr(obj, "fluidsim", None), "role", "NONE")
                == "SOURCE"
            ),
            None,
        )
        if src is not None:
            STATE["source_name"] = src.name
            if src.fluidsim.emission_mode != "INITIAL_FILL":
                raise RuntimeError(
                    "new Phase 3 source did not use Initial Fill: "
                    f"{src.fluidsim.emission_mode}")
            src.fluidsim.use_lifespan = True

        # Exercise normal-user role tagging through the surfaced operator seam.
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(4.0, 1.0, 1.0))
        sink = bpy.context.object
        sink.name = "FluidSim GUI Liquid Sink"
        sink_result = bpy.ops.fluidsim.tag_liquid_sink()
        if sink_result != {"FINISHED"} or sink.fluidsim.role != "SINK":
            raise RuntimeError(
                f"Tag Liquid Sink workflow failed: {sink_result}, "
                f"role={sink.fluidsim.role}")
        sink.fluidsim.sink_region_kind = "MESH_SHELL"
        sink.fluidsim.mesh_object = sink
        _assert_panel_state(
            "FLUIDSIM_PT_object_region",
            "configured Liquid Sink",
            property_name="sink_region_kind",
        )
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(4.0, -1.0, 1.0))
        force = bpy.context.object
        force.name = "FluidSim GUI Force Region"
        force_result = bpy.ops.fluidsim.tag_force_region()
        if force_result != {"FINISHED"} or force.fluidsim.role != "FORCE":
            raise RuntimeError(
                f"Tag Force Region workflow failed: {force_result}, "
                f"role={force.fluidsim.role}")
        force.fluidsim.force_type = "DIRECTIONAL"
        force.fluidsim.force_strength = 0.0
        _assert_panel_state(
            "FLUIDSIM_PT_object_force",
            "configured Force Region",
            property_name="force_type",
        )
        # These objects exist only to exercise the role-specific UI. They are
        # deliberately outside the simulation domain and must not enter the
        # strict runtime gather path during the playback workload.
        sink.fluidsim.enabled = False
        force.fluidsim.enabled = False
        if src is not None:
            bpy.ops.object.select_all(action="DESELECT")
            src.select_set(True)
            bpy.context.view_layer.objects.active = src
        result = bpy.ops.fluidsim.effect_add(
            effect_id="fluidsim.vortex_force")
        if result != {"FINISHED"}:
            raise RuntimeError(f"Custom Effect setup failed: {sorted(result)}")
        scene.fluidsim_effects.items[-1].enabled = False
        _assert_panel_state(
            "FLUIDSIM_PT_custom_effects",
            "Basic effect Play action",
            operator_id="screen.animation_play",
        )
        _assert_panel_state(
            "FLUIDSIM_PT_custom_effects",
            "Basic effect Apply Changes action",
            operator_id="fluidsim.apply_changes",
        )
        _draw_all_panels_smoke(
            "demo configured, solver inactive", STATE.get("source_name"))

        # Camera + light for the render check.
        cam_data = bpy.data.cameras.new("FluidSim Cam")
        cam = bpy.data.objects.new("FluidSim Cam", cam_data)
        scene.collection.objects.link(cam)
        cam.location = (7.5, -7.5, 5.5)
        cam.rotation_euler = (1.05, 0.0, 0.785)
        scene.camera = cam
        light_data = bpy.data.lights.new("FluidSim Sun", type="SUN")
        light_data.energy = 3.0
        light = bpy.data.objects.new("FluidSim Sun", light_data)
        scene.collection.objects.link(light)
        light.rotation_euler = (0.6, 0.2, 0.0)

        scene.fluidsim.ui_mode = "ADVANCED"
        result = bpy.ops.fluidsim.initialize()
        sess = session()
        check("initialize solver", result == {"FINISHED"} and sess.active,
              scene.fluidsim.status)
        check("cuda device", sess.active
              and str(sess.solver.device).startswith("cuda"))
        _draw_all_panels_smoke(
            "solver initialized", STATE.get("source_name"))
        # Post-D4 the runtime stats are consolidated into the Diagnostics
        # telemetry panel (same stable id) and rendered as read-only labels, so
        # the particle count surfaces under its "Particles" key label rather than
        # as an editable property.
        _assert_panel_state(
            "FLUIDSIM_PT_scene_runtime_stats",
            "initialized stats",
            label_contains="Particles",
        )
        _assert_panel_state(
            "FLUIDSIM_PT_scene_lifecycle",
            "Advanced manual Release",
            operator_id="fluidsim.release",
        )
        scene.fluidsim.ui_mode = "BASIC"
        STATE["phase"] = "play"
        return 0.2

    if phase == "play":
        fs = scene.fluidsim
        sess = session()
        scene.frame_set(scene.frame_start)
        STATE["frame_at_play"] = scene.frame_current
        STATE["solver_frame_at_play"] = sess.solver.frame
        STATE["t_play_start"] = time.perf_counter()
        # Blender's own Play must arm and admit the simulation by itself:
        # there is no Preview step in front of timeline playback any more.
        fs.live = False
        with bpy.context.temp_override(**_view3d_override()):
            result = bpy.ops.screen.animation_play()
        check("native Play arms simulation", result == {"FINISHED"} and fs.live,
              scene.fluidsim.status)
        owner_windows = [
            window for window in bpy.context.window_manager.windows
            if window.scene == scene
        ]
        check("animation playing", bool(owner_windows) and any(
            window.screen.is_animation_playing for window in owner_windows))
        STATE["phase"] = "playing"
        return 3.0

    if phase == "playing":
        sess = session()
        elapsed = time.perf_counter() - STATE["t_play_start"]
        scene_frames = bpy.context.scene.frame_current - STATE["frame_at_play"]
        solver_advanced = sess.solver.frame - STATE["solver_frame_at_play"]
        if (min(scene_frames, solver_advanced) < PLAYBACK_MIN_FRAMES
                and elapsed < PLAYBACK_PROGRESS_TIMEOUT_SECONDS):
            # A new Blender process can spend tens of seconds capturing the
            # first CUDA graphs.  Keep the timeline owner active until it has
            # produced the required live frames; a fixed one-shot timer would
            # otherwise cancel immediately after that cold first frame.
            return 0.2
        check("playback advanced timeline",
              scene_frames >= PLAYBACK_MIN_FRAMES,
              f"{scene_frames} frames in {elapsed:.1f}s")
        check("solver stepped during playback",
              solver_advanced >= PLAYBACK_MIN_FRAMES,
              f"{solver_advanced} solver frames")
        fs = bpy.context.scene.fluidsim
        check("particles during playback", fs.stat_particles > 1000,
              f"{fs.stat_particles:,}")
        from fluidSim_warp_blender51.blender import output_publication

        surf = output_publication.output_object(scene, "FluidSim Surface")
        check("surface visible in viewport",
              surf is not None and surf.visible_get()
              and len(surf.data.vertices) > 100,
              f"{len(surf.data.vertices) if surf else 0} verts")
        ww = output_publication.output_object(scene, "FluidSim Whitewater")
        check("whitewater object present", ww is not None,
              ww.type if ww else "")
        STATE["phase"] = "rewind"
        return 0.3

    if phase == "rewind":
        sess = session()
        scene = bpy.context.scene
        # Rewind while Preview is still armed so the production frame handler
        # owns the reset. Pausing first would intentionally bypass that handler.
        scene.frame_set(scene.frame_start)
        check("rewind resets solver", sess.solver.frame == 0,
              f"solver frame {sess.solver.frame}")
        pause_result = bpy.ops.fluidsim.preview_pause()
        owner_windows = [
            window for window in bpy.context.window_manager.windows
            if window.scene == scene
        ]
        check("Pause Preview", pause_result == {"FINISHED"}
              and not scene.fluidsim.live
              and bool(owner_windows)
              and all(not window.screen.is_animation_playing
                      for window in owner_windows),
              scene.fluidsim.status)
        # Step manually after reset through the Advanced compatibility path.
        scene.fluidsim.ui_mode = "ADVANCED"
        result = bpy.ops.fluidsim.step(frames=3)
        check("manual step after reset", result == {"FINISHED"},
              scene.fluidsim.status)
        _draw_all_panels_smoke(
            "post-playback manual step", STATE.get("source_name"))
        _assert_panel_state(
            "FLUIDSIM_PT_scene_control",
            "stepped status",
            label_contains="Frame ",
        )
        STATE["phase"] = "ui"
        return 0.2

    if phase == "ui":
        # Force full UI redraws so sidebar panels + overlay draw with a real
        # context. Panel draw coverage is checked explicitly above.
        _redraw_all()
        override = _view3d_override()
        if override:
            with bpy.context.temp_override(**override):
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
        missing = [p for p in PANEL_IDS if not hasattr(bpy.types, p)]
        check("panels registered in GUI", not missing, str(missing))
        check("ui redraw with overlay", True)
        _assert_panel_state(
            "FLUIDSIM_PT_header_popover",
            "Header popover quality and workspace button",
            property_name="quality",
            operator_id="fluidsim.open_workspace",
        )

        # Property edits through the UI data path (as the panels would).
        from fluidSim_warp_blender51.blender import pending_settings, scene_bridge

        scene = bpy.context.scene
        domain_settings = active_domain_settings(scene)
        if domain_settings is None:
            raise RuntimeError(
                "GUI property edits require an authoritative domain settings owner")
        domain_settings.flip_ratio = 0.9
        domain_settings.pressure_iterations = 150
        check("property edits applied",
              abs(domain_settings.flip_ratio - 0.9) < 1e-6
              and domain_settings.pressure_iterations == 150)

        # Pending state is owned by the simplified Scene Properties panel.
        sess = scene_bridge.session()
        solver_before_pending = sess.solver
        domain_settings.voxel_size *= 0.9
        pending = pending_settings.pending_presentation(scene)
        check("mid-run GUI edit is pending", pending["action"] == "REINIT_PENDING"
              and sess.solver is solver_before_pending, pending)
        pending_panel_result = _call_panel("FLUIDSIM_PT_scene_pending")
        check("pending Properties panel callable",
              pending_panel_result in PANEL_CALL_SUCCESS,
              pending_panel_result)
        revert_result = bpy.ops.fluidsim.revert_pending()
        check("GUI pending revert action", revert_result == {"FINISHED"}
              and pending_settings.pending_presentation(scene)["action"] == "NONE",
              revert_result)

        # Viewport proof. The offscreen draw runs on every platform; the window
        # grab is skipped on Linux, which has no window screenshot to take.
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.append(scripts_dir)
        from _offscreen_view3d import check_view3d_offscreen

        check("viewport offscreen draw", *check_view3d_offscreen(_view3d_override()))
        if not sys.platform.startswith("linux"):
            try:
                override = _view3d_override()
                with bpy.context.temp_override(**override):
                    bpy.ops.screen.screenshot_area(filepath=SCREENSHOT_PATH)
                ok, detail = _image_has_visible_content(
                    SCREENSHOT_PATH, min_width=320, min_height=200)
                check("viewport screenshot", ok, detail)
            except Exception as exc:  # noqa: BLE001
                check("viewport screenshot", False, repr(exc))
        STATE["phase"] = "workspace"
        return 0.3

    if phase == "workspace":
        # The header's workspace button in the real window manager. Blender applies
        # a window.workspace change on the next event-loop tick, so every switch is
        # verified one tick after it is requested.
        override = _view3d_override()
        window = override["window"]
        step = STATE.get("workspace_step", "create")
        if step == "create":
            STATE["workspace_home"] = window.workspace.name
            STATE["workspace_home_mode"] = window.workspace.object_mode
            STATE["workspace_count"] = len(bpy.data.workspaces)
            check("FluidSim workspace absent beforehand",
                  bpy.data.workspaces.get("FluidSim") is None)
            with bpy.context.temp_override(**override):
                result = bpy.ops.fluidsim.open_workspace()
            check("open_workspace creates the workspace", result == {"FINISHED"},
                  result)
            STATE["workspace_step"] = "verify"
            return 0.5
        home = bpy.data.workspaces.get(STATE["workspace_home"])
        if step == "verify":
            created = bpy.data.workspaces.get("FluidSim")
            check("FluidSim workspace added once",
                  created is not None
                  and len(bpy.data.workspaces) == STATE["workspace_count"] + 1)
            check("window switched to the FluidSim workspace",
                  window.workspace.name == "FluidSim", window.workspace.name)
            areas = [area for screen in created.screens for area in screen.areas]
            viewports = [area for area in areas if area.type == "VIEW_3D"]
            check("FluidSim workspace has a viewport with the sidebar open",
                  bool(viewports) and all(
                      space.show_region_ui
                      for area in viewports for space in area.spaces
                      if space.type == "VIEW_3D"),
                  [area.type for area in areas])
            check("FluidSim workspace has a timeline",
                  any(area.type == "DOPESHEET_EDITOR" and area.ui_type == "TIMELINE"
                      for area in areas),
                  [(area.type, area.ui_type) for area in areas])
            check("FluidSim workspace opens in Object mode",
                  created.object_mode == "OBJECT", created.object_mode)
            check("home workspace untouched",
                  home is not None
                  and home.object_mode == STATE["workspace_home_mode"],
                  home.object_mode if home is not None else None)
            window.workspace = home
            STATE["workspace_step"] = "reuse"
            return 0.5
        if step == "reuse":
            check("window returned to the home workspace",
                  window.workspace.name == STATE["workspace_home"],
                  window.workspace.name)
            with bpy.context.temp_override(**override):
                result = bpy.ops.fluidsim.open_workspace()
            check("open_workspace switches to the existing workspace",
                  result == {"FINISHED"}, result)
            STATE["workspace_step"] = "reused"
            return 0.5
        if step == "reused":
            check("second press reuses the workspace",
                  window.workspace.name == "FluidSim"
                  and len(bpy.data.workspaces) == STATE["workspace_count"] + 1,
                  f"{window.workspace.name}, {len(bpy.data.workspaces)} workspaces")
            window.workspace = home
            STATE["workspace_step"] = "restored"
            return 0.5
        check("window restored to the home workspace",
              window.workspace.name == STATE["workspace_home"],
              window.workspace.name)
        STATE["phase"] = "render"
        return 0.2

    if phase == "render":
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(
            bpy.types, "RenderSettings") and "BLENDER_EEVEE_NEXT" in {
            e.identifier for e in
            bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        } else "BLENDER_EEVEE"
        scene.render.resolution_x = 480
        scene.render.resolution_y = 320
        scene.render.filepath = RENDER_PATH
        bpy.ops.render.render(write_still=True)
        ok, detail = _image_has_visible_content(
            RENDER_PATH, min_width=480, min_height=320)
        check("render output written", os.path.exists(RENDER_PATH), detail)
        check("render contains visible content", ok, detail)
        STATE["phase"] = "diag"
        return 0.2

    if phase == "diag":
        sess = session()
        check("profiler frames in GUI",
              sess.profiler is not None and len(sess.profiler.history) >= 3,
              f"{len(sess.profiler.history) if sess.profiler else 0}")
        sess.validation_results = [
            {"name": "synthetic pass", "passed": True},
            {"name": "synthetic fail", "passed": False},
        ]
        _assert_panel_state(
            "FLUIDSIM_PT_scene_diagnostics",
            "validation result",
            operator_id="fluidsim.run_validation",
        )
        from fluidSim_warp_blender51.blender import pending_settings, scene_bridge

        scene_bridge.set_error(scene, "intentional panel-visible error")
        _assert_panel_state(
            "FLUIDSIM_PT_scene_control",
            "error visible",
            label_contains="intentional panel-visible error",
            operator_id="fluidsim.clear_error",
            icon="ERROR",
        )
        bpy.ops.fluidsim.clear_error()
        # Benchmark (short) through the operator.
        scene = bpy.context.scene
        scene.fluidsim.benchmark_frames = 6
        result = bpy.ops.fluidsim.run_benchmark(save_baseline=False)
        check("benchmark operator in GUI", result == {"FINISHED"},
              scene.fluidsim.status)
        ok = sess.benchmark_result is not None and \
            "timeStepAverage" in (sess.benchmark_result or {})
        check("benchmark result schema", ok)
        _assert_panel_state(
            "FLUIDSIM_PT_diagnostics_benchmark",
            "benchmark result",
            label_contains="Average step:",
        )
        _draw_all_panels_smoke(
            "diagnostics populated", STATE.get("source_name"))
        finish()
        return None

    finish()
    return None


def start():
    bpy.app.timers.register(state_machine, first_interval=0.5)


start()
