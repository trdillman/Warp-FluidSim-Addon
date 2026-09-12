"""Interactive Blender GUI acceptance test for the FluidSim Warp add-on.

Run with a real (non-background) Blender:
    blender --python tests/blender_gui_acceptance.py

Uses a timer-driven state machine so the UI event loop keeps running:
enables the add-on, builds the demo scene, initializes the CUDA solver,
starts viewport animation playback with live simulation, verifies the solver
advances during real playback, checks rewind-reset behavior, forces UI
redraws of the sidebar panels, captures a viewport screenshot, renders a
frame, and writes a JSON verdict before quitting Blender.

Set FLUIDSIM_PANEL_DRAW_INJECT_FAILURE=1 to prove the panel draw smoke gate
fails on real Blender panel draw exceptions.
"""

import json
import os
import sys
import tempfile
import time
import traceback

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon

ARTIFACTS = os.path.join(REPO_ROOT, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)
RESULT_PATH = os.environ.get(
    "FLUIDSIM_GUI_RESULT", os.path.join(ARTIFACTS, "phase3_gui_result.json"))
SCREENSHOT_PATH = os.path.join(ARTIFACTS, "phase3_gui_viewport.png")
RENDER_PATH = os.path.join(ARTIFACTS, "phase3_gui_render.png")
PANEL_DRAW_FAILURE_ENV = "FLUIDSIM_PANEL_DRAW_INJECT_FAILURE"
PANEL_DRAW_FAILURE_PANEL = "FLUIDSIM_PT_scene_control"
PANEL_CALL_SUCCESS = ({"FINISHED"}, {"RUNNING_MODAL"}, {"INTERFACE"})
PLAYBACK_MIN_FRAMES = 3
PLAYBACK_PROGRESS_TIMEOUT_SECONDS = 45.0

PANEL_IDS = [
    "FLUIDSIM_PT_viewport_status",
    "FLUIDSIM_VIEW3D_PT_live_tuning",
    "FLUIDSIM_PT_physics_root",
    "FLUIDSIM_PT_physics_surface_output",
    "FLUIDSIM_PT_scene_control", "FLUIDSIM_PT_scene_pending",
    "FLUIDSIM_PT_scene_runtime_stats", "FLUIDSIM_PT_scene_domain",
    "FLUIDSIM_PT_scene_simulation", "FLUIDSIM_PT_sim_pressure",
    "FLUIDSIM_PT_sim_viscosity", "FLUIDSIM_PT_scene_surface",
    "FLUIDSIM_PT_scene_whitewater", "FLUIDSIM_PT_scene_outputs",
    "FLUIDSIM_PT_scene_cache", "FLUIDSIM_PT_cache_playback",
    "FLUIDSIM_PT_cache_bake", "FLUIDSIM_PT_scene_diagnostics",
    "FLUIDSIM_PT_object_role", "FLUIDSIM_PT_object_source",
    "FLUIDSIM_PT_object_collider", "FLUIDSIM_PT_object_force",
    "FLUIDSIM_PT_custom_effects", "FLUIDSIM_PT_effect_packages",
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
    for window in wm.windows:
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
            panel_cls = getattr(bpy.types, panel_id)
            poll = getattr(panel_cls, "poll", None)
            override = _panel_override(panel_id)
            try:
                if override:
                    with bpy.context.temp_override(**override):
                        visible = poll(bpy.context) if callable(poll) else True
                else:
                    visible = poll(bpy.context) if callable(poll) else True
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
        try:
            enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
            check("addon enable", True)
        except Exception as exc:  # noqa: BLE001
            check("addon enable", False, repr(exc))
            finish()
            return None
        check("gui mode", not bpy.app.background)
        _draw_all_panels_smoke("addon enabled, solver inactive")
        # After the D3 hub split the Initialize action lives in the dedicated
        # Lifecycle panel; the read-only Status hub no longer draws operators.
        _assert_panel_state(
            "FLUIDSIM_PT_scene_lifecycle",
            "uninitialized status",
            operator_id="fluidsim.initialize",
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
        bpy.ops.fluidsim.setup_scene()
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
        STATE["phase"] = "play"
        return 0.2

    if phase == "play":
        fs = scene.fluidsim
        sess = session()
        bpy.ops.fluidsim.toggle_live()
        check("live toggled on", fs.live)
        scene.frame_set(scene.frame_start)
        STATE["frame_at_play"] = scene.frame_current
        STATE["solver_frame_at_play"] = sess.solver.frame
        STATE["t_play_start"] = time.perf_counter()
        override = _view3d_override()
        if override:
            with bpy.context.temp_override(**override):
                bpy.ops.screen.animation_play()
        else:
            bpy.ops.screen.animation_play()
        check("animation playing", bpy.context.screen.is_animation_playing
              if bpy.context.screen else True)
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
        # Stop playback.
        if bpy.context.screen and bpy.context.screen.is_animation_playing:
            override = _view3d_override()
            if override:
                with bpy.context.temp_override(**override):
                    bpy.ops.screen.animation_cancel(restore_frame=False)
            else:
                bpy.ops.screen.animation_cancel(restore_frame=False)
        check("playback advanced timeline",
              scene_frames >= PLAYBACK_MIN_FRAMES,
              f"{scene_frames} frames in {elapsed:.1f}s")
        check("solver stepped during playback",
              solver_advanced >= PLAYBACK_MIN_FRAMES,
              f"{solver_advanced} solver frames")
        fs = bpy.context.scene.fluidsim
        check("particles during playback", fs.stat_particles > 1000,
              f"{fs.stat_particles:,}")
        surf = bpy.data.objects.get("FluidSim Surface")
        check("surface visible in viewport",
              surf is not None and surf.visible_get()
              and len(surf.data.vertices) > 100,
              f"{len(surf.data.vertices) if surf else 0} verts")
        ww = bpy.data.objects.get("FluidSim Whitewater")
        check("whitewater object present", ww is not None,
              ww.type if ww else "")
        STATE["phase"] = "rewind"
        return 0.3

    if phase == "rewind":
        sess = session()
        scene = bpy.context.scene
        # Rewinding to the start frame must reset the simulation.
        scene.frame_set(scene.frame_start)
        check("rewind resets solver", sess.solver.frame == 0,
              f"solver frame {sess.solver.frame}")
        bpy.ops.fluidsim.toggle_live()
        check("live toggled off", not scene.fluidsim.live)
        # Step manually after reset (operator path in GUI context).
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

        # Property edits through the UI data path (as the panels would).
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
        from fluidSim_warp_blender51.blender import pending_settings, scene_bridge
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

        # Viewport screenshot (viewport-visible proof).
        try:
            override = _view3d_override()
            with bpy.context.temp_override(**override):
                bpy.ops.screen.screenshot_area(filepath=SCREENSHOT_PATH)
            ok, detail = _image_has_visible_content(
                SCREENSHOT_PATH, min_width=320, min_height=200)
            check("viewport screenshot", ok, detail)
        except Exception as exc:  # noqa: BLE001
            check("viewport screenshot", False, repr(exc))
        STATE["phase"] = "render"
        return 0.3

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
