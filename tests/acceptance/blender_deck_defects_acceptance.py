"""Real-GUI acceptance: the Liquid Deck defect fixes behave in a real Blender session.

Run with a real (non-background) Blender and simulated input events:
    blender --factory-startup --enable-event-simulate \
        --python tests/acceptance/blender_deck_defects_acceptance.py

Phase 1 runs in the unsaved startup file:

- Bake is greyed, with the reason, when there is no domain and when the cache
  path is relative in an unsaved file;
- Add > Fluid Domain lands on the 3D cursor;
- the effect-list buttons are greyed with "Add an effect first";
- turning on Liquid Preview with no particles warns;
- the domain's "Open Scene FluidSim Settings" opens the Scene tab, not Physics.

Phase 2 opens demos/dam_break.blend (never saved) and checks, in the live UI:

- Calculate Safe Options finishes;
- Step advances the solver frame the Deck readout shows; a clicked Restart asks
  first when stepped progress would be lost;
- the wheel greys Apply when nothing is pending and Step during cache playback;
- palette rows grey with reasons, the inline editor's start text reads back
  exactly, and a palette write is one real undo step;
- the bar hides with the viewport's Overlays toggle;
- the palette (Ctrl+K), wheel (Alt+Right-click) and header popover open from
  real input and draw without errors;
- a real modal Production Bake: the Bake slot shows progress, gizmo and palette
  edits are held, Add Domain is greyed, a Cancel from another scene leaves the
  bake running, a Cancel from this scene stops it, and the slot then offers
  Resume or Restart.

Writes report.json plus screenshots to FLUIDSIM_DEFECTS_OUT (default
artifacts/deck_defects) and prints DECK-DEFECTS-PASS or DECK-DEFECTS-FAIL.
"""

import json
import os
import shutil
import struct
import sys
import time
import traceback

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import ADDON_NAME, enable_repo_addon  # noqa: E402

OUT_DIR = os.environ.get("FLUIDSIM_DEFECTS_OUT",
                         os.path.join(REPO_ROOT, "artifacts", "deck_defects"))
DEMO = os.path.join(REPO_ROOT, "demos", "dam_break.blend")
PASS_MARKER = "DECK-DEFECTS-PASS"
FAIL_MARKER = "DECK-DEFECTS-FAIL"
OVERALL_TIMEOUT_SECONDS = 420.0

checks = {}
report = {"checks": checks, "details": {}, "screenshots": []}


def log(*parts):
    print("DECK-DEFECTS", *parts, flush=True)


def check(name, ok, detail=None):
    checks[name] = bool(ok)
    if detail is not None:
        report["details"][name] = detail
    log("check", name, "OK" if ok else "FAILED", "" if detail is None else detail)
    return bool(ok)


def mod(name):
    from importlib import import_module

    return import_module(f"{ADDON_NAME}.{name}")


# Window helpers ---------------------------------------------------------------------

class UI:
    """The window, its 3D View and Properties editor, and simulated input."""

    def __init__(self):
        self.refresh()
        self.x = self.y = 0

    def refresh(self):
        self.win = bpy.context.window_manager.windows[0]
        self.screen = self.win.screen
        self.v3d = next(a for a in self.screen.areas if a.type == "VIEW_3D")
        self.view = next(r for r in self.v3d.regions if r.type == "WINDOW")
        self.props = next((a for a in self.screen.areas if a.type == "PROPERTIES"), None)

    def ctx(self, area=None, **extra):
        area = area or self.v3d
        region = next(r for r in area.regions if r.type == "WINDOW")
        return bpy.context.temp_override(window=self.win, screen=self.screen, area=area,
                                         region=region, **extra)

    def move(self, x, y):
        self.x, self.y = int(x), int(y)
        self.win.event_simulate(type="MOUSEMOVE", value="NOTHING", x=self.x, y=self.y)

    def centre(self):
        self.move(self.view.x + self.view.width * 0.5, self.view.y + self.view.height * 0.5)

    def key(self, type_, unicode=None, **mods):
        extra = {"unicode": unicode} if unicode else {}
        self.win.event_simulate(type=type_, value="PRESS", x=self.x, y=self.y, **mods, **extra)
        self.win.event_simulate(type=type_, value="RELEASE", x=self.x, y=self.y, **mods)

    def screenshot(self, name, area=None):
        path = os.path.join(OUT_DIR, name + ".png")
        try:
            with bpy.context.temp_override(window=self.win, area=area or self.v3d):
                bpy.ops.screen.screenshot_area(filepath=path)
        except Exception as exc:  # noqa: BLE001 - a missing screenshot is not a defect
            log("screenshot failed", name, repr(exc))
            return
        report["screenshots"].append(os.path.basename(path))


def poll(ui, idname):
    """``(poll result, poll message)`` for ``idname`` in the 3D View."""
    controls = mod("blender.deck.controls")
    module, _, name = idname.partition(".")
    with ui.ctx():
        ok = getattr(getattr(bpy.ops, module), name).poll()
        return ok, ("" if ok else controls.poll_reason(bpy.context, idname))


def status(ui, control):
    controls = mod("blender.deck.controls")
    with ui.ctx():
        return controls.control_status(bpy.context, control)


# Phase 1: unsaved startup file ------------------------------------------------------

def phase_startup(ui):
    from mathutils import Vector

    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    yield 0.3

    ok, why = poll(ui, "fluidsim.bake_start")
    check("Bake greyed with no domain", not ok and "domain" in why.lower(), why)

    scene.cursor.location = (2.0, -1.5, 0.75)
    with ui.ctx():
        result = bpy.ops.fluidsim.add_domain("EXEC_DEFAULT")
    yield 0.3
    domain = bpy.context.active_object
    check("Add Fluid Domain finishes", result == {"FINISHED"}, sorted(result))
    if domain is None:
        return
    corners = [domain.matrix_world @ Vector(c) for c in domain.bound_box]
    base = (sum(c.x for c in corners) / 8, sum(c.y for c in corners) / 8,
            min(c.z for c in corners))
    check("Add Fluid Domain stands on the 3D cursor",
          all(abs(a - b) < 1e-4 for a, b in zip(base, (2.0, -1.5, 0.75))),
          [round(v, 4) for v in base])

    fs = scene.fluidsim
    fs.cache_path = "//fluidsim_cache/"
    ok, why = poll(ui, "fluidsim.bake_start")
    check("Bake greyed for a relative cache in an unsaved file",
          not ok and bool(why), why)

    for idname in ("fluidsim.effect_duplicate", "fluidsim.effect_move",
                   "fluidsim.effect_toggle_mute", "fluidsim.effect_toggle_solo"):
        ok, why = poll(ui, idname)
        check(f"{idname} greyed with no effects", not ok and why == "Add an effect first", why)

    controls = mod("blender.deck.controls")
    reports = []
    with ui.ctx():
        if fs.liquid_preview_enabled:
            controls.run_control(bpy.context, controls.LIQUID_PREVIEW)
        controls.run_control(bpy.context, controls.LIQUID_PREVIEW,
                             report=lambda kind, msg: reports.append(msg))
    check("Liquid Preview with no particles warns",
          controls.NO_PUBLISHED_PARTICLES in reports, reports)
    with ui.ctx():
        controls.run_control(bpy.context, controls.LIQUID_PREVIEW)

    if ui.props is None:
        check("a Properties editor is open", False)
        return
    space = ui.props.spaces.active
    space.context = "PHYSICS"
    with ui.ctx(active_object=domain, object=domain):
        result = bpy.ops.fluidsim.open_scene_properties("EXEC_DEFAULT", target="SCENE")
    yield 0.3
    check("Domain's Open Scene FluidSim Settings opens the Scene tab",
          result == {"FINISHED"} and space.context == "SCENE", [sorted(result), space.context])
    ui.screenshot("startup_scene_tab", ui.props)
    space.context = "PHYSICS"
    yield 0.4
    ui.screenshot("startup_physics_domain", ui.props)


# Phase 2: dam_break -----------------------------------------------------------------

def open_demo(ui):
    bpy.ops.wm.open_mainfile(filepath=DEMO, load_ui=False)
    yield 2.0
    ui.refresh()
    scene = bpy.context.scene
    scene.fluidsim.deck_shown = True
    check("dam_break opened", os.path.normcase(bpy.data.filepath) == os.path.normcase(DEMO),
          bpy.data.filepath)


def domain_settings(scene):
    return mod("blender.scene_bridge").active_domain_settings(scene)


def phase_session(ui):
    scene = bpy.context.scene
    fs = scene.fluidsim
    domain = domain_settings(scene)
    check("dam_break has a domain", domain is not None)
    if domain is None:
        return
    fs.live = False
    fs.cache_playback_enabled = False
    domain.voxel_size = max(float(domain.voxel_size), 0.14)
    domain.max_particles = min(int(domain.max_particles), 50_000)

    with ui.ctx():
        result = bpy.ops.fluidsim.apply_safe_settings("EXEC_DEFAULT")
    check("Calculate Safe Options finishes", result == {"FINISHED"},
          [sorted(result), fs.status])

    bar = mod("blender.deck.bar")
    controls = mod("blender.deck.controls")
    session = mod("operator_domains.session")
    ok, why = status(ui, controls.APPLY)
    check("Apply greyed when nothing is pending",
          not ok and why == "No changes are waiting to be applied", why)

    ok, why = status(ui, controls.STEP)
    check("Step greyed, with the reason, before the solver starts", not ok and bool(why), why)
    with ui.ctx():
        result = bpy.ops.fluidsim.initialize("EXEC_DEFAULT")
    yield 1.0
    check("solver initializes", result == {"FINISHED"}, [sorted(result), fs.status])
    for frame in (1, 2):
        with ui.ctx():
            result = bpy.ops.fluidsim.step("EXEC_DEFAULT", frames=1)
        yield 0.5
        check(f"Step {frame} finishes", result == {"FINISHED"}, [sorted(result), fs.status])
    solver_frame = bar._solver_frame(scene)
    readout = bar.sample_readout(scene)
    check("Deck readout follows the solver frame after Step",
          solver_frame is not None and readout.frame == solver_frame
          and solver_frame != scene.frame_current,
          {"solver": solver_frame, "readout": readout.frame,
           "timeline": scene.frame_current, "samples": list(readout.samples)})
    ui.centre()
    yield 0.4
    ui.screenshot("bar_after_step")

    check("stepped progress is at risk", session._preview_progress_at_risk(scene))
    with ui.ctx():
        result = bpy.ops.fluidsim.restart_simulation("INVOKE_DEFAULT")
    yield 0.6
    ui.screenshot("restart_confirm", None)
    check("clicked Restart asks first", result == {"RUNNING_MODAL"}, sorted(result))
    ui.key("ESC")
    yield 0.4
    check("dismissing the Restart dialog keeps the progress",
          bar._solver_frame(scene) == solver_frame, bar._solver_frame(scene))

    fs.cache_playback_enabled = True
    yield 0.2
    ok, why = status(ui, controls.STEP)
    check("Step greyed during cache playback",
          not ok and why.startswith("Cache playback is on"), why)
    fs.cache_playback_enabled = False

    wheel = mod("blender.deck.wheel")
    with ui.ctx():
        slots = wheel.stage_controls(bpy.context, "SIMULATE")
        statuses = [controls.control_status(bpy.context, c) for c in slots]
    report["details"]["wheel SIMULATE"] = [[c.label, ok, why]
                                           for c, (ok, why) in zip(slots, statuses)]
    apply_index = slots.index(controls.APPLY)
    tip = wheel.slot_tip("SIMULATE", apply_index, statuses, slots)
    check("wheel Apply slot says why it is greyed",
          tip[2] == "Unavailable: No changes are waiting to be applied", tip)

    with ui.ctx():
        result = bpy.ops.fluidsim.restart_simulation("EXEC_DEFAULT")
    yield 0.5
    check("Restart from a script runs without a dialog", result == {"FINISHED"}, sorted(result))


def phase_palette(ui):
    scene = bpy.context.scene
    palette = mod("blender.deck.palette")
    pending = mod("blender.pending_settings")

    with ui.ctx():
        ok, why = palette.row_status(bpy.context, palette.Entry(
            "operator", "fluidsim.bake_cancel", "Cancel Production Bake"))
    check("palette greys Cancel Bake with no bake", not ok and bool(why), why)

    entries = [e for e in palette.setting_entries(scene)
               if e.editable and e.value_type == "FLOAT"
               and getattr(pending.SETTING_TRANSITION_REGISTRY.get(e.target), "value", "")
               == "RESET"]
    check("a Reset-tier float setting is in the palette", bool(entries))
    if not entries:
        return
    entry = entries[0]
    group = pending.declaring_group(scene, entry.target)
    before = float(getattr(group, entry.target))
    text = palette.exact_value_text(before, entry.value_type)
    check("inline editor start text reads back exactly",
          struct.unpack("<f", struct.pack("<f", float(text)))[0] == before,
          [entry.target, text, before])

    with ui.ctx():
        bpy.ops.ed.undo_push(message="Deck defects baseline")
    new = before * 1.5 if before else 0.25
    message = palette.commit_inline(scene, entry, repr(new))
    after = float(getattr(group, entry.target))
    check("palette write reaches the property", abs(after - new) < 1e-5 * max(1.0, abs(new)),
          [entry.target, after])
    check("palette Reset-tier write says it is staged", bool(message), message)
    with ui.ctx():
        result = bpy.ops.ed.undo()
    yield 0.6
    scene = bpy.context.scene
    group = pending.declaring_group(scene, entry.target)
    check("one undo reverts the palette write",
          result == {"FINISHED"} and float(getattr(group, entry.target)) == before,
          [sorted(result), float(getattr(group, entry.target)), before])


def phase_surfaces(ui):
    ui.refresh()
    bar = mod("blender.deck.bar")
    registry = mod("blender.deck.registry").REGISTRY
    overlay = ui.v3d.spaces.active.overlay

    overlay.show_overlays = False
    with ui.ctx():
        hidden = bar.overlays_shown(bpy.context)
    yield 0.3
    ui.screenshot("bar_overlays_off")
    overlay.show_overlays = True
    with ui.ctx():
        shown = bar.overlays_shown(bpy.context)
    check("bar follows the Overlays toggle", hidden is False and shown is True, [hidden, shown])

    ui.centre()
    yield 0.3
    ui.key("K", ctrl=True)
    yield 0.5
    check("Ctrl+K opens the palette", registry.draw_handle("palette") is not None)
    for char in "visc":
        ui.key(char.upper(), unicode=char)
    yield 0.5
    ui.screenshot("palette_search_visc")
    ui.key("ESC")
    yield 0.4
    check("Esc closes the palette", registry.draw_handle("palette") is None)

    ui.centre()
    yield 0.2
    ui.win.event_simulate(type="RIGHTMOUSE", value="PRESS", x=ui.x, y=ui.y, alt=True)
    yield 0.5
    check("Alt+Right-click opens the wheel", registry.draw_handle("wheel") is not None)
    ui.screenshot("wheel_open")
    ui.win.event_simulate(type="RIGHTMOUSE", value="RELEASE", x=ui.x, y=ui.y, alt=True)
    yield 0.3
    if registry.draw_handle("wheel") is not None:
        ui.key("ESC")
        yield 0.3
    check("the wheel closes", registry.draw_handle("wheel") is None)

    with ui.ctx():
        result = bpy.ops.wm.call_panel(name="FLUIDSIM_PT_header_popover", keep_open=False)
    yield 0.6
    check("header popover opens", result == {"INTERFACE"}, sorted(result))
    ui.screenshot("header_popover", None)
    ui.move(ui.x + 400, ui.y - 300)
    yield 0.3
    ui.key("ESC")
    yield 0.3

    if ui.props is not None:
        space = ui.props.spaces.active
        for tab in ("SCENE", "PHYSICS"):
            space.context = tab
            yield 0.5
            ui.screenshot(f"dam_break_{tab.lower()}_tab", ui.props)


def phase_bake(ui):
    scene = bpy.context.scene
    fs = scene.fluidsim
    domain = domain_settings(scene)
    controls = mod("blender.deck.controls")
    gizmos = mod("blender.deck.gizmos")
    palette = mod("blender.deck.palette")
    production_bake = mod("blender.production_bake")
    bridge = mod("blender.scene_bridge")

    if bridge.session().owns_scene(scene):
        bridge.release_session(scene)
    cache = os.path.join(OUT_DIR, "cache")
    shutil.rmtree(cache, ignore_errors=True)
    fs.cache_path = cache
    fs.cache_compress = False
    fs.checkpoint_interval = 1
    fs.bake_start = int(scene.frame_start)
    fs.bake_end = int(scene.frame_start) + 1000
    domain.whitewater_enabled = False
    domain.surface_enabled = False
    other = bpy.data.scenes.new("Deck Defects Other Scene")

    check("Bake slot offers Bake before baking", controls.bake_control(scene) is controls.BAKE,
          controls.bake_control(scene).label)
    with ui.ctx():
        result = bpy.ops.fluidsim.bake_start("EXEC_DEFAULT", restart_existing=False)
    check("real modal Production Bake started", result == {"RUNNING_MODAL"}, sorted(result))
    if result != {"RUNNING_MODAL"}:
        return
    deadline = time.monotonic() + 60.0
    while (not production_bake.active_bakes()
           or production_bake.active_bakes()[0][1].status.value != "RUNNING"):
        if time.monotonic() > deadline:
            check("bake reaches RUNNING", False)
            return
        yield 0.1
    yield 1.0

    check("Bake slot shows progress while baking",
          controls.bake_control(scene) is controls.BAKE_PROGRESS)
    with ui.ctx():
        check("gizmo edits held during the bake", gizmos.bake_holds_edits(bpy.context))
    ok, why = poll(ui, "fluidsim.add_domain")
    check("Add Fluid Domain greyed during the bake", not ok and bool(why), why)
    entry = next((e for e in palette.setting_entries(scene)
                  if e.editable and e.value_type == "BOOLEAN" and e.target == "show_whitewater"),
                 None)
    check("Show Whitewater is editable in the palette", entry is not None)
    if entry is not None:
        message = palette.commit_inline(scene, entry, "")
        check("palette write during a bake says it is held",
              message == mod("blender.pending_settings")._BAKE_HOLD_MESSAGE, message)
        palette.commit_inline(scene, entry, "")  # toggle back
    ui.centre()
    yield 0.4
    ui.screenshot("bake_running_bar")

    ui.win.scene = other
    yield 0.3
    with ui.ctx():
        result = bpy.ops.fluidsim.bake_cancel("EXEC_DEFAULT")
    yield 0.3
    check("Cancel from another scene leaves the bake running",
          result == {"CANCELLED"} and production_bake.bake_is_active(scene), sorted(result))
    ui.win.scene = scene
    yield 0.3
    with ui.ctx():
        result = bpy.ops.fluidsim.bake_cancel("EXEC_DEFAULT")
    check("Cancel from the baking scene is accepted", result == {"FINISHED"}, sorted(result))
    deadline = time.monotonic() + 60.0
    while production_bake.active_bakes():
        if time.monotonic() > deadline:
            check("bake stops after Cancel", False)
            return
        yield 0.2
    check("bake stops after Cancel", True)
    yield 0.5
    slot = controls.bake_control(scene)
    check("Bake slot offers Resume or Restart after a cancelled bake",
          slot in (controls.BAKE_RESUME, controls.BAKE_RESTART), slot.label)
    ui.screenshot("bake_cancelled_bar")
    bpy.data.scenes.remove(other)


def workflow():
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    yield 2.0
    ui = UI()
    ui.centre()
    yield 0.1
    ui.key("ESC")  # dismiss the splash
    yield 0.3
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in os.listdir(OUT_DIR):
        if name.endswith(".png"):
            os.remove(os.path.join(OUT_DIR, name))
    prefs = bpy.context.preferences.addons[ADDON_NAME].preferences
    prefs.deck_mode = "ON"
    bpy.context.scene.fluidsim.deck_shown = True

    yield from phase_startup(ui)
    yield from open_demo(ui)
    for phase in (phase_session, phase_palette, phase_surfaces, phase_bake):
        try:
            yield from phase(ui)
        except Exception:  # noqa: BLE001 - one broken phase must not hide the others
            error = traceback.format_exc()
            report.setdefault("errors", {})[phase.__name__] = error
            check(f"{phase.__name__} ran without an exception", False, error.splitlines()[-1])
            log(error)
        ui.refresh()


def run():
    steps = workflow()
    started = time.monotonic()

    def finish():
        report["passed"] = bool(checks) and all(checks.values())
        report["elapsed_seconds"] = round(time.monotonic() - started, 1)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        failed = [name for name, ok in checks.items() if not ok]
        print(PASS_MARKER if report["passed"] else f"{FAIL_MARKER} {failed}", flush=True)
        shutil.rmtree(os.path.join(OUT_DIR, "cache"), ignore_errors=True)
        if os.environ.get("FLUIDSIM_DEFECTS_KEEP_OPEN") != "1":
            bpy.ops.wm.quit_blender()

    def tick():
        try:
            if time.monotonic() - started > OVERALL_TIMEOUT_SECONDS:
                check("finished_within_timeout", False)
                finish()
                return None
            return next(steps)
        except StopIteration:
            finish()
        except Exception:  # noqa: BLE001
            report["error"] = traceback.format_exc()
            check("no_exception", False, report["error"].splitlines()[-1])
            log(report["error"])
            finish()
        return None

    bpy.app.timers.register(tick, first_interval=1.0, persistent=True)


if bpy.app.background:
    print(f"{FAIL_MARKER} needs a GUI Blender (no --background)", flush=True)
    sys.exit(1)
run()
