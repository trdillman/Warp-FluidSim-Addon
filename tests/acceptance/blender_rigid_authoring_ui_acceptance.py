"""Real-GUI acceptance: an artist authors a Dynamic collider scene by hand.

Run with a real (non-background) Blender and simulated input events:
    blender --factory-startup --enable-event-simulate \
        --python tests/acceptance/blender_rigid_authoring_ui_acceptance.py

Every authoring step is driven through Blender's own event loop with
``Window.event_simulate`` -- key presses, mouse clicks and typed text land on
the same keymaps, buttons and modal operators an artist uses:

1. Shift+D, Z, "3", Enter duplicates the startup cube upwards; Shift+click
   adds the cube back to the selection.
2. Ctrl+K opens the command palette; typing "guided setup" + Enter runs Run
   Guided Setup and Enter confirms its dialog (domain + two liquid sources).
3. A click selects the cube alone; in Object Properties a click on "Set
   Selected Meshes as Colliders" tags it, a click on Dynamic turns two-way
   coupling on. The panel must then draw Density, Drag, Initial Velocity and
   the Lock toggles and stop drawing Transfer Velocity.
4. Clicks + typed text set Density 400 and Initial Velocity X 1; clicks on the
   Rotation X/Y lock toggles lock those axes.
5. I over Location keyframes the body; Space must be refused with the
   Dynamic collider "is animated" message. Alt+I deletes the key, and the
   emptied action left behind must no longer block Play.
6. Space plays the live simulation: the body gets an owned pose track, moves
   and keeps its locked rotations. Shift+Left (rewind) retires the track,
   restores the authored pose, and a second Play/rewind cycle does not crash.
7. Ctrl+K "water wheel" + Enter loads the Water Wheel demo from the gallery
   catalogue into a New Scene; Space turns the wheel about its axle.

Widgets are found the way a user would find them -- by hovering: the test
sweeps the Properties editor with the mouse and asks Blender what is under
the cursor (Ctrl+Shift+C copies a property's data path, Ctrl+C an operator
button's Python command). The OS clipboard is restored before Blender quits.

Writes artifacts/rigid_authoring_ui_report.json and prints
RIGID-AUTHORING-UI-ACCEPTANCE-PASS or -FAIL.
"""

import json
import math
import os
import sys
import time
import traceback

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

REPORT_PATH = os.environ.get(
    "FLUIDSIM_RIGID_UI_REPORT",
    os.path.join(REPO_ROOT, "artifacts", "rigid_authoring_ui_report.json"))
PASS_MARKER = "RIGID-AUTHORING-UI-ACCEPTANCE-PASS"
FAIL_MARKER = "RIGID-AUTHORING-UI-ACCEPTANCE-FAIL"
OVERALL_TIMEOUT_SECONDS = 480.0
LIVE_FRAMES = 12
WHEEL_FRAMES = 16
OWNED_ACTION_TAG = "fluidsim_rigid_poses"
RETIRED_ACTION_NAME = "FluidSim Dynamic (cleared)"
SENTINEL = "<fluidsim-ui-probe>"

KEY_TYPES = {
    **{str(d): name for d, name in enumerate(
        ("ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"))},
    " ": "SPACE", ".": "PERIOD", "-": "MINUS",
}

checks = {}
report = {"checks": checks, "steps": []}


def log(*parts):
    print("RIGID-UI", *parts, flush=True)


def check(name, ok, detail=None):
    checks[name] = bool(ok)
    if detail is not None:
        report.setdefault("details", {})[name] = detail
    log("check", name, "OK" if ok else "FAILED", "" if detail is None else detail)
    return bool(ok)


# Input ------------------------------------------------------------------------

class Input:
    """Simulated mouse/keyboard on the first window; every event carries x/y so
    it lands in the area under the cursor."""

    def __init__(self, window):
        self.win = window
        self.x = 0
        self.y = 0

    def move(self, x, y):
        self.x, self.y = int(x), int(y)
        self.win.event_simulate(type="MOUSEMOVE", value="NOTHING", x=self.x, y=self.y)

    def key(self, type_, unicode=None, **mods):
        extra = {"unicode": unicode} if unicode else {}
        self.win.event_simulate(type=type_, value="PRESS", x=self.x, y=self.y, **extra, **mods)
        self.win.event_simulate(type=type_, value="RELEASE", x=self.x, y=self.y, **mods)

    def click(self, x=None, y=None, **mods):
        if x is not None:
            self.move(x, y)
        self.win.event_simulate(type="LEFTMOUSE", value="PRESS", x=self.x, y=self.y, **mods)
        self.win.event_simulate(type="LEFTMOUSE", value="RELEASE", x=self.x, y=self.y, **mods)

    def text(self, value):
        for char in value:
            self.key(KEY_TYPES.get(char, char.upper()), unicode=char)

    def wheel(self, down=True, count=1):
        for _ in range(count):
            self.win.event_simulate(
                type="WHEELDOWNMOUSE" if down else "WHEELUPMOUSE", value="PRESS",
                x=self.x, y=self.y)


def area_of(window, kind):
    return next(a for a in window.screen.areas if a.type == kind)


def window_region(area):
    return next(r for r in area.regions if r.type == "WINDOW")


def scan_widgets(inp, region, x_fracs=(0.42, 0.8), step=8):
    """Hover down columns of the region (both halves of a two-column property
    grid); map what is under the cursor to its window position.
    Returns {data path or python command: (x, y)}."""
    wm = bpy.context.window_manager
    hits = {}
    for x_frac in x_fracs:
        x = region.x + int(region.width * x_frac)
        for y in range(region.y + region.height - 4, region.y + 4, -step):
            inp.move(x, y)
            yield 0.02
            for mods in ({"ctrl": True, "shift": True}, {"ctrl": True}):
                wm.clipboard = SENTINEL
                inp.key("C", **mods)
                yield 0.02
                text = wm.clipboard
                if text != SENTINEL and (text.startswith("bpy.ops.") or mods.get("shift")):
                    hits.setdefault(text, (x, y))
    return hits


def scan_row(inp, region, y, step=4):
    """Hover across one row (e.g. a row of toggles). Returns each widget's
    horizontal centre."""
    wm = bpy.context.window_manager
    spans = {}
    for x in range(region.x + 8, region.x + region.width - 8, step):
        inp.move(x, y)
        yield 0.02
        wm.clipboard = SENTINEL
        inp.key("C", ctrl=True, shift=True)
        yield 0.02
        if wm.clipboard != SENTINEL:
            low, high = spans.get(wm.clipboard, (x, x))
            spans[wm.clipboard] = (min(low, x), max(high, x))
    return {path: ((low + high) // 2, y) for path, (low, high) in spans.items()}


def find_widgets(inp, area, wanted, scrolls=6):
    """Scan the Properties editor from the top, scrolling down until every
    wanted path is under the cursor somewhere. Returns the last scan."""
    region = window_region(area)
    inp.move(region.x + region.width // 2, region.y + region.height // 2)
    inp.key("HOME")  # scroll the editor back to its top
    yield 0.2
    seen = {}
    for _ in range(scrolls + 1):
        hits = yield from scan_widgets(inp, region)
        seen = hits
        if all(path in hits for path in wanted):
            return hits
        inp.move(region.x + region.width // 2, region.y + region.height // 2)
        inp.wheel(count=8)
        yield 0.2
    return seen


def edit_number(inp, pos, value):
    """Click a number field (enters text editing), select all, type, Enter."""
    inp.click(*pos)
    yield 0.15
    inp.key("A", ctrl=True)
    inp.text(value)
    inp.key("RET")
    yield 0.2


def object_screen_position(window, obj, local=None):
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    area = area_of(window, "VIEW_3D")
    region = window_region(area)
    world = obj.matrix_world @ local if local is not None else obj.matrix_world.translation
    co = location_3d_to_region_2d(region, area.spaces.active.region_3d, world)
    return region.x + int(co.x), region.y + int(co.y)


def click_select(inp, window, obj, attempts=8):
    """Alt+A, then click the object -- at its centre, then on its edges (a
    wireframe-drawn object is only picked on its wires). True once ``obj`` is
    the only selected object and active."""
    mesh = obj.data
    targets = [None] + [
        (mesh.vertices[a].co + mesh.vertices[b].co) / 2.0
        for a, b in (edge.vertices for edge in mesh.edges)]
    for local in targets[:attempts]:
        inp.move(*object_screen_position(window, obj))
        inp.key("A", alt=True)
        yield 0.15
        inp.click(*object_screen_position(window, obj, local))
        yield 0.3
        if (bpy.context.selected_objects == [obj]
                and bpy.context.view_layer.objects.active == obj):
            return True
    return False


def wait_until(predicate, timeout, interval=0.1):
    start = time.monotonic()
    while not predicate():
        if time.monotonic() - start > timeout:
            return False
        yield interval
    return True


def owned_action(obj):
    animation = obj.animation_data
    action = animation.action if animation is not None else None
    if action is not None and action.get(OWNED_ACTION_TAG, False):
        return action
    return None


def key_frames(action):
    """Frames of every keyframe in a layered (Blender 4.4+) action."""
    frames = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fcurve in bag.fcurves:
                    frames.extend(point.co.x for point in fcurve.keyframe_points)
    return frames


def key_count(action):
    return len(key_frames(action))


def action_retired(pointer):
    """The pose action Reset replaced is renamed for removal or already freed."""
    for action in bpy.data.actions:
        if action.as_pointer() == pointer:
            return action.name.startswith(RETIRED_ACTION_NAME)
    return True


def track_restarted(obj, start):
    """After a rewind Reset the owned track holds only the frame-start pose."""
    action = owned_action(obj)
    return action is None or max(key_frames(action), default=start) <= start


# Workflow -----------------------------------------------------------------------

def workflow():
    wm = bpy.context.window_manager
    win = wm.windows[0]
    inp = Input(win)
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    yield 2.0

    v3d = area_of(win, "VIEW_3D")
    props = area_of(win, "PROPERTIES")
    view = window_region(v3d)
    centre = (view.x + view.width // 2, view.y + view.height // 2)
    inp.move(view.x + 300, view.y + 300)
    yield 0.1
    inp.key("ESC")  # dismiss the splash
    yield 0.3
    props.spaces.active.context = "OBJECT"
    yield 0.2

    scene = bpy.context.scene
    cube = bpy.data.objects["Cube"]
    check("startup_cube_active", bpy.context.view_layer.objects.active == cube)

    # 1. Duplicate the cube upwards (Shift+D, Z, 3, Enter), then Shift+click
    #    the original so both are selected and the original is active.
    report["steps"].append("duplicate")
    inp.move(*centre)
    inp.key("D", shift=True)
    yield 0.3
    inp.key("Z")
    inp.text("3")
    inp.key("RET")
    yield 0.4
    emitter = bpy.data.objects.get("Cube.001")
    check("duplicated_up", emitter is not None
          and abs(emitter.location.z - 3.0) < 1e-4,
          None if emitter is None else tuple(round(v, 4) for v in emitter.location))
    if emitter is None:
        return
    inp.click(*object_screen_position(win, cube), shift=True)
    yield 0.4
    check("both_selected", cube.select_get() and emitter.select_get()
          and bpy.context.view_layer.objects.active == cube)

    # 2. Ctrl+K palette -> Run Guided Setup -> Enter confirms the dialog.
    report["steps"].append("guided_setup")
    inp.move(*centre)
    inp.key("K", ctrl=True)
    yield 0.4
    inp.text("guided setup")
    yield 0.2
    inp.key("RET")
    yield 0.6
    inp.key("RET")  # confirm the Run Guided Setup dialog
    ok = yield from wait_until(
        lambda: getattr(scene.fluidsim, "domain_object", None) is not None, 10.0)
    check("guided_setup_domain", ok, scene.fluidsim.status)
    check("guided_setup_sources",
          cube.fluidsim.role == "SOURCE" and emitter.fluidsim.role == "SOURCE",
          (cube.fluidsim.role, emitter.fluidsim.role))
    if not ok:
        return

    # 3. Select the cube alone, tag it as a collider, turn Dynamic on.
    report["steps"].append("tag_collider")
    picked = yield from click_select(inp, win, cube)
    check("cube_alone_selected", picked,
          [o.name for o in bpy.context.selected_objects])
    tag_cmd = "bpy.ops.fluidsim.tag_collider()"
    hits = yield from find_widgets(inp, props, [tag_cmd])
    check("tag_collider_button_drawn", tag_cmd in hits, sorted(hits)[:40])
    if tag_cmd not in hits:
        return
    inp.click(*hits[tag_cmd])
    yield 0.4
    check("tagged_collider", cube.fluidsim.role == "COLLIDER", cube.fluidsim.role)

    hits = yield from find_widgets(inp, props, ["fluidsim.collider_dynamic"])
    check("static_panel_draws_transfer_velocity",
          "fluidsim.transfer_velocity" in hits
          and "fluidsim.collider_density" not in hits)
    if "fluidsim.collider_dynamic" not in hits:
        check("dynamic_toggle_drawn", False, sorted(hits))
        return
    inp.click(*hits["fluidsim.collider_dynamic"])
    yield 0.4
    check("dynamic_on", cube.fluidsim.collider_dynamic)

    dynamic_paths = ["fluidsim.collider_density", "fluidsim.collider_drag",
                     "fluidsim.collider_initial_velocity[0]"]
    hits = yield from find_widgets(inp, props, dynamic_paths)
    check("dynamic_panel_draws_rigid_settings",
          all(path in hits for path in dynamic_paths), sorted(hits))
    check("dynamic_panel_hides_transfer_velocity",
          "fluidsim.transfer_velocity" not in hits)
    if not all(path in hits for path in dynamic_paths):
        return

    # 4. Type values into the fields; click the rotation lock toggles.
    report["steps"].append("edit_settings")
    yield from edit_number(inp, hits["fluidsim.collider_density"], "400")
    check("density_typed", abs(cube.fluidsim.collider_density - 400.0) < 1e-3,
          cube.fluidsim.collider_density)
    hits = yield from find_widgets(inp, props, ["fluidsim.collider_initial_velocity[0]"])
    yield from edit_number(inp, hits["fluidsim.collider_initial_velocity[0]"], "1")
    check("initial_velocity_typed",
          abs(cube.fluidsim.collider_initial_velocity[0] - 1.0) < 1e-4,
          tuple(cube.fluidsim.collider_initial_velocity))

    region = window_region(props)
    lock_hits = {}
    for _attempt in range(2):
        column = yield from find_widgets(inp, props, ["fluidsim.collider_drag"])
        rows = sorted({pos[1] for path, pos in column.items()
                       if path.startswith("fluidsim.collider_lock_axes")})
        for row_y in rows:
            lock_hits.update((yield from scan_row(inp, region, row_y)))
        if "fluidsim.collider_lock_axes[3]" in lock_hits:
            break
    lock_paths = [f"fluidsim.collider_lock_axes[{i}]" for i in range(6)]
    check("lock_toggles_drawn", all(p in lock_hits for p in lock_paths), sorted(lock_hits))
    for index in (3, 4):
        path = lock_paths[index]
        if path in lock_hits:
            inp.move(*lock_hits[path])
            yield 0.2  # let the toggle take the hover before pressing
            inp.click()
            yield 0.6  # outside the double-click window
            report.setdefault("lock_clicks", []).append(
                (path, lock_hits[path], tuple(bool(v) for v in cube.fluidsim.collider_lock_axes)))
    locks = tuple(bool(v) for v in cube.fluidsim.collider_lock_axes)
    check("rotation_xy_locked", locks == (False, False, False, True, True, False), locks)

    # 5. Keyframing the body is refused at Play; Ctrl+Z removes the conflict.
    report["steps"].append("refusal")
    hits = yield from find_widgets(inp, props, ["location[0]"])
    inp.move(*hits["location[0]"])
    yield 0.1
    inp.key("I")
    yield 0.4
    check("body_keyframed", cube.animation_data is not None
          and cube.animation_data.action is not None)
    start = scene.frame_start
    inp.move(*centre)
    inp.key("SPACE")
    refused = yield from wait_until(
        lambda: "is animated" in str(scene.fluidsim.status), 15.0)
    check("play_refused_while_animated", refused, scene.fluidsim.status)
    inp.key("SPACE")
    yield 0.3
    inp.key("LEFT_ARROW", shift=True)
    yield 0.3
    hits = yield from find_widgets(inp, props, ["location[0]"])
    inp.move(*hits["location[0]"])
    yield 0.1
    inp.key("I", alt=True)  # Delete Keyframe over the keyed field
    yield 0.4
    animation = cube.animation_data
    report["after_delete_keyframe"] = {
        "action": None if animation is None or animation.action is None
        else animation.action.name,
        "keys": 0 if animation is None or animation.action is None
        else key_count(animation.action)}
    check("delete_keyframe_removed_keys", report["after_delete_keyframe"]["keys"] == 0,
          report["after_delete_keyframe"])
    check("authoring_kept",
          cube.fluidsim.collider_dynamic
          and abs(cube.fluidsim.collider_density - 400.0) < 1e-3,
          (cube.fluidsim.collider_dynamic, cube.fluidsim.collider_density))

    # 6. Play live; the body moves on an owned pose track. Rewind clears it.
    report["steps"].append("live_play")
    rest = cube.matrix_world.copy()
    inp.move(*centre)
    inp.key("SPACE")
    played = yield from wait_until(
        lambda: scene.frame_current >= start + LIVE_FRAMES, 150.0, 0.2)
    inp.key("SPACE")
    yield 0.5
    check("live_frames_played", played,
          {"frame": scene.frame_current, "status": scene.fluidsim.status})
    action = owned_action(cube)
    check("owned_pose_track", action is not None and key_count(action) > 0,
          None if action is None else key_count(action))
    moved = (cube.matrix_world.translation - rest.translation).length
    check("body_moved", moved > 1e-3, moved)
    rot = cube.matrix_world.to_euler()
    check("locked_rotation_held", abs(rot.x) < 1e-4 and abs(rot.y) < 1e-4,
          (rot.x, rot.y, rot.z))
    owned_pointer = action.as_pointer() if action is not None else 0

    inp.move(*centre)
    inp.key("LEFT_ARROW", shift=True)
    yield 1.0
    report["after_rewind"] = {"keys": len(key_frames(owned_action(cube) or action)),
                              "frame": scene.frame_current,
                              "status": scene.fluidsim.status}
    check("rewind_restarted_track", track_restarted(cube, start), report["after_rewind"])
    restored = max(abs(a - b) for row_a, row_b in zip(cube.matrix_world, rest)
                   for a, b in zip(row_a, row_b))
    check("rewind_restored_pose", restored < 1e-5, restored)
    check("rewind_retired_old_action", owned_pointer and action_retired(owned_pointer))
    # The crash path: play again and rewind again on the same session.
    inp.key("SPACE")
    yield from wait_until(lambda: scene.frame_current >= start + 3, 60.0, 0.2)
    inp.key("SPACE")
    yield 0.4
    inp.key("LEFT_ARROW", shift=True)
    yield 1.0
    check("second_play_rewind_survived", scene.frame_current == start
          and track_restarted(cube, start))

    # 7. Load the Water Wheel demo from the palette's gallery entries and play.
    report["steps"].append("water_wheel")
    inp.move(*centre)
    inp.key("K", ctrl=True)
    yield 0.4
    inp.text("water wheel")
    yield 0.2
    inp.key("RET")
    loaded = yield from wait_until(
        lambda: win.scene.objects.get("Water Wheel") is not None, 30.0, 0.2)
    check("water_wheel_loaded", loaded and win.scene != scene, win.scene.name)
    if not loaded:
        return
    demo = win.scene
    wheel = demo.objects["Water Wheel"]
    axis_before = wheel.matrix_world.col[0].xyz.copy()
    origin = wheel.matrix_world.translation.copy()
    start = demo.frame_start
    inp.move(*centre)
    inp.key("SPACE")
    played = yield from wait_until(
        lambda: demo.frame_current >= start + WHEEL_FRAMES, 150.0, 0.2)
    inp.key("SPACE")
    yield 0.5
    check("wheel_frames_played", played,
          {"frame": demo.frame_current, "status": demo.fluidsim.status})
    axis_after = wheel.matrix_world.col[0].xyz
    turned = math.acos(max(-1.0, min(1.0, axis_before.normalized().dot(axis_after.normalized()))))
    check("wheel_turned", turned > 0.05, turned)
    drift = (wheel.matrix_world.translation - origin).length
    check("wheel_axle_held", drift < 1e-4, drift)
    inp.key("LEFT_ARROW", shift=True)
    yield 0.8
    report["wheel_after_rewind"] = {
        "keys": key_frames(owned_action(wheel)) if owned_action(wheel) else [],
        "frame": demo.frame_current,
        "status": demo.fluidsim.status,
        "axis_dot": wheel.matrix_world.col[0].xyz.normalized().dot(axis_before.normalized())}
    check("wheel_rewind_restarted", track_restarted(wheel, start)
          and report["wheel_after_rewind"]["axis_dot"] > 1.0 - 1e-6,
          report["wheel_after_rewind"])


def run():
    wm = bpy.context.window_manager
    saved_clipboard = wm.clipboard
    steps = workflow()
    started = time.monotonic()

    def finish():
        try:
            wm.clipboard = saved_clipboard
        except Exception:  # noqa: BLE001
            pass
        report["passed"] = bool(checks) and all(checks.values())
        report["elapsed_seconds"] = round(time.monotonic() - started, 1)
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        failed = [name for name, ok in checks.items() if not ok]
        print(PASS_MARKER if report["passed"] else f"{FAIL_MARKER} {failed}", flush=True)
        bpy.ops.wm.quit_blender()

    def tick():
        try:
            if time.monotonic() - started > OVERALL_TIMEOUT_SECONDS:
                check("finished_within_timeout", False, report["steps"][-1:])
                finish()
                return None
            return next(steps)
        except StopIteration:
            finish()
        except Exception:  # noqa: BLE001
            report["error"] = traceback.format_exc()
            check("no_exception", False, report["error"].splitlines()[-1])
            finish()
        return None

    bpy.app.timers.register(tick, first_interval=1.0, persistent=True)


if bpy.app.background:
    print(f"{FAIL_MARKER} needs a GUI Blender (no --background)", flush=True)
    sys.exit(1)
run()
