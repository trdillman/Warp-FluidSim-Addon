"""Real-GUI acceptance: every Liquid Deck gizmo updates the viewport while it is dragged.

Run with a real (non-background) Blender and simulated input events:
    blender --factory-startup --enable-event-simulate \
        --python tests/acceptance/blender_deck_gizmo_realtime_acceptance.py

The scene holds one of every object that gets Deck gizmos:

- the domain (six face handles),
- emitters of each object type and Fill Target the gizmos treat differently: a
  mesh in Sampling Multiplier mode, a mesh in Legacy Absolute mode, a mesh in
  Inherit mode (velocity arrow only, no fill dial) and a point cloud,
- colliders: a static mesh (friction dial) and a Dynamic mesh (friction dial +
  Dynamic toggle).

Each handle is pressed with a simulated left click and dragged in steps through
Blender's own event loop. After every step, *before the button is released*,
the test records the bound value and how many times the 3D View redrew its
scene (a POST_VIEW draw handler) and saves a screenshot of the 3D View. A
gizmo passes when the value moved and the viewport redrew on every step of the
drag, i.e. the viewport follows the gizmo in real time, not only on release.
Escape-cancel is checked on the domain: the box must return to its start.

Writes artifacts/deck_gizmo_realtime/report.json plus the screenshots and prints
DECK-GIZMO-REALTIME-PASS or -FAIL.
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

from tools.repo_addon_loader import ADDON_NAME, enable_repo_addon  # noqa: E402

OUT_DIR = os.environ.get("FLUIDSIM_GIZMO_OUT",
                         os.path.join(REPO_ROOT, "artifacts", "deck_gizmo_realtime"))
SCREENSHOTS = os.environ.get("FLUIDSIM_GIZMO_SCREENSHOTS", "1") == "1"
PASS_MARKER = "DECK-GIZMO-REALTIME-PASS"
FAIL_MARKER = "DECK-GIZMO-REALTIME-FAIL"
OVERALL_TIMEOUT_SECONDS = 300.0
STEPS = 5
STEP_PIXELS = 18
STEP_WAIT = 0.25

checks = {}
report = {"checks": checks, "drags": []}
_GIZMOS = {}  # group bl_idname -> {gizmo name: (matrix_world, hidden)}
_REDRAWS = [0]


def log(*parts):
    print("DECK-GIZMO", *parts, flush=True)


def check(name, ok, detail=None):
    checks[name] = bool(ok)
    if detail is not None:
        report.setdefault("details", {})[name] = detail
    log("check", name, "OK" if ok else "FAILED", "" if detail is None else detail)
    return bool(ok)


# Input / view helpers --------------------------------------------------------------

class Input:
    def __init__(self, window):
        self.win = window
        self.x = self.y = 0

    def move(self, x, y):
        self.x, self.y = int(x), int(y)
        self.win.event_simulate(type="MOUSEMOVE", value="NOTHING", x=self.x, y=self.y)

    def key(self, type_, **mods):
        self.win.event_simulate(type=type_, value="PRESS", x=self.x, y=self.y, **mods)
        self.win.event_simulate(type=type_, value="RELEASE", x=self.x, y=self.y, **mods)

    def press(self):
        self.win.event_simulate(type="LEFTMOUSE", value="PRESS", x=self.x, y=self.y)

    def release(self):
        self.win.event_simulate(type="LEFTMOUSE", value="RELEASE", x=self.x, y=self.y)


def area_of(window, kind):
    return next(a for a in window.screen.areas if a.type == kind)


def window_region(area):
    return next(r for r in area.regions if r.type == "WINDOW")


def to_window(area, point):
    """Window pixel of world ``point`` in the 3D View, or None when behind the view."""
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    region = window_region(area)
    co = location_3d_to_region_2d(region, area.spaces.active.region_3d, point)
    return None if co is None else (region.x + co.x, region.y + co.y)


def record_gizmos():
    """Wrap each Deck group's draw_prepare to remember where its gizmos are drawn."""
    from importlib import import_module

    gizmos = import_module(f"{ADDON_NAME}.blender.deck.gizmos")
    # Patching a registered class's callback crashes Blender: unregister first.
    for group in gizmos.GROUP_CLASSES:
        bpy.utils.unregister_class(group)
    for group in gizmos.GROUP_CLASSES:
        group.draw_prepare = _recording(group.draw_prepare, group.bl_idname)
        bpy.utils.register_class(group)


def _recording(original, name):
    """``draw_prepare`` that also records gizmo placement (Blender wants exactly 2 args)."""
    def draw_prepare(self, context):
        original(self, context)
        entry = {}
        for attr in ("velocity", "dial", "toggle"):
            gz = getattr(self, attr, None)
            if gz is not None:
                entry[attr] = (gz.matrix_world.copy(), bool(gz.hide))
        for gz in getattr(self, "faces", ()):
            entry[f"face{gz.axis}{'+' if gz.side > 0 else '-'}"] = (
                gz.matrix_world.copy(), bool(gz.hide))
        _GIZMOS[name] = entry

    return draw_prepare


def count_redraw():
    _REDRAWS[0] += 1


def screenshot(window, area, name):
    if not SCREENSHOTS:
        return None
    path = os.path.join(OUT_DIR, name + ".png")
    try:
        with bpy.context.temp_override(window=window, area=area):
            bpy.ops.screen.screenshot_area(filepath=path)
    except Exception as exc:  # noqa: BLE001 - a missing screenshot is not a gizmo failure
        log("screenshot failed", name, repr(exc))
        return None
    return os.path.relpath(path, OUT_DIR)


# Scene ----------------------------------------------------------------------------

def make_scene(window, v3d):
    from mathutils import Vector

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    scene = bpy.context.scene

    def mesh(name, location, size, primitive="cube"):
        with bpy.context.temp_override(window=window, area=v3d):
            if primitive == "cube":
                bpy.ops.mesh.primitive_cube_add(size=size, location=location)
            else:
                bpy.ops.mesh.primitive_ico_sphere_add(radius=size / 2, location=location)
        obj = bpy.context.active_object
        obj.name = name
        return obj

    def run_on(obj, op):
        for other in scene.objects:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        with bpy.context.temp_override(window=window, area=v3d, active_object=obj,
                                       selected_objects=[obj]):
            op()

    domain = mesh("Domain", (0.0, 0.0, 1.5), 1.0)
    domain.scale = (5.0, 4.0, 3.0)
    run_on(domain, bpy.ops.fluidsim.use_active_as_domain)

    emitters = {}
    for name, x, mode in (("Emitter Mesh Multiplier", -1.6, "MULTIPLIER"),
                          ("Emitter Mesh Legacy", -0.5, "LEGACY_ABSOLUTE"),
                          ("Emitter Mesh Inherit", 0.6, "INHERIT")):
        obj = mesh(name, (x, -0.8, 2.2), 0.5, "ico")
        run_on(obj, bpy.ops.fluidsim.tag_liquid_emitter)
        obj.fluidsim.source_sampling_mode = mode
        obj.fluidsim.initial_velocity = (0.0, 0.0, -1.0)
        emitters[name] = obj

    with bpy.context.temp_override(window=window, area=v3d):
        bpy.ops.object.pointcloud_random_add(location=(1.7, -0.8, 2.2))
    cloud = bpy.context.active_object
    cloud.name = "Emitter Point Cloud"
    cloud.scale = (0.3, 0.3, 0.3)
    run_on(cloud, bpy.ops.fluidsim.tag_liquid_emitter)
    cloud.fluidsim.source_sampling_mode = "MULTIPLIER"
    cloud.fluidsim.initial_velocity = (0.0, 0.0, -1.0)
    emitters[cloud.name] = cloud

    colliders = {}
    for name, x, dynamic in (("Collider Static", -1.0, False), ("Collider Dynamic", 1.0, True)):
        obj = mesh(name, (x, 0.8, 0.6), 0.6)
        run_on(obj, bpy.ops.fluidsim.tag_collider)
        obj.fluidsim.collider_friction = 0.3
        if dynamic:
            obj.fluidsim.collider_dynamic = True
        colliders[name] = obj
    del Vector
    return domain, emitters, colliders


def focus(window, v3d, obj, distance):
    """Make ``obj`` the only selected, active object and frame it."""
    from mathutils import Euler

    scene = bpy.context.scene
    for other in scene.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    r3d = v3d.spaces.active.region_3d
    r3d.view_perspective = "PERSP"
    r3d.view_rotation = Euler((math.radians(62), 0.0, math.radians(35))).to_quaternion()
    r3d.view_location = obj.matrix_world.translation
    r3d.view_distance = distance
    v3d.tag_redraw()


# Drag driver ----------------------------------------------------------------------

def handle_point(area, group, name, local):
    entry = _GIZMOS.get(group, {}).get(name)
    if entry is None:
        return None
    matrix, hidden = entry
    if hidden:
        return None
    from mathutils import Vector

    return to_window(area, matrix @ Vector(local))


def drag(inp, window, area, label, group, name, local, direction, read, *, cancel=False):
    """Press on a handle, move it STEPS times by ``direction`` pixels, read the
    bound value and viewport redraws after every move, then release (or Escape)."""
    from mathutils import Vector

    shots = []
    start = handle_point(area, group, name, local)
    if start is None:
        check(f"{label}: handle drawn", False, sorted(_GIZMOS.get(group, {})))
        return None
    before = read()
    inp.move(*start)  # hover so the gizmo map highlights the handle
    yield 0.3
    inp.press()
    yield 0.2
    d = Vector(direction).normalized() * STEP_PIXELS
    steps = []
    for i in range(1, STEPS + 1):
        redraws = _REDRAWS[0]
        inp.move(start[0] + d.x * i, start[1] + d.y * i)
        yield STEP_WAIT
        steps.append({"value": read(), "viewport_redraws": _REDRAWS[0] - redraws})
        if i in (1, STEPS // 2 + 1, STEPS):
            shots.append(screenshot(window, area, f"{label}_drag{i}"))
    if cancel:
        inp.key("ESC")
    else:
        inp.release()
    yield 0.3
    after = read()
    shots.append(screenshot(window, area, f"{label}_released"))
    record = {"label": label, "before": before, "steps": steps, "after": after,
              "cancelled": cancel, "screenshots": [s for s in shots if s]}
    report["drags"].append(record)
    log(label, json.dumps({k: record[k] for k in ("before", "after")}, default=str),
        [(s["value"], s["viewport_redraws"]) for s in steps])
    return record


def judge_realtime(label, record, *, settle=None):
    """The value moved on every drag step and the viewport redrew each time."""
    if record is None:
        return
    values = [record["before"]] + [s["value"] for s in record["steps"]]
    changing = all(values[i] != values[i - 1] for i in range(1, len(values)))
    check(f"{label}: value follows the drag (before release)", changing, values)
    check(f"{label}: viewport redraws on every drag step",
          all(s["viewport_redraws"] > 0 for s in record["steps"]),
          [s["viewport_redraws"] for s in record["steps"]])
    if settle is not None:
        check(f"{label}: released value equals the last drag value",
              record["after"] == record["steps"][-1]["value"], record["after"])


def rounded(values, digits=4):
    return tuple(round(float(v), digits) for v in values)


def workflow():
    wm = bpy.context.window_manager
    win = wm.windows[0]
    inp = Input(win)
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    yield 2.0
    v3d = area_of(win, "VIEW_3D")
    view = window_region(v3d)
    inp.move(view.x + 300, view.y + 300)
    yield 0.1
    inp.key("ESC")  # dismiss the splash
    yield 0.3

    # Give the 3D View the whole window so the screenshots show it large.
    with bpy.context.temp_override(window=win, area=v3d):
        bpy.ops.screen.screen_full_area()
    yield 0.5
    v3d = area_of(win, "VIEW_3D")
    v3d.spaces.active.overlay.show_stats = False

    prefs = bpy.context.preferences.addons[ADDON_NAME].preferences
    prefs.deck_mode = "ON"
    scene = bpy.context.scene
    scene.fluidsim.deck_shown = True
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in os.listdir(OUT_DIR):
        if name.endswith(".png"):
            os.remove(os.path.join(OUT_DIR, name))

    record_gizmos()
    handler = bpy.types.SpaceView3D.draw_handler_add(count_redraw, (), "WINDOW", "POST_VIEW")
    try:
        domain, emitters, colliders = make_scene(win, v3d)
        report["objects"] = {
            "domain": domain.name,
            "emitters": {n: [o.type, o.fluidsim.source_sampling_mode] for n, o in emitters.items()},
            "colliders": {n: bool(o.fluidsim.collider_dynamic) for n, o in colliders.items()},
        }
        yield 0.5
        yield from drive_emitters(inp, win, v3d, emitters)
        yield from drive_colliders(inp, win, v3d, colliders)
        yield from drive_domain(inp, win, v3d, domain)
    finally:
        bpy.types.SpaceView3D.draw_handler_remove(handler, "WINDOW")


def drive_emitters(inp, win, v3d, emitters):
    for name, obj in emitters.items():
        slug = name.lower().replace(" ", "_")
        focus(win, v3d, obj, 4.0)
        _GIZMOS.clear()
        yield 0.5
        entry = _GIZMOS.get("FLUIDSIM_GGT_deck_emitter", {})
        check(f"{name}: emitter gizmos shown", "velocity" in entry, sorted(entry))
        screenshot(win, v3d, f"{slug}_idle")

        # Velocity arrow: grab the head, drag up-right on screen.
        record = yield from drag(
            inp, win, v3d, f"{slug}_velocity", "FLUIDSIM_GGT_deck_emitter", "velocity",
            (0.0, 0.0, 0.95), (1.0, 0.6), lambda o=obj: rounded(o.fluidsim.initial_velocity))
        judge_realtime(f"{name} velocity arrow", record, settle=True)

        mode = obj.fluidsim.source_sampling_mode
        dial_hidden = entry.get("dial", (None, True))[1]
        if mode == "INHERIT":
            check(f"{name}: no fill dial in Inherit mode", dial_hidden)
            continue
        prop = "source_sampling_multiplier" if mode == "MULTIPLIER" else "emit_rate"
        record = yield from drag(
            inp, win, v3d, f"{slug}_fill", "FLUIDSIM_GGT_deck_emitter", "dial",
            (1.0, 0.0, 0.0), (1.0, 0.0),
            lambda o=obj, p=prop: round(float(getattr(o.fluidsim, p)), 4))
        judge_realtime(f"{name} fill dial ({prop})", record, settle=True)


def drive_colliders(inp, win, v3d, colliders):
    for name, obj in colliders.items():
        slug = name.lower().replace(" ", "_")
        focus(win, v3d, obj, 4.0)
        _GIZMOS.clear()
        yield 0.5
        entry = _GIZMOS.get("FLUIDSIM_GGT_deck_collider", {})
        check(f"{name}: collider gizmos shown", "dial" in entry, sorted(entry))
        screenshot(win, v3d, f"{slug}_idle")
        record = yield from drag(
            inp, win, v3d, f"{slug}_friction", "FLUIDSIM_GGT_deck_collider", "dial",
            (1.0, 0.0, 0.0), (1.0, 0.0),
            lambda o=obj: round(float(o.fluidsim.collider_friction), 4))
        judge_realtime(f"{name} friction dial", record, settle=True)

        dynamic = bool(obj.fluidsim.collider_dynamic)
        point = handle_point(v3d, "FLUIDSIM_GGT_deck_collider", "toggle", (1.0, 0.0, 0.0))
        check(f"{name}: Dynamic toggle drawn", point is not None, sorted(entry))
        if point is None:
            continue
        inp.move(*point)
        yield 0.3
        inp.press()
        yield 0.1
        inp.release()
        yield 0.4
        screenshot(win, v3d, f"{slug}_toggle_clicked")
        check(f"{name}: Dynamic toggle click flips collider_dynamic",
              bool(obj.fluidsim.collider_dynamic) != dynamic,
              [dynamic, bool(obj.fluidsim.collider_dynamic)])

        # A press on the toggle that turns into a drag does not flip it.
        dynamic = bool(obj.fluidsim.collider_dynamic)
        inp.move(*point)
        yield 0.3
        inp.press()
        yield 0.1
        for i in range(1, 6):
            inp.move(point[0] + 12 * i, point[1])
            yield 0.05
        inp.move(*point)  # back over the toggle before the release
        yield 0.1
        inp.release()
        yield 0.4
        check(f"{name}: a drag released over the Dynamic toggle does not flip it",
              bool(obj.fluidsim.collider_dynamic) == dynamic,
              [dynamic, bool(obj.fluidsim.collider_dynamic)])


def drive_domain(inp, win, v3d, domain):
    from mathutils import Vector

    focus(win, v3d, domain, 14.0)
    _GIZMOS.clear()
    yield 0.5
    entry = _GIZMOS.get("FLUIDSIM_GGT_deck_domain", {})
    check("Domain: six face handles shown", len(entry) == 6, sorted(entry))
    screenshot(win, v3d, "domain_idle")

    def box():
        return rounded(tuple(domain.location) + tuple(domain.scale))

    for axis, side in ((0, 1), (1, -1), (2, 1)):
        name = f"face{axis}{'+' if side > 0 else '-'}"
        if name not in entry:
            continue
        centre = entry[name][0].translation
        normal = Vector((0.0, 0.0, 0.0))
        normal[axis] = side
        a, b = to_window(v3d, centre), to_window(v3d, centre + normal)
        direction = (b[0] - a[0], b[1] - a[1])
        label = f"domain_{'xyz'[axis]}{'+' if side > 0 else '-'}"
        record = yield from drag(inp, win, v3d, label, "FLUIDSIM_GGT_deck_domain", name,
                                 (1.0, 0.0, 0.0), direction, box)
        judge_realtime(f"Domain {'XYZ'[axis]}{'+' if side > 0 else '-'} face", record, settle=True)
        if record is not None:
            # The opposite face stays where it was.
            lo = axis_extent(record["before"], axis, -side)
            hi = axis_extent(record["after"], axis, -side)
            check(f"Domain {'XYZ'[axis]}{'+' if side > 0 else '-'} face: opposite face fixed",
                  abs(lo - hi) < 1e-3, [lo, hi])

    # Escape restores the box it started from.
    name = "face0-"
    if name in entry:
        centre = entry[name][0].translation
        a = to_window(v3d, centre)
        b = to_window(v3d, centre + Vector((-1.0, 0.0, 0.0)))
        record = yield from drag(inp, win, v3d, "domain_x-_cancel", "FLUIDSIM_GGT_deck_domain",
                                 name, (1.0, 0.0, 0.0), (b[0] - a[0], b[1] - a[1]), box,
                                 cancel=True)
        judge_realtime("Domain X- face (Esc)", record)
        if record is not None:
            check("Domain X- face: Esc restores the start box",
                  record["after"] == record["before"], [record["before"], record["after"]])


def axis_extent(box, axis, side):
    """World coordinate of the domain face on ``axis``/``side`` (unit cube mesh)."""
    location, scale = box[axis], box[3 + axis]
    return location + side * 0.5 * scale


def run():
    wm = bpy.context.window_manager
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
        if os.environ.get("FLUIDSIM_GIZMO_KEEP_OPEN") != "1":
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
            finish()
        return None

    del wm
    bpy.app.timers.register(tick, first_interval=1.0, persistent=True)


if bpy.app.background:
    print(f"{FAIL_MARKER} needs a GUI Blender (no --background)", flush=True)
    sys.exit(1)
run()
