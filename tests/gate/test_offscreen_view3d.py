"""The GUI acceptance's offscreen viewport check (`tests/acceptance/_offscreen_view3d.py`).

The check lets ``blender_gui_acceptance.py`` prove the 3D view draws something on
a Linux host, where the window screenshot it used before is unavailable. It
runs only inside a GUI Blender, which pytest does not have, so this file tests
what can be tested without one:

* the pixel counting, on synthetic buffers, including the cases the old
  "brighter than black" rule gets wrong (a uniform non-black frame);
* the draw orchestration, against a recording stand-in for the narrow ``bpy`` /
  ``gpu`` surface the module uses (names and signatures as introspected from
  Blender 5.2), so overlay restore, handler removal, buffer release and every
  failure message are exercised.

What it cannot show is that Blender really draws the scene into the buffer.
That needs one GUI run (see the lane E report).
"""

from __future__ import annotations

import importlib.util
import re
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tests" / "acceptance" / "_offscreen_view3d.py"
GUI_SCRIPT = ROOT / "tests" / "acceptance" / "blender_gui_acceptance.py"


def _load():
    spec = importlib.util.spec_from_file_location("offscreen_view3d_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def view():
    return _load()


def frame(width, height, background=(0.0, 0.0, 0.0, 0.0), rects=()):
    """A flat RGBA float frame: ``background`` with ``(x0, y0, x1, y1, colour)`` fills."""
    pixels = np.empty((height, width, 4), dtype=np.float32)
    pixels[:, :] = background
    for x0, y0, x1, y1, colour in rects:
        pixels[y0:y1, x0:x1] = colour
    return pixels.reshape(-1)


FLUID = (0.2, 0.5, 0.9, 1.0)


# --------------------------------------------------------------------------
# Importing needs no Blender
# --------------------------------------------------------------------------

def test_the_module_imports_without_bpy_or_gpu(monkeypatch):
    # A ``None`` entry makes ``import bpy`` / ``import gpu`` raise ImportError.
    monkeypatch.setitem(sys.modules, "bpy", None)
    monkeypatch.setitem(sys.modules, "gpu", None)

    module = _load()

    assert callable(module.check_view3d_offscreen)
    assert callable(module.evaluate_pixels)


def test_without_blender_the_check_reports_failure_instead_of_raising(monkeypatch, view):
    monkeypatch.setitem(sys.modules, "bpy", None)
    override = {"window": object(), "area": object(), "region": SimpleNamespace(width=800, height=600)}

    ok, detail = view.check_view3d_offscreen(override)

    assert ok is False
    assert "import of bpy halted" in detail


def test_no_3d_view_is_a_failed_check(view):
    assert view.check_view3d_offscreen(None) == (False, "no 3D view")


# --------------------------------------------------------------------------
# Pixel counting
# --------------------------------------------------------------------------

def test_a_blank_black_frame_has_no_visible_samples(view):
    ok, detail = view.evaluate_pixels(frame(640, 400), 640, 400, min_width=320, min_height=200)

    assert ok is False
    assert re.match(r"640x400, 0/\d+ visible samples", detail)


def test_a_uniform_non_black_frame_is_still_blank(view):
    """The case the old rule gets wrong: every pixel is bright, none is content.

    ``_image_has_visible_content`` counts luminance above 0.02, so a frame
    cleared to a grey (a world colour, an unlit viewport) would pass it. The
    check compares against the frame's own background, so it does not.
    """
    grey = frame(640, 400, background=(0.5, 0.5, 0.5, 1.0))

    ok, detail = view.evaluate_pixels(grey, 640, 400, min_width=320, min_height=200)

    assert ok is False
    assert re.match(r"640x400, 0/\d+ visible samples", detail)


@pytest.mark.parametrize(
    "background",
    [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 1.0), (0.9, 0.9, 0.9, 1.0)],
    ids=["transparent", "opaque-black", "grey", "light-grey"],
)
def test_drawn_content_is_found_on_any_background(view, background):
    pixels = frame(640, 400, background, [(100, 80, 400, 300, FLUID)])

    ok, detail = view.evaluate_pixels(pixels, 640, 400, min_width=320, min_height=200)

    assert ok is True, detail


def test_content_that_differs_only_in_alpha_counts(view):
    """Black geometry on a transparent clear is content, not background."""
    pixels = frame(640, 400, (0.0, 0.0, 0.0, 0.0), [(100, 80, 400, 300, (0.0, 0.0, 0.0, 1.0))])

    ok, _detail = view.evaluate_pixels(pixels, 640, 400, min_width=320, min_height=200)

    assert ok is True


def test_a_few_stray_pixels_are_not_enough(view):
    # A handful of pixels is noise: the rule needs more than max(20, sampled // 40).
    pixels = frame(640, 400, rects=[(300, 200, 303, 203, FLUID)])

    ok, _detail = view.evaluate_pixels(pixels, 640, 400, min_width=320, min_height=200)

    assert ok is False


def test_the_pass_threshold_matches_the_window_grab_rule(view):
    """More than ``max(20, sampled // 40)`` visible samples passes, no fewer."""
    width = height = 100
    sampled = 100 * 100  # fewer cells than the 4000 limit: every pixel is sampled
    needed = max(20, sampled // 40)  # 250
    # A band of full rows: each row is 100 samples.
    just_enough = frame(width, height, rects=[(0, 0, width, 3, FLUID)])  # 300 visible
    too_few = frame(width, height, rects=[(0, 0, width, 2, FLUID)])  # 200 visible
    assert 300 > needed >= 200

    kwargs = dict(min_width=50, min_height=50, sample_limit=sampled)
    assert view.evaluate_pixels(just_enough, width, height, **kwargs)[0] is True
    assert view.evaluate_pixels(too_few, width, height, **kwargs)[0] is False


def test_sampling_covers_the_whole_frame_not_a_few_columns(view):
    """Content confined to the last rows or the last columns is still seen.

    A stride through the flat row-major index can alias with the width and read
    a handful of columns; a regular 2D grid cannot.
    """
    width, height = 1000, 1000  # 1,000,000 px / 4000 = a flat stride of 250
    bottom = frame(width, height, rects=[(0, 940, width, height, FLUID)])
    right = frame(width, height, rects=[(940, 0, width, height, FLUID)])

    for pixels in (bottom, right):
        visible, sampled, _ = view.count_non_background(pixels, width, height)
        assert 0 < visible < sampled
        assert visible / sampled == pytest.approx(0.06, abs=0.02)


def test_the_sample_count_stays_within_the_limit(view):
    _visible, sampled, _ = view.count_non_background(frame(1920, 1080), 1920, 1080)

    assert 3000 < sampled <= 4000


def test_a_frame_smaller_than_the_grid_is_sampled_in_full(view):
    visible, sampled, _ = view.count_non_background(
        frame(10, 6, rects=[(0, 0, 10, 3, FLUID)]), 10, 6)

    assert (visible, sampled) == (30, 60)


@pytest.mark.parametrize("kind", ["list", "numpy", "bytes"])
def test_float_list_numpy_and_byte_buffers_agree(view, kind):
    floats = frame(400, 300, (0.1, 0.1, 0.1, 1.0), [(50, 50, 250, 200, FLUID)])
    if kind == "list":
        pixels, scale = floats.tolist(), 1.0
    elif kind == "numpy":
        pixels, scale = floats, 1.0
    else:
        pixels, scale = [int(round(v * 255)) for v in floats], 255.0

    result = view.count_non_background(pixels, 400, 300, value_range=scale)
    expected = view.count_non_background(floats, 400, 300)

    assert result[:2] == expected[:2]
    assert result[0] > 0


def test_detail_uses_the_window_grab_wording(view):
    """Same ``<w>x<h>, <n>/<sampled> visible samples`` shape the report parses."""
    pixels = frame(640, 400, rects=[(100, 80, 400, 300, FLUID)])

    _ok, detail = view.evaluate_pixels(pixels, 640, 400, min_width=320, min_height=200)

    assert re.match(r"^640x400, \d+/\d+ visible samples, background 0\.00/0\.00/0\.00/0\.00$", detail)


@pytest.mark.parametrize("size", [(319, 400), (640, 199), (0, 0)])
def test_a_frame_below_the_minimum_size_fails_on_size(view, size):
    width, height = size

    ok, detail = view.evaluate_pixels([], width, height, min_width=320, min_height=200)

    assert (ok, detail) == (False, f"{width}x{height}")


def test_a_short_buffer_is_an_error_not_a_pass(view):
    with pytest.raises(ValueError, match="pixel buffer holds"):
        view.count_non_background([0.0] * 100, 640, 400)


def test_unsupported_channel_counts_are_refused(view):
    with pytest.raises(ValueError, match="channels must be 3 or 4"):
        view.count_non_background([0.0] * 100, 5, 5, channels=2)


def test_three_channel_buffers_are_supported(view):
    pixels = np.zeros((300, 400, 3), dtype=np.float32)
    pixels[50:200, 50:250] = (0.2, 0.5, 0.9)

    ok, _ = view.evaluate_pixels(pixels.reshape(-1), 400, 300, min_width=320, min_height=200, channels=3)

    assert ok is True


# --------------------------------------------------------------------------
# Draw orchestration against a recording bpy/gpu stand-in
# --------------------------------------------------------------------------

WIDTH, HEIGHT = 640, 400


class _World:
    """Recording stand-ins for the ``bpy``/``gpu`` calls the module makes.

    Shapes follow Blender 5.2: ``GPUOffScreen(width, height, *, format)`` with
    ``bind()`` as a context manager, ``draw_view3d(scene, view_layer, view3d,
    region, view_matrix, projection_matrix, *, do_color_management,
    draw_background)`` and ``free()``; ``GPUFrameBuffer.read_color(x, y, w, h,
    channels, slot, format)`` returning a buffer whose ``dimensions`` is
    assignable; ``SpaceView3D.draw_handler_add``/``_remove``; and
    ``wm.redraw_timer`` running the draw handlers synchronously.
    """

    def __init__(self, *, draws_content=True, redraw_runs_handlers=True, draw_error=None,
                 redraw_error=None, background=(0, 0, 0, 0), content=(50, 128, 230, 255)):
        self.background = background  # RGBA bytes the offscreen is cleared to
        self.content = content  # RGBA bytes the scene is drawn in
        self.draws_content = draws_content
        self.redraw_runs_handlers = redraw_runs_handlers
        self.draw_error = draw_error
        self.redraw_error = redraw_error
        self.handlers = []
        self.removed = []
        self.offscreens = []
        self.draws = []
        self.overlay_during_draw = []
        self.redraws = 0
        self.tagged = 0
        self.overrides = []

        self.space = SimpleNamespace(
            overlay=SimpleNamespace(show_overlays=True),
            region_3d=SimpleNamespace(view_matrix="VIEW", window_matrix="PROJECTION"),
        )
        self.area = SimpleNamespace(spaces=SimpleNamespace(active=self.space), tag_redraw=self._tag)
        self.region = SimpleNamespace(width=WIDTH, height=HEIGHT)
        self.window = SimpleNamespace(scene="SCENE", view_layer="VIEW_LAYER")
        self.override = {"window": self.window, "area": self.area, "region": self.region,
                         "screen": object()}

    def _tag(self):
        self.tagged += 1

    # -- gpu ---------------------------------------------------------------
    def gpu_module(self):
        world = self

        class FrameBuffer:
            def __init__(self, offscreen):
                self.offscreen = offscreen
                self.cleared = None

            def clear(self, color=None, depth=None, stencil=None):
                self.cleared = color
                self.offscreen.cleared_to = color

            def read_color(self, x, y, xsize, ysize, channels, slot, fmt, *, data=None):
                assert (x, y, xsize, ysize, channels, slot) == (0, 0, WIDTH, HEIGHT, 4, 0)
                assert fmt == "UBYTE"
                assert self.offscreen.bound, "read outside bind()"
                assert self.offscreen.drawn, "read before draw_view3d"
                pixels = np.empty((ysize, xsize, 4), dtype=np.uint8)
                pixels[:, :] = world.background
                if world.draws_content:
                    pixels[80:300, 100:400] = world.content
                buffer = _Buffer(pixels.reshape(-1).tolist())
                return buffer

        class OffScreen:
            def __init__(self, width, height, *, format="RGBA8"):
                self.size = (width, height)
                self.bound = 0
                self.drawn = False
                self.freed = False
                self.cleared_to = None
                self.framebuffer = FrameBuffer(self)
                world.offscreens.append(self)

            @contextmanager
            def bind(self):
                self.bound += 1
                try:
                    yield
                finally:
                    self.bound -= 1

            def draw_view3d(self, scene, view_layer, view3d, region, view_matrix,
                            projection_matrix, *, do_color_management=False,
                            draw_background=True):
                assert self.bound, "draw_view3d outside bind()"
                if world.draw_error is not None:
                    raise world.draw_error
                world.overlay_during_draw.append(view3d.overlay.show_overlays)
                world.draws.append(dict(
                    scene=scene, view_layer=view_layer, view3d=view3d, region=region,
                    view_matrix=view_matrix, projection_matrix=projection_matrix,
                    do_color_management=do_color_management,
                    draw_background=draw_background, cleared_to=self.cleared_to))
                self.drawn = True

            def free(self):
                self.freed = True

        gpu = SimpleNamespace(
            types=SimpleNamespace(GPUOffScreen=OffScreen),
            state=SimpleNamespace(active_framebuffer_get=lambda: world.offscreens[-1].framebuffer),
        )
        return gpu

    # -- bpy ---------------------------------------------------------------
    def bpy_module(self):
        world = self

        def handler_add(function, args, region_type, draw_type):
            assert (region_type, draw_type) == ("WINDOW", "POST_PIXEL")
            handle = object()
            world.handlers.append((handle, function, args))
            return handle

        def handler_remove(handle, region_type):
            assert region_type == "WINDOW"
            world.removed.append(handle)
            world.handlers[:] = [h for h in world.handlers if h[0] is not handle]

        def redraw_timer(*, type, iterations):
            assert (type, iterations) == ("DRAW_WIN_SWAP", 1)
            world.redraws += 1
            if world.redraw_error is not None:
                raise world.redraw_error
            if world.redraw_runs_handlers:
                for _handle, function, args in list(world.handlers):
                    function(*args)

        @contextmanager
        def temp_override(**kwargs):
            world.overrides.append(kwargs)
            yield

        return SimpleNamespace(
            types=SimpleNamespace(SpaceView3D=SimpleNamespace(
                draw_handler_add=handler_add, draw_handler_remove=handler_remove)),
            ops=SimpleNamespace(wm=SimpleNamespace(redraw_timer=redraw_timer)),
            context=SimpleNamespace(temp_override=temp_override),
        )


class _Buffer(list):
    """``gpu.types.Buffer`` stand-in: indexable, with a settable ``dimensions``."""

    dimensions = None


@pytest.fixture
def world(monkeypatch):
    def build(**kwargs):
        built = _World(**kwargs)
        monkeypatch.setitem(sys.modules, "bpy", built.bpy_module())
        monkeypatch.setitem(sys.modules, "gpu", built.gpu_module())
        return built

    return build


def test_a_drawn_scene_passes_and_the_view_is_left_as_it_was(world, view):
    w = world()

    ok, detail = view.check_view3d_offscreen(w.override)

    assert ok is True, detail
    assert re.match(rf"^{WIDTH}x{HEIGHT}, \d+/\d+ visible samples", detail)
    # Overlays were off for the draw and are back on afterwards.
    assert w.overlay_during_draw == [False]
    assert w.space.overlay.show_overlays is True
    # The handler is gone and the offscreen buffer released.
    assert w.handlers == [] and len(w.removed) == 1
    assert [o.freed for o in w.offscreens] == [True]
    assert w.offscreens[0].size == (WIDTH, HEIGHT)
    # The redraw ran inside the view's context.
    assert w.overrides == [w.override]
    assert w.tagged == 1 and w.redraws == 1


def test_the_draw_uses_the_views_own_scene_matrices_and_no_background(world, view):
    w = world()

    view.check_view3d_offscreen(w.override)

    (draw,) = w.draws
    assert draw["scene"] == "SCENE" and draw["view_layer"] == "VIEW_LAYER"
    assert draw["view3d"] is w.space and draw["region"] is w.region
    assert (draw["view_matrix"], draw["projection_matrix"]) == ("VIEW", "PROJECTION")
    # No world colour, so an empty scene stays uniform; cleared first, so the
    # buffer holds nothing the draw did not put there.
    assert draw["draw_background"] is False
    assert draw["do_color_management"] is False
    assert draw["cleared_to"] == (0.0, 0.0, 0.0, 0.0)


def test_an_empty_scene_fails_the_check(world, view):
    w = world(draws_content=False)

    ok, detail = view.check_view3d_offscreen(w.override)

    assert ok is False
    assert re.match(rf"^{WIDTH}x{HEIGHT}, 0/\d+ visible samples", detail)
    assert w.space.overlay.show_overlays is True


def test_the_byte_buffer_is_judged_on_a_0_to_255_scale(world, view):
    """A 2/255 difference is below the 0.02 threshold: faint, not content.

    Reading the byte buffer as if it were 0..1 floats would call every
    non-identical byte visible.
    """
    w = world(background=(10, 10, 10, 255), content=(12, 12, 12, 255))

    ok, detail = view.check_view3d_offscreen(w.override)

    assert ok is False
    assert re.match(rf"^{WIDTH}x{HEIGHT}, 0/\d+ visible samples", detail)


def test_a_draw_that_raises_is_reported_and_cleaned_up(world, view):
    w = world(draw_error=RuntimeError("no GPU context"))

    ok, detail = view.check_view3d_offscreen(w.override)

    assert ok is False
    assert detail == "offscreen draw failed: RuntimeError('no GPU context')"
    assert w.handlers == [] and len(w.removed) == 1
    assert w.space.overlay.show_overlays is True
    assert [o.freed for o in w.offscreens] == [True]  # freed even though the draw raised


def test_a_handler_that_never_runs_is_a_failure_not_a_pass(world, view):
    w = world(redraw_runs_handlers=False)

    ok, detail = view.check_view3d_offscreen(w.override)

    assert (ok, detail) == (False, "the offscreen draw handler never ran")
    assert w.handlers == [] and len(w.removed) == 1
    assert w.space.overlay.show_overlays is True


def test_a_failing_redraw_still_restores_overlays_and_removes_the_handler(world, view):
    w = world(redraw_error=RuntimeError("redraw refused"))

    ok, detail = view.check_view3d_offscreen(w.override)

    assert ok is False
    assert "redraw refused" in detail
    assert w.handlers == [] and len(w.removed) == 1
    assert w.space.overlay.show_overlays is True


def test_the_draw_runs_once_even_if_the_handler_is_called_again(world, view):
    w = world()
    original = w.bpy_module().ops.wm.redraw_timer

    def redraw_twice(*, type, iterations):
        original(type=type, iterations=iterations)
        for _handle, function, args in list(w.handlers):
            function(*args)

    sys.modules["bpy"].ops.wm.redraw_timer = redraw_twice

    ok, _detail = view.check_view3d_offscreen(w.override)

    assert ok is True
    assert len(w.offscreens) == 1


def test_overlays_that_were_already_off_stay_off(world, view):
    w = world()
    w.space.overlay.show_overlays = False

    view.check_view3d_offscreen(w.override)

    assert w.overlay_during_draw == [False]
    assert w.space.overlay.show_overlays is False


def test_a_region_below_the_minimum_size_fails_before_drawing(world, view):
    w = world()
    w.region.width, w.region.height = 200, 100

    ok, detail = view.check_view3d_offscreen(w.override)

    assert (ok, detail) == (False, "200x100")
    assert w.offscreens == [] and w.handlers == [] and w.removed == []


# --------------------------------------------------------------------------
# The GUI script wires it in: Linux skips the window grab, Windows runs both
# --------------------------------------------------------------------------

def _viewport_block():
    """The GUI script's viewport-proof statements, dedented so they can be run.

    The script imports ``bpy`` at module level and drives a live window, so it
    cannot be imported here. The block is plain statements over names the
    script defines, which the tests below supply as recording stand-ins.
    """
    source = GUI_SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n")
    start = source.rindex("\n", 0, source.index("# Viewport proof")) + 1
    end = source.index('STATE["phase"] = "workspace"', start)
    return textwrap.dedent(source[start:end])


def _run_viewport_block(monkeypatch, platform, *, offscreen=(True, "ok"), grab_ok=True):
    """Execute the block as ``platform``; return the recorded checks and grabs."""
    checks, grabs = [], []
    fake_helper = SimpleNamespace(check_view3d_offscreen=lambda override: offscreen)
    monkeypatch.setitem(sys.modules, "_offscreen_view3d", fake_helper)

    @contextmanager
    def temp_override(**_kwargs):
        yield

    namespace = {
        "__file__": str(GUI_SCRIPT),
        "os": __import__("os"),
        "sys": SimpleNamespace(platform=platform, path=[]),
        "bpy": SimpleNamespace(
            context=SimpleNamespace(temp_override=temp_override),
            ops=SimpleNamespace(screen=SimpleNamespace(
                screenshot_area=lambda filepath: grabs.append(filepath))),
        ),
        "check": lambda name, cond, detail="": checks.append((name, bool(cond), detail)),
        "_view3d_override": lambda: {"area": object()},
        "_image_has_visible_content": lambda path, min_width, min_height: (
            grab_ok, "grabbed" if grab_ok else "blank"),
        "SCREENSHOT_PATH": "viewport.png",
    }
    exec(compile(_viewport_block(), str(GUI_SCRIPT), "exec"), namespace)
    return checks, grabs


def test_on_linux_the_offscreen_draw_replaces_the_window_grab(monkeypatch):
    checks, grabs = _run_viewport_block(monkeypatch, "linux")

    assert checks == [("viewport offscreen draw", True, "ok")]
    assert grabs == []


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_elsewhere_both_the_offscreen_draw_and_the_window_grab_run(monkeypatch, platform):
    checks, grabs = _run_viewport_block(monkeypatch, platform)

    assert [name for name, _ok, _detail in checks] == ["viewport offscreen draw", "viewport screenshot"]
    assert all(ok for _name, ok, _detail in checks)
    assert grabs == ["viewport.png"]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_a_failed_offscreen_draw_is_a_failed_check_on_every_platform(monkeypatch, platform):
    checks, _grabs = _run_viewport_block(monkeypatch, platform, offscreen=(False, "0/3969 visible samples"))

    assert ("viewport offscreen draw", False, "0/3969 visible samples") in checks


def test_on_windows_a_failed_window_grab_fails_even_when_the_offscreen_draw_passes(monkeypatch):
    checks, _grabs = _run_viewport_block(monkeypatch, "win32", grab_ok=False)

    assert ("viewport offscreen draw", True, "ok") in checks
    assert ("viewport screenshot", False, "blank") in checks


def test_the_helper_is_found_next_to_the_script(monkeypatch):
    """The script appends its own directory to ``sys.path`` before importing."""
    block = _viewport_block()

    assert "from _offscreen_view3d import check_view3d_offscreen" in block
    assert block.index("sys.path.append(scripts_dir)") < block.index("from _offscreen_view3d")
    assert (GUI_SCRIPT.parent / "_offscreen_view3d.py").is_file()
