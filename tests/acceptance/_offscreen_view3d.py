"""Prove the 3D viewport draws something, without grabbing a window.

``blender_gui_acceptance.py`` used to prove "the viewport is visible" by asking
the window system for a screenshot (``bpy.ops.screen.screenshot_area``). That
grab does not exist on a Linux host without a compositor, so the GUI gate could
not run there. This module draws the 3D view into a ``gpu.offscreen`` buffer
instead and counts the pixels that differ from the background.

The module has two halves:

* ``count_non_background`` / ``evaluate_pixels`` are pure Python over any
  indexable pixel sequence, so they are unit tested without Blender
  (``tests/gate/test_offscreen_view3d.py``).
* ``check_view3d_offscreen`` is the Blender half. ``bpy`` and ``gpu`` are
  imported inside it, so importing this module never needs Blender.

The pass rule is the one ``_image_has_visible_content`` in the GUI script uses
(sample about 4000 pixels, call a sample visible when it is brighter than 0.02,
pass when more than ``max(20, sampled // 40)`` are visible), generalised from
"brighter than black" to "differs from the background colour" because the
offscreen buffer's background is not necessarily black. With a black background
the two rules are identical.
"""

from __future__ import annotations

import math
from collections import Counter

SAMPLE_LIMIT = 4000
THRESHOLD = 0.02


def _sample_offsets(width, height, sample_limit):
    """Pixel offsets on a regular 2D grid of at most ``sample_limit`` cells.

    A grid over x and y covers the whole frame evenly. Striding through the
    flat row-major index instead can alias with the width and read only a few
    columns.
    """
    side = max(int(math.sqrt(sample_limit)), 1)
    columns = min(width, side)
    rows = min(height, side)
    xs = [min(width - 1, (2 * i + 1) * width // (2 * columns)) for i in range(columns)]
    ys = [min(height - 1, (2 * j + 1) * height // (2 * rows)) for j in range(rows)]
    return [y * width + x for y in ys for x in xs]


def count_non_background(pixels, width, height, *, channels=4, value_range=1.0,
                         sample_limit=SAMPLE_LIMIT, threshold=THRESHOLD):
    """Count sampled pixels that differ from the frame's background colour.

    ``pixels`` is a flat, row-major sequence of ``width * height * channels``
    values in ``0..value_range`` (floats in ``0..1``, or bytes with
    ``value_range=255``). The background is the most common sampled colour,
    compared at 8-bit precision.

    A sample is non-background when its colour differs from the background by
    more than ``threshold``: the mean absolute difference over R, G and B, or
    the alpha difference when a fourth channel is present, whichever is larger.
    Returns ``(non_background, sampled, background)``, with ``background`` the
    reference colour as a tuple of 0..1 floats (empty when nothing was sampled).
    """
    if channels not in (3, 4):
        raise ValueError(f"channels must be 3 or 4, not {channels!r}")
    if width <= 0 or height <= 0:
        return 0, 0, ()
    needed = width * height * channels
    if len(pixels) < needed:
        raise ValueError(
            f"pixel buffer holds {len(pixels)} values but {width}x{height}x{channels} "
            f"needs {needed}")

    samples = []
    for offset in _sample_offsets(width, height, sample_limit):
        base = offset * channels
        samples.append(tuple(float(pixels[base + c]) / value_range for c in range(channels)))
    if not samples:
        return 0, 0, ()

    reference = Counter(
        tuple(round(value * 255) for value in sample) for sample in samples
    ).most_common(1)[0][0]
    background = tuple(value / 255 for value in reference)

    visible = 0
    for sample in samples:
        colour = sum(abs(sample[c] - background[c]) for c in range(3)) / 3.0
        alpha = abs(sample[3] - background[3]) if channels == 4 else 0.0
        if max(colour, alpha) > threshold:
            visible += 1
    return visible, len(samples), background


def evaluate_pixels(pixels, width, height, *, min_width, min_height, channels=4,
                    value_range=1.0, sample_limit=SAMPLE_LIMIT, threshold=THRESHOLD):
    """Return ``(ok, detail)`` for a pixel buffer, in the GUI script's wording.

    Fails when the frame is smaller than ``min_width`` x ``min_height``, is
    empty, or has no more than ``max(20, sampled // 40)`` visible samples.
    ``detail`` reads ``"<w>x<h>, <n>/<sampled> visible samples"`` like the
    window-grab check, plus the background colour on a verdict from pixels.
    """
    if width < min_width or height < min_height:
        return False, f"{width}x{height}"
    if width * height == 0:
        return False, f"{width}x{height}, empty pixels"
    visible, sampled, background = count_non_background(
        pixels, width, height, channels=channels, value_range=value_range,
        sample_limit=sample_limit, threshold=threshold)
    shade = "/".join(f"{value:.2f}" for value in background)
    detail = f"{width}x{height}, {visible}/{sampled} visible samples, background {shade}"
    return visible > max(20, sampled // 40), detail


def _draw_to_buffer(window, area, region, width, height):
    """Draw ``area``'s 3D view into an offscreen buffer; return the RGBA bytes.

    Must run while the GPU context is current, i.e. inside a draw handler.
    """
    import gpu

    space = area.spaces.active
    region_3d = space.region_3d
    offscreen = gpu.types.GPUOffScreen(width, height)
    try:
        with offscreen.bind():
            framebuffer = gpu.state.active_framebuffer_get()
            framebuffer.clear(color=(0.0, 0.0, 0.0, 0.0))
            offscreen.draw_view3d(
                window.scene,
                window.view_layer,
                space,
                region,
                region_3d.view_matrix,
                region_3d.window_matrix,
                do_color_management=False,
                draw_background=False,
            )
            buffer = framebuffer.read_color(0, 0, width, height, 4, 0, "UBYTE")
        buffer.dimensions = width * height * 4
        return buffer
    finally:
        offscreen.free()


def check_view3d_offscreen(override, *, min_width=320, min_height=200):
    """Draw the GUI's 3D view offscreen and judge it; return ``(ok, detail)``.

    ``override`` is the ``{"window", "area", "region", "screen"}`` mapping the
    GUI script builds for its 3D view (``_view3d_override``), or ``None``.
    Never raises: any failure comes back as ``(False, <reason>)`` so the caller
    records it as a failed check rather than aborting the run.

    The draw happens inside a ``POST_PIXEL`` handler that a synchronous
    ``wm.redraw_timer`` triggers. A timer callback has no guaranteed current GPU
    context; a draw handler always does. Overlays are switched off for the draw
    (and restored) so the grid and gizmos cannot count as scene content.
    """
    if override is None:
        return False, "no 3D view"
    try:
        import bpy

        window, area, region = override["window"], override["area"], override["region"]
        width, height = int(region.width), int(region.height)
        if width < min_width or height < min_height:
            return False, f"{width}x{height}"
        space = area.spaces.active
        outcome = {}

        def draw():
            if outcome:
                return
            try:
                outcome["buffer"] = _draw_to_buffer(window, area, region, width, height)
            except Exception as exc:  # noqa: BLE001 - reported as the check's detail
                outcome["error"] = exc

        overlays = space.overlay.show_overlays
        space.overlay.show_overlays = False
        handle = bpy.types.SpaceView3D.draw_handler_add(draw, (), "WINDOW", "POST_PIXEL")
        try:
            area.tag_redraw()
            with bpy.context.temp_override(**override):
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        finally:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
            space.overlay.show_overlays = overlays

        if "error" in outcome:
            return False, f"offscreen draw failed: {outcome['error']!r}"
        if "buffer" not in outcome:
            return False, "the offscreen draw handler never ran"
        return evaluate_pixels(
            outcome["buffer"], width, height, min_width=min_width,
            min_height=min_height, channels=4, value_range=255.0)
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)
