"""Small ``gpu``/``blf`` drawing primitives shared by the Deck bar, palette and wheel.

Every function draws in region pixel space (a ``POST_PIXEL`` handler). ``gpu`` and
``blf`` are imported per call so the Deck modules import without Blender.
"""

from __future__ import annotations

import math

_SEGMENTS = 8  # per rounded corner


def _shader():
    import gpu

    return gpu.shader.from_builtin("UNIFORM_COLOR")


def _draw_tris(vertices, colour):
    if not vertices:
        return
    import gpu
    from gpu_extras.batch import batch_for_shader

    shader = _shader()
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices})
    gpu.state.blend_set("ALPHA")
    shader.bind()
    shader.uniform_float("color", tuple(colour))
    batch.draw(shader)
    gpu.state.blend_set("NONE")


def _fan(centre, outline):
    tris = []
    for i in range(len(outline)):
        tris.extend((centre, outline[i], outline[(i + 1) % len(outline)]))
    return tris


def rounded_rect_outline(x0, y0, x1, y1, radius):
    radius = max(0.0, min(radius, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    corners = ((x1 - radius, y1 - radius, 0.0), (x0 + radius, y1 - radius, 90.0),
               (x0 + radius, y0 + radius, 180.0), (x1 - radius, y0 + radius, 270.0))
    points = []
    for cx, cy, start in corners:
        for step in range(_SEGMENTS + 1):
            angle = math.radians(start + 90.0 * step / _SEGMENTS)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def rounded_rect(x0, y0, x1, y1, radius, colour):
    outline = rounded_rect_outline(x0, y0, x1, y1, radius)
    _draw_tris(_fan(((x0 + x1) / 2.0, (y0 + y1) / 2.0), outline), colour)


def circle(cx, cy, radius, colour, segments=32):
    outline = [(cx + radius * math.cos(2 * math.pi * i / segments),
                cy + radius * math.sin(2 * math.pi * i / segments)) for i in range(segments)]
    _draw_tris(_fan((cx, cy), outline), colour)


def ring(cx, cy, radius, thickness, fraction, colour, segments=48):
    """An arc from 12 o'clock, clockwise, covering ``fraction`` of the circle."""
    fraction = max(0.0, min(1.0, float(fraction)))
    count = max(1, int(segments * fraction))
    inner, outer = radius - thickness / 2.0, radius + thickness / 2.0
    tris = []
    for i in range(count):
        a0 = math.pi / 2 - 2 * math.pi * fraction * i / count
        a1 = math.pi / 2 - 2 * math.pi * fraction * (i + 1) / count
        p0 = (cx + inner * math.cos(a0), cy + inner * math.sin(a0))
        p1 = (cx + outer * math.cos(a0), cy + outer * math.sin(a0))
        p2 = (cx + outer * math.cos(a1), cy + outer * math.sin(a1))
        p3 = (cx + inner * math.cos(a1), cy + inner * math.sin(a1))
        tris.extend((p0, p1, p2, p0, p2, p3))
    _draw_tris(tris, colour)


def polyline(points, colour, width=1.5):
    """Connected line segments drawn as thin quads (no line-width state needed)."""
    tris = []
    half = width / 2.0
    for (xa, ya), (xb, yb) in zip(points, points[1:]):
        dx, dy = xb - xa, yb - ya
        length = math.hypot(dx, dy)
        if length <= 0.0:
            continue
        nx, ny = -dy / length * half, dx / length * half
        a, b = (xa + nx, ya + ny), (xb + nx, yb + ny)
        c, d = (xb - nx, yb - ny), (xa - nx, ya - ny)
        tris.extend((a, b, c, a, c, d))
    _draw_tris(tris, colour)


def text(x, y, string, size, colour):
    import blf

    blf.size(0, size)
    blf.color(0, *colour)
    blf.position(0, x, y, 0)
    blf.draw(0, str(string))


def text_width(string, size) -> float:
    import blf

    blf.size(0, size)
    return float(blf.dimensions(0, str(string))[0])
