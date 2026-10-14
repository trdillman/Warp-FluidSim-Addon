"""Deterministic offscreen references for Liquid Preview.

The visual acceptance is an offscreen render of a synthetic, seeded payload (a
sphere of liquid above a wavy sheet, plus whitewater) compared with committed
PNGs by SSIM. Everything except :func:`render_reference` is plain numpy, so
the comparison helpers are tested without Blender.
"""

from __future__ import annotations

import math
from pathlib import Path
import struct
import zlib

import numpy as np

REFERENCE_SIZE = (960, 540)
REFERENCE_VIEWS = ("sphere_sheet_persp", "sphere_sheet_ortho")
REFERENCE_PARTICLES = 60_000
REFERENCE_SEED = 7
REFERENCE_RADIUS = 0.02
BACKGROUND = (0.18, 0.19, 0.22, 1.0)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# --- SSIM ------------------------------------------------------------------

def _gaussian_window(size=11, sigma=1.5):
    offsets = np.arange(size) - (size - 1) / 2.0
    window = np.exp(-offsets ** 2 / (2.0 * sigma * sigma))
    return window / window.sum()


def _filter_valid(image, window):
    size = len(window)
    rows = image.shape[0] - size + 1
    cols = image.shape[1] - size + 1
    vertical = sum(w * image[k:k + rows] for k, w in enumerate(window))
    return sum(w * vertical[:, k:k + cols] for k, w in enumerate(window))


def ssim(first, second, *, data_range=255.0):
    """Mean structural similarity (Wang et al. 2004), Gaussian 11x11 window.

    Colour images are scored per RGB channel and averaged; alpha is ignored.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"image shapes differ: {a.shape} vs {b.shape}")
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    channels = min(a.shape[2], 3)
    window = _gaussian_window()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    scores = []
    for channel in range(channels):
        x, y = a[..., channel], b[..., channel]
        mu_x, mu_y = _filter_valid(x, window), _filter_valid(y, window)
        var_x = _filter_valid(x * x, window) - mu_x * mu_x
        var_y = _filter_valid(y * y, window) - mu_y * mu_y
        cov = _filter_valid(x * y, window) - mu_x * mu_y
        score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)
                 / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)))
        scores.append(float(score.mean()))
    return float(np.mean(scores))


# --- PNG (8-bit RGB/RGBA, no interlace) -------------------------------------

def _chunk(kind, data):
    body = kind + data
    return (struct.pack(">I", len(data)) + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))


def write_png(path, image):
    """Write an ``(h, w, 3|4)`` uint8 array as a PNG (top row first)."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    color_type = {3: 2, 4: 6}[channels]
    rows = np.concatenate(
        [np.zeros((height, 1), np.uint8), image.reshape(height, -1)], axis=1)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    Path(path).write_bytes(
        _PNG_SIGNATURE + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 9))
        + _chunk(b"IEND", b""))


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_png(path):
    """Read an 8-bit RGB/RGBA PNG into an ``(h, w, channels)`` uint8 array."""
    data = Path(path).read_bytes()
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError(f"{path} is not a PNG")
    offset, idat = len(_PNG_SIGNATURE), b""
    width = height = channels = None
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset:offset + 4])
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, depth, color_type, _c, _f, interlace = struct.unpack(
                ">IIBBBBB", body)
            if depth != 8 or color_type not in (2, 6) or interlace:
                raise ValueError("only 8-bit RGB/RGBA non-interlaced PNGs")
            channels = 3 if color_type == 2 else 4
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
    stride = width * channels
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(
        height, stride + 1)
    out = np.zeros((height, stride), dtype=np.int32)
    previous = np.zeros(stride, dtype=np.int32)
    for row in range(height):
        kind, line = raw[row, 0], raw[row, 1:].astype(np.int32)
        if kind == 0:
            current = line
        elif kind == 2:
            current = (line + previous) & 0xFF
        else:
            current = np.zeros(stride, dtype=np.int32)
            for i in range(stride):
                left = current[i - channels] if i >= channels else 0
                up = previous[i]
                corner = previous[i - channels] if i >= channels else 0
                if kind == 1:
                    predictor = left
                elif kind == 3:
                    predictor = (left + up) // 2
                elif kind == 4:
                    predictor = _paeth(left, up, corner)
                else:
                    raise ValueError(f"unknown PNG filter {kind}")
                current[i] = (line[i] + predictor) & 0xFF
        out[row] = current
        previous = current
    return out.astype(np.uint8).reshape(height, width, channels)


# --- synthetic payload -----------------------------------------------------

def synthetic_points(count=REFERENCE_PARTICLES, seed=REFERENCE_SEED):
    """Seeded liquid sphere over a wavy sheet, plus typed whitewater.

    Returns ``(positions, whitewater, whitewater_types)`` as float32, float32
    and int32 arrays. Types are 0 foam, 1 bubble, 2 spray.
    """
    rng = np.random.default_rng(seed)
    sphere_count = int(count * 0.35)
    sheet_count = count - sphere_count
    directions = rng.normal(size=(sphere_count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = 0.3 * np.cbrt(rng.random(sphere_count))[:, None]
    sphere = np.array([0.0, 0.0, 0.55]) + directions * radii
    x = rng.uniform(-1.0, 1.0, sheet_count)
    y = rng.uniform(-1.0, 1.0, sheet_count)
    z = (0.08 + rng.uniform(0.0, 0.1, sheet_count)  # clear of the floor
         + 0.05 * np.sin(3.0 * x) * np.cos(2.0 * y))
    sheet = np.stack([x, y, z], axis=1)
    positions = np.concatenate([sphere, sheet]).astype(np.float32)

    ww_count = max(count // 100, 1)
    shell = rng.normal(size=(ww_count, 3))
    shell /= np.linalg.norm(shell, axis=1, keepdims=True)
    shell[:, 2] = 0.75 + 0.25 * np.abs(shell[:, 2])  # a crown over the top
    shell /= np.linalg.norm(shell, axis=1, keepdims=True)
    lift = rng.uniform(0.0, 0.15, (ww_count, 1))
    whitewater = (np.array([0.0, 0.0, 0.55]) + shell * (0.32 + lift))
    types = rng.integers(0, 3, ww_count).astype(np.int32)
    return positions, whitewater.astype(np.float32), types


def reference_record(count=REFERENCE_PARTICLES, seed=REFERENCE_SEED):
    """A :class:`PublishedPoints` record holding the synthetic payload."""
    from ..output_publication import PublishedPoints

    positions, whitewater, types = synthetic_points(count, seed)
    for array in (positions, whitewater, types):
        array.flags.writeable = False
    return PublishedPoints(
        frame_id=1, scene_frame=0, scene_key=None, positions=positions,
        particle_radius=REFERENCE_RADIUS, whitewater=whitewater,
        whitewater_types=types, whitewater_radius=REFERENCE_RADIUS * 0.6)


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    true_up = np.cross(side, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = side, true_up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def reference_camera(width, height, *, orthographic=False):
    """World-to-eye and projection matrices for a reference view."""
    view = _look_at((2.2, -2.6, 1.6), (0.0, 0.0, 0.3))
    aspect = width / height
    near, far = 0.1, 100.0
    projection = np.zeros((4, 4))
    if orthographic:
        half = 1.2
        projection[0, 0] = 1.0 / (half * aspect)
        projection[1, 1] = 1.0 / half
        projection[2, 2] = -2.0 / (far - near)
        projection[2, 3] = -(far + near) / (far - near)
        projection[3, 3] = 1.0
    else:
        f = 1.0 / math.tan(math.radians(40.0) / 2.0)
        projection[0, 0] = f / aspect
        projection[1, 1] = f
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = 2.0 * far * near / (near - far)
        projection[3, 2] = -1.0
    return view, projection


# --- Blender-only rendering ------------------------------------------------

_BACKDROP_VERT = """
void main()
{
  shade = color;
  world = pos;
  gl_Position = mvp * vec4(pos, 1.0);
}
"""

_BACKDROP_FRAG = """
void main()
{
  float checker = mod(floor(world.x * 4.0) + floor(world.y * 4.0), 2.0);
  frag = vec4(shade * (0.85 + 0.15 * checker), 1.0);
}
"""


def _backdrop_geometry():
    """A ground plane and an opaque pillar between the camera and the liquid."""
    triangles, colors = [], []

    def quad(a, b, c, d, color):
        triangles.extend([a, b, c, a, c, d])
        colors.extend([color] * 6)

    ground = (0.42, 0.40, 0.36)
    quad((-3, -3, -0.03), (3, -3, -0.03), (3, 3, -0.03), (-3, 3, -0.03), ground)
    x0, x1, y0, y1, top = 0.75, 0.95, -1.15, -0.95, 0.9
    pillar = (0.62, 0.28, 0.22)
    quad((x0, y0, 0), (x1, y0, 0), (x1, y0, top), (x0, y0, top), pillar)
    quad((x1, y0, 0), (x1, y1, 0), (x1, y1, top), (x1, y0, top),
         tuple(c * 0.8 for c in pillar))
    quad((x0, y1, 0), (x0, y0, 0), (x0, y0, top), (x0, y1, top),
         tuple(c * 0.7 for c in pillar))
    quad((x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top),
         tuple(min(1.0, c * 1.2) for c in pillar))
    return (np.asarray(triangles, dtype=np.float32),
            np.asarray(colors, dtype=np.float32))


def draw_backdrop(gpu, view, projection):
    """Opaque test scene that writes depth, so occlusion is exercised."""
    iface = gpu.types.GPUStageInterfaceInfo("liquid_preview_backdrop_iface")
    iface.smooth("VEC3", "shade")
    iface.smooth("VEC3", "world")
    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, "VEC3", "pos")
    info.vertex_in(1, "VEC3", "color")
    info.vertex_out(iface)
    info.push_constant("MAT4", "mvp")
    info.fragment_out(0, "VEC4", "frag")
    info.vertex_source(_BACKDROP_VERT)
    info.fragment_source(_BACKDROP_FRAG)
    shader = gpu.shader.create_from_info(info)
    positions, colors = _backdrop_geometry()
    fmt = gpu.types.GPUVertFormat()
    fmt.attr_add(id="pos", comp_type="F32", len=3, fetch_mode="FLOAT")
    fmt.attr_add(id="color", comp_type="F32", len=3, fetch_mode="FLOAT")
    vbo = gpu.types.GPUVertBuf(fmt, len(positions))
    vbo.attr_fill("pos", positions)
    vbo.attr_fill("color", colors)
    batch = gpu.types.GPUBatch(type="TRIS", buf=vbo)
    gpu.state.depth_test_set("LESS_EQUAL")
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set("NONE")
    shader.bind()
    mvp = (np.asarray(projection) @ np.asarray(view)).astype(np.float32)
    shader.uniform_float("mvp", mvp.T.ravel().tolist())  # column-major
    batch.draw(shader)


def render_reference(view_name, *, size=REFERENCE_SIZE, renderer=None,
                     record=None, settings=None):
    """Render one reference view offscreen; returns ``(h, w, 4)`` uint8."""
    import gpu

    from .passes import LiquidPreviewRenderer, PreviewSettings

    width, height = size
    record = record if record is not None else reference_record()
    settings = settings if settings is not None else PreviewSettings()
    view, projection = reference_camera(
        width, height, orthographic=view_name.endswith("_ortho"))
    own_renderer = renderer is None
    renderer = renderer or LiquidPreviewRenderer()
    target = gpu.types.GPUOffScreen(width, height, format="RGBA8")
    try:
        with target.bind():
            framebuffer = gpu.state.active_framebuffer_get()
            framebuffer.clear(color=BACKGROUND, depth=1.0)
            gpu.state.viewport_set(0, 0, width, height)
            draw_backdrop(gpu, view, projection)
            renderer.draw(record, view_matrix=view, projection_matrix=projection,
                          viewport=(0, 0, width, height), settings=settings)
            buffer = framebuffer.read_color(0, 0, width, height, 4, 0, "UBYTE")
            buffer.dimensions = width * height * 4
            pixels = np.array(buffer, dtype=np.uint8)
    finally:
        target.free()
        if own_renderer:
            renderer.free()
    return pixels.reshape(height, width, 4)[::-1].copy()
