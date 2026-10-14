"""Liquid Preview passes: depth sprites, narrow-range filter, thickness, composite.

The pure functions at the top (filter radius, offscreen size, absorption,
blue-noise tile, uniform packing) carry the math and are unit tested without a
GPU. :class:`LiquidPreviewRenderer` owns every GPU resource and is the only
code that imports ``gpu``. It never reads the solver: its sole input is the
immutable :class:`PublishedPoints` record, uploaded once per ``frame_id``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from ..output_publication import _point_capacity_for
from . import shaders

MAX_FILTER_RADIUS = 24
REFRACTION_KAPPA = 0.05
DEPTH_RANGE_RADII = 10.0  # delta = 10 r, the narrow-range clamp window
BLUE_NOISE_SIZE = 64
_OFFSCREEN_CACHE_LIMIT = 4
_QUAD = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))


def filter_radius_pixels(sigma, height, depth, tan_half, *,
                         max_radius=MAX_FILTER_RADIUS):
    """Screen-space filter radius ``ceil(H*sigma / (2|z| tan(alpha/2)))``.

    ``sigma`` is the world-space filter size, ``height`` the target height in
    pixels, ``depth`` the eye-space depth and ``tan_half`` the tangent of half
    the vertical field of view (``1/proj[1][1]``; for orthographic views pass
    the half view height with ``depth=1``). Clamped to ``max_radius``.
    """
    depth = abs(float(depth))
    if sigma <= 0.0 or height <= 0 or depth <= 0.0 or tan_half <= 0.0:
        return 0
    radius = math.ceil(height * float(sigma) / (2.0 * depth * tan_half))
    return int(min(radius, max_radius))


def offscreen_size(width, height, scale):
    """Offscreen target size: ``scale`` x viewport, at least one pixel."""
    scale = min(max(float(scale), 0.0), 1.0)
    return (max(1, int(round(width * scale))),
            max(1, int(round(height * scale))))


def absorption_coefficients(color, distance=1.0):
    """Beer-Lambert ``a`` so ``exp(-a * distance)`` is the given colour."""
    rgb = np.clip(np.asarray(color, dtype=np.float64)[:3], 1e-3, 1.0)
    return tuple(float(value) for value in -np.log(rgb) / float(distance))


@lru_cache(maxsize=1)
def _blue_noise_ranks(size=BLUE_NOISE_SIZE, sigma=1.5):
    """Rank every texel by repeatedly filling the largest void.

    A toroidal Gaussian energy map accumulates around each placed texel; the
    next rank goes to the lowest-energy free texel. Deterministic: ties go to
    the first texel in row-major order, and the first rank is texel (0, 0).
    """
    offsets = np.minimum(np.arange(size), size - np.arange(size))
    kernel = np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2)
                    / (2.0 * sigma * sigma))
    energy = np.zeros((size, size))
    ranks = np.empty(size * size, dtype=np.int64)
    for rank in range(size * size):
        index = int(np.argmin(energy))
        y, x = divmod(index, size)
        ranks[index] = rank
        energy += np.roll(np.roll(kernel, y, axis=0), x, axis=1)
        energy[y, x] = np.inf
    return ranks.reshape(size, size)


def blue_noise_tile():
    """The 64x64 blue-noise tile as float32 ranks in ``[0, 1)``."""
    ranks = _blue_noise_ranks()
    return (ranks / float(ranks.size)).astype(np.float32)


@dataclass(frozen=True)
class PreviewSettings:
    """The scene display settings one redraw needs, read once per draw."""

    scale: float = 0.5
    radius_multiplier: float = 2.0
    absorption: tuple[float, float, float] = (0.25, 0.6, 0.75)
    whitewater: bool = True


def preview_settings(fs):
    """Read the scene display group (``scene.fluidsim``) once per redraw."""
    return PreviewSettings(
        scale=float(fs.liquid_preview_scale),
        radius_multiplier=float(fs.liquid_preview_radius_multiplier),
        absorption=tuple(float(c) for c in fs.liquid_preview_absorption)[:3],
        whitewater=bool(fs.liquid_preview_whitewater),
    )


def _matrix(values):
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def is_perspective(projection):
    """Blender projection matrices are perspective when ``m[3][3] == 0``."""
    return abs(float(_matrix(projection)[3, 3])) < 1e-9


def pack_params(view, projection, target_size, viewport, record, settings):
    """Pack the ``LiquidPreviewParams`` uniform block (std140, vec4-aligned)."""
    view = _matrix(view)
    projection = _matrix(projection)
    width, height = target_size
    radius = float(record.particle_radius) * float(settings.radius_multiplier)
    whitewater_radius = (float(record.whitewater_radius)
                         * float(settings.radius_multiplier))
    perspective = is_perspective(projection)
    tan_half = 1.0 / max(abs(float(projection[1, 1])), 1e-9)
    delta = DEPTH_RANGE_RADII * radius
    absorb = absorption_coefficients(settings.absorption)
    frame = float((record.scene_frame or 0) % BLUE_NOISE_SIZE)
    columns = [np.asarray(m, dtype=np.float32).T.ravel() for m in (
        view, projection, np.linalg.inv(projection), np.linalg.inv(view))]
    vectors = np.array([
        (width, height, 1.0 / width, 1.0 / height),
        (radius, 1.0 if perspective else 0.0, tan_half, whitewater_radius),
        (delta, delta, radius, MAX_FILTER_RADIUS),
        (*absorb, REFRACTION_KAPPA),
        (*settings.absorption[:3], frame),
        tuple(float(v) for v in viewport),
    ], dtype=np.float32).ravel()
    return np.concatenate([*columns, vectors]).astype(np.float32)


class _Target:
    """A colour texture (plus optional depth) behind one ``GPUFrameBuffer``.

    Not a ``GPUOffScreen``: in Blender 5.2 creating an offscreen leaves no
    framebuffer bound, which would drop the composite drawn after it on the
    floor. Creating textures and framebuffers does not touch the binding.
    """

    def __init__(self, gpu, size, color_format, *, with_depth=False):
        self.texture_color = gpu.types.GPUTexture(size, format=color_format)
        self.texture_depth = (
            gpu.types.GPUTexture(size, format="DEPTH_COMPONENT32F")
            if with_depth else None)
        self.framebuffer = gpu.types.GPUFrameBuffer(
            depth_slot=self.texture_depth, color_slots=self.texture_color)

    def bind(self):
        return self.framebuffer.bind()

    def free(self):
        self.framebuffer = self.texture_color = self.texture_depth = None


class LiquidPreviewRenderer:
    """GPU resources and passes for one Blender session.

    ``draw`` renders into whatever framebuffer is bound (the viewport in a
    ``POST_VIEW`` handler, an offscreen in tests and profiling) and uses its
    depth buffer as the scene depth, so solids in front occlude the liquid.
    """

    def __init__(self):
        self.uploads = 0
        self._shaders = None
        self._frame_id = None
        self._particles = None
        self._particle_count = 0
        self._whitewater = None
        self._whitewater_count = 0
        self._scratch = np.zeros((0, 3), dtype=np.float32)
        self._targets = OrderedDict()
        self._quad = None
        self._noise = None
        self._ubo = None

    # --- resources ---------------------------------------------------------

    def _ensure_static(self, gpu):
        if self._shaders is None:
            self._shaders = shaders.build_shaders(gpu)
        if self._quad is None:
            fmt = gpu.types.GPUVertFormat()
            fmt.attr_add(id="co", comp_type="F32", len=2, fetch_mode="FLOAT")
            vbo = gpu.types.GPUVertBuf(fmt, len(_QUAD))
            vbo.attr_fill("co", _QUAD)
            self._quad = gpu.types.GPUBatch(type="TRI_STRIP", buf=vbo)
        if self._noise is None:
            tile = blue_noise_tile()
            data = gpu.types.Buffer("FLOAT", tile.size, tile.ravel())
            self._noise = gpu.types.GPUTexture(
                (BLUE_NOISE_SIZE, BLUE_NOISE_SIZE), format="R32F", data=data)

    def _upload(self, gpu, record):
        """Copy one published frame into fresh, capacity-sized VBOs.

        Blender's ``GPUVertBuf`` cannot be refilled once drawn, so a new
        buffer is made per frame id; it is sized by the publication capacity
        planner so the allocation does not change with every live count.
        """
        count = record.particle_count
        self._particles = None
        if count:
            capacity = _point_capacity_for(count)
            if len(self._scratch) < capacity:
                self._scratch = np.zeros((capacity, 3), dtype=np.float32)
            scratch = self._scratch[:capacity]
            scratch[:count] = record.positions
            fmt = gpu.types.GPUVertFormat()
            fmt.attr_add(id="pos", comp_type="F32", len=3, fetch_mode="FLOAT")
            vbo = gpu.types.GPUVertBuf(fmt, capacity)
            vbo.attr_fill("pos", scratch)
            self._particles = gpu.types.GPUBatch(type="POINTS", buf=vbo)
        self._particle_count = count

        ww_count = record.whitewater_count
        self._whitewater = None
        if ww_count:
            types = (np.zeros(ww_count, dtype=np.float32)
                     if record.whitewater_types is None
                     else np.asarray(record.whitewater_types, dtype=np.float32))
            fmt = gpu.types.GPUVertFormat()
            fmt.attr_add(id="pos", comp_type="F32", len=3, fetch_mode="FLOAT")
            fmt.attr_add(id="wtype", comp_type="F32", len=1, fetch_mode="FLOAT")
            vbo = gpu.types.GPUVertBuf(fmt, ww_count)
            vbo.attr_fill("pos", np.ascontiguousarray(record.whitewater))
            vbo.attr_fill("wtype", types)
            self._whitewater = gpu.types.GPUBatch(type="POINTS", buf=vbo)
        self._whitewater_count = ww_count
        self._frame_id = record.frame_id
        self.uploads += 1

    def _targets_for(self, gpu, size):
        targets = self._targets.pop(size, None)
        if targets is None:
            targets = {
                "depth": _Target(gpu, size, "R32F", with_depth=True),
                "ping": _Target(gpu, size, "R32F"),
                "thickness": _Target(gpu, size, "R16F"),
            }
        self._targets[size] = targets
        while len(self._targets) > _OFFSCREEN_CACHE_LIMIT:
            _size, stale = self._targets.popitem(last=False)
            for target in stale.values():
                target.free()
        return targets

    def free(self):
        """Release every GPU resource (handler removal, add-on unload)."""
        for targets in self._targets.values():
            for target in targets.values():
                target.free()
        self._targets.clear()
        self._particles = self._whitewater = None
        self._quad = self._noise = self._ubo = None
        self._shaders = None
        self._frame_id = None

    # --- passes ------------------------------------------------------------

    def draw(self, record, *, view_matrix, projection_matrix, viewport,
             settings):
        """Render ``record`` into the bound framebuffer; GPU state restored."""
        import gpu

        width, height = int(viewport[2]), int(viewport[3])
        if record is None or width <= 0 or height <= 0:
            return False
        self._ensure_static(gpu)
        if record.frame_id != self._frame_id:
            self._upload(gpu, record)
        draw_whitewater = bool(settings.whitewater and self._whitewater)
        if self._particles is None and not draw_whitewater:
            return False

        state = gpu.state
        size = offscreen_size(width, height, settings.scale)
        params = pack_params(view_matrix, projection_matrix, size, viewport,
                             record, settings)
        if self._ubo is None:
            self._ubo = gpu.types.GPUUniformBuf(params)
        else:
            self._ubo.update(params)
        targets = self._targets_for(gpu, size)
        saved = (state.blend_get(), state.depth_test_get(),
                 state.depth_mask_get(), tuple(state.viewport_get()))
        try:
            state.program_point_size_set(True)
            if self._particles is not None:
                self._depth_pass(gpu, targets, size)
                self._thickness_pass(gpu, targets, size)
                liquid_depth = self._filter_passes(gpu, targets, size)
            else:
                liquid_depth = self._clear_depth(gpu, targets, size)
            state.viewport_set(*saved[3])
            if self._particles is not None:
                self._composite(gpu, liquid_depth, targets["thickness"])
            if draw_whitewater:
                self._whitewater_pass(gpu, liquid_depth)
        finally:
            state.program_point_size_set(False)
            state.blend_set(saved[0])
            state.depth_test_set(saved[1])
            state.depth_mask_set(saved[2])
            state.viewport_set(*saved[3])
        return True

    def _bind_shader(self, name):
        shader = self._shaders[name]
        shader.bind()
        shader.uniform_block("params", self._ubo)
        return shader

    def _depth_pass(self, gpu, targets, size):
        with targets["depth"].bind():
            gpu.state.viewport_set(0, 0, *size)
            gpu.state.active_framebuffer_get().clear(
                color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            gpu.state.blend_set("NONE")
            gpu.state.depth_test_set("LESS")
            gpu.state.depth_mask_set(True)
            shader = self._bind_shader("depth")
            self._particles.draw_range(
                shader, elem_start=0, elem_count=self._particle_count)

    def _thickness_pass(self, gpu, targets, size):
        with targets["thickness"].bind():
            gpu.state.viewport_set(0, 0, *size)
            gpu.state.active_framebuffer_get().clear(
                color=(0.0, 0.0, 0.0, 0.0))
            gpu.state.depth_test_set("NONE")
            gpu.state.depth_mask_set(False)
            gpu.state.blend_set("ADDITIVE_PREMULT")
            shader = self._bind_shader("thickness")
            shader.uniform_sampler("noise_tex", self._noise)
            self._particles.draw_range(
                shader, elem_start=0, elem_count=self._particle_count)

    def _fullscreen(self, gpu, target, size, name, source, **push):
        with target.bind():
            gpu.state.viewport_set(0, 0, *size)
            gpu.state.depth_test_set("NONE")
            gpu.state.depth_mask_set(False)
            gpu.state.blend_set("NONE")
            shader = self._bind_shader(name)
            shader.uniform_sampler("depth_tex", source)
            for key, value in push.items():
                shader.uniform_float(key, value)
            self._quad.draw(shader)
        return target.texture_color

    def _filter_passes(self, gpu, targets, size):
        """Narrow-range filter: horizontal, vertical, then a 2D clean-up."""
        depth, ping = targets["depth"], targets["ping"]
        self._fullscreen(gpu, ping, size, "filter", depth.texture_color,
                         filter_dir=(1.0, 0.0))
        self._fullscreen(gpu, depth, size, "filter", ping.texture_color,
                         filter_dir=(0.0, 1.0))
        return self._fullscreen(gpu, ping, size, "cleanup",
                                depth.texture_color)

    def _clear_depth(self, gpu, targets, size):
        with targets["ping"].bind():
            gpu.state.viewport_set(0, 0, *size)
            gpu.state.active_framebuffer_get().clear(
                color=(0.0, 0.0, 0.0, 0.0))
        return targets["ping"].texture_color

    def _composite(self, gpu, liquid_depth, thickness):
        gpu.state.blend_set("ALPHA_PREMULT")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)
        shader = self._bind_shader("composite")
        shader.uniform_sampler("depth_tex", liquid_depth)
        shader.uniform_sampler("thickness_tex", thickness.texture_color)
        self._quad.draw(shader)

    def _whitewater_pass(self, gpu, liquid_depth):
        gpu.state.blend_set("ALPHA_PREMULT")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)
        shader = self._bind_shader("whitewater")
        shader.uniform_sampler("depth_tex", liquid_depth)
        self._whitewater.draw_range(
            shader, elem_start=0, elem_count=self._whitewater_count)
