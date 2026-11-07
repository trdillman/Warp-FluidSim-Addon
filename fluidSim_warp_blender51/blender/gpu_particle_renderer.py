"""FluidSim texture-backed point renderer.

The caller owns the supplied textures and must draw on Blender's GPU thread.
"""
from collections import namedtuple

Packet = namedtuple('Packet', 'texture count revision generation width height radius color color_texture')


def prepare_packet(frame, *, width, height, radius, color, color_texture=None, color_size=None):
    width, height, count = int(width), int(height), int(frame.count)
    if width <= 0 or height <= 0 or count < 0 or count > width * height:
        raise ValueError('invalid texture dimensions or active particle count')
    if radius <= 0 or len(color) != 4:
        raise ValueError('radius must be positive and color must contain four values')
    if color_texture is not None and tuple(color_size or ()) != (width, height):
        raise ValueError('color texture dimensions must match positions')
    return Packet(frame.texture, count, frame.revision, frame.generation,
                  width, height, float(radius), tuple(color), color_texture)


class PointRenderer:
    """FluidSim adapter to the optional PRK external texture renderer."""
    def __init__(self):
        self._renderer = None

    def draw(self, packet, point_scale=1.0):
        if packet.color_texture is not None:
            raise ValueError('PRK external point renderer supports flat color only')
        if self._renderer is None:
            from prk.gpudraw import ExternalTexturePoints
            self._renderer = ExternalTexturePoints()
        self._renderer.draw(texture=packet.texture, width=packet.width,
                            height=packet.height, count=packet.count,
                            revision=packet.revision, generation=packet.generation,
                            radius=packet.radius, color=packet.color,
                            point_scale=point_scale)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
