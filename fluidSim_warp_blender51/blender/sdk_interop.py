"""FluidSim ownership and count adapter for the Vulkan interop SDK."""
from dataclasses import dataclass
import sys
import types


def _state():
    name = '_fluidsim_sdk_transport_state'
    if name not in sys.modules:
        state = types.ModuleType(name)
        state.quarantine = []
        state.restart_required = False
        sys.modules[name] = state
    return sys.modules[name]


@dataclass(frozen=True)
class TexturePacket:
    texture: object
    count: int
    width: int
    height: int
    revision: int
    generation: int


def _make_pack(wp):
    def pack(source, target, count, step):
        index = wp.tid()
        if index * step < count:
            target[index] = source[index * step]
        else:
            target[index] = wp.vec3(0.0)
    pack.__annotations__ = {
        'source': wp.array(dtype=wp.vec3), 'target': wp.array(dtype=wp.vec3),
        'count': int, 'step': int,
    }
    return wp.kernel(pack)


class SDKTransport:
    """Own one fixed-capacity transport; acquire frames only in its draw context."""
    def __init__(self, capacity, sdk=None, pack=None):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError('capacity must be a positive integer')
        if _state().restart_required:
            raise RuntimeError('interop resources quarantined; restart Blender')
        if sdk is None:
            import blender_vulkan_interop as sdk
        self.capacity = capacity
        self._transport = sdk.create_transport(capacity)
        self._closed = False
        self._counts = {}
        self._next_revision = 0
        self._shown_revision = None
        self._pack_override = pack
        self._buffer = self._kernel = None

    def can_publish(self):
        if _state().restart_required:
            raise RuntimeError('interop resources quarantined; restart Blender')
        return not self._closed and self._transport.can_publish()

    def publish(self, positions, count, step=1):
        if self._closed:
            return False
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('active count must be a nonnegative integer')
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError('step must be a positive integer')
        if count > len(positions) or (count + step - 1) // step > self.capacity:
            raise ValueError('active count exceeds transport capacity')
        if not self.can_publish():
            return False
        if self._pack_override is not None:
            packed = self._pack_override(positions, count, step)
        elif step == 1 and len(positions) == self.capacity:
            packed = positions
        else:
            import warp as wp
            if self._buffer is None:
                self._buffer = wp.empty(self.capacity, dtype=wp.vec3, device=positions.device)
                self._kernel = _make_pack(wp)
            wp.launch(self._kernel, dim=self.capacity,
                      inputs=[positions, self._buffer, count, step], device=positions.device)
            packed = self._buffer
        if not self._transport.publish(packed):
            return False
        self._counts[self._next_revision] = (count + step - 1) // step
        self._next_revision += 1
        return True

    def frame(self):
        if self._closed:
            return None
        if _state().restart_required:
            raise RuntimeError('interop resources quarantined; restart Blender')
        frame = self._transport.positions_texture()
        if frame is None:
            return None
        if frame.frame not in self._counts:
            raise RuntimeError('SDK returned an unknown frame revision')
        count = self._counts[frame.frame]
        self._shown_revision = frame.frame
        self._counts = {revision: n for revision, n in self._counts.items()
                        if revision >= frame.frame}
        return TexturePacket(frame.texture, count, frame.width, frame.height,
                             frame.frame, frame.generation)

    def close(self):
        if not self._closed:
            try:
                result = self._transport.close()
            except Exception:
                _state().quarantine.append(self)
                _state().restart_required = True
                self._closed = True
                self._close_result = 'failed'
                raise
            self._close_result = result
            self._closed = True
            if result != 'completed':
                _state().quarantine.append(self)
                _state().restart_required = True
        return self._close_result
