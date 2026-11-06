import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('sdk_interop_test', Path(__file__).parents[1] / 'fluidSim_warp_blender51/blender/sdk_interop.py')
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class Native:
    def __init__(self): self.shown = None; self.closed = 0
    def can_publish(self): return True
    def publish(self, positions): return True
    def positions_texture(self): return self.shown
    def close(self): self.closed += 1; return 'completed'

def test_frame_count_tracks_published_revision_and_dimensions():
    native = Native()
    adapter = module.SDKTransport(8, sdk=SimpleNamespace(create_transport=lambda n: native), pack=lambda positions, count, step: positions)
    assert adapter.publish([None] * 8, 7, 1)
    assert adapter.publish([None] * 8, 3, 1)
    native.shown = SimpleNamespace(texture='texture', count=8, width=4096, height=1, generation=5, frame=0)
    packet = adapter.frame()
    assert (packet.texture, packet.count, packet.width, packet.height, packet.generation, packet.revision) == ('texture', 7, 4096, 1, 5, 0)
    native.shown.frame = 1
    assert adapter.frame().count == 3
    assert adapter.close() == adapter.close() == 'completed'
    assert native.closed == 1

def test_close_error_quarantines_and_blocks_recreation():
    native = Native()
    def fail(): raise RuntimeError('cleanup failed')
    native.close = fail
    adapter = module.SDKTransport(8, sdk=SimpleNamespace(create_transport=lambda n: native), pack=lambda *args: None)
    with pytest.raises(RuntimeError): adapter.close()
    assert adapter.close() == 'failed'
    with pytest.raises(RuntimeError, match='restart Blender'):
        module.SDKTransport(8, sdk=SimpleNamespace(create_transport=lambda n: native))
    sys.modules.pop('_fluidsim_sdk_transport_state', None)

def test_real_warp_pack_strides_and_clears_tail():
    import numpy as np
    import warp as wp
    source = wp.array(np.arange(18, dtype=np.float32).reshape(6, 3), dtype=wp.vec3, device='cpu')
    target = wp.empty(3, dtype=wp.vec3, device='cpu')
    wp.launch(module._make_pack(wp), dim=3, inputs=[source, target, 3, 2], device='cpu')
    np.testing.assert_array_equal(target.numpy(), [[0, 1, 2], [6, 7, 8], [0, 0, 0]])
