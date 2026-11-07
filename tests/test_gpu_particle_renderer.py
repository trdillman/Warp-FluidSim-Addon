import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

path = Path(__file__).parents[1] / 'fluidSim_warp_blender51' / 'blender' / 'gpu_particle_renderer.py'
spec = importlib.util.spec_from_file_location('gpu_particle_renderer', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_packet_validates_capacity_and_counts():
    frame = SimpleNamespace(texture=object(), count=7, revision=2, generation=1)
    packet = module.prepare_packet(frame, width=4, height=2, radius=0.1, color=(1, 0, 0, 1))
    assert packet.count == 7
    assert packet.width == 4
    with pytest.raises(ValueError):
        module.prepare_packet(frame, width=2, height=3, radius=0.1, color=(1, 0, 0, 1))


def test_color_texture_requires_matching_dimensions():
    frame = SimpleNamespace(texture=object(), count=1, revision=1, generation=1)
    with pytest.raises(ValueError):
        module.prepare_packet(frame, width=4, height=2, radius=0.1, color=(1, 1, 1, 1), color_texture=object(), color_size=(2, 2))



def test_adapter_invokes_external_prk_renderer(monkeypatch):
    import sys
    from types import ModuleType
    calls = []
    class Renderer:
        def draw(self, **kwargs): calls.append(kwargs)
        def close(self): calls.append('closed')
    prk = ModuleType('prk')
    gpudraw = ModuleType('prk.gpudraw')
    gpudraw.ExternalTexturePoints = Renderer
    monkeypatch.setitem(sys.modules, 'prk', prk)
    monkeypatch.setitem(sys.modules, 'prk.gpudraw', gpudraw)
    frame = SimpleNamespace(texture=object(), count=3, revision=2, generation=1)
    packet = module.prepare_packet(frame, width=4, height=2, radius=0.1, color=(1, 0, 0, 1))
    renderer = module.PointRenderer()
    renderer.draw(packet)
    assert calls[0]['texture'] is frame.texture
    assert calls[0]['count'] == 3 and calls[0]['revision'] == 2
    renderer.close()
    assert calls[-1] == 'closed'
