import importlib.util
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).parents[1] / 'fluidSim_warp_blender51' / 'blender' / 'gpu_particle_owner.py'
spec = importlib.util.spec_from_file_location('gpu_particle_owner', path)
owner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner_module)

class Transport:
    made = []
    def __init__(self, capacity):
        self.capacity = capacity
        self.closed = False
        self.published = []
        self.__class__.made.append(self)
    def publish(self, positions, count, step=1):
        self.published.append((positions, count, step))
    def close(self):
        self.closed = True
        return 'completed'


def test_reuse_and_recreate_on_capacity_or_stride_change():
    owner = owner_module.ParticleOwner(Transport)
    particles = SimpleNamespace(capacity=100, host_count=9, position=object(), refresh_host_count=lambda: 9)
    owner.publish(particles, 2, 9)
    first = owner.transport
    assert first.capacity == 50
    assert first.published == [(particles.position, 9, 2)]
    owner.publish(particles, 2, 9)
    assert owner.transport is first
    owner.publish(particles, 4, 9)
    assert first.closed
    assert owner.transport.capacity == 25
    owner.close()
    assert owner.transport is None


def test_publish_scene_falls_back_for_attribute_colors(monkeypatch):
    import sys
    timers = SimpleNamespace(is_registered=lambda fn: False, register=lambda fn, **kw: None)
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(app=SimpleNamespace(timers=timers)))
    transport = Transport(4)
    owner = owner_module.ParticleOwner(Transport)
    owner.transport = transport
    owner_module._owner = owner
    fs = SimpleNamespace(gpu_particle_preview=True, show_particles=True,
                         particle_attribute_mode='SPEED')
    owner_module._publish_now(SimpleNamespace(fluidsim=fs), object(), 1)
    assert not transport.closed
    assert owner_module._owner is None
    owner_module._retire_from_timer()
    assert transport.closed


def test_failed_publish_closes_transport():
    class Broken(Transport):
        def publish(self, *args, **kwargs):
            raise RuntimeError('publish failed')
    owner = owner_module.ParticleOwner(Broken)
    particles = SimpleNamespace(capacity=3, host_count=1, position=object(), refresh_host_count=lambda: 1)
    import pytest
    with pytest.raises(RuntimeError, match='publish failed'):
        owner.publish(particles, 1, 1)
    assert owner.transport is None
    assert Broken.made[-1].closed


def test_restore_output_uses_original_view_layer_and_visibility():
    calls = []
    obj = SimpleNamespace(hide_set=lambda state, **kw: calls.append((state, kw['view_layer'])))
    layer = object()
    owner_module._hidden = (obj, layer, False)
    owner_module._restore_output()
    assert calls == [(False, layer)]
    assert owner_module._hidden is None


def test_failed_retirement_blocks_recreation():
    class Quarantined(Transport):
        def close(self): return 'quarantined'
    owner = owner_module.ParticleOwner(Quarantined)
    particles = SimpleNamespace(capacity=4, host_count=1, position=object(), refresh_host_count=lambda: 1)
    owner.publish(particles, 1, 1)
    import pytest
    try:
        with pytest.raises(RuntimeError, match='restart Blender'):
            owner.close()
        with pytest.raises(RuntimeError, match='restart Blender'):
            owner.publish(particles, 2, 1)
    finally:
        owner_module._restart_required = False


def test_paused_disable_restores_output_without_stepping_and_defers_close(monkeypatch):
    import sys
    timers = SimpleNamespace(is_registered=lambda fn: False, register=lambda fn, **kw: None)
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(app=SimpleNamespace(timers=timers)))
    calls = []
    obj = SimpleNamespace(hide_set=lambda state, **kw: calls.append(('visible', state)))
    owner = SimpleNamespace(close=lambda: calls.append(('closed',)))
    owner_module._owner = owner
    owner_module._hidden = (obj, object(), False)
    owner_module._retire_pending = False
    try:
        owner_module.disable_preview()
        assert calls == [('visible', False)]
        assert owner_module._retire_pending
        assert owner_module._owner is None
        assert owner_module._retired[-1][0] is owner
    finally:
        owner_module._owner = None
        owner_module._hidden = None
        owner_module._retired.clear()
        owner_module._retire_pending = False


def test_publish_timer_ignores_replaced_solver():
    solver = object()
    session = SimpleNamespace(solver=object(), owns_scene=lambda scene: True)
    owner_module._pending_frame = (object(), session, solver, 4)
    assert owner_module._publish_from_timer() is None
    assert owner_module._pending_frame is None


def test_missing_prk_falls_back_without_creating_transport(monkeypatch):
    import sys
    from types import ModuleType
    prk = ModuleType('prk'); gpudraw = ModuleType('prk.gpudraw')
    monkeypatch.setitem(sys.modules, 'prk', prk)
    monkeypatch.setitem(sys.modules, 'prk.gpudraw', gpudraw)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        gpu_particle_preview=True, show_particles=True,
        particle_attribute_mode='NONE', particle_material=None, status='Ready'))
    owner_module._publish_now(scene, SimpleNamespace(solver=object()), 1)
    assert owner_module._owner is None
    assert 'GPU preview unavailable' in scene.fluidsim.status


def test_reset_retains_solver_until_timer_retirement(monkeypatch):
    import sys
    timers = SimpleNamespace(is_registered=lambda fn: False, register=lambda fn, **kw: None)
    monkeypatch.setitem(sys.modules, 'bpy', SimpleNamespace(app=SimpleNamespace(timers=timers)))
    owner = owner_module.ParticleOwner(Transport)
    owner.transport = Transport(4)
    solver = object()
    owner_module._owner = owner
    owner_module._owner_solver = solver
    owner_module._pending_frame = ('stale',)
    owner_module.reset_runtime()
    assert owner_module._pending_frame is None
    assert owner_module._retired[-1] == (owner, None, solver)
    assert not owner.transport.closed
    owner_module._retire_from_timer()
    assert owner.transport is None and not owner_module._retired


def test_load_pre_retires_active_owner_and_clears_cancelled_timer(monkeypatch):
    import sys
    timers = SimpleNamespace(is_registered=lambda fn: False, register=lambda fn, **kw: None)
    monkeypatch.setitem(sys.modules, 'bpy', SimpleNamespace(app=SimpleNamespace(timers=timers)))
    owner = owner_module.ParticleOwner(Transport)
    transport = Transport(4)
    owner.transport = transport
    owner_module._owner = owner
    owner_module._owner_solver = object()
    owner_module._pending_frame = ('stale',)
    owner_module._on_gpu_preview_load_pre(None)
    assert transport.closed
    assert owner_module._owner is None and owner_module._pending_frame is None
    assert not owner_module._retire_pending and not owner_module._retired


def test_save_restores_visibility_and_suspends_drawing_until_completion():
    calls = []
    obj = SimpleNamespace(hide_set=lambda state, **kw: calls.append(state))
    owner_module._hidden = (obj, object(), False)
    owner_module._on_gpu_preview_save_pre(None)
    assert calls == [False] and owner_module._hidden is None
    assert owner_module._saving
    owner_module._on_gpu_preview_save_post(None)
    assert not owner_module._saving


def test_failed_save_also_resumes_preview():
    owner_module._on_gpu_preview_save_pre(None)
    assert owner_module._saving
    owner_module._on_gpu_preview_save_post_fail(None)
    assert not owner_module._saving


def test_renderer_retirement_failure_blocks_new_transport(monkeypatch):
    import pytest

    def fail():
        raise RuntimeError('renderer retirement failed')

    monkeypatch.setattr(owner_module, '_retired', [(None, SimpleNamespace(close=fail), object())])
    monkeypatch.setattr(owner_module, '_quarantined', [])
    monkeypatch.setattr(owner_module, '_restart_required', False)
    owner_module._retire_from_timer()
    assert owner_module._restart_required
    assert len(owner_module._quarantined) == 1
    made = []
    owner = owner_module.ParticleOwner(lambda capacity: made.append(capacity))
    with pytest.raises(RuntimeError, match='restart Blender'):
        owner.publish(SimpleNamespace(capacity=1), 1, 1)
    assert made == []


def test_visibility_restoration_obeys_current_policy_and_export_hold(monkeypatch):
    import sys
    from types import ModuleType

    package = ModuleType('preview_policy_test')
    policy = SimpleNamespace(_HELD=0, apply_viewport=lambda scene: queued.append(scene))
    package.liquid_display = policy
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setattr(owner_module, '__package__', package.__name__)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(liquid_display_mode='SURFACE', show_particles=True))
    monkeypatch.setattr(owner_module, '_owner_scene', scene)
    monkeypatch.setattr(owner_module, '_particles_visible',
                        lambda scene: scene.fluidsim.liquid_display_mode == 'SURFACE')
    writes, queued = [], []
    obj = SimpleNamespace(hide_set=lambda state, **kw: writes.append(state))
    for mode in ('OFF', 'PREVIEW', 'SURFACE'):
        scene.fluidsim.liquid_display_mode = mode
        assert owner_module._preview_particles_visible(scene) == (mode == 'SURFACE')
        owner_module._hidden = (obj, object(), False)
        owner_module._restore_output()
        assert writes[-1] == (mode != 'SURFACE')
    policy._HELD = 1
    owner_module._hidden = (obj, object(), False)
    owner_module._restore_output()
    assert len(writes) == 3 and queued == [scene]
    assert owner_module._hidden is None
