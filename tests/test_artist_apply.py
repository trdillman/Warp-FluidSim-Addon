"""Restart decisions follow the registry and preserve progress on cancellation."""
import pytest
from types import SimpleNamespace
from test_phase4_pending_interaction import _bridge, _scene, _activate


def test_compatible_resume_keeps_the_solver(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, solver = _activate(bridge, scene)
    assert not bridge.pending.changes_require_restart(scene)
    assert bridge.pending.apply_changes(scene) is False
    assert sess.solver is solver


def test_rebuild_requires_consent_then_applies_once(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, solver = _activate(bridge, scene)
    scene.frame_current = 5
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.05
    assert bridge.pending.changes_require_restart(scene)
    with pytest.raises(RuntimeError, match='Restart required'):
        bridge.pending.apply_changes(scene)
    assert sess.solver is solver
    assert scene.frame_current == 5
    calls = []
    def initialize(scene, **kwargs):
        calls.append(scene.frame_current)
        sess.solver = SimpleNamespace(frame=0, settings=solver.settings)
        bridge.pending.capture_applied_state(scene, sess)
    monkeypatch.setattr(sess, 'initialize', initialize)
    assert bridge.pending.apply_changes(scene, allow_restart=True)
    assert calls == [1]
    assert scene.frame_current == 1
    assert not bridge.pending.apply_changes(scene)


def test_failed_replacement_restores_frame_and_keeps_pending(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, solver = _activate(bridge, scene)
    scene.frame_current = 8
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.05
    def fail(*a, **kw):
        raise RuntimeError('candidate rejected')
    monkeypatch.setattr(sess, 'initialize', fail)
    with pytest.raises(RuntimeError, match='candidate rejected'):
        bridge.pending.apply_changes(scene, allow_restart=True)
    assert scene.frame_current == 8
    assert sess.solver is solver
    assert bridge.pending.changes_require_restart(scene)


@pytest.mark.parametrize('released', [False, True])
def test_explicit_restart_respects_rebuild_and_released_session(monkeypatch, released):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, solver = _activate(bridge, scene)
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.05
    if released:
        sess.solver = None
    calls = []
    def initialize(scene, **kwargs):
        calls.append('initialize')
        sess.solver = solver
        bridge.pending.capture_applied_state(scene, sess)
    monkeypatch.setattr(sess, 'initialize', initialize)
    monkeypatch.setattr(sess, 'reset', lambda *a, **kw: calls.append('reset'))
    assert bridge.pending.apply_changes(scene, allow_restart=True, force_restart=True)
    assert calls == ['initialize']
    assert bridge.pending.pending_presentation(scene)['action'] == 'NONE'


def test_fresh_preview_admits_at_scene_start_not_playhead(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    scene.frame_current = 27
    sess = bridge.session()
    from fluidSim_warp_blender51.blender import playback
    monkeypatch.setattr(playback, '_bridge', lambda: bridge)
    frames = []
    monkeypatch.setattr(sess, 'initialize', lambda scene, **kw: frames.append(scene.frame_current))
    playback.prepare_live_preview(scene)
    assert frames == [1]
    assert scene.frame_current == 1


@pytest.mark.parametrize('restart', [False, True])
def test_apply_cannot_allocate_behind_cache_playback(monkeypatch, restart):
    bridge = _bridge(monkeypatch)
    scene = _scene()
    sess, solver = _activate(bridge, scene)
    scene.fluidsim.domain_object.fluidsim_domain.voxel_size = 0.05
    scene.fluidsim.cache_playback_enabled = True
    sess.solver = None
    calls = []
    monkeypatch.setattr(sess, 'initialize', lambda *a, **kw: calls.append('initialize'))
    with pytest.raises(RuntimeError, match='Cache Preview'):
        bridge.pending.apply_changes(scene, allow_restart=True, force_restart=restart)
    assert not calls
    assert sess.solver is None
    assert scene.fluidsim.cache_playback_enabled


def test_guided_setup_cannot_mutate_a_baking_scene(monkeypatch):
    _bridge(monkeypatch)
    from fluidSim_warp_blender51.blender import production_bake, setup_wizard
    monkeypatch.setattr(production_bake, 'bake_is_active', lambda scene: True)
    with pytest.raises(RuntimeError, match='Production Bake'):
        setup_wizard.setup_selected(SimpleNamespace(scene=_scene()), 'QUICK_LIQUID')
