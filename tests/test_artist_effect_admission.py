"""Draft effects enter admission without prematurely publishing applied evidence."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_operator_panel_behavior import _install_bpy_stub, _reload_module
from test_simulation_admission import _Runtime, _Session, _Snapshot
from fluidSim_warp_blender51.blender.simulation_admission import SimulationAdmission
from fluidSim_warp_blender51.sim.effects.builtins import builtin_effect_definitions
from fluidSim_warp_blender51.sim.effects.schema import EffectInstanceV2
from fluidSim_warp_blender51.sim.effects.trust import TrustStore


def test_draft_capture_preserves_disabled_authoring_and_rejects_untrusted(monkeypatch, tmp_path):
    _install_bpy_stub(monkeypatch)
    effects = _reload_module('fluidSim_warp_blender51.blender.effects')
    definition = builtin_effect_definitions()[0]
    enabled = EffectInstanceV2('enabled', definition)
    disabled = replace(enabled, instance_id='disabled', enabled=False)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(snippets=[]),
                            fluidsim_effects=SimpleNamespace(migration_complete=True, items=[1], applied_signature=''))
    monkeypatch.setattr(effects, 'trust_store_for_scene', lambda scene: TrustStore(tmp_path / 'trust.json'))
    monkeypatch.setattr(effects, 'scene_effect_instances', lambda scene: (enabled, disabled))
    assert effects.build_effect_stack_for_admission(scene) == (enabled, disabled)
    assert scene.fluidsim_effects.applied_signature == ''
    imported = replace(enabled, definition=replace(definition, built_in=False))
    monkeypatch.setattr(effects, 'scene_effect_instances', lambda scene: (imported,))
    with pytest.raises(RuntimeError, match='trusted'):
        effects.build_effect_stack_for_admission(scene)


@pytest.mark.parametrize('failure', ['none', 'configure', 'output', 'stale'])
def test_effect_evidence_is_published_only_after_successful_admission(monkeypatch, failure):
    sess = _Session()
    old_solver = sess.solver
    scene = SimpleNamespace(fluidsim=SimpleNamespace(status=''), applied='old')
    snapshot = _Snapshot()
    snapshot.effects = ('captured',)
    def publish(target, instances):
        assert target is scene
        assert sess.solver.name == 'new'
        target.applied = instances
    owner = SimulationAdmission(sess, scene_identity=lambda s: 101,
        applied_state_capture=lambda *a, **kw: None, reset_settings_apply=lambda *a: None,
        effect_builder=lambda s: ('changed',) if failure == 'stale' else ('captured',),
        effect_publisher=publish)
    monkeypatch.setattr(owner, '_preflight', lambda *a, **kw: (snapshot, {}, None))
    def prepare(*args):
        if failure == 'configure':
            raise RuntimeError('configure failed')
        return _Runtime('new'), ()
    monkeypatch.setattr(owner, '_prepare_solver_replacement', prepare)
    def outputs(*args):
        if failure == 'output':
            raise RuntimeError('output failed')
    monkeypatch.setattr(owner, '_prepare_outputs', outputs)
    if failure == 'none':
        owner.initialize(scene)
        assert scene.applied == ('captured',)
    else:
        with pytest.raises(RuntimeError):
            owner.initialize(scene)
        assert scene.applied == 'old'
        if failure != 'output':
            assert sess.solver is old_solver
            assert not old_solver.released


def test_successful_replacement_clears_previous_runtime_error(monkeypatch):
    _install_bpy_stub(monkeypatch)
    effects = _reload_module('fluidSim_warp_blender51.blender.effects')
    instance = EffectInstanceV2('enabled', builtin_effect_definitions()[0])
    item = SimpleNamespace(instance_id='enabled', runtime_error='old runtime error',
                           diagnostics=[], kernels=[])
    scene = SimpleNamespace(fluidsim_effects=SimpleNamespace(items=[item]))
    effects.publish_admitted_effects(scene, (instance,))
    assert item.runtime_error == ''
    assert item.status == 'APPLIED'


def test_revoked_trust_pauses_and_cannot_publish_hot_bindings(monkeypatch, tmp_path):
    bpy = _install_bpy_stub(monkeypatch)
    effects = _reload_module('fluidSim_warp_blender51.blender.effects')
    from fluidSim_warp_blender51.blender import scene_bridge, playback
    from fluidSim_warp_blender51.sim.effects.trust import content_trust_digest
    definition = replace(builtin_effect_definitions()[0], built_in=False)
    instance = EffectInstanceV2('imported', definition)
    item = SimpleNamespace(instance_id='imported', runtime_error='', display_name='Imported', enabled=True)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(live=True, snippets=[], cache_playback_enabled=False),
                            fluidsim_effects=SimpleNamespace(items=[item], migration_complete=True))
    store = TrustStore(tmp_path / 'trust.json')
    store.trust(content_trust_digest(definition), 'Imported')
    monkeypatch.setattr(effects, 'trust_store_for_scene', lambda scene: store)
    monkeypatch.setattr(effects, 'scene_effect_instances', lambda scene: (instance,))
    monkeypatch.setattr(effects, 'effect_instance_from_pg', lambda scene, pg: instance)
    monkeypatch.setattr(playback, 'bpy', bpy)
    effects._publish_applied_state(scene, (instance,))
    signature = scene.fluidsim_effects.applied_signature
    bindings = []
    solver = SimpleNamespace(update_custom_effect_runtime_bindings=lambda instances: bindings.append(instances) or True)
    monkeypatch.setattr(scene_bridge, 'session', lambda: SimpleNamespace(active=True, owns_scene=lambda s: True, solver=solver))
    effects.untrust_effect(scene, 0)
    assert not scene.fluidsim.live
    assert scene.fluidsim_effects.applied_signature == signature
    assert not effects._try_live_parameter_update(scene)
    assert not bindings
    assert item.status == 'UNTRUSTED'
