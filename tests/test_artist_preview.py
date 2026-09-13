"""User actions must drive playback without resetting compatible progress."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from test_operator_panel_behavior import _install_bpy_stub, _reload_module


def test_registered_actions_do_not_inherit_other_registered_operators(monkeypatch):
    _install_bpy_stub(monkeypatch)
    ops = _reload_module('fluidSim_warp_blender51.operators')
    registered = set(ops.CLASSES)
    for cls in (ops.FLUIDSIM_OT_preview_start, ops.FLUIDSIM_OT_cache_preview_start,
                ops.FLUIDSIM_OT_apply_changes, ops.FLUIDSIM_OT_restart_simulation):
        assert not registered.intersection(cls.__mro__[1:]), cls.__name__


def preview_fixture(monkeypatch):
    bpy = _install_bpy_stub(monkeypatch)
    ops = _reload_module('fluidSim_warp_blender51.operators')
    from fluidSim_warp_blender51.blender import playback, pending_settings
    scene = SimpleNamespace(fluidsim=SimpleNamespace(live=False, cache_playback_enabled=False, status=''), frame_current=12, frame_start=1)
    screen = SimpleNamespace(
        is_animation_playing=False,
        areas=[SimpleNamespace(
            type='VIEW_3D',
            regions=[SimpleNamespace(type='WINDOW')],
            tag_redraw=lambda: None,
        )],
    )
    window = SimpleNamespace(scene=scene, screen=screen)
    context = SimpleNamespace(scene=scene, window=window, screen=screen, temp_override=lambda **kw: nullcontext(), evaluated_depsgraph_get=lambda: None)
    bpy.context = context
    calls = []
    class Animation:
        def poll(self):
            return True
        def __call__(self, **kwargs):
            calls.append('play')
            screen.is_animation_playing = True
            return {'FINISHED'}
    def cancel(**kwargs):
        assert kwargs.get('restore_frame') is False
        calls.append('pause')
        screen.is_animation_playing = False
        return {'FINISHED'}
    bpy.ops = SimpleNamespace(screen=SimpleNamespace(animation_play=Animation(), animation_cancel=cancel))
    monkeypatch.setattr(playback, 'bpy', bpy)
    monkeypatch.setattr(playback, 'prepare_live_preview', lambda *a, **kw: calls.append('prepare'), raising=False)
    monkeypatch.setattr(pending_settings, 'changes_require_restart', lambda s: False, raising=False)
    return bpy, ops, context, calls


def test_start_is_idempotent_and_pause_preserves_frame(monkeypatch):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    op = ops.FLUIDSIM_OT_preview_start()
    assert op.execute(context) == {'FINISHED'}
    assert context.scene.fluidsim.live and context.screen.is_animation_playing
    assert calls == ['prepare', 'play']
    assert op.execute(context) == {'FINISHED'}
    assert calls.count('play') == 1
    assert ops.FLUIDSIM_OT_preview_pause().execute(context) == {'FINISHED'}
    assert not context.scene.fluidsim.live
    assert not context.screen.is_animation_playing
    assert context.scene.frame_current == 12


def test_cache_action_is_independent_of_blender_instance_identifier(monkeypatch):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    from fluidSim_warp_blender51.blender import playback
    monkeypatch.setattr(playback, 'prepare_cache_preview', lambda scene: calls.append('cache'))
    op = ops.FLUIDSIM_OT_cache_preview_start()
    op.bl_idname = 'FLUIDSIM_OT_cache_preview_start'
    assert op.execute(context) == {'FINISHED'}
    assert calls == ['cache', 'play']
    assert context.scene.fluidsim.cache_playback_enabled


def test_restart_action_is_independent_of_blender_instance_identifier(monkeypatch):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    from fluidSim_warp_blender51.blender import pending_settings
    monkeypatch.setattr(pending_settings, 'apply_changes', lambda scene, **kw: calls.append(kw['force_restart']))
    op = ops.FLUIDSIM_OT_restart_simulation()
    op.bl_idname = 'FLUIDSIM_OT_restart_simulation'
    op.allow_restart = True
    assert op.execute(context) == {'FINISHED'}
    assert calls == [True]


@pytest.mark.parametrize('fault', ['no_window', 'prepare', 'play'])
def test_failed_start_never_leaves_preview_armed(monkeypatch, fault):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    from fluidSim_warp_blender51.blender import playback
    if fault == 'no_window':
        context.window = None
    elif fault == 'prepare':
        def fail(*a, **kw):
            raise RuntimeError('effect rejected')
        monkeypatch.setattr(playback, 'prepare_live_preview', fail)
    else:
        bpy.ops.screen.animation_play = SimpleNamespace(poll=lambda: False)
    assert ops.FLUIDSIM_OT_preview_start().execute(context) == {'CANCELLED'}
    assert not context.scene.fluidsim.live
    assert not context.screen.is_animation_playing
    assert 'play' not in calls


def test_legacy_cache_rejection_cannot_fall_into_live(monkeypatch):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    from fluidSim_warp_blender51.blender import playback
    monkeypatch.setattr(playback, 'set_cache_playback_enabled', lambda *a: False)
    assert ops.FLUIDSIM_OT_cache_playback_start().execute(context) == {'CANCELLED'}
    assert not context.scene.fluidsim.live
    assert not context.scene.fluidsim.cache_playback_enabled


def test_apply_preserves_playing_intent_and_passes_explicit_restart_consent(monkeypatch):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    from fluidSim_warp_blender51.blender import pending_settings
    context.screen.is_animation_playing = True
    context.scene.fluidsim.live = True
    def apply(scene, **kwargs):
        assert not scene.fluidsim.live
        assert not context.screen.is_animation_playing
        assert kwargs['allow_restart'] is True
        calls.append('apply')
        return True
    monkeypatch.setattr(pending_settings, 'apply_changes', apply)
    op = ops.FLUIDSIM_OT_apply_changes()
    op.allow_restart = True
    assert op.execute(context) == {'FINISHED'}
    assert calls == ['pause', 'apply', 'play']
    assert context.scene.fluidsim.live


def test_failed_apply_remains_paused(monkeypatch):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    from fluidSim_warp_blender51.blender import pending_settings
    context.screen.is_animation_playing = True
    context.scene.fluidsim.live = True
    def apply(*args, **kwargs):
        raise RuntimeError('admission rejected')
    monkeypatch.setattr(pending_settings, 'apply_changes', apply)
    assert ops.FLUIDSIM_OT_apply_changes().execute(context) == {'CANCELLED'}
    assert not context.scene.fluidsim.live
    assert not context.screen.is_animation_playing
    assert calls == ['pause']


def test_structural_change_pauses_only_the_affected_scene(monkeypatch):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    from fluidSim_warp_blender51.blender import playback
    context.scene.fluidsim.live = True
    context.screen.is_animation_playing = True
    foreign = SimpleNamespace(scene=object(), screen=SimpleNamespace(is_animation_playing=True))
    context.window_manager = SimpleNamespace(windows=[foreign, context.window])
    playback.pause_preview_for_changes(context.scene)
    assert not context.scene.fluidsim.live
    assert not context.screen.is_animation_playing
    assert foreign.screen.is_animation_playing
    assert calls == ['pause']


def test_invalid_cache_preview_does_not_release_live_solver(monkeypatch):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    from fluidSim_warp_blender51.blender import playback
    from fluidSim_warp_blender51.io import cache
    context.scene.fluidsim.cache_path = 'missing-cache'
    sess = SimpleNamespace(active=True, owns_scene=lambda s: True, release=lambda s: calls.append('release'))
    monkeypatch.setattr(playback, '_bridge', lambda: SimpleNamespace(session=lambda: sess))
    monkeypatch.setattr(cache, 'read_frame', lambda *a: None)
    with pytest.raises(FileNotFoundError):
        playback.prepare_cache_preview(context.scene)
    assert 'release' not in calls
    assert not context.scene.fluidsim.cache_playback_enabled


def test_cached_preview_does_not_require_a_gpu(monkeypatch):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    from fluidSim_warp_blender51.operator_domains import common
    monkeypatch.setattr(common, '_runtime_poll', lambda cls: False)
    assert ops.FLUIDSIM_OT_cache_preview_start.poll(context)


def test_start_exception_after_animation_begins_cancels_timeline(monkeypatch):
    bpy, ops, context, calls = preview_fixture(monkeypatch)
    class BrokenAnimation:
        def poll(self):
            return True
        def __call__(self):
            context.screen.is_animation_playing = True
            raise RuntimeError('animation startup failed')
    bpy.ops.screen.animation_play = BrokenAnimation()
    assert ops.FLUIDSIM_OT_preview_start().execute(context) == {'CANCELLED'}
    assert not context.screen.is_animation_playing
    assert not context.scene.fluidsim.live
