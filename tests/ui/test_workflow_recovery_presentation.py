"""Artist-facing state stays truthful across viewport and cache controls.

Recording layouts cover host presentation; the native companion acceptance
exercises real RNA callbacks, cache publication and recovery.
"""
from types import SimpleNamespace

import pytest
from test_operator_panel_behavior import (
    _draw_stage,
    _install_bpy_stub,
    _Layout,
    _reload_module,
    _stage_context,
)

PKG = 'fluidSim_warp_blender51'


@pytest.mark.parametrize(('authored', 'display', 'surface', 'expected'), [
    ('HIDDEN', 'SURFACE', True, 'Surface Mesh'),
    ('PARTICLES', 'SURFACE', True, 'Surface Mesh'),
    ('SURFACE', 'OFF', True, 'Off'),
    ('SURFACE', 'SURFACE', False, 'Particles'),
    ('HIDDEN', 'PREVIEW', True, 'Liquid Preview'),
])
def test_viewport_summary_follows_display_policy_not_render_outputs(
        monkeypatch, authored, display, surface, expected):
    _install_bpy_stub(monkeypatch)
    ui = _reload_module(f'{PKG}.blender.ui_scene')
    handler = _reload_module(f'{PKG}.blender.liquid_preview.handler')
    monkeypatch.setattr(handler, 'failure', lambda: None)
    output = _reload_module(f'{PKG}.blender.output_publication')
    monkeypatch.setattr(output, 'output_object', lambda *_args, **_kw: SimpleNamespace(
        data=SimpleNamespace(vertices=[1])))
    context, fs, _ = _stage_context(
        surface_enabled=surface,
        scene_overrides={'liquid_view_mode': authored, 'liquid_display_mode': display})
    layout = _draw_stage(ui.FLUIDSIM_PT_scene_outputs, context)
    assert f'Liquid Viewport: {expected}' in [x['text'] for x in layout.labels]
    assert fs.liquid_view_mode == authored  # a display read must not edit render outputs


@pytest.mark.parametrize('playing', [False, True])
def test_cache_exit_is_reachable_beside_both_shared_transports(monkeypatch, playing):
    _install_bpy_stub(monkeypatch)
    panels = _reload_module(f'{PKG}.panels')
    ui = _reload_module(f'{PKG}.blender.ui_scene')
    state = {'id': 'CACHE_PLAYBACK', 'owns_scene': False}
    monkeypatch.setattr(ui, 'activity_ui_state', lambda _scene: state)
    context = SimpleNamespace(scene=object(), screen=SimpleNamespace(is_animation_playing=playing))
    for drawer in (lambda layout: panels.draw_transport(layout, context, state, False),
                   lambda layout: ui.draw_cache_playback(layout, context)):
        layout = _Layout()
        drawer(layout)
        assert 'fluidsim.cache_playback_stop' in {op['id'] for op in layout.operators}
        assert ('fluidsim.preview_pause' if playing else 'screen.animation_play') in {
            op['id'] for op in layout.operators}


@pytest.mark.parametrize('module_name, class_name', [
    ('panels', 'FLUIDSIM_PT_viewport_status'),
    ('blender.ui_header', 'FLUIDSIM_PT_header_popover'),
])
def test_cache_error_is_retained_in_primary_surfaces_and_recovery_clears_it(
        monkeypatch, module_name, class_name):
    _install_bpy_stub(monkeypatch)
    module = _reload_module(f'{PKG}.{module_name}')
    bridge = _reload_module(f'{PKG}.blender.scene_bridge')
    error = ['Cache frame 8 is missing or corrupt in /review/cache']
    monkeypatch.setattr(bridge, 'current_error', lambda _scene: error[0])
    state = {'id': 'CACHE_PLAYBACK', 'label': 'Cache Playback Ready', 'icon': 'PLAY',
             'owns_scene': False, 'detail': error[0]}
    monkeypatch.setattr(module, 'activity_ui_state', lambda _scene: state)
    monkeypatch.setattr(module, 'runtime_ui_state', lambda: (False, 'Playback only'))
    monkeypatch.setattr(module, 'draw_pending_apply', lambda *_args: None)
    if module_name == 'panels':
        monkeypatch.setattr(module, 'playback_only', lambda: False)
    context, *_ = _stage_context()
    context.screen = SimpleNamespace(is_animation_playing=False)
    context.workspace = SimpleNamespace(name='FluidSim')
    cls = getattr(module, class_name)
    layout = _draw_stage(cls, context)
    text = ' '.join(x['text'] for x in layout.labels)
    assert 'Cache Playback Error' in text
    assert 'Cache frame 8' in text
    assert 'fluidsim.open_scene_properties' in {op['id'] for op in layout.operators}
    error[0] = ''
    state['detail'] = 'Cache frame 7: 6 particles'
    recovered = _draw_stage(cls, context)
    text = ' '.join(x['text'] for x in recovered.labels)
    assert 'Cache Playback Ready' in text
    assert 'Cache Playback Error' not in text


@pytest.mark.parametrize(('vertices', 'expected'), [
    ([1], 'Surface Mesh (preview fallback)'),
    ([], 'Particles (preview fallback)'),
])
def test_preview_failure_summary_matches_geometry_fallback(monkeypatch, vertices, expected):
    _install_bpy_stub(monkeypatch)
    display = _reload_module(f'{PKG}.blender.liquid_display')
    handler = _reload_module(f'{PKG}.blender.liquid_preview.handler')
    output = _reload_module(f'{PKG}.blender.output_publication')
    monkeypatch.setattr(handler, 'failure', lambda: 'graphics context unavailable')
    monkeypatch.setattr(output, 'output_object', lambda *_args, **_kw: SimpleNamespace(
        data=SimpleNamespace(vertices=vertices)))
    context, *_ = _stage_context(surface_enabled=True, scene_overrides={'liquid_display_mode': 'PREVIEW'})
    assert display.viewport_summary(context.scene) == expected


def test_long_cache_error_keeps_actionable_start_and_full_detail_link(monkeypatch):
    _install_bpy_stub(monkeypatch)
    common = _reload_module(f'{PKG}.blender.ui_common')
    bridge = _reload_module(f'{PKG}.blender.scene_bridge')
    detail = 'Cache frame 8 is missing or corrupt in /' + '/'.join(['long folder'] * 80)
    monkeypatch.setattr(bridge, 'current_error', lambda _scene: detail)
    layout = _Layout()
    state = {'id': 'CACHE_PLAYBACK', 'label': 'Cache Playback Ready', 'icon': 'PLAY', 'detail': detail}
    assert common.draw_activity_status(layout, object(), state)
    assert len(layout.labels) == 4
    assert 'Cache frame 8' in layout.labels[1]['text']
    assert all(len(row['text']) <= 42 for row in layout.labels[1:])
    assert layout.labels[-1]['text'].endswith('…')
    assert layout.operators[0]['id'] == 'fluidsim.open_scene_properties'


def test_retained_error_is_not_masked_by_a_newer_benign_status(monkeypatch):
    _install_bpy_stub(monkeypatch)
    common = _reload_module(f'{PKG}.blender.ui_common')
    bridge = _reload_module(f'{PKG}.blender.scene_bridge')
    monkeypatch.setattr(bridge, 'current_error', lambda _scene: 'Cache frame 8 is missing or corrupt')
    layout = _Layout()
    common.draw_activity_status(layout, object(), {
        'id': 'CACHE_PLAYBACK', 'label': 'Cache Playback Ready', 'icon': 'PLAY',
        'detail': 'Live simulation disabled'})
    assert 'Cache frame 8' in ' '.join(row['text'] for row in layout.labels)
