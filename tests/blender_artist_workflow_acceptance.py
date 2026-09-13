"""Timer-driven GUI proof; run in an isolated source-loaded demo session.

Call start(report_path) inside Blender MCP. The callback must advance through
real timeline playback, never manual frame_set loops. The source demo is not saved.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import bpy

STATE = {}


def start(report_path, with_effect=True):
    from fluidSim_warp_blender51.blender import effects, playback, scene_bridge
    scene = bpy.context.scene
    window = next(w for w in bpy.context.window_manager.windows if w.scene == scene)
    area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    STATE.clear()
    STATE.update(scene=scene, window=window, area=area, region=region, with_effect=with_effect,
                 path=Path(report_path), phase='start', deadline=time.monotonic() + 180,
                 report={'status': 'RUNNING', 'pid': os.getpid(),
                         'blend': bpy.data.filepath, 'blender': bpy.app.version_string,
                         'source': str(Path(effects.__file__).resolve()), 'checks': []})
    STATE['path'].parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    STATE['report']['head'] = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    STATE['report']['scene'] = scene.name
    STATE['report']['domain'] = scene.fluidsim.domain_object.name if scene.fluidsim.domain_object else None
    STATE['report']['frames'] = []
    with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
        if window.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        scene.fluidsim.live = False
        if scene_bridge.session().owns_scene(scene):
            scene_bridge.session().release(scene)
        with playback.suppress_frame_handlers():
            scene.frame_set(scene.frame_start)
        domain = scene_bridge.active_domain_settings(scene)
        assert scene.fluidsim.domain_object is not None, 'Demo has no domain'
        domain.voxel_size = 0.14
        domain.max_particles = 50000
        domain.max_whitewater = 1
        domain.whitewater_enabled = False
        domain.surface_enabled = False
        scene.fluidsim.show_particles = True
        effects.clear_effects(scene)
        if with_effect:
            result = bpy.ops.fluidsim.effect_add(effect_id='fluidsim.radial_pulse')
            assert result == {'FINISHED'}
    bpy.app.timers.register(_tick, first_interval=0.1)
    return {'scheduled': True, 'report': str(STATE['path'])}


def _check(name, condition):
    if not condition:
        raise AssertionError(name)
    STATE['report']['checks'].append(name)


def _output_digest(scene):
    from fluidSim_warp_blender51.blender.output_publication import output_object
    obj = output_object(scene, 'FluidSim Particles')
    assert obj is not None, 'Owned particles output missing'
    attr = obj.data.attributes.get('position')
    assert attr is not None and len(attr.data), 'Owned output is empty'
    values = [tuple(round(float(v), 6) for v in item.vector) for item in attr.data[:128]]
    return hashlib.sha256(repr(values).encode()).hexdigest()


def _tick():
    from fluidSim_warp_blender51.blender import effects, scene_bridge
    scene, window = STATE['scene'], STATE['window']
    try:
        assert time.monotonic() < STATE['deadline'], 'Playback acceptance timed out'
        with bpy.context.temp_override(window=window, screen=window.screen,
                                       area=STATE['area'], region=STATE['region']):
            sess = scene_bridge.session()
            phase = STATE['phase']
            if phase == 'start':
                _check('draft Preview succeeds', bpy.ops.fluidsim.preview_start() == {'FINISHED'})
                _check('Blender playback started', window.screen.is_animation_playing)
                _check('effect evidence applied automatically', bool(scene.fluidsim_effects.applied_signature))
                STATE.update(phase='advance', frame=scene.frame_current, solver=sess.solver)
            elif phase == 'advance' and scene.frame_current >= STATE['frame'] + 3:
                STATE['report']['frames'].append(scene.frame_current)
                STATE['report']['device'] = str(sess.solver.device)
                _check('real nonempty solver', sess.solver.particles.refresh_host_count() > 0)
                if STATE['with_effect']:
                    _check('production Radial Pulse execution', any(
                        s.effect_id == 'fluidsim.radial_pulse' and s.calls > 0
                        for s in sess.solver.effects.stats.values()))
                else:
                    _check('non-effect sibling has no effect dispatch', not sess.solver.effects.stats)
                STATE.update(output=_output_digest(scene), frame=scene.frame_current, phase='pause')
                _check('Pause succeeds', bpy.ops.fluidsim.preview_pause() == {'FINISHED'})
            elif phase == 'pause':
                _check('Pause preserves frame and solver', scene.frame_current == STATE['frame'] and sess.solver is STATE['solver'])
                _check('Pause stops timeline', not window.screen.is_animation_playing)
                entries = [e for group in sess.solver.effects.by_stage.values() for e in group]
                if STATE['with_effect']:
                    STATE['kernel'] = entries[0].compiled_kernel
                    item = scene.fluidsim_effects.items[0]
                    param = next(p for p in item.parameters if p.kind == 'FLOAT')
                    param.float_value = param.float_value + 0.1
                    entries = [e for group in sess.solver.effects.by_stage.values() for e in group]
                    _check('parameter edit preserves solver/kernel', sess.solver is STATE['solver'] and entries[0].compiled_kernel is STATE['kernel'])
                _check('Resume succeeds', bpy.ops.fluidsim.preview_start() == {'FINISHED'})
                STATE['phase'] = 'resume'
            elif phase == 'resume' and scene.frame_current >= STATE['frame'] + 3:
                STATE['report']['frames'].append(scene.frame_current)
                _check('owned output changes', _output_digest(scene) != STATE['output'])
                _check('Resume keeps solver', sess.solver is STATE['solver'])
                STATE['report']['particles'] = sess.solver.particles.refresh_host_count()
                STATE['report']['effect_calls'] = {s.effect_id: s.calls for s in sess.solver.effects.stats.values()}
                _check('structural edit added', bpy.ops.fluidsim.effect_add(effect_id='fluidsim.planar_split') == {'FINISHED'})
                _check('structural edit pauses simulation', not scene.fluidsim.live and not window.screen.is_animation_playing)
                frame = scene.frame_current
                try:
                    result = bpy.ops.fluidsim.apply_changes()
                except RuntimeError:
                    result = {'CANCELLED'}
                _check('no implicit progress loss', result == {'CANCELLED'} and sess.solver is STATE['solver'] and scene.frame_current == frame)
                _check('explicit apply succeeds', bpy.ops.fluidsim.apply_changes(allow_restart=True) == {'FINISHED'})
                _check('Apply restarts at start frame paused', scene.frame_current == scene.frame_start and not window.screen.is_animation_playing)
                _check('Apply installs current stack', not effects.production_blocking_reason(scene))
                _check('explicit Restart succeeds', bpy.ops.fluidsim.restart_simulation() == {'FINISHED'})
                _check('Restart remains at start', scene.frame_current == scene.frame_start)
                STATE['report']['status'] = 'PASS'
                STATE['path'].write_text(json.dumps(STATE['report'], indent=2), encoding='utf-8')
                print('ARTIST-WORKFLOW-ACCEPTANCE-PASS')
                return None
        return 0.05
    except Exception:
        scene.fluidsim.live = False
        with bpy.context.temp_override(window=window, screen=window.screen):
            if window.screen.is_animation_playing:
                bpy.ops.screen.animation_cancel(restore_frame=False)
        STATE['report'].update(status='FAIL', error=traceback.format_exc())
        STATE['path'].write_text(json.dumps(STATE['report'], indent=2), encoding='utf-8')
        print(STATE['report']['error'])
        return None
