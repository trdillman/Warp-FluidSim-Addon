"""Managed-GUI acceptance for the optional GPU particle preview.

Run only in a disposable managed Blender profile. Call start(report, profile_root)
from the managed Blender MCP main thread; the caller owns process shutdown.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def _module_info(module):
    path = Path(module.__file__).resolve()
    return {'path': str(path), 'version': getattr(module, '__version__', None),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class Acceptance:
    def __init__(self, report, profile_root):
        import bpy
        import gpu
        import warp
        import prk
        import blender_vulkan_interop as sdk
        import fluidSim_warp_blender51 as addon
        from fluidSim_warp_blender51.blender import sdk_interop, gpu_particle_owner, gpu_particle_renderer
        from prk.gpudraw import external_texture
        from prk.gpudraw.strategies import points
        from prk.gpudraw.strategies.points import PointsStrategy

        root = Path(profile_root).resolve(strict=True)
        resources = Path(os.environ.get('BLENDER_USER_RESOURCES', '')).resolve(strict=True)
        if root == resources or root not in resources.parents or 'sessions' not in resources.parts:
            raise RuntimeError('expected a disposable managed session under the supplied profile root')
        if bpy.data.filepath or len(bpy.context.scene.objects):
            raise RuntimeError('acceptance requires a fresh empty managed scene')
        if len(bpy.context.window_manager.windows) != 1:
            raise RuntimeError('acceptance requires one owned Blender window')
        self.report = Path(report)
        if root not in self.report.resolve().parents:
            raise RuntimeError('report must be inside the isolated profile root')
        self.bpy, self.gpu, self.warp = bpy, gpu, warp
        self.started = time.monotonic()
        self.stage = 'setup'
        self.original_draw = PointsStrategy.draw
        self.strategy = PointsStrategy
        self.calls = []
        owner = self
        def traced(strategy, packet, state):
            result = owner.original_draw(strategy, packet, state)
            owner.calls.append(int(packet.count))
            return result
        PointsStrategy.draw = traced
        self.data = {'status': 'running', 'blender_version': list(bpy.app.version),
                     'blender_build': bpy.app.build_hash.decode(),
                     'backend': gpu.platform.backend_type_get(),
                     'warp': _module_info(warp), 'sdk': _module_info(sdk),
                     'prk': _module_info(prk), 'addon': _module_info(addon),
                     'source_modules': {m.__name__: _module_info(m) for m in
                                        (sdk_interop, gpu_particle_owner, gpu_particle_renderer,
                                         external_texture, points)},
                     'correctness_position_readback': True,
                     'performance_qualification': False}
        self.save()

    def save(self):
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(json.dumps(self.data, indent=2) + '\n', encoding='utf-8')

    def finish(self, error=None):
        from fluidSim_warp_blender51.blender import gpu_particle_owner as owner
        self.strategy.draw = self.original_draw
        try:
            self.bpy.context.scene.fluidsim.live = False
            if self.bpy.ops.fluidsim.release.poll():
                self.bpy.ops.fluidsim.release()
        except Exception:
            self.data['cleanup_error'] = traceback.format_exc()
        self.data['status'] = 'failed' if error or self.data.get('cleanup_error') else 'passed'
        self.data['error'] = error
        self.data['elapsed_seconds'] = time.monotonic() - self.started
        self.data['retirement_pending_at_finish'] = bool(getattr(owner, '_retire_pending', False))
        self.save()

    def tick(self):
        import bpy
        from fluidSim_warp_blender51.blender import gpu_particle_owner as owner, scene_bridge
        try:
            if time.monotonic() - self.started > 90:
                raise TimeoutError('acceptance exceeded 90 seconds')
            scene = bpy.context.scene
            fs = scene.fluidsim
            if self.stage == 'setup':
                if self.gpu.platform.backend_type_get() != 'VULKAN':
                    raise RuntimeError('requires Vulkan')
                bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 1))
                bpy.context.active_object.display_type = 'WIRE'
                if bpy.ops.fluidsim.setup_scene(mode='QUICK_LIQUID') != {'FINISHED'}:
                    raise RuntimeError('guided setup failed')
                if fs.domain_object is None:
                    raise RuntimeError('guided setup did not designate a domain')
                # The current representation policy chooses ordinary particles
                # only when surface output is disabled. show_particles is a
                # render/export flag, not the viewport representation selector.
                fs.domain_object.fluidsim_domain.surface_enabled = False
                fs.liquid_display_mode = 'SURFACE'
                fs.show_particles = True
                fs.gpu_particle_preview = True
                if bpy.ops.fluidsim.initialize() != {'FINISHED'} or bpy.ops.fluidsim.step(frames=1) != {'FINISHED'}:
                    raise RuntimeError('solver initialization or step failed')
                self.stage = 'await_draw'
            elif self.stage == 'await_draw':
                if owner._error:
                    raise RuntimeError(owner._error)
                if not self.calls or owner._owner is None:
                    return 0.1
                count = int(fs.stat_particles)
                if count <= 0 or self.calls[-1] != count:
                    raise AssertionError('PRK draw count differs from active solver particles')
                if not bpy.data.objects['FluidSim Particles'].hide_get():
                    raise AssertionError('authoritative output was not hidden after draw')
                solver = scene_bridge.session().solver
                self.data['initial_count'] = count
                self.data['prk_strategy'] = self.strategy.__module__ + '.' + self.strategy.__name__
                self.data['before_positions'] = solver.particles.position.numpy()[:3].tolist()
                self.data['solver_frame_before'] = int(solver.frame)
                fs.live = True
                start = scene.frame_current
                for frame in range(start + 1, start + 13):
                    scene.frame_set(frame)
                fs.live = False
                self.data['scene_frame'] = scene.frame_current
                self.data['solver_frame_after'] = int(solver.frame)
                self.data['after_positions'] = solver.particles.position.numpy()[:3].tolist()
                if self.data['before_positions'] == self.data['after_positions'] or self.data['solver_frame_after'] <= self.data['solver_frame_before']:
                    raise AssertionError('timeline did not advance solver positions')
                fs.gpu_particle_preview = False
                if bpy.data.objects['FluidSim Particles'].hide_get():
                    raise AssertionError('paused disable did not restore output immediately')
                self.stage = 'await_retirement'
            elif self.stage == 'await_retirement':
                if owner._error:
                    raise RuntimeError(owner._error)
                if owner._owner is not None or getattr(owner, '_retire_pending', False):
                    return 0.1
                fs.gpu_particle_preview = True
                if bpy.ops.fluidsim.step(frames=1) != {'FINISHED'}:
                    raise RuntimeError('step after re-enable failed')
                self.calls.clear()
                self.stage = 'await_reenabled_draw'
            elif self.stage == 'await_reenabled_draw':
                if owner._error:
                    raise RuntimeError(owner._error)
                if not self.calls or owner._owner is None:
                    return 0.1
                self.data['reenabled_count'] = self.calls[-1]
                if self.calls[-1] != fs.stat_particles:
                    raise AssertionError('re-enabled draw count mismatch')
                self.stage = 'release'
                if bpy.ops.fluidsim.release() != {'FINISHED'}:
                    raise RuntimeError('release failed')
                if bpy.data.objects['FluidSim Particles'].hide_get():
                    raise AssertionError('release did not restore output')
            elif self.stage == 'release':
                if owner._error:
                    raise RuntimeError(owner._error)
                if owner._owner is not None or getattr(owner, '_retire_pending', False):
                    return 0.1
                self.data['release_visible'] = not bpy.data.objects['FluidSim Particles'].hide_get()
                self.finish()
                return None
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            self.save()
            return 0.1
        except Exception:
            self.finish(traceback.format_exc())
            return None


def start(report_path, isolated_profile_root):
    import bpy
    global _acceptance
    _acceptance = Acceptance(report_path, isolated_profile_root)
    bpy.app.timers.register(_acceptance.tick, first_interval=0.2)
    return _acceptance
