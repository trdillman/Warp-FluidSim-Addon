"""Blender child for a frozen D7 demo; never edits physics or saved workloads."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import bpy

root, blend, out, mode, expected_warp = sys.argv[sys.argv.index('--') + 1:]
sys.path.insert(0, root)
from tools.repo_addon_loader import enable_repo_addon

addon = enable_repo_addon(root, default_set=False, persistent=False)
from fluidSim_warp_blender51.blender import scene_bridge

legacy = not hasattr(scene_bridge, 'active_domain_settings')
if legacy:
    # Bind the old union RNA as a read-only view of the bridge's stored group.
    bpy.types.Object.fluidsim_domain = bpy.props.PointerProperty(type=type(bpy.context.scene.fluidsim))
bpy.ops.wm.open_mainfile(filepath=blend)
import warp as wp

from fluidSim_warp_blender51.blender import scene_bridge
from fluidSim_warp_blender51.sim.constants import TIMER_PHASE_SLOTS

wp.config.kernel_cache_dir = str(Path(out).parent / ('warp-cache-' + expected_warp))
wp.init()
assert wp.__version__ == expected_warp, wp.__version__
assert wp.is_cuda_available(), 'CUDA required'
scene = bpy.context.scene
scene.fluidsim.live = False
scene.fluidsim.cache_playback_enabled = False
getter = getattr(scene_bridge, 'active_domain_settings', None)
owner = getter(scene) if getter else scene.fluidsim
if getter is None:
    # Reverse the bridge's storage-only migration for the pre-split reference.
    # Values come from the SAME frozen Domain IDProperties, never a fast preset.
    import importlib.util
    spec = importlib.util.spec_from_file_location('budget_scene_compat', Path(__file__).with_name('budget_scene_compat.py'))
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    restore_legacy_domain = compat.restore_legacy_domain
    domain = scene.fluidsim.domain_object.fluidsim_domain
    authored = {key: getattr(domain, key) for key in domain.keys() if key != 'schema_version'}  # noqa: SIM118 - Blender RNA groups are not iterable
    restore_legacy_domain(scene.fluidsim, authored)
# 1.x always exported enabled output attributes. The switch did not exist
# in these frozen scenes; preserve their implicit workload in 2.x.
if hasattr(owner, 'export_attributes'):
    owner.export_attributes = True
owner.profiling_enabled = mode == 'stage'
owner.profiling_sync = mode == 'stage'
scene_bridge.release_session(scene)
session = scene_bridge.initialize_session(scene)
if mode == 'stage' and session.solver is not None:
    # Both versions use the same nesting-aware measurement implementation.
    # The old profiler double-counted nested spans into its solver total.
    import importlib.util
    name = 'fluidSim_warp_blender51.diagnostics._d7_profiler'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'fluidSim_warp_blender51/diagnostics/profiler.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    session.profiler = module.Profiler(device='cuda:0', sync=True)
    session.solver.profiler = session.profiler

assert session.solver is not None and not session.error, session.error
assert str(session.solver.settings.device).startswith('cuda'), 'CPU cannot certify the budget'
# Capture every serializable authored solver setting. Version-only additions
# are retained and reconciled by the comparator, rather than patched to win.
settings = {k: v for k, v in vars(session.solver.settings).items()
            if type(v) in (str, bool, int, float, tuple, list, type(None))
            and not k.startswith('_')}
from enum import Enum

settings.update({k: v.value for k, v in vars(session.solver.settings).items() if isinstance(v, Enum)})
records = []
try:
    for i in range(48):
        scene.frame_set(scene.frame_start + 1 + i)
        previous_time = session.solver.sim_time
        advanced = scene_bridge.step_frame(scene, session, write_cache=False, refresh_pending=False,
                                timeline_frame=scene.frame_start + 1 + i, publish_outputs=True)
        assert not session.error, session.error
        assert advanced is not None and session.solver.frame == i + 1, 'solver did not advance'
        assert session.solver.sim_time > previous_time, 'simulation time did not advance'
        stats = session.solver.stats
        timers = dict(stats.timers)
        stages = {name: float(timers.get(f'compute_{slot}', 0))
                  for name, slot in TIMER_PHASE_SLOTS.items()}
        stages['solver'] = float(timers.get('solver', 0))
        records.append({'frame': i + 1, 'step_ms': float(scene.fluidsim.stat_step_ms),
                        'solver_frame': int(session.solver.frame), 'sim_time': float(session.solver.sim_time),
                        'particles': int(session.solver.particles.refresh_host_count()),
                        'substeps': int(stats.substeps), 'stages': stages})
    result = {'blend_sha256': hashlib.sha256(Path(blend).read_bytes()).hexdigest(),
              'settings': settings, 'device': wp.get_device('cuda:0').name,
              'warp': wp.__version__, 'profiler_sync': mode == 'stage',
              'addon': addon.__file__, 'pid': os.getpid(),
              'source_head': subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD']).decode().strip(), 'particles': session.solver.particles.refresh_host_count(),
              'records': records}
    Path(out).write_text(json.dumps(result, indent=2, allow_nan=False))
finally:
    scene_bridge.release_session(scene)
print('D7-DEMO-TRIAL-PASS', flush=True)
