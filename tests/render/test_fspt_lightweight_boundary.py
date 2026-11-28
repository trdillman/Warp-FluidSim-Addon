"""Viewport scheduling must remain importable before Warp is installed."""
import json
import subprocess
import sys
from pathlib import Path


def test_viewport_import_does_not_reach_device_runtime_when_warp_is_absent():
    script = '''
import importlib.abc
import json
import sys

class RejectWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            raise ImportError("Warp is not installed in this registration process")

sys.meta_path.insert(0, RejectWarp())
from fluidSim_warp_blender51.fspt import viewport
from fluidSim_warp_blender51.fspt.contracts import DeviceSceneUnavailable
assert viewport.DeviceSceneUnavailable is DeviceSceneUnavailable
print(json.dumps(sorted(name for name in sys.modules if (
    name == "warp" or name.startswith("warp.")
    or name == "fluidSim_warp_blender51.fspt.surface_pool"
    or name == "fluidSim_warp_blender51.sim.solver"
))))
'''
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == []


def test_device_exception_reexport_preserves_class_and_subclass_identity():
    from fluidSim_warp_blender51.fspt.contracts import DeviceSceneUnavailable
    from fluidSim_warp_blender51.fspt.device_scene import (
        DeviceSceneUnavailable as DeviceReexport,
    )
    from fluidSim_warp_blender51.fspt.device_scene import GeometryBudgetExceeded

    assert DeviceReexport is DeviceSceneUnavailable
    assert issubclass(GeometryBudgetExceeded, DeviceSceneUnavailable)
