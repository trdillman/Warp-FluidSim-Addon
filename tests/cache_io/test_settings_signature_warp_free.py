"""``cache_identity.settings_signature`` is stable and needs no Warp.

Existing caches compare against this string, so the pinned value must never
change when the settings model is moved or refactored. The scene is a fixed
stand-in (RNA defaults on a default-box domain, fixed device and frame rate).
Each run happens in a subprocess so the Warp mask cannot leak.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "fluidSim_warp_blender51"

# Computed on the unchanged base (cfff382) before the settings model moved;
# re-pinned when lane J added narrow_band_enabled/narrow_band_voxels (every
# SolverSettings field is part of the signature, so existing caches re-bake),
# and again when owner decision D1 (2026-10-12) took the compute device out
# of cache identity.
PINNED_SIGNATURE = "6d5bb730f1b3c7c26485c54a49e3e85582d7d0d72903a68d3877e63df05c3012"
# The device-hashing value caches written before D1 carry. They must still
# match the same scene, or every 3.0/3.1 cache would be refused.
PRE_D1_SIGNATURE = "f812a312f5c45196d3e4709a368d9e6df192b633115c652e13a205d8c9d1f606"

_SCRIPT = r'''
import importlib
import importlib.abc
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

package, tests_dir, mask = sys.argv[1:4]


class BlockWarp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            raise ImportError("warp is masked")
        return None


if mask == "1":
    sys.meta_path.insert(0, BlockWarp())

bpy = MagicMock()
bpy.app.handlers.persistent = lambda function: function
sys.modules["bpy"] = bpy
mathutils = types.ModuleType("mathutils")
mathutils.Vector = tuple
sys.modules["mathutils"] = mathutils
preferences = types.ModuleType(package + ".preferences")
preferences.configured_device = lambda: "cuda:0"
sys.modules[package + ".preferences"] = preferences

sys.path.insert(0, tests_dir)
from settings_doubles import full_settings, split_settings

cache_identity = importlib.import_module(package + ".blender.cache_identity")
scene = SimpleNamespace(
    fluidsim=split_settings(full_settings(
        domain_object=None, quality="PREVIEW", use_scene_gravity=False,
        gravity=[0.0, 0.0, -9.81])),
    render=SimpleNamespace(fps=24, fps_base=1.0),
    use_gravity=True, gravity=[0.0, 0.0, -9.81],
)
print(json.dumps({
    "signature": cache_identity.settings_signature(scene),
    "legacy": list(cache_identity.legacy_settings_signatures(scene)),
    "warp_loaded": sorted(n for n in sys.modules
                          if n == "warp" or n.startswith("warp.")),
}))
'''


def _run(mask: bool) -> dict:
    env = dict(
        os.environ, PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT, PACKAGE_NAME,
         str(REPO / "tests" / "support"), "1" if mask else "0"],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_settings_signature_value_is_pinned():
    result = _run(mask=False)
    assert result["signature"] == PINNED_SIGNATURE
    assert PRE_D1_SIGNATURE in result["legacy"]


def test_settings_signature_needs_no_warp_and_is_unchanged_by_the_mask():
    masked = _run(mask=True)
    assert masked["warp_loaded"] == []
    assert masked["signature"] == PINNED_SIGNATURE
