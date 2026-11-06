"""Native RNA qualification for PR41 session-trust isolation (no GPU workload)."""
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.repo_addon_loader import enable_repo_addon

enable_repo_addon(str(ROOT), default_set=False, persistent=False)
from fluidSim_warp_blender51.blender import effect_workspace as workspace

scene = bpy.context.scene
item = scene.fluidsim_effects.items.add()
item.instance_id = "pr41-reused"
workspace.set_session_trust(item, True)
assert workspace.session_trust_enabled(scene.fluidsim_effects.items[0])
sibling = scene.fluidsim_effects.items.add()
sibling.instance_id = item.instance_id
assert not workspace.session_trust_enabled(sibling)
other = bpy.data.scenes.new("PR41 isolated scene")
replacement = other.fluidsim_effects.items.add()
replacement.instance_id = item.instance_id
assert not workspace.session_trust_enabled(replacement)
scene.fluidsim_effects.items.remove(0)
for _ in range(128):
    recycled = scene.fluidsim_effects.items.add()
    recycled.instance_id = "pr41-reused"
    assert not workspace.session_trust_enabled(recycled)
    scene.fluidsim_effects.items.remove(len(scene.fluidsim_effects.items) - 1)
item = scene.fluidsim_effects.items[0]
workspace._clear_session_trust_on_load(None)
assert not workspace.session_trust_enabled(item)
assert workspace._clear_session_trust_on_load in bpy.app.handlers.load_post
assert hasattr(workspace._clear_session_trust_on_load, "_bpy_persistent")
import tempfile

with tempfile.TemporaryDirectory(prefix="fluidsim-pr41-") as directory:
    workspace.set_session_trust(item, True)
    path = str(Path(directory) / "isolated.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    assert not workspace._SESSION_TRUST
    assert workspace._clear_session_trust_on_load in bpy.app.handlers.load_post
print("PR41-NATIVE-SAFETY-PASS", flush=True)
