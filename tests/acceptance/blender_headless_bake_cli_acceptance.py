"""Blender/CUDA acceptance: the documented headless bake command, with --scene.

The Bake panel's "Copy Bake Command" always passes ``--scene``. This saves a
file with two scenes, the FluidSim one not active, then runs exactly the
documented command in a child Blender:

  blender --background file.blend --python release/run_headless_bake.py -- \
    --scene "Liquid Scene" --start 1 --end 2 --cache ... --device ... --report-json ...

and checks the exit code, the JSON report and the cache it wrote.

Prints ``HEADLESS-BAKE-CLI-ACCEPTANCE-PASS`` and exits 0 when every check passes.

Run:
  blender --background --factory-startup --python-use-system-env --python tests/acceptance/blender_headless_bake_cli_acceptance.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import traceback

import bpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from _test_device import script_device  # noqa: E402

from tools.repo_addon_loader import ADDON_NAME, enable_repo_addon  # noqa: E402

CHILD_FLAG = "FLUIDSIM_HEADLESS_CLI_CHILD"
SCENE = "Liquid Scene"
FAILURES = []


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def child():
    """Inside the child Blender: load the repo add-on, then run the shipped shim."""
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)
    os.environ["FLUIDSIM_EXTENSION_MODULE"] = ADDON_NAME
    runpy.run_path(str(REPO_ROOT / ADDON_NAME / "release" / "run_headless_bake.py"),
                   run_name="__main__")  # exits the process with the CLI code


def build_file(path):
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)
    from fluidSim_warp_blender51.blender import scene_bridge

    liquid = bpy.context.scene
    liquid.name = SCENE
    bpy.ops.fluidsim.setup_scene()
    domain = scene_bridge.active_domain_settings(liquid)
    domain.voxel_size = 0.12
    domain.whitewater_enabled = False
    domain.surface_enabled = False
    liquid.fluidsim.live = False
    other = bpy.data.scenes.new("Other")
    for window in bpy.data.window_managers[0].windows:  # the saved active scene
        window.scene = other
    active = (bpy.data.window_managers[0].windows[0].scene.name
              if bpy.data.window_managers[0].windows else liquid.name)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    return active


def main(root):
    blend = root / "two_scenes.blend"
    cache = root / "cache"
    report = root / "report.json"
    active_name = build_file(blend)
    check("fixture: the FluidSim scene is not the active one", active_name != SCENE,
          active_name)
    env = dict(os.environ, **{CHILD_FLAG: "1"})
    # A farm bake needs the GPU; never inherit a GPU-hiding variable that an
    # earlier script in the gate's shared batch process left behind.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        bpy.app.binary_path, "--background", "--factory-startup",
        "--python-use-system-env", str(blend),
        "--python", str(Path(__file__).resolve()), "--",
        "--scene", SCENE, "--start", "1", "--end", "2",
        "--cache", str(cache), "--device", script_device(),
        "--resume", "auto", "--checkpoint-interval", "1",
        "--report-json", str(report),
    ]
    completed = subprocess.run(command, env=env, capture_output=True, text=True,
                               timeout=900, check=False)
    tail = (completed.stdout + completed.stderr)[-3000:]
    check("headless bake exits 0", completed.returncode == 0,
          f"exit {completed.returncode}\n{tail}")
    payload = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    check("report says success", payload.get("status") == "success"
          and payload.get("exit_code") == 0, json.dumps(payload)[:2000])
    manifest = cache / "fluidsim_cache.json"
    frames = []
    if manifest.exists():
        frames = sorted(int(key) for key in json.loads(
            manifest.read_text(encoding="utf-8")).get("frames", {}))
    check("cache holds the baked frames", frames == [1, 2], frames)


if __name__ == "__main__":
    if os.environ.get(CHILD_FLAG):
        child()
    root = Path(tempfile.mkdtemp(prefix="fluidsim-headless-cli-"))
    try:
        main(root)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        FAILURES.append("exception")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if FAILURES:
        print("HEADLESS-BAKE-CLI-ACCEPTANCE-FAIL", FAILURES, flush=True)
        sys.exit(1)
    print("HEADLESS-BAKE-CLI-ACCEPTANCE-PASS", flush=True)
    sys.exit(0)
