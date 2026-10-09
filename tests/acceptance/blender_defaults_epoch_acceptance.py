"""Acceptance: 3.0 domain defaults (APIC) and the pre-3.0 pin, in a real Blender.

Headless, no GPU. Checks that a newly designated domain reads APIC and survives
save/load, that a domain authored without the stamp reads FLIP/JACOBI after load
and after File > Append, and that a second load changes nothing.

Prints ``DEFAULTS_EPOCH_OK`` and exits 0 when every check passes.

Run:
  blender --background --factory-startup --python-use-system-env --python tests/acceptance/blender_defaults_epoch_acceptance.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import traceback

import bpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def domain_named(name):
    return bpy.data.objects[name].fluidsim_domain


def main():
    enable_repo_addon(REPO_ROOT, default_set=False, persistent=False)
    from fluidSim_warp_blender51.blender import file_compat
    from fluidSim_warp_blender51.blender.menus import create_domain_object

    key = file_compat.DEFAULTS_EPOCH_KEY
    scene = bpy.context.scene
    new = create_domain_object(scene)
    new.name = "NewDomain"
    check("new domain stamped", new.get(key) == file_compat.DEFAULTS_EPOCH)
    check("new domain APIC", new.fluidsim_domain.sample_type == "APIC")

    legacy = bpy.data.objects.new("LegacyDomain", bpy.data.meshes.new("LegacyMesh"))
    scene.collection.objects.link(legacy)
    legacy.fluidsim_domain.time_scale = 1.0  # authored settings => stored group
    scene.fluidsim.domain_object = legacy  # a 2.x file would carry this pointer
    check("legacy not stamped by designation", legacy.get(key) is None)
    kept = bpy.data.objects.new("PicDomain", bpy.data.meshes.new("PicMesh"))
    scene.collection.objects.link(kept)
    kept.fluidsim_domain.sample_type = "PIC"
    scene.fluidsim.domain_object = new

    tmp = Path(tempfile.mkdtemp()) / "defaults_epoch.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(tmp))
    for round_ in (1, 2):
        bpy.ops.wm.open_mainfile(filepath=str(tmp))
        check(f"load {round_}: new stays APIC", domain_named("NewDomain").sample_type == "APIC")
        legacy_group = domain_named("LegacyDomain")
        check(f"load {round_}: legacy FLIP", legacy_group.sample_type == "FLIP")
        check(f"load {round_}: legacy JACOBI", legacy_group.pressure_preconditioner == "JACOBI")
        check(f"load {round_}: explicit PIC kept", domain_named("PicDomain").sample_type == "PIC")
        check(f"load {round_}: legacy stamped",
              bpy.data.objects["LegacyDomain"].get(key) == file_compat.DEFAULTS_EPOCH)
        bpy.ops.wm.save_as_mainfile(filepath=str(tmp))

    # File > Append of a pre-3.0 domain: build the unstamped file, append into a fresh one.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    enable_repo_addon(REPO_ROOT, default_set=False, persistent=False)
    old = bpy.data.objects.new("AppendedLegacy", bpy.data.meshes.new("AppendedMesh"))
    bpy.context.scene.collection.objects.link(old)
    old.fluidsim_domain.time_scale = 1.0
    src = Path(tempfile.mkdtemp()) / "old.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(src))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    enable_repo_addon(REPO_ROOT, default_set=False, persistent=False)
    with bpy.data.libraries.load(str(src)) as (data_from, data_to):
        data_to.objects = ["AppendedLegacy"]
    appended = bpy.data.objects["AppendedLegacy"]
    check("append: legacy FLIP", appended.fluidsim_domain.sample_type == "FLIP",
          f"(got {appended.fluidsim_domain.sample_type})")

    print("DEFAULTS_EPOCH_OK" if not FAILURES else f"FAILED: {FAILURES}", flush=True)
    sys.exit(1 if FAILURES else 0)


try:
    main()
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    sys.exit(1)
