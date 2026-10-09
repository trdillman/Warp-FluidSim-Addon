"""Acceptance: Resolution Divisions replace the authored voxel size (lane G).

Headless Blender, no GPU. Checks, in a real Blender:

  1. ``voxel_size`` is a derived get/set property over ``resolution_divisions``
     and the domain object's world size; writing it sets the nearest divisions.
  2. A stored legacy ``voxel_size`` (no divisions) is migrated on load, the raw
     legacy value survives the get/set property, and a second pass is a no-op.
  3. Scaling the domain object changes the derived voxel and the REINIT
     signature (pending), and the grid never pads an extra leaf.
  4. Demo table: for each shipped demo, stored voxel, longest axis, migrated
     divisions, derived voxel, bitwise equality, and old/new padded dims. The
     demo files are only read, never saved.

Prints ``RESOLUTION_DIVISIONS_OK`` and exits 0 when every check passes.

Run:
  blender --background --factory-startup --python-use-system-env --python tests/acceptance/blender_resolution_divisions_acceptance.py
"""
from __future__ import annotations

from pathlib import Path
import struct
import sys
import tempfile
import traceback

import bpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

FAILURES = []
DEMOS = ("dam_break", "paddle_stir", "crown_splash", "chromatic_vortex")
# Padded dims each shipped demo must have after migration. Three equal the
# pre-3.0 dims. crown_splash's shipped 88,88,56 came from a pre-3.0 float32 ceil
# artifact (0.8000000119 / 0.00999999978 = 80.0000030 -> 81 cells -> pad to 88);
# the CELL_COUNT_TOLERANCE fix removes it, so the grid is 80,80,56.
EXPECTED_NEW_DIMS = {
    "dam_break": (192, 72, 56),
    "paddle_stir": (104, 104, 80),
    "crown_splash": (80, 80, 56),
    "chromatic_vortex": (32, 32, 32),
}


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def raw_domain_group(obj):
    return obj.bl_system_properties_get(do_create=True).get("fluidsim_domain")


def longest_axis(obj):
    from fluidSim_warp_blender51.properties import _domain_longest_axis

    return _domain_longest_axis(obj.fluidsim_domain)


def padded(voxel, scene, tolerance=None):
    """Padded dims; ``tolerance=0`` reproduces the pre-3.0 ceil exactly."""
    from fluidSim_warp_blender51.blender.solver_settings import domain_bounds
    from fluidSim_warp_blender51.sim import quality_contract as qc

    saved = qc.CELL_COUNT_TOLERANCE
    if tolerance is not None:
        qc.CELL_COUNT_TOLERANCE = tolerance
    try:
        return qc.resolve_quality(
            qc.AuthoredQualitySettings(voxel_size=voxel), domain_bounds(scene)
        ).padded_voxel_dimensions
    finally:
        qc.CELL_COUNT_TOLERANCE = saved


def fresh_domain():
    from fluidSim_warp_blender51.blender.menus import create_domain_object

    scene = bpy.context.scene
    scene.fluidsim.domain_object = None
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    domain = create_domain_object(scene)
    domain.scale = (2.0, 1.0, 0.5)  # the default 4 m cube mesh: an 8 x 4 x 2 m box
    bpy.context.view_layer.update()
    return scene, domain


def check_get_set():
    scene, domain = fresh_domain()
    group = domain.fluidsim_domain
    longest = longest_axis(domain)
    check("longest axis of the 8x4x2 box", abs(longest - 8.0) < 1e-5, longest)
    group.resolution_divisions = 100
    check("get = longest / divisions", group.voxel_size == f32(longest / 100),
          group.voxel_size)
    group.voxel_size = longest / 50
    check("set maps to the nearest divisions", group.resolution_divisions == 50,
          group.resolution_divisions)
    group.voxel_size = longest * 10
    check("set floors at 8 divisions", group.resolution_divisions == 8)
    check("padded grid at 64 divisions is 64 along the longest axis",
          (setattr(group, "resolution_divisions", 64) or True)
          and padded(group.voxel_size, scene)[0] == 64,
          padded(group.voxel_size, scene))


def check_migration(tmpdir):
    from fluidSim_warp_blender51.blender import file_compat

    scene, domain = fresh_domain()
    longest = longest_axis(domain)
    name = domain.name
    group = domain.fluidsim_domain
    group.resolution_divisions = 77  # materializes the stored group
    raw = raw_domain_group(domain)
    legacy = 0.05
    del raw["resolution_divisions"]
    raw["voxel_size"] = legacy
    check("legacy raw voxel_size is stored", raw["voxel_size"] == legacy)

    path = str(Path(tmpdir) / "legacy.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)  # load_post runs the migration
    domain = bpy.data.objects[name]
    expected = max(8, round(longest / legacy))
    check("load_post migrates voxel_size -> divisions",
          domain.fluidsim_domain.resolution_divisions == expected,
          (domain.fluidsim_domain.resolution_divisions, expected))
    raw = raw_domain_group(domain)
    check("legacy raw idprop survives the get/set property",
          "voxel_size" in raw and raw["voxel_size"] == legacy, dict(raw))
    check("derived voxel is read, not the stale raw one",
          abs(domain.fluidsim_domain.voxel_size - longest / expected) < 1e-6)

    domain.fluidsim_domain.resolution_divisions = 33
    changed = file_compat.migrate_domain_resolution()
    check("migration is idempotent and leaves authored divisions alone",
          changed == 0 and domain.fluidsim_domain.resolution_divisions == 33,
          (changed, domain.fluidsim_domain.resolution_divisions))

    # A file that already has divisions is never touched by a later load.
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    check("a reopened file keeps its divisions",
          bpy.data.objects[name].fluidsim_domain.resolution_divisions == 33)


def check_domain_scale_is_pending():
    from fluidSim_warp_blender51.blender import pending_settings

    scene, domain = fresh_domain()
    group = domain.fluidsim_domain
    group.resolution_divisions = 64
    before_voxel = group.voxel_size
    before = pending_settings.current_interaction_signatures(scene)
    domain.scale = (4.0, 1.0, 0.5)  # longest axis 8 m -> 16 m
    bpy.context.view_layer.update()
    after = pending_settings.current_interaction_signatures(scene)
    check("scaling the domain changes the derived voxel",
          group.voxel_size == f32(16.0 / 64) and group.voxel_size != before_voxel,
          group.voxel_size)
    changes = pending_settings._signature_changes(before[1], after[1])
    check("scaling the domain raises REINIT pending", "domain_bounds" in changes,
          changes)
    check("scaling the domain is not a RESET change",
          not pending_settings._signature_changes(before[0], after[0]))
    check("scaled domain still pads exactly 64 on its longest axis",
          padded(group.voxel_size, scene)[0] == 64, padded(group.voxel_size, scene))


def demo_table():
    from fluidSim_warp_blender51.blender import file_compat
    from fluidSim_warp_blender51.blender.scene_access import active_domain_object

    print("DEMO_TABLE demo | stored voxel | longest | divisions | derived voxel "
          "| bitwise equal | old dims | new dims", flush=True)
    for name in DEMOS:
        bpy.ops.wm.open_mainfile(filepath=str(REPO_ROOT / "demos" / f"{name}.blend"))
        scene = bpy.context.scene
        obj = active_domain_object(scene)
        raw = raw_domain_group(obj)
        check(f"{name}: demo stores a legacy voxel_size", raw is not None
              and "voxel_size" in raw)
        stored = float(raw["voxel_size"])
        group = obj.fluidsim_domain
        longest = longest_axis(obj)
        derived = float(group.voxel_size)
        old_dims = padded(stored, scene, tolerance=0.0)  # as shipped before 3.0
        new_dims = padded(derived, scene)
        equal = f32(stored) == derived
        print(f"DEMO_TABLE {name} | {stored!r} | {longest!r} | "
              f"{group.resolution_divisions} | {derived!r} | {equal} | "
              f"{old_dims} | {new_dims}", flush=True)
        check(f"{name}: migrated padded dims are the expected ones",
              new_dims == EXPECTED_NEW_DIMS[name], (old_dims, new_dims))
        if name != "crown_splash":
            check(f"{name}: migrated padded dims equal the stored ones",
                  old_dims == new_dims, (old_dims, new_dims))
        check(f"{name}: migration ran again is a no-op",
              file_compat.migrate_domain_resolution() == 0)


def main():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)
    from fluidSim_warp_blender51.blender import file_compat

    check("load_post migration registered",
          file_compat._on_load_post in bpy.app.handlers.load_post)
    check_get_set()
    with tempfile.TemporaryDirectory() as tmpdir:
        check_migration(tmpdir)
    check_domain_scale_is_pending()
    demo_table()


try:
    main()
except Exception:  # noqa: BLE001
    traceback.print_exc()
    FAILURES.append("exception")

if FAILURES:
    print("RESOLUTION_DIVISIONS_FAIL", FAILURES, flush=True)
    sys.exit(1)
print("RESOLUTION_DIVISIONS_OK", flush=True)
sys.exit(0)
