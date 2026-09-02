"""Headless .blend round-trip + idempotency gate for the Phase 3 domain migration.

Run:  blender --background --factory-startup --python tests/blender_domain_migration_acceptance.py

Scripts two pre-migration (``schema_version == 0``) scenes from a v0 template —
one with ``fluidsim.domain_object`` already set (the resolve branch of
``_resolve_or_create_domain``) and one unset (the create/designate branch that
spec §9 requires) — each carrying distinctive scalar/enum/vector values plus a
material and an image pointer (so the pointer path from Issue 3 is exercised).
Each fixture is saved, reopened so the ``load_post`` migration handler runs, and
asserted: every migrated field (scalars AND the two datablock pointers) landed
on ``obj.fluidsim_domain`` with identical values and ``schema_version == 1``.
The migrated state is then saved and reopened to prove the guard makes a second
load a no-op (no duplicate domain, values and schema unchanged).
"""

import os
import sys
import tempfile

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import ADDON_NAME, enable_repo_addon

FAILURES = []

# Distinctive, non-default values spanning float / int / bool / vector / enum,
# so a migration that silently keeps target defaults is caught.
DISTINCTIVE = {
    "voxel_size": 0.037,
    "max_particles": 1_234_567,
    "whitewater_enabled": True,
    "flip_ratio": 0.83,
    "particles_per_voxel": 12,
    "directional_force": (0.1, 0.2, 0.3),
    "height_back_mode": "FLAT",
    # ``live`` carries an RNA update callback that spawns a Solver Session.
    # It MUST migrate as a plain value: the copy fires the callback in real
    # Blender, so a non-suppressed copy would auto-start a solver mid-load.
    "live": True,
}


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def norm(value):
    """Compare values across a save/load boundary (datablocks by name)."""
    if isinstance(value, bpy.types.ID):
        return ("ID", type(value).__name__, value.name)
    if value is None:
        return None
    try:
        return tuple(round(float(component), 6) for component in value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return round(value, 6)
    return value


def author_v0_scene(with_domain):
    """Rebuild the current scene as a pre-migration (schema 0) template."""
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    fs = scene.fluidsim
    fs.domain_object = None
    fs.schema_version = 0
    for name, value in DISTINCTIVE.items():
        setattr(fs, name, value)
    material = bpy.data.materials.new("Migration Surface Mat")
    image = bpy.data.images.new("Migration Height Profile", width=2, height=2)
    fs.surface_material = material
    fs.height_profile_image = image
    if with_domain:
        mesh = bpy.data.meshes.new("Preset Domain")
        domain = bpy.data.objects.new("Preset Domain", mesh)
        scene.collection.objects.link(domain)
        fs.domain_object = domain
    return scene, material.name, image.name


def migrated_domains(scene):
    """Objects this scene has stamped as a migrated domain."""
    return [obj for obj in scene.objects
            if obj.fluidsim_domain.schema_version == 1]


def run_fixture(label, with_domain, tmpdir):
    from fluidSim_warp_blender51.blender.migration import migrated_field_names

    v0_path = os.path.join(tmpdir, f"{label}_v0.blend")
    v1_path = os.path.join(tmpdir, f"{label}_v1.blend")

    _scene, material_name, image_name = author_v0_scene(with_domain)
    check(f"{label}: authored schema_version == 0",
          bpy.context.scene.fluidsim.schema_version == 0)
    bpy.ops.wm.save_as_mainfile(filepath=v0_path)

    # -- reopen: the persistent load_post migration handler runs -------------
    bpy.ops.wm.open_mainfile(filepath=v0_path)
    scene = bpy.context.scene
    fs = scene.fluidsim
    check(f"{label}: source guard stamped 1", fs.schema_version == 1,
          fs.schema_version)

    # Regression (load semantics): migration is state restoration, not a user
    # action. Copying ``live=True`` fires its RNA update callback in real
    # Blender; without suppression that callback auto-starts a Solver Session
    # mid-load, before object transforms are evaluated, binding the scene to a
    # bogus unit-cube domain. Opening a file must never leave a live solver.
    from fluidSim_warp_blender51.blender import scene_bridge
    sess = scene_bridge.session()
    check(f"{label}: migration left no active session (side-effect-free copy)",
          not sess.active and not sess.error,
          f"active={sess.active} error={sess.error!r}")

    domains = migrated_domains(scene)
    check(f"{label}: exactly one migrated domain object", len(domains) == 1,
          [obj.name for obj in domains])
    if with_domain:
        check(f"{label}: designated the pre-set domain (resolve branch)",
              fs.domain_object is not None
              and fs.domain_object.name == "Preset Domain",
              getattr(fs.domain_object, "name", None))
    else:
        check(f"{label}: created + designated a domain (create branch)",
              fs.domain_object is not None
              and fs.domain_object.name in {obj.name for obj in scene.objects},
              getattr(fs.domain_object, "name", None))
        # Regression: an auto-created domain must span real bounds, or the
        # solver's "maximum must be greater than minimum on every axis" guard
        # rejects playback after a reopen. It must reproduce the exact fallback
        # box an unset domain used to yield, so output stays identical.
        from fluidSim_warp_blender51.blender.solver_settings import (
            DEFAULT_DOMAIN_MAX, DEFAULT_DOMAIN_MIN, domain_bounds)
        lo, hi = domain_bounds(scene)
        check(f"{label}: created domain bounds non-degenerate on every axis",
              all(hi[axis] > lo[axis] for axis in range(3)), (lo, hi))
        check(f"{label}: created domain bounds match the canonical fallback",
              norm(lo) == norm(DEFAULT_DOMAIN_MIN)
              and norm(hi) == norm(DEFAULT_DOMAIN_MAX), (lo, hi))

    if fs.domain_object is None:
        check(f"{label}: domain object present for field checks", False)
        return
    target = fs.domain_object.fluidsim_domain
    check(f"{label}: target schema_version == 1", target.schema_version == 1,
          target.schema_version)

    for name, value in DISTINCTIVE.items():
        check(f"{label}: scalar field {name!r} migrated",
              norm(getattr(target, name)) == norm(value),
              f"{norm(getattr(target, name))} != {norm(value)}")
    check(f"{label}: material pointer migrated (Issue 3)",
          target.surface_material is not None
          and target.surface_material.name == material_name,
          getattr(target.surface_material, "name", None))
    check(f"{label}: image pointer migrated (Issue 3)",
          target.height_profile_image is not None
          and target.height_profile_image.name == image_name,
          getattr(target.height_profile_image, "name", None))

    mismatches = [name for name in migrated_field_names(type(fs))
                  if norm(getattr(fs, name)) != norm(getattr(target, name))]
    check(f"{label}: every migrated field identical to source mirror",
          not mismatches, mismatches)

    # -- idempotency: save the migrated state, reopen, expect a no-op --------
    bpy.ops.wm.save_as_mainfile(filepath=v1_path)
    domain_name = fs.domain_object.name
    before_voxel = round(target.voxel_size, 6)
    bpy.ops.wm.open_mainfile(filepath=v1_path)
    scene = bpy.context.scene
    fs = scene.fluidsim
    check(f"{label}: reopen creates no duplicate domain",
          len(migrated_domains(scene)) == 1,
          [obj.name for obj in migrated_domains(scene)])
    check(f"{label}: reopen keeps schema_version == 1", fs.schema_version == 1,
          fs.schema_version)
    check(f"{label}: reopen keeps the designated domain",
          fs.domain_object is not None and fs.domain_object.name == domain_name,
          getattr(fs.domain_object, "name", None))
    check(f"{label}: reopen preserves a migrated value",
          fs.domain_object is not None
          and round(fs.domain_object.fluidsim_domain.voxel_size, 6) == before_voxel,
          getattr(getattr(fs.domain_object, "fluidsim_domain", None), "voxel_size", None))


def main():
    enabled = False
    try:
        enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
        enabled = True
    except Exception as exc:  # noqa: BLE001
        print(f"enable failed: {exc!r}")
    check("addon enable", enabled)
    if not enabled:
        return
    check("persistent addon preferences entry registered",
          ADDON_NAME in bpy.context.preferences.addons)
    check("migration load_post handler registered",
          any(getattr(handler, "__name__", "") == "_on_load_post"
              for handler in bpy.app.handlers.load_post))

    tmpdir = tempfile.mkdtemp(prefix="fluidsim_migration_")
    run_fixture("domain_set", True, tmpdir)
    run_fixture("domain_unset", False, tmpdir)


main()
print()
if FAILURES:
    print(f"DOMAIN-MIGRATION-ACCEPTANCE-FAIL ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("DOMAIN-MIGRATION-ACCEPTANCE-PASS")
sys.exit(0)
