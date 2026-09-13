"""Golden-frame equivalence gate for the Phase 3 domain migration (§9).

Run:  blender --background --factory-startup --python tests/blender_domain_migration_equivalence.py

Spec §9 ship criterion: a migrated scene and an equivalent freshly-authored
``obj.fluidsim_domain`` scene must produce *identical simulation output*.

The gate builds two scenes that share an identically-shaped domain cube and
identical distinctive solver values on ``obj.fluidsim_domain``:

  * Scene A is produced by the real ``load_post`` handler (author a
    ``schema_version == 0`` scene through the pre-split union class, save,
    reopen), then its hidden legacy scene mirror is poisoned with one set of
    wrong values.
  * Scene B authors the values directly on the object and poisons its hidden
    legacy scene mirror with a *different* set of wrong values.

Because the two mirrors disagree, any solver-determining read that still hits
``scene.fluidsim`` makes ``build_settings`` diverge between A and B — the gate
fails loudly. When every read is on the object group, the two resolved
``SolverSettings`` are field-identical, and an N-frame run of each yields an
exact active-voxel match (the benchmark determinism contract).
"""

import os
import sys
import tempfile

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon

FAILURES = []

# Domain AABB shared by both scenes (a well-behaved dam-break volume).
DOMAIN_LO = (-1.0, -1.0, 0.0)
DOMAIN_HI = (1.0, 1.0, 2.0)

# Distinctive solver-determining values authored onto obj.fluidsim_domain in
# BOTH scenes. build_settings must read these (never the poisoned mirror).
SOLVER_DISTINCTIVE = {
    "voxel_size": 0.08,
    "max_particles": 250_000,
    "flip_ratio": 0.97,
    "particles_per_voxel": 9,
    "whitewater_enabled": True,
    "surface_enabled": True,
    "cfl": 1.7,
    "max_speed": 12.5,
    "directional_force": (0.3, -0.2, 0.7),
    "pressure_iterations": 37,
}

# Two DIFFERENT poison sets written to each scene's scene.fluidsim mirror.
# They must differ from SOLVER_DISTINCTIVE and from each other so that a leak
# to the mirror makes scene A and scene B diverge.
POISON_A = {
    "voxel_size": 0.200,
    "max_particles": 11_111,
    "flip_ratio": 0.11,
    "particles_per_voxel": 3,
    "whitewater_enabled": False,
    "surface_enabled": False,
    "cfl": 0.4,
    "max_speed": 3.0,
    "directional_force": (9.0, 9.0, 9.0),
    "pressure_iterations": 5,
}
POISON_B = {
    "voxel_size": 0.155,
    "max_particles": 22_222,
    "flip_ratio": 0.55,
    "particles_per_voxel": 5,
    "whitewater_enabled": True,
    "surface_enabled": True,
    "cfl": 0.9,
    "max_speed": 7.0,
    "directional_force": (1.0, 2.0, 3.0),
    "pressure_iterations": 60,
}

# Kept small: the field-level equivalence is the decisive migration check; the
# live run only confirms the benchmark determinism contract end to end.
SIM_FRAMES = 6
SIM_EMIT_FRAMES = 2


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def make_domain_cube(name):
    verts = [(x, y, z)
             for x in (DOMAIN_LO[0], DOMAIN_HI[0])
             for y in (DOMAIN_LO[1], DOMAIN_HI[1])
             for z in (DOMAIN_LO[2], DOMAIN_HI[2])]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def author_group(group, values):
    for name, value in values.items():
        setattr(group, name, value)


def clear_scene_objects():
    scene = bpy.context.scene
    scene.fluidsim.domain_object = None
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def author_legacy_scene_group(scene, *, schema_version, domain_object, values):
    """Author values exactly as a pre-split Scene.fluidsim group stored them."""
    from fluidSim_warp_blender51 import properties

    with properties.legacy_scene_settings():
        fs = scene.fluidsim
        fs.schema_version = schema_version
        fs.domain_object = domain_object
        author_group(fs, values)


def poison_legacy_scene_mirror(scene, values):
    """Write differing stale domain values without widening normal runtime RNA."""
    from fluidSim_warp_blender51 import properties

    with properties.legacy_scene_settings():
        author_group(scene.fluidsim, values)


def build_migrated_scene_a(tmpdir):
    """Scene A: real load_post migration, then poison the scene mirror."""
    clear_scene_objects()
    scene = bpy.context.scene
    cube = make_domain_cube("Domain A")
    author_legacy_scene_group(
        scene,
        schema_version=0,
        domain_object=cube,
        values=SOLVER_DISTINCTIVE,
    )
    check("A: narrow scene binding restored before save",
          not hasattr(scene.fluidsim, "voxel_size"))

    path = os.path.join(tmpdir, "equivalence_a_v0.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)  # fires the migration handler

    scene = bpy.context.scene
    fs = scene.fluidsim
    check("A: migration stamped schema_version == 1", fs.schema_version == 1,
          fs.schema_version)
    check("A: domain object designated", fs.domain_object is not None,
          getattr(fs.domain_object, "name", None))
    # The hidden legacy mirror must be irrelevant: poison it with A's wrong
    # values while leaving the normal narrow scene binding in place afterward.
    poison_legacy_scene_mirror(scene, POISON_A)
    bpy.context.view_layer.update()


def build_fresh_scene_b():
    """Scene B: author values directly on the object; poison the mirror."""
    clear_scene_objects()
    scene = bpy.context.scene
    cube = make_domain_cube("Domain B")
    author_group(cube.fluidsim_domain, SOLVER_DISTINCTIVE)
    fs = scene.fluidsim
    fs.schema_version = 1
    fs.domain_object = cube
    poison_legacy_scene_mirror(scene, POISON_B)
    bpy.context.view_layer.update()


def resolved_settings():
    from fluidSim_warp_blender51.blender.solver_settings import build_settings
    return build_settings(bpy.context.scene)


def compare_settings(sa, sb):
    fields = type(sa).__dataclass_fields__
    mismatches = []
    for name in fields:
        va, vb = getattr(sa, name), getattr(sb, name)
        if va != vb:
            mismatches.append((name, va, vb))
    return mismatches


def run_equivalence():
    from fluidSim_warp_blender51.diagnostics.benchmark import (
        EXACT_FIELDS,
        compare_benchmark,
        default_benchmark_sources,
        run_benchmark,
    )

    tmpdir = tempfile.mkdtemp(prefix="fluidsim_equivalence_")

    build_migrated_scene_a(tmpdir)
    sa = resolved_settings()
    # Sanity: the resolved domain came from the shared cube, not the mirror.
    check("A: settings read distinctive voxel_size (not poison)",
          abs(sa.voxel_size - SOLVER_DISTINCTIVE["voxel_size"]) < 1e-6,
          sa.voxel_size)
    check("A: domain bounds match the cube",
          tuple(round(v, 6) for v in sa.domain_min) == DOMAIN_LO
          and tuple(round(v, 6) for v in sa.domain_max) == DOMAIN_HI,
          (tuple(sa.domain_min), tuple(sa.domain_max)))

    build_fresh_scene_b()
    sb = resolved_settings()
    check("B: settings read distinctive voxel_size (not poison)",
          abs(sb.voxel_size - SOLVER_DISTINCTIVE["voxel_size"]) < 1e-6,
          sb.voxel_size)

    mismatches = compare_settings(sa, sb)
    check("resolved SolverSettings are field-identical (migrated == fresh)",
          not mismatches, mismatches[:8])

    # Literal §9 criterion: simulate N frames from each and require an exact
    # active-voxel match (the benchmark determinism contract).
    res_a = run_benchmark(frames=SIM_FRAMES, emit_frames=SIM_EMIT_FRAMES,
                          settings=sa, sources=default_benchmark_sources()).as_dict()
    res_b = run_benchmark(frames=SIM_FRAMES, emit_frames=SIM_EMIT_FRAMES,
                          settings=sb, sources=default_benchmark_sources()).as_dict()
    report = compare_benchmark(res_a, res_b)
    check("N-frame active-voxel output is identical",
          all(report["exact"][name]["match"] for name in EXACT_FIELDS),
          {name: (res_a[name], res_b[name]) for name in EXACT_FIELDS})
    check("benchmark contract fields are valid on both runs",
          "error" not in report, report.get("error"))


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
    run_equivalence()


main()
print()
if FAILURES:
    print(f"DOMAIN-MIGRATION-EQUIVALENCE-FAIL ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("DOMAIN-MIGRATION-EQUIVALENCE-PASS")
sys.exit(0)
