"""Real Blender Append, library-load, Link and migration-lifecycle acceptance.

Run only through the profile-isolated project harness. ``main(output_dir)``
uses the installed add-on when already enabled; it never saves preferences.
The legacy fixture preserves the actual pre-3.0 stored domain schema.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import bpy


def main(output_dir=None):
    if "fluidSim_warp_blender51" not in sys.modules:
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from tools.repo_addon_loader import enable_repo_addon
        enable_repo_addon(str(root), default_set=True, persistent=False)
    import fluidSim_warp_blender51 as addon
    from fluidSim_warp_blender51.blender import file_compat
    from fluidSim_warp_blender51.blender.menus import create_domain_object
    from fluidSim_warp_blender51.properties import _domain_longest_axis

    out = Path(output_dir or tempfile.mkdtemp(prefix="fluidsim-append-"))
    out.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, condition, detail=None):
        checks.append(dict(name=name, passed=bool(condition), detail=detail))
        print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}", flush=True)

    target = bpy.data.scenes.new("Post30Target")
    bpy.context.window.scene = target
    target.fluidsim.live = False
    existing = create_domain_object(target)
    existing.name = "Post30Existing"
    existing.fluidsim_domain.resolution_divisions = 91
    source = bpy.data.scenes.new("Post30Source")
    bpy.context.window.scene = source
    source.fluidsim.live = False
    domain = create_domain_object(source)
    domain.name = "Post30Legacy"
    parent = bpy.data.objects.new("Post30Parent", None)
    source.collection.objects.link(parent)
    parent.scale = (1.0, 2.0, 1.0)
    domain.parent = parent
    domain.scale = (2.0, 1.0, 0.5)
    domain.rotation_euler.z = 0.23
    bpy.context.view_layer.update()
    expected = max(8, round(_domain_longest_axis(domain.fluidsim_domain) / 0.05))
    domain.fluidsim_domain.resolution_divisions = 77
    raw = domain.bl_system_properties_get(do_create=True)["fluidsim_domain"]
    del raw["resolution_divisions"]
    raw["voxel_size"] = 0.05
    fixture = out / "legacy.blend"
    bpy.data.libraries.write(str(fixture), {source})
    bpy.context.window.scene = target
    for obj in tuple(source.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.scenes.remove(source)

    appended = []
    for kind, name in (("Object", "Post30Legacy"), ("Scene", "Post30Source")):
        before = {obj.as_pointer() for obj in bpy.data.objects}
        bpy.ops.wm.append(directory=str(fixture) + "/" + kind + "/", filename=name)
        imported = [obj for obj in bpy.data.objects if obj.as_pointer() not in before
                    and obj.name.startswith("Post30Legacy") and obj.library is None]
        check(kind + " Append imports domain", len(imported) == 1)
        for obj in imported:
            got = obj.fluidsim_domain.resolution_divisions
            check(kind + " Append preserves authored resolution", got == expected,
                  dict(actual=got, expected=expected))
            appended.append(obj)
        check(kind + " Append leaves existing domain alone",
              existing.fluidsim_domain.resolution_divisions == 91)

    with bpy.data.libraries.load(str(fixture), link=False) as (_source, dest):
        dest.objects = ["Post30Legacy"]
    obj = dest.objects[0]
    check("unlinked import is not stamped using an unevaluated matrix",
          "resolution_divisions" not in obj.bl_system_properties_get()["fluidsim_domain"])
    parent = obj.parent
    while parent is not None:
        if parent.name not in target.objects:
            target.collection.objects.link(parent)
        parent = parent.parent
    target.collection.objects.link(obj)
    bpy.context.window.scene = target
    bpy.context.view_layer.update()
    check("first scene evaluation migrates the parent-transformed domain",
          obj.fluidsim_domain.resolution_divisions == expected,
          dict(actual=obj.fluidsim_domain.resolution_divisions, expected=expected))
    appended.append(obj)

    with bpy.data.libraries.load(str(fixture), link=True) as (_source, dest):
        dest.objects = ["Post30Legacy"]
    linked = dest.objects[0]
    raw = linked.bl_system_properties_get()["fluidsim_domain"]
    check("Link leaves library settings unchanged", "resolution_divisions" not in raw)
    # Explicitly author one appended domain, then import again: no whole-file sweep.
    appended[0].fluidsim_domain.resolution_divisions = 37
    bpy.ops.wm.append(directory=str(fixture) + "/Object/", filename="Post30Legacy")
    check("later imports preserve previously edited domains",
          appended[0].fluidsim_domain.resolution_divisions == 37)
    file_compat.register_handlers()
    file_compat.register_handlers()
    callbacks = [h for h in bpy.app.handlers.blend_import_post
                 if h.__module__ == file_compat.__name__]
    check("Append callback registered exactly once", len(callbacks) == 1)
    check("Append callback is persistent", bool(callbacks) and
          all(hasattr(h, "_bpy_persistent") for h in callbacks))
    reopen = out / "reopen.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(reopen), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(reopen))
    surviving = [h for h in bpy.app.handlers.blend_import_post
                 if h.__module__ == file_compat.__name__]
    check("Append handler survives save and reopen", len(surviving) == 1)
    check("linked domain remains unmodified after reopen",
          all("resolution_divisions" not in o.bl_system_properties_get()["fluidsim_domain"]
              for o in bpy.data.objects if o.library and o.name.startswith("Post30Legacy")))
    result = dict(passed=all(c["passed"] for c in checks), checks=checks,
                  blender=bpy.app.version_string, addon_file=addon.__file__)
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("APPEND-MIGRATION-ACCEPTANCE-PASS" if result["passed"] else "APPEND-MIGRATION-ACCEPTANCE-FAIL")
    return result


if __name__ == "__main__":
    if not main()["passed"]:
        raise RuntimeError("Append migration acceptance failed")
