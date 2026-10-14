"""Dump every shipped scene's domain settings, field by field, for the 3.0 split.

The settings-split completion must not change what a shipped scene loads as.
This script opens each shipped ``.blend`` headless, reads
``active_domain_settings(scene)`` for every scene in it, and writes the values
to a JSON file. Run it once on the base commit and once on the branch, then
compare the two files: they must be identical.

    blender -b --factory-startup --python-use-system-env ^
        --python tests/acceptance/blender_settings_split_3_0_acceptance.py -- OUT.json

``FLUIDSIM_REPO_ROOT`` selects the checkout to load the add-on and the scenes
from (default: the checkout this file is in). Pointing it at a base worktree
while running this same file gives the base dump.

The script only reads: it never saves, and it opens every file with
``load_ui=False``. Any ``.blend`` under ``demos/``, ``examples/`` or
``fluidSim_warp_blender51/resources/`` is covered; a file with no designated
domain (the render-node library, for one) is recorded with a null domain.
"""

import json
import os
from pathlib import Path
import sys

import bpy


ROOT = Path(os.environ.get("FLUIDSIM_REPO_ROOT")
            or Path(__file__).resolve().parents[2]).resolve()
sys.path.insert(0, str(ROOT))

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

SCENE_DIRECTORIES = ("demos", "examples", "fluidSim_warp_blender51/resources")
# RNA bookkeeping that is not a setting.
SKIP_PROPERTIES = {"rna_type", "name"}


def output_path():
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(arguments) != 1:
        raise SystemExit("usage: ... -- OUTPUT.json")
    return Path(arguments[0])


def shipped_blends():
    found = []
    for directory in SCENE_DIRECTORIES:
        base = ROOT / directory
        if base.is_dir():
            found.extend(base.rglob("*.blend"))
    return sorted(found, key=lambda path: path.relative_to(ROOT).as_posix())


def plain(value):
    """A JSON-safe copy of one RNA value."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, bpy.types.ID):
        return {"id": type(value).__name__, "name": value.name}
    try:
        return [plain(item) for item in value]
    except TypeError:
        return repr(value)


def dump_group(group):
    """Every RNA property of a property group, by name."""
    values = {}
    for prop in group.bl_rna.properties:
        if prop.identifier in SKIP_PROPERTIES:
            continue
        value = getattr(group, prop.identifier)
        if prop.type == "COLLECTION":
            values[prop.identifier] = [dump_group(item) for item in value]
        elif prop.type == "POINTER" and not isinstance(value, bpy.types.ID):
            values[prop.identifier] = None if value is None else dump_group(value)
        else:
            values[prop.identifier] = plain(value)
    return values


def dump_scene(scene):
    from fluidSim_warp_blender51.blender.scene_access import active_domain_settings

    domain_object = scene.fluidsim.domain_object
    settings = active_domain_settings(scene)
    return {
        "domain_object": None if domain_object is None else domain_object.name,
        "domain": None if settings is None else dump_group(settings),
        "scene_group": dump_group(scene.fluidsim),
    }


def dump_file(path):
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    return {scene.name: dump_scene(scene) for scene in bpy.data.scenes}


def main():
    out = output_path()
    enable_repo_addon(str(ROOT), default_set=True, persistent=True)
    files = {}
    for path in shipped_blends():
        key = path.relative_to(ROOT).as_posix()
        try:
            files[key] = dump_file(path)
        except Exception as error:  # a dump that hides a failure proves nothing
            files[key] = {"error": f"{type(error).__name__}: {error}"}
    failed = sorted(key for key, value in files.items() if "error" in value)
    report = {
        "blender": bpy.app.version_string,
        "root": ROOT.name,
        "file_count": len(files),
        "files": files,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    with_domain = sum(
        1 for value in files.values() if "error" not in value
        and any(scene["domain"] is not None for scene in value.values()))
    print(f"SETTINGS-SPLIT-3-0-DUMP files={len(files)} with_domain={with_domain} "
          f"failed={len(failed)} out={out}")
    if failed:
        raise SystemExit("could not dump: " + ", ".join(failed))


main()
