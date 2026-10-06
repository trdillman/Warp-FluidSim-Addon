"""Scene access is the one owner of the Active Domain and Scene identity.

``scene_access`` is a leaf module (no ``bpy``), so these tests cross its
interface directly with plain doubles. The architecture tests pin that the rest
of the package goes through it instead of re-deriving the same facts.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import scene_access


PACKAGE = Path(scene_access.__file__).resolve().parents[1]
ACCESSORS = ("active_domain_object", "active_domain_settings", "scene_identity")


def _scene_with_domain():
    group = SimpleNamespace(voxel_size=0.02, max_particles=9_000_000)
    obj = SimpleNamespace(fluidsim_domain=group)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(schema_version=1, domain_object=obj))
    return scene, obj, group


def test_active_domain_reads_only_the_domain_object():
    scene, obj, group = _scene_with_domain()
    assert scene_access.active_domain_object(scene) is obj
    assert scene_access.active_domain_settings(scene) is group

    # schema_version plays no part in the lookup.
    pre = SimpleNamespace(schema_version=0, domain_object=obj)
    assert scene_access.active_domain_settings(SimpleNamespace(fluidsim=pre)) is group


def test_no_designated_domain_is_none_never_the_scene_group():
    orphan = SimpleNamespace(schema_version=1, domain_object=None, voxel_size=0.5)
    scene = SimpleNamespace(fluidsim=orphan)
    assert scene_access.active_domain_object(scene) is None
    assert scene_access.active_domain_settings(scene) is None


def test_scene_without_fluidsim_is_no_domain_not_an_error():
    assert scene_access.active_domain_object(SimpleNamespace()) is None
    assert scene_access.active_domain_settings(SimpleNamespace()) is None
    assert scene_access.active_domain_settings(None) is None


def test_domain_object_without_a_settings_group_has_no_settings():
    scene = SimpleNamespace(fluidsim=SimpleNamespace(domain_object=SimpleNamespace()))
    assert scene_access.active_domain_object(scene) is not None
    assert scene_access.active_domain_settings(scene) is None


class _Scene:
    def __init__(self, pointer):
        self._pointer = pointer

    def as_pointer(self):
        if isinstance(self._pointer, BaseException):
            raise self._pointer
        return self._pointer


def test_scene_identity_is_the_datablock_pointer_across_wrappers():
    # Blender hands out a fresh wrapper per access; the pointer is what is stable.
    assert scene_access.scene_identity(_Scene(0xDEAD)) == 0xDEAD
    assert scene_access.scene_identity(_Scene(0xDEAD)) == scene_access.scene_identity(_Scene(0xDEAD))


@pytest.mark.parametrize(
    "pointer", [0, TypeError("x"), ValueError("x"), ReferenceError("x")])
def test_scene_identity_falls_back_to_object_id(pointer):
    scene = _Scene(pointer)
    assert scene_access.scene_identity(scene) == id(scene)


def test_scene_identity_of_a_double_without_as_pointer_is_object_id():
    scene = SimpleNamespace()
    assert scene_access.scene_identity(scene) == id(scene)


def test_scene_property_helpers_use_item_access_when_available():
    scene = {}
    scene_access.scene_set(scene, "key", "value")
    assert scene == {"key": "value"}
    assert scene_access.scene_get(scene, "key") == "value"
    assert scene_access.scene_get(scene, "missing", "fallback") == "fallback"
    assert scene_access.scene_get(scene, "missing") == ""


def test_scene_property_helpers_fall_back_to_attributes_for_plain_doubles():
    scene = SimpleNamespace()
    scene_access.scene_set(scene, "key", "value")
    assert scene.key == "value"
    assert scene_access.scene_get(scene, "key") == "value"
    assert scene_access.scene_get(scene, "missing", "fallback") == "fallback"


def _package_modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel.startswith("tests/"):
            continue
        yield rel, ast.parse(path.read_text(encoding="utf-8"))


def test_scene_access_is_a_leaf_with_no_bpy_or_addon_imports():
    imported = []
    for node in ast.walk(ast.parse(Path(scene_access.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(("." * node.level) + (node.module or ""))
    assert [name for name in imported if name != "__future__"] == []


def test_package_imports_the_accessors_from_scene_access_not_scene_bridge():
    offenders = []
    for rel, tree in _package_modules():
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom)
                    and (node.module or "").split(".")[-1] == "scene_bridge"):
                names = {alias.name for alias in node.names}
                if names & set(ACCESSORS):
                    offenders.append(f"{rel}:{node.lineno}")
    assert offenders == [], (
        "import the Active Domain / identity accessors from blender.scene_access; "
        f"scene_bridge only re-exposes them: {offenders}")


def test_only_scene_access_defines_the_accessors_or_reads_domain_object():
    duplicates, raw_reads = [], []
    for rel, tree in _package_modules():
        if rel == "blender/scene_access.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in (
                    *ACCESSORS, "_active_settings", "_scene_identity"):
                duplicates.append(f"{rel}:{node.lineno} {node.name}")
            if (isinstance(node, ast.Attribute) and node.attr == "domain_object"
                    and isinstance(node.ctx, ast.Load)):
                raw_reads.append(f"{rel}:{node.lineno}")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("getattr", "hasattr") and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "domain_object"):
                raw_reads.append(f"{rel}:{node.lineno}")
    assert duplicates == [], f"re-derived accessor: {duplicates}"
    assert raw_reads == [], (
        f"read the designated domain through scene_access.active_domain_object: {raw_reads}")
