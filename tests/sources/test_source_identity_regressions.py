"""B49: a Shift+D duplicate must not share its original's source identity."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from fluidSim_warp_blender51.blender.scene_inputs import persistent_source_key


class FakeObject(dict):
    def __init__(self, name: str, *, data_name: str = "Mesh"):
        super().__init__()
        self.name = self.name_full = name
        self.data = SimpleNamespace(name_full=data_name)
        self.library = None


def _install_objects(monkeypatch, *objects):
    registry = {obj.name_full: obj for obj in objects}
    fake_bpy = SimpleNamespace(data=SimpleNamespace(objects=registry, filepath=""))
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    return registry


def _duplicate(original, name, data_name):
    copy = FakeObject(name, data_name=data_name)
    copy.update(original)  # Shift+D copies every custom property.
    return copy


def test_duplicated_object_gets_its_own_identity(monkeypatch):
    original = FakeObject("Emitter", data_name="Mesh")
    registry = _install_objects(monkeypatch, original)
    original_key = persistent_source_key(original)

    copy = _duplicate(original, "Emitter.001", "Mesh.001")
    registry[copy.name_full] = copy
    copy_key = persistent_source_key(copy)

    assert copy_key != original_key
    assert persistent_source_key(original) == original_key
    assert persistent_source_key(copy) == copy_key
    assert copy["fluidsim_persistent_id"] == copy_key


def test_linked_duplicate_sharing_mesh_data_gets_its_own_identity(monkeypatch):
    original = FakeObject("Emitter", data_name="Mesh")
    registry = _install_objects(monkeypatch, original)
    original_key = persistent_source_key(original)
    copy = _duplicate(original, "Emitter.001", "Mesh")  # Alt+D shares data.
    registry[copy.name_full] = copy
    assert persistent_source_key(copy) != original_key


def test_renamed_object_keeps_its_identity(monkeypatch):
    source = FakeObject("Emitter")
    registry = _install_objects(monkeypatch, source)
    key = persistent_source_key(source)
    del registry["Emitter"]
    source.name = source.name_full = "Renamed Emitter"
    registry[source.name_full] = source
    assert persistent_source_key(source) == key
    # The rename is recorded, so a later duplicate of it is still detected.
    copy = _duplicate(source, "Renamed Emitter.001", "Mesh.001")
    registry[copy.name_full] = copy
    assert persistent_source_key(copy) != key
    assert persistent_source_key(source) == key
