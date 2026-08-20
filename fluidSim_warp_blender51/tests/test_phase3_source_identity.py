from __future__ import annotations

from types import SimpleNamespace

from fluidsim_testpkg.blender.scene_inputs import persistent_source_key


class FakeObject(dict):
    def __init__(self, name: str, *, data_name: str = "Mesh", library: str = ""):
        super().__init__()
        self.name = name
        self.name_full = name
        self.data = SimpleNamespace(name_full=data_name)
        self.library = SimpleNamespace(filepath=library) if library else None


def test_local_source_identity_is_deterministic_and_persisted():
    first = FakeObject("Emitter")
    second = FakeObject("Emitter")
    first_key = persistent_source_key(first)
    assert first_key == persistent_source_key(first)
    assert first_key == persistent_source_key(second)
    assert first["fluidsim_persistent_id"] == first_key
    assert first_key.startswith("object:")


def test_persisted_source_identity_survives_rename():
    source = FakeObject("Emitter")
    key = persistent_source_key(source)
    source.name = source.name_full = "Renamed Emitter"
    assert persistent_source_key(source) == key


def test_linked_source_identity_includes_library_path():
    first = FakeObject("Emitter", library="//library_a.blend")
    second = FakeObject("Emitter", library="//library_b.blend")
    assert persistent_source_key(first) != persistent_source_key(second)
