import json
import os
from types import SimpleNamespace

import pytest

from tools.demos.metadata import (
    assign_input_identities, prepare_scene_for_save, demo_object_identity, write_metadata,
)


def test_demo_object_identity_is_checkout_independent_and_unambiguous(monkeypatch, tmp_path):
    first = tmp_path / "first-checkout"
    second = tmp_path / "second-checkout"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    expected = demo_object_identity("Crown Pool Seed", "Crown Pool Seed")
    monkeypatch.chdir(second)
    assert demo_object_identity("Crown Pool Seed", "Crown Pool Seed") == expected
    assert demo_object_identity("Crown Droplet Seed", "Crown Droplet Seed") != expected
    assert demo_object_identity("a:b", "c") != demo_object_identity("a", "b:c")


def test_saved_scene_metadata_is_portable_without_changing_solver_inputs():
    scene = SimpleNamespace(
        render=SimpleNamespace(filepath="C:/old-checkout/demos/crown_splash_preview.png"),
        fluidsim=SimpleNamespace(stat_step_ms=123.45, dt=0.02, stat_particles=5984))
    prepare_scene_for_save(scene, "crown_splash")
    assert scene.render.filepath == "//crown_splash_preview.png"
    assert scene.fluidsim.stat_step_ms == 0
    assert scene.fluidsim.dt == 0.02
    assert scene.fluidsim.stat_particles == 5984


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_identical_metadata_preserves_bytes_and_mtime(tmp_path, newline):
    path = tmp_path / "meta.json"
    content = (json.dumps({"count": 12}, indent=2) + "\n").replace("\n", newline)
    path.write_bytes(content.encode())
    os.utime(path, ns=(1_600_000_000_000_000_000,) * 2)
    before = path.stat().st_mtime_ns
    write_metadata(path, {"count": 12})
    assert path.read_bytes() == content.encode()
    assert path.stat().st_mtime_ns == before


def test_new_and_changed_metadata_are_written(tmp_path):
    path = tmp_path / "meta.json"
    write_metadata(path, {"count": 12})
    assert json.loads(path.read_text()) == {"count": 12}
    write_metadata(path, {"count": 13})
    assert json.loads(path.read_text()) == {"count": 13}
    assert b"\r" not in path.read_bytes()


def test_all_authored_inputs_get_stable_distinct_ids_but_visuals_are_untouched():
    class Object(dict):
        def __init__(self, name, role, seed=False):
            self.name_full = name
            self.data = SimpleNamespace(name_full="Mesh")
            self.fluidsim = SimpleNamespace(role=role, auto_inject_seed=seed)
    objects = [Object(role, role) for role in ("SOURCE", "COLLIDER", "SINK", "FIELD", "FORCE")]
    objects += [Object("seed", "NONE", True), Object("visual", "NONE")]
    scene = SimpleNamespace(objects=objects)
    assign_input_identities(scene)
    expected = [dict(obj) for obj in objects]
    for obj in objects[:-1]:
        obj["fluidsim_persistent_id"] = "previous-checkout-or-scene-id"
    assign_input_identities(scene)
    assert [dict(obj) for obj in objects] == expected
    assert len({obj["fluidsim_persistent_id"] for obj in objects[:-1]}) == 6
    assert objects[-1] == {}
    assert all(obj["fluidsim_persistent_owner"] == obj.name_full for obj in objects[:-1])
