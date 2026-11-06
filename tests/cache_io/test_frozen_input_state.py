"""Frozen Blender inputs survive checkpoint2 round trips without runtime pointers."""

from dataclasses import replace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender import scene_inputs
from fluidSim_warp_blender51.io import checkpoint


def _captured(tmp_path):
    inputs = scene_inputs.SceneInputs()
    inputs._native_plan_keys = set()
    point_key = ("point-owner", "POINTS", "point-dependency")
    inputs._native_plan_input(
        point_key, True, lambda: np.array([[1, 2, 3]], np.float32)
    )
    volume = tmp_path / "start.vdb"
    volume.write_bytes(b"opaque VDB dependency")
    inputs._native_plan_input(
        ("sink-owner", "VOLUME", "volume-dependency"),
        True,
        lambda: (str(volume), tuple(np.eye(4).reshape(-1))),
    )
    inputs._native_plan_input(
        ("field-owner", "ANALYTIC", "field-owner"),
        True,
        lambda: {
            "position": (1.0, 0.0, 0.0),
            "rotation": (0.0, 0.0, 0.0, 1.0),
            "half_extents": (1.0, 1.0, 1.0),
            "radius": 1.0,
            "height": 1.0,
        },
    )
    for key in [
        ("FROZEN_SOURCE_MESH", "source", "mesh"),
        ("STATIC_MESH", "collider", "FIRST_SIM_FRAME"),
    ]:
        inputs._evaluated_mesh_cache[key] = {
            "vertices": np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], np.float32),
            "triangles": np.array([[0, 1, 2]], np.int32),
            "force_center": (1.0, 0.0, 0.0),
            "had_degenerate": False,
        }
    return inputs, point_key, volume


def test_frozen_native_mesh_and_static_inputs_survive_checkpoint_roundtrip(tmp_path):
    original, key, _ = _captured(tmp_path)
    state = original.export_frozen_state()
    payload = checkpoint.CheckpointPayload(10, 0.5, "settings", "sources")
    payload = checkpoint.attach_frozen_inputs(payload, state.metadata, state.arrays)
    path = tmp_path / "checkpoint.npz"
    digest = checkpoint.write_checkpoint_atomic(path, payload)
    loaded = checkpoint.read_checkpoint(path, expected_sha256=digest)
    metadata, arrays = checkpoint.frozen_inputs(loaded)
    restored = scene_inputs.SceneInputs()
    restored.restore_frozen_state(scene_inputs.FrozenInputState(metadata, arrays))
    # A fresh process sees the same persistent keys, despite new object pointers.
    continued = restored._native_plan_input(
        key, True, lambda: np.array([[10, 20, 30]], np.float32)
    )
    np.testing.assert_array_equal(continued, [[1, 2, 3]])
    for mesh_key in original._evaluated_mesh_cache:
        np.testing.assert_array_equal(
            restored._evaluated_mesh_cache[mesh_key]["vertices"],
            original._evaluated_mesh_cache[mesh_key]["vertices"],
        )
    assert loaded.format_version == 2
    assert state.metadata == restored.export_frozen_state().metadata
    for name, array in state.arrays.items():
        np.testing.assert_array_equal(
            array, restored.export_frozen_state().arrays[name]
        )
        assert not array.dtype.hasobject


def test_freezing_does_not_restore_every_frame_or_collider_velocity_history(tmp_path):
    original, key, _ = _captured(tmp_path)
    original._collider_prev["runtime-pointer"] = "old history"
    restored = scene_inputs.SceneInputs()
    restored.restore_frozen_state(original.export_frozen_state())
    assert restored._collider_prev == {}
    np.testing.assert_array_equal(
        restored._native_plan_input(
            key, False, lambda: np.array([[10, 0, 0]], np.float32)
        ),
        [[10, 0, 0]],
    )


def test_frozen_state_owns_immutable_snapshots(tmp_path):
    original, key, _ = _captured(tmp_path)
    state = original.export_frozen_state()
    original._native_plan_inputs[key][:] = 99
    restored = scene_inputs.SceneInputs()
    restored.restore_frozen_state(state)
    np.testing.assert_array_equal(restored._native_plan_inputs[key], [[1, 2, 3]])
    assert all(not value.flags.writeable for value in state.arrays.values())


def test_changed_volume_dependency_is_refused_before_restore(tmp_path):
    original, _, volume = _captured(tmp_path)
    state = original.export_frozen_state()
    volume.write_bytes(b"changed contents")
    restored = scene_inputs.SceneInputs()
    with pytest.raises(ValueError, match="[Vv]olume.*changed"):
        restored.restore_frozen_state(state)
    assert restored._native_plan_inputs == {}
    assert restored._evaluated_mesh_cache == {}


def test_malformed_frozen_state_is_refused_without_mutation(tmp_path):
    original, _, _ = _captured(tmp_path)
    state = original.export_frozen_state()
    name = next(iter(state.arrays))
    broken = replace(state, arrays={**state.arrays, name: np.array([object()])})
    restored = scene_inputs.SceneInputs()
    with pytest.raises(ValueError, match="frozen|Frozen"):
        restored.restore_frozen_state(broken)
    assert restored._native_plan_inputs == {}


def test_legacy_checkpoint_has_no_invented_frozen_state():
    payload = checkpoint.CheckpointPayload(10, 0.5, "settings", "sources").validated()
    assert checkpoint.frozen_inputs(payload) is None


def test_missing_frozen_extension_refuses_continuation_when_scene_requires_it(
    monkeypatch,
):
    from types import SimpleNamespace

    from fluidSim_warp_blender51.blender.production_bake import (
        ProductionBakeBackend,
        ResumeRefused,
    )

    monkeypatch.setattr(
        scene_inputs, "frozen_input_keys", lambda _scene: {("p", "POINTS", "p")}
    )
    payload = checkpoint.CheckpointPayload(10, 0.5, "settings", "sources").validated()
    with pytest.raises(ResumeRefused, match="FROZEN_INPUT_STATE_REQUIRED"):
        ProductionBakeBackend._resume_frozen_inputs(SimpleNamespace(), payload)


def test_missing_extension_is_provably_empty_without_frozen_scene_inputs(monkeypatch):
    from types import SimpleNamespace

    from fluidSim_warp_blender51.blender.production_bake import ProductionBakeBackend

    monkeypatch.setattr(scene_inputs, "frozen_input_keys", lambda _scene: set())
    payload = checkpoint.CheckpointPayload(10, 0.5, "settings", "sources").validated()
    state = ProductionBakeBackend._resume_frozen_inputs(SimpleNamespace(), payload)
    assert state.metadata == {"schema_version": 1, "records": []}


def test_changed_roster_refuses_valid_but_foreign_frozen_state(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fluidSim_warp_blender51.blender.production_bake import (
        ProductionBakeBackend,
        ResumeRefused,
    )

    original, _, _ = _captured(tmp_path)
    state = original.export_frozen_state()
    payload = checkpoint.attach_frozen_inputs(
        checkpoint.CheckpointPayload(10, 0.5, "settings", "sources"),
        state.metadata,
        state.arrays,
    )
    monkeypatch.setattr(
        scene_inputs,
        "frozen_input_keys",
        lambda _scene: {("foreign", "POINTS", "foreign")},
    )
    with pytest.raises(ResumeRefused, match="FROZEN_INPUT_STATE_INVALID"):
        ProductionBakeBackend._resume_frozen_inputs(SimpleNamespace(), payload)


def test_restored_shared_mesh_dependencies_keep_one_device_upload(tmp_path):
    original, _key, _volume = _captured(tmp_path)
    source_key = ("FROZEN_SOURCE_MESH", "source", "mesh")
    other_key = ("FROZEN_SOURCE_MESH", "second-source", "mesh")
    original._evaluated_mesh_cache[other_key] = original._evaluated_mesh_cache[
        source_key
    ]
    restored = scene_inputs.SceneInputs()
    restored.restore_frozen_state(original.export_frozen_state())
    assert (
        restored._evaluated_mesh_cache[source_key]
        is restored._evaluated_mesh_cache[other_key]
    )
