from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fluidsim_testpkg.sim.attributes import (
    ATTRIBUTE_SPECS,
    enabled_attribute_keys,
    motion_from_velocity,
    nearest_sample_indices,
    persistent_particle_channels,
    stable_source_id,
    transfer_nearest,
)


def test_attribute_names_are_unique():
    names = [spec.blender_name for spec in ATTRIBUTE_SPECS.values()]
    assert len(names) == len(set(names))


def test_persistent_particle_channels_are_checkpointed():
    for key in (
        "velocity", "age", "lifetime", "color", "source_id", "uid",
        "viscosity", "density",
    ):
        assert ATTRIBUTE_SPECS[key].checkpointed is True


def test_enabled_inventory_is_ordered_and_optional_channels_are_union():
    settings = SimpleNamespace(
        surface_attribute_velocity=True,
        surface_attribute_color=True,
        particles_attribute_uid=True,
    )
    assert enabled_attribute_keys(settings, "surface") == ("velocity", "color")
    assert enabled_attribute_keys(settings, "whitewater") == ("type",)
    assert persistent_particle_channels(settings) == ("color", "uid")


def test_whitewater_type_is_stable_int_checkpointed_state():
    spec = ATTRIBUTE_SPECS["type"]
    assert spec.blender_name == "fluidsim_whitewater_type"
    assert spec.data_type == "INT"
    assert spec.domain == "POINT"
    assert spec.cached is True
    assert spec.checkpointed is True


def test_source_id_is_deterministic_positive_int32():
    first = stable_source_id("object:Cube:stable-id")
    assert first == stable_source_id("object:Cube:stable-id")
    assert first != stable_source_id("object:Other:stable-id")
    assert 0 < first <= np.iinfo(np.int32).max


def test_motion_is_displacement_per_scene_frame():
    result = motion_from_velocity([[24.0, 0.0, -12.0]], frame_rate=24.0)
    np.testing.assert_allclose(result, [[1.0, 0.0, -0.5]])


def test_nearest_transfer_supports_vectors_and_deterministic_ties():
    targets = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]], dtype=np.float32)
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        transfer_nearest(targets, points, values), values)


def test_spatial_nearest_lookup_resolves_ties_and_far_fallback():
    points = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    targets = np.array([[0.0, 0.0, 0.0], [99.9, 0.0, 0.0]], dtype=np.float32)
    indices = nearest_sample_indices(targets, points, cell_size=0.25)
    np.testing.assert_array_equal(indices, [0, 2])
