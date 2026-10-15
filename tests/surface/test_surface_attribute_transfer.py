from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim.attributes import nearest_sample_indices


def _transfer_type():
    try:
        from fluidSim_warp_blender51.sim.surface_attribute_transfer import (
            SurfaceAttributeTransfer,
        )
    except ModuleNotFoundError:
        pytest.fail("SurfaceAttributeTransfer API is missing")
    return SurfaceAttributeTransfer


def _particles(device, positions, **channels):
    arrays = {
        name: wp.array(values, dtype=dtype, device=device)
        for name, (values, dtype) in channels.items()
    }
    return SimpleNamespace(
        position=wp.array(
            np.asarray(positions, dtype=np.float32),
            dtype=wp.vec3,
            device=device,
        ),
        attr=lambda name: arrays[name],
    )


def _devices():
    devices = ["cpu"]
    if wp.is_cuda_available():
        devices.append("cuda:0")
    return devices


def test_surface_transfer_preserves_exact_tie_and_global_fallback_on_cpu():
    """A lowest-index tie or absent local bucket must not change the contract."""
    transfer = _transfer_type()(device="cpu")
    particles = _particles(
        "cpu",
        [[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [10.0, 0.0, 0.0]],
    )
    targets = wp.array(
        np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device="cpu",
    )

    transfer.prepare(particles, live_count=3, cell_size=1.0)
    result = transfer.nearest_indices(targets, target_count=2).numpy()

    assert np.array_equal(result[:2], np.array([0, 2], dtype=np.int32))
    assert transfer.operation_counts() == {
        "sample_gathers": 1,
        "hash_grid_builds": 1,
        "local_lookups": 1,
        "fallback_launches": 1,
        "gathers": 0,
        "readbacks": 0,
    }


@pytest.mark.parametrize("device", _devices())
def test_surface_transfer_gathers_all_supported_dtypes_from_one_nearest_map(
    device,
):
    """Adding channels must gather values, not repeat the spatial lookup."""
    transfer = _transfer_type()(device=device)
    particles = _particles(
        device,
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        scalar=([1.25, 2.5, 5.0], wp.float32),
        integer=([7, 8, 9], wp.int32),
        vector3=([[1, 2, 3], [4, 5, 6], [7, 8, 9]], wp.vec3),
        vector4=(
            [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
            wp.vec4,
        ),
    )
    targets = wp.array(
        np.array([[0.25, 0.0, 0.0], [3.75, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    transfer.prepare(particles, live_count=3, cell_size=1.0)
    transfer.nearest_indices(targets, target_count=2)

    expected = {
        "scalar": np.array([1.25, 5.0], dtype=np.float32),
        "integer": np.array([7, 9], dtype=np.int32),
        "vector3": np.array([[1, 2, 3], [7, 8, 9]], dtype=np.float32),
        "vector4": np.array(
            [[1, 0, 0, 1], [0, 0, 1, 1]], dtype=np.float32
        ),
    }
    for key, expected_value in expected.items():
        transfer.gather(key, particles.attr(key), target_count=2)
        actual = transfer.readback(key, target_count=2)
        assert actual.dtype == expected_value.dtype
        assert actual.flags.c_contiguous
        assert np.array_equal(actual, expected_value)

    assert transfer.operation_counts() == {
        "sample_gathers": 1,
        "hash_grid_builds": 1,
        "local_lookups": 1,
        "fallback_launches": 1,
        "gathers": 4,
        "readbacks": 4,
    }


@pytest.mark.parametrize(
    ("positions", "targets", "expected"),
    [
        ([], [[0.0, 0.0, 0.0]], [-1]),
        ([[0.0, 0.0, 0.0]], [], []),
    ],
)
@pytest.mark.parametrize("device", _devices())
def test_surface_transfer_handles_empty_sources_and_targets(
    positions, targets, expected, device
):
    """Empty live prefixes must not launch invalid zero-size Warp work."""
    transfer = _transfer_type()(device=device)
    particles = _particles(device, positions)
    targets_device = wp.array(
        np.asarray(targets, dtype=np.float32).reshape((-1, 3)),
        dtype=wp.vec3,
        device=device,
    )

    transfer.prepare(particles, live_count=len(positions), cell_size=1.0)
    first_device = transfer.nearest_indices(
        targets_device, target_count=len(targets)
    )
    first = first_device.numpy()
    second = transfer.nearest_indices(
        targets_device, target_count=len(targets)
    ).numpy()

    expected_array = np.asarray(expected, dtype=np.int32)
    assert first_device.dtype == wp.int32
    assert first.shape == expected_array.shape
    assert np.array_equal(first, expected_array)
    assert np.array_equal(second, expected_array)
    counts = transfer.operation_counts()
    assert counts["local_lookups"] == 0
    assert counts["fallback_launches"] == 0
    assert counts["gathers"] == 0
    assert counts["readbacks"] == 0


def test_hash_collision_is_filtered_before_global_fallback():
    """A wrapped HashGrid bucket must not masquerade as one of 27 cells."""
    transfer = _transfer_type()(device="cpu")
    # The minimum 4-cell grid hashes x=0 and x=4 into the same bucket.
    particles = _particles("cpu", [[4.25, 0.0, 0.0]])
    targets = wp.array(
        np.array([[0.25, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device="cpu",
    )
    transfer.prepare(particles, live_count=1, cell_size=1.0)

    actual = transfer.nearest_indices(targets, target_count=1).numpy()

    assert np.array_equal(actual, np.array([0], dtype=np.int32))
    assert transfer.fallback_target_count() == 1


@pytest.mark.parametrize("device", _devices())
def test_nearest_indices_match_cpu_contract_for_boundaries_ties_and_negatives(
    device,
):
    """Negative floors, boundary cells and duplicate ties stay byte exact."""
    positions = np.array(
        [
            [-1.25, 0.0, 0.0],
            [-0.75, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    expected = np.array([0, 2, 2, 5], dtype=np.int32)
    assert np.array_equal(
        nearest_sample_indices(targets, positions, cell_size=1.0), expected
    )
    transfer = _transfer_type()(device=device)
    transfer.prepare(_particles(device, positions), len(positions), 1.0)
    targets_device = wp.array(targets, dtype=wp.vec3, device=device)

    first = transfer.nearest_indices(targets_device, len(targets)).numpy()
    second = transfer.nearest_indices(targets_device, len(targets)).numpy()

    assert np.array_equal(first, expected)
    assert first.tobytes() == second.tobytes()
    assert transfer.operation_counts()["local_lookups"] == 1


@pytest.mark.parametrize("device", _devices())
def test_one_ulp_near_ties_match_numpy_exactly(device):
    """CUDA arithmetic must not flip winners at one-ULP boundaries."""
    one = np.float32(1.0)
    below = np.nextafter(one, np.float32(-np.inf), dtype=np.float32)
    above = np.nextafter(one, np.float32(np.inf), dtype=np.float32)
    positions = np.array(
        [
            [below, above, one],
            [above, below, one],
            [below, one, one],
            [above, one, one],
        ],
        dtype=np.float32,
    )
    targets = np.array(
        [[one, one, one], [one, above, one]], dtype=np.float32
    )
    expected = nearest_sample_indices(targets, positions, cell_size=0.25)
    assert np.array_equal(expected, np.array([2, 0], dtype=np.intp))
    transfer = _transfer_type()(device=device)
    transfer.prepare(_particles(device, positions), len(positions), 0.25)

    actual = transfer.nearest_indices(
        wp.array(targets, dtype=wp.vec3, device=device), len(targets)
    ).numpy()

    assert np.array_equal(actual, expected.astype(np.int32))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("live_count", [119_999, 120_000, 120_001, 240_001])
def test_stable_stride_cap_preserves_original_particle_indices(
    device, live_count
):
    """The 120k cap must retain stable original indices at every boundary."""
    stride = int(np.ceil(live_count / 120_000))
    expected_index = ((live_count - 1) // stride) * stride
    positions = np.zeros((live_count, 3), dtype=np.float32)
    positions[:, 0] = np.arange(live_count, dtype=np.float32) * np.float32(0.001)
    target = positions[expected_index : expected_index + 1].copy()
    transfer = _transfer_type()(device=device)
    transfer.prepare(_particles(device, positions), live_count, 0.01)

    actual = transfer.nearest_indices(
        wp.array(target, dtype=wp.vec3, device=device), 1
    ).numpy()

    assert np.array_equal(actual, np.array([expected_index], dtype=np.int32))


def test_invalidation_rebuilds_once_without_reallocating_publication_state():
    """A stale HostFrame signal must rebuild lookup state exactly once."""
    transfer = _transfer_type()(device="cpu")
    particles = _particles("cpu", [[0.0, 0.0, 0.0]])
    targets = wp.array(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device="cpu",
    )
    transfer.prepare(particles, 1, 1.0)
    transfer.nearest_indices(targets, 1)
    transfer.invalidate()
    transfer.prepare(particles, 1, 1.0)
    transfer.nearest_indices(targets, 1)
    transfer.prepare(particles, 1, 1.0)
    transfer.nearest_indices(targets, 1)

    counts = transfer.operation_counts()
    assert counts["sample_gathers"] == 2
    assert counts["hash_grid_builds"] == 2
    assert counts["local_lookups"] == 2


def test_global_transfer_counters_accumulate_across_host_frames():
    """The real profiler must see work from every short-lived HostFrame."""
    from fluidSim_warp_blender51.sim import surface_attribute_transfer as module

    module.surface_attribute_transfer_operation_counts(reset=True)
    for _ in range(2):
        transfer = _transfer_type()(device="cpu")
        particles = _particles(
            "cpu",
            [[0.0, 0.0, 0.0]],
            age=([1.0], wp.float32),
        )
        target = wp.array(
            np.array([[0.0, 0.0, 0.0]], np.float32),
            dtype=wp.vec3,
            device="cpu",
        )
        transfer.prepare(particles, 1, 1.0)
        transfer.nearest_indices(target, 1)
        transfer.gather("age", particles.attr("age"), 1)
        transfer.readback("age", 1)

    assert module.surface_attribute_transfer_operation_counts() == {
        "sample_gathers": 2,
        "hash_grid_builds": 2,
        "local_lookups": 2,
        "fallback_launches": 2,
        "gathers": 2,
        "readbacks": 2,
    }


def test_prepare_retries_after_hash_grid_build_failure():
    """A transient grid-build failure must not poison the frame signature."""
    transfer = _transfer_type()(device="cpu")
    particles = _particles("cpu", [[0.0, 0.0, 0.0]])

    class FailOnceGrid:
        def __init__(self):
            self.calls = 0

        def build(self, _points, _cell_size):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("injected hash-grid failure")

    grid = FailOnceGrid()
    transfer._grid = grid
    transfer._grid_dim = 1_000

    with pytest.raises(RuntimeError, match="injected hash-grid failure"):
        transfer.prepare(particles, 1, 1.0)
    transfer.prepare(particles, 1, 1.0)

    assert grid.calls == 2
    assert transfer.operation_counts()["sample_gathers"] == 2
    assert transfer.operation_counts()["hash_grid_builds"] == 1


def test_nearest_lookup_retries_after_launch_failure(monkeypatch):
    """A failed lookup launch must not expose an uninitialized cached map."""
    from fluidSim_warp_blender51.sim import surface_attribute_transfer as module

    transfer = _transfer_type()(device="cpu")
    particles = _particles("cpu", [[0.0, 0.0, 0.0]])
    targets = wp.array(
        np.array([[0.0, 0.0, 0.0]], np.float32),
        dtype=wp.vec3,
        device="cpu",
    )
    transfer.prepare(particles, 1, 1.0)
    real_launch = module.wp.launch
    fail = {"pending": True}

    def launch(kernel, *args, **kwargs):
        if kernel is module.k_nearest_local_27_cells and fail["pending"]:
            fail["pending"] = False
            raise RuntimeError("injected lookup failure")
        return real_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(module.wp, "launch", launch)
    with pytest.raises(RuntimeError, match="injected lookup failure"):
        transfer.nearest_indices(targets, 1)

    actual = transfer.nearest_indices(targets, 1).numpy()

    assert np.array_equal(actual, np.array([0], np.int32))
    assert transfer.operation_counts()["local_lookups"] == 1


# Display-path lookup: the device nearest map that replaced the NumPy
# grid-hash in output_publication._surface_values must reproduce
# _nearest_values bitwise (same cells, same tie order, same fallback).

_FIXTURE = Path(__file__).parent / "fixtures" / "chromatic_vortex_f24_surface_colour.npz"


def _display_values(device, verts, positions, values, cell_size):
    from fluidSim_warp_blender51.blender import output_publication

    transfer = _transfer_type()(device=device)
    transfer.prepare(
        _particles(device, positions), len(positions), max(float(cell_size), 1.0e-5))
    targets = wp.array(np.asarray(verts, np.float32), dtype=wp.vec3, device=device)
    nearest = transfer.display_nearest_indices(targets, len(verts))
    return output_publication._values_at_nearest(nearest, values)


def _cpu_values(verts, positions, values, cell_size):
    from fluidSim_warp_blender51.blender import output_publication

    return output_publication._nearest_values(verts, positions, values, cell_size)


@pytest.mark.parametrize("device", _devices())
def test_display_lookup_matches_cpu_on_chromatic_vortex_fixture(device):
    """Real frame-24 surface colours: the device map equals the CPU lookup bitwise."""
    data = np.load(_FIXTURE)
    args = (data["verts"], data["positions"], data["values"], float(data["cell_size"]))
    cpu = _cpu_values(*args)

    assert np.array_equal(cpu[:-3], data["full_frame_values"])
    assert np.array_equal(_display_values(device, *args), cpu)
    # The three far vertices exercise the no-neighbour mean fallback.
    assert np.array_equal(cpu[-3:], np.broadcast_to(np.nanmean(data["values"], 0), cpu[-3:].shape))


@pytest.mark.parametrize("device", _devices())
def test_display_lookup_matches_cpu_tie_order(device):
    """Duplicates and equidistant samples in different cells pick the CPU winner."""
    positions = np.array([
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],          # duplicate, same cell
        [-0.5, 0.0, 0.0], [1.5, 0.0, 0.0],         # equidistant, different cells
        [0.5, -0.5, 0.5], [0.5, 1.5, 0.5],
    ], np.float32)
    values = np.arange(len(positions) * 3, dtype=np.float32).reshape(-1, 3)
    verts = np.array([[0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5],
                      [0.5, 0.5, 0.25]], np.float32)
    args = (verts, positions, values, 1.0)

    assert np.array_equal(_display_values(device, *args), _cpu_values(*args))


@pytest.mark.parametrize("device", _devices())
def test_display_lookup_matches_cpu_above_sample_cap(device):
    """Above the 120k cap both paths decimate with the same stride."""
    from fluidSim_warp_blender51.blender import output_publication
    from fluidSim_warp_blender51.sim import surface_attribute_transfer

    assert (output_publication._DISPLAY_MAX_SAMPLES
            == surface_attribute_transfer.MAX_SURFACE_ATTRIBUTE_SAMPLES)
    rng = np.random.default_rng(7)
    positions = rng.random((250_001, 3), dtype=np.float32)
    values = rng.random((250_001,), dtype=np.float32)
    verts = rng.random((2_000, 3), dtype=np.float32) * 1.2 - 0.1
    args = (verts, positions, values, 0.02)

    assert np.array_equal(_display_values(device, *args), _cpu_values(*args))


@pytest.mark.parametrize("device", _devices())
def test_display_lookup_empty_inputs_and_export_map_isolation(device):
    """Empty inputs follow the CPU contract; the export map cache is untouched."""
    transfer = _transfer_type()(device=device)
    targets = wp.array(np.array([[0, 0, 0], [20, 0, 0]], np.float32),
                       dtype=wp.vec3, device=device)
    with pytest.raises(RuntimeError, match="prepare"):
        transfer.display_nearest_indices(targets, 2)

    transfer.prepare(_particles(device, np.zeros((0, 3))), 0, 1.0)
    assert np.array_equal(transfer.display_nearest_indices(targets, 2), [-1, -1])

    transfer.prepare(_particles(device, [[0.2, 0, 0], [8.8, 0, 0]]), 2, 1.0)
    export = transfer.nearest_indices(targets, 2).numpy().copy()
    assert transfer.display_nearest_indices(targets, 0).shape == (0,)
    assert np.array_equal(transfer.display_nearest_indices(targets, 2), [0, -1])
    assert np.array_equal(transfer.nearest_indices(targets, 2).numpy(), export)
    assert transfer.operation_counts()["local_lookups"] == 1
