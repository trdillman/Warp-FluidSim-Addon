"""GPU-resident nearest-particle transfer for opt-in surface attributes."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

MAX_SURFACE_ATTRIBUTE_SAMPLES = 120_000
_COUNTER_NAMES = (
    "sample_gathers",
    "hash_grid_builds",
    "local_lookups",
    "fallback_launches",
    "gathers",
    "readbacks",
)
_GLOBAL_OPERATIONS = Counter()


def surface_attribute_transfer_operation_counts(*, reset=False):
    """Aggregate transfer operations across short-lived HostFrames."""
    result = {
        name: int(_GLOBAL_OPERATIONS[name]) for name in _COUNTER_NAMES
    }
    if reset:
        _GLOBAL_OPERATIONS.clear()
    return result


@wp.kernel
def k_gather_sample_positions(
    positions: wp.array(dtype=wp.vec3),
    live_count: int,
    stride: int,
    sampled: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    source_index = tid * stride
    if source_index < live_count:
        sampled[tid] = positions[source_index]


@wp.kernel
def k_nearest_local_27_cells(
    grid_id: wp.uint64,
    sampled: wp.array(dtype=wp.vec3),
    sample_count: int,
    stride: int,
    targets: wp.array(dtype=wp.vec3),
    target_count: int,
    cell_size: float,
    nearest: wp.array(dtype=wp.int32),
    fallback_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= target_count:
        return
    target = targets[tid]
    target_x = int(wp.floor(target[0] / cell_size))
    target_y = int(wp.floor(target[1] / cell_size))
    target_z = int(wp.floor(target[2] / cell_size))
    best_distance = wp.float32(3.402823466e38)
    best_index = wp.int32(2147483647)

    # Two cell widths are deliberately conservative. Exact cell filtering
    # below preserves the CPU contract even when the grid hashes collide.
    query = wp.hash_grid_query(grid_id, target, cell_size * 2.0)
    for sample_index in query:
        if sample_index >= sample_count:
            continue
        point = sampled[sample_index]
        point_x = int(wp.floor(point[0] / cell_size))
        point_y = int(wp.floor(point[1] / cell_size))
        point_z = int(wp.floor(point[2] / cell_size))
        if (
            wp.abs(point_x - target_x) > 1
            or wp.abs(point_y - target_y) > 1
            or wp.abs(point_z - target_z) > 1
        ):
            continue
        delta = target - point
        distance = wp.dot(delta, delta)
        original_index = sample_index * stride
        if distance < best_distance or (
            distance == best_distance and original_index < best_index
        ):
            best_distance = distance
            best_index = original_index
    if best_index == 2147483647:
        nearest[tid] = -1
        wp.atomic_add(fallback_count, 0, 1)
    else:
        nearest[tid] = best_index


@wp.kernel
def k_nearest_global_fallback(
    sampled: wp.array(dtype=wp.vec3),
    sample_count: int,
    stride: int,
    targets: wp.array(dtype=wp.vec3),
    target_count: int,
    nearest: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= target_count or nearest[tid] >= 0:
        return
    target = targets[tid]
    best_distance = wp.float32(3.402823466e38)
    best_index = wp.int32(2147483647)
    for sample_index in range(sample_count):
        delta = target - sampled[sample_index]
        distance = wp.dot(delta, delta)
        original_index = sample_index * stride
        if distance < best_distance or (
            distance == best_distance and original_index < best_index
        ):
            best_distance = distance
            best_index = original_index
    nearest[tid] = best_index


@wp.kernel
def k_gather_float(
    nearest: wp.array(dtype=wp.int32),
    source: wp.array(dtype=float),
    target_count: int,
    output: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid < target_count:
        output[tid] = source[nearest[tid]]


@wp.kernel
def k_gather_int32(
    nearest: wp.array(dtype=wp.int32),
    source: wp.array(dtype=wp.int32),
    target_count: int,
    output: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid < target_count:
        output[tid] = source[nearest[tid]]


@wp.kernel
def k_gather_vec3(
    nearest: wp.array(dtype=wp.int32),
    source: wp.array(dtype=wp.vec3),
    target_count: int,
    output: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid < target_count:
        output[tid] = source[nearest[tid]]


@wp.kernel
def k_gather_vec4(
    nearest: wp.array(dtype=wp.int32),
    source: wp.array(dtype=wp.vec4),
    target_count: int,
    output: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    if tid < target_count:
        output[tid] = source[nearest[tid]]


class SurfaceAttributeTransfer:
    """Persistent buffers and exact nearest lookup for one output owner."""

    def __init__(self, *, device):
        self.device = wp.get_device(device)
        self._counts = Counter()
        self._sample_capacity = 0
        self._target_capacity = 0
        self._sampled = None
        self._nearest = None
        # Warp 1.16 CUDA zero-length slices cannot be materialized with
        # ``.numpy()``. Keep an owned empty allocation for the direct API.
        self._empty_nearest = wp.empty(
            0, dtype=wp.int32, device=self.device
        )
        self._grid = None
        self._grid_dim = 0
        self._prepared_signature = None
        self._nearest_signature = None
        self._sample_count = 0
        self._stride = 1
        self._cell_size = 0.0
        self._outputs = {}
        self._fallback_count = wp.zeros(1, dtype=wp.int32, device=self.device)

    def operation_counts(self) -> dict[str, int]:
        return {name: int(self._counts[name]) for name in _COUNTER_NAMES}

    def _record(self, name: str) -> None:
        self._counts[name] += 1
        _GLOBAL_OPERATIONS[name] += 1

    def fallback_target_count(self) -> int:
        """Synchronize the optional profiler metric for the latest lookup."""
        return int(self._fallback_count.numpy()[0])

    def invalidate(self) -> None:
        self._prepared_signature = None
        self._nearest_signature = None

    def _ensure_sample_capacity(self, capacity: int) -> None:
        if capacity <= self._sample_capacity:
            return
        self._sample_capacity = max(capacity, self._sample_capacity * 2, 1)
        self._sampled = wp.empty(
            self._sample_capacity, dtype=wp.vec3, device=self.device
        )

    def _ensure_target_capacity(self, capacity: int) -> None:
        if capacity <= self._target_capacity:
            return
        self._target_capacity = max(capacity, self._target_capacity * 2, 1)
        self._nearest = wp.empty(
            self._target_capacity, dtype=wp.int32, device=self.device
        )

    def _ensure_grid(self, sample_count: int) -> None:
        grid_dim = max(4, int(math.ceil(max(1, sample_count) ** (1.0 / 3.0))))
        if self._grid is not None and grid_dim <= self._grid_dim:
            return
        self._grid_dim = grid_dim
        self._grid = wp.HashGrid(
            grid_dim, grid_dim, grid_dim, device=self.device, dtype=wp.float32
        )

    def prepare(self, particles, live_count: int, cell_size: float) -> None:
        live_count = int(live_count)
        cell_size = float(cell_size)
        if live_count < 0 or live_count > int(particles.position.size):
            raise ValueError("live particle count is outside the position buffer")
        if not math.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError("cell_size must be finite and positive")
        signature = (id(particles), id(particles.position), live_count, cell_size)
        if signature == self._prepared_signature:
            return
        self._nearest_signature = None
        self._prepared_signature = None
        self._cell_size = cell_size
        self._stride = max(
            1, int(math.ceil(live_count / MAX_SURFACE_ATTRIBUTE_SAMPLES))
        )
        self._sample_count = (
            int(math.ceil(live_count / self._stride)) if live_count else 0
        )
        if not self._sample_count:
            self._prepared_signature = signature
            return
        self._ensure_sample_capacity(self._sample_count)
        wp.launch(
            k_gather_sample_positions,
            dim=self._sample_count,
            inputs=[
                particles.position,
                live_count,
                self._stride,
                self._sampled,
            ],
            device=self.device,
        )
        self._record("sample_gathers")
        self._ensure_grid(self._sample_count)
        self._grid.build(self._sampled[: self._sample_count], cell_size)
        self._record("hash_grid_builds")
        self._prepared_signature = signature

    def nearest_indices(self, targets_device, target_count: int):
        if self._prepared_signature is None:
            raise RuntimeError("prepare() must be called before nearest_indices()")
        target_count = int(target_count)
        if target_count < 0 or target_count > int(targets_device.size):
            raise ValueError("target count is outside the target buffer")
        signature = (id(targets_device), target_count, self._prepared_signature)
        if signature == self._nearest_signature:
            if not target_count:
                return self._empty_nearest
            return self._nearest[:target_count]
        self._nearest_signature = None
        self._ensure_target_capacity(max(1, target_count))
        if not target_count:
            self._nearest_signature = signature
            return self._empty_nearest
        if not self._sample_count:
            self._nearest[:target_count].fill_(-1)
            self._fallback_count.fill_(target_count)
            self._nearest_signature = signature
            return self._nearest[:target_count]
        self._fallback_count.zero_()
        wp.launch(
            k_nearest_local_27_cells,
            dim=target_count,
            inputs=[
                self._grid.id,
                self._sampled,
                self._sample_count,
                self._stride,
                targets_device,
                target_count,
                self._cell_size,
                self._nearest,
                self._fallback_count,
            ],
            device=self.device,
        )
        self._record("local_lookups")
        wp.launch(
            k_nearest_global_fallback,
            dim=target_count,
            inputs=[
                self._sampled,
                self._sample_count,
                self._stride,
                targets_device,
                target_count,
                self._nearest,
            ],
            device=self.device,
        )
        self._record("fallback_launches")
        self._nearest_signature = signature
        return self._nearest[:target_count]

    def gather(self, key: str, particle_array, target_count: int):
        """Gather one particle channel through the prepared nearest map."""
        if self._nearest_signature is None:
            raise RuntimeError("nearest_indices() must be called before gather()")
        target_count = int(target_count)
        dtype = particle_array.dtype
        kernels = {
            wp.float32: k_gather_float,
            wp.int32: k_gather_int32,
            wp.vec3: k_gather_vec3,
            wp.vec4: k_gather_vec4,
        }
        if dtype not in kernels:
            raise TypeError(f"unsupported surface attribute dtype {dtype}")
        key = str(key)
        existing = self._outputs.get(key)
        if existing is None or target_count > existing[1]:
            capacity = max(target_count, existing[1] * 2 if existing else 1)
            output = wp.empty(capacity, dtype=dtype, device=self.device)
            self._outputs[key] = (output, capacity, dtype)
        else:
            output, _, existing_dtype = existing
            if dtype != existing_dtype:
                raise TypeError(
                    f"surface attribute {key!r} changed device dtype"
                )
        wp.launch(
            kernels[dtype],
            dim=target_count,
            inputs=[self._nearest, particle_array, target_count, output],
            device=self.device,
        )
        self._record("gathers")
        return output[:target_count]

    def readback(self, key: str, target_count: int) -> np.ndarray:
        """Read back only the gathered live surface prefix."""
        key = str(key)
        if key not in self._outputs:
            raise KeyError(key)
        output = self._outputs[key][0]
        target_count = int(target_count)
        if target_count < 0 or target_count > self._outputs[key][1]:
            raise ValueError("target count is outside the gathered output buffer")
        if not target_count:
            dtype = np.int32 if output.dtype == wp.int32 else np.float32
            components = int(getattr(output.dtype, "_length_", 1))
            shape = (0,) if components == 1 else (0, components)
            return np.zeros(shape, dtype=dtype)
        self._record("readbacks")
        return np.ascontiguousarray(output[:target_count].numpy())
