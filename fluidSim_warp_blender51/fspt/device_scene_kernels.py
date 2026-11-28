"""Validated device-to-aggregate copies; malformed values never reach the BVH."""
import warp as wp


@wp.kernel
def copy_positions(source: wp.array(dtype=wp.vec3), destination: wp.array(dtype=wp.vec3),
                   offset: int, status: wp.array(dtype=wp.int32)):
    i = wp.tid()
    value = source[i]
    if not (wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])):
        wp.atomic_max(status, 0, 1)
        value = wp.vec3(0.0)
    destination[offset + i] = value


@wp.kernel
def copy_indices(source: wp.array(dtype=wp.int32), destination: wp.array(dtype=wp.int32),
                 index_offset: int, vertex_offset: int, vertex_count: int,
                 status: wp.array(dtype=wp.int32)):
    i = wp.tid()
    value = source[i]
    if value < 0 or value >= vertex_count:
        wp.atomic_max(status, 0, 2)
        value = 0
    destination[index_offset + i] = value + vertex_offset
