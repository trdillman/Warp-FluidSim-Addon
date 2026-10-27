"""Pure layout facts for ParticleSystem's optional double-buffered state."""

from .attributes import ATTRIBUTE_SPECS


PHYSICAL_ATTR_NAMES = ("age", "life", "temperature", "viscosity", "density")
OPTIONAL_VEC4_ATTR_NAMES = ("color",)
OPTIONAL_INT_ATTR_NAMES = ("source_id", "uid")
AFFINE_BUFFER_NAMES = ("affine", "_affine2")
AFFINE_BYTES_PER_PARTICLE = len(AFFINE_BUFFER_NAMES) * 9 * 4  # float32 mat33
BIN_VOXEL_BUFFER_NAMES = ("counts", "starts")
BIN_PARTICLE_BUFFER_NAMES = ("indices", "arrival", "unordered")


def bin_voxel_bytes(voxel_capacity: int) -> int:
    """Full voxel-sized bin scratch replaced alongside a growing liquid pool."""
    return max(0, int(voxel_capacity)) * len(BIN_VOXEL_BUFFER_NAMES) * 4


def optional_particle_bytes_per_slot(channels) -> int:
    """Payload of selected product channels and their compaction buffers."""
    selected = frozenset(channels)
    known = set(OPTIONAL_VEC4_ATTR_NAMES) | set(OPTIONAL_INT_ATTR_NAMES)
    unknown = selected - known
    if unknown:
        raise ValueError(f"unsupported optional particle channels: {sorted(unknown)}")
    return 2 * sum(ATTRIBUTE_SPECS[name].bytes_per_element for name in selected)
