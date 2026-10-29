"""Whitewater emission-workspace layout shared by storage and costing.

Keep this module free of Warp: admission must not initialize a device merely
to learn the payload of the workspace that liquid-capacity growth replaces.
"""

# dtype names are resolved by the allocation owner, never by admission.
WHITEWATER_BUFFER_LAYOUT = {
    "position": ("vec3", 12),
    "velocity": ("vec3", 12),
    "age": ("float32", 4),
    "life": ("float32", 4),
    "alive": ("int32", 4),
    "type": ("int32", 4),
    "free_slots": ("int32", 4),
    "dead": ("int32", 4),
    "dead_offsets": ("int32", 4),
    "_position2": ("vec3", 12),
    "_velocity2": ("vec3", 12),
    "_age2": ("float32", 4),
    "_life2": ("float32", 4),
    "_type2": ("int32", 4),
    "_scan": ("int32", 4),
}
WHITEWATER_BYTES_PER_PARTICLE = sum(size for _, size in WHITEWATER_BUFFER_LAYOUT.values())

CANDIDATE_BUFFER_NAMES = ("candidates", "candidate_offsets", "candidate_types")
CANDIDATE_BYTES_PER_LIQUID = len(CANDIDATE_BUFFER_NAMES) * 4  # int32


def candidate_workspace_bytes(capacity: int) -> int:
    """Payload of all replacement candidate arrays, excluding allocator overhead."""
    return max(1, int(capacity)) * CANDIDATE_BYTES_PER_LIQUID
