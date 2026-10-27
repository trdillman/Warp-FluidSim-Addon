"""Pure allocation facts consumed by grid owners and conservative costing.

Widths count 32-bit words, not allocator reservation or whole-solver overhead.
Keep allocation order stable; the owners resolve the Warp types and extents.
"""

GRID_BUFFER_LAYOUT = (
    ("leaf_table", "cells", 1),
    ("leaf_coords", "leaves", 3),
    ("num_leaves", "scalar", 1),
    ("overflow", "scalar", 1),
    ("_candidate_leaf_table", "cells", 1),
    ("_candidate_leaf_coords", "leaves", 3),
    ("_candidate_num_leaves", "scalar", 1),
    ("_candidate_overflow", "scalar", 1),
    ("leaf_flags", "cells", 1),
    ("leaf_flags_tmp", "cells", 1),
    ("_leaf_pending", "cells", 1),
    ("_leaf_pending_rank", "cells", 1),
    ("_leaf_slot_base", "scalar", 1),
)
SIMULATION_TABLE_NAMES = ("old_leaf_table",)
GRID_TABLE_BYTES_PER_CELL = sum(
    words * 4 for _, extent, words in GRID_BUFFER_LAYOUT if extent == "cells")
GRID_COORD_BYTES_PER_LEAF = sum(
    words * 4 for _, extent, words in GRID_BUFFER_LAYOUT if extent == "leaves")
SIMULATION_TABLE_BYTES_PER_CELL = len(SIMULATION_TABLE_NAMES) * 4

# SurfaceTensionSolver publishes these only after every allocation succeeds.
CAPILLARY_BUFFER_LAYOUT = (
    ("_normal_a", 3), ("_normal_b", 3), ("_phi_a", 1), ("_phi_b", 1),
)
CAPILLARY_BYTES_PER_VOXEL = sum(words * 4 for _, words in CAPILLARY_BUFFER_LAYOUT)
