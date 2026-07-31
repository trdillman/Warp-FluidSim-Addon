"""Blender/CUDA acceptance for transactional sparse-topology refusal."""

from pathlib import Path
import sys

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluidSim_warp_blender51.sim.fields import (  # noqa: E402
    SparseGrid,
    TopologyCapacityError,
)


def main():
    wp.init()
    device = "cuda:0"
    if not wp.get_device(device).is_cuda:
        raise RuntimeError("CUDA device required")
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0),
        voxel_size=0.1, leaf_capacity=1, device=device)

    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    assert grid.finalize_topology() == 1
    live_table = grid.leaf_table.numpy().copy()
    live_coords = grid.leaf_coords.numpy().copy()
    live_count = grid.num_leaves.numpy().copy()

    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (1.999, 1.999, 1.999))
    try:
        grid.finalize_topology()
    except TopologyCapacityError as exc:
        assert exc.required > exc.capacity == 1
    else:
        raise AssertionError("expected sparse-topology refusal")

    assert grid.active_leaf_count == 1
    np.testing.assert_array_equal(grid.leaf_table.numpy(), live_table)
    np.testing.assert_array_equal(grid.leaf_coords.numpy(), live_coords)
    np.testing.assert_array_equal(grid.num_leaves.numpy(), live_count)
    print("TOPOLOGY-TRANSACTION-ACCEPTANCE-PASS")


if __name__ == "__main__":
    main()
