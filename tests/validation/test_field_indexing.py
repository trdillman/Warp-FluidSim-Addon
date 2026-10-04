"""Sparse tiled leaf-grid indexing/addressing validation (tests 1-3)."""
import sys

import pytest

from _subset_runner import check_validation, run_subset, validation_id

INDICES = [0, 1, 2]


@pytest.mark.parametrize("index", INDICES, ids=validation_id)
def test_leaf_grid_indexing_validation(cuda_device, index):
    check_validation(index, cuda_device)


if __name__ == "__main__":
    sys.exit(run_subset(INDICES, "LEAF-GRID-INDEXING"))
