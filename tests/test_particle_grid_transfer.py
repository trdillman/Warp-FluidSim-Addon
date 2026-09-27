"""P2G/G2P transfer validation (tests 4-5)."""
import sys

import pytest

from _subset_runner import check_validation, run_subset, validation_id

INDICES = [3, 4]


@pytest.mark.parametrize("index", INDICES, ids=validation_id)
def test_particle_grid_transfer_validation(cuda_device, index):
    check_validation(index, cuda_device)


if __name__ == "__main__":
    sys.exit(run_subset(INDICES, "PARTICLE-GRID-TRANSFER"))
