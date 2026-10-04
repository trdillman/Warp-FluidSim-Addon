"""End-to-end solver validation: sourcing, collision, surface, whitewater."""
import sys

import pytest

from _subset_runner import check_validation, run_subset, validation_id

INDICES = [5, 6, 7, 8, 11, 13, 14]


@pytest.mark.parametrize("index", INDICES, ids=validation_id)
def test_solver_validation(cuda_device, index):
    check_validation(index, cuda_device)


if __name__ == "__main__":
    sys.exit(run_subset(INDICES, "SOLVER-VALIDATION"))
