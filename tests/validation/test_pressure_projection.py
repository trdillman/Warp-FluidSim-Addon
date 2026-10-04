"""Pressure projection + solver convergence validation (tests 8-9)."""
import sys

import pytest

from _subset_runner import check_validation, run_subset, validation_id

INDICES = [7, 8]


@pytest.mark.parametrize("index", INDICES, ids=validation_id)
def test_pressure_projection_validation(cuda_device, index):
    check_validation(index, cuda_device)


if __name__ == "__main__":
    sys.exit(run_subset(INDICES, "PRESSURE-PROJECTION"))
