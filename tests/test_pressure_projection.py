"""Pressure projection + solver convergence validation (tests 8-9)."""
import sys
from _subset_runner import run_subset

if __name__ == "__main__":
    sys.exit(run_subset([7, 8], "PRESSURE-PROJECTION"))
