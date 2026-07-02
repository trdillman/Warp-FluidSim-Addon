"""End-to-end solver validation: sourcing, collision bits, surface,
whitewater, benchmark schema (tests 6-7, 10, 12-13)."""
import sys
from _subset_runner import run_subset

if __name__ == "__main__":
    sys.exit(run_subset([5, 6, 9, 11, 12], "SOLVER-VALIDATION"))
