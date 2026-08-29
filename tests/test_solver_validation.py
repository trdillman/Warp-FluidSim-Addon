"""End-to-end solver validation: sourcing, collision, surface, whitewater."""
import sys
from _subset_runner import run_subset

if __name__ == "__main__":
    sys.exit(run_subset([5, 6, 7, 8, 11, 13, 14], "SOLVER-VALIDATION"))
