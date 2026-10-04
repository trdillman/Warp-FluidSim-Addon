"""Focused contracts for ST-FLIP transfer behavior at support/grid boundaries.

The real-kernel behavior is exercised by ``headless_spatiotemporal_validation.py``.
These source-level checks keep the two narrow regression fixes explicit so a future
kernel refactor cannot silently restore particle loss or sparse-fringe dilution.
"""

from pathlib import Path


KERNELS = Path(__file__).parents[2] / "fluidSim_warp_blender51" / "sim" / "kernels_fluid.py"


def _source() -> str:
    return KERNELS.read_text(encoding="utf-8")


def test_spacetime_transfer_clamps_residual_time_to_temporal_support():
    source = _source()
    assert (
        "tau = wp.clamp(-time_offsets[tid] / safe_prev_dt, -0.5, 0.5)"
        in source
    )
    assert "wt = _st_temporal_weight(tau)" in source


def test_sparse_fringe_phase_averages_only_faces_that_exist():
    source = _source()
    assert "nfaces = float(3.0)" in source
    assert source.count("nfaces += 1.0") == 3
    assert "phi = accum / nfaces" in source
    assert "phi = accum / 6.0" not in source
