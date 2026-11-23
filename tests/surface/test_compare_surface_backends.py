"""The surface-backend comparison tool reports sane numbers on a known sphere."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import surface as SURF  # noqa: E402
from tools.profiling import compare_surface_backends as CMP  # noqa: E402

from test_surface_extractor_parity import CASES, _sphere_field  # noqa: E402


def test_mesh_stats_on_a_known_octahedron():
    verts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                      [0, 0, 1], [0, 0, -1]], np.float32)
    tris = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                     [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], np.int32)
    stats = CMP.mesh_stats(verts, tris)
    assert stats["tris"] == 8 and stats["verts"] == 6
    assert stats["open_edges"] == 0 and stats["bad_edges"] == 0
    assert stats["quality_mean"] == pytest.approx(1.0, abs=1e-6)
    assert stats["sliver_percent"] == 0.0
    # Removing one face opens three edges.
    assert CMP.mesh_stats(verts, tris[1:])["open_edges"] == 3


def test_compare_reports_every_available_backend(cuda_device):
    grid, field = _sphere_field(cuda_device, *CASES["interior"])
    result = CMP.compare_backends_on_sdf(grid, field, cuda_device, 512, repeats=2)
    assert set(result) == set(CMP.BACKENDS)
    for name, record in result.items():
        if "skipped" in record:
            continue
        assert "error" not in record, (name, record)
        assert record["tris"] > 0 and record["ms"] >= 0.0
        assert record["repeatable"] is True
        assert record["open_edges"] == 0 and record["bad_edges"] == 0
    if SURF.SURFACE_NETS_AVAILABLE:
        assert result["nets"]["quality_mean"] > result["dense"]["quality_mean"]


@pytest.mark.parametrize("emit_frames", [0, 1, 4])
def test_solver_comparison_emits_for_requested_steps(monkeypatch, emit_frames):
    from fluidSim_warp_blender51.sim import solver
    from tools.profiling import profile_gpu_solver as profile

    active_steps = []
    released = []
    sources = object()

    class RecordingSolver:
        active = False

        def __init__(self, settings):
            pass

        def set_source_batch(self, batch=None):
            self.active = batch is sources

        def step_frame(self):
            active_steps.append(self.active)

        def release(self):
            released.append(True)

    monkeypatch.setattr(solver, "FluidSolver", RecordingSolver)
    monkeypatch.setattr(profile, "_build_settings", lambda args: None)
    monkeypatch.setattr(profile, "build_sources", lambda args: sources)
    monkeypatch.setattr(CMP, "_sample", lambda *args: {"backends": {}})
    args = CMP.parse(["--solver-preset", "dam_break", "--frames", "5",
                      "--emit-frames", str(emit_frames)])
    CMP.run_solver_preset(args)

    assert active_steps == [True] * emit_frames + [False] * (5 - emit_frames)
    assert released == [True]
