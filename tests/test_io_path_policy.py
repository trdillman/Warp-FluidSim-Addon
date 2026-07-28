import json
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.diagnostics.benchmark import save_benchmark
from fluidSim_warp_blender51.diagnostics.report import write_report
from fluidSim_warp_blender51.io.export import (
    export_frame,
    export_points_ply,
    export_surface_obj,
)


def test_export_surface_obj_rejects_missing_parent_before_write(tmp_path):
    path = tmp_path / "missing" / "surface.obj"
    solver = SimpleNamespace(
        surface_mesh=lambda: (
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
        )
    )

    with pytest.raises(FileNotFoundError, match="Output parent directory does not exist"):
        export_surface_obj(solver, str(path))

    assert not path.exists()


def test_export_points_ply_rejects_missing_parent_before_write(tmp_path):
    path = tmp_path / "missing" / "points.ply"

    with pytest.raises(FileNotFoundError, match="Output parent directory does not exist"):
        export_points_ply(np.zeros((1, 3), dtype=np.float32), str(path))

    assert not path.exists()


def test_export_frame_rejects_missing_parent_before_solver_reads(tmp_path):
    path = tmp_path / "missing" / "frame"
    solver = SimpleNamespace(settings=SimpleNamespace(surface_enabled=True))

    with pytest.raises(FileNotFoundError, match="Output parent directory does not exist"):
        export_frame(solver, str(path))

    assert not path.parent.exists()


def test_write_report_rejects_missing_parent_before_write(tmp_path):
    path = tmp_path / "missing" / "report.json"

    with pytest.raises(FileNotFoundError, match="Output parent directory does not exist"):
        write_report(str(path), {"ok": True})

    assert not path.exists()


def test_save_benchmark_rejects_missing_parent_before_write(tmp_path):
    path = tmp_path / "missing" / "benchmark.json"

    with pytest.raises(FileNotFoundError, match="Output parent directory does not exist"):
        save_benchmark(str(path), {"result": "completed"})

    assert not path.exists()


def test_report_and_benchmark_write_existing_parent(tmp_path):
    report_path = tmp_path / "report.json"
    benchmark_path = tmp_path / "benchmark.json"

    assert write_report(str(report_path), {"ok": True}) == str(report_path)
    save_benchmark(str(benchmark_path), {"result": "completed"})

    assert json.loads(report_path.read_text(encoding="utf-8")) == {"ok": True}
    assert json.loads(benchmark_path.read_text(encoding="utf-8")) == {
        "result": "completed"
    }
