import json
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.diagnostics.benchmark import save_benchmark
from fluidSim_warp_blender51.diagnostics.report import write_report
from fluidSim_warp_blender51.io import export as export_mod
from fluidSim_warp_blender51.io.export import (
    export_frame,
    export_points_ply,
    export_surface_obj,
)


def _export_solver(*, whitewater=True):
    host = SimpleNamespace(
        surface_mesh=(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
        ),
        positions=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        velocities=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        whitewater=(
            np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
            if whitewater else np.zeros((0, 3), dtype=np.float32)
        ),
    )
    return SimpleNamespace(
        host_frame=lambda: host,
        settings=SimpleNamespace(
            surface_enabled=True,
            surface_resolution_mode=0,
            surface_resolution_multiplier=1.0,
            surface_smooth_iterations=0,
        ),
        surface=SimpleNamespace(metadata=lambda: {"voxel_size": 0.1}),
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


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_export_surface_obj_rejects_non_finite_vertices_before_write(
        tmp_path, bad_value):
    path = tmp_path / "surface.obj"
    solver = _export_solver()
    host = solver.host_frame()
    host.surface_mesh[0][0, 0] = bad_value

    with pytest.raises(ValueError, match="surface vertices must be finite"):
        export_surface_obj(solver, str(path), host=host)

    assert not path.exists()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["positions", "velocities"])
def test_export_points_ply_rejects_non_finite_data_before_write(
        tmp_path, bad_value, field):
    path = tmp_path / "points.ply"
    positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    velocities = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    if field == "positions":
        positions[0, 0] = bad_value
    else:
        velocities[0, 0] = bad_value

    with pytest.raises(ValueError, match=f"{field} must be finite"):
        export_points_ply(positions, str(path), velocities=velocities)

    assert not path.exists()


def test_export_frame_rejects_missing_parent_before_solver_reads(tmp_path):
    path = tmp_path / "missing" / "frame"
    solver = SimpleNamespace(settings=SimpleNamespace(surface_enabled=True))

    with pytest.raises(FileNotFoundError, match="Output parent directory does not exist"):
        export_frame(solver, str(path))

    assert not path.parent.exists()


def test_export_frame_rolls_back_complete_bundle_after_late_publish_failure(
        tmp_path, monkeypatch):
    base = tmp_path / "frame"
    prior = {
        tmp_path / "frame_surface.obj": b"prior surface\n",
        tmp_path / "frame_particles.ply": b"prior particles\n",
        tmp_path / "frame_whitewater.ply": b"prior whitewater\n",
    }
    for path, content in prior.items():
        path.write_bytes(content)

    real_replace = export_mod.os.replace

    def fail_final_sibling_publish(source, destination):
        if str(source).endswith(".stage") and str(destination).endswith(
                "_whitewater.ply"):
            raise OSError("injected final sibling publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(export_mod.os, "replace", fail_final_sibling_publish)

    with pytest.raises(OSError, match="final sibling publication failure"):
        export_frame(_export_solver(), str(base))

    assert {path: path.read_bytes() for path in prior} == prior
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        path.name for path in prior
    )


def test_export_frame_prevalidates_every_sibling_before_touching_final_paths(
        tmp_path):
    base = tmp_path / "frame"
    prior = {
        tmp_path / "frame_surface.obj": b"prior surface\n",
        tmp_path / "frame_particles.ply": b"prior particles\n",
        tmp_path / "frame_whitewater.ply": b"prior whitewater\n",
    }
    for path, content in prior.items():
        path.write_bytes(content)
    solver = _export_solver()
    solver.host_frame().whitewater[0, 0] = np.nan

    with pytest.raises(ValueError, match="positions must be finite"):
        export_frame(solver, str(base))

    assert {path: path.read_bytes() for path in prior} == prior
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        path.name for path in prior
    )


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
