"""Jacobi's residual is reduced on the device with one scalar readback (P10).

``measure_residual`` used to read the whole rhs and A*p arrays back and take
``np.linalg.norm`` of their float32 difference. It now forms the same float32
difference on the device and accumulates it in float64. The value must match
the old host result to float32 tolerance and the exact float64 norm closely,
and no array longer than one element may be read back.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

wp = pytest.importorskip("warp")


def _solved_block(device, side=0.16, iterations=12):
    from fluidSim_warp_blender51.sim import pressure as P
    from fluidSim_warp_blender51.sim.constants import VOXEL_FLUID
    from fluidSim_warp_blender51.sim.fields import (
        MacVelocityField, ScalarField, SparseGrid,
    )

    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(side,) * 3, voxel_size=0.01,
                      leaf_capacity=64, device=device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (side - 1e-4,) * 3)
    grid.finalize_topology()
    grid.refresh_topology_counts()
    vt = ScalarField(grid, dtype=wp.int32)
    vt.data.fill_(VOXEL_FLUID)
    vel = MacVelocityField(grid)
    vel.zero()
    rng = np.random.default_rng(3)
    vel.u.assign(wp.array(rng.standard_normal(grid.voxel_capacity).astype(np.float32),
                          device=device))
    counts = ScalarField(grid, dtype=wp.int32)
    collision = wp.zeros(grid.voxel_capacity, dtype=wp.vec3, device=device)
    solver = P.PressureSolver(grid, device)
    solver.build_rhs(vel, vt.data, collision, counts.data)
    solver.build_preconditioner(vt.data)
    solver.solve_jacobi(vt.data, iterations)
    return solver, vt


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
def test_device_residual_matches_the_host_norm(device_kind, request, monkeypatch):
    device = (request.getfixturevalue("cuda_device")
              if device_kind == "cuda" else "cpu")
    solver, vt = _solved_block(device)
    n = solver._dim()
    assert n > 1_000

    sizes = []
    original = wp.array.numpy

    def recording_numpy(self):
        sizes.append(int(self.size))
        return original(self)

    monkeypatch.setattr(wp.array, "numpy", recording_numpy)
    value = solver.measure_residual(vt.data)
    monkeypatch.setattr(wp.array, "numpy", original)
    assert max(sizes) == 1, sizes

    # measure_residual leaves A*p in ap: rebuild both host references from it.
    difference = solver.rhs[:n].numpy() - solver.ap[:n].numpy()
    old_host = float(np.linalg.norm(difference))
    exact = math.sqrt(float(np.sum(difference.astype(np.float64) ** 2)))
    assert value > 0.0
    assert value == pytest.approx(exact, rel=1e-12)
    assert value == pytest.approx(old_host, rel=1e-5)
    assert solver.measure_residual(vt.data) == value  # fixed reduction order
    assert solver.last_residual == pytest.approx(value, rel=1e-12)


def test_non_finite_rhs_still_reports_an_infinite_jacobi_residual():
    solver, vt = _solved_block("cpu", iterations=2)
    rhs = solver.rhs.numpy()
    rhs[0] = np.nan
    solver.rhs.assign(wp.array(rhs, dtype=float, device="cpu"))
    assert math.isnan(solver.measure_residual(vt.data))
    solver.solve_jacobi(vt.data, 2)
    assert solver.last_residual == math.inf
    assert solver.last_residual_squared == math.inf
