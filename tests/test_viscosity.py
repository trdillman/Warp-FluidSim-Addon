"""Viscosity solve stability validation (test 11)."""
import sys
from types import SimpleNamespace

import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.constants import LEAF_VOLUME
from fluidSim_warp_blender51.sim import viscosity
from fluidSim_warp_blender51.sim.viscosity import ViscositySolver

from _subset_runner import run_subset


def _fake_viscosity_solver(active_voxels=LEAF_VOLUME, capacity=LEAF_VOLUME * 8):
    solver = object.__new__(ViscositySolver)
    solver.grid = SimpleNamespace(
        active_voxel_count=active_voxels,
        voxel_capacity=capacity,
        leaf_table=object(),
        leaf_coords=object(),
        voxels_dim=object(),
        leaves_dim=object(),
        num_leaves=object(),
        voxel_size=0.1,
    )
    solver.device = "fake-device"
    for name in (
        "rhs", "x", "inv_diag", "r", "z", "p_dir", "ap", "tmp_u", "tmp_v",
        "tmp_w", "counter", "rz", "rz_new", "p_ap", "alpha", "beta",
        "residual",
    ):
        setattr(solver, name, object())
    return solver


def test_viscosity_smooth_scales_with_active_voxel_prefix(monkeypatch):
    solver = _fake_viscosity_solver()
    vel = SimpleNamespace(
        u=object(), v=object(), w=object(), wu=object(), wv=object(), ww=object()
    )
    launch_dims = []
    copy_counts = []

    monkeypatch.setattr(
        viscosity.wp,
        "launch",
        lambda _kernel, dim, inputs, device: launch_dims.append(dim),
    )
    monkeypatch.setattr(
        viscosity.wp,
        "copy",
        lambda _dst, _src, count=None: copy_counts.append(count),
    )

    solver.smooth(vel, object(), object(), dt=0.01, iterations=1)

    assert launch_dims == [solver.grid.active_voxel_count]
    assert copy_counts == [solver.grid.active_voxel_count] * 3


def test_viscosity_pcg_uses_active_prefix_for_launches_reductions_and_copy(monkeypatch):
    solver = _fake_viscosity_solver()
    launch_dims = []
    reduction_counts = []
    copy_counts = []

    monkeypatch.setattr(
        viscosity.wp,
        "launch",
        lambda _kernel, dim, inputs, device: launch_dims.append(dim),
    )
    monkeypatch.setattr(
        viscosity.wp,
        "copy",
        lambda _dst, _src, count=None: copy_counts.append(count),
    )
    monkeypatch.setattr(
        viscosity.wp.utils,
        "array_inner",
        lambda _a, _b, out, count=None: reduction_counts.append(count),
    )

    solver._solve_component(
        comp=object(),
        weight=object(),
        viscosity_field=object(),
        voxel_type=object(),
        dt=0.01,
        use_pcg=True,
        iterations=0,
        tolerance=1.0e-4,
    )

    active = solver.grid.active_voxel_count
    assert solver.grid.voxel_capacity not in launch_dims
    assert set(launch_dims) == {1, active}
    assert reduction_counts == [active]
    assert copy_counts == [active]


if __name__ == "__main__":
    sys.exit(run_subset([10], "VISCOSITY"))
