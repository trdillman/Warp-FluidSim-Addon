"""The implicit viscosity operator must be symmetric for PCG to be valid.

Review finding, 2026-10-26: each row used only its own cell's coefficient
(A[i,j] = -c_i, A[j,i] = -c_j), so with spatially varying viscosity
(viscosity_from_particles, or any authored field) the matrix was
non-symmetric and conjugate gradients solved the wrong system.  Runs the real
Warp kernels on the CPU device over one 8^3 leaf.
"""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.constants import LEAF_VOLUME  # noqa: E402
from fluidSim_warp_blender51.sim import viscosity  # noqa: E402

DEVICE = "cpu"
VOXEL_SIZE = 0.1
DT = 0.02


def _grid(nu):
    arrays = dict(
        viscosity_field=wp.array(nu.astype(np.float32), dtype=float, device=DEVICE),
        weight=wp.array(np.ones(LEAF_VOLUME, np.float32), dtype=float, device=DEVICE),
        voxel_type=wp.zeros(LEAF_VOLUME, dtype=wp.int32, device=DEVICE),
        leaf_table=wp.array(np.zeros(1, np.int32), dtype=wp.int32, device=DEVICE),
        leaf_coords=wp.array(np.zeros((1, 3), np.int32), dtype=wp.vec3i, device=DEVICE),
        voxels_dim=wp.vec3i(8, 8, 8),
        leaves_dim=wp.vec3i(1, 1, 1),
        num_leaves=wp.array(np.ones(1, np.int32), dtype=wp.int32, device=DEVICE),
        voxel_size=VOXEL_SIZE,
        dt=wp.array(np.array([DT], np.float32), dtype=float, device=DEVICE),
    )
    return arrays


def _matrix(grid):
    """Dense A from k_viscosity_apply_matrix applied to every unit vector."""
    out = wp.zeros(LEAF_VOLUME, dtype=float, device=DEVICE)
    columns = []
    for j in range(LEAF_VOLUME):
        unit = np.zeros(LEAF_VOLUME, np.float32)
        unit[j] = 1.0
        x = wp.array(unit, dtype=float, device=DEVICE)
        wp.launch(
            viscosity.k_viscosity_apply_matrix, dim=LEAF_VOLUME, device=DEVICE,
            inputs=[x, grid["viscosity_field"], grid["weight"], grid["voxel_type"],
                    grid["leaf_table"], grid["leaf_coords"], grid["voxels_dim"],
                    grid["leaves_dim"], grid["num_leaves"], grid["voxel_size"],
                    grid["dt"], out])
        columns.append(out.numpy().copy())
    return np.stack(columns, axis=1)


def _inverse_diagonal(grid):
    inv_diag = wp.zeros(LEAF_VOLUME, dtype=float, device=DEVICE)
    wp.launch(
        viscosity.k_viscosity_compute_inverse_diagonal, dim=LEAF_VOLUME, device=DEVICE,
        inputs=[grid["viscosity_field"], grid["weight"], grid["voxel_type"],
                grid["leaf_table"], grid["leaf_coords"], grid["voxels_dim"],
                grid["leaves_dim"], grid["num_leaves"], grid["voxel_size"],
                grid["dt"], inv_diag])
    return inv_diag.numpy()


@pytest.fixture(scope="module")
def varying():
    rng = np.random.default_rng(7)
    grid = _grid(rng.uniform(0.001, 0.5, LEAF_VOLUME))
    return grid, _matrix(grid)


def test_operator_is_symmetric_with_varying_viscosity(varying):
    _grid_arrays, matrix = varying
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-5, atol=1e-7)


def test_operator_is_positive_definite_with_varying_viscosity(varying):
    _grid_arrays, matrix = varying
    assert np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min() > 0.0


def test_jacobi_preconditioner_matches_the_operator_diagonal(varying):
    grid, matrix = varying
    np.testing.assert_allclose(_inverse_diagonal(grid), 1.0 / np.diag(matrix), rtol=1e-6)


def test_uniform_viscosity_operator_is_unchanged():
    """Uniform fields (every shipped demo) keep the original coefficients."""
    nu = 0.05
    grid = _grid(np.full(LEAF_VOLUME, nu))
    matrix = _matrix(grid)
    c = np.float32(np.float32(nu) * np.float32(DT) / np.float32(VOXEL_SIZE * VOXEL_SIZE))
    corner, interior = 0, 1 + 8 + 64  # voxel (0,0,0) has 3 neighbours, (1,1,1) has 6
    assert matrix[corner, corner] == pytest.approx(1.0 + 3 * c, rel=1e-6)
    assert matrix[interior, interior] == pytest.approx(1.0 + 6 * c, rel=1e-6)
    assert matrix[interior, interior + 1] == pytest.approx(-c, rel=1e-6)
