"""Checkpoint export slices on the device; payloads stay byte-identical (P01).

The oracle is the previous implementation, ``array.numpy()[:count]``: a
full-capacity readback sliced on the host. Patching it back in and exporting
the same solver state again must give identical arrays, and identical NPZ
members once written through the checkpoint writer.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
from _narrow_band_harness import run_tank

from fluidSim_warp_blender51.io.checkpoint import (
    capture_solver, write_checkpoint_atomic,
)
from fluidSim_warp_blender51.sim import narrow_band_state, particles, solver as solver_module
from fluidSim_warp_blender51.sim.checkpoint_state import host_prefix
from fluidSim_warp_blender51.sim.enums import ParticleSampleType
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

wp = pytest.importorskip("warp")


def _full_readback_prefix(array, count):
    return array.numpy()[:count]


_PATCHED_MODULES = (solver_module, particles, narrow_band_state)


def _with_full_readback(monkeypatch, fn):
    for module in _PATCHED_MODULES:
        monkeypatch.setattr(module, "host_prefix", _full_readback_prefix)
    try:
        return fn()
    finally:
        for module in _PATCHED_MODULES:
            monkeypatch.setattr(module, "host_prefix", host_prefix)


def _assert_identical(expected, actual, path="state"):
    if isinstance(expected, dict):
        assert set(expected) == set(actual), path
        for key in expected:
            _assert_identical(expected[key], actual[key], f"{path}[{key!r}]")
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype, path
        assert actual.shape == expected.shape, path
        assert actual.tobytes() == expected.tobytes(), path
    else:
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def _npz_members(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in sorted(archive.namelist())}


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "dtype", [float, wp.int32, wp.uint32, wp.vec3, wp.vec4, wp.mat33])
@pytest.mark.parametrize("count", [0, 1, 5, 16, 40])
def test_host_prefix_matches_full_readback(device_kind, dtype, count, request):
    device = (request.getfixturevalue("cuda_device")
              if device_kind == "cuda" else "cpu")
    rng = np.random.default_rng(7)
    array = wp.zeros(16, dtype=dtype, device=device)
    host = rng.standard_normal(array.numpy().shape).astype(array.numpy().dtype)
    array.assign(host)
    expected = array.numpy()[:count]
    actual = host_prefix(array, count)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def _sparse_solver(device):
    # Generous capacity, small live set: the case the full readback wasted.
    solver = FluidSolver(SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.0625, max_particles=65536, max_whitewater=8192,
        surface_enabled=False, whitewater_enabled=False,
        sample_type=int(ParticleSampleType.APIC), species_enabled=True,
        device=device))
    axes = [0.3 + 0.03125 * np.arange(n) for n in (8, 8, 8)]
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)
    solver.inject_particles(points, velocity=(0.25, 0.0, 0.0))
    for _ in range(2):
        solver.step_frame()
    return solver


def test_export_checkpoint_state_is_byte_identical_to_full_readback(
        cuda_device, monkeypatch, tmp_path):
    solver = _sparse_solver(cuda_device)
    try:
        assert solver.particles.refresh_host_count() < solver.particles.capacity
        assert solver.whitewater.refresh_host_count() == 0
        sliced = solver.export_checkpoint_state()
        full = _with_full_readback(monkeypatch, solver.export_checkpoint_state)
        _assert_identical(full, sliced)
        assert sliced["arrays"]["particle_affine"].shape[0] > 0
        assert "particle_attr_color" in sliced["arrays"]

        new_path = tmp_path / "sliced.npz"
        old_path = tmp_path / "full.npz"
        write_checkpoint_atomic(new_path, capture_solver(
            solver, settings_signature="s", source_signature="q"))
        write_checkpoint_atomic(old_path, _with_full_readback(
            monkeypatch, lambda: capture_solver(
                solver, settings_signature="s", source_signature="q")))
        assert _npz_members(new_path) == _npz_members(old_path)
    finally:
        solver.release()


def test_narrow_band_checkpoint_and_leaf_phi_are_byte_identical(
        cuda_device, monkeypatch):
    solver = run_tank(cuda_device, narrow_band=True, frames=3)
    try:
        sliced = solver.export_checkpoint_state()
        full = _with_full_readback(monkeypatch, solver.export_checkpoint_state)
        _assert_identical(full, sliced)
        coords, phi = solver.narrow_band.leaf_phi()
        ref_coords, ref_phi = _with_full_readback(
            monkeypatch, solver.narrow_band.leaf_phi)
        _assert_identical(
            {"coords": ref_coords, "phi": ref_phi},
            {"coords": coords, "phi": phi})
        assert phi.shape[0] == coords.shape[0] > 0
    finally:
        solver.release()
