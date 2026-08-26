"""Numerical contracts for sparse two-species diffusion."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim.fields import SparseGrid
from fluidSim_warp_blender51.sim.particles import ParticleSystem
from fluidSim_warp_blender51.sim.registry import lookup
from fluidSim_warp_blender51.sim.solver import FluidSolver, FrameStats


DEVICE = "cuda:0"
KERNEL_NAMES = (
    "species_p2g_clear",
    "species_p2g_accumulate",
    "species_face_conductance_measure",
    "species_node_scale",
    "species_face_flux_apply",
    "species_g2p_delta",
    "species_particle_alpha",
    "species_delta_apply",
    "species_bounds_budget_reduce",
    "species_preview_derive",
)


def _species_api():
    try:
        return importlib.import_module("fluidSim_warp_blender51.sim.species")
    except ModuleNotFoundError:
        pytest.fail("sim.species is required for conservative species diffusion")


def _fixture(x_coordinates, concentrations, *, diffusivity=0.1, volumes=None,
             solid_coordinates=()):
    api = _species_api()
    x_coordinates = tuple(float(value) for value in x_coordinates)
    count = len(x_coordinates)
    if volumes is None:
        volumes = np.ones(count, dtype=np.float32)
    positions = np.asarray(
        [(value + 0.5, 0.5, 0.5) for value in x_coordinates],
        dtype=np.float32,
    )
    concentration = np.zeros((count, 4), dtype=np.float32)
    concentration[:, :2] = np.asarray(concentrations, dtype=np.float32)

    particles = ParticleSystem(
        max(count, 1), DEVICE,
        ("represented_volume", "species_concentration"),
    )
    wp.copy(particles.position, wp.array(positions, dtype=wp.vec3, device=DEVICE))
    wp.copy(
        particles.attr("represented_volume"),
        wp.array(np.asarray(volumes, dtype=np.float32), dtype=wp.float32, device=DEVICE),
    )
    wp.copy(
        particles.attr("species_concentration"),
        wp.array(concentration, dtype=wp.vec4, device=DEVICE),
    )
    wp.copy(
        particles.count,
        wp.array(np.asarray([count], dtype=np.int32), dtype=wp.int32, device=DEVICE),
    )
    particles.host_count = count

    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(24.0, 8.0, 8.0),
        voxel_size=1.0, leaf_capacity=3, device=DEVICE,
    )
    grid.begin_topology()
    grid.mark_points(particles.position, particles.count, count)
    grid.expand(iterations=1)
    grid.finalize_topology()

    collision_values = np.full(
        grid.voxel_capacity, 1.0e6, dtype=np.float32)
    leaf_table = grid.leaf_table.numpy()
    for x in solid_coordinates:
        x = int(x)
        leaf_x = x >> 3
        dense = leaf_x
        slot = int(leaf_table[dense])
        assert slot >= 0
        collision_values[slot * 512 + (x & 7)] = -1.0
    collision_sdf = wp.array(
        collision_values, dtype=wp.float32, device=DEVICE)

    palette = api.SpeciesPalette.create(
        ids=("a", "b"),
        absorption_rgb=((0.1, 1.0, 2.0), (2.0, 1.0, 0.1)),
        reference_path_length=1.0,
    )
    transport = api.SpeciesTransport(
        particles, grid, DEVICE, enabled=True,
        diffusivity=diffusivity, max_diffusion_substeps=16,
        palette=palette, collision_sdf=collision_sdf,
        collision_mask_enabled=bool(solid_coordinates),
    )
    return particles, grid, transport


def _amounts(particles):
    count = particles.host_count
    volume = particles.attr("represented_volume").numpy()[:count]
    concentration = particles.attr("species_concentration").numpy()[:count, :2]
    return np.sum(volume[:, None] * concentration, axis=0)


def _concentrations(particles):
    return particles.attr("species_concentration").numpy()[
        :particles.host_count
    ]


def test_mixing_off_is_an_exact_identity_and_registers_no_diffusion_work():
    particles, _grid, transport = _fixture(
        (6, 7), ((1.0, 0.0), (0.0, 1.0)), diffusivity=0.0,
    )
    before = _concentrations(particles).copy()

    transport.diffuse(1.0)

    np.testing.assert_array_equal(_concentrations(particles), before)
    assert transport.frame_diffusion_substeps == 0


def test_two_cell_exchange_is_equal_opposite_and_matches_hand_solution():
    particles, _grid, transport = _fixture(
        (6, 7), ((1.0, 0.0), (0.0, 1.0)), diffusivity=0.1,
    )
    before = _amounts(particles)

    transport.diffuse(1.0)

    result = _concentrations(particles)
    np.testing.assert_allclose(
        result[:, :2], ((0.9, 0.1), (0.1, 0.9)), rtol=0.0, atol=2.0e-6,
    )
    np.testing.assert_allclose(_amounts(particles), before, rtol=0.0, atol=2.0e-6)
    assert result[0, 2] == 0.0 and result[0, 3] == 0.0
    assert result[1, 2] == 0.0 and result[1, 3] == 0.0


def test_dense_saturated_interface_is_not_frozen_by_float_roundoff():
    count = 512
    half = count // 2
    particles, _grid, transport = _fixture(
        (6.49,) * half + (6.51,) * half,
        ((1.0, 0.0),) * half + ((0.0, 1.0),) * half,
        diffusivity=0.1,
        volumes=np.full(count, 1.25e-4, dtype=np.float32),
    )
    before = _amounts(particles)

    minimum_alpha = 1.0
    for _ in range(240):
        transport.diffuse(0.1)
        minimum_alpha = min(
            minimum_alpha, float(transport.global_alpha.numpy()[0]))

    result = _concentrations(particles)
    assert minimum_alpha > 0.0
    assert np.count_nonzero(
        (result[:, 0] > 1.0e-8) & (result[:, 1] > 1.0e-8)) > 0
    np.testing.assert_array_equal(np.sum(result[:, :2], axis=1), 1.0)
    np.testing.assert_allclose(_amounts(particles), before, rtol=0.0, atol=3.0e-6)


def test_in_domain_solid_cell_blocks_species_exchange():
    particles, _grid, transport = _fixture(
        (6, 7), ((1.0, 0.0), (0.0, 1.0)), diffusivity=0.1,
        solid_coordinates=(7,),
    )
    before = _concentrations(particles).copy()

    transport.diffuse(1.0)

    np.testing.assert_array_equal(_concentrations(particles), before)


def test_zero_collision_sdf_is_fluid_and_allows_species_exchange():
    particles, _grid, transport = _fixture(
        (6, 7), ((1.0, 0.0), (0.0, 1.0)), diffusivity=0.1,
    )
    transport.collision_sdf.fill_(0.0)
    transport.collision_mask_enabled = True

    transport.diffuse(1.0)

    np.testing.assert_allclose(
        _concentrations(particles)[:, :2],
        ((0.9, 0.1), (0.1, 0.9)), rtol=0.0, atol=2.0e-6,
    )


def test_in_domain_air_cell_blocks_species_exchange():
    particles, _grid, transport = _fixture(
        (6, 8), ((1.0, 0.0), (0.0, 1.0)), diffusivity=0.1,
    )
    before = _concentrations(particles).copy()

    transport.diffuse(1.0)

    np.testing.assert_array_equal(_concentrations(particles), before)


def test_global_limiter_preserves_conservation_and_simplex_bounds():
    particles, _grid, transport = _fixture(
        (6, 6, 7, 7),
        ((0.01, 0.89), (0.99, 0.0), (0.4, 0.1), (0.4, 0.5)),
        diffusivity=0.5,
    )
    before = _amounts(particles)

    transport.diffuse(1.0)

    result = _concentrations(particles)
    assert np.all(np.isfinite(result))
    assert np.min(result[:, :2]) >= -1.0e-7
    assert np.max(np.sum(result[:, :2], axis=1)) <= 1.0 + 1.0e-7
    np.testing.assert_allclose(_amounts(particles), before, rtol=0.0, atol=3.0e-6)


def test_positive_face_owner_applies_one_exchange_only():
    particles, _grid, transport = _fixture(
        (6, 7), ((1.0, 0.0), (0.0, 1.0)), diffusivity=0.1,
    )

    transport.diffuse(1.0)

    np.testing.assert_allclose(
        _concentrations(particles)[:, :2],
        ((0.9, 0.1), (0.1, 0.9)), rtol=0.0, atol=2.0e-6)


def test_cross_leaf_translation_matches_interior_exchange():
    interior, _grid_a, transport_a = _fixture(
        (6, 7), ((1.0, 0.0), (0.0, 1.0)), diffusivity=0.1,
    )
    seam, _grid_b, transport_b = _fixture(
        (7, 8), ((1.0, 0.0), (0.0, 1.0)), diffusivity=0.1,
    )

    transport_a.diffuse(1.0)
    transport_b.diffuse(1.0)

    np.testing.assert_allclose(
        _concentrations(seam), _concentrations(interior),
        rtol=0.0, atol=2.0e-6,
    )


def test_diffuse_does_not_synchronize_or_convert_device_arrays(monkeypatch):
    _particles, _grid, transport = _fixture(
        (6, 7), ((1.0, 0.0), (0.0, 1.0)), diffusivity=0.1,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("diffuse crossed the device/host boundary")

    monkeypatch.setattr(wp, "synchronize", forbidden)
    monkeypatch.setattr(wp, "synchronize_device", forbidden)
    monkeypatch.setattr(type(transport.global_alpha), "numpy", forbidden)

    transport.diffuse(1.0)


def test_canonical_kernel_names_resolve_after_species_module_import():
    _species_api()

    for name in KERNEL_NAMES:
        assert lookup(name).fn is not None


def _reset_fixture(*, enabled, with_transport):
    solver = object.__new__(FluidSolver)
    solver.device = DEVICE
    solver.settings = SimpleNamespace(
        species_enabled=enabled, species_ids=("a", "b"),
        species_absorption_rgb=((0.1, 1.0, 2.0), (2.0, 1.0, 0.1)),
        species_reference_path_length=1.0, species_diffusivity=0.1,
        species_max_diffusion_substeps=16, spacetime_seed=1,
        spacetime_reference_mass=0.0, spacetime_enabled=False,
    )
    channels = (("represented_volume", "species_concentration")
                if with_transport else ())
    solver.particles = ParticleSystem(2, DEVICE, channels)
    solver.grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(8.0, 8.0, 8.0),
        voxel_size=1.0, leaf_capacity=1, device=DEVICE)
    solver.collision_sdf = SimpleNamespace(data=wp.full(
        solver.grid.voxel_capacity, 1.0e6,
        dtype=wp.float32, device=DEVICE))
    solver.sourcing = SimpleNamespace(reset=lambda: None)
    solver.whitewater = SimpleNamespace(clear=lambda: None)
    solver.churn = SimpleNamespace(zero=lambda: None)
    solver.surface = SimpleNamespace(reset=lambda: None)
    solver._host_frame_cache = None
    solver.frame = 0
    solver.sim_time = 0.0
    if with_transport:
        api = _species_api()
        solver.species = api.SpeciesTransport(
            solver.particles, solver.grid, DEVICE, enabled=True,
            diffusivity=0.1, max_diffusion_substeps=16,
            palette=api.SpeciesPalette.create(
                ids=("a", "b"), absorption_rgb=(
                    (0.1, 1.0, 2.0), (2.0, 1.0, 0.1)),
                reference_path_length=1.0),
            collision_sdf=solver.collision_sdf.data,
            collision_mask_enabled=True)
    return solver


def test_reset_removes_transport_and_channels_when_feature_is_disabled():
    solver = _reset_fixture(enabled=False, with_transport=True)

    solver.reset()

    assert not hasattr(solver, "species")
    assert not solver.particles.has_optional_channel("species_concentration")


def test_reset_constructs_transport_when_feature_is_enabled():
    solver = _reset_fixture(enabled=True, with_transport=False)

    solver.reset()

    assert solver.species.particles is solver.particles
    assert solver.species.collision_sdf is solver.collision_sdf.data
    assert solver.species.collision_mask_enabled is True
    assert solver.particles.has_optional_channel("species_concentration")


def test_step_frame_orders_species_between_p2g_and_forces():
    events = []
    solver = object.__new__(FluidSolver)
    solver.settings = SimpleNamespace(
        time_scale=1.0, frame_rate=1.0, topology_expand=0, cfl=1.0,
    )
    solver.profiler = None
    solver.device = DEVICE
    solver.frame = 0
    solver.sim_time = 0.0
    solver.stats = FrameStats()
    solver.colliders = []
    solver.force_fields = []
    solver.snippets = SimpleNamespace(has_any=lambda: False, run=lambda *_args: None)
    solver.whitewater = SimpleNamespace(
        host_count=0,
        reset_frame_diagnostics=lambda: None,
        refresh_diagnostics=lambda: {
            "required": 0, "available": 0, "admitted": 0, "dropped": 0,
            "policy": "NONE", "status": "OK", "message": "",
        },
    )
    solver.surface = SimpleNamespace(
        after_substep=lambda: False, after_frame=lambda: None,
        metadata=lambda: {"voxel_size": 1.0, "active_leaves": 0,
                          "active_voxels": 0},
    )
    solver.grid = SimpleNamespace(
        active_voxel_count=0, voxel_capacity=1,
        total_voxels=lambda: 1, refresh_topology_counts=lambda: 0,
    )
    solver.particles = SimpleNamespace(
        capacity=1, refresh_host_count=lambda: 0,
    )
    solver.species = SimpleNamespace(
        diffuse=lambda _dt: events.append("species"),
        finalize_frame_receipt=lambda: None,
    )
    solver._compute_substeps = lambda _dt: (1, False)
    solver._whitewater_topology_expand = lambda _dt: 0
    solver._rebuild_topology = lambda *_args: None
    solver._clear_grid_fields = lambda **_kwargs: None
    solver._apply_collisions = lambda: None
    solver._update_moving_colliders = lambda _dt: None
    solver._index_particles = lambda: None
    solver._source = lambda _dt: False
    solver._p2g = lambda _dt: events.append("p2g")
    solver._save_velocities = lambda: None
    solver._forces = lambda _dt: events.append("forces")
    solver._viscosity = lambda _dt: events.append("viscosity")
    solver._collision_faces = lambda: None
    solver._pressure = lambda _dt: events.append("pressure")
    solver._collider_material_response = lambda: None
    solver._sanitize = lambda _dt: None
    solver._g2p_advect = lambda _dt: events.append("g2p")
    solver._sanitize_particles = lambda _dt: None
    solver._lifecycle = lambda _dt, **_kwargs: None
    solver._relax = lambda: None
    solver._whitewater = lambda *_args, **_kwargs: None
    solver._finalize_whitewater = lambda: 0
    solver.collision_impact_u = wp.zeros(1, dtype=float, device=DEVICE)
    solver.collision_impact_v = wp.zeros(1, dtype=float, device=DEVICE)
    solver.collision_impact_w = wp.zeros(1, dtype=float, device=DEVICE)
    solver.vel = SimpleNamespace(
        u=wp.zeros(1, dtype=float, device=DEVICE),
        v=wp.zeros(1, dtype=float, device=DEVICE),
        w=wp.zeros(1, dtype=float, device=DEVICE),
    )
    solver._nvox = 1

    @contextmanager
    def phase(name):
        if name in {"sourcing", "p2g", "species", "forces", "viscosity",
                    "pressure", "g2p"}:
            events.append(f"phase:{name}")
        yield

    solver._phase = phase

    solver.step_frame()

    assert events == [
        "phase:sourcing", "phase:p2g", "p2g", "phase:species", "species",
        "phase:forces", "forces", "phase:viscosity", "viscosity",
        "phase:pressure", "pressure", "phase:g2p", "g2p",
    ]
