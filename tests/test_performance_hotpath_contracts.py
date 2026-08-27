"""Small contracts preventing the measured live-playback hot paths returning."""

from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim.particles import ParticleSystem
from fluidSim_warp_blender51.sim.constants import VOXEL_COLLISION, VOXEL_FLUID
from fluidSim_warp_blender51.sim.fields import SparseGrid
from fluidSim_warp_blender51.sim.pressure import PressureSolver
from fluidSim_warp_blender51.sim.solver import (
    FluidSolver,
    SolverSettings,
    SourceShapeDesc,
    _motion_leaf_radius,
)
from fluidSim_warp_blender51.sim.sourcing import (
    SourcePipeline,
    SourceShape,
    k_source_shapes_particles,
)
from fluidSim_warp_blender51.sim.whitewater import WhitewaterSystem


def test_user_path_profile_resolves_surface_attribute_modes(monkeypatch):
    from fluidSim_warp_blender51.sim.attributes import OUTPUT_TOGGLE_NAMES

    spec = importlib.util.spec_from_file_location(
        "fluidsim_profile_user_path",
        Path(__file__).parents[1] / "tools" / "profile_user_path.py",
    )
    profile_user_path = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile_user_path)

    settings = SimpleNamespace()
    for toggle in OUTPUT_TOGGLE_NAMES["surface"].values():
        setattr(settings, toggle, True)

    assert profile_user_path.configure_surface_attribute_profile(
        settings, "off") == ()
    assert profile_user_path.configure_surface_attribute_profile(
        settings, "default") == ("velocity", "motion")
    assert profile_user_path.configure_surface_attribute_profile(
        settings, "all") == tuple(OUTPUT_TOGGLE_NAMES["surface"])
    assert profile_user_path.expected_surface_attribute_readbacks("off") == 0
    assert profile_user_path.expected_surface_attribute_readbacks("default") == 1
    assert profile_user_path.expected_surface_attribute_readbacks("all") == 9
    assert profile_user_path.expected_surface_attribute_numpy_calls("off") == 0
    assert profile_user_path.expected_surface_attribute_numpy_calls("default") == 1
    assert profile_user_path.expected_surface_attribute_numpy_calls("all") == 10

    monkeypatch.setattr(
        sys, "argv", ["profile_user_path.py", "--", "--surface-attributes", "off"])
    assert profile_user_path.parse_args().surface_attributes == "off"


def test_particle_emission_saturates_device_and_host_counts():
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    device = "cuda:0"
    particles = ParticleSystem(capacity=64, device=device)
    wp.copy(particles.count, wp.array([60], dtype=wp.int32, device=device))
    shape = SourceShapeDesc(
        enabled=True, shape=2, stype=0, mode=1,
        position=(0.4, 0.4, 0.4), half_extents=(1.0, 1.0, 1.0),
        emit_rate=100.0,
    ).to_struct()
    shapes = wp.array([shape], dtype=SourceShape, device=device)
    leaf_coords = wp.array([(0, 0, 0)], dtype=wp.vec3i, device=device)
    num_leaves = wp.array([1], dtype=wp.int32, device=device)
    voxel_counts = wp.zeros(512, dtype=wp.int32, device=device)
    emission_state = wp.array(
        [0, 1, 0, 0], dtype=wp.int32, device=device)

    wp.launch(
        k_source_shapes_particles, dim=512,
        inputs=[shapes, 1, leaf_coords, num_leaves, wp.vec3(0.0), 0.1,
                voxel_counts, 1, 0, 0, particles.capacity],
        outputs=[particles.position, particles.velocity,
                 particles.attr("age"), particles.attr("life"),
                 particles.attr("temperature"), particles.attr("viscosity"),
                 particles.attr("density"),
                 particles.optional_array("color"),
                 particles.optional_array("source_id"),
                 particles.time_offset,
                 particles.alive, particles.count, emission_state],
        device=device,
    )

    assert particles.refresh_host_count() == particles.capacity
    assert int(particles.count.numpy()[0]) == particles.capacity
    assert np.isfinite(particles.position.numpy()[:particles.capacity]).all()
    assert int(emission_state.numpy()[3]) == 1


def test_pcg_reuses_scalar_buffers_without_device_copy():
    source = inspect.getsource(PressureSolver.solve_pcg)
    batch_source = inspect.getsource(PressureSolver._launch_pcg_batch)
    assert "wp.copy(self.rz, self.rz_new)" not in source
    assert "wp.copy(self.rz, self.rz_new)" not in batch_source
    assert "self.rz, self.rz_new = self.rz_new, self.rz" in batch_source


def test_particle_compaction_scatter_publishes_count_without_extra_launch():
    """The bounded scatter must publish its final count before dead returns."""
    from fluidSim_warp_blender51.sim import particles

    scatter_source = inspect.getsource(particles.k_compact_alive_scatter)
    compact_source = inspect.getsource(ParticleSystem.compact)

    count_write = "count_out[0] = offsets[tid] + alive[tid]"
    assert count_write in scatter_source
    assert scatter_source.index("if tid == n - 1:") < scatter_source.index(
        "if alive[tid] == 0:")
    assert "wp.launch(k_last_offset_count" not in compact_source


def test_pcg_centralizes_required_full_grid_work_buffer_clears():
    source = inspect.getsource(PressureSolver.solve_pcg)
    helper = inspect.getsource(PressureSolver._zero_pcg_vectors)
    assert "for array in (x, self.r, self.z, self.p_dir, self.ap)" not in source
    assert "self.ap[:full_n].zero_()" not in source
    assert source.count("x[:full_n].zero_()") == 1
    assert source.count("self._zero_pcg_vectors(full_n, x)") == 1
    assert "for array in (x, self.r, self.z, self.p_dir, self.ap)" in helper


def test_pcg_cuda_conditional_path_does_not_preclassify_residual_on_host():
    source = inspect.getsource(PressureSolver.solve_pcg)
    conditional_anchor = '        if use_conditional:\n'
    fallback_anchor = (
        '        else:\n'
        '            self.last_pcg_execution_mode = "host_batched"\n'
    )
    conditional_source = source.split(conditional_anchor, 1)[1].split(fallback_anchor, 1)[0]
    fallback_source = source.split(fallback_anchor, 1)[1].split(
        "        # Batched device-resident iterations", 1)[0]

    assert "self.residual.numpy()" not in conditional_source
    assert "self.residual.numpy()" in fallback_source


def test_source_admission_has_one_aggregate_readback_and_no_per_source_sync():
    begin = inspect.getsource(SourcePipeline._begin_emission_transaction)
    finish = inspect.getsource(SourcePipeline._finish_emission_transaction)
    compiled = inspect.getsource(SourcePipeline.apply_compiled)

    assert "k_admit_emission_batch" in begin
    assert begin.index("if not attempted") < begin.index("wp.copy")
    assert begin.count(".numpy()") == 0
    assert compiled.count("refresh_host_count") == 0
    assert compiled.count(".numpy()") == 0
    assert finish.count("self._emission_state.numpy()") == 1


def test_live_surface_publish_cadence_defaults_to_every_frame():
    assert SolverSettings().live_surface_publish_cadence == 1


def test_viewport_replaces_full_mesh_repair_with_index_validation():
    from fluidSim_warp_blender51.blender import output_publication

    source = inspect.getsource(output_publication._swap_mesh)
    assert "mesh.validate()" not in source
    assert "surface triangle index is outside the vertex range" in source


def test_skipped_surface_publish_preserves_existing_mesh(monkeypatch):
    from fluidSim_warp_blender51 import blender
    from fluidSim_warp_blender51.blender import output_publication

    fake_materials = types.ModuleType("fluidSim_warp_blender51.blender.materials")
    fake_materials.ensure_surface_material = lambda *_args: None
    fake_materials.ensure_particle_material = lambda *_args: None
    fake_materials.ensure_whitewater_material = lambda *_args: None
    monkeypatch.setitem(sys.modules, fake_materials.__name__, fake_materials)
    monkeypatch.setattr(blender, "materials", fake_materials, raising=False)

    fake_bpy = types.ModuleType("bpy")
    monkeypatch.setattr(output_publication, "bpy", fake_bpy)
    existing = SimpleNamespace(type="MESH", data=SimpleNamespace(vertices=[1]))
    monkeypatch.setattr(
        output_publication.bpy, "data",
        SimpleNamespace(objects=SimpleNamespace(get=lambda _name: existing)),
        raising=False)
    swaps = []
    monkeypatch.setattr(output_publication, "_swap_mesh", lambda *args: swaps.append(args))
    monkeypatch.setattr(output_publication, "_clear_points", lambda _scene, _name: None)
    monkeypatch.setattr(
        output_publication, "_hide_and_clear_output",
        lambda _scene, _name: None,
    )
    scene = SimpleNamespace(fluidsim=SimpleNamespace())
    payload = SimpleNamespace(
        surface_visible=True, publish_surface=False,
        particles_visible=False, whitewater_visible=False,
        whitewater_output_mode="COMBINED",
    )

    output_publication.publish_payload(scene, payload)

    assert swaps == []


def test_whitewater_compaction_is_frame_batched_without_substep_readback():
    substep_source = inspect.getsource(FluidSolver._whitewater)
    finalize_source = inspect.getsource(FluidSolver._finalize_whitewater)
    frame_source = inspect.getsource(FluidSolver.step_frame)

    assert "synchronize_device" not in substep_source
    assert "ww.compact()" not in substep_source
    assert "ww.free_slots" in substep_source
    assert "ww.free_count" in substep_source
    emit_source = inspect.getsource(WhitewaterSystem.emit)
    assert "ww.compact(active_whitewater)" in finalize_source
    assert "dim=max(1, active_whitewater)" in finalize_source
    assert "self.candidates[:n]" in emit_source
    assert "synchronize" not in emit_source
    assert "whitewater_particles = self._finalize_whitewater()" in frame_source


def test_whitewater_finalization_uses_fixed_active_work_across_capacities(monkeypatch):
    """Compaction and density scatter must not grow with allocated capacity."""
    launches = []
    monkeypatch.setattr(
        wp, "launch",
        lambda _kernel, **kwargs: launches.append(kwargs["dim"]),
    )

    class Density:
        data = None

        def __init__(self):
            self.data = self

        def __getitem__(self, _key):
            return self

        def zero_(self):
            pass

    class Whitewater:
        def __init__(self, capacity, before_compaction, after_compaction):
            self.capacity = capacity
            self.position = object()
            self.alive = object()
            self.count = object()
            self._counts = iter((before_compaction, after_compaction))
            self.compact_hints = []
            self.type_count_hints = []

        def refresh_host_count(self):
            return next(self._counts)

        def compact(self, count_hint):
            self.compact_hints.append(count_hint)
            self.update_type_counts(count_hint)

        def update_type_counts(self, count_hint):
            self.type_count_hints.append(count_hint)

    results = []
    for capacity in (128, 131_072):
        launches.clear()
        ww = Whitewater(capacity, before_compaction=19, after_compaction=11)
        solver = SimpleNamespace(
            settings=SimpleNamespace(whitewater_enabled=True, voxel_size=0.1),
            whitewater=ww,
            grid=SimpleNamespace(
                leaf_table=object(), voxels_dim=object(), leaves_dim=object(),
                origin=object(),
            ),
            ww_density=Density(),
            _nvox=1,
            device="cuda:0",
        )

        active_count = FluidSolver._finalize_whitewater(solver)
        results.append((
            active_count, ww.compact_hints, ww.type_count_hints,
            launches.copy(),
        ))

    assert results == [
        (11, [19], [19], [11]),
        (11, [19], [19], [11]),
    ]


def test_whitewater_topology_budget_covers_ballistic_motion():
    method_source = inspect.getsource(FluidSolver._whitewater_topology_expand)
    frame_source = inspect.getsource(FluidSolver.step_frame)

    assert _motion_leaf_radius(0.0, 0.0, 9.81, 0.25) == 0
    assert _motion_leaf_radius(16.0, 0.25, 0.0, 0.25) == 3
    assert "ww.velocity" in method_source
    assert "self._whitewater_topology_expand(frame_dt)" in frame_source


def test_whitewater_emission_reuses_dead_slot_at_capacity():
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    device = "cuda:0"
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8),
        voxel_size=0.1, leaf_capacity=4, device=device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.79, 0.79, 0.79))
    assert grid.finalize_topology() == 1
    ww = WhitewaterSystem(capacity=2, device=device)
    wp.copy(ww.count, wp.array([2], dtype=wp.int32, device=device))
    wp.copy(ww.alive, wp.array([1, 0], dtype=wp.int32, device=device))
    wp.copy(ww.free_slots, wp.array([1, 0], dtype=wp.int32, device=device))
    wp.copy(ww.free_count, wp.array([1], dtype=wp.int32, device=device))
    liquid_position = wp.array([(0.35, 0.35, 0.35)], dtype=wp.vec3, device=device)
    liquid_velocity = wp.array([(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    liquid_count = wp.array([1], dtype=wp.int32, device=device)
    churn = wp.ones(grid.voxel_capacity, dtype=float, device=device)
    surface_sdf = wp.zeros(grid.voxel_capacity, dtype=float, device=device)

    ww.emit(
        candidate_count=1,
        liquid_positions=liquid_position,
        liquid_velocities=liquid_velocity,
        liquid_count=liquid_count,
        churn=churn,
        surface_sdf=surface_sdf,
        leaf_table=grid.leaf_table,
        voxels_dim=grid.voxels_dim,
        leaves_dim=grid.leaves_dim,
        origin=grid.origin,
        voxel_size=grid.voxel_size,
        dt=1.0,
        churn_threshold=0.0,
        emission_rate=100.0,
        surface_band=10.0,
        min_speed=0.0,
        foam_band=1.0,
        min_spray_speed=2.0,
        life_min=1.0,
        life_max=1.0,
        foam_life_multiplier=1.0,
        bubble_life_multiplier=1.0,
        spray_life_multiplier=1.0,
        seed=123,
    )

    assert int(ww.count.numpy()[0]) == ww.capacity
    assert int(ww.free_count.numpy()[0]) == 0
    assert int(ww.alive.numpy()[1]) == 1


def test_pressure_index_contains_only_fluid_non_collision_voxels():
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    device = "cuda:0"
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8),
        voxel_size=0.1, leaf_capacity=4, device=device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.79, 0.79, 0.79))
    assert grid.finalize_topology() == 1
    types = np.zeros(grid.voxel_capacity, dtype=np.int32)
    types[[0, 7, 511]] = VOXEL_FLUID
    types[7] |= VOXEL_COLLISION
    solver = PressureSolver(grid, device)

    count = solver._build_pressure_index(
        wp.array(types, dtype=wp.int32, device=device))

    assert count == 2
    assert solver.pressure_indices[:count].numpy().tolist() == [0, 511]

    grid.begin_topology()
    assert grid.finalize_topology() == 0
    assert solver._build_pressure_index(
        wp.array(types, dtype=wp.int32, device=device)) == 0
