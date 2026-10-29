"""The cached solver SDF must not outlive the particles or leaf layout it was
built from.

Regression for review item B03. The solver-grid SDF cache was keyed only on
frame, time and surface settings:
(a) a topology rebuild re-ranks leaf slots without moving time, so a
    frame-end SDF was read through the next frame's leaf table (probe: up to
    0.189 m error in surface tension);
(b) a Custom Effect writing ``particle_position`` moved particles without
    invalidating the SDF (probe: unchanged SDF after a +0.25 m shift).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

wp = pytest.importorskip("warp")


def test_solver_sdf_key_changes_when_the_leaf_layout_is_rebuilt():
    from fluidSim_warp_blender51.sim.fields import SparseGrid
    from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings
    from fluidSim_warp_blender51.sim.surface import SurfacePipeline

    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8),
                      voxel_size=0.1, leaf_capacity=1, device="cpu")
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.79, 0.79, 0.79))
    grid.finalize_topology()
    pipeline = object.__new__(SurfacePipeline)
    pipeline.solver = SimpleNamespace(
        frame=3, sim_time=0.125, grid=grid, settings=SolverSettings())
    before = pipeline.solver_sdf_build_key()

    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.79, 0.79, 0.79))
    grid.finalize_topology()

    assert pipeline.solver_sdf_build_key() != before


def _state_pipeline():
    from fluidSim_warp_blender51.sim.solver import FluidSolver
    from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings
    from fluidSim_warp_blender51.sim.surface import SurfacePipeline

    solver = object.__new__(FluidSolver)
    solver.settings = SolverSettings()
    solver.grid = SimpleNamespace(topology_generation=1)
    solver.frame, solver.sim_time = 3, 0.125
    solver.liquid_state_revision = 0
    solver._host_frame_cache = None
    surface = object.__new__(SurfacePipeline)
    surface.solver = solver
    surface.topology_generation = 0
    solver.surface = surface
    return solver, surface


def test_surface_freshness_follows_liquid_state_instead_of_clock_advances():
    solver, surface = _state_pipeline()
    sdf_key, render_key = surface.solver_sdf_build_key(), surface.build_key()
    solver.frame += 1
    solver.sim_time += 0.5
    assert surface.solver_sdf_build_key() == sdf_key
    assert surface.build_key() == render_key


def test_same_count_liquid_edit_invalidates_physical_and_render_products():
    solver, surface = _state_pipeline()
    sdf_key, render_key = surface.solver_sdf_build_key(), surface.build_key()
    surface.mesh_cache = object()
    solver.mark_liquid_state_changed()
    assert solver.liquid_state_revision == 1
    assert surface.solver_sdf_build_key() != sdf_key
    assert surface.build_key() != render_key
    assert surface.mesh_cache is None


@pytest.mark.parametrize("name, value", [
    ("surface_tension_enabled", False),
    ("surface_tension_band_voxels", 4.0),
    ("surface_tension_smoothing_iterations", 3),
])
def test_surface_support_setting_edits_change_both_product_identities(name, value):
    solver, surface = _state_pipeline()
    solver.settings.surface_tension_enabled = True
    solver.settings.surface_tension_coefficient = 1.0
    before = surface.solver_sdf_build_key(), surface.build_key()
    setattr(solver.settings, name, value)
    assert surface.solver_sdf_build_key() != before[0]
    assert surface.build_key() != before[1]


def test_partial_direct_source_write_cannot_leave_surface_products_current(monkeypatch):
    from fluidSim_warp_blender51.sim import source_execution as execution

    solver, surface = _state_pipeline()
    before = surface.solver_sdf_build_key()
    surface.mesh_cache = object()
    count = SimpleNamespace(value=0)
    particles = SimpleNamespace(refresh_host_count=lambda: 0, next_uid=0, count=count, capacity=10)
    source = object.__new__(execution.CompiledSourceExecution)
    source.runtime = SimpleNamespace(
        particles=particles, invalidate_host_cache=lambda: None,
        invalidate_particle_surface=solver.mark_liquid_state_changed)
    source.device = "cpu"
    source._emission_state = source._compiled_result = SimpleNamespace(zero_=lambda: None)
    source._compiled_empty_scalar = None
    source._point_slot_base = source._point_slot_limit = None

    def fail_after_reserve(kernel, **_kwargs):
        if kernel is execution.k_reserve_owned_point_slots:
            count.value = 1
            raise RuntimeError("partial source reserve")

    monkeypatch.setattr(execution, "fast_launch", fail_after_reserve)
    packet = SimpleNamespace(positions=[(0.0, 0.0, 0.0)])
    with pytest.raises(RuntimeError, match="partial source reserve"):
        source._direct_packet_attempt(((packet, None, None),), 1)
    assert count.value == 1
    assert solver.liquid_state_revision == 1
    assert surface.solver_sdf_build_key() != before
    assert surface.mesh_cache is None


@pytest.mark.parametrize("raises", [False, True])
def test_position_effects_record_mutation_even_after_partial_failure(raises):
    from fluidSim_warp_blender51.sim.effects.runtime import EffectRuntimeRequirements
    from fluidSim_warp_blender51.sim.frame_features import FrameFeaturePhase, FrameFeatures

    solver, surface = _state_pipeline()
    before = surface.solver_sdf_build_key()
    features = object.__new__(FrameFeatures)

    def run(*_args):
        if raises:
            raise RuntimeError("partial position write")

    features.effects = SimpleNamespace(
        requirements_for_stage=lambda _: EffectRuntimeRequirements(particles=True, moves_particles=True),
        run=run,
    )
    if raises:
        with pytest.raises(RuntimeError, match="partial position write"):
            features.run(FrameFeaturePhase.AFTER_PARTICLE_UPDATE, solver, {})
    else:
        features.run(FrameFeaturePhase.AFTER_PARTICLE_UPDATE, solver, {})
    assert solver.liquid_state_revision == 1
    assert surface.solver_sdf_build_key() != before


def _entry(access):
    from fluidSim_warp_blender51.sim.effects.enums import EffectDomain
    from fluidSim_warp_blender51.sim.effects.runtime import EffectRuntimeEntry
    from fluidSim_warp_blender51.sim.effects.schema import FieldBindingV2

    kernel = SimpleNamespace(
        domain=EffectDomain.PARTICLES,
        fields=(FieldBindingV2("particle_position", access),),
        prerequisites=(), kernel_id="k")
    return EffectRuntimeEntry(
        instance=SimpleNamespace(instance_id="i"), compiled_effect=None,
        compiled_kernel=SimpleNamespace(kernel_definition=kernel),
        order=0, kernel_index=0)


def test_a_position_writing_binding_marks_the_stage_as_moving_particles():
    from fluidSim_warp_blender51.sim.effects.enums import FieldAccess
    from fluidSim_warp_blender51.sim.effects.runtime import _requirements_for_entry

    assert _requirements_for_entry(_entry(FieldAccess.WRITE_CURRENT)).moves_particles
    assert _requirements_for_entry(_entry(FieldAccess.READ_WRITE_CURRENT)).moves_particles
    assert not _requirements_for_entry(_entry(FieldAccess.READ)).moves_particles


@pytest.mark.parametrize("moves", [True, False])
def test_a_stage_that_moves_particles_invalidates_the_solver_sdf(moves):
    from fluidSim_warp_blender51.sim.effects.runtime import EffectRuntimeRequirements
    from fluidSim_warp_blender51.sim.frame_features import (
        FrameFeaturePhase, FrameFeatures,
    )

    invalidated = []
    features = object.__new__(FrameFeatures)
    features.effects = SimpleNamespace(
        requirements_for_stage=lambda _stage: EffectRuntimeRequirements(
            particles=True, moves_particles=moves),
        run=lambda *_args: None,
        runtime_errors=[],
    )
    solver = SimpleNamespace(mark_liquid_state_changed=lambda: invalidated.append(True))

    features.run(FrameFeaturePhase.AFTER_PARTICLE_TRANSFER, solver, {})

    assert invalidated == ([True] if moves else [])
