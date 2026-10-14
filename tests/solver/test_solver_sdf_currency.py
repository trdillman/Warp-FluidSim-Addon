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
    solver = SimpleNamespace(surface=SimpleNamespace(
        invalidate_solver_sdf=lambda: invalidated.append(True)))

    features.run(FrameFeaturePhase.AFTER_PARTICLE_TRANSFER, solver, {})

    assert invalidated == ([True] if moves else [])
