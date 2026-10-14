"""B39: adaptive substep scheduling counts sources that are about to emit.

``_compute_substeps`` runs before emission. It measured only existing
particles and colliders, so an empty scene with an 8 m/s jet picked one
substep; the grid CFL clamp (``cfl * h / dt``) then cut the jet to 2.4 m/s on
its first frame, while a fixed 8-substep run kept 8 m/s.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.enums import (  # noqa: E402
    SourceEmissionMode,
    SourceShapeShape,
    SourceShapeSourceMode,
    SourceShapeType,
)
from fluidSim_warp_blender51.sim.frame_runtime import (  # noqa: E402
    SubstepPolicy,
    select_substeps,
)
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings  # noqa: E402
from fluidSim_warp_blender51.sim.sourcing import SourceShapeDesc  # noqa: E402

JET = 8.0


def _jet_solver(device, emission_mode=SourceEmissionMode.INITIAL_FILL):
    s = SolverSettings()
    s.domain_min = (-2.0, -1.0, 0.0)
    s.domain_max = (2.0, 1.0, 2.0)
    s.voxel_size = 0.1
    s.max_particles = 20_000
    s.whitewater_enabled = False
    s.surface_enabled = False
    s.min_substeps = 1
    s.max_substeps = 8
    s.device = device
    solver = FluidSolver(s)
    solver.set_source_batch([SourceShapeDesc(
        enabled=True, shape=SourceShapeShape.BOX, stype=SourceShapeType.SOURCE,
        mode=SourceShapeSourceMode.PARTICLES, position=(-1.6, 0.0, 1.0),
        rotation=(0.0, 0.0, 0.0, 1.0), half_extents=(0.15, 0.15, 0.15),
        radius=0.15, height=0.15, emit_rate=6.0, emission_mode=emission_mode,
        source_id="jet", velocity=(JET, 0.0, 0.0))])
    return solver


def test_first_frame_jet_keeps_its_speed(cuda_device):
    solver = _jet_solver(cuda_device)
    try:
        st = solver.step_frame()
        n = solver.particles.refresh_host_count()
        assert n > 0
        vx = solver.particles.velocity.numpy()[:n, 0]
        assert st.substeps > 1
        assert vx.mean() > 0.95 * JET, f"jet slowed to {vx.mean():.2f} m/s"
        # once fired, an initial-fill source no longer drives scheduling
        assert solver._armed_source_max_speed() == 0.0
    finally:
        solver.release()


def test_schedule_without_particles_or_sources_is_unchanged():
    class _S:
        gravity = (0.0, 0.0, -9.81)
        min_substeps = 1
        max_substeps = 8
        cfl = 1.0
        voxel_size = 0.1

    policy = SubstepPolicy.from_settings(_S())
    frame_dt = 1.0 / 24.0
    assert select_substeps(policy, frame_dt, has_particles=False).substeps == 1
    # a still source adds nothing; a moving one schedules like fluid would
    assert select_substeps(policy, frame_dt, has_particles=False,
                           source_max_speed=0.0).substeps == 1
    assert (select_substeps(policy, frame_dt, has_particles=False,
                            source_max_speed=JET).substeps
            == select_substeps(policy, frame_dt, has_particles=True,
                               fluid_max_speed=JET).substeps)
