"""Built-in particle-velocity effects must reach the solve under every transfer.

Regression for review item B68: built-ins wrote particle velocity at
AFTER_PARTICLE_TRANSFER, after particle-to-grid transfer, where PIC (and APIC)
grid-to-particle transfer replaces the particle velocity with the grid value,
so Velocity Damping and every force built-in did nothing under PIC.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.effects.builtins import builtin_effect_definitions
from fluidSim_warp_blender51.sim.effects.compiler import compile_effect
from fluidSim_warp_blender51.sim.effects.enums import EffectStage
from fluidSim_warp_blender51.sim.effects.schema import EffectInstanceV2
from fluidSim_warp_blender51.sim.enums import ParticleSampleType
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

pytest.importorskip("warp")

FRAMES = 10
SPEED = 1.0


def _damping(stage=None):
    definition = next(item for item in builtin_effect_definitions()
                      if item.effect_id == "fluidsim.velocity_damping")
    if stage is not None:
        kernels = tuple(replace(kernel, stage=stage) for kernel in definition.kernels)
        definition = replace(definition, kernels=kernels)
    return EffectInstanceV2("damping-1", definition)


def _mean_speed_after(device, instance, sample_type):
    solver = FluidSolver(SolverSettings(
        domain_min=(-1.0, -1.0, -0.5), domain_max=(1.5, 1.0, 0.5),
        voxel_size=0.0625, leaf_capacity=256, max_particles=8192,
        max_whitewater=64, surface_enabled=False, whitewater_enabled=False,
        gravity=(0.0, 0.0, 0.0), relax_enabled=False, open_boundaries=True,
        domain_sdf_walls=False, sample_type=int(sample_type), device=device,
    ))
    try:
        axis = np.arange(-0.25, 0.25, 0.03125)
        thin = np.arange(-0.125, 0.125, 0.03125)
        positions = np.stack(np.meshgrid(axis, axis, thin, indexing="ij"),
                             axis=-1).reshape(-1, 3).astype(np.float32)
        velocities = np.zeros_like(positions)
        velocities[:, 0] = SPEED
        solver.inject_particles(positions, velocities=velocities)
        if instance is not None:
            compiled = compile_effect(instance, device=device)
            assert compiled.success, compiled.issues
            assert solver.set_custom_effect_candidate(
                SimpleNamespace(success=True, compiled_effects=(compiled,)))
        for _ in range(FRAMES):
            solver.step_frame()
        return float(np.mean(solver.particle_velocities()[:, 0]))
    finally:
        solver.release()


def test_builtin_damping_slows_liquid_under_pic(cuda_device):
    undamped = _mean_speed_after(cuda_device, None, ParticleSampleType.PIC)
    damped = _mean_speed_after(cuda_device, _damping(), ParticleSampleType.PIC)
    assert undamped == pytest.approx(SPEED, rel=0.05)
    # 1/s damping for 10 frames at 24 fps keeps about exp(-10/24) = 0.66.
    assert damped < 0.8 * undamped, (undamped, damped)


def test_velocity_written_after_transfer_is_discarded_by_pic(cuda_device):
    """The mechanism the AFTER_PARTICLE_TRANSFER warning describes."""
    undamped = _mean_speed_after(cuda_device, None, ParticleSampleType.PIC)
    late = _mean_speed_after(
        cuda_device, _damping(EffectStage.AFTER_PARTICLE_TRANSFER), ParticleSampleType.PIC)
    assert late == pytest.approx(undamped, rel=0.01)
