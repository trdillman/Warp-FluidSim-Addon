"""Regression contracts for the force settings and dispatch seam."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from fluidSim_warp_blender51.sim import solver as solver_mod
from fluidSim_warp_blender51.sim.kernels_fluid import (
    ForceFieldProxy,
    force_field_proxy_acceleration,
)
from fluidSim_warp_blender51.sim.solver import (
    FluidSolver,
    ForceFieldDesc,
    SolverSettings,
)


@wp.kernel
def _evaluate_force_proxy(
    fields: wp.array(dtype=ForceFieldProxy),
    point: wp.vec3,
    velocity: wp.vec3,
    time: float,
    result: wp.array(dtype=wp.vec3),
):
    result[0] = force_field_proxy_acceleration(
        fields[0], point, velocity, time)


def _force_harness():
    class Particles:
        count = "particle_count"

        @staticmethod
        def attr(name):
            return f"particle_attr:{name}"

    settings = SimpleNamespace(
        gravity=(0.0, 0.0, -9.81),
        directional_force=(1.0, 2.0, 3.0),
        omnidirectional_force_center=(4.0, 5.0, 6.0),
        omnidirectional_force_strength=-7.0,
        omnidirectional_force_radius=8.0,
        force_diffusion=0.25,
        dissipation_enabled=True,
        dissipation_attribute="viscosity",
        dissipation_rate=0.5,
        dissipation_target=0.75,
        attraction_enabled=True,
        attraction_strength=9.0,
        attraction_max_distance=10.0,
        max_speed=11.0,
        speed_limit_behavior=0,
        speed_limit_percent=0.1,
    )
    source_force_calls = []
    surface_tension_calls = []
    return SimpleNamespace(
        settings=settings,
        grid=SimpleNamespace(
            num_leaves="num_leaves", leaf_coords="leaf_coords",
            leaf_table="leaf_table", voxels_dim="voxels_dim",
            leaves_dim="leaves_dim", origin="origin", voxel_size=0.125,
        ),
        vel=SimpleNamespace(
            u="u", v="v", w="w", wu="wu", wv="wv", ww="ww"),
        tmp_u="tmp_u", tmp_v="tmp_v", tmp_w="tmp_w",
        particles=Particles(),
        collision_sdf=SimpleNamespace(data="collision_sdf"),
        force_fields=[object()],
        sourcing=SimpleNamespace(
            apply_forces=lambda dt: source_force_calls.append(dt)),
        source_force_calls=source_force_calls,
        surface_tension=SimpleNamespace(
            apply=lambda dt: surface_tension_calls.append(dt)),
        surface_tension_calls=surface_tension_calls,
        _force_field_data=lambda: "force_field_array",
        _nvox=512,
        _np_dim=lambda: 13,
        device="fake-device",
        sim_time=1.25,
    )


def test_force_stage_dispatches_all_enabled_global_force_kernels(monkeypatch):
    launches = []
    monkeypatch.setattr(
        solver_mod.wp, "launch",
        lambda kernel, **kwargs: launches.append((kernel, kwargs)),
    )
    solver = _force_harness()

    FluidSolver._forces(solver, 0.125)

    assert [kernel for kernel, _kwargs in launches] == [
        solver_mod.k_forces_gravity,
        solver_mod.k_forces_directional,
        solver_mod.k_forces_omnidirectional,
        solver_mod.k_forces_proxy_fields,
        solver_mod.k_forces_diffusion,
        solver_mod.k_dissipate_particle_attribute,
        solver_mod.k_attract_sdf_gradient_direction,
        solver_mod.k_forces_speed_limit,
    ]
    assert solver.source_force_calls == [0.125]
    assert solver.surface_tension_calls == [0.125]
    assert launches[3][1]["inputs"][:2] == ["force_field_array", 1]
    assert launches[5][1]["inputs"][0] == "particle_attr:viscosity"
    assert launches[6][1]["inputs"][0] == "collision_sdf"
    assert launches[-1][1]["inputs"][0:3] == ["tmp_u", "tmp_v", "tmp_w"]
    # k_forces_speed_limit reads +x/+y/+z neighbor faces while scaling, so it
    # must write to separate output buffers (like k_forces_diffusion above it)
    # rather than in place, or concurrent threads race on partially-scaled
    # neighbor reads. Assert the double-buffer swap explicitly: the diffusion
    # step already swapped vel<->tmp once (hence "inputs" above is tmp_*), and
    # speed_limit's own output+swap cancels that, returning both pairs to
    # their original identity.
    assert launches[-1][1]["outputs"] == ["u", "v", "w"]
    assert (solver.vel.u, solver.vel.v, solver.vel.w) == ("u", "v", "w")
    assert (solver.tmp_u, solver.tmp_v, solver.tmp_w) == ("tmp_u", "tmp_v", "tmp_w")


def _evaluate(desc, point=(1.0, 0.2, 0.3), velocity=(0.4, 0.5, 0.6)):
    fields = wp.array([desc.to_struct()], dtype=ForceFieldProxy, device="cpu")
    result = wp.zeros(1, dtype=wp.vec3, device="cpu")
    wp.launch(
        _evaluate_force_proxy, dim=1,
        inputs=[fields, wp.vec3(*point), wp.vec3(*velocity), 0.75],
        outputs=[result], device="cpu",
    )
    return result.numpy()[0]


def test_all_native_blender_force_proxy_branches_produce_finite_acceleration():
    shared = dict(direction=(0.0, 0.0, 1.0), strength=2.0, size=0.5)
    descriptors = [
        ForceFieldDesc(force_type=1, **shared),
        ForceFieldDesc(force_type=2, **shared),
        ForceFieldDesc(force_type=3, **shared),
        ForceFieldDesc(force_type=4, **shared),
        ForceFieldDesc(force_type=5, rest_length=0.5, damping=0.2, **shared),
        ForceFieldDesc(force_type=6, **shared),
        ForceFieldDesc(force_type=7, **shared),
        ForceFieldDesc(force_type=8, texture_mode=2, seed=3, **shared),
        ForceFieldDesc(
            force_type=9, guide_points=((0.0, 0.0, 0.0),
                                        (1.0, 0.0, 0.0),
                                        (2.0, 0.5, 0.0)),
            guide_minimum=0.5, guide_clump=0.2, **shared),
        ForceFieldDesc(force_type=10, **shared),
        ForceFieldDesc(force_type=11, seed=7, **shared),
        ForceFieldDesc(
            force_type=12, linear_drag=0.4, quadratic_drag=0.2, **shared),
        ForceFieldDesc(
            force_type=13, source_velocity=(1.0, 0.0, 0.0), flow=0.5,
            **shared),
    ]

    accelerations = np.asarray([_evaluate(desc) for desc in descriptors])

    assert np.isfinite(accelerations).all()
    assert np.linalg.norm(accelerations, axis=1).min() > 1.0e-6


def test_drag_proxy_uses_blender_linear_and_quadratic_drag_properties():
    acceleration = _evaluate(
        ForceFieldDesc(force_type=12, linear_drag=0.5, quadratic_drag=0.25),
        point=(1.0, 0.0, 0.0), velocity=(2.0, 0.0, 0.0))

    assert np.allclose(acceleration, (-2.0, 0.0, 0.0), atol=1.0e-6)


def test_force_defaults_are_inert_and_preserve_valid_attribute_selection():
    settings = SolverSettings()

    assert settings.directional_force == (0.0, 0.0, 0.0)
    assert settings.omnidirectional_force_strength == 0.0
    assert settings.force_diffusion == 0.0
    assert settings.dissipation_enabled is False
    assert settings.dissipation_attribute == "temperature"
    assert settings.attraction_enabled is False


def test_force_panel_exposes_spin_axis_and_global_attraction_controls():
    root = Path(__file__).parents[1] / "fluidSim_warp_blender51" / "blender"
    scene_source = (root / "ui_scene.py").read_text(encoding="utf-8")
    object_source = (root / "ui_object.py").read_text(encoding="utf-8")

    assert 'layout.prop(fs, "attraction_enabled")' in scene_source
    assert 'layout.prop(fs, "force_diffusion")' in scene_source
    assert 'settings.force_type in {"DIRECTIONAL", "SPIN"}' in object_source
    assert 'layout.prop(settings, "force_direction")' in object_source
