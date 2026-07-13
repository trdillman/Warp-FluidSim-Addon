"""Regression tests for the realtime liquid surfacing filter stack."""

from __future__ import annotations

from types import SimpleNamespace


def _fake_solver(solver_mod, settings):
    solver = object.__new__(solver_mod.FluidSolver)
    solver.settings = settings
    solver.device = "cuda:test"
    solver._nvox = 11
    solver._np_dim = lambda: 7
    solver.grid = SimpleNamespace(
        leaf_table="leaf_table",
        leaf_coords="leaf_coords",
        voxels_dim=(8, 8, 8),
        leaves_dim=(1, 1, 1),
        num_leaves=1,
        origin=(0.0, 0.0, 0.0),
    )
    solver.particles = SimpleNamespace(position="position", count=5)
    from fluidSim_warp_blender51.sim.surface import SurfacePipeline
    pipeline = object.__new__(SurfacePipeline)
    pipeline.solver = solver
    pipeline.sdf = SimpleNamespace(data="sdf")
    pipeline.tmp = SimpleNamespace(data="tmp")
    return pipeline


def test_surface_filters_stack_in_fixed_houdini_style_order(monkeypatch):
    import fluidSim_warp_blender51.sim.solver as solver_mod

    launches = []
    import fluidSim_warp_blender51.sim.surface as surface_mod
    monkeypatch.setattr(
        surface_mod.wp,
        "launch",
        lambda kernel, **kwargs: launches.append((kernel, kwargs)),
    )
    settings = solver_mod.SolverSettings()
    settings.surface_dilate_iterations = 1
    settings.surface_dilate_strength = 0.2
    settings.surface_erode_iterations = 2
    settings.surface_erode_strength = 0.3
    settings.surface_smooth_type = int(solver_mod.SdfFilterType.MEAN_CURVATURE_FLOW)
    settings.surface_smooth_iterations = 3
    settings.surface_smooth_strength = 0.4
    pipeline = _fake_solver(solver_mod, settings)

    pipeline.build(build_render=False)

    assert [call[0] for call in launches] == [
        solver_mod.k_surface_sdf_reset,
        solver_mod.k_build_surface_sdf,
        solver_mod.k_smooth_surface_sdf_dilate,
        solver_mod.k_smooth_surface_sdf_erode,
        solver_mod.k_smooth_surface_sdf_erode,
        solver_mod.k_smooth_surface_sdf_mean_curvature,
        solver_mod.k_smooth_surface_sdf_mean_curvature,
        solver_mod.k_smooth_surface_sdf_mean_curvature,
    ]
    strengths = [call[1]["inputs"][-1] for call in launches[2:]]
    assert strengths == [0.2, 0.3, 0.3, 0.4, 0.4, 0.4]


def test_surface_stack_defaults_preserve_legacy_box_smoothing(monkeypatch):
    import fluidSim_warp_blender51.sim.solver as solver_mod

    launches = []
    import fluidSim_warp_blender51.sim.surface as surface_mod
    monkeypatch.setattr(
        surface_mod.wp,
        "launch",
        lambda kernel, **kwargs: launches.append((kernel, kwargs)),
    )
    pipeline = _fake_solver(solver_mod, solver_mod.SolverSettings())

    pipeline.build(build_render=False)

    assert [call[0] for call in launches] == [
        solver_mod.k_surface_sdf_reset,
        solver_mod.k_build_surface_sdf,
        solver_mod.k_smooth_surface_sdf_box,
    ]
    assert launches[-1][1]["inputs"][-1] == 0.7
