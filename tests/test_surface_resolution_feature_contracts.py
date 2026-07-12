"""Contracts for decoupled render-surface resolution."""

from pathlib import Path
import inspect

from fluidSim_warp_blender51.sim.enums import (
    SurfaceResolutionMode,
    SurfaceUpdateCadence,
)
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings
from fluidSim_warp_blender51.sim.surface import SurfacePipeline


def test_surface_settings_preserve_fast_default_with_frame_cadence():
    settings = SolverSettings()

    assert settings.surface_resolution_mode == SurfaceResolutionMode.SIMULATION
    assert settings.surface_resolution_multiplier == 2.0
    assert settings.surface_update_cadence == SurfaceUpdateCadence.FRAME
    assert settings.surface_build_lowres_sdf is True
    assert settings.surface_max_dense_dim == 512


def test_render_surface_voxel_size_is_decoupled_from_simulation_grid():
    settings = SolverSettings(voxel_size=0.1)

    assert SurfacePipeline.voxel_size_for(settings) == 0.1

    settings.surface_resolution_mode = SurfaceResolutionMode.MULTIPLIER
    assert SurfacePipeline.voxel_size_for(settings) == 0.05

    settings.surface_resolution_mode = SurfaceResolutionMode.CUSTOM
    settings.surface_voxel_size = 0.025
    assert SurfacePipeline.voxel_size_for(settings) == 0.025


def test_surface_panel_exposes_resolution_and_cadence_controls():
    source = (Path(__file__).parents[1] / "fluidSim_warp_blender51" / "panels.py").read_text()

    for control in (
        "surface_resolution_mode",
        "surface_resolution_multiplier",
        "surface_voxel_size",
        "surface_leaf_capacity_scale",
        "surface_max_dense_dim",
        "surface_update_cadence",
        "surface_build_lowres_sdf",
    ):
        assert control in source


def test_frame_identity_advances_before_frame_cadence_surface_build():
    source = inspect.getsource(FluidSolver.step_frame)

    frame_advance = source.index("self.frame += 1")
    final_surface = source.index("cadence == int(SurfaceUpdateCadence.FRAME)")
    assert frame_advance < final_surface


def test_blender_surface_prefilter_default_preserves_single_legacy_smooth_pass():
    source = (Path(__file__).parents[1] / "fluidSim_warp_blender51" / "properties.py").read_text()
    marker = "surface_filter_iterations: IntProperty"
    declaration = source[source.index(marker):source.index(marker) + 180]

    assert "default=0" in declaration
