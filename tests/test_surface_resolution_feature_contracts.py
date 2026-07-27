"""Contracts for decoupled render-surface resolution."""

from pathlib import Path
import inspect
import re
from contextlib import nullcontext
from types import SimpleNamespace

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
    final_surface = source.index("surface.after_frame")
    assert frame_advance < final_surface


def test_surface_pipeline_observes_substep_frame_and_manual_cadence():
    calls = []
    pipeline = object.__new__(SurfacePipeline)
    pipeline.solver = SimpleNamespace(settings=SimpleNamespace(
        surface_enabled=True,
        surface_build_lowres_sdf=False,
        surface_update_cadence=SurfaceUpdateCadence.SUBSTEP,
        whitewater_enabled=False,
    ), _phase=lambda _name: nullcontext())
    pipeline.build = lambda build_render=True, build_lowres=None: calls.append(
        (build_render, build_lowres))

    pipeline.after_substep()
    pipeline.after_frame()
    assert calls == [(True, True)]

    calls.clear()
    pipeline.solver.settings.surface_update_cadence = SurfaceUpdateCadence.FRAME
    pipeline.solver.settings.whitewater_enabled = True
    pipeline.after_substep()
    pipeline.after_frame()
    assert calls == [(False, True), (True, False)]

    calls.clear()
    pipeline.solver.settings.surface_update_cadence = SurfaceUpdateCadence.MANUAL
    pipeline.solver.settings.whitewater_enabled = False
    pipeline.after_substep()
    pipeline.after_frame()
    assert calls == []


def test_surface_metadata_reports_active_render_grid():
    pipeline = object.__new__(SurfacePipeline)
    pipeline.solver = SimpleNamespace(settings=SimpleNamespace(
        surface_resolution_mode=1, surface_resolution_multiplier=2.0,
        surface_update_cadence=1, surface_smooth_iterations=3))
    pipeline.grid = SimpleNamespace(active_leaf_count=7)
    pipeline.voxel_size = lambda: 0.025

    assert pipeline.metadata() == {
        "voxel_size": 0.025, "resolution_mode": 1,
        "resolution_multiplier": 2.0, "update_cadence": 1,
        "smooth_iterations": 3, "active_leaves": 7,
        "active_voxels": 7 * 512,
    }


def test_blender_surface_prefilter_default_preserves_single_legacy_smooth_pass():
    source = (Path(__file__).parents[1] / "fluidSim_warp_blender51" / "properties.py").read_text()
    declaration = re.search(
        r"surface_filter_iterations\s*:\s*IntProperty\((.*?)\)\s*$",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )

    assert declaration is not None
    assert re.search(r"\bdefault\s*=\s*0\b", declaration.group(1))
