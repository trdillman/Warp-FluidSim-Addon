import inspect

from fluidSim_warp_blender51.sim import solver as solver_mod
from fluidSim_warp_blender51.sim.surface import SurfacePipeline


def test_surface_pipeline_owns_surface_state_and_solver_only_delegates():
    source = inspect.getsource(solver_mod.FluidSolver)
    assert "def _ensure_render_surface_resources" not in source
    assert "def _build_surface_sdf_on_grid" not in source
    assert "def _rebuild_render_surface_topology" not in source
    assert "_surface_mesh_cache =" not in source
    assert SurfacePipeline
