import numpy as np

from fluidSim_warp_blender51.sim.enums import MeshSourceRegion, SourceShapeSourceMode
from fluidSim_warp_blender51.sim.source_compat import descriptor_to_plan
from fluidSim_warp_blender51.sim.sourcing import MeshSourceDesc, SourceShapeDesc


def test_legacy_analytic_is_copied_and_field_maps_only_to_pressure():
    desc = SourceShapeDesc(source_id="blender-object:7",runtime_source_key="session:3:object:7",
                           mode=SourceShapeSourceMode.FIELD)
    plan = descriptor_to_plan(desc,session_generation=3)
    assert str(plan.stable_id) == "session:3:object:7"
    assert plan.operation.name == "PRESSURE"
    assert plan.payload.kind.name == "PRESSURE"


def test_mesh_plan_owns_arrays_and_never_retains_warp_mesh():
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    desc = MeshSourceDesc(vertices=vertices,triangles=np.array([[0,1,2]],np.int32),
                          source_region=MeshSourceRegion.SURFACE_SHELL)
    plan = descriptor_to_plan(desc)
    vertices[:] = 12
    assert not hasattr(plan.region,"mesh")
    assert not plan.region.vertices.flags.writeable
    assert float(plan.region.vertices.max()) == 1.0
