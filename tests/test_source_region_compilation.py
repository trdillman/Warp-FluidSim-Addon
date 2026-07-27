import numpy as np

from fluidSim_warp_blender51.sim.enums import (
    SourceEvaluationCadence, SourceLifecycle, SourceOperation,
    SourcePayloadKind, SourceShapeShape,
)
from fluidSim_warp_blender51.sim.source_regions import (
    AnalyticRegionPlan, CompileContext, MeshRegionPlan, PayloadPlan,
    SourceCompiler, SourceKey, SourcePlan,
)


def cube(open_top=False):
    v = np.array([(0.5,0.5,0.5),(1.5,0.5,0.5),(1.5,1.5,0.5),(0.5,1.5,0.5),
                  (0.5,0.5,1.5),(1.5,0.5,1.5),(1.5,1.5,1.5),(0.5,1.5,1.5)],np.float32)
    t = np.array([(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
                  (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)],np.int32)
    return v, t[:-2] if open_top else t


def ctx():
    return CompileContext(1,0,1,"cpu",(0,0,0),0.25,(0,0,0),(2,2,2),upload=False)


def plan(region, key="x"):
    return SourcePlan(SourceKey(key),SourceOperation.LIQUID,SourceLifecycle.TOP_UP,
                      SourceEvaluationCadence.STATIC,region,
                      PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))


def test_watertight_mesh_solid_matches_analytic_box_at_voxel_centers():
    vertices, triangles = cube()
    compiler = SourceCompiler()
    mesh = compiler.compile((plan(MeshRegionPlan(vertices,triangles,solid=True),"mesh"),),ctx())
    box = compiler.compile((plan(AnalyticRegionPlan(
        shape=SourceShapeShape.BOX,position=(1,1,1),half_extents=(0.5,0.5,0.5)),"box"),),ctx())
    mesh_count = int(np.count_nonzero(mesh.sources[0].region.coverage))
    box_count = int(np.count_nonzero(box.sources[0].region.coverage))
    assert mesh_count == box_count == 64
    assert (mesh.sources[0].region.sdf[mesh.sources[0].region.coverage > 0] <= 0).all()


def test_open_mesh_rejects_solid_but_passes_explicit_shell_without_fallback():
    vertices, triangles = cube(open_top=True)
    compiler = SourceCompiler()
    solid = compiler.compile((plan(MeshRegionPlan(vertices,triangles,solid=True)),),ctx()).sources[0].region
    assert solid.kind.name == "MESH_SOLID"
    assert any(d.code == "MESH_NOT_WATERTIGHT" for d in solid.diagnostics)
    assert solid.leaf_count == 0
    shell = compiler.compile((plan(MeshRegionPlan(
        vertices, triangles, solid=False, shell_half_width=0.15),"shell"),),ctx()).sources[0].region
    assert shell.kind.name == "MESH_SHELL"
    assert shell.leaf_count > 0
    assert np.count_nonzero(shell.coverage) > 0
    assert not shell.diagnostics


def test_capacity_failure_is_structured_and_visible():
    vertices, triangles = cube()
    limited = CompileContext(1,0,1,"cpu",(0,0,0),0.05,(0,0,0),(2,2,2),
                             leaf_capacity=1,upload=False)
    batch = SourceCompiler().compile((plan(MeshRegionPlan(vertices,triangles,solid=True)),),limited)
    assert any(d.code == "SOURCE_LEAF_CAPACITY" and d.severity == "ERROR"
               for d in batch.diagnostics)


def test_static_source_edit_releases_superseded_cache_generation():
    compiler = SourceCompiler()
    first = plan(AnalyticRegionPlan(
        shape=SourceShapeShape.BOX, position=(0.75, 1, 1),
        half_extents=(0.25, 0.25, 0.25)), "stable")
    second = plan(AnalyticRegionPlan(
        shape=SourceShapeShape.BOX, position=(1.25, 1, 1),
        half_extents=(0.25, 0.25, 0.25)), "stable")
    compiler.compile((first,), ctx())
    old_region = next(iter(compiler._cache.values()))
    compiler.compile((second,), ctx())
    assert len(compiler._cache) == 1
    assert old_region.device_leaf_coords is None
    assert compiler.live_cache_bytes == next(iter(compiler._cache.values())).device_bytes
