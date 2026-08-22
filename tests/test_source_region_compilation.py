from dataclasses import replace

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.enums import (
    SourceEvaluationCadence, SourceLifecycle, SourceOperation,
    SourcePayloadKind, SourceShapeShape,
)
from fluidSim_warp_blender51.sim.source_regions import (
    AnalyticRegionPlan, CompileContext, MeshRegionPlan, PayloadPlan,
    PointRegionPlan, SourceCompiler, SourceKey, SourcePlan,
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


@pytest.mark.parametrize(
    ("vertices", "triangles"),
    [
        (np.zeros((3,), np.float32), np.zeros((0, 3), np.int32)),
        (np.zeros((1, 3, 2), np.float32), np.zeros((0, 3), np.int32)),
        (np.zeros((3, 2), np.float32), np.zeros((0, 3), np.int32)),
        (np.zeros((0, 3), np.float32), np.zeros((3, 2), np.int32)),
    ],
)
def test_mesh_region_rejects_malformed_vector_shapes(vertices, triangles):
    with pytest.raises(ValueError):
        MeshRegionPlan(vertices, triangles)


def test_mesh_region_accepts_exact_empty_and_populated_vector_shapes():
    empty = MeshRegionPlan(
        np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    populated = MeshRegionPlan(
        np.zeros((2, 3), np.float32), np.zeros((1, 3), np.int32))

    assert empty.vertices.shape == (0, 3)
    assert empty.triangles.shape == (0, 3)
    assert populated.vertices.shape == (2, 3)
    assert populated.triangles.shape == (1, 3)


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


def test_no_upload_planning_cache_cannot_poison_runtime_upload():
    compiler = SourceCompiler()
    source = plan(AnalyticRegionPlan(
        shape=SourceShapeShape.BOX, position=(1, 1, 1),
        half_extents=(0.5, 0.5, 0.5)), "upload-isolation")

    planned = compiler.compile((source,), ctx())
    runtime = compiler.compile((source,), replace(ctx(), upload=True))

    assert planned.sources[0].region.device_coverage is None
    assert runtime.sources[0].region.device_coverage is not None
    assert runtime.sources[0].region.device_leaf_coords is not None


def test_compiled_footprints_expose_conservative_particle_and_leaf_demand():
    source = plan(AnalyticRegionPlan(
        shape=SourceShapeShape.BOX, position=(1, 1, 1),
        half_extents=(0.5, 0.5, 0.5)), "footprint")
    source = replace(source, payload=replace(
        source.payload, target_particles_per_voxel=2.0))

    batch = SourceCompiler().compile((source,), ctx())
    footprint = batch.footprints[0]

    # A unit box at h=.25 covers 4^3 voxels, each with full coverage.
    assert footprint.stable_id == "footprint"
    assert footprint.particle_demand == 128
    assert footprint.particle_quality.value == "upper_bound"
    assert len(footprint.leaf_coords) == batch.sources[0].region.leaf_count
    assert footprint.staging_bytes == batch.sources[0].region.device_bytes


def test_zero_liquid_occupancy_has_no_particle_or_leaf_demand():
    source = plan(AnalyticRegionPlan(
        shape=SourceShapeShape.BOX, position=(1, 1, 1),
        half_extents=(0.5, 0.5, 0.5)), "zero-footprint")
    source = replace(source, payload=replace(
        source.payload, target_particles_per_voxel=0.0))

    footprint = SourceCompiler().compile((source,), ctx()).footprints[0]

    assert footprint.particle_demand == 0
    assert footprint.particle_quality.value == "exact"
    assert footprint.leaf_coords == ()


def test_direct_point_footprint_count_is_exact():
    points = np.array(((0.25, 0.25, 0.25), (1.25, 1.25, 1.25)), np.float32)
    direct = SourcePlan(
        SourceKey("direct"), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC,
        PointRegionPlan(points),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES),
    )

    footprint = SourceCompiler().compile((direct,), ctx()).footprints[0]

    assert footprint.particle_demand == 2
    assert footprint.particle_quality.value == "exact"


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
