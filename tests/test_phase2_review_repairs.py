import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim.enums import (
    SourceEvaluationCadence, SourceLifecycle, SourceOperation,
    SourcePayloadKind, SourceShapeShape,
)
from fluidSim_warp_blender51.sim.source_regions import (
    AnalyticRegionPlan, CompileContext, MeshRegionPlan, PayloadPlan,
    SourceCompiler, SourceKey, SourcePlan,
)


def context(frame=1, *, leaf_capacity=4096, max_bytes=512 * 1024 * 1024):
    return CompileContext(frame, 0, 1, "cpu", (0, 0, 0), .25,
                          (0, 0, 0), (2, 2, 2), leaf_capacity=leaf_capacity,
                          max_source_bytes=max_bytes, upload=False)


def plan(region, *, key="x", cadence=SourceEvaluationCadence.STATIC,
         operation=SourceOperation.LIQUID, ppv=8.0):
    kind = (SourcePayloadKind.LIQUID_ATTRIBUTES
            if operation == SourceOperation.LIQUID else SourcePayloadKind.NONE)
    return SourcePlan(SourceKey(key), operation, SourceLifecycle.TOP_UP,
                      cadence, region,
                      PayloadPlan(kind=kind, target_particles_per_voxel=ppv))


def cube():
    vertices = np.array([
        (.5,.5,.5),(1.5,.5,.5),(1.5,1.5,.5),(.5,1.5,.5),
        (.5,.5,1.5),(1.5,.5,1.5),(1.5,1.5,1.5),(.5,1.5,1.5)], np.float32)
    triangles = np.array([
        (0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
        (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)], np.int32)
    return vertices, triangles


def test_every_frame_cache_evicts_previous_generation_and_releases_live_bytes():
    compiler = SourceCompiler()
    dynamic = plan(AnalyticRegionPlan(
        shape=SourceShapeShape.SPHERE, position=(1,1,1), radius=.4),
        cadence=SourceEvaluationCadence.EVERY_SIM_FRAME)
    for frame in range(50):
        compiler.compile((dynamic,), context(frame))
        assert len(compiler._cache) == 1
    assert compiler.live_cache_bytes > 0
    compiler.release()
    assert compiler.live_cache_bytes == 0


def test_leaf_and_memory_preflight_reject_before_voxel_point_allocation(monkeypatch):
    compiler = SourceCompiler()
    source = plan(AnalyticRegionPlan(
        shape=SourceShapeShape.BOX, position=(1,1,1), half_extents=(1,1,1)))
    monkeypatch.setattr(compiler, "_voxel_points", lambda *_: (_ for _ in ()).throw(
        AssertionError("preflight failed to stop allocation")))
    batch = compiler.compile((source,), context(leaf_capacity=0))
    assert any(d.code == "SOURCE_LEAF_CAPACITY" for d in batch.diagnostics)


def test_empty_candidate_leaves_are_compacted_and_coverage_count_precomputed():
    compiler = SourceCompiler()
    sphere = plan(AnalyticRegionPlan(
        shape=SourceShapeShape.SPHERE, position=(1,1,1), radius=.3))
    region = compiler.compile((sphere,), context()).sources[0].region
    assert region.leaf_count == 1
    assert region.covered_voxel_count == np.count_nonzero(region.coverage)


def test_compiled_force_is_actionably_rejected_and_high_occupancy_warns():
    region = AnalyticRegionPlan(shape=SourceShapeShape.BOX, position=(1,1,1),
                                half_extents=(.25,.25,.25))
    force = SourceCompiler().compile((plan(region, operation=SourceOperation.FORCE),),
                                     context())
    assert any(d.code == "COMPILED_FORCE_UNSUPPORTED" and "legacy" in d.message
               for d in force.diagnostics)
    occupancy = SourceCompiler().compile((plan(region, ppv=96),), context())
    assert any(d.code == "TARGET_OCCUPANCY_CLAMPED" for d in occupancy.diagnostics)


def test_realistic_dynamic_mesh_work_is_bounded_before_distance_evaluation(monkeypatch):
    vertices, triangles = cube()
    triangles = np.tile(triangles, (4167, 1))[:50000]
    compiler = SourceCompiler()
    monkeypatch.setattr(compiler, "_mesh_distances", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(AssertionError("work bound was bypassed")))
    source = plan(MeshRegionPlan(vertices, triangles, solid=False,
                                 shell_half_width=.1),
                  cadence=SourceEvaluationCadence.EVERY_SIM_FRAME)
    fine = CompileContext(1,0,1,"cpu",(0,0,0),.02,(0,0,0),(2,2,2),upload=False)
    batch = compiler.compile((source,), fine)
    assert any(d.code == "SOURCE_DYNAMIC_MESH_BUDGET" for d in batch.diagnostics)


def test_cuda_bvh_shell_matches_cpu_compiler_coverage_and_distance():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    vertices, triangles = cube()
    cpu_source = plan(MeshRegionPlan(
        vertices, triangles, solid=False, shell_half_width=.2), key="cpu")
    # Repeating identical triangles preserves the surface while crossing the
    # 500k-comparison switch into Warp's mesh-BVH staging path.
    gpu_triangles = np.tile(triangles, (88, 1))
    gpu_source = plan(MeshRegionPlan(
        vertices, gpu_triangles, solid=False, shell_half_width=.2), key="gpu")
    cpu_region = SourceCompiler().compile((cpu_source,), context()).sources[0].region
    gpu_context = CompileContext(
        1, 0, 1, "cuda:0", (0,0,0), .25, (0,0,0), (2,2,2), upload=False)
    gpu_region = SourceCompiler().compile((gpu_source,), gpu_context).sources[0].region
    assert not any(d.severity == "ERROR" for d in gpu_region.diagnostics)
    assert np.array_equal(gpu_region.leaf_coords, cpu_region.leaf_coords)
    assert np.array_equal(gpu_region.coverage, cpu_region.coverage)
    assert np.allclose(gpu_region.sdf, cpu_region.sdf, atol=1.0e-5)
