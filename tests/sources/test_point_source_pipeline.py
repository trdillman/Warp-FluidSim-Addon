import numpy as np
import pytest

warp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.enums import (
    SourceEvaluationCadence, SourceLifecycle, SourceOperation, SourcePayloadKind,
)
from fluidSim_warp_blender51.sim.source_regions import (
    CompileContext, PayloadPlan, PointRegionPlan, SourceCompiler, SourceKey, SourcePlan,
)
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings


def context():
    return CompileContext(1,0,1,"cpu",(0,0,0),0.25,(0,0,0),(2,2,2),upload=False)


def make(region, lifecycle):
    return SourcePlan(SourceKey("points"),SourceOperation.LIQUID,lifecycle,
                      SourceEvaluationCadence.STATIC,region,
                      PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))


def test_direct_points_preserve_owned_attributes_and_have_no_volume_coverage():
    values = np.array([[0.5,0.5,0.5],[1,1,1]],np.float32)
    region = PointRegionPlan(values, velocities=values+1,
                             temperature=[2,3],viscosity=[4,5],density=[6,7],
                             lifespan=[8,9])
    values[:] = 99
    compiled = SourceCompiler().compile((make(region,SourceLifecycle.DIRECT_APPEND),),context()).sources[0].region
    assert compiled.leaf_count == 0
    np.testing.assert_allclose(compiled.points.positions, [[0.5,0.5,0.5],[1,1,1]])
    np.testing.assert_allclose(compiled.points.temperature,[2,3])
    assert not compiled.points.positions.flags.writeable


def test_rasterized_points_are_explicit_lifecycle_aware_regions():
    region = PointRegionPlan([[0.5,0.5,0.5]],rasterize_radius=0.3)
    compiled = SourceCompiler().compile((make(region,SourceLifecycle.TOP_UP),),context()).sources[0].region
    assert compiled.leaf_count > 0
    assert np.count_nonzero(compiled.coverage) > 0


def test_point_attribute_lengths_are_validated():
    with pytest.raises(ValueError,match="length"):
        PointRegionPlan([[0,0,0],[1,1,1]],temperature=[1])


@pytest.mark.parametrize(
    "positions",
    [
        np.zeros((3,), np.float32),
        np.zeros((1, 3, 2), np.float32),
        np.zeros((3, 2), np.float32),
    ],
)
def test_point_region_rejects_malformed_position_shapes(positions):
    with pytest.raises(ValueError):
        PointRegionPlan(positions)


def test_point_region_rejects_wrong_vector_tail_shape():
    with pytest.raises(ValueError):
        PointRegionPlan(
            np.zeros((2, 3), np.float32),
            velocities=np.zeros((3, 2), np.float32),
        )


@pytest.mark.parametrize(
    "attribute", ["temperature", "viscosity", "density", "lifespan"])
def test_point_region_rejects_rank_two_scalar_attributes(attribute):
    values = {attribute: np.zeros((2, 1), np.float32)}

    with pytest.raises(ValueError):
        PointRegionPlan(np.zeros((2, 3), np.float32), **values)


def test_point_region_accepts_exact_vector_and_scalar_ranks():
    empty = PointRegionPlan(
        np.zeros((0, 3), np.float32),
        velocities=np.zeros((0, 3), np.float32),
        temperature=np.zeros((0,), np.float32),
    )
    populated = PointRegionPlan(
        np.zeros((2, 3), np.float32),
        velocities=np.zeros((2, 3), np.float32),
        temperature=np.zeros((2,), np.float32),
    )

    assert empty.positions.shape == (0, 3)
    assert empty.velocities.shape == (0, 3)
    assert empty.temperature.shape == (0,)
    assert populated.positions.shape == (2, 3)
    assert populated.velocities.shape == (2, 3)
    assert populated.temperature.shape == (2,)


def test_direct_point_arrays_reach_live_particles_without_constant_substitution(cuda_device):
    settings = SolverSettings()
    settings.domain_min = (0.0, 0.0, 0.0)
    settings.domain_max = (2.0, 2.0, 2.0)
    settings.voxel_size = 0.25
    settings.max_particles = 32
    settings.surface_enabled = False
    settings.whitewater_enabled = False
    settings.device = cuda_device
    solver = FluidSolver(settings)
    region = PointRegionPlan(
        [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        velocities=[[1, 2, 3], [4, 5, 6]],
        temperature=[7, 8], viscosity=[9, 10], density=[11, 12],
        lifespan=[13, 14],
    )
    direct = SourcePlan(
        SourceKey("direct-attrs"), SourceOperation.LIQUID,
        SourceLifecycle.DIRECT_APPEND, SourceEvaluationCadence.STATIC, region,
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
                    velocity=(-1, -1, -1), temperature=-1, viscosity=-1,
                    density=-1, lifespan=-1, use_lifespan=True),
    )
    solver.set_source_batch(native_plans=(direct,))
    solver._rebuild_topology(expand=0)
    solver._clear_grid_fields()
    assert solver.sourcing.apply(1.0 / 24.0)
    assert solver.particles.host_count == 2
    np.testing.assert_allclose(solver.particles.velocity.numpy()[:2], region.velocities)
    np.testing.assert_allclose(solver.particles.attr("temperature").numpy()[:2], region.temperature)
    np.testing.assert_allclose(solver.particles.attr("viscosity").numpy()[:2], region.viscosity)
    np.testing.assert_allclose(solver.particles.attr("density").numpy()[:2], region.density)
    np.testing.assert_allclose(solver.particles.attr("life").numpy()[:2], region.lifespan)
    solver.release()
