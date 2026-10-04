import numpy as np
import pytest

from fluidSim_warp_blender51.sim.enums import (
    SourceEvaluationCadence, SourceHeightBackMode, SourceLifecycle,
    SourceOperation, SourcePayloadKind,
)
from fluidSim_warp_blender51.sim.source_regions import (
    CompileContext, HeightRegionPlan, PayloadPlan, SourceCompiler, SourceKey, SourcePlan,
)


def test_profile_requires_real_input_and_changes_height_coverage():
    with pytest.raises(ValueError,match="explicit profile"):
        HeightRegionPlan(0.5,SourceHeightBackMode.PROFILE)
    with pytest.raises(ValueError, match="not defined"):
        HeightRegionPlan(0.5, SourceHeightBackMode.INFINITE)
    context = CompileContext(1,0,1,"cpu",(0,0,0),0.25,(0,0,0),(1,1,1),upload=False)
    profile = np.array([[0,0],[0.5,0.5]],np.float32)
    plan = SourcePlan(SourceKey("height"),SourceOperation.LIQUID,SourceLifecycle.ONCE,
                      SourceEvaluationCadence.STATIC,
                      HeightRegionPlan(0.25,SourceHeightBackMode.PROFILE,profile),
                      PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))
    region = SourceCompiler().compile((plan,),context).sources[0].region
    assert np.count_nonzero(region.coverage) > 0
    assert region.coverage.min() == 0
    assert region.coverage.max() == 1


def test_height_compiler_includes_runtime_four_voxel_topology_safety_band():
    context = CompileContext(
        1, 0, 1, "cpu", (0, 0, 0), 0.1,
        (0, 0, 0), (1.6, 1.6, 1.6), upload=False)
    plan = SourcePlan(
        SourceKey("height-safety"), SourceOperation.LIQUID,
        SourceLifecycle.EVERY_FRAME, SourceEvaluationCadence.EVERY_SIM_FRAME,
        HeightRegionPlan(0.55, SourceHeightBackMode.FLAT),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))

    region = SourceCompiler().compile((plan,), context).sources[0].region

    # Leaf z=1 starts at 0.8. A one-voxel band would stop at 0.65 and miss it;
    # the runtime's four-voxel band reaches 0.95 and must be planned too.
    assert max(int(coord[2]) for coord in region.leaf_coords) == 0
    assert max(int(coord[2]) for coord in region.topology_leaf_coords) == 1
