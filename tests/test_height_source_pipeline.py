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
