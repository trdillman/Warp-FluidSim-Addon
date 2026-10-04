from dataclasses import FrozenInstanceError

import pytest

from fluidSim_warp_blender51.sim.sourcing import SourceApplyResult, SourcePipeline


def test_source_apply_result_is_complete_immutable_mapping():
    result = SourceApplyResult(
        emitted=2,
        capacity_limited=False,
        required=2,
        available=8,
        admitted=True,
    )

    assert result["emitted"] == 2
    assert result.get("growth_status") == "UNCHANGED"
    assert dict(result)["pause_requested"] is False
    with pytest.raises(FrozenInstanceError):
        result.available = 4


def test_pipeline_capacity_update_replaces_receipt():
    pipeline = SourcePipeline.__new__(SourcePipeline)
    original = SourceApplyResult(
        emitted=0,
        capacity_limited=True,
        required=6,
        available=4,
        admitted=False,
        deferred=True,
    )
    pipeline.last_apply_result = original

    pipeline.update_available_capacity(12)

    assert pipeline.last_apply_result.available == 12
    assert pipeline.last_apply_result.deferred is True
    assert original.available == 4
