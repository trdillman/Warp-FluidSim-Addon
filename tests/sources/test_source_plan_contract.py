from dataclasses import fields

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.enums import (
    SourceEvaluationCadence, SourceLifecycle, SourceOperation,
    SourcePayloadKind, SourceShapeShape,
)
from fluidSim_warp_blender51.sim.source_regions import (
    ActivationPlan, AnalyticRegionPlan, CompileContext, PayloadPlan,
    PointRegionPlan, SourceCompiler, SourceKey, SourcePlan,
)


def _context(**changes):
    values = dict(frame=1, substep=0, session_generation=1, device="cpu",
                  origin=(0, 0, 0), voxel_size=0.25,
                  domain_min=(0, 0, 0), domain_max=(2, 2, 2), upload=False)
    values.update(changes)
    return CompileContext(**values)


def _plan(region=None, **changes):
    values = dict(stable_id=SourceKey("session:1:object:42"),
                  operation=SourceOperation.LIQUID,
                  lifecycle=SourceLifecycle.TOP_UP,
                  cadence=SourceEvaluationCadence.STATIC,
                  region=region or AnalyticRegionPlan(
                      shape=SourceShapeShape.BOX, position=(1, 1, 1),
                      half_extents=(0.5, 0.5, 0.5)),
                  payload=PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))
    values.update(changes)
    return SourcePlan(**values)


def test_plan_axes_are_independent_and_blender_free():
    names = {item.name for item in fields(SourcePlan)}
    assert names == {"stable_id", "operation", "lifecycle", "cadence",
                     "region", "payload", "activation", "blend"}
    assert "bpy" not in open(
        "fluidSim_warp_blender51/sim/source_regions.py", encoding="utf8").read()


def test_field_payload_cannot_pretend_to_emit_liquid():
    with pytest.raises(ValueError, match="cannot be represented as liquid"):
        _plan(payload=PayloadPlan(kind=SourcePayloadKind.PRESSURE))


def test_direct_append_is_points_only():
    with pytest.raises(ValueError, match="point inputs"):
        _plan(lifecycle=SourceLifecycle.DIRECT_APPEND)
    plan = _plan(
        region=PointRegionPlan(np.zeros((1, 3), np.float32)),
        lifecycle=SourceLifecycle.DIRECT_APPEND)
    assert plan.region.kind.name == "POINTS"


def test_compiler_allocates_only_intersecting_flat_8_cubed_leaves_and_caches():
    compiler = SourceCompiler()
    batch = compiler.compile((_plan(),), _context())
    assert 0 < len(batch.required_leaf_coords) <= 8
    assert batch.sources[0].region.coverage.shape[1] == 8 ** 3
    assert not any("tree" in item.name.lower() for item in fields(
        type(batch.sources[0].region)))
    again = compiler.compile((_plan(),), _context(frame=9))
    assert compiler.compile_count == 1
    assert compiler.cache_hits == 1
    assert again.sources[0].region is batch.sources[0].region
    compiler.release()
    assert compiler._cache == {}


def test_deterministic_seed_includes_source_voxel_frame_and_sample():
    seed = SourceCompiler.deterministic_seed("a", (1, 2, 3), 4, 5)
    assert seed == SourceCompiler.deterministic_seed("a", (1, 2, 3), 4, 5)
    assert len({seed,
                SourceCompiler.deterministic_seed("b", (1, 2, 3), 4, 5),
                SourceCompiler.deterministic_seed("a", (2, 2, 3), 4, 5),
                SourceCompiler.deterministic_seed("a", (1, 2, 3), 5, 5),
                SourceCompiler.deterministic_seed("a", (1, 2, 3), 4, 6)}) == 5


def test_activation_and_every_frame_invalidation_are_separate():
    compiler = SourceCompiler()
    inactive = _plan(activation=ActivationPlan(False))
    assert compiler.compile((inactive,), _context()).sources == ()
    dynamic = _plan(cadence=SourceEvaluationCadence.EVERY_SIM_FRAME)
    compiler.compile((dynamic,), _context(frame=1))
    compiler.compile((dynamic,), _context(frame=2))
    assert compiler.compile_count == 2
