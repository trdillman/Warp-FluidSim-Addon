"""Frame feature sequencing contracts."""

from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.sim.enums import CustomKernelOrder
from fluidSim_warp_blender51.sim.effects.enums import EffectStage
from fluidSim_warp_blender51.sim.frame_features import (
    FrameFeaturePhase,
    FrameFeatures,
)


class _Snippets:
    def __init__(self, events):
        self.events = events
        self.errors = []

    def has_any(self):
        return True

    def run(self, order, solver, time_context):
        self.events.append(("snippet", order, solver, time_context))


class _Effects:
    def __init__(self, events, *, surface_sdf=False):
        self.events = events
        self.runtime_errors = []
        self.requirements = SimpleNamespace(
            rebuild_topology_each_substep=False)
        self.surface_sdf = surface_sdf

    def requirements_for_stage(self, stage):
        self.events.append(("requirements", stage))
        return SimpleNamespace(surface_sdf=self.surface_sdf)

    def run(self, stage, solver, time_context):
        self.events.append(("effect", stage, solver, time_context))


def features(events, *, surface_sdf=False):
    value = FrameFeatures.__new__(FrameFeatures)
    value.snippets = _Snippets(events)
    value.effects = _Effects(events, surface_sdf=surface_sdf)
    return value


@pytest.mark.parametrize(("phase", "snippet", "effect"), [
    (FrameFeaturePhase.BEFORE_EMISSION,
     CustomKernelOrder.FIRST_STEP, EffectStage.BEFORE_EMISSION),
    (FrameFeaturePhase.BEFORE_SOURCING,
     CustomKernelOrder.BEFORE_SOURCING, None),
    (FrameFeaturePhase.AFTER_PARTICLE_TRANSFER,
     CustomKernelOrder.BEFORE_FORCES, EffectStage.AFTER_PARTICLE_TRANSFER),
    (FrameFeaturePhase.BEFORE_PRESSURE,
     CustomKernelOrder.BEFORE_PROJECT_NON_DIVERGENT_AND_ADVECTION,
     EffectStage.BEFORE_PRESSURE),
    (FrameFeaturePhase.AFTER_PARTICLE_UPDATE,
     CustomKernelOrder.FINAL_STEP, EffectStage.AFTER_PARTICLE_UPDATE),
])
def test_frame_feature_phase_preserves_hook_order(phase, snippet, effect):
    events = []
    value = features(events)
    solver = SimpleNamespace(
        surface=SimpleNamespace(ensure_solver_sdf_current=lambda: None))
    context = {"frame": 4}

    value.run(phase, solver, context)

    assert events[0] == ("snippet", snippet, solver, context)
    if effect is None:
        assert events == [("snippet", snippet, solver, context)]
    else:
        assert events[1:] == [
            ("requirements", effect),
            ("effect", effect, solver, context),
        ]


def test_frame_features_prepare_surface_before_effect():
    events = []
    value = features(events, surface_sdf=True)
    solver = SimpleNamespace(surface=SimpleNamespace(
        ensure_solver_sdf_current=lambda: events.append(("surface",))))

    value.run(FrameFeaturePhase.BEFORE_PRESSURE, solver, {})

    assert [event[0] for event in events] == [
        "snippet", "requirements", "surface", "effect"]


def test_frame_features_report_combined_topology_requirement():
    value = features([])

    assert value.rebuild_topology_each_substep is True
