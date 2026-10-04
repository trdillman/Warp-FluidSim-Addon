"""Frame feature sequencing contracts."""

from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.sim.effects.enums import EffectStage
from fluidSim_warp_blender51.sim.frame_features import (
    FrameFeaturePhase,
    FrameFeatures,
)


class _Effects:
    def __init__(self, events, *, surface_sdf=False, rebuild_topology=False):
        self.events = events
        self.runtime_errors = []
        self.requirements = SimpleNamespace(
            rebuild_topology_each_substep=rebuild_topology)
        self.surface_sdf = surface_sdf

    def requirements_for_stage(self, stage):
        self.events.append(("requirements", stage))
        return SimpleNamespace(surface_sdf=self.surface_sdf)

    def run(self, stage, solver, time_context):
        self.events.append(("effect", stage, solver, time_context))


def features(events, *, surface_sdf=False, rebuild_topology=False):
    value = FrameFeatures.__new__(FrameFeatures)
    value.effects = _Effects(
        events,
        surface_sdf=surface_sdf,
        rebuild_topology=rebuild_topology,
    )
    return value


@pytest.mark.parametrize(("phase", "effect"), [
    (FrameFeaturePhase.BEFORE_EMISSION, EffectStage.BEFORE_EMISSION),
    (FrameFeaturePhase.BEFORE_SOURCING, None),
    (FrameFeaturePhase.AFTER_PARTICLE_TRANSFER, EffectStage.AFTER_PARTICLE_TRANSFER),
    (FrameFeaturePhase.BEFORE_PRESSURE, EffectStage.BEFORE_PRESSURE),
    (FrameFeaturePhase.AFTER_PARTICLE_UPDATE, EffectStage.AFTER_PARTICLE_UPDATE),
])
def test_frame_feature_phase_runs_only_the_v2_effect_stage(phase, effect):
    events = []
    value = features(events)
    solver = SimpleNamespace(
        surface=SimpleNamespace(ensure_solver_sdf_current=lambda: None))
    context = {"frame": 4}

    value.run(phase, solver, context)

    if effect is None:
        assert events == []
    else:
        assert events == [
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
        "requirements", "surface", "effect"]


def test_frame_features_report_v2_topology_requirement():
    value = features([], rebuild_topology=True)

    assert value.rebuild_topology_each_substep is True
