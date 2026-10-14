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
        return SimpleNamespace(surface_sdf=self.surface_sdf, moves_particles=False)

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
        ensure_solver_sdf_current=lambda: events.append(("surface",)) or True))

    value.run(FrameFeaturePhase.BEFORE_PRESSURE, solver, {})

    assert [event[0] for event in events] == [
        "requirements", "surface", "effect"]


def _sdf_reading_features(launches, monkeypatch):
    """Real FrameFeatures with one installed voxel kernel that reads surface_sdf."""
    from fluidSim_warp_blender51.sim.effects import runtime
    from fluidSim_warp_blender51.sim.effects.enums import (
        EffectCadence, EffectDomain, FieldAccess)
    from fluidSim_warp_blender51.sim.effects.schema import (
        EffectDefinitionV2, EffectInstanceV2, FieldBindingV2, KernelDefinitionV2)

    kernel = KernelDefinitionV2(
        kernel_id="reads_sdf", label="Reads SDF",
        domain=EffectDomain.ACTIVE_VOXELS, stage=EffectStage.BEFORE_PRESSURE,
        cadence=EffectCadence.EVERY_SUBSTEP,
        source="value = surface_sdf[flat_voxel_index]",
        fields=(FieldBindingV2("surface_sdf", FieldAccess.READ),))
    definition = EffectDefinitionV2(
        effect_id="test.sdf_reader", version="1.0.0", name="SDF Reader",
        summary="s", description="d", category="Tests", author="Tests",
        license="GPL-3.0-or-later", kernels=(kernel,))
    instance = EffectInstanceV2("sdf-reader-1", definition)
    compiled = SimpleNamespace(
        success=True, instance=instance,
        kernels=(SimpleNamespace(kernel_definition=kernel,
                                 generated=SimpleNamespace(kernel_index=0)),))
    value = FrameFeatures(device="cpu")
    assert value.swap_candidate(
        SimpleNamespace(success=True, compiled_effects=(compiled,)))
    monkeypatch.setattr(
        runtime, "launch_compiled_kernel",
        lambda entry, solver, ctx, device: launches.append(entry.key) or (8, 0.0))
    return value


def test_failed_required_sdf_refresh_fails_closed_with_structured_error(monkeypatch):
    # B69: the Boolean result of ensure_solver_sdf_current() was ignored, so
    # the effect ran on stale or zero-initialized SDF storage.
    from fluidSim_warp_blender51.sim.effects.runtime import EffectExecutionError

    launches = []
    value = _sdf_reading_features(launches, monkeypatch)
    surface = SimpleNamespace(
        ensure_solver_sdf_current=lambda: False,
        solver_sdf_error="Simulation liquid SDF unavailable: boom")
    solver = SimpleNamespace(surface=surface)

    with pytest.raises(EffectExecutionError) as caught:
        value.run(FrameFeaturePhase.BEFORE_PRESSURE, solver,
                  {"frame": 7, "substep_index": 1})

    assert launches == []
    error = caught.value.runtime_error
    assert (error.effect_id, error.instance_id, error.kernel_id) == (
        "test.sdf_reader", "sdf-reader-1", "reads_sdf")
    assert (error.stage, error.frame, error.substep_index) == (
        EffectStage.BEFORE_PRESSURE.value, 7, 1)
    assert "SDF" in error.message and "boom" in error.message
    assert value.effects.failed_error == error


def test_current_sdf_lets_the_effect_run(monkeypatch):
    launches = []
    value = _sdf_reading_features(launches, monkeypatch)
    solver = SimpleNamespace(surface=SimpleNamespace(
        ensure_solver_sdf_current=lambda: True, solver_sdf_error=""))

    value.run(FrameFeaturePhase.BEFORE_PRESSURE, solver, {"frame": 7})

    assert launches == ["sdf-reader-1:reads_sdf:0"]


def test_frame_features_report_v2_topology_requirement():
    value = features([], rebuild_topology=True)

    assert value.rebuild_topology_each_substep is True
