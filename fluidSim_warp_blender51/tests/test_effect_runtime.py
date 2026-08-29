from __future__ import annotations

from dataclasses import dataclass

import pytest
from fluidsim_testpkg.sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    FieldAccess,
)
from fluidsim_testpkg.sim.effects.runtime import (
    EffectCandidateInstallError,
    EffectExecutionError,
    EffectRuntimeManager,
    EffectRuntimeRequirements,
)
from fluidsim_testpkg.sim.effects.schema import (
    EffectDefinitionV2,
    EffectInstanceV2,
    FieldBindingV2,
    KernelDefinitionV2,
)


@dataclass(frozen=True)
class FakeGenerated:
    kernel_index: int = 0


@dataclass(frozen=True)
class FakeCompiledKernel:
    kernel_definition: KernelDefinitionV2
    generated: FakeGenerated = FakeGenerated()
    kernel: object = object()


@dataclass(frozen=True)
class FakeCompiledEffect:
    instance: EffectInstanceV2
    kernels: tuple[FakeCompiledKernel, ...]
    success: bool = True


@dataclass(frozen=True)
class FakeCandidate:
    success: bool
    compiled_effects: tuple[FakeCompiledEffect, ...] = ()
    issues: tuple[object, ...] = ()


def _instance(
    instance_id: str,
    *,
    domain=EffectDomain.PARTICLES,
    stage=EffectStage.AFTER_PARTICLE_TRANSFER,
    cadence=EffectCadence.EVERY_SUBSTEP,
    fields=(),
    order=0,
    frame_start=0,
    frame_end=0,
    frame_step=1,
):
    kernel = KernelDefinitionV2(
        kernel_id=f"kernel_{instance_id.replace('-', '_')}",
        label="Kernel",
        domain=domain,
        stage=stage,
        cadence=cadence,
        source="pass",
        fields=tuple(fields),
    )
    definition = EffectDefinitionV2(
        effect_id=f"test.runtime-{instance_id}",
        version="1.0.0",
        name=instance_id,
        summary=instance_id,
        description=instance_id,
        category="Tests",
        author="Tests",
        license="GPL-3.0-or-later",
        built_in=True,
        kernels=(kernel,),
    )
    instance = EffectInstanceV2(
        instance_id,
        definition,
        order=order,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_step=frame_step,
    )
    return instance, FakeCompiledEffect(instance, (FakeCompiledKernel(kernel),))


def test_failed_candidate_preserves_previous_applied_stack():
    instance, compiled = _instance("previous")
    manager = EffectRuntimeManager(device="cpu")
    assert manager.swap_candidate(FakeCandidate(True, (compiled,))) is True
    previous_signature = manager.applied_signature
    previous_entries = manager.entry_count

    assert manager.swap_candidate(FakeCandidate(False, (), ("compile failed",))) is False
    assert manager.applied_signature == previous_signature
    assert manager.entry_count == previous_entries == 1


def test_candidate_preparation_error_preserves_stack_and_original_exception():
    _instance_value, compiled = _instance("previous")
    manager = EffectRuntimeManager(device="cpu")
    assert manager.swap_candidate(FakeCandidate(True, (compiled,))) is True
    previous_signature = manager.applied_signature
    broken = FakeCompiledEffect(compiled.instance, compiled.kernels, success=False)

    with pytest.raises(
        EffectCandidateInstallError,
        match="PREPARE_CANDIDATE: ValueError",
    ) as captured:
        manager.swap_candidate(FakeCandidate(True, (broken,)))

    error = captured.value
    assert error.diagnostic.phase == "PREPARE_CANDIDATE"
    assert error.diagnostic.exception_type == "ValueError"
    assert error.diagnostic.message == (
        "candidate contains an unsuccessful compiled effect"
    )
    assert isinstance(error.original_exception, ValueError)
    assert error.__cause__ is error.original_exception
    assert manager.applied_signature == previous_signature
    assert manager.entry_count == 1


def test_stage_order_and_every_substep_launches(monkeypatch):
    first, compiled_first = _instance("first", order=0)
    second, compiled_second = _instance("second", order=1)
    calls = []

    def fake_launch(entry, _solver, _time_ctx, _device):
        calls.append(entry.instance.instance_id)
        return 4, 0.001

    monkeypatch.setattr("fluidsim_testpkg.sim.effects.runtime.launch_compiled_kernel", fake_launch)
    manager = EffectRuntimeManager(device="cpu")
    manager.swap_candidate(FakeCandidate(True, (compiled_second, compiled_first)))
    manager.run(EffectStage.AFTER_PARTICLE_TRANSFER, object(), {"frame": 1, "substep_index": 0})
    manager.run(EffectStage.AFTER_PARTICLE_TRANSFER, object(), {"frame": 1, "substep_index": 1})
    assert calls == ["first", "second", "first", "second"]


def test_frame_stats_snapshot_reports_deltas_without_resetting_totals(
        monkeypatch):
    _instance_value, compiled = _instance("profile")

    monkeypatch.setattr(
        "fluidsim_testpkg.sim.effects.runtime.launch_compiled_kernel",
        lambda *_args, **_kwargs: (5, 0.002),
    )
    manager = EffectRuntimeManager(device="cpu")
    manager.swap_candidate(FakeCandidate(True, (compiled,)))

    manager.begin_frame()
    manager.run(
        EffectStage.AFTER_PARTICLE_TRANSFER,
        object(),
        {"frame": 1, "substep_index": 0},
    )
    manager.run(
        EffectStage.AFTER_PARTICLE_TRANSFER,
        object(),
        {"frame": 1, "substep_index": 1},
    )
    first = manager.frame_stats_snapshot()

    manager.begin_frame()
    manager.run(
        EffectStage.AFTER_PARTICLE_TRANSFER,
        object(),
        {"frame": 2, "substep_index": 0},
    )
    second = manager.frame_stats_snapshot()
    cumulative = next(iter(manager.stats.values()))

    assert first[0]["calls"] == 2
    assert first[0]["elements"] == 10
    assert first[0]["submissionSeconds"] == pytest.approx(0.004)
    assert second[0]["calls"] == 1
    assert second[0]["elements"] == 5
    assert second[0]["submissionSeconds"] == pytest.approx(0.002)
    assert cumulative.calls == 3
    assert cumulative.elements == 15
    assert cumulative.submission_seconds == pytest.approx(0.006)


def test_once_per_frame_and_initialization_cadence(monkeypatch):
    once, compiled_once = _instance("once", cadence=EffectCadence.ONCE_PER_FRAME)
    init, compiled_init = _instance("init", cadence=EffectCadence.INITIALIZATION, order=1)
    calls = []

    def fake_launch(entry, _solver, time_ctx, _device):
        calls.append((entry.instance.instance_id, time_ctx["frame"], time_ctx["substep_index"]))
        return 1, 0.0

    monkeypatch.setattr("fluidsim_testpkg.sim.effects.runtime.launch_compiled_kernel", fake_launch)
    manager = EffectRuntimeManager(device="cpu")
    manager.swap_candidate(FakeCandidate(True, (compiled_once, compiled_init)))
    for frame in (1, 2):
        for substep in (0, 1, 2):
            manager.run(
                EffectStage.AFTER_PARTICLE_TRANSFER,
                object(),
                {"frame": frame, "substep_index": substep},
            )
    assert calls == [("once", 1, 0), ("init", 1, 0), ("once", 2, 0)]


def test_frame_window_and_step_are_enforced(monkeypatch):
    instance, compiled = _instance("window", frame_start=2, frame_end=6, frame_step=2)
    calls = []

    def fake_launch(entry, _solver, time_ctx, _device):
        calls.append(time_ctx["frame"])
        return 1, 0.0

    monkeypatch.setattr("fluidsim_testpkg.sim.effects.runtime.launch_compiled_kernel", fake_launch)
    manager = EffectRuntimeManager(device="cpu")
    manager.swap_candidate(FakeCandidate(True, (compiled,)))
    for frame in range(1, 8):
        manager.run(
            EffectStage.AFTER_PARTICLE_TRANSFER,
            object(),
            {"frame": frame, "substep_index": 0},
        )
    assert calls == [2, 4, 6]


def test_particle_only_effect_does_not_require_topology_rebuild():
    _instance_value, compiled = _instance(
        "particle",
        fields=(FieldBindingV2("particle_velocity", FieldAccess.READ_WRITE_CURRENT),),
    )
    manager = EffectRuntimeManager(device="cpu")
    manager.swap_candidate(FakeCandidate(True, (compiled,)))
    assert manager.requirements == EffectRuntimeRequirements(
        particles=True,
        topology=False,
        surface_sdf=False,
        collision=False,
        rebuild_topology_each_substep=False,
    )


def test_active_voxel_and_sdf_effect_derives_prerequisites():
    _instance_value, compiled = _instance(
        "voxel",
        domain=EffectDomain.ACTIVE_VOXELS,
        stage=EffectStage.BEFORE_PRESSURE,
        fields=(FieldBindingV2("surface_sdf", FieldAccess.READ),),
    )
    manager = EffectRuntimeManager(device="cpu")
    manager.swap_candidate(FakeCandidate(True, (compiled,)))
    assert manager.requirements.topology is True
    assert manager.requirements.surface_sdf is True
    assert manager.requirements.rebuild_topology_each_substep is True


def test_runtime_error_fails_closed_until_a_new_candidate_is_applied(monkeypatch):
    _instance_value, compiled = _instance("stats")
    replacement_instance, replacement = _instance("replacement")
    launches = iter([(3, 0.01), RuntimeError("bad launch"), (2, 0.02)])

    def fake_launch(_entry, _solver, _time_ctx, _device):
        result = next(launches)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("fluidsim_testpkg.sim.effects.runtime.launch_compiled_kernel", fake_launch)
    manager = EffectRuntimeManager(device="cpu")
    manager.swap_candidate(FakeCandidate(True, (compiled,)))
    manager.run(EffectStage.AFTER_PARTICLE_TRANSFER, object(), {"frame": 1, "substep_index": 0})
    with pytest.raises(EffectExecutionError, match="bad launch") as captured:
        manager.run(
            EffectStage.AFTER_PARTICLE_TRANSFER,
            object(),
            {"frame": 1, "substep_index": 1},
        )
    stats = next(iter(manager.stats.values()))
    assert stats.calls == 1
    assert stats.elements == 3
    assert stats.submission_seconds == 0.01
    assert "bad launch" in stats.last_error
    assert manager.runtime_errors
    assert captured.value.runtime_error.instance_id == "stats"

    with pytest.raises(EffectExecutionError, match="bad launch"):
        manager.run(
            EffectStage.AFTER_PARTICLE_TRANSFER,
            object(),
            {"frame": 2, "substep_index": 0},
        )

    assert manager.swap_candidate(FakeCandidate(True, (replacement,))) is True
    manager.run(
        EffectStage.AFTER_PARTICLE_TRANSFER,
        object(),
        {"frame": 2, "substep_index": 0},
    )
    replacement_stats = next(iter(manager.stats.values()))
    assert replacement_instance.instance_id == "replacement"
    assert replacement_stats.calls == 1
    assert replacement_stats.elements == 2


def test_applied_signature_is_sha256_string_and_inventory_is_separate():
    _instance_value, compiled = _instance("signature")
    manager = EffectRuntimeManager(device="cpu")
    manager.swap_candidate(FakeCandidate(True, (compiled,)))
    assert isinstance(manager.applied_signature, str)
    assert len(manager.applied_signature) == 64
    assert manager.inventory[0]["instance_id"] == "signature"
