from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from fluidsim_testpkg.sim.effects.enums import (
    EffectCadence,
    EffectDomain,
    EffectStage,
    FieldAccess,
    ParameterKind,
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
    ParameterDefinitionV2,
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
    parameters=(),
    parameter_values=None,
):
    kernel = KernelDefinitionV2(
        kernel_id=f"kernel_{instance_id.replace('-', '_')}",
        label="Kernel",
        domain=domain,
        stage=stage,
        cadence=cadence,
        source="pass",
        fields=tuple(fields),
        parameter_keys=tuple(item.key for item in parameters),
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
        parameters=tuple(parameters),
        kernels=(kernel,),
    )
    instance = EffectInstanceV2(
        instance_id,
        definition,
        order=order,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_step=frame_step,
        parameter_values=parameter_values or {},
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


def test_runtime_bindable_parameter_update_preserves_compiled_runtime_state(monkeypatch):
    strength = ParameterDefinitionV2(
        key="strength",
        label="Strength",
        kind=ParameterKind.FLOAT,
        default=1.0,
        minimum=0.0,
        maximum=100.0,
        runtime_bindable=True,
    )
    instance, compiled = _instance(
        "hot-update",
        parameters=(strength,),
        parameter_values={"strength": 1.0},
    )
    seen = []

    def fake_launch(entry, _solver, _time_ctx, _device):
        seen.append(entry.instance.effective_parameter_values()["strength"])
        return 1, 0.0

    monkeypatch.setattr(
        "fluidsim_testpkg.sim.effects.runtime.launch_compiled_kernel",
        fake_launch,
    )
    manager = EffectRuntimeManager(device="cpu")
    manager.swap_candidate(FakeCandidate(True, (compiled,)))
    entry_before = manager.by_stage[EffectStage.AFTER_PARTICLE_TRANSFER][0]
    compiled_kernel_before = entry_before.compiled_kernel
    stats_before = manager.stats[entry_before.key]
    manager._once_per_frame[entry_before.key] = 12
    updated = replace(instance, parameter_values={"strength": 48.54})

    assert manager.update_runtime_bindings((updated,)) is True
    entry_after = manager.by_stage[EffectStage.AFTER_PARTICLE_TRANSFER][0]
    assert entry_after.compiled_kernel is compiled_kernel_before
    assert manager.stats[entry_after.key] is stats_before
    assert manager._once_per_frame[entry_after.key] == 12

    manager.run(
        EffectStage.AFTER_PARTICLE_TRANSFER,
        object(),
        {"frame": 13, "substep_index": 0},
    )
    assert seen == [48.54]


_DAMP_SOURCE = "particle_velocity[particle_index] = particle_velocity[particle_index] * factor"


def _compiled_particle_entry(device, *, access=FieldAccess.READ_WRITE_CURRENT,
                             expert=False):
    """Compile a real particle-velocity kernel and wrap it as a runtime entry."""
    from fluidsim_testpkg.sim.effects.compiler import compile_effect
    from fluidsim_testpkg.sim.effects.runtime import EffectRuntimeEntry

    factor = ParameterDefinitionV2("factor", "Factor", ParameterKind.FLOAT, 0.5)
    kernel = KernelDefinitionV2(
        kernel_id="damp", label="Damp", domain=EffectDomain.PARTICLES,
        stage=EffectStage.BEFORE_EMISSION, cadence=EffectCadence.EVERY_SUBSTEP,
        source=_DAMP_SOURCE,
        fields=(FieldBindingV2("particle_velocity", access),),
        parameter_keys=("factor",))
    definition = EffectDefinitionV2(
        effect_id="test.launch", version="1.0.0", name="Launch", summary="s",
        description="d", category="Tests", author="Tests",
        license="GPL-3.0-or-later", parameters=(factor,), kernels=(kernel,))
    instance = EffectInstanceV2(
        "launch-1", definition, safe_mode=not expert, expert_acknowledged=expert)
    compiled = compile_effect(instance, device=device)
    assert compiled.success, compiled.issues
    return EffectRuntimeEntry(instance, compiled, compiled.kernels[0], 0, 0)


def _particle_solver(device, *, live=5, capacity=8):
    from types import SimpleNamespace

    import numpy as np
    import warp as wp

    refreshes = []
    velocity = np.arange(capacity * 3, dtype=np.float32).reshape(capacity, 3)
    particles = SimpleNamespace(
        position=wp.zeros(capacity, dtype=wp.vec3, device=device),
        velocity=wp.array(velocity, dtype=wp.vec3, device=device),
        alive=wp.array(np.ones(capacity, dtype=np.int32), dtype=wp.int32, device=device),
        count=wp.array([live], dtype=wp.int32, device=device),
        capacity=capacity,
        attrs={},
        refresh_host_count=lambda: refreshes.append(True) or live,
    )
    solver = SimpleNamespace(particles=particles, settings=SimpleNamespace(voxel_size=0.1))
    return solver, velocity, refreshes


def _counting_synchronize(monkeypatch):
    import warp as wp

    syncs = []
    original = wp.synchronize_device

    def counting(device=None):
        syncs.append(device)
        return original(device)

    monkeypatch.setattr(wp, "synchronize_device", counting)
    return syncs


def test_safe_launch_skips_host_count_readback_and_device_sync(monkeypatch):
    # P13: every launch refreshed the particle count from the device (unused:
    # the launch spans capacity under the kernel's count guard) and blocked
    # on wp.synchronize_device.
    from fluidsim_testpkg.sim.effects.runtime import launch_compiled_kernel

    entry = _compiled_particle_entry("cpu")
    solver, velocity, refreshes = _particle_solver("cpu")
    syncs = _counting_synchronize(monkeypatch)

    elements, _duration = launch_compiled_kernel(entry, solver, {"frame": 1}, "cpu")

    assert (refreshes, syncs) == ([], [])
    assert elements == 8
    expected = velocity.copy()
    expected[:5] *= 0.5
    assert (solver.particles.velocity.numpy() == expected).all()


def test_expert_launch_still_synchronizes_so_faults_name_the_effect(monkeypatch):
    from fluidsim_testpkg.sim.effects.runtime import launch_compiled_kernel

    entry = _compiled_particle_entry(
        "cpu", access=FieldAccess.UNSAFE_ARBITRARY, expert=True)
    solver, _velocity, refreshes = _particle_solver("cpu")
    syncs = _counting_synchronize(monkeypatch)

    launch_compiled_kernel(entry, solver, {"frame": 1}, "cpu")

    assert refreshes == []
    assert syncs == ["cpu"]


def test_unsynchronized_launch_matches_synchronized_result_cpu(monkeypatch):
    _assert_sync_equivalence(monkeypatch, "cpu")


def test_unsynchronized_launch_matches_synchronized_result_cuda(monkeypatch, cuda_device):
    _assert_sync_equivalence(monkeypatch, cuda_device)


def _assert_sync_equivalence(monkeypatch, device):
    from fluidsim_testpkg.sim.effects import runtime

    entry = _compiled_particle_entry(device)
    results = []
    for force_sync in (True, False):
        monkeypatch.setattr(runtime, "_launch_needs_sync", lambda _entry, f=force_sync: f)
        solver, _velocity, _refreshes = _particle_solver(device, live=97, capacity=128)
        for _ in range(3):
            runtime.launch_compiled_kernel(entry, solver, {"frame": 1}, device)
        results.append(solver.particles.velocity.numpy())
    assert results[0].tobytes() == results[1].tobytes()


def test_runtime_binding_update_rejects_non_bindable_and_structural_changes():
    locked = ParameterDefinitionV2(
        key="locked",
        label="Locked",
        kind=ParameterKind.FLOAT,
        default=1.0,
        runtime_bindable=False,
    )
    instance, compiled = _instance(
        "locked",
        parameters=(locked,),
        parameter_values={"locked": 1.0},
    )
    manager = EffectRuntimeManager(device="cpu")
    manager.swap_candidate(FakeCandidate(True, (compiled,)))
    signature_before = manager.applied_signature
    entry_before = manager.by_stage[EffectStage.AFTER_PARTICLE_TRANSFER][0]

    assert manager.update_runtime_bindings((
        replace(instance, parameter_values={"locked": 2.0}),
    )) is False
    assert manager.update_runtime_bindings((replace(instance, order=1),)) is False
    assert manager.applied_signature == signature_before
    assert manager.by_stage[EffectStage.AFTER_PARTICLE_TRANSFER][0] is entry_before
