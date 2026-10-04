"""Trace regressions for active capacity and scene-owned plan transactions."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from settings_doubles import split_settings


@dataclass
class _Settings:
    device: str = "cuda:0"
    max_particles: int = 8
    auto_grow_particles: bool = False
    particle_growth_failure_policy: str = "DEFER"


def _bridge(monkeypatch):
    bpy = types.ModuleType("bpy")
    bpy.context = SimpleNamespace()
    bpy.app = SimpleNamespace(handlers=SimpleNamespace(frame_change_post=[]))
    bpy.path = SimpleNamespace(abspath=lambda value: value)
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    output_publication = types.ModuleType(
        "fluidSim_warp_blender51.blender.output_publication")
    output_publication.update_outputs = lambda *_args, **_kwargs: None
    output_publication.ensure_outputs = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.output_publication",
        output_publication)
    blender_package = importlib.import_module("fluidSim_warp_blender51.blender")
    monkeypatch.setattr(
        blender_package, "output_publication", output_publication, raising=False)
    sys.modules.pop("fluidSim_warp_blender51.blender.scene_bridge", None)
    bridge = importlib.import_module(
        "fluidSim_warp_blender51.blender.scene_bridge")
    from fluidSim_warp_blender51.sim import memory_safety

    monkeypatch.setattr(memory_safety, "vram_snapshot", lambda _device: {
        "gpu_total_bytes": 16 * 1024**3,
        "gpu_free_bytes": 12 * 1024**3,
    })
    monkeypatch.setattr(memory_safety, "release_cuda_memory", lambda _device: None)
    from fluidSim_warp_blender51.blender import output_publication, pending_settings
    monkeypatch.setattr(
        pending_settings, "capture_applied_state", lambda *_a, **_k: None)
    monkeypatch.setattr(
        output_publication, "update_outputs", lambda *_a, **_k: None)
    return bridge


def _active_reset(
        monkeypatch, *, authored, runtime, demand, auto_grow=False,
        prepare_error=None, memory_blocked_at=None, patch_prepare=True):
    bridge = _bridge(monkeypatch)
    scene = SimpleNamespace(
        frame_start=1,
        fluidsim=split_settings(SimpleNamespace(
            max_particles=authored,
            auto_grow_particles=auto_grow,
            particle_growth_failure_policy="DEFER",
            snippets=[],
            status="before",
        )),
    )
    released = []
    old_solver = SimpleNamespace(
        settings=SimpleNamespace(
            max_particles=runtime,
            auto_grow_particles=True,
            particle_growth_failure_policy="PAUSE",
        ),
        particles=SimpleNamespace(
            capacity=runtime, host_count=7, payload=object()),
        frame=23,
        stats=SimpleNamespace(marker="old-stats"),
        device="cuda:0",
        release=lambda: released.append("old"),
    )
    sess = bridge.Session()
    sess.solver = old_solver
    sess.scene_identity = bridge.scene_identity(scene)
    sess.start_frame = 4
    sess.last_frame = 23
    sess.frame_stats = [{"frame": 23}]
    sess.cache_written_start = 3
    sess.cache_written_end = 23
    sess.cache_settings_records = {23: "old-cache"}
    old_inputs = SimpleNamespace(clear=lambda: None)
    sess.scene_inputs = old_inputs

    from fluidSim_warp_blender51.blender import simulation_admission
    monkeypatch.setattr(
        simulation_admission, "build_settings",
        lambda _scene: _Settings(
            max_particles=authored,
            auto_grow_particles=auto_grow,
        ),
    )
    planned = []
    from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes
    expected_live_solver_bytes = estimate_solver_bytes(old_solver.settings)

    def plan_scene(
            _scene, settings, _resources, *,
            additional_transaction_bytes=0):
        assert additional_transaction_bytes == expected_live_solver_bytes
        planned.append(int(settings.max_particles))
        blockers = []
        if demand > int(settings.max_particles):
            blockers.append(SimpleNamespace(
                code="PARTICLE_CAPACITY",
                required=demand,
                available=int(settings.max_particles),
                message="particle capacity",
            ))
        if (memory_blocked_at is not None
                and int(settings.max_particles) >= memory_blocked_at):
            blockers.append(SimpleNamespace(
                code="MEMORY_ESTIMATE",
                required=2_000,
                available=1_000,
                message="memory estimate",
            ))
        plan = SimpleNamespace(
            blockers=tuple(blockers),
            scene_signature=f"capacity-{settings.max_particles}",
        )
        return SimpleNamespace(
            capacity_plan=plan,
            seeds=(),
            effects=(),
            scene_inputs=SimpleNamespace(clear=lambda: None),
        )

    monkeypatch.setattr(sess._admission(), "_plan_scene", plan_scene)
    prepared = []

    def prepare(_scene, settings, _snapshot, _profiler):
        if prepare_error is not None:
            raise prepare_error
        candidate = SimpleNamespace(
            settings=settings,
            particles=SimpleNamespace(capacity=int(settings.max_particles)),
            device="cuda:0",
            release=lambda: released.append("candidate"),
        )
        prepared.append(candidate)
        return candidate, ()

    if patch_prepare:
        monkeypatch.setattr(
            simulation_admission.SimulationAdmission,
            "_prepare_solver_replacement",
            lambda _owner, *args, **kwargs: prepare(*args, **kwargs),
        )
    return bridge, scene, sess, old_solver, released, planned, prepared


def test_reset_autogrow_off_raised_reserve_refuses_against_runtime_without_mutation(
        monkeypatch):
    bridge, scene, sess, old, released, planned, prepared = _active_reset(
        monkeypatch, authored=12, runtime=8, demand=10, auto_grow=False)
    authoritative = (
        sess.solver, old.particles.payload, old.frame, old.stats,
        sess.start_frame, sess.last_frame, sess.frame_stats,
        sess.cache_written_start, sess.cache_written_end,
        sess.cache_settings_records,
    )

    with pytest.raises(RuntimeError, match="required 10, available 8"):
        sess.reset(scene)

    assert planned == [8]
    assert prepared == []
    assert released == []
    assert (
        sess.solver, old.particles.payload, old.frame, old.stats,
        sess.start_frame, sess.last_frame, sess.frame_stats,
        sess.cache_written_start, sess.cache_written_end,
        sess.cache_settings_records,
    ) == authoritative
    assert old.settings.particle_growth_failure_policy == "PAUSE"
    assert sess.capacity_plan_for_scene(scene).blockers[0].available == 8


def test_reset_autogrow_off_lowered_reserve_uses_grown_runtime_without_shrink(
        monkeypatch):
    _bridge_mod, scene, sess, old, released, planned, prepared = _active_reset(
        monkeypatch, authored=4, runtime=12, demand=10, auto_grow=False)

    sess.reset(scene)

    assert planned == [12, 12]  # initial admission + pre-allocation VRAM recheck
    assert prepared and sess.solver is prepared[0]
    assert sess.solver.particles.capacity == 12
    assert scene.fluidsim.domain_object.fluidsim_domain.max_particles == 4
    assert released == ["old"]
    assert old.particles.capacity == 12


def test_reset_autogrow_rejects_memory_blocked_growth_candidate(monkeypatch):
    _bridge_mod, scene, sess, old, released, planned, prepared = _active_reset(
        monkeypatch,
        authored=8,
        runtime=8,
        demand=10,
        auto_grow=True,
        memory_blocked_at=9,
    )
    authoritative = (
        sess.solver, old.particles.payload, old.frame, old.stats,
        sess.start_frame, sess.last_frame, sess.frame_stats,
        sess.cache_written_start, sess.cache_written_end,
        sess.cache_settings_records,
    )

    with pytest.raises(RuntimeError, match="required 10, available 8"):
        sess.reset(scene)

    assert planned[0] == 8
    assert len(planned) == 2 and planned[1] > planned[0]
    assert prepared == []
    assert released == []
    assert (
        sess.solver, old.particles.payload, old.frame, old.stats,
        sess.start_frame, sess.last_frame, sess.frame_stats,
        sess.cache_written_start, sess.cache_written_end,
        sess.cache_settings_records,
    ) == authoritative
    assert sess.capacity_plan_for_scene(scene).blockers[0].code == (
        "PARTICLE_CAPACITY")


def test_reset_ungrown_peak_memory_refusal_preserves_live_solver_and_status(
        monkeypatch):
    _bridge_mod, scene, sess, old, released, planned, prepared = _active_reset(
        monkeypatch,
        authored=8,
        runtime=8,
        demand=7,
        auto_grow=False,
        memory_blocked_at=8,
    )

    with pytest.raises(RuntimeError, match=r"old-solver \+ candidate"):
        sess.reset(scene)

    assert planned == [8]
    assert prepared == []
    assert released == []
    assert sess.solver is old
    assert old.frame == 23
    assert sess.last_frame == 23
    assert scene.fluidsim.status == "Reset blocked — live solver preserved"
    assert "old-solver + candidate" in sess.current_error(scene)
    assert sess.capacity_plan_for_scene(scene).blockers[0].code == (
        "MEMORY_ESTIMATE")


def test_failed_empty_reset_replacement_preserves_authoritative_session(
        monkeypatch):
    _bridge_mod, scene, sess, old, released, planned, _prepared = _active_reset(
        monkeypatch,
        authored=8,
        runtime=8,
        demand=7,
        auto_grow=False,
        prepare_error=RuntimeError("candidate allocation failed"),
    )
    frame_stats = sess.frame_stats
    cache_records = sess.cache_settings_records

    with pytest.raises(RuntimeError, match="candidate allocation failed"):
        sess.reset(scene)

    assert planned == [8, 8]  # initial admission + pre-allocation VRAM recheck
    assert sess.solver is old
    assert sess.last_frame == 23
    assert sess.frame_stats is frame_stats
    assert sess.cache_settings_records is cache_records
    assert old.particles.host_count == 7
    assert released == []


def test_candidate_constructor_failure_cleans_candidate_pool_only(monkeypatch):
    bridge, scene, sess, old, released, _planned, _prepared = _active_reset(
        monkeypatch, authored=8, runtime=8, demand=7, auto_grow=False,
        patch_prepare=False)

    class Candidate:
        def __init__(self, _settings, profiler=None):
            raise RuntimeError("candidate allocation failed")

    solver_module = types.ModuleType("fluidSim_warp_blender51.sim.solver")
    solver_module.FluidSolver = Candidate
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.sim.solver", solver_module)
    from fluidSim_warp_blender51.sim import memory_safety

    cuda_releases = []
    monkeypatch.setattr(
        memory_safety, "release_cuda_memory",
        lambda device: cuda_releases.append(device))

    owner = sess._admission()
    settings = _Settings(max_particles=8)
    with pytest.raises(RuntimeError, match="candidate allocation failed"):
        owner._prepare_solver_replacement(
            scene, settings, SimpleNamespace(settings=settings), None)

    assert sess.solver is old
    assert released == []
    assert cuda_releases == ["cuda:0"]


def test_prepared_seed_capacity_mismatch_discards_candidate_not_live_solver(
        monkeypatch):
    bridge, scene, sess, old, released, _planned, _prepared = _active_reset(
        monkeypatch, authored=8, runtime=8, demand=7, auto_grow=False,
        patch_prepare=False)
    candidates = []

    class Candidate:
        device = "cuda:0"

        def __init__(self, settings, profiler=None):
            self.settings = settings
            self.particles = SimpleNamespace(capacity=8, host_count=0)
            self.sourcing = SimpleNamespace(last_apply_result={})
            candidates.append(self)

        def release(self):
            released.append("candidate")

        def set_custom_snippets(self, _snippets):
            return ()

    solver_module = types.ModuleType("fluidSim_warp_blender51.sim.solver")
    solver_module.FluidSolver = Candidate
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.sim.solver", solver_module)
    from fluidSim_warp_blender51.blender import sources

    def refuse(candidate_session, _seeds):
        candidate_session.solver.sourcing.last_apply_result = {
            "admitted": False,
            "required": 9,
            "available": 8,
        }
        return 0

    monkeypatch.setattr(sources, "commit_prepared_seeds", refuse)
    owner = sess._admission()
    monkeypatch.setattr(owner, "_configure_candidate", lambda *_a, **_k: ())
    settings = _Settings(max_particles=8)
    snapshot = SimpleNamespace(seeds=(object(),), settings=settings)

    with pytest.raises(RuntimeError, match="required 9, available 8"):
        owner._prepare_solver_replacement(
            scene, settings, snapshot, None)

    assert candidates
    assert sess.solver is old
    assert old.particles.host_count == 7
    assert released == ["candidate"]


def test_reinitialize_bypassed_memory_plan_retains_truth_after_candidate_failure(
        monkeypatch):
    bridge, scene, sess, old, released, _planned, _prepared = _active_reset(
        monkeypatch, authored=8, runtime=8, demand=7, auto_grow=False)
    scene.fluidsim.domain_object.fluidsim_domain.profiling_enabled = False
    scene.fluidsim.domain_object.fluidsim_domain.profiling_sync = False
    blocker = SimpleNamespace(
        code="MEMORY_ESTIMATE", required=2048, available=1024,
        message="memory estimate")
    plan = SimpleNamespace(
        blockers=(blocker,), estimated_bytes=2048, budget_bytes=1024,
        suggestions=(), scene_signature="reinit")
    snapshot = SimpleNamespace(capacity_plan=plan, seeds=())
    from fluidSim_warp_blender51.sim import memory_safety

    expected_live_solver_bytes = memory_safety.estimate_solver_bytes(old.settings)
    planned_residencies = []

    def plan_scene(
            _scene, _settings, _resources, *,
            additional_transaction_bytes=0):
        planned_residencies.append(additional_transaction_bytes)
        return snapshot

    monkeypatch.setattr(sess._admission(), "_plan_scene", plan_scene)
    from fluidSim_warp_blender51.blender import simulation_admission
    monkeypatch.setattr(
        simulation_admission.SimulationAdmission,
        "_prepare_solver_replacement",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("replacement allocation failed")),
    )

    with pytest.raises(RuntimeError, match="replacement allocation failed"):
        sess.initialize(scene, ignore_memory_safeguard=True)

    assert planned_residencies == [
        expected_live_solver_bytes, expected_live_solver_bytes]
    assert sess.solver is old
    assert old.frame == 23
    assert old.particles.host_count == 7
    assert sess.frame_stats == [{"frame": 23}]
    assert released == []
    assert sess.capacity_plan_for_scene(scene) is plan
    assert sess.capacity_plan_memory_bypassed is True


def test_capacity_plan_owner_signature_and_release_load_invalidation(monkeypatch):
    bridge = _bridge(monkeypatch)
    first = SimpleNamespace()
    second = SimpleNamespace()
    plan = SimpleNamespace(scene_signature="identical-content", suggestions=(1,))
    sess = bridge.Session()
    sess._set_capacity_plan(first, plan, memory_bypassed=True)
    sess.particle_capacity_notice_scene_identity = bridge.scene_identity(first)

    assert sess.capacity_plan_for_scene(first) is plan
    assert "|operation=INITIALIZE" in sess.capacity_plan_scene_signature
    sess.capacity_plan_operation = "RESET"
    assert sess.capacity_plan_for_scene(first) is None
    sess.capacity_plan_operation = "INITIALIZE"
    assert sess.capacity_plan_for_scene(second) is None
    assert sess.particle_capacity_notice_for_scene(second) == ""
    plan.scene_signature = "changed"
    assert sess.capacity_plan_for_scene(first) is None

    plan.scene_signature = "identical-content"
    sess._set_capacity_plan(first, plan, memory_bypassed=True)
    sess.release(first)
    assert sess.capacity_plan is None
    assert sess.capacity_plan_scene_identity is None
    assert sess.capacity_plan_scene_signature == ""
    assert sess.capacity_plan_operation is None
    assert sess.capacity_plan_memory_bypassed is False
    assert sess.particle_capacity_notice_for_scene(first) == ""

    global_session = bridge.session()
    global_session._set_capacity_plan(first, plan, memory_bypassed=True)
    from fluidSim_warp_blender51.blender import pending_settings
    pending_settings._on_load_pre("")
    assert global_session.capacity_plan is None
    assert global_session.capacity_plan_scene_identity is None
    assert global_session.capacity_plan_operation is None
    assert global_session.capacity_plan_memory_bypassed is False


def test_first_initialize_plans_with_exactly_zero_live_residency(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = SimpleNamespace(
        fluidsim=split_settings(SimpleNamespace(
            profiling_enabled=False,
            profiling_sync=False,
            snippets=[],
            status="before",
        )),
    )
    sess = bridge.Session()
    from fluidSim_warp_blender51.blender import simulation_admission
    monkeypatch.setattr(
        simulation_admission, "build_settings",
        lambda _scene: _Settings(max_particles=8))
    blocker = SimpleNamespace(
        code="PARTICLE_DEMAND_UNKNOWN", required=None, available=8,
        message="unknown demand")
    plan = SimpleNamespace(
        blockers=(blocker,), estimated_bytes=0, budget_bytes=1024,
        suggestions=(), scene_signature="first-init")
    snapshot = SimpleNamespace(capacity_plan=plan, seeds=())
    planned_models = []

    def plan_scene(
            _scene, settings, _resources, *,
            additional_transaction_bytes=0):
        planned_models.append((
            int(settings.max_particles), additional_transaction_bytes))
        return snapshot

    monkeypatch.setattr(sess._admission(), "_plan_scene", plan_scene)
    prepared = []
    monkeypatch.setattr(
        simulation_admission.SimulationAdmission,
        "_prepare_solver_replacement",
        lambda *_a, **_k: prepared.append(True))

    with pytest.raises(RuntimeError, match="particle demand is unknown"):
        sess.initialize(scene)

    assert planned_models == [(8, 0)]
    assert prepared == []
    assert sess.solver is None
    assert sess.capacity_plan_operation == "INITIALIZE"


def test_cross_scene_initialize_charges_existing_solver_residency(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene_a = SimpleNamespace(fluidsim=split_settings(SimpleNamespace()))
    scene_b = SimpleNamespace(
        fluidsim=split_settings(SimpleNamespace(
            profiling_enabled=False,
            profiling_sync=False,
            snippets=[],
            status="before",
        )),
    )
    old_settings = _Settings(max_particles=4_000_000)
    sess = bridge.Session()
    sess.solver = SimpleNamespace(
        settings=old_settings,
        particles=SimpleNamespace(capacity=4_000_000, host_count=0),
        device="cuda:0",
    )
    sess.scene_identity = bridge.scene_identity(scene_a)
    sess.scene_inputs = SimpleNamespace(clear=lambda: None)
    from fluidSim_warp_blender51.blender import simulation_admission
    monkeypatch.setattr(
        simulation_admission, "build_settings",
        lambda _scene: _Settings(max_particles=64))
    blocker = SimpleNamespace(
        code="PARTICLE_DEMAND_UNKNOWN", required=None, available=64,
        message="unknown demand")
    plan = SimpleNamespace(
        blockers=(blocker,), estimated_bytes=0, budget_bytes=1024,
        suggestions=(), scene_signature="cross-scene-init")
    seen_residencies = []

    def plan_scene(
            _scene, _settings, _resources, *,
            additional_transaction_bytes=0):
        seen_residencies.append(additional_transaction_bytes)
        return SimpleNamespace(capacity_plan=plan, seeds=())

    monkeypatch.setattr(sess._admission(), "_plan_scene", plan_scene)
    from fluidSim_warp_blender51.sim import memory_safety

    with pytest.raises(RuntimeError, match="particle demand is unknown"):
        sess.initialize(scene_b)

    expected = memory_safety.estimate_solver_bytes(old_settings)
    assert seen_residencies == [expected]
    assert sess.solver is not None
    assert sess.scene_identity == bridge.scene_identity(scene_a)
    assert sess.capacity_plan_operation == "INITIALIZE"


def test_owned_reset_refresh_uses_runtime_capacity_and_exact_residency(monkeypatch):
    _bridge_mod, scene, sess, old, _released, _planned, _prepared = _active_reset(
        monkeypatch, authored=4, runtime=12, demand=7, auto_grow=False)
    from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes

    expected_live_solver_bytes = estimate_solver_bytes(old.settings)
    calls = []
    original = SimpleNamespace(scene_signature="owned-refresh", suggestions=())
    plan = SimpleNamespace(scene_signature="owned-refresh", suggestions=())
    sess._set_capacity_plan(scene, original, operation="RESET")

    def plan_scene(
            _scene, settings, _resources, *,
            additional_transaction_bytes=0):
        calls.append((int(settings.max_particles), additional_transaction_bytes))
        return SimpleNamespace(capacity_plan=plan)

    monkeypatch.setattr(sess._admission(), "_plan_scene", plan_scene)

    expected_signature = sess.capacity_plan_signature_for_scene(scene)
    assert sess.refresh_capacity_plan(scene) is plan
    assert calls == [(12, expected_live_solver_bytes)]
    assert sess.capacity_plan_for_scene(scene) is plan
    assert sess.capacity_plan_operation == "RESET"
    assert sess.capacity_plan_signature_for_scene(scene) == expected_signature
    assert sess.capacity_plan_memory_bypassed is False


def test_owned_reinitialize_refresh_uses_authored_reserve_after_runtime_growth(
        monkeypatch):
    _bridge_mod, scene, sess, old, _released, _planned, _prepared = _active_reset(
        monkeypatch, authored=1024, runtime=65_536, demand=2048,
        auto_grow=False)
    from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes

    expected_live_solver_bytes = estimate_solver_bytes(old.settings)
    suggestion = SimpleNamespace(key="preserve_simulation")
    original = SimpleNamespace(
        scene_signature="capacity-1024", suggestions=(suggestion,),
        blockers=(SimpleNamespace(code="PARTICLE_CAPACITY"),),
    )
    sess._set_capacity_plan(scene, original, operation="REINITIALIZE")
    expected_signature = sess.capacity_plan_signature_for_scene(scene)
    calls = []

    def plan_scene(
            _scene, settings, _resources, *,
            additional_transaction_bytes=0):
        capacity = int(settings.max_particles)
        calls.append((capacity, additional_transaction_bytes))
        plan = SimpleNamespace(
            scene_signature=f"capacity-{capacity}",
            suggestions=(suggestion,) if capacity < 2048 else (),
            blockers=((SimpleNamespace(code="PARTICLE_CAPACITY"),)
                      if capacity < 2048 else ()),
        )
        return SimpleNamespace(capacity_plan=plan)

    monkeypatch.setattr(sess._admission(), "_plan_scene", plan_scene)

    refreshed = sess.refresh_capacity_plan(scene)

    assert calls == [(1024, expected_live_solver_bytes)]
    assert refreshed.blockers[0].code == "PARTICLE_CAPACITY"
    assert [item.key for item in refreshed.suggestions] == [
        "preserve_simulation"]
    assert sess.capacity_plan_operation == "REINITIALIZE"
    assert sess.capacity_plan_signature_for_scene(scene) == expected_signature


def test_reinitialize_memory_bypass_selects_grown_peak_plan_truthfully(monkeypatch):
    _bridge_mod, scene, sess, old, released, planned, prepared = _active_reset(
        monkeypatch,
        authored=8,
        runtime=8,
        demand=10,
        auto_grow=True,
        memory_blocked_at=9,
        prepare_error=RuntimeError("candidate allocation attempted"),
    )
    scene.fluidsim.domain_object.fluidsim_domain.profiling_enabled = False
    scene.fluidsim.domain_object.fluidsim_domain.profiling_sync = False

    with pytest.raises(RuntimeError, match="candidate allocation attempted"):
        sess.initialize(scene, ignore_memory_safeguard=True)

    assert len(planned) == 3
    assert planned[0] == 8
    assert planned[1] == planned[2] > planned[0]
    assert prepared == []
    assert released == []
    assert sess.solver is old
    assert sess.capacity_plan_for_scene(scene).blockers[0].code == (
        "MEMORY_ESTIMATE")
    assert sess.capacity_plan_memory_bypassed is True


def test_capacity_notice_uses_live_pool_after_growth_and_replacement(monkeypatch):
    bridge = _bridge(monkeypatch)
    scene = SimpleNamespace(fluidsim=split_settings(SimpleNamespace(max_particles=4)))
    sess = bridge.Session()
    sess.scene_identity = bridge.scene_identity(scene)
    sess.solver = SimpleNamespace(particles=SimpleNamespace(capacity=8))
    sess.particle_capacity_notice_scene_identity = bridge.scene_identity(scene)

    assert "Active pool 8; Initial Reserve 4" in (
        sess.particle_capacity_notice_for_scene(scene))

    # Playback growth mutates the authoritative pool in place.
    sess.solver.particles.capacity = 65_536
    assert "Active pool 65,536; Initial Reserve 4" in (
        sess.particle_capacity_notice_for_scene(scene))

    # A successful reset/reinitialize swaps in another authoritative solver.
    sess.solver = SimpleNamespace(particles=SimpleNamespace(capacity=12))
    assert "Active pool 12; Initial Reserve 4" in (
        sess.particle_capacity_notice_for_scene(scene))


def test_capacity_policy_edit_does_not_discard_another_scenes_plan(monkeypatch):
    bridge = _bridge(monkeypatch)
    owner = SimpleNamespace()
    edited = SimpleNamespace(fluidsim=split_settings(SimpleNamespace(
        max_particles=4,
        auto_grow_particles=True,
        particle_growth_failure_policy="DEFER",
        status="",
    )))
    plan = SimpleNamespace(scene_signature="owner", suggestions=())
    sess = bridge.session()
    sess._set_capacity_plan(owner, plan)
    from fluidSim_warp_blender51.blender import pending_settings
    monkeypatch.setattr(pending_settings, "_tag_view3d_redraw", lambda: None)

    pending_settings.update_particle_capacity_policy(edited)

    assert sess.capacity_plan_for_scene(owner) is plan
    assert "Initial Reserve 4" in edited.fluidsim.status


def test_reinitialize_capacity_preflight_refusal_retains_active_solver(monkeypatch):
    bridge, scene, sess, old, released, _planned, _prepared = _active_reset(
        monkeypatch, authored=8, runtime=12, demand=7, auto_grow=False)
    scene.fluidsim.domain_object.fluidsim_domain.profiling_enabled = False
    scene.fluidsim.domain_object.fluidsim_domain.profiling_sync = False
    blocker = SimpleNamespace(
        code="PARTICLE_CAPACITY", required=9, available=8,
        message="particle capacity")
    plan = SimpleNamespace(
        blockers=(blocker,), estimated_bytes=0, budget_bytes=1024,
        suggestions=(), scene_signature="blocked-reinit")
    snapshot = SimpleNamespace(capacity_plan=plan, seeds=())
    from fluidSim_warp_blender51.sim import memory_safety

    expected_live_solver_bytes = memory_safety.estimate_solver_bytes(old.settings)
    planned_residencies = []

    def plan_scene(
            _scene, _settings, _resources, *,
            additional_transaction_bytes=0):
        planned_residencies.append(additional_transaction_bytes)
        return snapshot

    monkeypatch.setattr(sess._admission(), "_plan_scene", plan_scene)
    prepared = []
    from fluidSim_warp_blender51.blender import simulation_admission
    monkeypatch.setattr(
        simulation_admission.SimulationAdmission,
        "_prepare_solver_replacement",
        lambda *_a, **_k: prepared.append(True))

    cuda_releases = []
    monkeypatch.setattr(
        memory_safety, "release_cuda_memory",
        lambda device: cuda_releases.append(device))

    with pytest.raises(RuntimeError, match="required 9, available 8"):
        sess.initialize(scene)

    assert planned_residencies == [expected_live_solver_bytes]
    assert sess.solver is old
    assert old.frame == 23
    assert old.particles.host_count == 7
    assert sess.frame_stats == [{"frame": 23}]
    assert released == []
    assert prepared == []
    assert cuda_releases == []
    assert sess.capacity_plan_for_scene(scene) is plan
    assert sess.capacity_plan_operation == "REINITIALIZE"
