"""Regression contracts for pre-allocation GPU memory recovery."""

import sys
from types import SimpleNamespace
from pathlib import Path

from fluidSim_warp_blender51.sim.memory_safety import (
    SIM_BYTES_PER_VOXEL,
    WHITEWATER_BYTES,
    WHITEWATER_EMISSION_BYTES_PER_LIQUID,
    assess_configuration,
    estimate_solver_bytes,
    is_memory_capacity_error,
    recommend_safe_settings,
    safe_memory_budget,
)


def test_memory_estimate_includes_compact_pcg_and_whitewater_free_list():
    assert SIM_BYTES_PER_VOXEL == 356
    assert WHITEWATER_BYTES == 76
    assert WHITEWATER_EMISSION_BYTES_PER_LIQUID == 8


def _risky_settings():
    return SimpleNamespace(
        leaf_capacity=262_144,
        max_particles=8_000_000,
        max_whitewater=4_000_000,
        voxel_size=0.02,
        particles_per_voxel=16.0,
        topology_expand=3,
        domain_min=(-6.0, -2.0, 0.0),
        domain_max=(6.0, 2.0, 6.0),
        surface_resolution_mode=1,
        surface_resolution_multiplier=4.0,
        surface_leaf_capacity_scale=64.0,
        surface_max_dense_dim=2048,
    )


def test_16_gib_card_can_use_11_6_gib_when_14_gib_is_free():
    gib = 1024**3

    assert safe_memory_budget(16 * gib, 14 * gib) >= int(11.6 * gib)


def test_memory_estimate_flags_capacity_that_exceeds_budget_before_allocation():
    settings = _risky_settings()
    budget = 8 * 1024**3

    assert estimate_solver_bytes(settings) > budget


def test_recommended_settings_fit_budget_and_remove_secondary_surface_grid():
    settings = _risky_settings()
    budget = 8 * 1024**3

    recommended = recommend_safe_settings(settings, budget)

    assert recommended["voxel_size"] > settings.voxel_size
    assert "leaf_capacity" not in recommended
    assert recommended["surface_resolution_mode"] == 0
    assert recommended["surface_leaf_capacity_scale"] == 1.0
    assert recommended["surface_max_dense_dim"] <= 256
    safe = SimpleNamespace(**{**vars(settings), **recommended})
    assert estimate_solver_bytes(safe) <= budget


def test_memory_errors_are_routed_to_recovery_ui():
    assert is_memory_capacity_error("FluidSim: leaf capacity exceeded (9 > 8)")
    assert is_memory_capacity_error("CUDA error 2: out of memory")
    assert is_memory_capacity_error("FluidSim memory safeguard stopped initialization")
    assert not is_memory_capacity_error("pressure solver stalled")


def test_high_machine_load_recommendation_adjusts_all_major_cost_drivers():
    settings = _risky_settings()
    machine = {
        "gpu_utilization": 96.0,
        "cpu_utilization": 92.0,
        "ram_utilization": 90.0,
    }

    recommended = recommend_safe_settings(settings, 8 * 1024**3, machine)

    assert recommended["voxel_size"] > settings.voxel_size
    assert recommended["particles_per_voxel"] < settings.particles_per_voxel
    assert recommended["topology_expand"] < settings.topology_expand
    assert recommended["max_particles"] < settings.max_particles
    assert recommended["max_whitewater"] < settings.max_whitewater


def test_comfortable_configuration_is_left_unchanged():
    settings = SimpleNamespace(
        leaf_capacity=4096,
        max_particles=250_000,
        max_whitewater=100_000,
        voxel_size=0.1,
        particles_per_voxel=8.0,
        topology_expand=1,
        domain_min=(-2.0, -2.0, 0.0),
        domain_max=(2.0, 2.0, 4.0),
        surface_resolution_mode=0,
        surface_resolution_multiplier=2.0,
        surface_leaf_capacity_scale=1.0,
        surface_max_dense_dim=256,
    )

    assert (
        recommend_safe_settings(
            settings,
            8 * 1024**3,
            {
                "gpu_utilization": 20.0,
                "cpu_utilization": 20.0,
                "ram_utilization": 30.0,
            },
        )
        == {}
    )


def test_assessment_stops_before_allocation_and_returns_recovery_plan():
    settings = _risky_settings()
    resources = {
        "gpu_total_bytes": 16 * 1024**3,
        "gpu_free_bytes": 10 * 1024**3,
        "gpu_utilization": 70.0,
        "cpu_utilization": 40.0,
        "ram_utilization": 50.0,
    }

    assessment = assess_configuration(settings, resources)

    assert assessment["safe"] is False
    assert assessment["estimated_bytes"] > assessment["budget_bytes"]
    assert assessment["recommended"]


def test_capacity_recovery_ui_uses_scene_properties_and_verified_plan_flow():
    root = Path(__file__).parents[1] / "fluidSim_warp_blender51"
    operators = (root / "operator_domains" / "session.py").read_text(encoding="utf-8")
    panels = (root / "panels.py").read_text()
    scene_ui = (root / "blender" / "ui_scene.py").read_text()

    assert 'bl_idname = "fluidsim.apply_safe_settings"' in operators
    assert 'operator("fluidsim.apply_safe_settings"' not in panels
    assert '"fluidsim.apply_capacity_suggestion"' in scene_ui
    assert "capacity_plan_for_scene" in scene_ui
    assert "is_memory_capacity_error" not in panels
    assert "is_memory_capacity_error" not in scene_ui


def test_production_gate_includes_real_blender_memory_recovery_lane():
    gate = (Path(__file__).parents[1] / "tools" / "run_production_gate.py").read_text()
    reference = "tests/blender_memory_safety_gate.py"

    # Limitation: this source-text check proves a live-looking reference, not
    # that the production gate actually executes Blender successfully.
    assert reference in gate
    assert any(
        reference in line and not line.lstrip().startswith("#")
        for line in gate.splitlines()
    )


def test_estimate_caps_leaf_allocation_to_domain_cells():
    settings = _risky_settings()
    settings.leaf_capacity = 1_000_000
    settings.domain_min = (0.0, 0.0, 0.0)
    settings.domain_max = (0.8, 0.8, 0.8)
    settings.voxel_size = 0.1
    settings.max_particles = 0
    settings.max_whitewater = 0
    settings.surface_resolution_mode = 0

    one_leaf = SimpleNamespace(**vars(settings))
    one_leaf.leaf_capacity = 1
    assert estimate_solver_bytes(settings) == estimate_solver_bytes(one_leaf)


def test_safe_settings_uses_voxel_size_instead_of_starving_leaf_capacity():
    settings = _risky_settings()
    recommended = recommend_safe_settings(settings, 3 * 1024**3)

    assert recommended["voxel_size"] > settings.voxel_size
    assert "leaf_capacity" not in recommended


def test_surface_dense_memory_is_counted_in_simulation_mode():
    settings = _risky_settings()
    settings.surface_resolution_mode = 0
    settings.surface_max_dense_dim = 512
    with_surface = estimate_solver_bytes(settings)
    settings.surface_enabled = False

    assert with_surface > estimate_solver_bytes(settings)


def test_disabled_surface_does_not_allocate_secondary_grid_estimate():
    settings = _risky_settings()
    settings.surface_enabled = False
    secondary_mode = estimate_solver_bytes(settings)
    settings.surface_resolution_mode = 0

    assert secondary_mode == estimate_solver_bytes(settings)


def test_solver_release_breaks_gpu_owning_backreferences():
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    solver = FluidSolver.__new__(FluidSolver)
    extractor = SimpleNamespace(reset_buffers=lambda: None)
    surface = SimpleNamespace(
        solver=solver, extractor=extractor, render_extractor=extractor
    )
    host_frame = SimpleNamespace(solver=solver)
    sourcing = SimpleNamespace(owner=solver)
    solver.surface = surface
    solver.sourcing = sourcing
    solver._host_frame_cache = host_frame

    solver.release()

    assert surface.solver is None
    assert sourcing.owner is None
    assert host_frame.solver is None
    # Dropped, not nulled: _render_surface_voxel_size (solver.py:795) and
    # io.cache._frame_payload (cache.py:123) both gate on hasattr, so a None
    # left in place turns the released-state guard into an AttributeError.
    assert not hasattr(solver, "surface")
    assert solver.sourcing is None
    assert solver._host_frame_cache is None


def test_cuda_release_collects_cycles_and_flushes_pool_before_restoring_threshold(
    monkeypatch,
):
    from fluidSim_warp_blender51.sim import memory_safety

    calls = []
    device = SimpleNamespace(is_cuda=True, is_mempool_supported=True)
    fake_warp = SimpleNamespace(
        get_device=lambda name: calls.append(("device", name)) or device,
        get_mempool_release_threshold=lambda dev: calls.append(("get", dev)) or 123,
        set_mempool_release_threshold=lambda dev, value: calls.append(
            ("set", dev, value)
        ),
        synchronize_device=lambda dev: calls.append(("sync", dev)),
    )
    monkeypatch.setattr(memory_safety.gc, "collect", lambda: calls.append(("gc",)))
    monkeypatch.setitem(sys.modules, "warp", fake_warp)

    memory_safety.release_cuda_memory("cuda:0")

    assert calls == [
        ("gc",),
        ("device", "cuda:0"),
        ("get", device),
        ("set", device, 0),
        ("sync", device),
        ("set", device, 123),
    ]
