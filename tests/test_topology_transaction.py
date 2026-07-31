"""Behavioral contracts for transactional sparse topology and leaf-demand quality."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim.capacity_planner import (
    CapacityVerdict,
    DemandQuality,
    plan_initialization,
)
from fluidSim_warp_blender51.sim.fields import SparseGrid, TopologyCapacityError


def _grid(*, capacity=1):
    wp.init()
    return SparseGrid(
        origin=(0.0, 0.0, 0.0),
        size=(2.0, 2.0, 2.0),
        voxel_size=0.1,
        leaf_capacity=capacity,
        device="cpu",
    )


def test_overflow_preserves_live_leaf_table_coords_and_count():
    grid = _grid(capacity=1)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    assert grid.finalize_topology() == 1

    live_table = grid.leaf_table.numpy().copy()
    live_coords = grid.leaf_coords.numpy().copy()
    live_count = grid.num_leaves.numpy().copy()

    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (1.999, 1.999, 1.999))
    with pytest.raises(TopologyCapacityError) as caught:
        grid.finalize_topology()

    assert caught.value.required > caught.value.capacity == 1
    assert grid.active_leaf_count == 1
    np.testing.assert_array_equal(grid.leaf_table.numpy(), live_table)
    np.testing.assert_array_equal(grid.leaf_coords.numpy(), live_coords)
    np.testing.assert_array_equal(grid.num_leaves.numpy(), live_count)

    # A refused candidate is disposable; the next admitted transaction can
    # reuse its buffers without disturbing the still-live table first.
    grid.begin_topology()
    grid.mark_world_box((0.8, 0.0, 0.0), (0.9, 0.1, 0.1))
    assert grid.finalize_topology() == 1
    assert not np.array_equal(grid.leaf_table.numpy(), live_table)


def test_deferred_overflow_does_not_commit_candidate_topology():
    grid = _grid(capacity=1)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    assert grid.finalize_topology() == 1
    live_table = grid.leaf_table.numpy().copy()

    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (1.999, 1.999, 1.999))
    assert grid.finalize_topology(sync=False) == 1
    with pytest.raises(TopologyCapacityError):
        grid.refresh_topology_counts()
    np.testing.assert_array_equal(grid.leaf_table.numpy(), live_table)


def _settings():
    return SimpleNamespace(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.6, 1.6, 1.6),
        voxel_size=0.1,
        max_particles=1024,
        leaf_capacity=64,
        particles_per_voxel=8.0,
        topology_expand=0,
        max_whitewater=0,
        surface_enabled=False,
        surface_resolution_mode=0,
        surface_resolution_multiplier=1.0,
        surface_leaf_capacity_scale=1.0,
        surface_max_dense_dim=64,
        surface_voxel_size=0.0,
    )


def _resources():
    return {"gpu_total_bytes": 64 * 1024**3, "gpu_free_bytes": 64 * 1024**3}


def test_leaf_demand_propagates_upper_bound_quality():
    footprint = SimpleNamespace(
        stable_id="legacy-aabb",
        particle_demand=0,
        particle_quality=DemandQuality.EXACT,
        leaf_coords=((0, 0, 0),),
        leaf_quality=DemandQuality.UPPER_BOUND,
        leaf_reason="bounded AABB",
        staging_bytes=0,
    )
    plan = plan_initialization(_settings(), (footprint,), (), (), _resources())
    assert plan.verdict is CapacityVerdict.ADMITTED
    assert plan.leaf_demand.value == 1
    assert plan.leaf_demand.quality is DemandQuality.UPPER_BOUND


def test_unknown_leaf_demand_fails_closed():
    footprint = SimpleNamespace(
        stable_id="failed-source",
        particle_demand=0,
        particle_quality=DemandQuality.EXACT,
        leaf_coords=(),
        leaf_quality=DemandQuality.UNKNOWN,
        leaf_reason="source topology compile failed",
        staging_bytes=0,
    )
    plan = plan_initialization(_settings(), (footprint,), (), (), _resources())
    assert plan.verdict is CapacityVerdict.BLOCKED
    assert plan.leaf_demand.value is None
    assert plan.leaf_demand.quality is DemandQuality.UNKNOWN
    assert any(item.code == "LEAF_DEMAND_UNKNOWN" for item in plan.blockers)
