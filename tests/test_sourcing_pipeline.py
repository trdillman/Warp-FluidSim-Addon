"""Interface checks for the stateful fluid sourcing Module."""

from types import SimpleNamespace

import numpy as np

from fluidSim_warp_blender51.sim import sourcing
from fluidSim_warp_blender51.sim.sourcing import (
    HeightSourceDesc,
    SourcePipeline,
    SourceShapeDesc,
)


class _Particles:
    def __init__(self, host_count=0, capacity=8, refreshed_count=1):
        self.host_count = host_count
        self.capacity = capacity
        self.refreshed_count = refreshed_count
        self.refresh_calls = 0
        self.position = "position"
        self.velocity = "velocity"
        self.time_offset = "time_offset"
        self.alive = "alive"
        self.count = "count"

    def attr(self, name):
        return f"attr:{name}"

    def refresh_host_count(self):
        self.refresh_calls += 1
        return self.refreshed_count


def _owner(*, host_count=0, capacity=8, refreshed_count=1):
    return SimpleNamespace(
        device="cuda:0",
        grid=SimpleNamespace(
            leaf_coords="leaf_coords",
            num_leaves="num_leaves",
            origin="origin",
        ),
        settings=SimpleNamespace(
            voxel_size=0.1,
            domain_min=(-1.0, -2.0, 0.0),
            domain_max=(1.0, 2.0, 3.0),
        ),
        particles=_Particles(host_count, capacity, refreshed_count),
        voxel_type=SimpleNamespace(data="voxel_type"),
        collision_sdf=SimpleNamespace(data="collision_sdf"),
        collision_vel="collision_vel",
        vel=SimpleNamespace(u="u", v="v", w="w"),
        temperature=SimpleNamespace(data="temperature"),
        viscosity_field=SimpleNamespace(data="viscosity"),
        voxel_counts=SimpleNamespace(data="voxel_counts"),
        _nvox=512,
        _host_frame_cache=object(),
    )


def _record_warp(monkeypatch):
    arrays = []
    launches = []

    monkeypatch.setattr(
        sourcing.wp,
        "zeros",
        lambda shape, **kwargs: ("zeros", shape, kwargs),
    )

    def array(values, **kwargs):
        result = ("array", len(values), len(arrays))
        arrays.append((values, kwargs, result))
        return result

    monkeypatch.setattr(sourcing.wp, "array", array)
    monkeypatch.setattr(
        sourcing.wp,
        "launch",
        lambda kernel, **kwargs: launches.append((kernel, kwargs)),
    )
    monkeypatch.setattr(
        SourceShapeDesc,
        "to_struct",
        lambda self: ("shape", self.enabled, self.shape),
    )
    return arrays, launches


def test_apply_owns_launch_order_cache_and_full_capacity_signal(monkeypatch):
    arrays, launches = _record_warp(monkeypatch)
    owner = _owner(host_count=8, capacity=8)
    pipeline = SourcePipeline(owner)
    pipeline.set_sources([SourceShapeDesc(mode=1)])
    pipeline.set_height_source(HeightSourceDesc(
        enabled=True,
        base_height=0.75,
        emit_particles=False,
    ))

    first_seed = pipeline._emit_seed
    assert pipeline.apply(1.0 / 24.0) is False
    assert pipeline.apply(1.0 / 24.0) is False

    assert [kernel for kernel, _ in launches[:3]] == [
        sourcing.k_source_shapes,
        sourcing.k_source_height,
        sourcing.k_source_resolve_type,
    ]
    assert [kernel for kernel, _ in launches[3:]] == [
        sourcing.k_source_shapes,
        sourcing.k_source_height,
        sourcing.k_source_resolve_type,
    ]
    assert len(arrays) == 1
    assert owner.particles.refresh_calls == 0
    expected_seed = (first_seed * 1664525 + 1013904223) & 0x7FFFFFFF
    expected_seed = (expected_seed * 1664525 + 1013904223) & 0x7FFFFFFF
    assert pipeline._emit_seed == expected_seed


def test_height_emission_reuses_cached_shapes_and_refreshes_after_growth(monkeypatch):
    arrays, launches = _record_warp(monkeypatch)
    owner = _owner()
    pipeline = SourcePipeline(owner)
    pipeline.set_height_source(HeightSourceDesc(
        enabled=True,
        base_height=1.0,
        velocity=(0.0, 0.0, 2.0),
        emit_particles=True,
        emit_rate=6.0,
    ))

    assert pipeline.apply(0.01) is True
    assert pipeline.apply(0.01) is True

    assert [kernel for kernel, _ in launches[:3]] == [
        sourcing.k_source_height,
        sourcing.k_source_resolve_type,
        sourcing.k_source_shapes_particles,
    ]
    assert len(arrays) == 2  # disabled field placeholder + derived height box
    assert owner.particles.refresh_calls == 2


def test_inject_particles_owns_upload_launch_and_count_refresh(monkeypatch):
    arrays, launches = _record_warp(monkeypatch)
    owner = _owner(refreshed_count=2)
    pipeline = SourcePipeline(owner)

    result = pipeline.inject_particles(
        np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        velocity=(1.0, 0.0, 0.0),
        temperature=2.0,
    )

    assert result == 2
    assert len(arrays) == 2
    assert [kernel for kernel, _ in launches] == [sourcing.k_source_particles]
    assert launches[0][1]["dim"] == 2
    assert launches[0][1]["inputs"][-1] == pipeline._first_slot_dev
    assert owner.particles.refresh_calls == 1
    assert owner._host_frame_cache is None

    assert pipeline.inject_particles([]) == 0
    assert len(launches) == 1
    assert owner.particles.refresh_calls == 1
