"""GPU-first fixed-N surface output contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp

from fluidSim_warp_blender51.sim.fields import SparseGrid
from fluidSim_warp_blender51.sim.frame_output import HostFrame
from fluidSim_warp_blender51.sim.species import k_species_surface_color
from fluidSim_warp_blender51.sim.surface import (
    DeviceSurfaceResult,
    HostSurfaceResult,
    SurfaceExtractor,
    SurfacePipeline,
    k_surface_vertices_to_world,
)


DEVICE = "cuda:0"


def test_device_surface_result_materializes_geometry_and_colors_together(monkeypatch):
    syncs = []
    monkeypatch.setattr(wp, "synchronize_device", lambda device: syncs.append(str(device)))
    result = DeviceSurfaceResult(
        vertices_world=wp.array([(1.0, 2.0, 3.0)], dtype=wp.vec3, device=DEVICE),
        triangles=wp.array([0, 0, 0], dtype=wp.int32, device=DEVICE),
        vertex_count=1,
        triangle_count=1,
        colors=wp.array([(0.2, 0.3, 0.4, 1.0)], dtype=wp.vec4, device=DEVICE),
        device=DEVICE,
    )

    host = result.to_host()

    assert syncs == [DEVICE]
    np.testing.assert_array_equal(host.vertices, ((1.0, 2.0, 3.0),))
    np.testing.assert_array_equal(host.triangles, ((0, 0, 0),))
    np.testing.assert_allclose(host.colors, ((0.2, 0.3, 0.4, 1.0),))


def test_surface_world_transform_runs_on_device():
    vertices = wp.array([(0.0, 1.0, 2.0)], dtype=wp.vec3, device=DEVICE)
    wp.launch(
        k_surface_vertices_to_world,
        dim=1,
        inputs=[vertices, wp.vec3(10.0, 20.0, 30.0), 0.5, 1],
        device=DEVICE,
    )
    np.testing.assert_allclose(vertices.numpy(), ((10.0, 20.5, 31.0),))


def test_gpu_surface_color_matches_hand_derived_beer_lambert_reference():
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(8.0, 8.0, 8.0),
        voxel_size=1.0, leaf_capacity=1, device=DEVICE)
    marker = wp.array([(0.5, 0.5, 0.5)], dtype=wp.vec3, device=DEVICE)
    marker_count = wp.array([1], dtype=wp.int32, device=DEVICE)
    grid.begin_topology()
    grid.mark_points(marker, marker_count, 1)
    grid.finalize_topology()
    volume = wp.ones(grid.voxel_capacity, dtype=float, device=DEVICE)
    mass = wp.full(
        grid.voxel_capacity, wp.vec4(0.5, 0.5, 0.0, 0.0),
        dtype=wp.vec4, device=DEVICE)
    colors = wp.empty(1, dtype=wp.vec4, device=DEVICE)

    wp.launch(k_species_surface_color, dim=1, inputs=[
        marker, 1, grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
        grid.origin, grid.voxel_size, volume, mass,
        wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), 1.0, colors,
    ], device=DEVICE)

    np.testing.assert_allclose(
        colors.numpy(), [(np.exp(-0.5), np.exp(-0.5), 1.0, 1.0)],
        rtol=1.0e-6)


def test_live_clear_dye_uses_paired_gpu_surface_color_without_particle_readback():
    from fluidSim_warp_blender51.blender.output_publication import prepare_live_payload

    def forbidden(*_args, **_kwargs):
        raise AssertionError("live Clear Dye read particle positions or concentrations")

    result = HostSurfaceResult(
        vertices=np.array([(0.0, 0.0, 0.0)], dtype=np.float32),
        triangles=np.zeros((0, 3), dtype=np.int32),
        colors=np.array([(0.25, 0.5, 0.75, 1.0)], dtype=np.float32),
    )
    solver = SimpleNamespace(
        settings=SimpleNamespace(voxel_size=0.5),
        species=SimpleNamespace(),
    )
    host = SimpleNamespace(
        solver=solver,
        positions=property(forbidden),
        species_concentrations=forbidden,
        positions_strided=forbidden,
        whitewater=np.zeros((0, 3), dtype=np.float32),
        surface_result=lambda *, species_colors=False: result,
        output_attributes=lambda *_args, **_kwargs: {},
        _species_enabled=lambda: True,
    )
    fs = SimpleNamespace(
        show_particles=False,
        show_surface=True,
        show_whitewater=False,
        whitewater_enabled=False,
        particle_display_fraction=1.0,
        particle_display_scale=1.0,
        whitewater_display_scale=1.0,
        particle_attribute_mode="NONE",
        surface_attribute_mode="CLEAR_DYE",
        whitewater_attribute_mode="NONE",
        surface_enabled=True,
    )

    payload = prepare_live_payload(
        SimpleNamespace(fluidsim=fs), host, publish_surface=True)

    assert set(payload.surface_attributes) == {"color"}
    np.testing.assert_array_equal(payload.surface_attributes["color"], result.colors)


def test_surface_cadence_cache_reuses_the_complete_geometry_color_pair():
    calls = []
    first = HostSurfaceResult(
        vertices=np.array([(1.0, 0.0, 0.0)], dtype=np.float32),
        triangles=np.zeros((0, 3), dtype=np.int32),
        colors=np.array([(0.1, 0.2, 0.3, 1.0)], dtype=np.float32),
    )
    solver = SimpleNamespace(
        frame=7,
        surface_mesh=lambda **kwargs: calls.append(kwargs) or first,
    )
    host = HostFrame(solver)

    result_a = host.surface_result(species_colors=True)
    result_b = host.surface_result(species_colors=True)

    assert result_a is result_b
    assert calls == [{"species_colors": True}]
    np.testing.assert_array_equal(result_b.vertices, first.vertices)
    np.testing.assert_array_equal(result_b.colors, first.colors)


def test_outstanding_device_result_is_stable_after_marching_cubes_reuse():
    extractor = object.__new__(SurfaceExtractor)
    extractor.device = DEVICE
    extractor._mc = SimpleNamespace(
        verts=wp.array([(1.0, 2.0, 3.0)], dtype=wp.vec3, device=DEVICE),
        indices=wp.array([0, 0, 0], dtype=wp.int32, device=DEVICE),
    )
    first = extractor._lease_marching_cubes_result(
        offset=wp.vec3(0.0), voxel_size=1.0, color_fn=None)

    extractor._mc.verts.fill_(9.0)
    extractor._mc.indices.fill_(7)
    extractor._lease_marching_cubes_result(
        offset=wp.vec3(0.0), voxel_size=1.0, color_fn=None)

    host = first.to_host()
    np.testing.assert_array_equal(host.vertices, ((1.0, 2.0, 3.0),))
    np.testing.assert_array_equal(host.triangles, ((0, 0, 0),))


def test_surface_pipeline_routes_colors_through_its_decoupled_render_grid():
    render_grid = object()
    simulation_grid = object()
    routed = []
    host_result = HostSurfaceResult(
        np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
        np.zeros((0, 4), np.float32))
    species = SimpleNamespace(
        derive_surface_colors=lambda vertices, count, *, grid: (
            routed.append(grid)
            or wp.empty(count, dtype=wp.vec4, device=DEVICE)))
    extractor = SimpleNamespace(
        extract_device=lambda _sdf, *, color_fn: (
            color_fn(wp.empty(0, dtype=wp.vec3, device=DEVICE), 0),
            SimpleNamespace(to_host=lambda: host_result),
        )[1])
    pipeline = object.__new__(SurfacePipeline)
    pipeline.solver = SimpleNamespace(
        settings=SimpleNamespace(surface_enabled=False),
        species=species, grid=simulation_grid)
    pipeline.grid = render_grid
    pipeline.render_sdf = object()
    pipeline.render_extractor = extractor
    pipeline.mesh_cache = None
    pipeline.mesh_key = None
    pipeline.build_key = lambda: "frame-9"

    pipeline.mesh(species_colors=True)

    assert routed == [render_grid]
