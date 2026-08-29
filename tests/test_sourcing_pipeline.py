"""Interface checks for the stateful fluid sourcing Module."""

from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.sim import source_kernels, sourcing

# Import before monkeypatching Warp constructors in host-transaction tests.
from fluidSim_warp_blender51.sim.source_regions import SourceCompiler  # noqa: F401
from fluidSim_warp_blender51.sim.sourcing import (
    HeightSourceDesc,
    SourceEmissionMode,
    SourcePipeline,
    SourceShapeDesc,
)


@sourcing.wp.kernel
def _sample_shape_local_field(
    shape: int,
    points: sourcing.wp.array(dtype=sourcing.wp.vec3),
    values: sourcing.wp.array(dtype=float),
):
    tid = sourcing.wp.tid()
    values[tid] = sourcing.shape_local_sdf(
        shape,
        points[tid],
        sourcing.wp.vec3(1.0, 1.0, 1.0),
        1.0,
        1.0,
    )


@sourcing.wp.kernel
def _sample_shape_world_field(
    shape: sourcing.SourceShape,
    points: sourcing.wp.array(dtype=sourcing.wp.vec3),
    values: sourcing.wp.array(dtype=float),
):
    tid = sourcing.wp.tid()
    values[tid] = sourcing.shape_sdf_world(shape, points[tid])


@sourcing.wp.kernel
def _sample_degenerate_shape_field(
    shape: int,
    radius: float,
    height: float,
    points: sourcing.wp.array(dtype=sourcing.wp.vec3),
    values: sourcing.wp.array(dtype=float),
):
    tid = sourcing.wp.tid()
    values[tid] = sourcing.shape_local_sdf(
        shape, points[tid], sourcing.wp.vec3(1.0, 1.0, 1.0), radius, height
    )


@sourcing.wp.kernel
def _sample_watertight_mesh_inside(
    mesh_id: sourcing.wp.uint64,
    points: sourcing.wp.array(dtype=sourcing.wp.vec3),
    values: sourcing.wp.array(dtype=sourcing.wp.int32),
):
    tid = sourcing.wp.tid()
    if sourcing._watertight_mesh_contains(mesh_id, points[tid], 10.0):
        values[tid] = 1
    else:
        values[tid] = 0


@pytest.mark.parametrize("shape", (3, 4), ids=("pyramid", "cone"))
def test_cone_and_pyramid_exact_surface_sign_and_feature_distances(shape):
    side_point = (0.0, 0.5, 0.0) if shape == 3 else (0.5, 0.0, 0.0)
    points = np.asarray(
        [
            (0.0, 0.0, 1.0),  # apex
            (0.0, 0.0, -1.0),  # base
            side_point,  # sloped face
            (0.0, 0.0, 0.0),  # interior
            (0.0, 0.0, -1.25),  # below base
            (0.0, 0.0, -0.75),  # inside above base
            (1.0, 1.0, -1.0) if shape == 3 else (1.0, 0.0, -1.0),
        ],
        dtype=np.float32,
    )
    device = sourcing.wp.get_preferred_device()
    values = sourcing.wp.empty(len(points), dtype=float, device=device)
    sourcing.wp.launch(
        _sample_shape_local_field,
        dim=len(points),
        inputs=[
            shape,
            sourcing.wp.array(points, dtype=sourcing.wp.vec3, device=device),
            values,
        ],
        device=device,
    )

    sampled = values.numpy()
    np.testing.assert_allclose(sampled[:3], 0.0, atol=1.0e-6)
    assert sampled[3] < 0.0
    np.testing.assert_allclose(sampled[4], 0.25, atol=1.0e-6)
    np.testing.assert_allclose(sampled[5], -0.25, atol=1.0e-6)
    np.testing.assert_allclose(sampled[6], 0.0, atol=1.0e-6)


@pytest.mark.parametrize("shape", (3, 4), ids=("pyramid", "cone"))
@pytest.mark.parametrize("voxel_size", (0.25, 0.1, 0.04))
def test_cone_and_pyramid_resolution_offsets_follow_euclidean_side_distance(
    shape,
    voxel_size,
):
    boundary = np.array(
        (0.0, 0.5, 0.0) if shape == 3 else (0.5, 0.0, 0.0),
        dtype=np.float32,
    )
    axis = 1 if shape == 3 else 0
    inward, outward = boundary.copy(), boundary.copy()
    inward[axis] -= voxel_size * 0.5
    outward[axis] += voxel_size * 0.5
    points = np.asarray([inward, outward], dtype=np.float32)
    device = sourcing.wp.get_preferred_device()
    values = sourcing.wp.empty(2, dtype=float, device=device)
    sourcing.wp.launch(
        _sample_shape_local_field,
        dim=2,
        inputs=[
            shape,
            sourcing.wp.array(points, dtype=sourcing.wp.vec3, device=device),
            values,
        ],
        device=device,
    )
    expected = voxel_size / np.sqrt(5.0)
    np.testing.assert_allclose(values.numpy(), (-expected, expected), atol=2.0e-6)


@pytest.mark.parametrize("shape", (3, 4), ids=("pyramid", "cone"))
def test_cone_and_pyramid_world_transform_preserves_exact_distance(shape):
    desc = SourceShapeDesc(
        shape=shape,
        position=(2.0, 3.0, 4.0),
        rotation=(0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)),
        radius=1.0,
        height=1.0,
    )
    local = np.asarray([(0.0, 0.5, 0.0), (0.0, 0.75, 0.0)], dtype=np.float32)
    world = np.column_stack((-local[:, 1], local[:, 0], local[:, 2]))
    world += np.asarray(desc.position, dtype=np.float32)
    device = sourcing.wp.get_preferred_device()
    values = sourcing.wp.empty(2, dtype=float, device=device)
    sourcing.wp.launch(
        _sample_shape_world_field,
        dim=2,
        inputs=[
            desc.to_struct(),
            sourcing.wp.array(world, dtype=sourcing.wp.vec3, device=device),
            values,
        ],
        device=device,
    )
    assert abs(values.numpy()[0]) < 1.0e-6
    assert values.numpy()[1] > 0.0


@pytest.mark.parametrize("shape", (3, 4), ids=("pyramid", "cone"))
@pytest.mark.parametrize(("radius", "height"), ((0.0, 1.0), (1.0, 0.0), (0.0, 0.0)))
def test_cone_and_pyramid_degenerate_dimensions_are_finite_and_nonnegative(
    shape,
    radius,
    height,
):
    points = np.asarray(
        [(0.0, 0.0, 0.0), (0.25, 0.5, -0.75), (2.0, -1.0, 3.0)], dtype=np.float32
    )
    device = sourcing.wp.get_preferred_device()
    values = sourcing.wp.empty(len(points), dtype=float, device=device)
    sourcing.wp.launch(
        _sample_degenerate_shape_field,
        dim=len(points),
        inputs=[
            shape,
            radius,
            height,
            sourcing.wp.array(points, dtype=sourcing.wp.vec3, device=device),
            values,
        ],
        device=device,
    )
    sampled = values.numpy()
    assert np.isfinite(sampled).all()
    assert (sampled >= 0.0).all()


def test_cone_and_pyramid_exact_distance_is_explicit_in_product_copy():
    kernel_text = open(source_kernels.__file__, encoding="utf-8").read()
    properties_text = open(
        "fluidSim_warp_blender51/properties.py", encoding="utf-8"
    ).read()
    guide_text = open(
        "fluidSim_warp_blender51/docs/user_guide.md", encoding="utf-8"
    ).read()

    assert "Exact Euclidean signed distance" in kernel_text
    assert "exact local Euclidean distance" in properties_text
    assert "exact Euclidean boundary distance" in guide_text


@pytest.mark.parametrize("reverse_winding", (False, True))
def test_watertight_mesh_inside_is_independent_of_triangle_winding(reverse_winding):
    vertices = np.asarray(
        [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ],
        dtype=np.float32,
    )
    triangles = np.asarray(
        [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=np.int32,
    )
    if reverse_winding:
        triangles = triangles[:, (0, 2, 1)]
    device = sourcing.wp.get_preferred_device()
    mesh = sourcing.wp.Mesh(
        points=sourcing.wp.array(vertices, dtype=sourcing.wp.vec3, device=device),
        indices=sourcing.wp.array(triangles.reshape(-1), dtype=int, device=device),
    )
    points = sourcing.wp.array(
        np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=np.float32),
        dtype=sourcing.wp.vec3,
        device=device,
    )
    values = sourcing.wp.empty(2, dtype=sourcing.wp.int32, device=device)
    sourcing.wp.launch(
        _sample_watertight_mesh_inside,
        dim=2,
        inputs=[mesh.id, points, values],
        device=device,
    )
    np.testing.assert_array_equal(values.numpy(), (1, 0))


class _Particles:
    def __init__(self, host_count=0, capacity=8, refreshed_count=1):
        self.host_count = host_count
        self.capacity = capacity
        self.refreshed_count = refreshed_count
        self.refresh_calls = 0
        self.next_uid = 0
        self.position = "position"
        self.velocity = "velocity"
        self.time_offset = "time_offset"
        self.alive = "alive"
        self.count = self

    def attr(self, name):
        return f"attr:{name}"

    def has_optional_channel(self, _name):
        return False

    def optional_array(self, name):
        return f"optional:{name}"

    def refresh_host_count(self):
        self.refresh_calls += 1
        if isinstance(self.refreshed_count, list):
            # Successive device readbacks: emitters read the counter before and
            # after committing, and those two values are not the same.
            return self.refreshed_count.pop(0)
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


def _pipeline(owner):
    return SourcePipeline(
        device=owner.device,
        grid=owner.grid,
        particles=owner.particles,
        settings=owner.settings,
        active_voxel_count=lambda: owner._nvox,
        current_frame=lambda: 0,
        ensure_particle_capacity=getattr(owner, "ensure_particle_capacity", None),
        invalidate_host_cache=lambda: setattr(owner, "_host_frame_cache", None),
        invalidate_particle_surface=lambda: None,
        voxel_type=owner.voxel_type,
        collision_sdf=owner.collision_sdf,
        collision_velocity=owner.collision_vel,
        velocity=owner.vel,
        temperature=owner.temperature,
        viscosity_field=owner.viscosity_field,
        density=getattr(owner, "density", None),
        voxel_counts=owner.voxel_counts,
        pressure_solver=getattr(owner, "pressure_solver", None),
    )


def _record_warp(monkeypatch):
    arrays = []
    launches = []

    class DeviceArray:
        def __init__(self, shape):
            self.values = np.zeros(shape, dtype=np.int32)

        def zero_(self):
            self.values.fill(0)

        def numpy(self):
            return self.values.copy()

    monkeypatch.setattr(
        sourcing.wp,
        "zeros",
        lambda shape, **kwargs: DeviceArray(shape),
    )
    monkeypatch.setattr(sourcing.wp, "copy", lambda *args, **kwargs: None)
    monkeypatch.setattr(sourcing.wp, "empty", lambda shape, **kwargs: ("empty", shape))

    def array(values, **kwargs):
        result = ("array", len(values), len(arrays))
        arrays.append((values, kwargs, result))
        return result

    monkeypatch.setattr(sourcing.wp, "array", array)

    def launch(kernel, **kwargs):
        launches.append((kernel, kwargs))
        inputs = kwargs.get("inputs", ())
        outputs = kwargs.get("outputs", ())
        if kernel in {
            sourcing.k_count_source_shapes_particles,
            sourcing.k_count_mesh_source_particles,
            sourcing.k_count_compiled_source_particles,
        }:
            inputs[-1].values[0] += 1
        elif kernel is sourcing.k_add_emission_demand:
            inputs[-1].values[0] += int(inputs[0])
        elif kernel is sourcing.k_admit_emission_batch:
            particles, capacity, state = inputs
            available = capacity - particles.host_count
            state.values[4] = available
            state.values[1] = int(state.values[0] <= available)
        elif kernel in {
            sourcing.k_source_shapes_particles,
            sourcing.k_mesh_source_particles,
            sourcing.k_emit_compiled_source_particles,
            sourcing.k_source_owned_points,
        }:
            state = outputs[-1]
            if state.values[1]:
                state.values[2] += (
                    int(inputs[6]) if kernel is sourcing.k_source_owned_points else 1
                )

    monkeypatch.setattr(sourcing.wp, "launch", launch)

    monkeypatch.setattr(
        SourceShapeDesc,
        "to_struct",
        lambda self: ("shape", self.enabled, self.shape),
    )
    return arrays, launches


def test_apply_owns_launch_order_cache_and_full_capacity_signal(monkeypatch):
    arrays, launches = _record_warp(monkeypatch)
    owner = _owner(host_count=8, capacity=8)
    pipeline = _pipeline(owner)
    pipeline.set_sources([SourceShapeDesc(mode=1)])
    pipeline.set_height_source(
        HeightSourceDesc(
            enabled=True,
            base_height=0.75,
            emit_particles=False,
        )
    )

    first_seed = pipeline._emit_seed
    assert pipeline.apply(1.0 / 24.0) is False
    assert pipeline.apply(1.0 / 24.0) is False

    kernels = [kernel for kernel, _ in launches]
    assert kernels.count(sourcing.k_count_source_shapes_particles) == 2
    assert kernels.count(sourcing.k_admit_emission_batch) == 2
    assert kernels.count(sourcing.k_source_shapes_particles) == 2
    assert kernels.count(sourcing.k_source_height) == 2
    assert kernels.count(sourcing.k_source_resolve_type) == 2
    assert len(arrays) == 2  # guarded emission array plus field-stamp array
    assert owner.particles.refresh_calls == 0
    # Both transactions were rejected by the full pool, so retry state is
    # byte-for-byte deterministic rather than silently advancing the RNG.
    assert pipeline._emit_seed == first_seed
    result = pipeline.last_apply_result
    assert result["emitted"] == 0
    assert result["capacity_limited"] is True
    assert result["required"] == 1
    assert result["available"] == 0
    assert result["admitted"] is False
    assert result["deferred"] is True
    assert result["growth_status"] == "DISABLED"


def test_height_emission_reuses_cached_shapes_and_refreshes_after_growth(monkeypatch):
    arrays, launches = _record_warp(monkeypatch)
    owner = _owner()
    pipeline = _pipeline(owner)
    pipeline.set_height_source(
        HeightSourceDesc(
            enabled=True,
            base_height=1.0,
            velocity=(0.0, 0.0, 2.0),
            emit_particles=True,
            emit_rate=6.0,
            # Cache reuse is only observable across repeated emissions, so this
            # case needs the mode that emits every substep.
            emission_mode=SourceEmissionMode.CONTINUOUS_INFLOW,
        )
    )

    assert pipeline.apply(0.01) is True
    assert pipeline.apply(0.01) is True

    assert [kernel for kernel, _ in launches[:5]] == [
        sourcing.k_count_source_shapes_particles,
        sourcing.k_admit_emission_batch,
        sourcing.k_source_height,
        sourcing.k_source_resolve_type,
        sourcing.k_source_shapes_particles,
    ]
    assert len(arrays) == 2  # disabled field placeholder + derived height box
    assert owner.particles.refresh_calls == 2


def test_inject_particles_owns_upload_launch_and_count_refresh(monkeypatch):
    arrays, launches = _record_warp(monkeypatch)
    # Admission reads the device counter in the kernel; the single host refresh
    # reports the committed total after the exact packet transaction.
    owner = _owner(refreshed_count=[0, 2])
    pipeline = _pipeline(owner)

    result = pipeline.inject_particles(
        np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        velocity=(1.0, 0.0, 0.0),
        temperature=2.0,
    )

    assert result == 2
    assert len(arrays) == 1
    assert [kernel for kernel, _ in launches] == [
        sourcing.k_add_emission_demand,
        sourcing.k_admit_emission_batch,
        sourcing.k_source_owned_points,
    ]
    assert launches[-1][1]["dim"] == 2
    assert owner.particles.refresh_calls == 2
    assert owner._host_frame_cache is None
    assert pipeline.last_injection_result == sourcing.InjectionResult(
        admitted=True,
        emitted=2,
        required=2,
        available=8,
        capacity_limited=False,
        total=2,
    )

    assert pipeline.inject_particles([]) == 0
    assert len(launches) == 3
    assert owner.particles.refresh_calls == 2
