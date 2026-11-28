"""Actual Warp CPU path samples: keyed seeds and absolute progressive indices."""
import numpy as np
import pytest

wp = pytest.importorskip('warp')


@pytest.fixture(scope='module')
def glass_paths():
    from fluidSim_warp_blender51.fspt.kernels import trace_paths

    wp.init()
    width, height = 128, 64
    count = width * height
    points, indices = [], []
    for z, facing in [(0, -1), (0.5, 1), (-1.5, 1)]:
        start = len(points)
        points.extend([(-8, -8, z), (0, 8, z), (8, -8, z)])
        winding = (0, 1, 2) if facing == -1 else (0, 2, 1)
        indices.extend(start + i for i in winding)
    mesh = wp.Mesh(points=wp.array(points, dtype=wp.vec3, device='cpu'),
                   indices=wp.array(indices, dtype=wp.int32, device='cpu'))
    materials = wp.array([0, 1, 2], dtype=wp.int32, device='cpu')
    kinds = wp.array([1, 2, 2], dtype=wp.int32, device='cpu')
    colors = wp.array([(1, 1, 1)] * 3, dtype=wp.vec3, device='cpu')
    emissions = wp.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=wp.vec3, device='cpu')
    iors = wp.array([1.5] * 3, dtype=float, device='cpu')
    matrix = wp.mat44(np.eye(4, dtype=np.float32))

    def trace(seed, first, samples, accumulation=None):
        if accumulation is None:
            accumulation = wp.zeros(count, dtype=wp.vec3, device='cpu')
        output = wp.empty(count, dtype=wp.vec4, device='cpu')
        wp.launch(trace_paths, dim=count, inputs=[
            mesh.id, 3, materials, kinds, colors, emissions, iors, wp.vec3(0.0),
            matrix, width, height, first, samples, 2, seed, accumulation, output,
        ], device='cpu')
        return accumulation, output.numpy()
    return trace


def test_adjacent_render_seeds_do_not_reuse_shifted_paths(glass_paths):
    first = glass_paths(31, 1, 1)[0].numpy()
    adjacent = glass_paths(32, 0, 1)[0].numpy()
    assert np.mean(np.any(first != adjacent, axis=1)) > 0.01


def test_same_seed_progressive_dispatch_is_identical_to_one_dispatch(glass_paths):
    accumulation, _ = glass_paths(31, 0, 1)
    _, split = glass_paths(31, 1, 1, accumulation)
    _, whole = glass_paths(31, 0, 2)
    np.testing.assert_array_equal(split, whole)


def test_keyed_samples_retain_normal_incidence_fresnel_and_radiance_weight(glass_paths):
    _, pixels = glass_paths(31, 0, 1)
    reflected = np.count_nonzero(pixels[:, 1]) / len(pixels)
    assert abs(reflected - 0.04) < 0.015
    assert np.all(np.isclose(pixels[:, 0], 4 / 9) | np.isclose(pixels[:, 0], 0))
    assert np.isfinite(pixels).all() and np.all(pixels[:, 3] == 1)


def test_sampler_wraps_supported_boundary_indices_with_independent_known_vectors():
    from fluidSim_warp_blender51.fspt.kernels import path_rng

    @wp.kernel
    def sample_keys(seeds: wp.array(dtype=wp.int32), pixels: wp.array(dtype=wp.int32),
                    samples: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.uint32)):
        i = wp.tid()
        values[i] = path_rng(seeds[i], pixels[i], samples[i])

    # Explicit integer reference to Warp 1.19 native rand.h. Do not invoke
    # either the production sampler or Warp's host random functions here.
    mask = (1 << 32) - 1
    def pcg(value):
        b = (value * 747796405 + 2891336453) & mask
        c = (((b >> ((b >> 28) + 4)) ^ b) * 277803737) & mask
        return ((c >> 22) ^ c) & mask
    cases = [(0, 0, 0), (31, 0, 1), (32, 0, 0), (100000, 16777215, 4095),
             (31, 16777215, 4094), (32, 16777215, 4093)]
    expected = [pcg((pcg((seed + pcg(pixel)) & mask) + pcg(sample)) & mask)
                for seed, pixel, sample in cases]
    arrays = [wp.array([row[col] for row in cases], dtype=wp.int32, device='cpu')
              for col in range(3)]
    result = wp.empty(len(cases), dtype=wp.uint32, device='cpu')
    wp.launch(sample_keys, dim=len(cases), inputs=[*arrays, result], device='cpu')
    np.testing.assert_array_equal(result.numpy(), np.asarray(expected, dtype=np.uint32))
    assert expected[1] != expected[2] and expected[4] != expected[5]
