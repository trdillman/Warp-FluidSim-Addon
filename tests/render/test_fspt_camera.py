"""Final camera capability controls; unsupported optics must be explicit."""
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender.fspt_scene import (
    UnsupportedSceneError,
    camera_snapshot,
)


class Matrix:
    def __matmul__(self, other):
        return self

    def normalized(self):
        return self

    def inverted(self):
        return self

    def __array__(self, dtype=None, copy=None):
        return np.eye(4, dtype=dtype)


def camera_graph(*, motion_blur=False, depth_of_field=False):
    camera = SimpleNamespace(name="Camera", data=SimpleNamespace(type="PERSP",
                              dof=SimpleNamespace(use_dof=depth_of_field)),
                              matrix_world=Matrix(), calc_matrix_camera=lambda *args, **kwargs: Matrix())
    camera.evaluated_get = lambda _: camera
    scene = SimpleNamespace(name="Scene", camera=camera,
                            render=SimpleNamespace(use_motion_blur=motion_blur,
                                                   pixel_aspect_x=1.0, pixel_aspect_y=1.0))
    return SimpleNamespace(scene=scene)


def test_supported_static_camera_still_produces_finite_ray_projection():
    result = camera_snapshot(camera_graph(), 128, 96)
    np.testing.assert_array_equal(result.inverse_view_projection, np.eye(4))


def test_authored_motion_blur_is_refused_instead_of_silently_ignored():
    with pytest.raises(UnsupportedSceneError, match="motion blur"):
        camera_snapshot(camera_graph(motion_blur=True), 128, 96)


def test_authored_depth_of_field_remains_an_explicit_refusal():
    with pytest.raises(UnsupportedSceneError, match="depth of field"):
        camera_snapshot(camera_graph(depth_of_field=True), 128, 96)
