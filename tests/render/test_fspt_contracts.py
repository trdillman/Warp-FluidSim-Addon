"""Observable ownership and image-history contracts; no Blender or GPU doubles."""
import importlib
import importlib.util

import numpy as np
import pytest


def contracts():
    assert importlib.util.find_spec("fluidSim_warp_blender51.fspt"), "FSPT is not implemented"
    return importlib.import_module("fluidSim_warp_blender51.fspt.contracts")


def fixture(c, **overrides):
    args = {"vertices": np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                "triangles": np.array([[0, 1, 2]]), "material_indices": np.array([0]),
                "materials": (c.Material(),), "world": (0.1, 0.2, 0.3), "frame": 1}
    args.update(overrides)
    return c.SceneSnapshot(**args)


def test_snapshot_owns_source_arrays_and_is_read_only():
    c = contracts()
    original = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    s = fixture(c, vertices=original)
    before = s.digest
    original[0, 0] = 99
    assert s.vertices[0, 0] == 0
    assert s.digest == before
    with pytest.raises(ValueError):
        s.vertices[0, 0] = 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_geometry_is_rejected(bad):
    c = contracts()
    with pytest.raises(ValueError, match="finite"):
        fixture(c, vertices=[[bad, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_out_of_bounds_geometry_and_materials_are_rejected():
    c = contracts()
    with pytest.raises(ValueError, match="triangle"):
        fixture(c, triangles=[[0, 1, 9]])
    with pytest.raises(ValueError, match="material"):
        fixture(c, material_indices=[2])


def test_scene_identity_changes_for_material_geometry_world_and_frame():
    c = contracts()
    a = fixture(c)
    variants = [fixture(c, materials=(c.Material(color=(1., 0., 0.)),)),
                fixture(c, vertices=[[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
                fixture(c, world=(1., 1., 1.)), fixture(c, frame=2)]
    assert all(s.digest != a.digest for s in variants)


def test_material_rejects_invalid_optical_inputs():
    c = contracts()
    for kwargs in [{"ior": 0}, {"color": (-1, 0, 0)},
                   {"emission": (float("nan"), 0, 0)}, {"kind": "unknown"}]:
        with pytest.raises(ValueError):
            c.Material(**kwargs)


def test_camera_matrix_is_owned_finite_and_invertible():
    c = contracts()
    original = np.eye(4)
    camera = c.CameraSnapshot(original)
    original[0, 0] = 2
    assert camera.inverse_view_projection[0, 0] == 1
    with pytest.raises(ValueError):
        c.CameraSnapshot(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        c.CameraSnapshot(np.full((4, 4), np.nan))


def test_accumulation_resets_for_camera_resize_scene_or_settings():
    c = contracts()
    scene = fixture(c)
    camera = c.CameraSnapshot(np.eye(4))
    key = c.frame_key(scene, camera, 64, 32, 4, 0)
    history = c.AccumulationState()
    assert history.begin(key)
    history.advance(2)
    assert not history.begin(key)
    assert history.samples == 2
    moved = np.eye(4)
    moved[0, 3] = 1
    for new in [c.frame_key(scene, c.CameraSnapshot(moved), 64, 32, 4, 0),
                c.frame_key(scene, camera, 32, 32, 4, 0),
                c.frame_key(scene, camera, 64, 32, 8, 0),
                c.frame_key(fixture(c, frame=2), camera, 64, 32, 4, 0)]:
        history.advance(1)
        assert history.begin(new)
        assert history.samples == 0


def test_stale_frame_is_rejected_before_presentation():
    c = contracts()
    with pytest.raises(RuntimeError, match="stale"):
        c.require_current("new scene", "old scene")
    c.require_current("same", "same")


def test_empty_scene_is_valid_and_has_a_stable_identity():
    c = contracts()
    a = c.SceneSnapshot([], [], [], (c.Material(),), (0., 0., 0.), 1)
    b = c.SceneSnapshot([], [], [], (c.Material(),), (0., 0., 0.), 1)
    assert a.vertices.shape == (0, 3)
    assert a.triangles.shape == (0, 3)
    assert a.digest == b.digest


def test_lazy_package_does_not_import_warp_or_blender():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c",
        ("import sys; import fluidSim_warp_blender51.fspt; "
        "assert 'bpy' not in sys.modules; assert 'warp' not in sys.modules")],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
