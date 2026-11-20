"""Liquid Preview draws the points a file was saved with, before the first step."""
import importlib
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_scene_bridge_cache_playback import _install_scene_bridge_stubs


@pytest.fixture
def output(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51 import blender

    monkeypatch.setattr(blender, "output_publication", None, raising=False)
    monkeypatch.delitem(
        sys.modules, f"{blender.__name__}.output_publication", raising=False)
    module = importlib.import_module(f"{blender.__name__}.output_publication")
    module.clear_published_points()
    yield module
    module.clear_published_points()


class _Column:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def __len__(self):
        return len(self.values)

    def foreach_get(self, _name, out):
        out[:] = self.values.reshape(-1)


def _cloud(live=5, capacity=8, radius=0.0125):
    positions = np.zeros((capacity, 3), dtype=np.float32)
    positions[:live] = np.arange(live * 3, dtype=np.float32).reshape(live, 3)
    radii = np.full(capacity, radius, dtype=np.float32)
    attributes = {"position": NS(data=_Column(positions)),
                  "radius": NS(data=_Column(radii))}
    obj = {"fluidsim_live_point_count": live}
    return NS(type="POINTCLOUD", data=NS(attributes=attributes),
              get=obj.get)


def _scene(pointer=11):
    return NS(frame_current=1, as_pointer=lambda: pointer)


def test_saved_points_become_the_preview_record(output, monkeypatch):
    cloud = _cloud()
    monkeypatch.setattr(output, "output_object", lambda *_a, **_k: cloud)

    record = output.seed_published_points_from_saved_outputs(_scene())

    assert output.latest_published_points() is record
    assert record.particle_count == 5
    np.testing.assert_array_equal(
        record.positions, np.arange(15, dtype=np.float32).reshape(5, 3))
    assert record.particle_radius == pytest.approx(0.0125)
    assert record.whitewater_count == 0
    assert record.scene_key == 11
    assert not record.positions.flags.writeable


def test_a_record_for_this_scene_is_never_replaced(output, monkeypatch):
    monkeypatch.setattr(output, "output_object", lambda *_a, **_k: _cloud())
    first = output.seed_published_points_from_saved_outputs(_scene())

    assert output.seed_published_points_from_saved_outputs(_scene()) is None
    assert output.latest_published_points() is first


def test_another_scenes_record_is_replaced(output, monkeypatch):
    monkeypatch.setattr(output, "output_object", lambda *_a, **_k: _cloud())
    output.seed_published_points_from_saved_outputs(_scene(pointer=7))

    record = output.seed_published_points_from_saved_outputs(_scene(pointer=9))

    assert record is not None and record.scene_key == 9


@pytest.mark.parametrize("obj", [None, NS(type="MESH", data=None, get=lambda *_a: 3),
                                 _cloud(live=0)])
def test_nothing_is_published_without_saved_points(output, monkeypatch, obj):
    monkeypatch.setattr(output, "output_object", lambda *_a, **_k: obj)

    assert output.seed_published_points_from_saved_outputs(_scene()) is None
    assert output.latest_published_points() is None


def test_saving_surface_preview_preserves_retained_cache_attributes(output, monkeypatch):
    from fluidSim_warp_blender51 import blender

    scene = _scene()
    scene.fluidsim = NS(liquid_preview_enabled=True, show_particles=False)
    cloud = _cloud()
    ids = np.arange(5, dtype=np.int32) + 100
    positions = np.ones((5, 3), dtype=np.float32)
    replacements = []

    def swap(obj, points, radius, colors=None, attributes=None):
        replacements.append(attributes)
        obj.data.attributes["uid"] = ids.copy() if attributes is not None else np.zeros(5, np.int32)

    monkeypatch.setattr(blender, "materials", NS(ensure_particle_material=None), raising=False)
    monkeypatch.setattr(blender, "liquid_display", NS(apply_viewport=lambda *_a, **_k: None), raising=False)
    monkeypatch.setattr(output, "_require_bpy", lambda: None)
    monkeypatch.setattr(output, "_ensure_point_object", lambda *_a: cloud)
    monkeypatch.setattr(output, "output_object", lambda *_a, **_k: cloud)
    monkeypatch.setattr(output, "_swap_points", swap)
    monkeypatch.setattr(output, "_supports_pointcloud", lambda: True)
    monkeypatch.setattr(output, "_ensure_material", lambda *_a: None)
    monkeypatch.setattr(output, "_set_hidden", lambda *_a, **_k: None)
    monkeypatch.setattr(output, "_hide_and_clear_output", lambda *_a: None)
    monkeypatch.setattr(output, "_sync_render_inputs", lambda *_a: None)
    monkeypatch.setattr(output, "output_family_policy", lambda *_a, **_k: [
        NS(name=output.SURFACE_NAME, visible=False),
        NS(name=output.PARTICLES_NAME, visible=False, radius=0.0125,
           material="ensure_particle_material"),
    ])
    payload = NS(
        retain_cache_geometry=True, surface_visible=False, viewport_surface=False,
        publish_surface=False, particles_visible=False, viewport_particles=False,
        particles=positions, particle_radius=0.0125, particle_attributes={"uid": ids},
        particle_attribute_mode="NONE", whitewater_visible=False,
    )

    output.publish_payload(scene, payload)
    output.persist_preview_points_for_save(scene)

    assert len(replacements) == 1
    np.testing.assert_array_equal(cloud.data.attributes["uid"], ids)
