"""Display edits must update published geometry independently of render flags."""
import importlib
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest
from settings_doubles import full_settings, split_settings
from test_scene_bridge_cache_playback import _install_scene_bridge_stubs


@pytest.fixture
def world(monkeypatch):
    bridge = _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51 import blender
    modules = {}
    for name in ("output_publication", "pending_settings"):
        # Record the package attribute before importlib replaces it, so later
        # tests retain the same module object as their collection-time imports.
        monkeypatch.setattr(blender, name, None, raising=False)
        monkeypatch.delitem(sys.modules, f"{blender.__name__}.{name}", raising=False)
        module = importlib.import_module(f"{blender.__name__}.{name}")
        modules[name] = module
    op, pending = modules["output_publication"], modules["pending_settings"]
    scene = NS(frame_current=7, fluidsim=split_settings(full_settings(
        quality="CUSTOM", liquid_display_mode="OFF", live=True,
        cache_playback_enabled=True, show_surface=False, show_particles=False,
        show_whitewater=True, whitewater_enabled=True, surface_enabled=False,
        whitewater_output_mode="COMBINED", cache_path="test-cache")))
    bpy = sys.modules["bpy"]
    bpy.context.scene = scene
    monkeypatch.setattr(pending, "_tag_view3d_redraw", lambda: None)
    materials = types.ModuleType(f"{blender.__name__}.materials")
    for name in ("surface", "particle", "whitewater", "foam", "bubble", "spray"):
        setattr(materials, f"ensure_{name}_material", lambda *_args: None)
    monkeypatch.setitem(sys.modules, materials.__name__, materials)
    monkeypatch.setattr(blender, "materials", materials, raising=False)

    class Object(dict):
        def __init__(self):
            super().__init__()
            self.type = "MESH"
            self.data = NS(vertices=[])
            self.hide_viewport = self.hide_render = self.hidden = False
            self.attributes = {}

        def hide_set(self, hidden, **_kwargs):
            self.hidden = hidden

        def hide_get(self, **_kwargs):
            return self.hidden

    objects = {}

    def ensure(_scene, name, *_args):
        return objects.setdefault(name, Object())

    def write_points(obj, positions, _colors=None, attributes=None):
        obj.data.vertices = list(positions)
        obj.attributes = attributes or {}

    monkeypatch.setattr(op, "_require_bpy", lambda: None)
    monkeypatch.setattr(op, "output_object", lambda _scene, name: objects.get(name))
    monkeypatch.setattr(op, "_ensure_point_object", ensure)
    monkeypatch.setattr(op, "_ensure_object", ensure)
    monkeypatch.setattr(op, "_supports_pointcloud", lambda: False)
    monkeypatch.setattr(op, "_swap_vertex_cloud", write_points)
    monkeypatch.setattr(op, "_swap_mesh", lambda obj, positions, *_args: write_points(obj, positions))
    monkeypatch.setattr(op, "_ensure_material", lambda *_args: None)
    monkeypatch.setattr(op, "_set_point_radius", lambda *_args: None)
    monkeypatch.setattr(op, "_sync_render_inputs", lambda *_args: NS(warning=""))
    from fluidSim_warp_blender51.io import cache
    data = {
        "positions": np.arange(18, dtype=np.float32).reshape(6, 3),
        "velocities": np.tile(np.array([[3, 4, 0]], np.float32), (6, 1)),
        "whitewater": np.ones((3, 3), np.float32),
        "whitewater_type": np.array([0, 1, 2], np.int32),
        "attribute_schema_version": 2,
        "attribute_inventory_json": '{"surface":[],"particles":[],"whitewater":["type"]}',
        "surface_attribute_transfer_method": cache.SURFACE_TRANSFER_METHOD,
    }
    reads = []

    def read_frame(base, frame):
        reads.append((base, frame))
        return data

    monkeypatch.setattr(cache, "read_frame", read_frame)
    return NS(bridge=bridge, output=op, pending=pending, scene=scene, data=data,
              objects=objects, reads=reads)


@pytest.mark.parametrize("empty_surface", [False, True])
@pytest.mark.parametrize("mode,want", [("HEIGHT", [2, 5, 8, 11, 14, 17]), ("SPEED", [5] * 6)])
def test_live_fallback_attributes_match_rendered_and_cached_controls(world, empty_surface, mode, want):
    w = world
    fs = w.scene.fluidsim
    fs.liquid_display_mode = "SURFACE"
    fs.show_whitewater = False
    fs.particle_attribute_mode = mode
    attribute_key = "speed" if mode == "SPEED" else "__visualization__"
    fs.domain_object.fluidsim_domain.surface_enabled = empty_surface
    host = NS(
        solver=NS(settings=NS(voxel_size=.1)),
        positions_strided=lambda _step: w.data["positions"],
        velocities_strided=lambda _step: w.data["velocities"],
        surface_mesh=(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)),
        output_attributes=lambda *_args, **_kwargs: {})
    for render_particles in (False, True):
        fs.show_particles = render_particles
        payload = w.output.prepare_live_payload(w.scene, host)
        assert payload.viewport_particles and len(payload.particles) == 6
        np.testing.assert_array_equal(payload.particle_attributes.get(attribute_key), want)
        assert fs.show_particles is render_particles
    fs.show_particles = False
    cached = w.output.prepare_cache_payload(w.scene, w.data)
    np.testing.assert_array_equal(cached.particle_attributes[attribute_key], want)
