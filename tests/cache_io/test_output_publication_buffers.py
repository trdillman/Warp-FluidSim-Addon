"""Exercise publication against Blender RNA's bulk-buffer contract.

The fake collections reject the per-element conversion fallback rather than
merely checking the resulting geometry. Actual Blender acceptance covers the
same shared writer separately.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender import output_publication as output
from fluidSim_warp_blender51.sim.attributes import ATTRIBUTE_SPECS


class BulkCollection:
    def __init__(self, formats):
        self.formats = formats
        self.count = 0
        self.values = {}

    def __len__(self):
        return self.count

    def add(self, count):
        self.count += count

    def foreach_set(self, name, values):
        view = memoryview(values)
        assert view.format == self.formats[name], (
            f"{name}: {view.format!r} would miss Blender's bulk-buffer path")
        assert view.c_contiguous
        self.values[name] = np.asarray(values).copy()


class Attributes(dict):
    def new(self, *, name, type, domain):
        collection = BulkCollection({"value": "i" if type == "INT" else "f"})
        attribute = SimpleNamespace(data_type=type, domain=domain, data=collection)
        self[name] = attribute
        return attribute

    def remove(self, attribute):
        for name, value in list(self.items()):
            if value is attribute:
                del self[name]


class Mesh:
    users = 1

    def __init__(self):
        self.vertices = BulkCollection({"co": "f"})
        self.loops = BulkCollection({"vertex_index": "I"})
        self.polygons = BulkCollection({"loop_start": "I", "loop_total": "I"})
        self.attributes = Attributes()
        self.updated = False
        self.smooth = False

    def clear_geometry(self):
        for collection in (self.vertices, self.loops, self.polygons):
            collection.count = 0
            collection.values.clear()

    def shade_smooth(self):
        self.smooth = True

    def update(self):
        self.updated = True


def test_mesh_topology_uses_unsigned_bulk_buffers_and_preserves_indices():
    mesh = Mesh()
    obj = SimpleNamespace(data=mesh, type="MESH")
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    output._swap_mesh(obj, verts, tris)
    np.testing.assert_array_equal(mesh.loops.values["vertex_index"], tris.ravel())
    np.testing.assert_array_equal(mesh.polygons.values["loop_start"], [0, 3])
    np.testing.assert_array_equal(mesh.polygons.values["loop_total"], [3, 3])
    assert mesh.updated and mesh.smooth
    # An empty following frame must still clear the existing geometry.
    output._swap_mesh(obj, np.empty((0, 3), np.float32), np.empty((0, 3), np.int32))
    assert len(mesh.vertices) == len(mesh.loops) == len(mesh.polygons) == 0


def test_integer_attribute_uses_signed_bulk_buffer_and_preserves_padding():
    mesh = Mesh()
    mesh.vertices.add(4)
    values = np.array([-1, 23], np.int32)
    attribute = output.publish_attribute(
        mesh, ATTRIBUTE_SPECS["source_id"], values, live_count=2)
    np.testing.assert_array_equal(attribute.data.values["value"], [-1, 23, 0, 0])


@pytest.mark.parametrize("bad_index", [-1, 4])
def test_invalid_triangle_indices_are_rejected_before_unsigned_upload(bad_index):
    mesh = Mesh()
    obj = SimpleNamespace(data=mesh, type="MESH")
    with pytest.raises(ValueError, match="outside the vertex range"):
        output._swap_mesh(
            obj, np.zeros((4, 3), np.float32), np.array([[0, 1, bad_index]], np.int32))
    assert not mesh.loops.values


class _PointAttribute:
    def __init__(self, name, data_type, capacity, components):
        self.name = name
        self.data_type = data_type
        self.domain = "POINT"
        self.uploads = []
        self.data = self
        self._shape = (capacity,) if components == 1 else (capacity, components)

    def foreach_set(self, _prop, values):
        array = np.asarray(values)
        self.uploads.append((array.__array_interface__["data"][0], array.copy()))


class _PointCloud:
    users = 1

    def __init__(self, capacity):
        self.points = [None] * capacity
        self.attributes = _PointAttributes({
            "position": _PointAttribute("position", "FLOAT_VECTOR", capacity, 3),
            "radius": _PointAttribute("radius", "FLOAT", capacity, 1),
            output.COLOR_ATTR: _PointAttribute(
                output.COLOR_ATTR, "FLOAT_COLOR", capacity, 4),
        })

    def resize(self, _count):
        raise AssertionError("steady-state frames must not reallocate")

    def update_tag(self):
        pass


class _PointAttributes(dict):
    def __iter__(self):
        return iter(self.values())


def test_point_publication_reuses_zero_padded_scratch_across_frames(monkeypatch):
    """P18: capacity-sized scratch is reused, and the padding stays zero when
    the live count drops (equivalent to a fresh np.zeros every frame)."""
    scratch = getattr(output, "_SCRATCH", None)
    if scratch is not None:
        scratch.clear()
    capacity = output._point_capacity_for(1000)
    allocations = []

    class _CountingNumpy:
        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def zeros(shape, *args, **kwargs):
            first = shape[0] if isinstance(shape, tuple) else shape
            if first == capacity:
                allocations.append(shape)
            return np.zeros(shape, *args, **kwargs)

    monkeypatch.setattr(output, "np", _CountingNumpy())
    cloud = _PointCloud(capacity)
    obj = type("Obj", (dict,), {})()
    obj.type = "POINTCLOUD"
    obj.data = cloud
    obj.name = "FluidSim Particles"
    rng = np.random.default_rng(3)

    for frame, n in enumerate((1000, 800, 900)):
        if frame == 1:
            allocations.clear()  # the first frame may allocate the scratch
        positions = rng.random((n, 3), dtype=np.float32)
        colors = rng.random((n, 4), dtype=np.float32)
        output._swap_points(obj, positions, 0.25, colors)
        expected_coords = np.zeros((capacity, 3), np.float32)
        expected_coords[:n] = positions
        expected_radii = np.zeros(capacity, np.float32)
        expected_radii[:n] = 0.25
        expected_colors = np.zeros((capacity, 4), np.float32)
        expected_colors[:n] = colors
        for name, expected in (("position", expected_coords),
                               ("radius", expected_radii),
                               (output.COLOR_ATTR, expected_colors)):
            np.testing.assert_array_equal(
                cloud.attributes[name].uploads[-1][1], expected.ravel(), err_msg=name)
        assert obj[output.POINT_COUNT_TAG] == n

    assert allocations == [], "steady-state frames allocated capacity-sized scratch"


def test_float_attribute_keeps_its_bulk_buffer_and_padding():
    mesh = Mesh()
    mesh.vertices.add(3)
    attribute = output.publish_attribute(
        mesh, ATTRIBUTE_SPECS["age"], np.array([1.5], np.float32), live_count=1)
    np.testing.assert_array_equal(attribute.data.values["value"], [1.5, 0, 0])
