"""USD export needs one point-cloud datablock per output for the whole range.

Blender 5.2's USD exporter writes per-frame time samples for a point cloud only
while the frame handler rewrites the *same* datablock; a replaced datablock is
written as a separate prim. Publication replaces a cloud when it must grow or
can shrink, so the exporter reserves capacity for the range's largest frame and
holds shrinking off while it runs. These tests pin both halves outside Blender.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender import output_publication as output
from fluidSim_warp_blender51.sim.attributes import ATTRIBUTE_SPECS


class _Data:
    def __init__(self, owner, components):
        self.owner = owner
        self.components = components
        self.last = None

    def foreach_set(self, _prop, values):
        array = np.asarray(values)
        assert array.size == len(self.owner.points) * self.components
        self.last = array.copy()


class _Attribute:
    def __init__(self, owner, name, data_type, domain):
        self.name = name
        self.data_type = data_type
        self.domain = domain
        components = {"FLOAT_VECTOR": 3, "FLOAT_COLOR": 4}.get(data_type, 1)
        self.data = _Data(owner, components)


class _Attributes(dict):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def new(self, name=None, type=None, domain=None):  # noqa: A002 - RNA names
        attribute = _Attribute(self.owner, name, type, domain)
        self[name] = attribute
        return attribute

    def __iter__(self):
        return iter(list(self.values()))


class _Cloud:
    users = 1

    def __init__(self, name):
        self.name = name
        self.points = []
        self.attributes = _Attributes(self)
        self.materials = []

    def resize(self, count):
        self.points = [None] * int(count)

    def update_tag(self):
        pass


class _Clouds:
    def __init__(self):
        self.created = []

    def new(self, name):
        cloud = _Cloud(name)
        self.created.append(cloud)
        return cloud


class _Obj(dict):
    type = "POINTCLOUD"
    name = "FluidSim Particles"


@pytest.fixture
def clouds(monkeypatch):
    factory = _Clouds()
    monkeypatch.setattr(output, "bpy", SimpleNamespace(
        data=SimpleNamespace(pointclouds=factory),
        types=SimpleNamespace(Mesh=type("Mesh", (), {}),
                              PointCloud=_Cloud)))
    output._SCRATCH.clear()
    return factory


def _published_cloud(clouds, count, *, color=True, keys=("age",)):
    """A particle output as publication leaves it after one frame of ``count``."""
    obj = _Obj()
    cloud = clouds.new(obj.name)
    cloud.resize(output._point_capacity_for(count))
    output._ensure_point_attributes(cloud, color, attribute_keys=keys)
    obj.data = cloud
    obj[output.POINT_COUNT_TAG] = count
    clouds.created.clear()
    return obj


def _frame(obj, count, *, color=True):
    rng = np.random.default_rng(count)
    output._swap_points(
        obj, rng.random((count, 3), dtype=np.float32), 0.05,
        rng.random((count, 4), dtype=np.float32) if color else None,
        {"age": np.full(count, 0.5, np.float32)})


def test_shrink_replacement_is_held_inside_hold_point_capacity(clouds):
    obj = _published_cloud(clouds, 20_000)
    reason = output._point_replacement_reason(obj.data, 10, True, ("age",))
    assert reason == "shrink"
    with output.hold_point_capacity():
        assert output._point_replacement_reason(
            obj.data, 10, True, ("age",)) is None
        with output.hold_point_capacity():  # holds nest
            pass
        assert output._point_replacement_reason(
            obj.data, 10, True, ("age",)) is None
    assert output._point_replacement_reason(
        obj.data, 10, True, ("age",)) == "shrink"


def test_growth_and_schema_drift_are_never_held(clouds):
    obj = _published_cloud(clouds, 100)
    capacity = len(obj.data.points)
    with output.hold_point_capacity():
        assert output._point_replacement_reason(
            obj.data, capacity + 1, True, ("age",)) == "growth"
        assert output._point_replacement_reason(
            obj.data, 10, False, ("age",)) == "schema"


def test_hold_is_released_when_the_body_raises(clouds):
    obj = _published_cloud(clouds, 20_000)
    with pytest.raises(RuntimeError):
        with output.hold_point_capacity():
            raise RuntimeError("exporter failed")
    assert output._point_replacement_reason(
        obj.data, 10, True, ("age",)) == "shrink"


def test_reserve_grows_once_with_the_same_schema(clouds):
    obj = _published_cloud(clouds, 100)
    old = obj.data
    assert output.reserve_point_capacity(obj, 5_000) is True
    new = obj.data
    assert new is not old
    assert len(new.points) == output._point_capacity_for(5_000)
    assert {(a.name, a.data_type, a.domain) for a in new.attributes} == {
        (a.name, a.data_type, a.domain) for a in old.attributes}
    assert ATTRIBUTE_SPECS["age"].blender_name in new.attributes
    # Already large enough: no second replacement.
    assert output.reserve_point_capacity(obj, 5_000) is False
    assert output.reserve_point_capacity(obj, 10) is False
    assert obj.data is new


def test_reserved_cloud_survives_a_varying_range_under_a_hold(clouds):
    counts = (40, 5_000, 3, 2_500, 0, 4_999)
    obj = _published_cloud(clouds, counts[0])
    output.reserve_point_capacity(obj, max(counts))
    reserved = obj.data
    with output.hold_point_capacity():
        for count in counts:
            _frame(obj, count)
            assert obj.data is reserved, count
            assert obj[output.POINT_COUNT_TAG] == count
            radius = reserved.attributes["radius"].data.last
            # Padding stays zero-radius: the export trims it by radius.
            assert np.count_nonzero(radius) == count


def test_reserve_ignores_non_pointcloud_outputs(clouds):
    obj = _Obj()
    obj.type = "MESH"
    obj.data = SimpleNamespace(vertices=[])
    assert output.reserve_point_capacity(obj, 10_000) is False
