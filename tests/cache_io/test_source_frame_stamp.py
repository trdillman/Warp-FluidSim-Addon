"""The per-frame source stamp, as the Blender adapter produces it.

``source_frame_stamp`` is what a v3 bake writes for every frame and what a
resume compares against, so these pin what moves it (a transform, geometry)
and what must not (a rename, a panel setting, the frame number itself).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender.cache_identity import source_frame_stamp

_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class _Object(dict):
    def __init__(self, key, *, role="COLLIDER"):
        super().__init__(fluidsim_persistent_id=key)
        properties = {
            "role": SimpleNamespace(type="ENUM", is_array=False),
            "friction": SimpleNamespace(type="FLOAT", is_array=False),
        }
        self.name = key
        self.name_full = key
        self.library = None
        self.type = "EMPTY"
        self.hide_viewport = False
        self.data = SimpleNamespace(points=())
        self.matrix_world = _IDENTITY
        self.fluidsim = SimpleNamespace(
            role=role,
            enabled=True,
            source_ui_schema_version=3,
            friction=0.5,
            bl_rna=SimpleNamespace(properties=properties),
        )


def _scene(*objects, frame=1):
    return SimpleNamespace(objects=tuple(objects), frame_current=frame)


def _translated(x):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_stamp_follows_a_moving_collider():
    paddle = _Object("paddle")
    scene = _scene(paddle, frame=1)
    at_rest = source_frame_stamp(scene, None, 1)

    paddle.matrix_world = _translated(0.25)
    scene.frame_current = 30

    assert source_frame_stamp(scene, None, 30) != at_rest


def test_stamp_is_repeatable_for_a_static_scene():
    scene = _scene(_Object("a"), _Object("b", role="SOURCE"), frame=1)
    first = source_frame_stamp(scene, None, 1)
    scene.frame_current = 30

    assert source_frame_stamp(scene, None, 30) == first


def test_stamp_ignores_panel_settings_and_names():
    paddle = _Object("paddle")
    scene = _scene(paddle)
    before = source_frame_stamp(scene, None, 1)

    paddle.fluidsim.friction = 0.9
    paddle.name_full = "paddle.001"

    assert source_frame_stamp(scene, None, 1) == before


def test_stamp_does_not_depend_on_object_order():
    a, b = _Object("a"), _Object("b")
    b.matrix_world = _translated(2.0)

    assert (
        source_frame_stamp(_scene(a, b), None, 1)
        == source_frame_stamp(_scene(b, a), None, 1)
    )


def test_untagged_objects_do_not_contribute():
    paddle = _Object("paddle")
    bystander = _Object("bystander", role="NONE")
    before = source_frame_stamp(_scene(paddle), None, 1)
    bystander.matrix_world = _translated(5.0)

    assert source_frame_stamp(_scene(paddle, bystander), None, 1) == before


def test_stamp_refuses_to_hash_another_frames_scene_under_this_frames_name():
    scene = _scene(_Object("paddle"), frame=7)

    with pytest.raises(ValueError, match="frame_set"):
        source_frame_stamp(scene, None, 8)


class _Collection(list):
    def foreach_get(self, name, target):
        import numpy as np
        target[:] = np.asarray([getattr(item, name) for item in self]).reshape(-1)


def _mesh_object(key):
    obj = _Object(key)
    obj.type = 'MESH'
    obj.data = SimpleNamespace(
        vertices=_Collection([SimpleNamespace(co=point) for point in
                              [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]]),
        polygons=[SimpleNamespace(vertices=(0, 1, 2)), SimpleNamespace(vertices=(0, 2, 3))],
    )
    return obj


def test_stamp_follows_triangle_connectivity_with_same_counts():
    obj = _mesh_object('mesh')
    scene = _scene(obj)
    before = source_frame_stamp(scene, None, 1)
    obj.data.polygons[0].vertices = (0, 1, 3)
    obj.data.polygons[1].vertices = (1, 2, 3)
    assert source_frame_stamp(scene, None, 1) != before


def test_stamp_follows_same_count_point_movement():
    obj = _Object('points')
    obj.type = 'POINTCLOUD'
    obj.data.points = _Collection([SimpleNamespace(co=(0, 0, 0))])
    scene = _scene(obj)
    before = source_frame_stamp(scene, None, 1)
    obj.data.points[0].co = (1, 0, 0)
    assert source_frame_stamp(scene, None, 1) != before


@pytest.mark.parametrize('edit', ['geometry', 'transform', 'replacement'])
def test_referenced_untagged_mesh_contributes(edit):
    owner = _Object('source', role='SOURCE')
    ref = _mesh_object('reference')
    ref.fluidsim.role = 'NONE'
    owner.fluidsim.geometry_mode = 'MESH'
    owner.fluidsim.mesh_object = ref
    scene = _scene(owner, ref)
    before = source_frame_stamp(scene, None, 1)
    if edit == 'geometry':
        ref.data.vertices[0].co = (0, 0, 1)
    elif edit == 'transform':
        ref.matrix_world = _translated(2)
    else:
        owner.fluidsim.mesh_object = _mesh_object('replacement')
    assert source_frame_stamp(scene, None, 1) != before


def test_volume_same_path_content_change_contributes(tmp_path):
    owner = _Object('source', role='SOURCE')
    owner.fluidsim.source_region_kind = 'VOLUME_GRID'
    volume = _Object('volume', role='NONE')
    volume.type = 'VOLUME'
    path = tmp_path / 'input.vdb'
    path.write_bytes(b'first content')
    volume.data = SimpleNamespace(filepath=str(path), grids=SimpleNamespace(frame_filepath=''))
    owner.fluidsim.volume_input_object = volume
    scene = _scene(owner, volume)
    before = source_frame_stamp(scene, None, 1)
    path.write_bytes(b'other content')
    assert source_frame_stamp(scene, None, 1) != before


def test_schema_version_is_explicit_in_new_bundles():
    from fluidSim_warp_blender51.io.cache import signature_bundle_from_records
    assert signature_bundle_from_records({}, ())['input_identity_schema_version'] == '2'


@pytest.mark.parametrize('role', ['FIELD', 'SOURCE'])
@pytest.mark.parametrize('schema', [2, 3])
@pytest.mark.parametrize('cadence', ['FIRST_SIM_FRAME', 'EVERY_SIM_FRAME'])
@pytest.mark.parametrize('value_source', ['VOLUME_GRID', 'CONSTANT'])
@pytest.mark.parametrize('edit', ['content', 'transform', 'reference'])
def test_analytic_field_volume_value_dependency(tmp_path, role, schema, cadence, value_source, edit):
    from fluidSim_warp_blender51.blender.cache_identity import source_signature

    owner = _Object('field', role=role)
    owner.fluidsim.source_ui_schema_version = schema
    # Pre-schema-3 files must ignore stale modern region selections.
    owner.fluidsim.source_region_kind = 'POINTS' if schema < 3 else 'ANALYTIC'
    owner.fluidsim.source_mode = 'FIELD'
    owner.fluidsim.field_region_kind = 'ANALYTIC'
    owner.fluidsim.field_value_source = value_source
    owner.fluidsim.mesh_evaluation_mode = cadence
    volume = _Object('volume', role='NONE')
    volume.type = 'VOLUME'
    path = tmp_path / 'values.vdb'
    path.write_bytes(b'initial field values')
    volume.data = SimpleNamespace(filepath=str(path), grids=SimpleNamespace(frame_filepath=''))
    owner.fluidsim.volume_input_object = volume
    scene = _scene(owner, volume)
    before = (source_signature(scene), source_frame_stamp(scene, None, 1))
    if edit == 'content':
        path.write_bytes(b'changed field values')
    elif edit == 'transform':
        volume.matrix_world = _translated(2)
    else:
        replacement = _Object('replacement-volume', role='NONE')
        replacement.type = 'VOLUME'
        replacement.data = volume.data
        owner.fluidsim.volume_input_object = replacement
    after = (source_signature(scene), source_frame_stamp(scene, None, 1))
    if value_source == 'VOLUME_GRID':
        assert after[0] != before[0]
        assert after[1] != before[1]
    else:
        assert after == before


def test_point_source_precedes_unused_field_volume_value():
    from fluidSim_warp_blender51.blender.cache_identity import source_signature

    owner = _Object('source', role='SOURCE')
    owner.fluidsim.source_mode = 'FIELD'
    owner.fluidsim.source_region_kind = 'POINTS'
    owner.fluidsim.field_value_source = 'VOLUME_GRID'
    points = _Object('points', role='NONE')
    points.type = 'POINTCLOUD'
    points.data = SimpleNamespace(points=_Collection([SimpleNamespace(co=(0., 0., 0.))]))
    owner.fluidsim.point_input_object = points
    unused = _Object('unused-volume', role='NONE')
    unused.type = 'VOLUME'
    unused.data = SimpleNamespace(filepath='must-not-open-unused.vdb')
    owner.fluidsim.volume_input_object = unused
    scene = _scene(owner, points, unused)
    before = (source_signature(scene), source_frame_stamp(scene, None, 1))
    unused.data.filepath = 'another-unused.vdb'
    assert (source_signature(scene), source_frame_stamp(scene, None, 1)) == before
    points.data.points[0].co = (1., 0., 0.)
    after = (source_signature(scene), source_frame_stamp(scene, None, 1))
    assert after[0] != before[0] and after[1] != before[1]


def test_unselected_dependency_does_not_contribute():
    owner = _Object('source', role='SOURCE')
    ref = _mesh_object('unused')
    ref.fluidsim.role = 'NONE'
    owner.fluidsim.mesh_object = ref
    scene = _scene(owner, ref)
    before = source_frame_stamp(scene, None, 1)
    ref.data.vertices[0].co = (0, 0, 4)
    ref.matrix_world = _translated(3)
    assert source_frame_stamp(scene, None, 1) == before


def test_native_force_reference_orientation_contributes():
    wind = _Object('wind', role='NONE')
    wind.hide_viewport = False
    ref = _Object('velocity', role='NONE')
    wind.field = SimpleNamespace(type='WIND', strength=1.0, source_object=ref)
    scene = _scene(wind, ref)
    before = source_frame_stamp(scene, None, 1)
    ref.matrix_world = ((0., 0., 1., 0.), (0., 1., 0., 0.),
                        (-1., 0., 0., 0.), (0., 0., 0., 1.))
    assert source_frame_stamp(scene, None, 1) != before


def test_height_profile_pixel_edit_contributes():
    domain = _Object('domain', role='NONE')
    image = SimpleNamespace(size=(2, 2), pixels=[0.0, 0., 0., 1.] * 4)
    domain.fluidsim_domain = SimpleNamespace(
        height_enabled=True, height_back_mode='PROFILE', height_profile_image=image,
        height_profile_scale=1.)
    scene = _scene(domain)
    scene.fluidsim = SimpleNamespace(domain_object=domain)
    from fluidSim_warp_blender51.blender.cache_identity import source_signature
    before = source_frame_stamp(scene, None, 1)
    before_signature = source_signature(scene)
    image.pixels[0] = 1.0
    assert source_frame_stamp(scene, None, 1) != before
    assert source_signature(scene) != before_signature


@pytest.mark.parametrize("inactive", ["disabled", "hidden"])
@pytest.mark.parametrize("exists", [False, True])
def test_inactive_volume_does_not_read_or_hash_external_bytes(tmp_path, inactive, exists):
    owner = _Object('source', role='SOURCE')
    owner.fluidsim.source_region_kind = 'VOLUME_GRID'
    volume = _Object('volume', role='NONE')
    volume.type = 'VOLUME'
    path = tmp_path / 'unused.vdb'
    if exists:
        path.write_bytes(b'initial')
    volume.data = SimpleNamespace(filepath=str(path), grids=SimpleNamespace(frame_filepath=''))
    owner.fluidsim.volume_input_object = volume
    if inactive == 'disabled':
        owner.fluidsim.enabled = False
    else:
        owner.hide_viewport = True
    scene = _scene(owner, volume)
    from fluidSim_warp_blender51.blender.cache_identity import source_signature
    before = source_frame_stamp(scene, None, 1)
    before_signature = source_signature(scene)
    path.write_bytes(b'changed unused content')
    assert source_frame_stamp(scene, None, 1) == before
    assert source_signature(scene) == before_signature
    owner.fluidsim.enabled = True
    owner.hide_viewport = False
    assert source_frame_stamp(scene, None, 1) != before
    assert source_signature(scene) != before_signature


def test_unselected_domain_profile_does_not_contribute():
    domain = _Object('other-domain', role='NONE')
    image = SimpleNamespace(size=(2, 2), pixels=[0.0, 0., 0., 1.] * 4)
    domain.fluidsim_domain = SimpleNamespace(
        height_enabled=True, height_back_mode='PROFILE', height_profile_image=image,
        height_profile_scale=1.)
    scene = _scene(domain)
    before = source_frame_stamp(scene, None, 1)
    image.pixels[0] = 1.0
    assert source_frame_stamp(scene, None, 1) == before


def test_hidden_automatic_seed_points_still_follow_seed_consumer():
    owner = _mesh_object('hidden-seed')
    owner.fluidsim.role = 'SOURCE'
    owner.fluidsim.auto_inject_seed = True
    owner.fluidsim.source_region_kind = 'VOLUME_GRID'
    owner.hide_viewport = True
    # Production excludes hidden source plans, but prepare_auto_seeds still
    # consumes enabled seed points. The inactive volume path must stay unused.
    scene = _scene(owner)
    before = source_frame_stamp(scene, None, 1)
    owner.data.vertices[0].co = (0, 0, 2)
    assert source_frame_stamp(scene, None, 1) != before
