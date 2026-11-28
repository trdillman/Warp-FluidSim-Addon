"""GPU scene adaptation omits fluid mesh reads and preserves global ordering."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_fspt_device_scene import key
from test_fspt_material_translation import diffuse, graph

from fluidSim_warp_blender51.blender import fspt_bridge, surface_publication
from fluidSim_warp_blender51.fspt.device_scene import DeviceGeometryReference, DeviceSceneUnavailable


class Rows:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __len__(self):
        return len(self.values)

    def foreach_get(self, name, target):
        values = np.zeros(len(self.values), np.int32) if name == 'material_index' else self.values
        target[:] = values.ravel()


class Obj(SimpleNamespace):
    def get(self, name, default=None):
        return 'FluidSim Surface' if name == 'fluidsim_output_kind' and self.fluid else default


class FluidMesh(SimpleNamespace):
    @property
    def vertices(self):
        raise AssertionError('fluid vertex array was read')

    @property
    def loop_triangles(self):
        raise AssertionError('fluid triangle array was read')


@pytest.fixture
def fixture(monkeypatch):
    material, _ = graph(diffuse())
    material.as_pointer = lambda: 10
    extraction = []
    geometry = SimpleNamespace(pointcloud=None, curves=None, volume=None, grease_pencil=None,
                               instance_references=lambda: ())

    def obj(uid, fluid, offset):
        mesh = SimpleNamespace(session_uid=uid + 100, vertices=Rows(np.eye(3) + offset),
            loop_triangles=Rows([[0, 1, 2]]), materials=[material], shape_keys=None,
            has_custom_normals=False, calc_loop_triangles=lambda: None)
        if fluid:
            mesh = FluidMesh(session_uid=uid + 100, materials=[material], shape_keys=None,
                             has_custom_normals=False)
        item = Obj(session_uid=uid, name=f'Object {uid}', type='MESH', fluid=fluid, data=mesh,
            fluidsim=SimpleNamespace(role='NONE'), hide_render=False, matrix_world=np.eye(4),
            visible_get=lambda **_: True, modifiers=[], material_slots=[SimpleNamespace(
                link='DATA', material=material)], evaluated_geometry=lambda: geometry,
            to_mesh_clear=lambda: None)
        item.original = item
        if fluid:
            def forbidden_geometry():
                raise AssertionError('fluid evaluated geometry was queried')
            item.evaluated_geometry = forbidden_geometry

        def to_mesh(**_):
            extraction.append(uid)
            if fluid:
                raise AssertionError('fluid evaluated mesh was read')
            return mesh

        item.to_mesh = to_mesh
        return item

    left, fluid, right = obj(11, False, -2), obj(12, True, 0), obj(13, False, 2)
    scene = SimpleNamespace(session_uid=100, frame_current=9, world=None,
        fluidsim=SimpleNamespace(domain_object=None, output_collection_name='Output',
                                 cache_playback_enabled=False),
        objects={o.name: o for o in (left, fluid, right)})
    layer = SimpleNamespace(session_uid=201, material_override=None, world_override=None,
                            objects=scene.objects)
    instances = [SimpleNamespace(object=o, is_instance=False, show_self=True,
                                 matrix_world=np.eye(4)) for o in (left, fluid, right)]
    depsgraph = SimpleNamespace(scene=scene, view_layer=layer, object_instances=instances)
    source_key = replace(key(1), scene_token=100, owner_token=(None, 'Output'))
    stamp = surface_publication.SurfacePublication(
        100, (None, 'Output'), source_key.epoch, surface_publication.SurfaceChannel.GPU_FINAL,
        17, surface_publication.SurfaceSource.LIVE, 9, source_key.solver_frame,
        source_key.state_revision, source_key.simulation_time, None, None,
        source_key.build_stamp.build_key, fluid.session_uid, fluid.data.session_uid, 3, 1,
        True, 4.0, 3.0, source_key.source_timeline_frame)
    pool = SimpleNamespace(_key=source_key, device='cuda:0')
    budget = SimpleNamespace(surface_ready=lambda k: (3, 1) if k == pool._key else None)
    pool._geometry_array_budget = budget
    selection = SimpleNamespace(publication=stamp, key=source_key, pool=pool,
                                vertex_count=3, triangle_count=1)
    monkeypatch.setattr(fspt_bridge, '_surface_object', lambda _: fluid)
    current = [stamp]
    monkeypatch.setattr(surface_publication, 'current_surface_publication', lambda *_: current[0])
    validation = []
    def validate(owner, captured, **kwargs):
        validation.append((owner, captured, kwargs))
    monkeypatch.setattr(fspt_bridge, 'validate_publications', validate)
    return SimpleNamespace(scene=scene, layer=layer, graph=depsgraph, fluid=fluid,
        left=left, right=right, selection=selection, extraction=extraction, current=current,
        validation=validation)


def module():
    from fluidSim_warp_blender51.blender import fspt_device_scene
    return fspt_device_scene


def test_ordered_static_capture_never_calls_fluid_mesh_api(fixture):
    a = fixture
    result = module().capture_device_scene(a.graph, a.selection)
    assert a.extraction == [11, 13]
    assert isinstance(result.parts[1], DeviceGeometryReference)
    np.testing.assert_array_equal(result.parts[0].vertices, np.eye(3) - 2)
    np.testing.assert_array_equal(result.parts[2].vertices, np.eye(3) + 2)
    assert [part.vertex_offset for part in result.layout] == [0, 3, 6]
    assert [part.index_offset for part in result.layout] == [0, 3, 6]
    np.testing.assert_array_equal(result.material_indices, [1, 1, 1])
    assert result.materials[1].color == pytest.approx((0.2, 0.4, 0.6))
    assert result.publications == (a.selection.publication,)


def test_capability_is_metadata_only_and_rejects_visible_modifiers(fixture):
    a = fixture
    assert module().surface_capability_reason(a.scene, a.layer) == ''
    a.fluid.modifiers.append(SimpleNamespace(show_viewport=True, name='Subdivision'))
    assert 'modifier' in module().surface_capability_reason(a.scene, a.layer)
    assert not a.extraction
    with pytest.raises(DeviceSceneUnavailable, match='modifier'):
        module().capture_device_scene(a.graph, a.selection)
    assert not a.extraction


@pytest.mark.parametrize('kind', ['transform', 'instance', 'duplicate', 'normals', 'shape', 'materials'])
def test_incompatible_fluid_semantics_fall_back_before_static_capture(fixture, kind):
    a = fixture
    if kind == 'transform':
        a.fluid.matrix_world[0, 3] = 1
    elif kind == 'instance':
        a.graph.object_instances[1].is_instance = True
    elif kind == 'duplicate':
        a.graph.object_instances.append(a.graph.object_instances[1])
    elif kind == 'normals':
        a.fluid.data.has_custom_normals = True
    elif kind == 'shape':
        a.fluid.data.shape_keys = object()
    else:
        a.fluid.material_slots.append(a.fluid.material_slots[0])
    with pytest.raises(DeviceSceneUnavailable):
        module().capture_device_scene(a.graph, a.selection)
    assert not a.extraction


def test_selected_gpu_generation_is_revalidated_after_static_capture(fixture):
    a = fixture
    previous = a.right.to_mesh

    def replace_generation(**kwargs):
        a.current[0] = replace(a.current[0], generation=18)
        return previous(**kwargs)

    a.right.to_mesh = replace_generation
    with pytest.raises(DeviceSceneUnavailable, match='publication'):
        module().capture_device_scene(a.graph, a.selection)


def test_owned_static_cache_reuses_only_with_explicit_matching_revision(fixture):
    a = fixture
    cache = module().StaticSceneCache()
    first = module().capture_device_scene(a.graph, a.selection, static_revision=1, cache=cache)
    second = module().capture_device_scene(a.graph, a.selection, static_revision=1, cache=cache)
    assert a.extraction == [11, 13]
    assert second.parts[0] is first.parts[0] and second.parts[2] is first.parts[2]
    third = module().capture_device_scene(a.graph, a.selection, static_revision=2, cache=cache)
    assert a.extraction == [11, 13, 11, 13]
    assert third.parts[0] is not first.parts[0]


def test_static_transform_winding_and_material_override_follow_reference(fixture):
    a = fixture
    a.graph.object_instances[0].matrix_world[0, 0] = -1
    result = module().capture_device_scene(a.graph, a.selection)
    np.testing.assert_array_equal(result.parts[0].triangles, [[0, 2, 1]])
    a.layer.material_override = object()
    from fluidSim_warp_blender51.blender.fspt_scene import UnsupportedSceneError
    with pytest.raises(UnsupportedSceneError, match='override'):
        module().capture_device_scene(a.graph, a.selection)


def test_stale_key_and_wrong_source_facts_are_not_rendered(fixture):
    a = fixture
    a.selection.pool._key = replace(a.selection.key, generation=2)
    with pytest.raises(DeviceSceneUnavailable):
        module().capture_device_scene(a.graph, a.selection)
    a.selection.pool._key = a.selection.key
    a.selection.publication = replace(a.selection.publication, state_revision=999)
    a.current[0] = a.selection.publication
    with pytest.raises(DeviceSceneUnavailable):
        module().capture_device_scene(a.graph, a.selection)


@pytest.mark.parametrize('unsupported', ['type', 'component'])
def test_hidden_static_self_keeps_reference_unsupported_geometry_errors(fixture, unsupported):
    a = fixture
    a.graph.object_instances[0].show_self = False
    if unsupported == 'type':
        a.left.type = 'POINTCLOUD'
    else:
        a.left.evaluated_geometry = lambda: SimpleNamespace(pointcloud=object())
    from fluidSim_warp_blender51.blender.fspt_scene import UnsupportedSceneError
    with pytest.raises(UnsupportedSceneError, match='unsupported|not supported'):
        module().capture_device_scene(a.graph, a.selection)


def test_capture_uses_channel_aware_validation_before_and_after(fixture):
    a = fixture
    module().capture_device_scene(a.graph, a.selection)
    assert len(a.validation) == 2
    for scene, captured, options in a.validation:
        assert scene is a.scene and captured == (a.selection.publication,)
        assert options == dict(for_viewport=True, view_layer=a.layer, device='cuda:0')
