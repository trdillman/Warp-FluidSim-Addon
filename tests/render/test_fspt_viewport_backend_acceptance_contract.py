"""Grader controls only; native CUDA broker execution is a separate gate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_fspt_device_scene_capture import fixture as live_scene  # noqa: F401 - pytest fixture


@pytest.fixture
def verifier():
    path = Path(__file__).resolve().parents[1] / 'acceptance' / 'blender_fspt_viewport_backend_acceptance.py'
    spec = importlib.util.spec_from_file_location('fspt_broker_verifier_controls', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example():
    pixels = np.ones((3, 4, 4), np.float32)
    pixels[0, :, :3] = (0.1, 0.2, 0.3)
    frame = SimpleNamespace(device='cuda:0', key=('captured', 7), samples=1, pixels=pixels)
    return frame, pixels.copy()


def test_native_grader_accepts_exact_source_bound_nontrivial_image(verifier):
    frame, expected = example()
    verifier.check_frame(frame, expected, 'cuda:0', ('captured', 7))


@pytest.mark.parametrize('fault', ['stale', 'device', 'samples', 'constant', 'nonfinite', 'shape', 'missing'])
def test_native_grader_rejects_reached_bad_result(verifier, fault):
    frame, expected = example()
    if fault == 'stale':
        frame.key = ('captured', 6)
    elif fault == 'device':
        frame.device = 'cpu'
    elif fault == 'samples':
        frame.samples = 2
    elif fault == 'constant':
        frame.pixels[:] = frame.pixels[0, 0]
    elif fault == 'nonfinite':
        frame.pixels[0, 0, 0] = float('nan')
    elif fault == 'shape':
        frame.pixels = frame.pixels[:1]
    else:
        frame = None
    with pytest.raises(AssertionError):
        verifier.check_frame(frame, expected, 'cuda:0', ('captured', 7))


def test_native_grader_rejects_a_trivial_matching_oracle(verifier):
    frame, _ = example()
    frame.pixels[:] = (0.1, 0.2, 0.3, 1.0)
    with pytest.raises(AssertionError, match='oracle'):
        verifier.check_frame(frame, frame.pixels.copy(), 'cuda:0', ('captured', 7))


def test_native_deadline_does_not_turn_missing_completion_into_success(verifier):
    with pytest.raises(TimeoutError, match='no completion'):
        verifier.until(lambda: None, lambda value: value is not None, 'no completion', seconds=0)


def test_protected_static_rna_receives_original_graph(verifier):
    class NativeDepsgraph:
        scene, view_layer = object(), object()

    graph = NativeDepsgraph()
    access = []
    calls = []
    fluid = SimpleNamespace(session_uid=1, data=SimpleNamespace())
    fluid.original = fluid
    mesh = object()

    def static_to_mesh(*, preserve_all_data_layers, depsgraph):
        if type(depsgraph) is not NativeDepsgraph:
            raise TypeError('Depsgraph expected')
        assert depsgraph is graph and preserve_all_data_layers is False
        access.append('to_mesh')
        return mesh

    static = SimpleNamespace(session_uid=2, to_mesh=static_to_mesh,
                             to_mesh_clear=lambda: access.append('to_mesh_clear'))
    static.original = static
    graph.object_instances = [SimpleNamespace(object=fluid), SimpleNamespace(object=static)]
    guarded_graph, guarded_fluid = verifier._guard_live_graph(
        graph, fluid, lambda item: item.session_uid, calls)
    with pytest.raises(TypeError, match='Depsgraph expected'):
        static.to_mesh(preserve_all_data_layers=False, depsgraph=guarded_graph)
    fluid_instance, static_instance = list(guarded_graph.object_instances)
    assert fluid_instance.object.original is guarded_fluid
    assert static_instance.object.original is static
    evaluated = static_instance.object
    assert evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=guarded_graph) is mesh
    evaluated.to_mesh_clear()
    assert access == ['to_mesh', 'to_mesh_clear'] and not calls

    for source in (guarded_fluid, fluid_instance.object):
        for name in ('to_mesh', 'evaluated_geometry'):
            with pytest.raises(AssertionError, match='fluid evaluated geometry'):
                getattr(source, name)
        with pytest.raises(AssertionError, match='fluid bulk mesh'):
            source.data.vertices.foreach_get('co', np.empty(0))
    assert calls == ['to_mesh', 'evaluated_geometry', 'vertices'] * 2


def test_actual_adapter_static_rna_boundary_with_fluid_guard(verifier, live_scene, monkeypatch):  # noqa: F811
    from fluidSim_warp_blender51.blender import fspt_bridge, fspt_device_scene, surface_publication

    class NativeDepsgraph(SimpleNamespace):
        pass

    scene = live_scene
    graph = NativeDepsgraph(**vars(scene.graph))
    native_reads, calls = [], []

    def strict_to_mesh(original):
        def extract(*, preserve_all_data_layers, depsgraph):
            if type(depsgraph) is not NativeDepsgraph:
                raise TypeError('Depsgraph expected')
            assert depsgraph is graph
            native_reads.append(depsgraph)
            return original(preserve_all_data_layers=preserve_all_data_layers, depsgraph=depsgraph)
        return extract

    for static in (scene.left, scene.right):
        static.to_mesh = strict_to_mesh(static.to_mesh)
    guarded_graph, guarded_fluid = verifier._guard_live_graph(
        graph, scene.fluid, surface_publication.id_token, calls)
    monkeypatch.setattr(fspt_bridge, '_surface_object', lambda _: guarded_fluid)

    request = fspt_device_scene.capture_device_scene(guarded_graph, scene.selection)
    assert native_reads == [graph, graph] and scene.extraction == [11, 13]
    assert request.reference.key == scene.selection.key
    assert [layout.vertex_offset for layout in request.layout] == [0, 3, 6]
    assert not calls
